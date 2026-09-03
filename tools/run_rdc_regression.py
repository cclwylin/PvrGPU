#!/usr/bin/env python3
"""RDC Test Pattern Batch Regression Runner for PvrGPU.

Recursively discovers and executes .rdc / .drc test patterns against the
PvrGPU Gallium driver and SystemC hardware model, aggregating pass/fail
statistics, execution times, and generating comprehensive reports.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional


DEFAULT_PATTERNS_DIR = Path("/Users/linwanyi/Downloads/Working/GPU_TestPatterns")
DEFAULT_OUTPUT_DIR = Path("outputs/rdc_regression")
DEFAULT_CANDIDATE_PVRGPU_BINS = [
    Path("/Users/linwanyi/Downloads/_Codex/Working/PvrGPU/build/bin/pvrgpu"),
    Path("build/bin/pvrgpu"),
]


@dataclass
class TestCase:
    rdc_path: Path
    rel_path: Path
    suite: str
    case_name: str
    out_dir: Path


@dataclass
class TestResult:
    case_name: str
    suite: str
    rel_path: str
    rdc_path: str
    out_dir: str
    status: str  # "PASS", "FAIL", "TIMEOUT", "CRASH", "SKIPPED"
    duration_ms: int = 0
    exit_code: int = 0
    reason: str = ""
    error_summary: str = ""
    frame_artifact: str = ""
    stdout_tail: str = ""
    stderr_tail: str = ""


_cancel_event = threading.Event()


def find_pvrgpu_binary(explicit: Optional[Path] = None) -> Optional[Path]:
    if explicit and explicit.is_file() and os.access(explicit, os.X_OK):
        return explicit.resolve()
    env_bin = os.environ.get("PVRGPU_BIN")
    if env_bin:
        p = Path(env_bin)
        if p.is_file() and os.access(p, os.X_OK):
            return p.resolve()
    for candidate in DEFAULT_CANDIDATE_PVRGPU_BINS:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    which = shutil.which("pvrgpu")
    if which:
        return Path(which).resolve()
    return None


def discover_rdc_patterns(
    root: Path,
    out_root: Path,
    suite_filter: Optional[str] = None,
    pattern_regex: Optional[str] = None,
) -> List[TestCase]:
    cases: List[TestCase] = []
    regex = re.compile(pattern_regex) if pattern_regex else None

    if not root.is_dir():
        print(f"Warning: pattern root does not exist: {root}", file=sys.stderr)
        return cases

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext not in (".rdc", ".drc"):
            continue

        rel_path = path.relative_to(root)
        parts = rel_path.parts
        suite = parts[0] if parts else "root"

        if suite_filter and suite_filter.lower() not in suite.lower():
            continue

        rel_str = str(rel_path)
        if regex and not regex.search(rel_str):
            continue

        # Generate a clean outdir mirror
        stem_rel = rel_path.with_suffix("")
        out_dir = out_root / stem_rel
        case_name = path.stem

        cases.append(
            TestCase(
                rdc_path=path.resolve(),
                rel_path=rel_path,
                suite=suite,
                case_name=case_name,
                out_dir=out_dir.resolve(),
            )
        )

    return cases


def inspect_existing_pass(case: TestCase) -> Optional[TestResult]:
    result_json = case.out_dir / "backend-result.json"
    if not result_json.is_file():
        return None
    try:
        with open(result_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("status") == "PASS":
            duration = int(data.get("duration_ms", 0))
            frame = data.get("artifacts", {}).get("frame", "")
            return TestResult(
                case_name=case.case_name,
                suite=case.suite,
                rel_path=str(case.rel_path),
                rdc_path=str(case.rdc_path),
                out_dir=str(case.out_dir),
                status="PASS",
                duration_ms=duration,
                exit_code=0,
                reason="Pre-existing successful run",
                frame_artifact=frame,
            )
    except Exception:
        pass
    return None


def run_single_test(
    pvrgpu_bin: Path,
    case: TestCase,
    timeout_sec: int,
    skip_passed: bool = False,
) -> TestResult:
    if _cancel_event.is_set():
        return TestResult(
            case_name=case.case_name,
            suite=case.suite,
            rel_path=str(case.rel_path),
            rdc_path=str(case.rdc_path),
            out_dir=str(case.out_dir),
            status="SKIPPED",
            reason="Execution cancelled",
        )

    if skip_passed:
        cached = inspect_existing_pass(case)
        if cached:
            return cached

    case.out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(pvrgpu_bin),
        str(case.rdc_path),
        "--outdir",
        str(case.out_dir),
    ]

    start_time = time.monotonic()
    timed_out = False
    proc = None
    stdout_data = ""
    stderr_data = ""

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        stdout_data, stderr_data = proc.communicate(timeout=timeout_sec)
        exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        exit_code = -1
        if proc:
            proc.kill()
            try:
                stdout_data, stderr_data = proc.communicate(timeout=5)
            except Exception:
                pass
    except Exception as e:
        exit_code = -2
        stderr_data = str(e)

    duration_ms = int((time.monotonic() - start_time) * 1000)

    # Check backend-result.json
    result_json = case.out_dir / "backend-result.json"
    status = "FAIL"
    reason = ""
    frame_artifact = ""

    if timed_out:
        status = "TIMEOUT"
        reason = f"Execution timed out after {timeout_sec}s"
    elif result_json.is_file():
        try:
            with open(result_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            backend_status = data.get("status", "")
            if backend_status == "PASS" and exit_code == 0:
                status = "PASS"
            else:
                status = "FAIL"
            reason = data.get("reason", "")
            if "duration_ms" in data and data["duration_ms"]:
                duration_ms = int(data["duration_ms"])
            frame_artifact = data.get("artifacts", {}).get("frame", "")
        except Exception as e:
            reason = f"Invalid backend-result.json: {e}"
    elif exit_code == 0:
        status = "PASS"
    else:
        status = "FAIL"

    # Extract concise error summary from stderr / stdout if reason is empty
    if status != "PASS" and not reason:
        lines = (stderr_data + "\n" + stdout_data).strip().splitlines()
        for line in reversed(lines):
            line_str = line.strip()
            if any(k in line_str.lower() for k in ("failed", "error", "exception", "unsupported", "reject")):
                reason = line_str
                break
        if not reason and lines:
            reason = lines[-1].strip()

    stdout_tail = "\n".join(stdout_data.strip().splitlines()[-10:])
    stderr_tail = "\n".join(stderr_data.strip().splitlines()[-10:])

    return TestResult(
        case_name=case.case_name,
        suite=case.suite,
        rel_path=str(case.rel_path),
        rdc_path=str(case.rdc_path),
        out_dir=str(case.out_dir),
        status=status,
        duration_ms=duration_ms,
        exit_code=exit_code,
        reason=reason,
        frame_artifact=frame_artifact,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
    )


def print_progress(index: int, total: int, result: TestResult) -> None:
    # ANSI color codes
    colors = {
        "PASS": "\033[92m",     # Green
        "FAIL": "\033[91m",     # Red
        "TIMEOUT": "\033[93m",  # Yellow
        "SKIPPED": "\033[90m",  # Gray
    }
    reset = "\033[0m"
    color = colors.get(result.status, "")
    status_str = f"{color}[{result.status:7s}]{reset}"
    time_str = f"{result.duration_ms / 1000.0:6.2f}s"
    short_rel = result.rel_path
    if len(short_rel) > 55:
        short_rel = "..." + short_rel[-52:]
    reason_snippet = f" -> {result.reason[:60]}" if result.status != "PASS" and result.reason else ""
    print(f"[{index:4d}/{total:4d}] {status_str} {time_str} {short_rel:<55}{reason_snippet}", flush=True)


def generate_reports(out_dir: Path, results: List[TestResult], total_time_sec: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. JSON Report
    json_path = out_dir / "regression_summary.json"
    summary_data = {
        "timestamp": datetime.now().isoformat(),
        "total": len(results),
        "passed": sum(1 for r in results if r.status == "PASS"),
        "failed": sum(1 for r in results if r.status == "FAIL"),
        "timeout": sum(1 for r in results if r.status == "TIMEOUT"),
        "skipped": sum(1 for r in results if r.status == "SKIPPED"),
        "total_time_sec": round(total_time_sec, 2),
        "results": [asdict(r) for r in results],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    # 2. CSV Report
    csv_path = out_dir / "regression_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Suite", "Case", "Status", "Duration(ms)", "ExitCode", "Reason", "RelativePath"])
        for r in results:
            writer.writerow([r.suite, r.case_name, r.status, r.duration_ms, r.exit_code, r.reason, r.rel_path])

    # 3. Markdown Report
    md_path = out_dir / "regression_report.md"
    passed = summary_data["passed"]
    failed = summary_data["failed"]
    timeout = summary_data["timeout"]
    skipped = summary_data["skipped"]
    total = summary_data["total"]
    pass_rate = (passed / total * 100.0) if total > 0 else 0.0

    # Per-suite statistics
    suites: Dict[str, Dict[str, int]] = {}
    for r in results:
        st = suites.setdefault(r.suite, {"total": 0, "pass": 0, "fail": 0, "timeout": 0, "skipped": 0})
        st["total"] += 1
        if r.status == "PASS":
            st["pass"] += 1
        elif r.status == "FAIL":
            st["fail"] += 1
        elif r.status == "TIMEOUT":
            st["timeout"] += 1
        elif r.status == "SKIPPED":
            st["skipped"] += 1

    # Failure category breakdown
    fail_reasons: Dict[str, int] = {}
    for r in results:
        if r.status in ("FAIL", "TIMEOUT"):
            key = r.reason if r.reason else "Unknown failure"
            if len(key) > 80:
                key = key[:77] + "..."
            fail_reasons[key] = fail_reasons.get(key, 0) + 1

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# PvrGPU RDC Test Pattern Regression Report\n\n")
        f.write(f"- **Generated At**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Total Patterns**: {total}\n")
        f.write(f"- **Passed**: {passed} ({pass_rate:.1f}%)\n")
        f.write(f"- **Failed**: {failed}\n")
        f.write(f"- **Timed Out**: {timeout}\n")
        f.write(f"- **Skipped**: {skipped}\n")
        f.write(f"- **Total Duration**: {total_time_sec:.1f}s\n\n")

        f.write("## Suite Summary\n\n")
        f.write("| Suite | Total | Passed | Failed | Timeout | Skipped | Pass Rate |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for sname, st in sorted(suites.items()):
            s_rate = (st["pass"] / st["total"] * 100.0) if st["total"] > 0 else 0.0
            f.write(f"| `{sname}` | {st['total']} | {st['pass']} | {st['fail']} | {st['timeout']} | {st['skipped']} | {s_rate:.1f}% |\n")
        f.write("\n")

        if fail_reasons:
            f.write("## Top Failure Reasons\n\n")
            f.write("| Count | Error Reason |\n")
            f.write("| :---: | :--- |\n")
            for reason, count in sorted(fail_reasons.items(), key=lambda x: x[1], reverse=True)[:15]:
                f.write(f"| {count} | `{reason}` |\n")
            f.write("\n")

        f.write("## Test Cases Detail\n\n")
        f.write("| Case | Suite | Status | Duration | Reason |\n")
        f.write("| :--- | :--- | :---: | :---: | :--- |\n")
        for r in results:
            badge = "✅ PASS" if r.status == "PASS" else ("⏱️ TIMEOUT" if r.status == "TIMEOUT" else "❌ FAIL")
            f.write(f"| `{r.case_name}` | `{r.suite}` | {badge} | {r.duration_ms / 1000.0:.2f}s | {r.reason} |\n")

    print(f"\nReports written to:")
    print(f"  Markdown: {md_path}")
    print(f"  JSON:     {json_path}")
    print(f"  CSV:      {csv_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run batch regression on .rdc patterns for PvrGPU",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pattern-dir",
        type=Path,
        default=DEFAULT_PATTERNS_DIR,
        help="Directory containing test patterns",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where test artifacts and reports are saved",
    )
    parser.add_argument(
        "--pvrgpu-bin",
        type=Path,
        default=None,
        help="Path to pvrgpu executable (auto-detected if omitted)",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default=None,
        help="Filter tests by suite substring (e.g. GLBench, dEQP, glmark2, GFXBench)",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Regex pattern to filter relative paths",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of test patterns to run (0 = all)",
    )
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=4,
        help="Number of parallel execution jobs",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds per test pattern (default: 300s)",
    )
    parser.add_argument(
        "--skip-passed",
        action="store_true",
        help="Skip test cases that have already passed in output directory",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list discovered test patterns without executing",
    )

    args = parser.parse_args()

    # Locate pvrgpu binary
    pvrgpu_bin = find_pvrgpu_binary(args.pvrgpu_bin)
    if not args.list_only:
        if not pvrgpu_bin:
            print("Error: Could not locate pvrgpu binary. Build PvrGPU or pass --pvrgpu-bin", file=sys.stderr)
            return 1
        print(f"Using pvrgpu binary: {pvrgpu_bin}")

    # Discover patterns
    print(f"Scanning for .rdc test patterns in: {args.pattern_dir}")
    cases = discover_rdc_patterns(
        root=args.pattern_dir,
        out_root=args.out_dir,
        suite_filter=args.suite,
        pattern_regex=args.filter,
    )

    if args.limit > 0:
        cases = cases[: args.limit]

    total_cases = len(cases)
    print(f"Discovered {total_cases} test pattern(s).")

    if total_cases == 0:
        return 0

    if args.list_only:
        for i, c in enumerate(cases, 1):
            print(f"[{i:4d}] {c.suite:<15} {c.rel_path}")
        return 0

    # Handle graceful termination on Ctrl+C
    results: List[TestResult] = []

    def sigint_handler(signum, frame):
        print("\n[!] Interrupt received, aborting remaining tests and writing report...", file=sys.stderr)
        _cancel_event.set()

    prev_sigint = signal.signal(signal.SIGINT, sigint_handler)

    print(f"Starting regression with {args.jobs} worker(s), timeout={args.timeout}s per test...")
    start_total = time.monotonic()

    if args.jobs <= 1:
        for idx, case in enumerate(cases, 1):
            if _cancel_event.is_set():
                break
            res = run_single_test(pvrgpu_bin, case, args.timeout, args.skip_passed)
            results.append(res)
            print_progress(idx, total_cases, res)
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            future_to_case = {
                executor.submit(run_single_test, pvrgpu_bin, case, args.timeout, args.skip_passed): (idx, case)
                for idx, case in enumerate(cases, 1)
            }
            completed_count = 0
            for future in as_completed(future_to_case):
                idx, case = future_to_case[future]
                completed_count += 1
                try:
                    res = future.result()
                except Exception as e:
                    res = TestResult(
                        case_name=case.case_name,
                        suite=case.suite,
                        rel_path=str(case.rel_path),
                        rdc_path=str(case.rdc_path),
                        out_dir=str(case.out_dir),
                        status="FAIL",
                        reason=f"Executor exception: {e}",
                    )
                results.append(res)
                print_progress(completed_count, total_cases, res)

    total_time = time.monotonic() - start_total
    generate_reports(args.out_dir, results, total_time)

    passed_count = sum(1 for r in results if r.status == "PASS")
    return 0 if (passed_count == len(results) and len(results) > 0) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        sys.exit(0)
