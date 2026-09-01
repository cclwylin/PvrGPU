#!/usr/bin/env python3
"""Expand one catalogued dEQP group and run every leaf in a fresh process.

The fresh-process boundary is required by the current PvrGPU SystemC bridge:
it defers simulation until process teardown and retains only the latest pending
driver command.  Passing a wildcard directly to pvrgpu-deqp would therefore
produce incomplete evidence for all but the final submitting case.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time

from deqp_groups import filter_exact_cases, get_group


EVENT_PREFIX = "PVRGPU_BATCH "
CASE_BLOCK_RE = re.compile(
    r"^#beginTestCaseResult\s+([^\r\n]+).*?^#endTestCaseResult\s*$",
    re.DOTALL | re.MULTILINE,
)
RESULT_RE = re.compile(r'<Result\s+StatusCode="([^"]+)">([^<]*)</Result>')
PASS_STATUSES = {"pass"}
SKIP_STATUSES = {"notsupported", "waiver"}
WARNING_STATUSES = {"qualitywarning", "compatibilitywarning"}

_cancel_requested = False
_child: subprocess.Popen[str] | None = None


def emit(event: str, **payload: object) -> None:
    print(
        EVENT_PREFIX
        + json.dumps({"event": event, **payload}, ensure_ascii=False, sort_keys=True),
        flush=True,
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--group-id", required=True)
    parser.add_argument("--surface", default="pbuffer")
    parser.add_argument("--watchdog", default="disable")
    parser.add_argument("--log-images", choices=("enable", "disable"), default="disable")
    parser.add_argument("--gl-config", default="")
    parser.add_argument("--systemc-api-lib", default="")
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="limit execution after discovery; 0 runs the complete group",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="discover and write case-list.txt without executing cases",
    )
    return parser.parse_args(argv)


def _signal_handler(_signum: int, _frame: object) -> None:
    global _cancel_requested
    _cancel_requested = True
    if _child is not None and _child.poll() is None:
        _child.kill()


def _runner_common_args(args: argparse.Namespace, output_dir: Path) -> list[str]:
    values = [
        f"--pvrgpu-output-dir={output_dir}",
        f"--deqp-surface-type={args.surface}",
        f"--deqp-watchdog={args.watchdog}",
        f"--deqp-log-images={args.log_images}",
    ]
    if args.gl_config:
        values.append(f"--deqp-gl-config-name={args.gl_config}")
    if args.systemc_api_lib:
        values.append(f"--pvrgpu-systemc-api-lib={args.systemc_api_lib}")
    return values


def _discovery_override(suite: str) -> str | None:
    # Listing does not execute GL tests.  The override is used only so the
    # package can build its hierarchy when the real EGL config lacks ES3 bits.
    return {
        "dEQP-GLES3": "3.0",
        "dEQP-GLES31": "3.1",
    }.get(suite)


def discover_cases(args: argparse.Namespace) -> tuple[str, ...]:
    group = get_group(args.group_id)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".caselist-discovery-", dir=args.output_dir
    ) as temporary:
        temporary_dir = Path(temporary)
        export_file = temporary_dir / "all-cases.txt"
        artifact_dir = temporary_dir / "artifacts"
        command = [
            str(args.runner),
            *_runner_common_args(args, artifact_dir),
            f"--deqp-case={group.suite}.*",
            "--deqp-runmode=txt-caselist",
            f"--deqp-caselist-export-file={export_file}",
        ]
        environment = os.environ.copy()
        environment.pop("MESA_GLES_VERSION_OVERRIDE", None)
        override = _discovery_override(group.suite)
        if override:
            environment["MESA_GLES_VERSION_OVERRIDE"] = override
        emit("discovery_started", group_id=group.id, suite=group.suite)
        completed = subprocess.run(
            command,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            check=False,
        )
        if completed.stdout:
            print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
        if not export_file.is_file():
            raise RuntimeError(
                "dEQP caselist discovery produced no export "
                f"(exit {completed.returncode})"
            )
        discovered = []
        for line in export_file.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("TEST:"):
                discovered.append(line.partition(":")[2].strip())
        selected = filter_exact_cases(group, discovered)
        emit(
            "discovery_finished",
            group_id=group.id,
            discovered=len(discovered),
            selected=len(selected),
            runner_exit=completed.returncode,
        )
        return selected


def _synthetic_case_block(case_name: str, status: str, message: str) -> str:
    safe_message = (
        message.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return (
        f"#beginTestCaseResult {case_name}\n"
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<TestCaseResult CasePath="{case_name}" Version="0.3.4" '
        'CaseType="SelfValidate">\n'
        f" <Result StatusCode=\"{status}\">{safe_message}</Result>\n"
        "</TestCaseResult>\n\n"
        "#endTestCaseResult\n"
    )


def _extract_case_block(qpa_text: str, case_name: str, return_code: int) -> tuple[str, str]:
    for match in CASE_BLOCK_RE.finditer(qpa_text):
        if match.group(1).strip() != case_name:
            continue
        block = match.group(0).rstrip() + "\n"
        result = RESULT_RE.search(block)
        status = result.group(1).strip() if result else "Incomplete"
        return block, status
    status = "InternalError"
    message = f"runner exit {return_code} without a closed QPA result"
    return _synthetic_case_block(case_name, status, message), status


def _qpa_preamble(qpa_text: str, group_id: str) -> str:
    marker = "#beginTestCaseResult"
    if marker in qpa_text:
        return qpa_text.split(marker, 1)[0]
    return (
        '#sessionInfo logFormatVersion "0.3.4"\n'
        '#sessionInfo vendor "PvrGPU"\n'
        '#sessionInfo renderer "PvrGPU grouped exact-case runner"\n'
        f'#sessionInfo commandLineParameters "--group-id={group_id}"\n\n'
        "#beginSession\n\n"
    )


def _write_summary(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _run_exact_case(
    args: argparse.Namespace,
    case_name: str,
    index: int,
    total: int,
) -> int:
    global _child
    command = [
        str(args.runner),
        *_runner_common_args(args, args.output_dir),
        f"--deqp-case={case_name}",
    ]
    emit("case_started", index=index, total=total, case=case_name)
    environment = os.environ.copy()
    environment.pop("MESA_GLES_VERSION_OVERRIDE", None)
    _child = subprocess.Popen(
        command,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
    )
    assert _child.stdout is not None
    for line in _child.stdout:
        print(f"[{index}/{total}] {line}", end="")
    return_code = _child.wait()
    _child = None
    return return_code


def _status_bucket(status: str) -> str:
    normalized = status.casefold()
    if normalized in PASS_STATUSES:
        return "passed"
    if normalized in SKIP_STATUSES:
        return "skipped"
    if normalized in WARNING_STATUSES:
        return "warnings"
    return "failed"


def execute_group(args: argparse.Namespace, cases: tuple[str, ...]) -> int:
    group = get_group(args.group_id)
    selected = cases[: args.max_cases] if args.max_cases > 0 else cases
    case_list = args.output_dir / "case-list.txt"
    case_list.write_text("".join(f"{case}\n" for case in selected), encoding="utf-8")
    if args.list_only:
        emit("listed", group_id=group.id, total=len(selected), path=str(case_list))
        return 0
    if not group.available:
        raise RuntimeError(group.availability_reason or "group is unavailable")
    if group.id == "gles3-stress-shaders" and args.watchdog != "enable":
        raise RuntimeError(
            "the long-running shader group includes infinite-loop cases; "
            "enable the dEQP watchdog"
        )
    if not selected:
        raise RuntimeError("the selected group contains no exact cases in this CTS build")

    started = time.monotonic()
    aggregate_path = args.output_dir / "batch-results.qpa"
    final_qpa = args.output_dir / "results.qpa"
    journal_path = args.output_dir / "batch-results.jsonl"
    summary_path = args.output_dir / "batch-summary.json"
    counts: Counter[str] = Counter()
    completed_count = 0
    aggregate_started = False

    initial_summary: dict[str, object] = {
        "schema": "pvrgpu.deqp-group-run.v1",
        "group_id": group.id,
        "label": group.label,
        "suite": group.suite,
        "selectors": list(group.selectors),
        "total": len(selected),
        "completed": 0,
        "passed": 0,
        "skipped": 0,
        "warnings": 0,
        "failed": 0,
        "cancelled": False,
    }
    _write_summary(summary_path, initial_summary)
    emit("batch_started", group_id=group.id, total=len(selected))

    with aggregate_path.open("w", encoding="utf-8") as aggregate, journal_path.open(
        "w", encoding="utf-8"
    ) as journal:
        for index, case_name in enumerate(selected, start=1):
            if _cancel_requested:
                break
            case_started = time.monotonic()
            return_code = _run_exact_case(args, case_name, index, len(selected))
            qpa_text = (
                final_qpa.read_text(encoding="utf-8", errors="replace")
                if final_qpa.is_file()
                else ""
            )
            if not aggregate_started:
                aggregate.write(_qpa_preamble(qpa_text, group.id))
                aggregate_started = True
            block, status = _extract_case_block(qpa_text, case_name, return_code)
            aggregate.write(block + "\n")
            aggregate.flush()
            bucket = _status_bucket(status)
            counts[bucket] += 1
            completed_count += 1
            record = {
                "index": index,
                "total": len(selected),
                "case": case_name,
                "status": status,
                "runner_exit": return_code,
                "duration_seconds": round(time.monotonic() - case_started, 6),
            }
            journal.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            journal.flush()
            emit("case_finished", **record)
        if not aggregate_started:
            aggregate.write(_qpa_preamble("", group.id))
        aggregate.write(
            f"Run took {time.monotonic() - started:.2f} seconds\n\n#endSession\n"
        )

    os.replace(aggregate_path, final_qpa)
    summary = {
        **initial_summary,
        "completed": completed_count,
        "passed": counts["passed"],
        "skipped": counts["skipped"],
        "warnings": counts["warnings"],
        "failed": counts["failed"],
        "cancelled": _cancel_requested,
        "duration_seconds": round(time.monotonic() - started, 6),
    }
    _write_summary(summary_path, summary)
    emit("batch_finished", **summary)
    if _cancel_requested:
        return 130
    return 1 if counts["failed"] else 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(sys.argv[1:] if argv is None else argv))
    args.runner = args.runner.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.runner.is_file():
        print(f"deqp group runner: pvrgpu-deqp not found: {args.runner}", file=sys.stderr)
        return 2
    if os.name != "nt" and not os.access(args.runner, os.X_OK):
        print(f"deqp group runner: runner is not executable: {args.runner}", file=sys.stderr)
        return 2
    if args.max_cases < 0:
        print("deqp group runner: --max-cases cannot be negative", file=sys.stderr)
        return 2
    try:
        get_group(args.group_id)
    except KeyError as error:
        print(f"deqp group runner: {error}", file=sys.stderr)
        return 2
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    try:
        cases = discover_cases(args)
        return execute_group(args, cases)
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        emit("error", group_id=args.group_id, message=str(error))
        print(f"deqp group runner: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
