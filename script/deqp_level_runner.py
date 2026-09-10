#!/usr/bin/env python3
"""Headless execution of the same four-tier plan used by the desktop UI."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
import csv
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time

if __package__:
    from .deqp_4level_catalog import (
        DEFAULT_GL_CONFIG, EXACT_CASE_RE, GROUPS_4LEVEL, L4_MODULES,
        TIERS_BY_ID, build_plan, module_for_case, round_robin_shards,
    )
else:
    from deqp_4level_catalog import (
        DEFAULT_GL_CONFIG, EXACT_CASE_RE, GROUPS_4LEVEL, L4_MODULES,
        TIERS_BY_ID, build_plan, module_for_case, round_robin_shards,
    )

REPO = Path(__file__).resolve().parents[1]
FIELDS = ["case", "status", "exit_code", "duration_ms", "case_dir"]
NONFAIL = {"Pass", "NotSupported", "Waiver", "QualityWarning", "CompatibilityWarning"}


def nonnegative(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("必須是非負整數")
    return parsed


def arguments(argv=None):
    parser = argparse.ArgumentParser(
        prog="script/run_deqp_level.sh",
        description="執行 dEQP 四級回歸；預設 --1。共用 UI 的分級規則，不需要 PySide6。",
        epilog="例如：script/run_deqp_level.sh --2 --shards 4；使用 --dry-run 先產生計畫。",
    )
    levels = parser.add_mutually_exclusive_group()
    for level in range(1, 5):
        levels.add_argument(f"--{level}", action="store_const", const=f"L{level}",
                            dest="tier", help=f"執行 Level {level}")
    parser.set_defaults(tier="L1")
    parser.add_argument("--shards", type=int, help="同時執行上限：L1–L3 預設 8；L4 固定 1")
    parser.add_argument("--timeout", type=nonnegative, default=0, help="單案例秒數上限，預設 0（不限）")
    parser.add_argument("--dry-run", action="store_true", help="只用 discovery 快取產生計畫，不啟動 dEQP")
    parser.add_argument("--refresh-discovery", action="store_true", help="重新列舉各 module 的完整測項")
    parser.add_argument("--discovery-dir", type=Path, help="列舉快取目錄（預設 <output-root>/deqp_groups/discovery）")
    parser.add_argument("--output-root", type=Path, default=Path(os.environ.get("PVRGPU_OUTPUT_ROOT") or REPO / "outputs"))
    parser.add_argument("--output-dir", type=Path, help="本次結果目錄，必須尚未存在；預設建立時間戳目錄")
    parser.add_argument("--deqp-build-dir", type=Path, default=os.environ.get("PVRGPU_DEQP_BUILD_DIR") or None)
    parser.add_argument("--deqp-project", type=Path, default=os.environ.get("PVRGPU_DEQP_PROJECT_DIR") or None)
    parser.add_argument("--mesa-prefix", type=Path, default=os.environ.get("PVRGPU_MESA_PVRGPU_PREFIX") or None)
    parser.add_argument("--systemc-lib", type=Path, default=os.environ.get("PVRGPU_SYSTEMC_API_LIB") or None)
    parser.add_argument("--verify-link", action="store_true", help="記錄實際載入的 driver/bridge")
    parser.add_argument("--runner", type=Path, default=REPO / "script/run_deqp_dynamic.sh", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.shards is None:
        args.shards = TIERS_BY_ID[args.tier].shards
    if not 1 <= args.shards <= 8:
        parser.error("--shards 必須介於 1 與 8")
    if args.tier == "L4" and args.shards != 1:
        parser.error("Level 4 依序執行，--shards 必須是 1")
    if args.dry_run and args.refresh_discovery:
        parser.error("--dry-run 不會重新列舉；請移除 --refresh-discovery")
    for key in ("output_root", "output_dir", "discovery_dir", "deqp_build_dir",
                "deqp_project", "mesa_prefix", "systemc_lib", "runner"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, Path(value).expanduser().resolve())
    if args.discovery_dir is None:
        args.discovery_dir = args.output_root / "deqp_groups/discovery"
    return args


def save_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_cases(path: Path, module: str) -> list[str]:
    cases = [line.split("#", 1)[0].strip() for line in path.read_text(encoding="utf-8").splitlines()]
    cases = [case for case in cases if case]
    if not cases:
        raise ValueError(f"列舉快取是空的：{path}")
    if len(cases) != len(set(cases)):
        raise ValueError(f"列舉快取有重複測項：{path}")
    for case in cases:
        if not EXACT_CASE_RE.fullmatch(case) or module_for_case(case) != module:
            raise ValueError(f"{path} 含無效或錯誤 module 的測項：{case}")
    return cases


class Processes:
    """Own each runner's process group, including its dEQP/timeout children."""
    def __init__(self):
        self.lock = threading.Lock()
        self.active = set()
        self.cancelled = False

    def run(self, command, log: Path) -> int:
        with log.open("w", encoding="utf-8") as stream:
            with self.lock:
                if self.cancelled:
                    return 130
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                self.active.add(process)
            try:
                return process.wait()
            except BaseException:
                self.cancel()
                raise
            finally:
                with self.lock:
                    self.active.discard(process)

    def cancel(self):
        with self.lock:
            self.cancelled = True
            active = list(self.active)
        for process in active:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        # A bounded grace period, then stop stubborn descendants too.
        for process in active:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def common_command(args) -> list[str]:
    command = ["bash", str(args.runner)]
    for key, flag in (("mesa_prefix", "--mesa-prefix"), ("systemc_lib", "--systemc-lib"),
                      ("deqp_build_dir", "--deqp-build-dir"), ("deqp_project", "--deqp-project")):
        if getattr(args, key):
            command += [flag, str(getattr(args, key))]
    return command


def prepare(args, output: Path, processes: Processes):
    modules = list(L4_MODULES) if args.tier == "L4" else sorted({g.module for g in GROUPS_4LEVEL})
    discovery = {}
    snapshot = output / "discovery"
    snapshot.mkdir()
    for module in modules:
        cached = args.discovery_dir / f"{module}.txt"
        if args.refresh_discovery or not cached.is_file():
            if args.dry_run:
                raise ValueError(f"--dry-run 需要列舉快取：{cached}；請先執行 --refresh-discovery")
            print(f"列舉 {module} …", flush=True)
            generated = snapshot / f"{module}.txt"
            command = common_command(args) + ["--module", module, "--discover", "--caselist-out",
                str(generated), "--output-dir", str(output / "discovery_runs" / module)]
            log = snapshot / f"{module}.log"
            code = processes.run(command, log)
            if code:
                raise RuntimeError(f"{module} 列舉失敗（exit {code}）：{log}")
            cases = read_cases(generated, module)
            cached.parent.mkdir(parents=True, exist_ok=True)
            # Publish only a complete, validated discovery; keep old cache on error.
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=cached.parent,
                                             prefix=f".{module}-", delete=False) as stream:
                stream.write("\n".join(cases) + "\n")
                temporary = Path(stream.name)
            temporary.replace(cached)
        else:
            cases = read_cases(cached, module)
        discovery[module] = cases
        (snapshot / f"{module}.txt").write_text("\n".join(cases) + "\n", encoding="utf-8")

    tier = TIERS_BY_ID[args.tier]
    plan = build_plan(name=args.tier, tier=tier, groups=list(GROUPS_4LEVEL),
                      discovery=discovery, shards=args.shards, log_images=tier.log_images,
                      default_config=DEFAULT_GL_CONFIG, whole_modules=tier.whole_modules)
    if not plan.total_cases:
        raise ValueError("計畫沒有測項")
    if args.tier != "L4":
        for entry in plan.groups:
            if entry.quota and not entry.selected:
                raise ValueError(f"第 {entry.group.number} 組沒有一般測項；請更新 discovery 快取")
    caselists = output / "caselists"
    caselists.mkdir()
    jobs = []
    for bucket in plan.buckets:
        for index, cases in enumerate(round_robin_shards(bucket.cases, args.shards)):
            tag = f"{bucket.tag}-shard{index}"
            path = caselists / f"{tag}.txt"
            path.write_text("\n".join(cases) + "\n", encoding="utf-8")
            directory = output / "runs" / tag
            command = common_command(args) + ["--module", bucket.module, "--caselist", str(path),
                "--keep-going", "--surface-type", "pbuffer", "--size", "256x256",
                "--gl-config", bucket.gl_config, "--log-images", plan.log_images,
                "--timeout", str(args.timeout), "--output-dir", str(directory)]
            if args.verify_link:
                command.append("--verify-link")
            jobs.append(dict(tag=tag, module=bucket.module, gl_config=bucket.gl_config,
                             count=len(cases), caselist=str(path), output_dir=str(directory),
                             log=str(output / f"{tag}.log"), command=command))
    all_cases = [case for bucket in plan.buckets for case in bucket.cases]
    (caselists / "all.txt").write_text("\n".join(all_cases) + "\n", encoding="utf-8")
    payload = dict(tier=args.tier, total_cases=plan.total_cases, shards=plan.shards,
                   log_images=plan.log_images, timeout=args.timeout,
                   long_excluded_l4_only=plan.long_excluded_count,
                   description=plan.description, groups=[asdict(entry) for entry in plan.groups], jobs=jobs)
    save_json(output / "plan.json", payload)
    (output / "commands.sh").write_text(
        "#!/usr/bin/env bash\nset -uo pipefail\n" +
        f"if [[ -e {shlex.quote(str(output / 'runs'))} ]]; then\n" +
        "    echo '結果已存在，請重新產生新的計畫目錄。' >&2\n    exit 1\nfi\nstatus=0\n" +
        "\n".join(shlex.join(job["command"]) + " || status=1" for job in jobs) +
        '\nexit "$status"\n', encoding="utf-8")
    scope = "包含全部長測項" if args.tier == "L4" else f"{plan.long_excluded_count} 項長測項留在 L4"
    print(f"{args.tier}：{plan.total_cases:,} 項，{scope}，"
          f"{args.shards} worker，存圖 {plan.log_images}", flush=True)
    drift = [entry.group.number for entry in plan.groups if entry.drift]
    if drift:
        print(f"discovery 數量與文件不同的組：{', '.join(map(str, drift))}（詳見 plan.json）", flush=True)
    return plan, jobs, all_cases


def merge(output: Path, jobs, expected, codes, started, cancelled=False):
    by_case = {}
    issues = []
    for job in jobs:
        summary = Path(job["output_dir"]) / "summary.tsv"
        wanted = set(Path(job["caselist"]).read_text().splitlines())
        rows = []
        if summary.is_file():
            with summary.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream, delimiter="\t")
                if reader.fieldnames != FIELDS:
                    issues.append(f"{job['tag']}: invalid summary columns")
                else:
                    rows = list(reader)
        for row in rows:
            case = row.get("case")
            if (case in by_case or case not in wanted or None in row or
                    any(row.get(k) is None for k in FIELDS)):
                issues.append(f"{job['tag']}: duplicate/unexpected/incomplete row {case}")
                continue
            try:
                int(row["exit_code"])
                if int(row["duration_ms"]) < 0:
                    raise ValueError("negative duration")
            except ValueError:
                issues.append(f"{job['tag']}: invalid exit/duration for {case}")
                continue
            by_case[case] = row
    rows = [by_case[case] for case in expected if case in by_case]
    with (output / "summary.tsv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    missing = [case for case in expected if case not in by_case]
    failures = [row for row in rows if row["status"] not in NONFAIL or int(row["exit_code"]) != 0]
    job_failures = {job["tag"]: codes.get(job["tag"]) for job in jobs if codes.get(job["tag"]) != 0}
    result = dict(tier=json.loads((output / "plan.json").read_text())["tier"],
                  total=len(expected), completed=len(rows), statuses=dict(Counter(row["status"] for row in rows)),
                  failures=failures, incomplete_cases=missing, job_failures=job_failures,
                  evidence_issues=issues, cancelled=cancelled, seconds=round(time.monotonic()-started, 3))
    save_json(output / "result.json", result)
    print(f"完成 {len(rows):,}/{len(expected):,}：" + json.dumps(result["statuses"], ensure_ascii=False), flush=True)
    if missing or issues or job_failures:
        print(f"未完成 {len(missing)} 項；分片錯誤 {len(job_failures)}；資料錯誤 {len(issues)}", flush=True)
    return 130 if cancelled else int(bool(failures or missing or issues or job_failures))


def main(argv=None):
    args = arguments(argv)
    output = None
    processes = Processes()
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    old_term = signal.signal(signal.SIGTERM, interrupted)
    try:
        if args.output_dir:
            output = args.output_dir
            output.mkdir(parents=True, exist_ok=False)
        else:
            root = args.output_root / "deqp_4level"
            root.mkdir(parents=True, exist_ok=True)
            output = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("%Y%m%d_%H%M%S_") + args.tier + "_", dir=root))
        print(f"結果目錄：{output}", flush=True)
        plan, jobs, expected = prepare(args, output, processes)
        if args.dry_run:
            print(f"計畫已產生：{output / 'plan.json'}", flush=True)
            return 0
        # Resolve every module/config before dispatching any test jobs.
        for bucket in plan.buckets:
            log = output / f"preflight-{bucket.tag}.log"
            code = processes.run(common_command(args) + ["--module", bucket.module,
                "--case", bucket.cases[0], "--gl-config", bucket.gl_config,
                "--check", "--output-dir", str(output / "preflight" / bucket.tag)], log)
            if code:
                raise RuntimeError(f"{bucket.tag} runtime 檢查失敗（exit {code}）：{log}")
        started = time.monotonic()
        codes = {}
        cancelled = False
        pool = ThreadPoolExecutor(max_workers=args.shards)
        try:
            futures = {pool.submit(processes.run, job["command"], Path(job["log"])): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                codes[job["tag"]] = future.result()
                print(f"分片 {len(codes)}/{len(jobs)} 完成：{job['tag']}（exit {codes[job['tag']]}）", flush=True)
        except KeyboardInterrupt:
            cancelled = True
            processes.cancel()
        except Exception:
            processes.cancel()
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        return merge(output, jobs, expected, codes, started, cancelled)
    except KeyboardInterrupt:
        processes.cancel()
        print("執行已中止，已完成的結果保留。", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as error:
        processes.cancel()
        print(f"run_deqp_level: {error}", file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGTERM, old_term)


if __name__ == "__main__":
    raise SystemExit(main())
