#!/usr/bin/env python3
"""Run Golden and PvrGPU counters for every RDC below a directory.

The worker is intentionally UI-agnostic.  It emits optional JSONL progress
events for the standalone Qt front end, keeps backend diagnostics in the run
artifacts, and always writes a run-level ``report.md`` after discovery.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import difflib
import hashlib
import html
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Mapping, Sequence
from urllib.parse import quote


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from counter_protocol import CounterProtocolError  # noqa: E402
from rdc.write_counter_txt import (  # noqa: E402
    counters_from_golden_report,
    counters_from_pvrgpu_jsonl,
    format_counter_text,
)


EVENT_SCHEMA = "pvrgpu.rdc-counter-run.v1"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
CANCEL_GRACE_SECONDS = 0.75


class BatchSetupError(RuntimeError):
    """A run-level configuration error that prevents discovery/execution."""


class RunnerFailure(RuntimeError):
    """One backend failed for one RDC."""


class BatchCancelled(RuntimeError):
    """The UI or terminal requested cancellation."""


@dataclass(frozen=True)
class ManifestEntry:
    index: int
    case: str
    rdc_path: str
    rdc_sha256: str
    width: int
    height: int


@dataclass(frozen=True)
class DiscoveredRdc:
    path: Path
    relative_path: str
    sha256: str
    manifest: ManifestEntry | None


@dataclass
class CaseResult:
    index: int
    rdc: str
    sha256: str
    case: str
    manifest_index: int | None
    artifact_dir: str
    status: str = "FAIL"
    golden: str = "SKIP"
    pvrgpu: str = "SKIP"
    compare: str = "SKIP"
    stage: str = "manifest-map"
    reason: str = ""
    golden_counter: str = ""
    pvrgpu_counter: str = ""
    diff: str = ""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_name(value: str) -> str:
    cleaned = SAFE_NAME_RE.sub("-", value).strip("-.")
    return cleaned[:80] or "rdc"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def load_manifest(path: Path) -> dict[str, ManifestEntry]:
    if not path.is_file():
        raise BatchSetupError(f"Manifest does not exist: {path}")

    required = {"index", "case", "rdc_path", "rdc_sha256", "width", "height"}
    by_digest: dict[str, ManifestEntry] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            missing = required.difference(reader.fieldnames or ())
            if missing:
                raise BatchSetupError(
                    "Manifest is missing columns: " + ", ".join(sorted(missing))
                )
            for row_number, row in enumerate(reader, start=2):
                digest = (row.get("rdc_sha256") or "").strip().lower()
                case = (row.get("case") or "").strip()
                rdc_path = (row.get("rdc_path") or "").strip()
                if SHA256_RE.fullmatch(digest) is None:
                    raise BatchSetupError(
                        f"Manifest row {row_number} has an invalid RDC SHA-256"
                    )
                if not case or not rdc_path:
                    raise BatchSetupError(
                        f"Manifest row {row_number} has an empty case or RDC path"
                    )
                try:
                    entry = ManifestEntry(
                        index=int(row["index"]),
                        case=case,
                        rdc_path=rdc_path,
                        rdc_sha256=digest,
                        width=int(row["width"]),
                        height=int(row["height"]),
                    )
                except (TypeError, ValueError, KeyError) as exc:
                    raise BatchSetupError(
                        f"Manifest row {row_number} has invalid numeric metadata"
                    ) from exc
                if entry.index <= 0 or entry.width <= 0 or entry.height <= 0:
                    raise BatchSetupError(
                        f"Manifest row {row_number} requires positive index/size values"
                    )
                if digest in by_digest:
                    raise BatchSetupError(
                        f"Manifest contains duplicate RDC SHA-256: {digest}"
                    )
                by_digest[digest] = entry
    except (OSError, UnicodeError) as exc:
        raise BatchSetupError(f"Cannot read manifest {path}: {exc}") from exc

    if not by_digest:
        raise BatchSetupError(f"Manifest contains no RDC rows: {path}")
    return by_digest


def discover_rdcs(root: Path, manifest: Mapping[str, ManifestEntry]) -> list[DiscoveredRdc]:
    if not root.is_dir():
        raise BatchSetupError(f"RDC directory does not exist: {root}")
    walk_errors: list[OSError] = []
    paths: list[Path] = []
    for directory, directory_names, file_names in os.walk(
        root, topdown=True, onerror=walk_errors.append, followlinks=False
    ):
        directory_names.sort(key=str.casefold)
        for file_name in sorted(file_names, key=str.casefold):
            path = Path(directory) / file_name
            if path.suffix.lower() == ".rdc":
                paths.append(path)
    if walk_errors:
        error = walk_errors[0]
        raise BatchSetupError(f"Cannot scan RDC directory: {error}")
    paths.sort(
        key=lambda path: (path.relative_to(root).as_posix().casefold(), path.as_posix())
    )
    discovered: list[DiscoveredRdc] = []
    for path in paths:
        try:
            digest = sha256_file(path)
        except OSError as exc:
            raise BatchSetupError(f"Cannot read RDC {path}: {exc}") from exc
        discovered.append(
            DiscoveredRdc(
                path=path,
                relative_path=path.relative_to(root).as_posix(),
                sha256=digest,
                manifest=manifest.get(digest),
            )
        )
    return discovered


class EventSink:
    def __init__(self, json_events: bool) -> None:
        self.json_events = json_events

    def emit(self, event_type: str, **fields: object) -> None:
        payload = {"schema": EVENT_SCHEMA, "type": event_type, **fields}
        if self.json_events:
            print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)
            return

        if event_type == "scan_complete":
            print(f"Found {fields['total']} RDC file(s).", flush=True)
        elif event_type == "rdc_started":
            print(
                f"[{fields['index']}/{fields['total']}] {fields['rdc']}",
                flush=True,
            )
        elif event_type == "rdc_result":
            reason = f" - {fields['reason']}" if fields.get("reason") else ""
            print(f"  {fields['status']}{reason}", flush=True)
        elif event_type == "report_written":
            print(f"Report: {fields['path']}", flush=True)
        elif event_type == "summary":
            print(
                "Summary: "
                f"{fields['passed']} PASS / {fields['failed']} FAIL "
                f"({fields['total']} total)",
                flush=True,
            )


class DirectoryCounterRun:
    def __init__(
        self,
        *,
        input_root: Path,
        output_root: Path,
        manifest_path: Path,
        golden_runner: Path,
        pvrgpu_runner: Path,
        require_mesa_ingest: bool,
        timeout_seconds: float,
        events: EventSink,
    ) -> None:
        self.input_root = input_root.expanduser().resolve()
        self.output_root = output_root.expanduser().resolve()
        self.manifest_path = manifest_path.expanduser().resolve()
        self.golden_runner = golden_runner.expanduser().resolve()
        self.pvrgpu_runner = pvrgpu_runner.expanduser().resolve()
        self.require_mesa_ingest = require_mesa_ingest
        self.timeout_seconds = timeout_seconds
        self.events = events
        self.cancel_requested = False
        self.cancel_requested_at: float | None = None
        self.active_process: subprocess.Popen[bytes] | None = None
        self.run_root: Path | None = None
        self.started_at = utc_now()

    def request_cancel(self) -> None:
        if not self.cancel_requested:
            self.cancel_requested = True
            self.cancel_requested_at = time.monotonic()
        if self.active_process is not None:
            self._terminate_process(self.active_process)

    @staticmethod
    def _runner_argv(path: Path) -> list[str]:
        if path.suffix.lower() == ".sh":
            return ["bash", str(path)]
        return [str(path)]

    @staticmethod
    def _terminate_process(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except ProcessLookupError:
            return

    @staticmethod
    def _kill_process(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            return

    def _run_command(
        self,
        arguments: Sequence[str],
        *,
        stdout_path: Path,
        stderr_path: Path,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        with stdout_path.open("wb") as stdout_stream, stderr_path.open("wb") as stderr_stream:
            try:
                process = subprocess.Popen(
                    list(arguments),
                    stdout=stdout_stream,
                    stderr=stderr_stream,
                    env=dict(environment) if environment is not None else None,
                    start_new_session=(os.name == "posix"),
                )
            except OSError as exc:
                raise RunnerFailure(f"Could not launch runner: {exc}") from exc
            self.active_process = process
            try:
                started_at = time.monotonic()
                while True:
                    try:
                        return_code = process.wait(timeout=0.1)
                        break
                    except subprocess.TimeoutExpired:
                        now = time.monotonic()
                        if self.cancel_requested:
                            cancelled_at = self.cancel_requested_at or now
                            if now - cancelled_at >= CANCEL_GRACE_SECONDS:
                                self._kill_process(process)
                                process.wait()
                                raise BatchCancelled("Run cancelled by user")
                            continue
                        if (
                            self.timeout_seconds > 0
                            and now - started_at >= self.timeout_seconds
                        ):
                            self._terminate_process(process)
                            try:
                                process.wait(timeout=CANCEL_GRACE_SECONDS)
                            except subprocess.TimeoutExpired:
                                self._kill_process(process)
                                process.wait()
                            raise RunnerFailure(
                                f"Runner exceeded the {self.timeout_seconds:g}-second timeout"
                            )
            finally:
                self.active_process = None

        if self.cancel_requested:
            raise BatchCancelled("Run cancelled by user")
        if return_code != 0:
            raise RunnerFailure(f"Runner exited with code {return_code}")

    def _emit_stage(self, index: int, stage: str, status: str) -> None:
        self.events.emit("stage", index=index, stage=stage, status=status)

    def _store_case_result(self, result: CaseResult, case_root: Path) -> CaseResult:
        atomic_write_text(
            case_root / "result.json",
            json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n",
        )
        self.events.emit(
            "rdc_result",
            index=result.index,
            rdc=result.rdc,
            case=result.case,
            status=result.status,
            reason=result.reason,
            stage=result.stage,
            golden=result.golden,
            pvrgpu=result.pvrgpu,
            compare=result.compare,
        )
        return result

    def _stage_input(self, item: DiscoveredRdc, case_root: Path) -> Path:
        assert item.manifest is not None
        input_dir = case_root / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        staged = input_dir / Path(item.manifest.rdc_path).name
        try:
            staged.symlink_to(item.path.resolve())
        except OSError as exc:
            raise RunnerFailure(f"Could not stage RDC input: {exc}") from exc
        return staged

    def _run_one(self, item: DiscoveredRdc, index: int, total: int) -> CaseResult:
        entry = item.manifest
        label = entry.case if entry is not None else item.path.stem
        case_dir_name = f"{index:04d}-{safe_name(label)}-{item.sha256[:8]}"
        assert self.run_root is not None
        case_root = self.run_root / "cases" / case_dir_name
        case_root.mkdir(parents=True, exist_ok=False)
        result = CaseResult(
            index=index,
            rdc=item.relative_path,
            sha256=item.sha256,
            case=entry.case if entry is not None else "UNMAPPED",
            manifest_index=entry.index if entry is not None else None,
            artifact_dir=f"cases/{case_dir_name}",
        )

        self.events.emit(
            "rdc_started",
            index=index,
            total=total,
            rdc=item.relative_path,
            case=result.case,
        )

        if entry is None:
            result.reason = "RDC SHA-256 is not present in the frozen manifest"
            self._emit_stage(index, "golden", "SKIP")
            self._emit_stage(index, "pvrgpu", "SKIP")
            self._emit_stage(index, "compare", "SKIP")
            return self._store_case_result(result, case_root)

        try:
            staged_rdc = self._stage_input(item, case_root)
        except RunnerFailure as exc:
            result.stage = "input"
            result.reason = str(exc)
            self._emit_stage(index, "golden", "SKIP")
            self._emit_stage(index, "pvrgpu", "SKIP")
            self._emit_stage(index, "compare", "SKIP")
            return self._store_case_result(result, case_root)

        runner_arguments = [
            "--rdc",
            str(staged_rdc),
            "--case",
            entry.case,
            "--width",
            str(entry.width),
            "--height",
            str(entry.height),
        ]

        golden_dir = case_root / "golden"
        golden_dir.mkdir()
        result.stage = "golden"
        self._emit_stage(index, "golden", "RUNNING")
        try:
            self._run_command(
                [
                    *self._runner_argv(self.golden_runner),
                    *runner_arguments,
                    "--outdir",
                    str(golden_dir),
                ],
                stdout_path=golden_dir / "stdout.log",
                stderr_path=golden_dir / "stderr.log",
                environment=os.environ,
            )
            golden_report = golden_dir / "Report.md"
            counters = counters_from_golden_report(golden_report)
            golden_counter = case_root / "counter_golden.txt"
            atomic_write_text(golden_counter, format_counter_text(counters))
            result.golden = "PASS"
            result.golden_counter = golden_counter.name
            self._emit_stage(index, "golden", "PASS")
        except BatchCancelled as exc:
            result.stage = "cancelled"
            result.reason = str(exc)
            self._emit_stage(index, "golden", "FAIL")
            self._emit_stage(index, "pvrgpu", "SKIP")
            self._emit_stage(index, "compare", "SKIP")
            return self._store_case_result(result, case_root)
        except (RunnerFailure, CounterProtocolError, OSError, UnicodeError) as exc:
            result.reason = f"Golden failed: {exc}"
            self._emit_stage(index, "golden", "FAIL")
            self._emit_stage(index, "pvrgpu", "SKIP")
            self._emit_stage(index, "compare", "SKIP")
            return self._store_case_result(result, case_root)

        pvrgpu_dir = case_root / "pvrgpu"
        (pvrgpu_dir / "png").mkdir(parents=True)
        result.stage = "pvrgpu"
        self._emit_stage(index, "pvrgpu", "RUNNING")
        pvrgpu_environment = dict(os.environ)
        pvrgpu_environment["PVRGPU_RDC_MANIFEST"] = str(self.manifest_path)
        pvrgpu_environment["PVRGPU_RDC_POC_ARTIFACT_DIR"] = str(
            pvrgpu_dir / "mesa-poc"
        )
        try:
            self._run_command(
                [
                    *self._runner_argv(self.pvrgpu_runner),
                    *runner_arguments,
                    "--outdir",
                    str(pvrgpu_dir / "png"),
                ],
                stdout_path=pvrgpu_dir / "stdout.jsonl",
                stderr_path=pvrgpu_dir / "stderr.log",
                environment=pvrgpu_environment,
            )
            counters = counters_from_pvrgpu_jsonl(
                pvrgpu_dir / "stdout.jsonl",
                require_mesa_ingest=self.require_mesa_ingest,
                expected_rdc_sha256=item.sha256,
            )
            pvrgpu_counter = case_root / "counter_pvrgpu.txt"
            atomic_write_text(pvrgpu_counter, format_counter_text(counters))
            result.pvrgpu = "PASS"
            result.pvrgpu_counter = pvrgpu_counter.name
            self._emit_stage(index, "pvrgpu", "PASS")
        except BatchCancelled as exc:
            result.stage = "cancelled"
            result.reason = str(exc)
            self._emit_stage(index, "pvrgpu", "FAIL")
            self._emit_stage(index, "compare", "SKIP")
            return self._store_case_result(result, case_root)
        except (RunnerFailure, CounterProtocolError, OSError, UnicodeError) as exc:
            result.reason = f"PvrGPU failed: {exc}"
            self._emit_stage(index, "pvrgpu", "FAIL")
            self._emit_stage(index, "compare", "SKIP")
            return self._store_case_result(result, case_root)

        result.stage = "compare"
        self._emit_stage(index, "compare", "RUNNING")
        golden_text = (case_root / result.golden_counter).read_text(encoding="utf-8")
        pvrgpu_text = (case_root / result.pvrgpu_counter).read_text(encoding="utf-8")
        if golden_text != pvrgpu_text:
            diff_path = case_root / "counter_diff.txt"
            diff_text = "".join(
                difflib.unified_diff(
                    golden_text.splitlines(keepends=True),
                    pvrgpu_text.splitlines(keepends=True),
                    fromfile="counter_golden.txt",
                    tofile="counter_pvrgpu.txt",
                )
            )
            atomic_write_text(diff_path, diff_text)
            result.compare = "FAIL"
            result.diff = diff_path.name
            result.reason = "17-counter exact comparison mismatch"
            self._emit_stage(index, "compare", "FAIL")
            return self._store_case_result(result, case_root)

        result.status = "PASS"
        result.compare = "PASS"
        result.reason = ""
        self._emit_stage(index, "compare", "PASS")
        return self._store_case_result(result, case_root)

    @staticmethod
    def _markdown_text(value: str) -> str:
        return (
            html.escape(value, quote=False)
            .replace("`", "&#96;")
            .replace("\\", "\\\\")
            .replace("|", "\\|")
            .replace("\r", "")
            .replace("\n", "<br>")
        )

    @staticmethod
    def _markdown_link(label: str, target: str) -> str:
        return f"[{label}]({quote(target, safe='/._-')})"

    def _write_report(
        self,
        discovered: Sequence[DiscoveredRdc],
        results: Sequence[CaseResult],
        *,
        cancelled: bool,
        global_reason: str = "",
    ) -> tuple[Path, dict[str, object]]:
        assert self.run_root is not None
        finished_at = utc_now()
        passed = sum(result.status == "PASS" for result in results)
        failed = len(results) - passed
        overall = "CANCELLED" if cancelled else "PASS" if results and failed == 0 else "FAIL"
        summary: dict[str, object] = {
            "schema": EVENT_SCHEMA,
            "input_root": str(self.input_root),
            "run_root": str(self.run_root),
            "manifest": str(self.manifest_path),
            "started_at": self.started_at,
            "finished_at": finished_at,
            "status": overall,
            "total": len(discovered),
            "passed": passed,
            "failed": failed,
            "cancelled": cancelled,
            "reason": global_reason,
            "require_mesa_ingest": self.require_mesa_ingest,
            "results": [asdict(result) for result in results],
        }
        atomic_write_text(
            self.run_root / "run.json",
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        )

        lines = [
            "# RDC Counter Pass/Fail Report",
            "",
            f"- Overall: **{overall}**",
            f"- Input directory: `{self._markdown_text(str(self.input_root))}`",
            f"- Generated: `{finished_at}`",
            f"- Total: **{len(discovered)}**",
            f"- PASS: **{passed}**",
            f"- FAIL: **{failed}**",
            "- Comparison: normalized 17-counter exact text comparison",
            "- PNG comparison: not used",
        ]
        if global_reason:
            lines.append(f"- Note: {self._markdown_text(global_reason)}")
        lines.extend(
            [
                "",
                "## Results",
                "",
                "| # | RDC | Case | Golden | PvrGPU | Result |",
                "|---:|---|---|---|---|---|",
            ]
        )
        for result in results:
            artifact = result.artifact_dir
            golden_cell = result.golden
            if result.golden_counter:
                golden_cell += " " + self._markdown_link(
                    "counter", f"{artifact}/{result.golden_counter}"
                )
            pvrgpu_cell = result.pvrgpu
            if result.pvrgpu_counter:
                pvrgpu_cell += " " + self._markdown_link(
                    "counter", f"{artifact}/{result.pvrgpu_counter}"
                )
            result_cell = f"**{result.status}**"
            if result.diff:
                result_cell += " " + self._markdown_link(
                    "diff", f"{artifact}/{result.diff}"
                )
            lines.append(
                "| "
                + " | ".join(
                    (
                        str(result.index),
                        self._markdown_text(result.rdc),
                        self._markdown_text(result.case),
                        golden_cell,
                        pvrgpu_cell,
                        result_cell,
                    )
                )
                + " |"
            )

        failures = [result for result in results if result.status != "PASS"]
        if failures:
            lines.extend(["", "## Failures", ""])
            for result in failures:
                lines.extend(
                    [
                        f"### {result.index}. {self._markdown_text(result.rdc)}",
                        "",
                        f"- Stage: `{self._markdown_text(result.stage)}`",
                        f"- Reason: {self._markdown_text(result.reason or 'Unspecified failure')}",
                        "- Artifacts: "
                        + self._markdown_link(
                            "result.json", f"{result.artifact_dir}/result.json"
                        ),
                        "",
                    ]
                )

        report_path = self.run_root / "report.md"
        atomic_write_text(report_path, "\n".join(lines).rstrip() + "\n")
        return report_path, summary

    def run(self) -> int:
        if self.timeout_seconds < 0:
            raise BatchSetupError("Timeout must be zero or a positive number")
        if self.output_root == self.input_root or self.input_root in self.output_root.parents:
            raise BatchSetupError(
                "Output root must be outside the RDC input directory"
            )
        for label, runner in (
            ("Golden", self.golden_runner),
            ("PvrGPU", self.pvrgpu_runner),
        ):
            if not runner.is_file():
                raise BatchSetupError(f"{label} runner does not exist: {runner}")
            if runner.suffix.lower() != ".sh" and not os.access(runner, os.X_OK):
                raise BatchSetupError(f"{label} runner is not executable: {runner}")

        manifest = load_manifest(self.manifest_path)
        discovered = discover_rdcs(self.input_root, manifest)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.run_root = self.output_root / f"rdc-counter-{stamp}-{os.getpid()}"
        self.run_root.mkdir(parents=False, exist_ok=False)
        self.events.emit(
            "run_started",
            input_root=str(self.input_root),
            run_root=str(self.run_root),
        )
        self.events.emit("scan_complete", total=len(discovered))

        results: list[CaseResult] = []
        global_reason = ""
        if not discovered:
            global_reason = "No .rdc files were found below the selected directory"
        else:
            for index, item in enumerate(discovered, start=1):
                if self.cancel_requested:
                    break
                results.append(self._run_one(item, index, len(discovered)))

        if self.cancel_requested and len(results) < len(discovered):
            for index, item in enumerate(discovered[len(results) :], start=len(results) + 1):
                label = item.manifest.case if item.manifest else "UNMAPPED"
                case_dir_name = f"{index:04d}-{safe_name(label)}-{item.sha256[:8]}"
                case_root = self.run_root / "cases" / case_dir_name
                case_root.mkdir(parents=True, exist_ok=True)
                result = CaseResult(
                    index=index,
                    rdc=item.relative_path,
                    sha256=item.sha256,
                    case=label,
                    manifest_index=item.manifest.index if item.manifest else None,
                    artifact_dir=f"cases/{case_dir_name}",
                    stage="cancelled",
                    reason="Cancelled before execution",
                )
                results.append(self._store_case_result(result, case_root))

        report_path, summary = self._write_report(
            discovered,
            results,
            cancelled=self.cancel_requested,
            global_reason=global_reason,
        )
        self.events.emit("report_written", path=str(report_path))
        self.events.emit(
            "summary",
            total=summary["total"],
            passed=summary["passed"],
            failed=summary["failed"],
            status=summary["status"],
            report=str(report_path),
        )
        if self.cancel_requested:
            return 130
        return 0 if summary["status"] == "PASS" else 1


def build_argument_parser() -> argparse.ArgumentParser:
    work_root = Path(
        os.environ.get(
            "PVRGPU_WORK_ROOT",
            str(Path.home() / "Downloads" / "_Codex" / "Working" / "PvrGPU"),
        )
    )
    parser = argparse.ArgumentParser(
        description=(
            "Recursively run Golden/PvrGPU counter comparison for every RDC and "
            "write report.md."
        )
    )
    parser.add_argument(
        "--rdc-dir",
        "--input-dir",
        dest="rdc_dir",
        type=Path,
        required=True,
        help="directory to scan recursively for .rdc files",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            os.environ.get(
                "PVRGPU_RDC_COUNTER_OUTPUT", str(work_root / "out" / "rdc-counter-report")
            )
        ),
        help="parent directory for a timestamped run directory",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "config" / "rdc-glbench-v1.tsv",
    )
    parser.add_argument(
        "--golden-runner",
        type=Path,
        default=PROJECT_ROOT / "scripts" / "run-rdc-golden-counter.sh",
    )
    parser.add_argument(
        "--pvrgpu-runner",
        type=Path,
        default=PROJECT_ROOT / "scripts" / "run-rdc-pvrgpu-poc.sh",
    )
    parser.add_argument(
        "--require-mesa-ingest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="require formal RenderDoc+Mesa ingest evidence in PvrGPU JSONL",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=0,
        help="per-runner timeout; zero disables the timeout",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable JSONL progress events for the UI",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    options = parser.parse_args(argv)
    events = EventSink(options.json)
    run = DirectoryCounterRun(
        input_root=options.rdc_dir,
        output_root=options.output_root,
        manifest_path=options.manifest,
        golden_runner=options.golden_runner,
        pvrgpu_runner=options.pvrgpu_runner,
        require_mesa_ingest=options.require_mesa_ingest,
        timeout_seconds=options.timeout_seconds,
        events=events,
    )

    def cancel_handler(_signum: int, _frame: object) -> None:
        run.request_cancel()

    signal.signal(signal.SIGINT, cancel_handler)
    signal.signal(signal.SIGTERM, cancel_handler)
    try:
        return run.run()
    except BatchSetupError as exc:
        events.emit("fatal", message=str(exc))
        print(f"RDC counter run setup failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
