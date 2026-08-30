#!/usr/bin/env python3
"""Catalog and optionally replay pre-recorded dEQP RDC captures.

This worker deliberately consumes only captured .rdc files. It never invokes a
dEQP binary; on-the-fly dEQP execution belongs to a later phase.
"""

from __future__ import annotations

import argparse
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
EVENT_SCHEMA = "pvrgpu.deqp-capture-report.v1"
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
MANIFEST_ROW_RE = re.compile(r"^\s*(\d+)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*$")
CANCEL_GRACE_SECONDS = 0.75


@dataclass(frozen=True)
class PhaseDefinition:
    number: int
    name: str
    title: str
    goal: str

    @property
    def key(self) -> str:
        return f"phase{self.number}-{self.name}"


PHASES: tuple[PhaseDefinition, ...] = (
    PhaseDefinition(
        0,
        "context-info",
        "Context and Info",
        "Capture metadata, EGL/context creation, and no-render smoke.",
    ),
    PhaseDefinition(
        1,
        "clear-scissor",
        "Clear and Scissor",
        "Color clears, clear masks, and scissor-limited clears.",
    ),
    PhaseDefinition(
        2,
        "basic-draw",
        "Basic Draw",
        "Primitive draw, vertex arrays, attributes, instancing, and simple shaders.",
    ),
    PhaseDefinition(
        3,
        "fragment-state",
        "Fragment State",
        "Rasterization, depth/stencil, blend, and fragment operations.",
    ),
    PhaseDefinition(
        4,
        "texture-sampler",
        "Texture and Sampler",
        "Texture upload, filtering, compressed formats, mipmaps, and samplers.",
    ),
    PhaseDefinition(
        5,
        "fbo-sync-image",
        "FBO, Sync, and Image",
        "FBOs, framebuffer blits, EGL images, sync, and multi-pass setup.",
    ),
    PhaseDefinition(
        6,
        "advanced-large",
        "Advanced and Large Features",
        "Compute, SSBO/UBO, transform feedback, tessellation, geometry, robustness, stress, and Manhattan-class features.",
    ),
)
PHASE_BY_NUMBER = {phase.number: phase for phase in PHASES}
PHASE_BY_KEY = {phase.key: phase for phase in PHASES}


def default_work_root() -> Path:
    return Path.home() / "Downloads" / "_Codex" / "Working" / "PvrGPU"


def default_deqp_rdc_root() -> Path:
    return Path.home() / "Downloads" / "_Codex" / "Working" / "deqp"


class ReportSetupError(RuntimeError):
    """A setup error that prevents the report from running."""


class RunnerFailure(RuntimeError):
    """One replay runner failed."""

    def __init__(self, message: str, return_code: int | None = None) -> None:
        super().__init__(message)
        self.return_code = return_code


class ReportCancelled(RuntimeError):
    """The terminal or UI requested cancellation."""


@dataclass(frozen=True)
class ManifestRecord:
    index: int
    case: str
    rdc_filename: str
    status: str


@dataclass(frozen=True)
class Capture:
    index: int
    path: Path
    relative_path: str
    sha256: str
    suite: str
    group: str
    case: str
    manifest_status: str
    qpa_path: str
    phase: int
    phase_key: str
    phase_title: str


@dataclass
class CaptureResult:
    index: int
    rdc: str
    sha256: str
    suite: str
    group: str
    case: str
    manifest_status: str
    qpa: str
    phase: int
    phase_key: str
    phase_title: str
    artifact_dir: str
    status: str = "READY"
    stage: str = "catalog"
    reason: str = ""
    golden: str = "SKIP"
    golden_counter: str = ""
    pvrgpu: str = "SKIP"
    pvrgpu_counter: str = ""
    compare: str = "SKIP"
    diff: str = ""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_name(value: str) -> str:
    cleaned = SAFE_NAME_RE.sub("-", value).strip("-.")
    return cleaned[:96] or "capture"


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


def parse_manifest(path: Path) -> dict[str, ManifestRecord]:
    records: dict[str, ManifestRecord] = {}
    if not path.is_file():
        return records
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            match = MANIFEST_ROW_RE.match(line)
            if match is None:
                continue
            index_text, case, rdc_filename, status = match.groups()
            rdc_filename = rdc_filename.strip()
            if not rdc_filename or rdc_filename == "(none)":
                continue
            try:
                index = int(index_text)
            except ValueError:
                continue
            records[rdc_filename] = ManifestRecord(
                index=index,
                case=case.strip(),
                rdc_filename=rdc_filename,
                status=status.strip(),
            )
    except OSError as exc:
        raise ReportSetupError(f"Cannot read dEQP capture manifest {path}: {exc}") from exc
    return records


def nearest_recorder_dir(path: Path) -> Path | None:
    current = path.parent
    while current != current.parent:
        if current.name == "recorder":
            return current
        current = current.parent
    return None


def derive_suite_and_group(root: Path, path: Path, recorder_dir: Path | None) -> tuple[str, str]:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    suite = parts[0] if parts else "unknown"
    group = "unknown"
    if recorder_dir is not None:
        try:
            rel_recorder = recorder_dir.relative_to(root).parts
            if len(rel_recorder) >= 2:
                group = rel_recorder[-2].rstrip(".")
        except ValueError:
            pass
    elif len(parts) >= 2:
        group = parts[-2].rstrip(".")
    return suite, group


def derive_case_from_filename(group: str, path: Path) -> str:
    stem = path.stem
    if stem.endswith("_capture"):
        stem = stem[: -len("_capture")]
    stem = re.sub(r"^\d+_", "", stem)
    stem = stem.strip("_")
    if group != "unknown" and stem:
        return f"{group}.{stem.replace('_', '.')}"
    return stem or path.stem


def qpa_for_case(recorder_dir: Path | None, manifest: ManifestRecord | None, path: Path) -> str:
    if recorder_dir is None:
        return ""
    qpa_dir = recorder_dir / "qpa"
    if not qpa_dir.is_dir():
        return ""
    candidates: list[Path] = []
    if manifest is not None:
        candidates.extend(sorted(qpa_dir.glob(f"{manifest.index:04d}_*.qpa")))
    stem = path.stem
    if stem.endswith("_capture"):
        stem = stem[: -len("_capture")]
    candidates.extend(sorted(qpa_dir.glob(f"{stem}.qpa")))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return ""


def classify_phase(case: str, group: str) -> PhaseDefinition:
    text = f"{case} {group}".lower()
    if any(
        token in text
        for token in (
            "compute",
            "ssbo",
            "ubo",
            "transform_feedback",
            "transform.feedback",
            "tessellation",
            "geometry_shading",
            "geometry.shading",
            "robustness",
            ".debug.",
            "stress.",
            ".multisample",
            "multisample.",
            "primitive_bounding_box",
            "primitive.bounding.box",
            "builtin_functions",
            "builtin.functions",
            "multisample_interpolation",
            "multisample.interpolation",
        )
    ):
        return PHASE_BY_NUMBER[6]
    if ".info." in text or "create_context" in text or ".context." in text:
        return PHASE_BY_NUMBER[0]
    if "color_clear" in text or ".clear." in text or ".scissor." in text:
        return PHASE_BY_NUMBER[1]
    if ".draw." in text or "vertex_arrays" in text or "rasterization.primitives" in text:
        return PHASE_BY_NUMBER[2]
    if "fragment_ops" in text or "depth" in text or "stencil" in text or "blend" in text:
        return PHASE_BY_NUMBER[3]
    if ".texture." in text or "texture." in text:
        return PHASE_BY_NUMBER[4]
    if ".fbo." in text or ".sync." in text:
        return PHASE_BY_NUMBER[5]
    return PHASE_BY_NUMBER[6]


def parse_phase_selector(value: str) -> int:
    normalized = value.strip().casefold()
    if normalized.startswith("phase"):
        normalized = normalized[5:]
    if "-" in normalized:
        normalized = normalized.split("-", 1)[0]
    try:
        phase = int(normalized, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid phase selector: {value!r}") from exc
    if phase not in PHASE_BY_NUMBER:
        raise argparse.ArgumentTypeError("phase must be in the range 0..6")
    return phase


def discover_captures(
    root: Path,
    *,
    filters: Sequence[str],
    phases: Sequence[int],
    phase_max: int | None,
    limit: int,
    sample_per_group: int,
) -> list[Capture]:
    if not root.is_dir():
        raise ReportSetupError(f"dEQP capture directory does not exist: {root}")

    paths: list[Path] = []
    walk_errors: list[OSError] = []
    for directory, directory_names, file_names in os.walk(
        root, topdown=True, onerror=walk_errors.append, followlinks=False
    ):
        directory_names[:] = [
            name for name in directory_names if name not in {".git", "__pycache__"}
        ]
        directory_names.sort(key=str.casefold)
        for file_name in sorted(file_names, key=str.casefold):
            path = Path(directory) / file_name
            if path.suffix.lower() == ".rdc":
                paths.append(path)
    if walk_errors:
        raise ReportSetupError(f"Cannot scan dEQP capture directory: {walk_errors[0]}")
    paths.sort(key=lambda path: (path.relative_to(root).as_posix().casefold(), path.as_posix()))

    manifest_cache: dict[Path, dict[str, ManifestRecord]] = {}
    captures: list[Capture] = []
    per_group_counts: dict[str, int] = {}
    normalized_filters = [item.casefold() for item in filters]
    selected_phases = set(phases)
    for path in paths:
        recorder_dir = nearest_recorder_dir(path)
        manifest_records: Mapping[str, ManifestRecord] = {}
        if recorder_dir is not None:
            manifest_path = recorder_dir / "manifest.txt"
            if manifest_path not in manifest_cache:
                manifest_cache[manifest_path] = parse_manifest(manifest_path)
            manifest_records = manifest_cache[manifest_path]
        manifest = manifest_records.get(path.name)
        suite, group = derive_suite_and_group(root, path, recorder_dir)
        case = manifest.case if manifest is not None else derive_case_from_filename(group, path)
        phase = classify_phase(case, group)
        if selected_phases and phase.number not in selected_phases:
            continue
        if phase_max is not None and phase.number > phase_max:
            continue
        searchable = " ".join((path.relative_to(root).as_posix(), suite, group, case)).casefold()
        if normalized_filters and not all(item in searchable for item in normalized_filters):
            continue
        if sample_per_group > 0:
            group_count = per_group_counts.get(group, 0)
            if group_count >= sample_per_group:
                continue
            per_group_counts[group] = group_count + 1
        digest = sha256_file(path)
        captures.append(
            Capture(
                index=len(captures) + 1,
                path=path,
                relative_path=path.relative_to(root).as_posix(),
                sha256=digest,
                suite=suite,
                group=group,
                case=case,
                manifest_status=manifest.status if manifest is not None else "NO_MANIFEST_ROW",
                qpa_path=qpa_for_case(recorder_dir, manifest, path),
                phase=phase.number,
                phase_key=phase.key,
                phase_title=phase.title,
            )
        )
        if limit > 0 and len(captures) >= limit:
            break
    return captures


class EventSink:
    def __init__(self, json_events: bool) -> None:
        self.json_events = json_events

    def emit(self, event_type: str, **fields: object) -> None:
        payload = {"schema": EVENT_SCHEMA, "type": event_type, **fields}
        if self.json_events:
            print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)
            return
        if event_type == "scan_complete":
            print(f"Found {fields['total']} dEQP RDC capture(s).", flush=True)
        elif event_type == "capture_started":
            print(f"[{fields['index']}/{fields['total']}] {fields['case']}", flush=True)
        elif event_type == "capture_result":
            reason = f" - {fields['reason']}" if fields.get("reason") else ""
            print(f"  {fields['status']}{reason}", flush=True)
        elif event_type == "report_written":
            print(f"Report: {fields['path']}", flush=True)
        elif event_type == "summary":
            print(
                f"Summary: {fields['passed']} PASS / {fields['failed']} FAIL / "
                f"{fields['ready']} READY / {fields['golden_pass']} GOLDEN_PASS / "
                f"{fields['pvrgpu_pass']} PVRGPU_PASS / "
                f"{fields['unsupported']} UNSUPPORTED / {fields['errors']} ERROR "
                f"({fields['total']} total)",
                flush=True,
            )


class DeqpCaptureReport:
    def __init__(
        self,
        *,
        input_root: Path,
        output_root: Path,
        filters: Sequence[str],
        phases: Sequence[int],
        phase_max: int | None,
        limit: int,
        sample_per_group: int,
        run_golden: bool,
        run_pvrgpu: bool,
        golden_runner: Path,
        pvrgpu_runner: Path,
        timeout_seconds: float,
        events: EventSink,
    ) -> None:
        self.input_root = input_root.expanduser().resolve()
        self.output_root = output_root.expanduser().resolve()
        self.filters = filters
        self.phases = phases
        self.phase_max = phase_max
        self.limit = limit
        self.sample_per_group = sample_per_group
        self.run_golden = run_golden
        self.run_pvrgpu = run_pvrgpu
        self.golden_runner = golden_runner.expanduser().resolve()
        self.pvrgpu_runner = pvrgpu_runner.expanduser().resolve()
        self.timeout_seconds = timeout_seconds
        self.events = events
        self.started_at = utc_now()
        self.run_root: Path | None = None
        self.cancel_requested = False
        self.cancel_requested_at: float | None = None
        self.active_process: subprocess.Popen[bytes] | None = None

    def request_cancel(self) -> None:
        if not self.cancel_requested:
            self.cancel_requested = True
            self.cancel_requested_at = time.monotonic()
        if self.active_process is not None:
            self._terminate_process(self.active_process)

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
    ) -> None:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        with stdout_path.open("wb") as stdout_stream, stderr_path.open("wb") as stderr_stream:
            try:
                process = subprocess.Popen(
                    list(arguments),
                    stdout=stdout_stream,
                    stderr=stderr_stream,
                    env=os.environ.copy(),
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
                                raise ReportCancelled("Run cancelled by user")
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
            raise ReportCancelled("Run cancelled by user")
        if return_code != 0:
            raise RunnerFailure(f"Runner exited with code {return_code}", return_code)

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

    def _write_capture_input(self, capture: Capture, result: CaptureResult, case_root: Path) -> None:
        lines = [
            "schema=pvrgpu.deqp-capture-input.v1",
            f"rdc={capture.path}",
            f"relative_rdc={capture.relative_path}",
            f"sha256={capture.sha256}",
            f"suite={capture.suite}",
            f"group={capture.group}",
            f"case={capture.case}",
            f"manifest_status={capture.manifest_status}",
            f"qpa={capture.qpa_path}",
            f"phase={capture.phase}",
            f"phase_key={capture.phase_key}",
            f"phase_title={capture.phase_title}",
            f"artifact_dir={result.artifact_dir}",
            "uses_deqp_binary=false",
        ]
        atomic_write_text(case_root / "input.txt", "\n".join(lines) + "\n")

    def _run_one(self, capture: Capture, total: int) -> CaptureResult:
        assert self.run_root is not None
        case_dir_name = f"{capture.index:04d}-{safe_name(capture.case)}-{capture.sha256[:12]}"
        case_root = self.run_root / "cases" / case_dir_name
        case_root.mkdir(parents=True, exist_ok=False)
        result = CaptureResult(
            index=capture.index,
            rdc=capture.relative_path,
            sha256=capture.sha256,
            suite=capture.suite,
            group=capture.group,
            case=capture.case,
            manifest_status=capture.manifest_status,
            qpa=capture.qpa_path,
            phase=capture.phase,
            phase_key=capture.phase_key,
            phase_title=capture.phase_title,
            artifact_dir=f"cases/{case_dir_name}",
        )
        self._write_capture_input(capture, result, case_root)
        self.events.emit(
            "capture_started",
            index=capture.index,
            total=total,
            case=capture.case,
            rdc=capture.relative_path,
        )

        if not self.run_golden and not self.run_pvrgpu:
            return self._store_result(result, case_root)

        if self.run_golden:
            golden_dir = case_root / "golden"
            result.stage = "golden"
            result.golden = "RUNNING"
            try:
                self._run_command(
                    [
                        "bash",
                        str(self.golden_runner),
                        "--rdc",
                        str(capture.path),
                        "--case",
                        capture.case,
                        "--outdir",
                        str(golden_dir),
                    ],
                    stdout_path=golden_dir / "runner.stdout.log",
                    stderr_path=golden_dir / "runner.stderr.log",
                )
                counter_path = golden_dir / "counter_golden.txt"
                if not counter_path.is_file() or counter_path.stat().st_size == 0:
                    raise RunnerFailure(f"Golden counter was not produced: {counter_path}")
                result.status = "GOLDEN_PASS"
                result.stage = "golden"
                result.golden = "PASS"
                result.golden_counter = "golden/counter_golden.txt"
            except ReportCancelled as exc:
                result.status = "CANCELLED"
                result.stage = "cancelled"
                result.golden = "FAIL"
                result.reason = str(exc)
                return self._store_result(result, case_root)
            except (RunnerFailure, OSError) as exc:
                result.status = "ERROR"
                result.stage = "golden"
                result.golden = "FAIL"
                result.reason = str(exc)

        if self.run_pvrgpu and result.status != "ERROR":
            pvrgpu_dir = case_root / "pvrgpu"
            result.stage = "pvrgpu"
            result.pvrgpu = "RUNNING"
            try:
                self._run_command(
                    [
                        "bash",
                        str(self.pvrgpu_runner),
                        "--rdc",
                        str(capture.path),
                        "--case",
                        capture.case,
                        "--phase",
                        str(capture.phase),
                        "--phase-key",
                        capture.phase_key,
                        "--outdir",
                        str(pvrgpu_dir),
                    ],
                    stdout_path=pvrgpu_dir / "runner.stdout.log",
                    stderr_path=pvrgpu_dir / "runner.stderr.log",
                )
                counter_path = pvrgpu_dir / "counter_pvrgpu.txt"
                if not counter_path.is_file() or counter_path.stat().st_size == 0:
                    raise RunnerFailure(f"PvrGPU counter was not produced: {counter_path}")
                result.stage = "pvrgpu"
                result.pvrgpu = "PASS"
                result.pvrgpu_counter = "pvrgpu/counter_pvrgpu.txt"
                result.status = "PVRGPU_PASS"
            except RunnerFailure as exc:
                result.stage = "pvrgpu"
                result.pvrgpu = "UNSUPPORTED" if exc.return_code == 3 else "FAIL"
                if exc.return_code == 3:
                    result.status = "UNSUPPORTED"
                    unsupported = pvrgpu_dir / "unsupported.txt"
                    if unsupported.is_file():
                        result.reason = unsupported.read_text(
                            encoding="utf-8", errors="replace"
                        ).strip()
                    if not result.reason:
                        result.reason = "PvrGPU lowering is not implemented for this capture phase"
                else:
                    result.status = "ERROR"
                    result.reason = str(exc)
            except ReportCancelled as exc:
                result.status = "CANCELLED"
                result.stage = "cancelled"
                result.pvrgpu = "FAIL"
                result.reason = str(exc)
                return self._store_result(result, case_root)
            except OSError as exc:
                result.status = "ERROR"
                result.stage = "pvrgpu"
                result.pvrgpu = "FAIL"
                result.reason = str(exc)

        if result.golden_counter and result.pvrgpu_counter:
            result.stage = "compare"
            golden_text = (case_root / result.golden_counter).read_text(encoding="utf-8")
            pvrgpu_text = (case_root / result.pvrgpu_counter).read_text(encoding="utf-8")
            if golden_text == pvrgpu_text:
                result.status = "PASS"
                result.compare = "PASS"
                result.reason = ""
            else:
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
                result.status = "FAIL"
                result.compare = "FAIL"
                result.diff = "counter_diff.txt"
                result.reason = "17-counter exact comparison mismatch"
        return self._store_result(result, case_root)

    def _store_result(self, result: CaptureResult, case_root: Path) -> CaptureResult:
        atomic_write_text(
            case_root / "result.json",
            json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n",
        )
        self.events.emit(
            "capture_result",
            index=result.index,
            case=result.case,
            status=result.status,
            stage=result.stage,
            reason=result.reason,
        )
        return result

    def _write_report(
        self,
        captures: Sequence[Capture],
        results: Sequence[CaptureResult],
        *,
        cancelled: bool,
        global_reason: str = "",
    ) -> tuple[Path, dict[str, object]]:
        assert self.run_root is not None
        finished_at = utc_now()
        passed = sum(result.status == "PASS" for result in results)
        failed = sum(result.status == "FAIL" for result in results)
        ready = sum(result.status == "READY" for result in results)
        golden_pass = sum(result.status == "GOLDEN_PASS" for result in results)
        pvrgpu_pass = sum(result.status == "PVRGPU_PASS" for result in results)
        unsupported = sum(result.status == "UNSUPPORTED" for result in results)
        errors = sum(result.status == "ERROR" for result in results)
        cancelled_count = sum(result.status == "CANCELLED" for result in results)
        bad = failed + errors
        status = "CANCELLED" if cancelled else "PASS" if bad == 0 and captures else "FAIL"
        phase_counts: dict[str, int] = {}
        for capture in captures:
            phase_counts[capture.phase_key] = phase_counts.get(capture.phase_key, 0) + 1
        summary: dict[str, object] = {
            "schema": EVENT_SCHEMA,
            "input_root": str(self.input_root),
            "run_root": str(self.run_root),
            "started_at": self.started_at,
            "finished_at": finished_at,
            "status": status,
            "total": len(captures),
            "passed": passed,
            "failed": failed,
            "ready": ready,
            "golden_pass": golden_pass,
            "pvrgpu_pass": pvrgpu_pass,
            "unsupported": unsupported,
            "errors": errors,
            "cancelled": cancelled_count,
            "uses_deqp_binary": False,
            "run_golden": self.run_golden,
            "run_pvrgpu": self.run_pvrgpu,
            "filters": list(self.filters),
            "phases": list(self.phases),
            "phase_max": self.phase_max,
            "limit": self.limit,
            "sample_per_group": self.sample_per_group,
            "phase_counts": phase_counts,
            "reason": global_reason,
            "results": [asdict(result) for result in results],
        }
        atomic_write_text(
            self.run_root / "run.json",
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        )
        atomic_write_text(
            self.run_root / "discovered-rdc.txt",
            "".join(f"{capture.sha256}  {capture.relative_path}\n" for capture in captures),
        )

        if self.run_golden and self.run_pvrgpu:
            mode = "golden+pvrgpu"
        elif self.run_golden:
            mode = "golden-replay"
        elif self.run_pvrgpu:
            mode = "pvrgpu-probe"
        else:
            mode = "catalog"

        lines = [
            "# dEQP RDC Capture Report",
            "",
            f"- Overall: **{status}**",
            f"- Input directory: `{self._markdown_text(str(self.input_root))}`",
            f"- Generated: `{finished_at}`",
            "- Uses dEQP binary: **no**",
            f"- Mode: `{mode}`",
            f"- Total captures: **{len(captures)}**",
            f"- PASS: **{passed}**",
            f"- FAIL: **{failed}**",
            f"- READY: **{ready}**",
            f"- GOLDEN_PASS: **{golden_pass}**",
            f"- PVRGPU_PASS: **{pvrgpu_pass}**",
            f"- UNSUPPORTED: **{unsupported}**",
            f"- ERROR: **{errors}**",
        ]
        if global_reason:
            lines.append(f"- Note: {self._markdown_text(global_reason)}")

        lines.extend(["", "## Phase Hints", "", "| Phase | Captures |", "|---|---:|"])
        for phase, count in sorted(phase_counts.items()):
            lines.append(f"| {self._markdown_text(phase)} | {count} |")

        lines.extend(["", "## Phase Contract", "", "| Phase | Name | Goal |", "|---:|---|---|"])
        for phase in PHASES:
            lines.append(
                "| "
                + " | ".join(
                    (
                        str(phase.number),
                        self._markdown_text(phase.title),
                        self._markdown_text(phase.goal),
                    )
                )
                + " |"
            )

        lines.extend(
            [
                "",
                "## Results",
                "",
                "| # | Case | Phase | Suite | Manifest | Golden | PvrGPU | Compare | Result | Artifacts |",
                "|---:|---|---|---|---|---|---|---|---|---|",
            ]
        )
        for result in results:
            golden_cell = result.golden
            if result.golden_counter:
                golden_cell += " " + self._markdown_link(
                    "counter", f"{result.artifact_dir}/{result.golden_counter}"
                )
            pvrgpu_cell = result.pvrgpu
            if result.pvrgpu_counter:
                pvrgpu_cell += " " + self._markdown_link(
                    "counter", f"{result.artifact_dir}/{result.pvrgpu_counter}"
                )
            result_cell = f"**{self._markdown_text(result.status)}**"
            if result.diff:
                result_cell += " " + self._markdown_link(
                    "diff", f"{result.artifact_dir}/{result.diff}"
                )
            lines.append(
                "| "
                + " | ".join(
                    (
                        str(result.index),
                        self._markdown_text(result.case),
                        self._markdown_text(result.phase_key),
                        self._markdown_text(result.suite),
                        self._markdown_text(result.manifest_status),
                        golden_cell,
                        pvrgpu_cell,
                        result.compare,
                        result_cell,
                        self._markdown_link("result", f"{result.artifact_dir}/result.json"),
                    )
                )
                + " |"
            )

        failures = [
            result
            for result in results
            if result.status in {"FAIL", "ERROR", "CANCELLED", "UNSUPPORTED"}
        ]
        if failures:
            lines.extend(["", "## Failures", ""])
            for result in failures:
                lines.extend(
                    [
                        f"### {result.index}. {self._markdown_text(result.case)}",
                        "",
                        f"- Stage: `{self._markdown_text(result.stage)}`",
                        f"- Reason: {self._markdown_text(result.reason or 'Unspecified failure')}",
                        f"- RDC: `{self._markdown_text(result.rdc)}`",
                        "- Artifacts: "
                        + self._markdown_link("result.json", f"{result.artifact_dir}/result.json"),
                        "",
                    ]
                )

        report_path = self.run_root / "report.md"
        atomic_write_text(report_path, "\n".join(lines).rstrip() + "\n")
        return report_path, summary

    def run(self) -> int:
        if self.phase_max is not None and self.phase_max not in PHASE_BY_NUMBER:
            raise ReportSetupError("--phase-max must be in the range 0..6")
        if self.limit < 0:
            raise ReportSetupError("--limit must be zero or a positive integer")
        if self.sample_per_group < 0:
            raise ReportSetupError("--sample-per-group must be zero or a positive integer")
        if self.timeout_seconds < 0:
            raise ReportSetupError("--timeout-seconds must be zero or a positive number")
        if self.output_root == self.input_root or self.input_root in self.output_root.parents:
            raise ReportSetupError("Output root must be outside the dEQP capture input directory")
        if self.run_golden and not self.golden_runner.is_file():
            raise ReportSetupError(f"Golden capture runner does not exist: {self.golden_runner}")
        if self.run_pvrgpu and not self.pvrgpu_runner.is_file():
            raise ReportSetupError(f"PvrGPU capture runner does not exist: {self.pvrgpu_runner}")

        captures = discover_captures(
            self.input_root,
            filters=self.filters,
            phases=self.phases,
            phase_max=self.phase_max,
            limit=self.limit,
            sample_per_group=self.sample_per_group,
        )
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.run_root = self.output_root / f"deqp-capture-{stamp}-{os.getpid()}"
        self.run_root.mkdir(parents=False, exist_ok=False)
        self.events.emit("scan_complete", total=len(captures))

        results: list[CaptureResult] = []
        global_reason = ""
        if not captures:
            global_reason = "No .rdc captures matched the selected directory/filter"
        for capture in captures:
            if self.cancel_requested:
                break
            results.append(self._run_one(capture, len(captures)))

        report_path, summary = self._write_report(
            captures,
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
            ready=summary["ready"],
            golden_pass=summary["golden_pass"],
            pvrgpu_pass=summary["pvrgpu_pass"],
            unsupported=summary["unsupported"],
            errors=summary["errors"],
            status=summary["status"],
        )
        if self.cancel_requested:
            return 130
        return 0 if summary["status"] == "PASS" else 1


def build_argument_parser() -> argparse.ArgumentParser:
    work_root = Path(os.environ.get("PVRGPU_WORK_ROOT", str(default_work_root())))
    parser = argparse.ArgumentParser(
        description=(
            "Scan pre-recorded dEQP RDC captures and optionally replay them "
            "through the Golden path. This command never invokes a dEQP binary."
        )
    )
    parser.add_argument(
        "--rdc-dir",
        "--input-dir",
        dest="rdc_dir",
        type=Path,
        default=Path(
            os.environ.get(
                "PVRGPU_DEQP_RDC_ROOT",
                str(default_deqp_rdc_root()),
            )
        ),
        help="directory to scan recursively for captured .rdc files",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            os.environ.get(
                "PVRGPU_DEQP_CAPTURE_OUTPUT",
                str(work_root / "out" / "deqp-capture-report"),
            )
        ),
        help="parent directory for a timestamped report directory",
    )
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        help="case/path substring filter; may be specified more than once",
    )
    parser.add_argument(
        "--phase",
        action="append",
        type=parse_phase_selector,
        default=[],
        help="select one capture phase, 0 through 6; may be specified more than once",
    )
    parser.add_argument(
        "--phase-max",
        type=parse_phase_selector,
        help="select every capture whose phase number is less than or equal to this value",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="limit captures after filtering; zero means no limit",
    )
    parser.add_argument(
        "--sample-per-group",
        type=int,
        default=0,
        help="limit captures per dEQP group; zero means no per-group limit",
    )
    parser.add_argument(
        "--run-golden",
        action="store_true",
        help="replay each selected RDC through RenderDoc + Mesa + llvmpipe",
    )
    parser.add_argument(
        "--run-pvrgpu",
        action="store_true",
        help="run the selected captures through the PvrGPU capture hook",
    )
    parser.add_argument(
        "--golden-runner",
        type=Path,
        default=PROJECT_ROOT / "scripts" / "run-deqp-capture-golden.sh",
        help="single-capture Golden runner used only with --run-golden",
    )
    parser.add_argument(
        "--pvrgpu-runner",
        type=Path,
        default=PROJECT_ROOT / "scripts" / "run-deqp-capture-pvrgpu-probe.sh",
        help="single-capture PvrGPU runner used only with --run-pvrgpu",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=0,
        help="per-capture replay timeout; zero disables timeout",
    )
    parser.add_argument(
        "--list-phases",
        action="store_true",
        help="print the capture phase contract and exit",
    )
    parser.add_argument("--json", action="store_true", help="emit JSONL progress events")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    options = parser.parse_args(argv)
    if options.list_phases:
        for phase in PHASES:
            print(f"{phase.number}\t{phase.key}\t{phase.title}\t{phase.goal}")
        return 0
    events = EventSink(options.json)
    report = DeqpCaptureReport(
        input_root=options.rdc_dir,
        output_root=options.output_root,
        filters=options.filter,
        phases=options.phase,
        phase_max=options.phase_max,
        limit=options.limit,
        sample_per_group=options.sample_per_group,
        run_golden=options.run_golden,
        run_pvrgpu=options.run_pvrgpu,
        golden_runner=options.golden_runner,
        pvrgpu_runner=options.pvrgpu_runner,
        timeout_seconds=options.timeout_seconds,
        events=events,
    )

    def cancel_handler(_signum: int, _frame: object) -> None:
        report.request_cancel()

    signal.signal(signal.SIGINT, cancel_handler)
    signal.signal(signal.SIGTERM, cancel_handler)
    try:
        return report.run()
    except ReportSetupError as exc:
        events.emit("fatal", message=str(exc))
        print(f"dEQP capture report setup failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
