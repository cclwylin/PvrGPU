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
import shutil
import signal
import subprocess
import sys
import time
import struct
from typing import Mapping, Sequence
from urllib.parse import quote
import zlib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from counter_protocol import CounterProtocolError  # noqa: E402
from counter_protocol import STANDARD_COUNTER_FIELDS  # noqa: E402
from rdc.write_counter_txt import (  # noqa: E402
    counters_from_golden_report,
    counters_from_pvrgpu_jsonl,
    format_counter_text,
)


EVENT_SCHEMA = "pvrgpu.rdc-counter-run.v1"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
TRACE_DRAW_ACTIONS_RE = re.compile(r"^Trace draw actions:\s*([0-9]+)\s*$")
DEQP_MANIFEST_ROW_RE = re.compile(
    r"^\s*(\d+)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*$"
)
CANCEL_GRACE_SECONDS = 0.75
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def native_executable_name(name: str) -> str:
    return f"{name}.exe" if os.name == "nt" else name


def default_runner_path(work_root: Path, name: str) -> Path:
    build_root = Path(
        os.environ.get("PVRGPU_BUILD_DIR", str(work_root / "build"))
    ).expanduser()
    return build_root / "bin" / native_executable_name(name)


class BatchSetupError(RuntimeError):
    """A run-level configuration error that prevents discovery/execution."""


class RunnerFailure(RuntimeError):
    """One backend failed for one RDC."""


class PngCompareError(RuntimeError):
    """Framebuffer PNG comparison failed or could not be decoded."""


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
class DeqpManifestRecord:
    index: int
    case: str
    rdc_filename: str
    status: str


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
    png: str = "SKIP"
    stage: str = "manifest-map"
    reason: str = ""
    golden_counter: str = ""
    pvrgpu_counter: str = ""
    diff: str = ""
    golden_png: str = ""
    pvrgpu_png: str = ""
    png_diff: str = ""
    golden_cache: str = ""
    trace_draw_actions: int | None = None


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


def normalized_counter_text_from_file(path: Path) -> str:
    """Read an existing strict 17-counter text file and normalize formatting."""

    counters: dict[str, int] = {}
    expected = set(STANDARD_COUNTER_FIELDS)
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        if "=" not in line:
            raise CounterProtocolError(
                f"{path}: counter line {line_number} is not name=value"
            )
        name, value_text = (part.strip() for part in line.split("=", 1))
        if name not in expected:
            raise CounterProtocolError(
                f"{path}: unexpected counter field {name!r}"
            )
        if name in counters:
            raise CounterProtocolError(
                f"{path}: duplicate counter field {name!r}"
            )
        try:
            value = int(value_text, 10)
        except ValueError as exc:
            raise CounterProtocolError(
                f"{path}: counter field {name!r} is not an integer"
            ) from exc
        if value < 0:
            raise CounterProtocolError(
                f"{path}: counter field {name!r} is negative"
            )
        counters[name] = value
    missing = [field for field in STANDARD_COUNTER_FIELDS if field not in counters]
    if missing:
        raise CounterProtocolError(
            f"{path}: missing counter fields: {', '.join(missing)}"
        )
    return format_counter_text(counters)


def parse_trace_draw_actions(path: Path) -> int | None:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in lines:
        match = TRACE_DRAW_ACTIONS_RE.match(line.strip())
        if match is not None:
            return int(match.group(1))
    return None


def counter_values_from_text(counter_text: str) -> dict[str, int]:
    counters: dict[str, int] = {}
    for raw_line in counter_text.splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        name, value_text = (part.strip() for part in line.split("=", 1))
        if name not in STANDARD_COUNTER_FIELDS:
            continue
        counters[name] = int(value_text, 10)
    return counters


def trace_draw_actions_from_counter_text(counter_text: str) -> int:
    counters = counter_values_from_text(counter_text)
    return counters.get("drawlists", 0)


def png_compare_skip_reason_from_golden_counter(counter_text: str) -> str:
    """Return a reason when the selected counter row has no color writes.

    RenderDoc's replay PNG is a final framebuffer snapshot.  It is only a
    strict oracle for this batch when the selected API work actually writes
    color.  Clear-only captures still keep PNG comparison enabled because a
    clear is a real framebuffer write even though it has no pixel shader
    invocations.
    """

    counters = counter_values_from_text(counter_text)
    if counters.get("drawlists", 0) > 0 and counters.get("ps_invocations", 0) == 0:
        return (
            "Selected draw counter has drawlists but ps_invocations=0; "
            "no measured framebuffer color writes, so the replay PNG is not "
            "used as this frame's comparison oracle."
        )
    return ""


def golden_player_logs_no_color_output(golden_dir: Path) -> bool:
    """Return true when the Golden player says the selected replay has no color target."""

    for relative in (
        "player-wrapper.stdout.log",
        "stdout.log",
        "player.stdout.log",
    ):
        log_path = golden_dir / relative
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if re.search(r"(?m)^Color output:\s*none\s*$", text):
            return True
        if "selected replay range has no color output" in text:
            return True
    return False


def paeth_predictor(left: int, above: int, upper_left: int) -> int:
    prediction = left + above - upper_left
    distance_left = abs(prediction - left)
    distance_above = abs(prediction - above)
    distance_upper_left = abs(prediction - upper_left)
    if distance_left <= distance_above and distance_left <= distance_upper_left:
        return left
    if distance_above <= distance_upper_left:
        return above
    return upper_left


def decode_rgba8_png(path: Path) -> tuple[int, int, bytes]:
    """Decode a non-interlaced RGBA8 PNG into top-to-bottom RGBA bytes."""

    try:
        data = path.read_bytes()
    except OSError as exc:
        raise PngCompareError(f"{path}: cannot read PNG: {exc}") from exc
    if not data.startswith(PNG_SIGNATURE):
        raise PngCompareError(f"{path}: invalid PNG signature")

    offset = len(PNG_SIGNATURE)
    ihdr: tuple[int, int, int, int, int, int, int] | None = None
    idat_parts: list[bytes] = []
    saw_iend = False
    chunk_index = 0

    while offset < len(data):
        if offset + 12 > len(data):
            raise PngCompareError(f"{path}: truncated PNG chunk header")
        length = struct.unpack_from(">I", data, offset)[0]
        chunk_type = data[offset + 4 : offset + 8]
        payload_start = offset + 8
        payload_end = payload_start + length
        crc_end = payload_end + 4
        if crc_end > len(data):
            raise PngCompareError(f"{path}: truncated {chunk_type!r} chunk")

        payload = data[payload_start:payload_end]
        expected_crc = struct.unpack_from(">I", data, payload_end)[0]
        actual_crc = zlib.crc32(chunk_type)
        actual_crc = zlib.crc32(payload, actual_crc) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise PngCompareError(f"{path}: bad {chunk_type!r} CRC")

        if chunk_type == b"IHDR":
            if chunk_index != 0:
                raise PngCompareError(f"{path}: IHDR is not the first chunk")
            if ihdr is not None:
                raise PngCompareError(f"{path}: duplicate IHDR")
            if length != 13:
                raise PngCompareError(f"{path}: invalid IHDR length")
            ihdr = struct.unpack(">IIBBBBB", payload)
        elif chunk_type == b"IDAT":
            if ihdr is None:
                raise PngCompareError(f"{path}: IDAT precedes IHDR")
            idat_parts.append(payload)
        elif chunk_type == b"IEND":
            if length != 0:
                raise PngCompareError(f"{path}: invalid IEND length")
            saw_iend = True
            offset = crc_end
            break

        offset = crc_end
        chunk_index += 1

    if ihdr is None:
        raise PngCompareError(f"{path}: missing IHDR")
    if not idat_parts:
        raise PngCompareError(f"{path}: missing IDAT")
    if not saw_iend:
        raise PngCompareError(f"{path}: missing IEND")
    if offset != len(data):
        raise PngCompareError(f"{path}: trailing bytes after IEND")

    width, height, bit_depth, color_type, compression, filtering, interlace = ihdr
    if bit_depth != 8:
        raise PngCompareError(f"{path}: expected 8-bit channels, got {bit_depth}")
    if color_type != 6:
        raise PngCompareError(f"{path}: expected RGBA color type, got {color_type}")
    if compression != 0:
        raise PngCompareError(f"{path}: unsupported compression method")
    if filtering != 0:
        raise PngCompareError(f"{path}: unsupported PNG filtering method")
    if interlace != 0:
        raise PngCompareError(f"{path}: interlaced PNG is not supported")

    try:
        decompressor = zlib.decompressobj()
        filtered = decompressor.decompress(b"".join(idat_parts))
        filtered += decompressor.flush()
    except zlib.error as exc:
        raise PngCompareError(f"{path}: invalid zlib stream: {exc}") from exc
    if not decompressor.eof:
        raise PngCompareError(f"{path}: incomplete zlib stream")
    if decompressor.unused_data:
        raise PngCompareError(f"{path}: trailing zlib stream data")
    if decompressor.unconsumed_tail:
        raise PngCompareError(f"{path}: unconsumed zlib input")

    bytes_per_pixel = 4
    row_bytes = width * bytes_per_pixel
    expected_filtered_bytes = height * (row_bytes + 1)
    if len(filtered) != expected_filtered_bytes:
        raise PngCompareError(
            f"{path}: expected {expected_filtered_bytes} filtered bytes, "
            f"got {len(filtered)}"
        )

    pixels = bytearray()
    previous_row = bytearray(row_bytes)
    cursor = 0
    for row_index in range(height):
        filter_type = filtered[cursor]
        cursor += 1
        encoded = filtered[cursor : cursor + row_bytes]
        cursor += row_bytes
        reconstructed = bytearray(row_bytes)

        for byte_index, encoded_byte in enumerate(encoded):
            left = (
                reconstructed[byte_index - bytes_per_pixel]
                if byte_index >= bytes_per_pixel
                else 0
            )
            above = previous_row[byte_index]
            upper_left = (
                previous_row[byte_index - bytes_per_pixel]
                if byte_index >= bytes_per_pixel
                else 0
            )

            if filter_type == 0:
                predictor = 0
            elif filter_type == 1:
                predictor = left
            elif filter_type == 2:
                predictor = above
            elif filter_type == 3:
                predictor = (left + above) // 2
            elif filter_type == 4:
                predictor = paeth_predictor(left, above, upper_left)
            else:
                raise PngCompareError(
                    f"{path}: row {row_index} has invalid filter {filter_type}"
                )
            reconstructed[byte_index] = (encoded_byte + predictor) & 0xFF

        pixels.extend(reconstructed)
        previous_row = reconstructed

    return width, height, bytes(pixels)


def compare_rgba8_pngs(golden_path: Path, pvrgpu_path: Path) -> tuple[bool, str]:
    """Return exact decoded-RGBA framebuffer comparison status and summary."""

    golden_width, golden_height, golden_pixels = decode_rgba8_png(golden_path)
    pvrgpu_width, pvrgpu_height, pvrgpu_pixels = decode_rgba8_png(pvrgpu_path)
    if (golden_width, golden_height) != (pvrgpu_width, pvrgpu_height):
        return (
            False,
            "PNG dimension mismatch: "
            f"golden={golden_width}x{golden_height}, "
            f"pvrgpu={pvrgpu_width}x{pvrgpu_height}",
        )

    differing_pixels = 0
    max_channel_delta = 0
    first_difference: tuple[int, int, bytes, bytes] | None = None
    for pixel in range(golden_width * golden_height):
        offset = pixel * 4
        golden_pixel = golden_pixels[offset : offset + 4]
        pvrgpu_pixel = pvrgpu_pixels[offset : offset + 4]
        if golden_pixel == pvrgpu_pixel:
            continue
        differing_pixels += 1
        max_channel_delta = max(
            max_channel_delta,
            *(abs(first - second) for first, second in zip(golden_pixel, pvrgpu_pixel)),
        )
        if first_difference is None:
            first_difference = (
                pixel % golden_width,
                pixel // golden_width,
                golden_pixel,
                pvrgpu_pixel,
            )

    if first_difference is not None:
        x, y, golden_pixel, pvrgpu_pixel = first_difference
        return (
            False,
            f"RGBA_MISMATCH pixels={differing_pixels} "
            f"max_channel_delta={max_channel_delta} first=({x},{y}) "
            f"golden={tuple(golden_pixel)} pvrgpu={tuple(pvrgpu_pixel)}",
        )

    return (
        True,
        f"RGBA_MATCH size={golden_width}x{golden_height} "
        "differing_pixels=0 max_channel_delta=0",
    )


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


def parse_deqp_manifest(path: Path) -> dict[str, DeqpManifestRecord]:
    records: dict[str, DeqpManifestRecord] = {}
    if not path.is_file():
        return records
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            match = DEQP_MANIFEST_ROW_RE.match(line)
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
            records[rdc_filename] = DeqpManifestRecord(
                index=index,
                case=case.strip(),
                rdc_filename=rdc_filename,
                status=status.strip(),
            )
    except OSError as exc:
        raise BatchSetupError(f"Cannot read dEQP manifest {path}: {exc}") from exc
    return records


def nearest_recorder_dir(path: Path) -> Path | None:
    current = path.parent
    while current != current.parent:
        if current.name == "recorder":
            return current
        current = current.parent
    return None


def derive_deqp_group(root: Path, path: Path, recorder_dir: Path | None) -> str:
    if recorder_dir is not None:
        try:
            relative_recorder = recorder_dir.relative_to(root).parts
            if len(relative_recorder) >= 2:
                return relative_recorder[-2].rstrip(".")
        except ValueError:
            pass
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    if len(parts) >= 2:
        return parts[-2].rstrip(".")
    return "unknown"


def derive_case_from_path(root: Path, path: Path) -> tuple[str, int | None]:
    recorder_dir = nearest_recorder_dir(path)
    if recorder_dir is not None:
        records = parse_deqp_manifest(recorder_dir / "manifest.txt")
        record = records.get(path.name)
        if record is not None and record.case:
            return record.case, record.index

    group = derive_deqp_group(root, path, recorder_dir)
    stem = path.stem
    if stem.endswith("_capture"):
        stem = stem[: -len("_capture")]
    stem = re.sub(r"^\d+_", "", stem).strip("_")
    if group != "unknown" and stem:
        return f"{group}.{stem.replace('_', '.')}", None
    return stem or path.stem, None


def sanitize_runner_case(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    return cleaned[:128] or fallback


def natural_path_sort_key(value: str) -> tuple[tuple[int, object], ...]:
    parts: list[tuple[int, object]] = []
    for part in re.split(r"([0-9]+)", value):
        if not part:
            continue
        if part.isdigit():
            parts.append((0, int(part)))
        else:
            parts.append((1, part.casefold()))
    return tuple(parts)


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
        key=lambda path: (
            natural_path_sort_key(path.relative_to(root).as_posix()),
            path.as_posix(),
        )
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
        timeout_seconds: float,
        require_manifest: bool,
        reuse_golden_cache: bool,
        events: EventSink,
    ) -> None:
        self.input_root = input_root.expanduser().resolve()
        self.output_root = output_root.expanduser().resolve()
        self.manifest_path = manifest_path.expanduser().resolve()
        self.golden_runner = golden_runner.expanduser().resolve()
        self.pvrgpu_runner = pvrgpu_runner.expanduser().resolve()
        self.timeout_seconds = timeout_seconds
        self.require_manifest = require_manifest
        self.reuse_golden_cache = reuse_golden_cache
        self.events = events
        self.cancel_requested = False
        self.cancel_requested_at: float | None = None
        self.active_process: subprocess.Popen[bytes] | None = None
        self.run_root: Path | None = None
        self.started_at = utc_now()
        self.golden_cache_dir = self.output_root / "golden-cache"
        self.golden_counter_cache_index: dict[str, Path] = {}
        self.golden_png_cache_index: dict[str, Path] = {}
        self.golden_no_color_output_cache_index: set[str] = set()

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
            png=result.png,
        )
        return result

    @staticmethod
    def _valid_counter_text_path(path: Path) -> bool:
        try:
            normalized_counter_text_from_file(path)
        except (CounterProtocolError, OSError, UnicodeError):
            return False
        return True

    @staticmethod
    def _valid_png_path(path: Path) -> bool:
        try:
            decode_rgba8_png(path)
        except (PngCompareError, OSError, ValueError):
            return False
        return True

    @staticmethod
    def _first_valid_png(paths: Sequence[Path]) -> Path | None:
        for path in paths:
            if path.is_file() and DirectoryCounterRun._valid_png_path(path):
                return path
        return None

    @staticmethod
    def _find_golden_png(golden_dir: Path) -> Path | None:
        preferred = sorted(golden_dir.glob("player-output/*/png/*_replay.png"))
        fallback = sorted(golden_dir.rglob("*.png"))
        return DirectoryCounterRun._first_valid_png([*preferred, *fallback])

    @staticmethod
    def _find_pvrgpu_png(pvrgpu_dir: Path) -> Path | None:
        jsonl_path = pvrgpu_dir / "stdout.jsonl"
        candidates: list[Path] = []
        if jsonl_path.is_file():
            try:
                for line in jsonl_path.read_text(encoding="utf-8").splitlines():
                    stripped = line.lstrip()
                    if not stripped.startswith("{"):
                        continue
                    try:
                        message = json.loads(stripped)
                    except json.JSONDecodeError:
                        continue
                    artifact = message.get("artifact_png")
                    if isinstance(artifact, str) and artifact.strip():
                        artifact_path = Path(artifact)
                        if not artifact_path.is_absolute():
                            artifact_path = (pvrgpu_dir / artifact_path).resolve()
                        candidates.append(artifact_path)
            except (OSError, UnicodeError):
                pass
        candidates.extend(sorted((pvrgpu_dir / "png").glob("*.png")))
        # Deliberately do not fall back to pvrgpu/player-png here: that is the
        # RenderDoc/Mesa replay image, while this report is checking the
        # framebuffer emitted by the PvrGPU SystemC/model backend.
        return DirectoryCounterRun._first_valid_png(candidates)

    @staticmethod
    def _copy_png_artifact(source: Path, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
        return destination

    @staticmethod
    def _relative_artifact_path(case_root: Path, path: Path) -> str:
        try:
            return path.relative_to(case_root).as_posix()
        except ValueError:
            return str(path)

    def _build_golden_counter_cache_index(self) -> dict[str, Path]:
        index: dict[str, Path] = {}
        if not self.reuse_golden_cache:
            return index

        if self.golden_cache_dir.is_dir():
            for counter_path in self.golden_cache_dir.glob("*/counter_golden.txt"):
                digest = counter_path.parent.name.lower()
                if (
                    SHA256_RE.fullmatch(digest) is not None
                    and digest not in index
                    and self._valid_counter_text_path(counter_path)
                ):
                    index[digest] = counter_path
                    cached_png = counter_path.parent / "golden.png"
                    if self._valid_png_path(cached_png):
                        self.golden_png_cache_index[digest] = cached_png
                    metadata_path = counter_path.parent / "metadata.json"
                    try:
                        metadata = json.loads(
                            metadata_path.read_text(encoding="utf-8")
                        )
                    except (OSError, UnicodeError, json.JSONDecodeError):
                        metadata = {}
                    if (
                        isinstance(metadata, dict)
                        and metadata.get("no_color_output") is True
                    ):
                        self.golden_no_color_output_cache_index.add(digest)

        for run_root in sorted(self.output_root.glob("rdc-counter-*"), reverse=True):
            cases_root = run_root / "cases"
            if not cases_root.is_dir():
                continue
            for result_path in sorted(cases_root.glob("*/result.json")):
                try:
                    result_data = json.loads(result_path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError):
                    continue
                if not isinstance(result_data, dict):
                    continue
                digest = str(result_data.get("sha256", "")).lower()
                if digest in index or SHA256_RE.fullmatch(digest) is None:
                    continue
                if result_data.get("golden") not in {"PASS", "CACHED"}:
                    continue
                counter_name = str(result_data.get("golden_counter", "")).strip()
                candidates = []
                if counter_name:
                    candidates.append(result_path.parent / counter_name)
                candidates.extend(
                    (
                        result_path.parent / "counter_golden.txt",
                        result_path.parent / "counter.txt",
                        result_path.parent / "golden" / "counter.txt",
                    )
                )
                for candidate in candidates:
                    if candidate.is_file() and self._valid_counter_text_path(candidate):
                        index[digest] = candidate
                        if digest not in self.golden_png_cache_index:
                            golden_png = self._find_golden_png(result_path.parent / "golden")
                            if golden_png is not None:
                                self.golden_png_cache_index[digest] = golden_png
                        if golden_player_logs_no_color_output(
                            result_path.parent / "golden"
                        ):
                            self.golden_no_color_output_cache_index.add(digest)
                        break
        return index

    def _lookup_golden_counter_cache(self, sha256: str) -> Path | None:
        cached = self.golden_counter_cache_index.get(sha256)
        if cached is not None and self._valid_counter_text_path(cached):
            return cached
        self.golden_counter_cache_index.pop(sha256, None)
        stable = self.golden_cache_dir / sha256 / "counter_golden.txt"
        if stable.is_file() and self._valid_counter_text_path(stable):
            self.golden_counter_cache_index[sha256] = stable
            return stable
        return None

    def _lookup_golden_png_cache(self, sha256: str) -> Path | None:
        cached = self.golden_png_cache_index.get(sha256)
        if cached is not None and self._valid_png_path(cached):
            return cached
        self.golden_png_cache_index.pop(sha256, None)
        stable = self.golden_cache_dir / sha256 / "golden.png"
        if self._valid_png_path(stable):
            self.golden_png_cache_index[sha256] = stable
            return stable
        return None

    def _lookup_golden_trace_draw_actions_cache(self, sha256: str) -> int | None:
        metadata_path = self.golden_cache_dir / sha256 / "metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        if not isinstance(metadata, dict):
            return None
        value = metadata.get("trace_draw_actions")
        return value if isinstance(value, int) and value >= 0 else None

    def _lookup_golden_no_color_output_cache(self, sha256: str) -> bool:
        if sha256 in self.golden_no_color_output_cache_index:
            return True
        metadata_path = self.golden_cache_dir / sha256 / "metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return False
        if isinstance(metadata, dict) and metadata.get("no_color_output") is True:
            self.golden_no_color_output_cache_index.add(sha256)
            return True
        return False

    def _store_golden_counter_cache(
        self,
        *,
        sha256: str,
        counter_text: str,
        source_counter: Path,
        source_png: Path | None = None,
        trace_draw_actions: int | None = None,
        no_color_output: bool = False,
    ) -> None:
        cache_dir = self.golden_cache_dir / sha256
        counter_path = cache_dir / "counter_golden.txt"
        atomic_write_text(counter_path, counter_text)
        png_path = cache_dir / "golden.png"
        cached_png = ""
        if source_png is not None and self._valid_png_path(source_png):
            self._copy_png_artifact(source_png, png_path)
            cached_png = str(png_path)
            self.golden_png_cache_index[sha256] = png_path
        elif png_path.exists():
            png_path.unlink()
        atomic_write_text(
            cache_dir / "metadata.json",
            json.dumps(
                {
                    "schema": EVENT_SCHEMA,
                    "type": "golden_counter_cache",
                    "rdc_sha256": sha256,
                    "source_counter": str(source_counter),
                    "source_png": str(source_png) if source_png is not None else "",
                    "cached_png": cached_png,
                    "trace_draw_actions": trace_draw_actions,
                    "no_color_output": no_color_output,
                    "created_at": utc_now(),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
        )
        self.golden_counter_cache_index[sha256] = counter_path
        if no_color_output:
            self.golden_no_color_output_cache_index.add(sha256)
        else:
            self.golden_no_color_output_cache_index.discard(sha256)

    def _entry_for_item(self, item: DiscoveredRdc) -> tuple[ManifestEntry | None, bool]:
        if item.manifest is not None:
            return item.manifest, False
        if self.require_manifest:
            return None, False

        derived_case, derived_index = derive_case_from_path(self.input_root, item.path)
        safe_case = sanitize_runner_case(derived_case, item.path.stem)
        return (
            ManifestEntry(
                index=derived_index or 0,
                case=safe_case,
                rdc_path=item.relative_path,
                rdc_sha256=item.sha256,
                width=1,
                height=1,
            ),
            True,
        )

    def _stage_input(
        self, item: DiscoveredRdc, case_root: Path, entry: ManifestEntry
    ) -> Path:
        input_dir = case_root / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        staged = input_dir / Path(entry.rdc_path).name
        source = item.path.resolve()
        try:
            staged.symlink_to(source)
        except OSError as symlink_error:
            try:
                os.link(source, staged)
            except OSError:
                try:
                    shutil.copy2(source, staged)
                except OSError as copy_error:
                    raise RunnerFailure(
                        "Could not stage RDC input by symlink, hardlink, or "
                        f"copy: {copy_error}; symlink error: {symlink_error}"
                    ) from copy_error
        return staged

    def _run_one(self, item: DiscoveredRdc, index: int, total: int) -> CaseResult:
        entry, derived_metadata = self._entry_for_item(item)
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
            manifest_index=(
                entry.index
                if entry is not None and not derived_metadata and entry.index > 0
                else None
            ),
            artifact_dir=f"cases/{case_dir_name}",
        )
        trace_draw_actions: int | None = None
        golden_no_color_output = False

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
            staged_rdc = self._stage_input(item, case_root, entry)
        except RunnerFailure as exc:
            result.stage = "input"
            result.reason = str(exc)
            self._emit_stage(index, "golden", "SKIP")
            self._emit_stage(index, "pvrgpu", "SKIP")
            self._emit_stage(index, "compare", "SKIP")
            return self._store_case_result(result, case_root)

        def runner_arguments_for(width: int, height: int) -> list[str]:
            return [
                "--rdc",
                str(staged_rdc),
                "--case",
                entry.case,
                "--width",
                str(width),
                "--height",
                str(height),
            ]

        runner_arguments = runner_arguments_for(entry.width, entry.height)

        result.stage = "golden"
        self._emit_stage(index, "golden", "RUNNING")
        golden_dir = case_root / "golden"
        golden_dir.mkdir()
        golden_counter = case_root / "counter_golden.txt"
        cached_golden_counter = (
            self._lookup_golden_counter_cache(item.sha256)
            if self.reuse_golden_cache
            else None
        )
        if cached_golden_counter is not None:
            try:
                counter_text = normalized_counter_text_from_file(cached_golden_counter)
                atomic_write_text(golden_counter, counter_text)
                result.golden = "CACHED"
                result.golden_counter = golden_counter.name
                result.golden_cache = str(cached_golden_counter)
                trace_draw_actions = self._lookup_golden_trace_draw_actions_cache(
                    item.sha256
                )
                if trace_draw_actions is None:
                    trace_draw_actions = trace_draw_actions_from_counter_text(
                        counter_text
                    )
                result.trace_draw_actions = trace_draw_actions
                golden_no_color_output = self._lookup_golden_no_color_output_cache(
                    item.sha256
                )
                cached_golden_png = (
                    None
                    if golden_no_color_output
                    else self._lookup_golden_png_cache(item.sha256)
                )
                if (
                    cached_golden_png is None
                    and not golden_no_color_output
                    and not png_compare_skip_reason_from_golden_counter(counter_text)
                ):
                    raise CounterProtocolError(
                        "cached Golden result has no framebuffer PNG or "
                        "explicit no-color evidence"
                    )
                if cached_golden_png is not None:
                    case_golden_png = self._copy_png_artifact(
                        cached_golden_png, case_root / "golden.png"
                    )
                    result.golden_png = self._relative_artifact_path(
                        case_root, case_golden_png
                    )
                atomic_write_text(
                    golden_dir / "cache-hit.txt",
                    "Golden counter cache hit; llvmpipe replay was skipped.\n"
                    f"source={cached_golden_counter}\n",
                )
                self._emit_stage(index, "golden", "CACHED")
            except (CounterProtocolError, OSError, UnicodeError):
                self.golden_counter_cache_index.pop(item.sha256, None)
                cached_golden_counter = None

        if cached_golden_counter is None:
            try:
                self._run_command(
                    [
                        str(self.golden_runner),
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
                counter_text = format_counter_text(counters)
                trace_draw_actions = parse_trace_draw_actions(
                    golden_dir / "player-wrapper.stdout.log"
                )
                if trace_draw_actions is None:
                    trace_draw_actions = parse_trace_draw_actions(
                        golden_dir / "stdout.log"
                    )
                if trace_draw_actions is None:
                    trace_draw_actions = trace_draw_actions_from_counter_text(
                        counter_text
                    )
                result.trace_draw_actions = trace_draw_actions
                atomic_write_text(golden_counter, counter_text)
                source_golden_png = self._find_golden_png(golden_dir)
                case_golden_png: Path | None = None
                if source_golden_png is not None:
                    case_golden_png = self._copy_png_artifact(
                        source_golden_png, case_root / "golden.png"
                    )
                    result.golden_png = self._relative_artifact_path(
                        case_root, case_golden_png
                    )
                golden_no_color_output = golden_player_logs_no_color_output(golden_dir)
                self._store_golden_counter_cache(
                    sha256=item.sha256,
                    counter_text=counter_text,
                    source_counter=golden_counter,
                    source_png=case_golden_png,
                    trace_draw_actions=trace_draw_actions,
                    no_color_output=golden_no_color_output,
                )
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
                result.golden = "FAIL"
                self._emit_stage(index, "golden", "FAIL")
                self._emit_stage(index, "pvrgpu", "SKIP")
                self._emit_stage(index, "compare", "SKIP")
                return self._store_case_result(result, case_root)

        pvrgpu_dir = case_root / "pvrgpu"
        (pvrgpu_dir / "png").mkdir(parents=True)
        result.stage = "pvrgpu"
        self._emit_stage(index, "pvrgpu", "RUNNING")
        pvrgpu_runner_arguments = runner_arguments
        golden_png_path_for_extent = (
            case_root / result.golden_png if result.golden_png else None
        )
        if golden_png_path_for_extent is not None:
            try:
                golden_png_width, golden_png_height, _ = decode_rgba8_png(
                    golden_png_path_for_extent
                )
                pvrgpu_runner_arguments = runner_arguments_for(
                    golden_png_width, golden_png_height
                )
            except (OSError, PngCompareError):
                pvrgpu_runner_arguments = runner_arguments
        pvrgpu_environment = dict(os.environ)
        pvrgpu_environment["PVRGPU_RDC_MANIFEST"] = str(self.manifest_path)
        if trace_draw_actions is not None:
            pvrgpu_environment["PVRGPU_RDC_TRACE_DRAW_ACTIONS"] = str(
                trace_draw_actions
            )
        try:
            self._run_command(
                [
                    str(self.pvrgpu_runner),
                    *pvrgpu_runner_arguments,
                    "--outdir",
                    str(pvrgpu_dir),
                ],
                stdout_path=pvrgpu_dir / "stdout.jsonl",
                stderr_path=pvrgpu_dir / "stderr.log",
                environment=pvrgpu_environment,
            )
            counters = counters_from_pvrgpu_jsonl(
                pvrgpu_dir / "stdout.jsonl",
                expected_rdc_sha256=item.sha256,
            )
            pvrgpu_counter = case_root / "counter_pvrgpu.txt"
            atomic_write_text(pvrgpu_counter, format_counter_text(counters))
            pvrgpu_png = self._find_pvrgpu_png(pvrgpu_dir)
            if pvrgpu_png is not None:
                result.pvrgpu_png = self._relative_artifact_path(case_root, pvrgpu_png)
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
            result.pvrgpu = "FAIL"
            self._emit_stage(index, "pvrgpu", "FAIL")
            self._emit_stage(index, "compare", "SKIP")
            return self._store_case_result(result, case_root)

        result.stage = "compare"
        self._emit_stage(index, "compare", "RUNNING")
        golden_text = (case_root / result.golden_counter).read_text(encoding="utf-8")
        pvrgpu_text = (case_root / result.pvrgpu_counter).read_text(encoding="utf-8")
        failure_reasons: list[str] = []
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
            failure_reasons.append("17-counter exact comparison mismatch")
            self._emit_stage(index, "compare", "FAIL")
        else:
            result.compare = "PASS"
            self._emit_stage(index, "compare", "PASS")

        result.stage = "png"
        self._emit_stage(index, "png", "RUNNING")
        golden_png_path = (
            case_root / result.golden_png if result.golden_png else None
        )
        pvrgpu_png_path = (
            case_root / result.pvrgpu_png if result.pvrgpu_png else None
        )
        png_skip_reason = png_compare_skip_reason_from_golden_counter(golden_text)
        if not png_skip_reason and golden_no_color_output:
            png_skip_reason = (
                "Golden RenderDoc replay selected range has no color output; "
                "no framebuffer PNG is used as this frame's comparison oracle."
            )
        if png_skip_reason:
            png_skip_path = case_root / "png_skip.txt"
            atomic_write_text(
                png_skip_path,
                "\n".join(
                    (
                        "PNG framebuffer comparison: SKIP",
                        f"reason={png_skip_reason}",
                        f"golden_png={result.golden_png}",
                        f"pvrgpu_png={result.pvrgpu_png}",
                    )
                )
                + "\n",
            )
            result.png = "SKIP"
            result.png_diff = png_skip_path.name
            self._emit_stage(index, "png", "SKIP")
        elif golden_png_path is None or pvrgpu_png_path is None:
            result.png = "FAIL"
            missing_sides = [
                side
                for side, path in (
                    ("Golden", golden_png_path),
                    ("PvrGPU", pvrgpu_png_path),
                )
                if path is None
            ]
            if len(missing_sides) == 2:
                message = (
                    "Golden and PvrGPU PNG outputs are both missing without "
                    "explicit no-color evidence"
                )
            else:
                missing_side = missing_sides[0]
                existing_side = "PvrGPU" if missing_side == "Golden" else "Golden"
                message = (
                    f"{missing_side} PNG output is missing while {existing_side} "
                    "wrote a framebuffer PNG"
                )
            png_diff_path = case_root / "png_diff.txt"
            atomic_write_text(
                png_diff_path,
                "\n".join(
                    (
                        "PNG framebuffer comparison: FAIL",
                        f"reason={message}",
                        f"golden_png={result.golden_png}",
                        f"pvrgpu_png={result.pvrgpu_png}",
                    )
                )
                + "\n",
            )
            result.png_diff = png_diff_path.name
            failure_reasons.append(message)
            self._emit_stage(index, "png", "FAIL")
        else:
            try:
                png_match, png_message = compare_rgba8_pngs(
                    golden_png_path, pvrgpu_png_path
                )
            except PngCompareError as exc:
                png_match = False
                png_message = f"PNG decode failed: {exc}"
            if png_match:
                result.png = "PASS"
                self._emit_stage(index, "png", "PASS")
            else:
                png_diff_path = case_root / "png_diff.txt"
                atomic_write_text(
                    png_diff_path,
                    "\n".join(
                        (
                            "PNG framebuffer comparison: FAIL",
                            f"reason={png_message}",
                            f"golden_png={result.golden_png}",
                            f"pvrgpu_png={result.pvrgpu_png}",
                        )
                    )
                    + "\n",
                )
                result.png = "FAIL"
                result.png_diff = png_diff_path.name
                failure_reasons.append(png_message)
                self._emit_stage(index, "png", "FAIL")

        if failure_reasons:
            result.status = "FAIL"
            result.stage = "compare" if result.compare == "FAIL" else "png"
            result.reason = "; ".join(failure_reasons)
        else:
            result.status = "PASS"
            result.stage = "compare"
            result.reason = ""
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
            (
                "- PNG comparison: decoded RGBA exact comparison when the "
                "selected frame has measured framebuffer color writes"
            ),
        ]
        if global_reason:
            lines.append(f"- Note: {self._markdown_text(global_reason)}")
        lines.extend(
            [
                "",
                "## Results",
                "",
                "| # | RDC | Case | Golden | PvrGPU | Counter | PNG | Result |",
                "|---:|---|---|---|---|---|---|---|",
            ]
        )
        for result in results:
            artifact = result.artifact_dir
            golden_cell = result.golden
            if result.golden_counter:
                golden_cell += " " + self._markdown_link(
                    "counter", f"{artifact}/{result.golden_counter}"
                )
            if result.golden_png:
                golden_cell += " " + self._markdown_link(
                    "png", f"{artifact}/{result.golden_png}"
                )
            pvrgpu_cell = result.pvrgpu
            if result.pvrgpu_counter:
                pvrgpu_cell += " " + self._markdown_link(
                    "counter", f"{artifact}/{result.pvrgpu_counter}"
                )
            if result.pvrgpu_png:
                pvrgpu_cell += " " + self._markdown_link(
                    "png", f"{artifact}/{result.pvrgpu_png}"
                )
            png_cell = result.png
            if result.png_diff:
                png_cell += " " + self._markdown_link(
                    "diff", f"{artifact}/{result.png_diff}"
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
                        result.compare,
                        png_cell,
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
                        f"- Counter compare: `{self._markdown_text(result.compare)}`",
                        f"- PNG compare: `{self._markdown_text(result.png)}`",
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
            if os.name != "nt" and not os.access(runner, os.X_OK):
                raise BatchSetupError(f"{label} runner is not executable: {runner}")

        manifest = load_manifest(self.manifest_path)
        discovered = discover_rdcs(self.input_root, manifest)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.golden_png_cache_index = {}
        self.golden_counter_cache_index = self._build_golden_counter_cache_index()
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
                entry, derived_metadata = self._entry_for_item(item)
                label = entry.case if entry is not None else "UNMAPPED"
                case_dir_name = f"{index:04d}-{safe_name(label)}-{item.sha256[:8]}"
                case_root = self.run_root / "cases" / case_dir_name
                case_root.mkdir(parents=True, exist_ok=True)
                result = CaseResult(
                    index=index,
                    rdc=item.relative_path,
                    sha256=item.sha256,
                    case=label,
                    manifest_index=(
                        entry.index
                        if entry is not None and not derived_metadata and entry.index > 0
                        else None
                    ),
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
            "Recursively run Golden/PvrGPU counter and framebuffer comparison "
            "for every RDC and write report.md."
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
        default=Path(
            os.environ.get(
                "PVRGPU_RDC_GOLDEN_RUNNER",
                str(default_runner_path(work_root, "llvmpipe")),
            )
        ),
        help="single-RDC Mesa llvmpipe executable",
    )
    parser.add_argument(
        "--pvrgpu-runner",
        type=Path,
        default=Path(
            os.environ.get(
                "PVRGPU_RDC_PVRGPU_RUNNER",
                str(default_runner_path(work_root, "pvrgpu")),
            )
        ),
        help="single-RDC Mesa/Gallium pvrgpu executable",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=0,
        help="per-runner timeout; zero disables the timeout",
    )
    parser.add_argument(
        "--require-manifest",
        action="store_true",
        help=(
            "fail unmapped RDCs instead of deriving replay metadata from the "
            "RDC path or dEQP recorder manifest"
        ),
    )
    parser.add_argument(
        "--reuse-golden-cache",
        action="store_true",
        help=(
            "reuse prior Golden artifacts by RDC SHA; opt in only when the "
            "llvmpipe player, Mesa runtime, and replay policy are unchanged"
        ),
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
        timeout_seconds=options.timeout_seconds,
        require_manifest=options.require_manifest,
        reuse_golden_cache=options.reuse_golden_cache,
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
