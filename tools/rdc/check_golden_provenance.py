#!/usr/bin/env python3
"""Validate the frozen RDC suite and llvmpipe Golden provenance.

The Golden runner deliberately keeps selection/presentation fields separate
from semantic counters.  Frame, Marker, and Frame selection markers are
recorded as provenance, but they are not part of the 17-field exact counter
comparison.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import sys
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from counter_protocol import (  # noqa: E402
    STANDARD_COUNTER_FIELDS,
    parse_markdown_report,
)


SUITE = "rdc-glbench-v1"
INPUT_SCHEMA = "pvrgpu.rdc.input-provenance.v1"
RUNTIME_SCHEMA = "pvrgpu.rdc.golden-runtime.v1"
COUNTER_SCHEMA = "pvrgpu.rdc.golden-counter-self-check.v1"
SHA256_RE = re.compile(r"[0-9a-f]{64}")

EXPECTED_CASES = (
    "fill_solid",
    "fill_solid_depth_never",
    "fill_solid_depth_neq",
    "fill_solid_blended",
    "triangle_setup",
    "triangle_setup_all_culled",
    "triangle_setup_half_culled",
    "attribute_fetch_shader",
    "attribute_fetch_shader_2_attr",
    "attribute_fetch_shader_4_attr",
    "attribute_fetch_shader_8_attr",
    "varyings_shader_1",
    "varyings_shader_2",
    "varyings_shader_4",
    "varyings_shader_8",
    "fill_tex_nearest",
    "fill_tex_bilinear",
    "fill_tex_trilinear_linear_01",
    "fill_tex_trilinear_linear_04",
    "fill_tex_trilinear_linear_05",
)

EXPECTED_HEADER = (
    "index",
    "case",
    "rdc_path",
    "rdc_sha256",
    "recorder_png_path",
    "recorder_png_sha256",
    "recorder_report_path",
    "recorder_report_sha256",
    "target_event_policy",
    "output_attachment_policy",
    "width",
    "height",
    "format",
    "samples",
    "color_space",
    "compare_policy",
    "counter_policy",
    "determinism_policy",
    "feature_gate",
)

ASSETS = (
    ("rdc_path", "rdc_sha256"),
    ("recorder_png_path", "recorder_png_sha256"),
    ("recorder_report_path", "recorder_report_sha256"),
)

FIXED_POLICIES = {
    "target_event_policy": "final-leaf-event-v1",
    "output_attachment_policy": "bound-draw-fbo-color-v1",
    "width": "512",
    "height": "512",
    "format": "R8G8B8A8_UNORM",
    "samples": "1",
    "color_space": "png-decoded-rgba8",
    "compare_policy": "rgba8-exact-regression-v1",
    "counter_policy": "llvmpipe-17-exact-selection-normalized-v1",
    "determinism_policy": "two-cold-replays-exact-v1",
}


def sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(output_path: Path, payload: Any) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)


def checked_relative_asset(root: Path, relative_text: str) -> Path:
    relative = PurePosixPath(relative_text)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != relative_text:
        raise ValueError(f"unsafe or non-canonical manifest path: {relative_text!r}")
    resolved_root = root.resolve(strict=True)
    resolved_asset = (resolved_root / Path(*relative.parts)).resolve(strict=True)
    try:
        resolved_asset.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"manifest asset escapes RDC root: {relative_text!r}") from exc
    if not resolved_asset.is_file():
        raise ValueError(f"manifest asset is not a regular file: {resolved_asset}")
    return resolved_asset


def expected_asset_paths(case_name: str) -> dict[str, str]:
    return {
        "rdc_path": f"{case_name}/recorder/trace/{case_name}_capture_1.rdc",
        "recorder_png_path": f"{case_name}/recorder/png/{case_name}_sample_000001.png",
        "recorder_report_path": f"{case_name}/recorder/Report.md",
    }


def read_and_validate_manifest(manifest_path: Path, rdc_root: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if tuple(reader.fieldnames or ()) != EXPECTED_HEADER:
            raise ValueError(
                "manifest header mismatch: "
                f"expected {EXPECTED_HEADER!r}, got {tuple(reader.fieldnames or ())!r}"
            )
        rows = list(reader)

    if len(rows) != len(EXPECTED_CASES):
        raise ValueError(f"manifest must contain exactly 20 cases, found {len(rows)}")

    seen_assets: set[str] = set()
    seen_rdc_hashes: set[str] = set()
    for expected_index, (expected_case, row) in enumerate(
        zip(EXPECTED_CASES, rows, strict=True), start=1
    ):
        if any(row.get(field, "") == "" for field in EXPECTED_HEADER):
            raise ValueError(f"manifest row {expected_index} contains an empty field")
        if row["index"] != str(expected_index) or row["case"] != expected_case:
            raise ValueError(
                f"manifest order mismatch at index {expected_index}: "
                f"index={row['index']!r}, case={row['case']!r}"
            )

        expected_paths = expected_asset_paths(expected_case)
        for field, expected_path in expected_paths.items():
            if row[field] != expected_path:
                raise ValueError(
                    f"{expected_case}: {field} must be {expected_path!r}, got {row[field]!r}"
                )
        for field, expected_value in FIXED_POLICIES.items():
            if row[field] != expected_value:
                raise ValueError(
                    f"{expected_case}: {field} must be {expected_value!r}, got {row[field]!r}"
                )

        for path_field, hash_field in ASSETS:
            expected_hash = row[hash_field]
            if SHA256_RE.fullmatch(expected_hash) is None:
                raise ValueError(f"{expected_case}: invalid {hash_field}: {expected_hash!r}")
            relative_path = row[path_field]
            if relative_path in seen_assets:
                raise ValueError(f"duplicate manifest asset path: {relative_path}")
            seen_assets.add(relative_path)
            asset_path = checked_relative_asset(rdc_root, relative_path)
            actual_hash = sha256_file(asset_path)
            if actual_hash != expected_hash:
                raise ValueError(
                    f"{expected_case}: {path_field} SHA-256 mismatch: "
                    f"expected {expected_hash}, got {actual_hash}"
                )

        if row["rdc_sha256"] in seen_rdc_hashes:
            raise ValueError(f"duplicate RDC SHA-256 at {expected_case}")
        seen_rdc_hashes.add(row["rdc_sha256"])

    return rows


def command_manifest(options: argparse.Namespace) -> None:
    manifest_path = options.manifest.resolve(strict=True)
    rdc_root = options.rdc_root.resolve(strict=True)
    rows = read_and_validate_manifest(manifest_path, rdc_root)
    selected = next((row for row in rows if row["case"] == options.case), None)
    if selected is None:
        raise ValueError(f"case is not in the frozen suite: {options.case!r}")

    selected_assets = {
        path_field: str(checked_relative_asset(rdc_root, selected[path_field]))
        for path_field, _ in ASSETS
    }
    payload = {
        "schema": INPUT_SCHEMA,
        "suite": SUITE,
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "rdc_root": str(rdc_root),
        "verified_assets": len(rows) * len(ASSETS),
        "verified_cases": len(rows),
        "ordered_cases": list(EXPECTED_CASES),
        "selected": {**selected, "resolved_assets": selected_assets},
    }
    write_json_atomic(options.output, payload)
    print(
        f"MANIFEST_OK suite={SUITE} cases={len(rows)} assets={len(rows) * len(ASSETS)} "
        f"selected={options.case}"
    )


def checked_runtime_file(file_path: Path, label: str) -> dict[str, str]:
    resolved = file_path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file: {resolved}")
    return {
        "path": str(file_path.expanduser().absolute()),
        "resolved_path": str(resolved),
        "sha256": sha256_file(resolved),
    }


def one_glob(parent: Path, pattern: str, label: str) -> Path:
    matches = sorted(parent.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected one {label} matching {parent / pattern}, found {len(matches)}")
    return matches[0]


def command_runtime(options: argparse.Namespace) -> None:
    renderdoc_root = options.renderdoc_root.expanduser().resolve(strict=True)
    mesa_prefix = options.mesa_prefix.expanduser().resolve(strict=True)
    files = {
        "suite_manifest": checked_runtime_file(options.manifest, "suite manifest"),
        "benchscope_glbench_wrapper": checked_runtime_file(
            options.benchscope_script, "BenchScope GLBench replay wrapper"
        ),
        "benchscope_player_wrapper": checked_runtime_file(
            options.gfx_player_script, "BenchScope RenderDoc/Mesa player wrapper"
        ),
        "renderdoc_mesa_player": checked_runtime_file(
            renderdoc_root / "bin/renderdoc-mesa-player", "RenderDoc Mesa player"
        ),
        "renderdoc_real_egl": checked_runtime_file(
            renderdoc_root / "real/libMesaEGL.dylib", "RenderDoc real Mesa EGL"
        ),
        "renderdoc_real_gles": checked_runtime_file(
            renderdoc_root / "real/libMesaGLESv2.dylib", "RenderDoc real Mesa GLES"
        ),
        "mesa_egl": checked_runtime_file(mesa_prefix / "lib/libEGL.1.dylib", "Mesa EGL"),
        "mesa_gles": checked_runtime_file(
            mesa_prefix / "lib/libGLESv2.2.dylib", "Mesa GLES"
        ),
        "mesa_gallium": checked_runtime_file(
            one_glob(mesa_prefix / "lib", "libgallium-*.dylib", "Mesa Gallium library"),
            "Mesa Gallium library",
        ),
        "mesa_dri": checked_runtime_file(
            mesa_prefix / "lib/dri/libdril_dri.dylib", "Mesa DRI driver"
        ),
    }
    payload = {
        "schema": RUNTIME_SCHEMA,
        "suite": SUITE,
        "platform": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "paths": {
            "mesa_prefix": str(mesa_prefix),
            "renderdoc_root": str(renderdoc_root),
        },
        "runtime_guards": {
            "EGL_PLATFORM": "surfaceless",
            "GALLIUM_DRIVER": "llvmpipe",
            "LIBGL_ALWAYS_SOFTWARE": "1",
            "MESA_LOADER_DRIVER_OVERRIDE": "swrast",
            "MESA_SHADER_CACHE_DISABLE": "true",
            "RENDERDOC_OUTPUT_FROM_DRAW_FBO": "1",
        },
        "files": files,
    }
    write_json_atomic(options.output, payload)
    print(f"RUNTIME_PROVENANCE_OK files={len(files)} renderer_guard=llvmpipe")


def report_record(report_path: Path) -> tuple[Any, dict[str, str]]:
    parsed = parse_markdown_report(report_path)
    renderer = parsed.metadata.get("Renderer", "")
    if "llvmpipe" not in renderer.lower():
        raise ValueError(f"report renderer is not llvmpipe: {report_path}: {renderer!r}")
    if len(parsed.records) != 1:
        raise ValueError(f"report must contain exactly one frame: {report_path}")
    record = parsed.records[0]
    actual_fields = set(record.values)
    expected_fields = set(STANDARD_COUNTER_FIELDS)
    if actual_fields != expected_fields:
        missing = sorted(expected_fields - actual_fields)
        extra = sorted(actual_fields - expected_fields)
        raise ValueError(
            f"report does not have the exact 17 semantic fields: {report_path}: "
            f"missing={missing}, extra={extra}"
        )
    return parsed, {
        "frame": str(record.frame),
        "marker": record.marker,
        "frame_selection_markers": parsed.metadata.get("Frame selection markers", ""),
    }


def equal_metadata(reports: Iterable[Any]) -> dict[str, str]:
    reports = tuple(reports)
    fields = ("Mesa", "Renderer", "Counter owner", "Schema")
    reference = {field: reports[0].metadata.get(field, "") for field in fields}
    for report in reports[1:]:
        candidate = {field: report.metadata.get(field, "") for field in fields}
        if candidate != reference:
            raise ValueError(
                f"semantic report metadata mismatch: reference={reference!r}, "
                f"candidate={candidate!r}"
            )
    return reference


def command_counters(options: argparse.Namespace) -> None:
    report_paths = [options.recorder, *options.replay]
    if len(options.replay) != 2:
        raise ValueError("the determinism gate requires exactly two cold replay reports")

    parsed_reports = []
    ignored_selection = []
    for role, report_path in zip(("recorder", "run-01", "run-02"), report_paths, strict=True):
        resolved = report_path.resolve(strict=True)
        parsed, ignored = report_record(resolved)
        parsed_reports.append(parsed)
        ignored_selection.append(
            {
                "role": role,
                "report": str(resolved),
                **ignored,
            }
        )

    metadata = equal_metadata(parsed_reports)
    reference_values = dict(parsed_reports[0].records[0].values)
    for role, report in zip(("run-01", "run-02"), parsed_reports[1:], strict=True):
        candidate_values = dict(report.records[0].values)
        if candidate_values != reference_values:
            mismatches = {
                field: {
                    "recorder": reference_values[field],
                    role: candidate_values[field],
                }
                for field in STANDARD_COUNTER_FIELDS
                if candidate_values[field] != reference_values[field]
            }
            raise ValueError(f"{options.case}: semantic counter mismatch: {mismatches}")

    payload = {
        "schema": COUNTER_SCHEMA,
        "suite": SUITE,
        "case": options.case,
        "policy": "llvmpipe-17-exact-selection-normalized-v1",
        "result": "MATCH",
        "compared_fields": list(STANDARD_COUNTER_FIELDS),
        "values": reference_values,
        "semantic_metadata": metadata,
        "ignored_selection_provenance": ignored_selection,
        "reports": [
            {
                "role": role,
                "path": str(report_path.resolve(strict=True)),
                "sha256": sha256_file(report_path.resolve(strict=True)),
            }
            for role, report_path in zip(
                ("recorder", "run-01", "run-02"), report_paths, strict=True
            )
        ],
    }
    write_json_atomic(options.output, payload)
    print(
        f"COUNTERS_MATCH case={options.case} fields={len(STANDARD_COUNTER_FIELDS)} "
        "replays=2 selection_fields=ignored"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    manifest = commands.add_parser("manifest", help="verify all 20 frozen input capsules")
    manifest.add_argument("--manifest", type=Path, required=True)
    manifest.add_argument("--rdc-root", type=Path, required=True)
    manifest.add_argument("--case", required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.set_defaults(function=command_manifest)

    runtime = commands.add_parser("runtime", help="hash the pinned Golden runtime")
    runtime.add_argument("--manifest", type=Path, required=True)
    runtime.add_argument("--benchscope-script", type=Path, required=True)
    runtime.add_argument("--gfx-player-script", type=Path, required=True)
    runtime.add_argument("--renderdoc-root", type=Path, required=True)
    runtime.add_argument("--mesa-prefix", type=Path, required=True)
    runtime.add_argument("--output", type=Path, required=True)
    runtime.set_defaults(function=command_runtime)

    counters = commands.add_parser(
        "counters", help="compare recorder and two replay reports"
    )
    counters.add_argument("--case", required=True)
    counters.add_argument("--recorder", type=Path, required=True)
    counters.add_argument("--replay", type=Path, action="append", required=True)
    counters.add_argument("--output", type=Path, required=True)
    counters.set_defaults(function=command_counters)
    return parser


def main() -> int:
    parser = build_parser()
    options = parser.parse_args()
    try:
        options.function(options)
    except (OSError, ValueError) as exc:
        print(f"PROVENANCE_FAIL command={options.command}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
