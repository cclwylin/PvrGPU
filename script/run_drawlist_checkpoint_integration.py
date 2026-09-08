#!/usr/bin/env python3
"""Generate a tiny COMPLETE RDC; optionally verify real checkpoint continuation.

Capture always uses explicitly selected llvmpipe. Replay may use llvmpipe or a
real PvrGPU bridge. Omitting --player creates and verifies the producer artifact
only; it never claims checkpoint replay passed. Builds just the tiny executable
in a fresh external directory, never Mesa/RenderDoc/the bridge/shared CMake.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import run_drawlist_replay as contract

REPO = Path(__file__).resolve().parents[1]


def expected_colors(depth: bool = False) -> list[bytes]:
    """Independent integer oracle for the documented GLSL math at pixel centers.

    Every seed channel is even, so scale=1/2 produces exact integer UNORM
    codes. Draw1 is A/2+(32,16,8); draw2 is 1-B. For optional shared depth,
    draw0 writes z=.1+.8*(x+.5)/8; draw1's z=.5 passes LESS only for x>=4.
    Rejected draw1 pixels retain its pre-capture initialized (12,20,28,255).
    """
    colors = [bytearray(), bytearray(), bytearray()]
    for y in range(8):
        for x in range(8):
            a = (8 + 16 * x, 24 + 20 * y, 16 + 8 * ((x + 2 * y) % 8))
            b = tuple(channel // 2 + bias for channel, bias in zip(a, (32, 16, 8)))
            if depth and x < 4:
                b = (12, 20, 28)
            c = tuple(255 - channel for channel in b)
            for target, rgb in zip(colors, (a, b, c)):
                target.extend((*rgb, 255))
    return [bytes(color) for color in colors]


def compare_color(path: Path, expected: bytes) -> dict:
    identity = contract.file_identity(path, single_link=True)
    observed = path.read_bytes()
    equal = observed == expected
    return dict(identity, equal=equal, expected_sha256=hashlib.sha256(expected).hexdigest(),
                differing_bytes=sum(a != b for a, b in zip(observed, expected)) + abs(len(observed) - len(expected)))


def compare_checkpoint_states(uninterrupted: Path, resumed: Path) -> dict:
    """Exact functional archive invariant for this deterministic tiny fixture.

    Not a production cross-driver acceptance gate: unrelated program linkages
    can use different numeric uniform locations while remaining equivalent.
    Here both runs use the same capture, renderer and frozen runtime, so the
    entire resource/context archive must be equal, independently of its PNG.
    """
    first = contract.file_identity(uninterrupted, single_link=True)
    second = contract.file_identity(resumed, single_link=True)
    equal = True
    with uninterrupted.open("rb") as left, resumed.open("rb") as right:
        while True:
            a, b = left.read(1024 * 1024), right.read(1024 * 1024)
            if a != b:
                equal = False
                break
            if not a:
                break
    contract.require(contract.file_identity(uninterrupted, single_link=True) == first and
                     contract.file_identity(resumed, single_link=True) == second,
                     "checkpoint archive changed during comparison")
    return {"uninterrupted": first, "resumed": second, "byte_equal": equal,
            "scope": "deterministic same-capture same-runtime fixture; not a cross-driver production gate"}


def checked_run(command: list[str], env: dict, out: Path, timeout: float) -> None:
    contract.run_child(command, env, out, timeout)


def native_fixture_work(run_out: Path, expected_draws: int) -> dict:
    """Independent command-count check for THIS explicit three-draw producer.

    This is test evidence, never a production submission limit or an estimate
    from a customer's capture name. The engine separately audits completion.
    """
    def events(path: Path) -> list[str]:
        text = path.read_text()
        contract.require(not text or text.endswith("\n"), "truncated fixture driver log")
        result = []
        for line in text.splitlines():
            if not line.strip():
                continue
            fields = dict(token.split("=", 1) for token in line.split() if "=" in token)
            contract.require(fields.get("schema") == "pvrgpu.driver-counter.v1" and
                             fields.get("producer") == "pvrgpu-gallium-driver" and fields.get("event"),
                             "invalid fixture driver event")
            result.append(fields["event"])
        return result
    actual = events(run_out / "driver-counter.txt")
    recorded = actual.count("draw_array_primitive_recorded")
    contract.require(recorded == expected_draws,
                     f"fixture replayed {recorded} accepted draws, expected exactly {expected_draws}")
    contract.require(actual.count("systemc_api_submit") > 0 and actual.count("systemc_api_done") > 0,
                     "fixture has no actual native submission/completion")
    contract.require(not any(event.startswith("compute_api_") for event in actual),
                     "graphics-only fixture unexpectedly dispatched compute")
    helpers = {}
    for name in ("restore", "save"):
        helper = events(run_out / f"replay.json.{name}-driver.txt")
        contract.require(not any(event.startswith(("draw", "systemc_api_", "compute_api_")) or
                                 event == "launch_grid" for event in helper),
                         f"{name} helper executed native/shader work")
        helpers[name] = {"events": len(helper), "native_draw_dispatch_events": 0}
    return {"expected_accepted_draws": expected_draws, "actual_accepted_draws": recorded,
            "api_submissions": actual.count("systemc_api_submit"), "helper_scopes": helpers}


def execute(args: argparse.Namespace) -> tuple[Path, bool]:
    out = contract.safe_path(args.outdir)
    contract.require(not out.exists() and out.parent.is_dir(), "outdir must be fresh with existing parent")
    contract.require(out != REPO and REPO not in out.parents, "generated fixture must stay outside repository")
    source = contract.safe_path(args.renderdoc_source)
    contract.require((source / "renderdoc/api/app/renderdoc_app.h").is_file(), "RenderDoc app header missing")
    runtime = contract.runtime(SimpleNamespace(player=sys.executable, renderdoc_lib=args.renderdoc_lib,
        mesa_prefix=args.mesa_prefix, backend="llvmpipe", bridge=None, allow_model_change=False))
    out.mkdir(mode=0o700)
    binary = out / "checkpoint-capture"
    build = [args.cxx, "-std=c++17", "-O1", "-g", "-Wall", "-Wextra", "-Wpedantic",
             "-isystem", str(Path(runtime["mesa_prefix"]) / "include"),
             "-isystem", str(source / "renderdoc"), "-I" + str(REPO / "src"),
             "-I" + str(REPO / "tools"), str(REPO / "tests/drawlist_checkpoint_capture.cpp"),
             str(REPO / "src/rdc_runner/sha256.cpp"), runtime["renderdoc"]["path"],
             "-Wl,-rpath," + str(Path(runtime["renderdoc"]["path"]).parent), "-o", str(binary)]
    if sys.platform != "darwin":
        build.append("-ldl")
    with (out / "build.stdout.log").open("xb") as stdout, (out / "build.stderr.log").open("xb") as stderr:
        result = subprocess.run(build, stdout=stdout, stderr=stderr, timeout=120)
    contract.require(result.returncode == 0, f"fixture build failed; inspect {out / 'build.stderr.log'}")
    runtime["player"] = contract.file_identity(binary)
    capture_out = out / "capture"
    capture_out.mkdir(mode=0o700)
    for name in ("dri", "tmp", "cache", "model"):
        (capture_out / name).mkdir(mode=0o700)
    env = contract.child_environment(SimpleNamespace(backend="llvmpipe"), runtime, capture_out)
    command = [str(binary), str(capture_out), "1" if args.depth else "0", "1" if args.hidden_mip else "0"]
    contract.atomic_json(out / "build-and-capture.json", {"build": build, "capture_command": command,
        "environment": env, "runtime": runtime,
        "sources": [contract.file_identity(REPO / name) for name in
                    ("tests/drawlist_checkpoint_capture.cpp", "tools/drawlist-color-readback.h",
                     "script/run_drawlist_checkpoint_integration.py")]})
    checked_run(command, env, capture_out, args.timeout)
    producer = contract.read_json(capture_out / "capture.json")
    contract.require(producer.get("schema") == "pvrgpu.checkpoint-fixture.v1" and producer.get("draws") == 3 and
                     producer.get("depth") is args.depth and producer.get("readback_state_restored") is True and
                     producer.get("hidden_mip") is args.hidden_mip and producer.get("tail_poisoned_prefix_texture") is True and
                     producer.get("readback_negative_checks") == 4 and
                     producer.get("renderer", "").lower().startswith("llvmpipe"),
                     "unexpected producer receipt")
    capture = contract.safe_path(producer["capture"])
    contract.require(capture.parent == capture_out and capture.suffix == ".rdc", "capture escaped output directory")
    capture_id = contract.file_identity(capture)
    expected = expected_colors(args.depth)
    producer_comparisons = [compare_color(capture_out / f"draw{i}.rgba", expected[i]) for i in range(3)]
    contract.require(all(row["equal"] for row in producer_comparisons), "producer GL result disagrees with CPU oracle")
    for key in contract.RUNTIME_FILES:
        if runtime[key] is not None:
            contract.require(contract.file_identity(Path(runtime[key]["path"])) == runtime[key],
                             f"capture runtime changed: {key}")
    report = {"schema": "pvrgpu.checkpoint-integration.v1", "capture": capture_id,
              "depth": args.depth, "hidden_mip": args.hidden_mip, "capture_runtime": runtime, "producer": producer,
              "producer_comparisons": producer_comparisons,
              "checkpoint_verified": False, "status": "capture_created_only"}
    if args.player:
        replay_prefix = args.replay_mesa_prefix or args.mesa_prefix
        runs: list[dict] = []
        for name, draw, resume_name in (("uninterrupted2", 2, None), ("save0", 0, None),
                                      ("resume1", 1, "save0"), ("resume2", 2, "resume1")):
            run_out = out / name
            replay = [sys.executable, str(REPO / "script/run_drawlist_replay.py"), str(capture),
                "--player", args.player, "--renderdoc-lib", args.replay_renderdoc_lib or args.renderdoc_lib,
                "--mesa-prefix", replay_prefix, "--backend", args.backend,
                "--through-draw", str(draw), "--outdir", str(run_out), "--timeout", str(args.timeout), "--verify-color"]
            if args.backend == "pvrgpu":
                contract.require(args.bridge is not None, "native replay requires explicit --bridge")
                replay += ["--bridge", args.bridge]
            if resume_name:
                replay += ["--resume", str(out / resume_name / "snapshot")]
            with (out / (name + ".stdout.log")).open("xb") as stdout, (out / (name + ".stderr.log")).open("xb") as stderr:
                # The wrapper owns the replay child's separate process group
                # and its timeout cleanup. Killing just that wrapper with an
                # outer timeout could orphan a still-running native replay.
                result = subprocess.run(replay, stdout=stdout, stderr=stderr)
            contract.require(result.returncode == 0, f"{name} failed; inspect {out / (name + '.stderr.log')}")
            receipt = contract.read_json(run_out / "replay.json")
            contract.require(receipt.get("trace_draw_actions") == 3, "fixture must replay exactly three original Drawcall actions")
            comparison = compare_color(run_out / "color.rgba", expected[draw])
            contract.require(comparison["equal"], f"{name} output disagrees with independent CPU oracle")
            work = native_fixture_work(run_out, 3 if name == "uninterrupted2" else 1) if args.backend == "pvrgpu" else None
            runs.append({"name": name, "argv": replay, "receipt": receipt, "comparison": comparison,
                         "independent_native_work": work})
        continuous = (out / "uninterrupted2/color.rgba").read_bytes()
        resumed = (out / "resume2/color.rgba").read_bytes()
        contract.require(continuous == resumed, "uninterrupted vs resumed output mismatch")
        state_comparison = compare_checkpoint_states(out / "uninterrupted2/snapshot/state.bin",
                                                      out / "resume2/snapshot/state.bin")
        contract.require(state_comparison["byte_equal"], "uninterrupted vs resumed complete state mismatch")
        report.update(status="PASS", checkpoint_verified=True, runs=runs,
                      uninterrupted_resumed_raw_equal=True, cold_cache=True,
                      complete_state_comparison=state_comparison,
                      comparison_scope="CPU-oracle RGBA8 at all boundaries and exact final complete resource/context archive")
    contract.atomic_json(out / "integration.json", report)
    return out / "integration.json", report["checkpoint_verified"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesa-prefix", required=True, help="llvmpipe capture prefix")
    parser.add_argument("--renderdoc-source", required=True, help="headers for explicit capture RenderDoc library")
    parser.add_argument("--renderdoc-lib", required=True, help="capture RenderDoc EGL/GLES interception library")
    parser.add_argument("--outdir", required=True, help="new external output directory")
    parser.add_argument("--depth", action="store_true", help="also consume a shared depth attachment across draws")
    parser.add_argument("--hidden-mip", action="store_true", help="draw into a hidden physical mip, then expose and consume it after restore")
    parser.add_argument("--cxx", default="/usr/local/opt/llvm/bin/clang++" if sys.platform == "darwin" else "c++")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--player", help="snapshot-capable player; omit to generate capture only")
    parser.add_argument("--replay-renderdoc-lib", help="snapshot extension library linked by --player; defaults to capture library")
    parser.add_argument("--replay-mesa-prefix", help="replay Mesa prefix; defaults to capture prefix")
    parser.add_argument("--backend", choices=("pvrgpu", "llvmpipe"), default="llvmpipe")
    parser.add_argument("--bridge", help="required for native replay")
    try:
        path, verified = execute(parser.parse_args())
    except (contract.ReplayError, OSError, ValueError, TypeError, KeyError, subprocess.TimeoutExpired) as error:
        print(f"checkpoint integration: {error}", file=sys.stderr)
        return 1
    print(f"{'PASS checkpoint continuation' if verified else 'CAPTURE ONLY; checkpoint not yet tested'}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
