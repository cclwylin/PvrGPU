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
import re
import subprocess
import sys
from types import SimpleNamespace

import run_drawlist_replay as contract

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import inspect_drawlist_snapshot as snapshot_inventory


def expected_colors(depth: bool = False, depth_renderbuffer: bool = False,
                    depth_low_codes: bool = False, rgb9e5: bool = False) -> list[bytes]:
    """Independent integer oracle for the documented GLSL math at pixel centers.

    Every seed channel is even, so scale=1/2 produces exact integer UNORM
    codes. Draw1 is A/2+(32,16,8); draw2 is 1-B. For optional shared depth,
    draw0 writes z=.1+.8*(x+.5)/8; draw1's z=.5 passes LESS only for x>=4.
    Rejected draw1 pixels retain its pre-capture initialized (12,20,28,255).

    The separate 7x5 D16-RB fixture writes depth classes 1/8,3/8,5/8,7/8.
    Draw1 at 1/2 passes classes2/3. Draw2 at 1/4 (classes0..2) or3/4
    (class3) passes only classes1/2 after draw1 updated the same RB. Class3
    would incorrectly pass if FBO aliases diverged or save1 lost the update.
    """
    contract.require(sum((depth, depth_renderbuffer, depth_low_codes, rgb9e5)) <= 1, "fixture modes are mutually exclusive")
    if rgb9e5:
        colors = [bytearray() for _ in range(3)]
        for y in range(8):
            for x in range(8):
                a, b = rgb9_codes(0, x, y), rgb9_codes(1, x, y)
                for target, rgb in zip(colors, (tuple(32*k for k in a), tuple(24*k for k in a),
                                                tuple(12*k+8*v for k, v in zip(a, b)))):
                    target.extend((*rgb, 255))
        return [bytes(color) for color in colors]
    aliased_depth = depth_renderbuffer or depth_low_codes
    width, height = (7, 5) if aliased_depth else (8, 8)
    colors = [bytearray(), bytearray(), bytearray()]
    for y in range(height):
        for x in range(width):
            a = (8 + 16 * x, 24 + 20 * y, 16 + 8 * ((x + 2 * y) % 8))
            b = tuple(channel // 2 + bias for channel, bias in zip(a, (32, 16, 8)))
            cell = (3 * x + 5 * y) % 4
            if (depth and x < 4) or (aliased_depth and cell < 2):
                b = (12, 20, 28)
            c = tuple(255 - channel for channel in b)
            if aliased_depth and cell in (0, 3):
                c = (12, 20, 28)
            for target, rgb in zip(colors, (a, b, c)):
                target.extend((*rgb, 255))
    return [bytes(color) for color in colors]


def rgb9_codes(seed: int, x: int, y: int) -> tuple[int, int, int]:
    """Independent exact values in units of 1/8; not a float repacking oracle."""
    values = (0, 1, 2, 4)
    return ((0, 0, 0) if (x+3*y+seed) % 11 == 0 else
            (values[(x+y+seed) % 4], values[(x+2*y+seed+1) % 4], values[(2*x+y+seed+2) % 4]))


def expected_rgb9e5_words(seed: int, level: int) -> bytes:
    contract.require(seed in (0, 1) and level in (0, 1), "RGB9E5 fixture input index")
    result = bytearray()
    for y in range(16 >> level):
        for x in range(16 >> level):
            codes = rgb9_codes(seed, x, y)
            shift = (x+y+seed) % 4
            word = ((31 if not any(codes) else 15+shift) << 27)
            for channel, value in enumerate(codes):
                word |= ((value*64) >> shift) << (9*channel)
            result.extend(word.to_bytes(4, "little"))
    return bytes(result)


def rgb9e5_mips(identity: int, signature: bytes, native: bytes, section: int) -> list[bytes]:
    """Locked RenderDoc TextureStateInitialData parser for this 2D/two-mip test.

    Every array length, native identity, body span and alignment is checked;
    no pixels/ResourceIds are located by searching for expected payloads.
    This is a fixture verifier, not a production texture codec.
    """
    contract.require(section in (0x23, 0x24) and len(signature) == 20*8, "RGB9E5 signature/section")
    shape = [int.from_bytes(signature[i:i+8], "little") for i in range(0, len(signature), 8)]
    contract.require(shape[:8] == [2, 0x0de1, 0x8c3d, 16, 16, 1, 0, 2] and
                     shape[10:12] == [3, 2] and  # mipsValid bitmask, physical mip count
                     shape[12:] == [16, 16, 1, 0x8c3d, 8, 8, 1, 0x8c3d], "RGB9E5 physical shape")
    n = snapshot_inventory.Reader(native)
    def integer(width: int = 8) -> int:
        return int.from_bytes(n.raw(width), "little")
    contract.require(integer(4) == 3, "RGB9E5 native chunk")
    body_length = integer(4)
    contract.require((integer(), integer(4), integer(4), integer(1)) ==
                     (identity, 2, 0x8c3d, 0), "RGB9E5 native identity/format/view")
    contract.require([integer(4) for _ in range(7)] == [16, 16, 1, 0, 2, 0x0de1, 2], "RGB9E5 native shape")
    n.raw(11*4)
    for size in (4, 3, 4):
        contract.require(integer() == size, "RGB9E5 fixed-array count")
        n.raw(size*4)
    n.raw(4)
    contract.require((integer(), integer(4), integer(4)) == (0, 0, 0), "RGB9E5 buffer metadata")
    n.raw(4)
    contract.require(n.at == 189, "RGB9E5 fixed metadata span")
    result = []
    for level in range(2):
        size = integer()
        contract.require(size == (16 >> level)**2*4, "RGB9E5 mip raw size")
        padding = (-n.at) % 64
        contract.require(bytes(n.raw(padding)) == bytes(padding), "RGB9E5 raw alignment padding")
        result.append(bytes(n.raw(size)))
    contract.require(n.at == body_length+8, "RGB9E5 native body span")
    padding = (-n.at) % 64
    contract.require(bytes(n.raw(padding)) == bytes(padding) and n.at == len(native), "RGB9E5 native trailing padding")
    return result


def verify_rgb9e5_checkpoints(out: Path, capture_sha: str) -> dict:
    """Check both complete physical mips and stable identities at save0/save1."""
    results, first_identity = [], None
    for name in ("save0", "resume1"):
        path = out / name / "snapshot/state.bin"
        pin = contract.file_identity(path, single_link=True)
        inventory = snapshot_inventory.inspect(path)
        contract.require(inventory["capture_sha256"] == capture_sha and inventory["snapshot_api_version"] == 2,
                         "RGB9E5 checkpoint capture/version identity")
        r = snapshot_inventory.Reader(path.read_bytes()[48:])
        r.text(); section = r.u64(); r.u64(); r.u64()
        for _ in range(3): r.text()
        r.blob()
        for _ in range(25): r.u64()
        resources = []
        for _ in range(r.u64()):
            identity, namespace, signature = r.u64(), r.u64(), bytes(r.blob())
            for _ in range(r.u64()*2): r.u64()
            native = bytes(r.blob())
            if namespace == 2 and signature[:24] == b"".join(v.to_bytes(8, "little") for v in (2, 0x0de1, 0x8c3d)):
                resources.append((identity, rgb9e5_mips(identity, signature, native, section)))
        contract.require(r.at == len(r.data) and len(resources) == 2, "RGB9E5 fixture resource inventory")
        image0 = [row for row in resources if row[1] == [expected_rgb9e5_words(0, level) for level in range(2)]]
        contract.require(len(image0) == 1, "RGB9E5 saved input0 packed bits differ")
        if first_identity is None:
            first_identity = image0[0][0]
        contract.require(image0[0][0] == first_identity, "RGB9E5 restored resource identity changed")
        image1 = [row for row in resources if row[0] != first_identity][0]
        expected1 = [bytes((16 >> level)**2*4) if name == "save0" else expected_rgb9e5_words(1, level) for level in range(2)]
        contract.require(image1[1] == expected1, "RGB9E5 saved input1 packed bits differ")
        contract.require(contract.file_identity(path, single_link=True) == pin, "RGB9E5 archive changed during verification")
        results.append({"name": name, "archive": pin, "resources": [
            {"id": identity, "mips": [{"level": level, "size_bytes": len(raw),
             "sha256": hashlib.sha256(raw).hexdigest(), "byte_equal": True} for level, raw in enumerate(mips)]}
            for identity, mips in resources]})
    contract.require({r["id"] for r in results[0]["resources"]} == {r["id"] for r in results[1]["resources"]},
                     "RGB9E5 resource identities did not persist")
    return {"scope": "exact locked native mip arrays, including noncanonical shared exponents", "checkpoints": results}


def expected_low_depth_codes() -> list[bytes]:
    """Integer D24 LESS/write oracle; GL_UINT expansion for these codes is <<8.

    All codes are below 65536, so the low repeated bits of the normalized
    24-to-32-bit conversion are zero. This does not emulate float upload.
    """
    result = [bytearray() for _ in range(3)]
    for y in range(5):
        for x in range(7):
            cell = (3 * x + 5 * y) % 4
            stored = (1, 159, 256, 65535)[cell]
            result[0].extend((stored << 8).to_bytes(4, "little"))
            for draw, incoming in ((1, 160), (2, 200 if cell == 3 else 100)):
                if incoming < stored:
                    stored = incoming
                result[draw].extend((stored << 8).to_bytes(4, "little"))
    return [bytes(value) for value in result]


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


def capture_runtime(args: argparse.Namespace) -> dict:
    # This fixture creates an explicit GLES3.1 context and GLSL310 program.
    # Keep the runtime-v2 version field explicit, including capture-only runs.
    return contract.runtime(SimpleNamespace(player=sys.executable, renderdoc_lib=args.renderdoc_lib,
        mesa_prefix=args.mesa_prefix, backend="llvmpipe", bridge=None,
        allow_model_change=False, gles_version="3.1"))


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
    model_text = (run_out / "model.jsonl").read_text()
    contract.require(model_text.endswith("\n"), "missing/truncated fixture model log")
    counters = []
    capture_markers = 0
    for line in model_text.splitlines():
        if not line.strip():
            continue
        # The native logger emits this explicit PNG notice between its JSON
        # counter and done records. It is not a physical-draw counter. Never
        # discard other non-JSON text, truncated notices, or arbitrary markers.
        marker = re.fullmatch(r"@CAPTURE: driver_pco_triangles sample=([1-9][0-9]*) "
                              r"png=driver_pco_triangles_sample_([0-9]{6,})\.png", line)
        if marker:
            contract.require(int(marker[1]) == int(marker[2]), "inconsistent fixture capture marker")
            capture_markers += 1
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise contract.ReplayError("invalid fixture model log line") from error
        contract.require(isinstance(row, dict) and row.get("type") in ("hello", "counter", "done"),
                         "invalid fixture model record")
        if row["type"] == "counter":
            counters.append(row)
    contract.require(counters and all(row.get("source") == "pvrgpu-systemc" and
                     isinstance(row.get("counters"), dict) and
                     type(row["counters"].get("drawlists")) is int and
                     row["counters"]["drawlists"] >= 0 for row in counters),
                     "invalid fixture physical-draw counter")
    physical_draws = sum(row["counters"]["drawlists"] for row in counters)
    contract.require(physical_draws == expected_draws,
                     f"fixture executed {physical_draws} physical draws, expected exactly {expected_draws}")
    helpers = {}
    for name in ("restore", "save"):
        helper = events(run_out / f"replay.json.{name}-driver.txt")
        contract.require(not any(event.startswith(("draw", "systemc_api_", "compute_api_")) or
                                 event == "launch_grid" for event in helper),
                         f"{name} helper executed native/shader work")
        helpers[name] = {"events": len(helper), "native_draw_dispatch_events": 0}
    return {"expected_accepted_draws": expected_draws, "actual_accepted_draws": recorded,
            "actual_physical_draws": physical_draws,
            "capture_markers": capture_markers,
            "api_submissions": actual.count("systemc_api_submit"), "helper_scopes": helpers}


def execute(args: argparse.Namespace) -> tuple[Path, bool]:
    out = contract.safe_path(args.outdir)
    contract.require(not out.exists() and out.parent.is_dir(), "outdir must be fresh with existing parent")
    contract.require(out != REPO and REPO not in out.parents, "generated fixture must stay outside repository")
    source = contract.safe_path(args.renderdoc_source)
    contract.require((source / "renderdoc/api/app/renderdoc_app.h").is_file(), "RenderDoc app header missing")
    runtime = capture_runtime(args)
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
    command = [str(binary), str(capture_out), "1" if args.depth else "0", "1" if args.hidden_mip else "0",
               "1" if args.depth_renderbuffer else "0", "1" if args.depth_low_codes else "0",
               "1" if args.buffer_target_alias else "0", "1" if args.rgb9e5 else "0"]
    contract.atomic_json(out / "build-and-capture.json", {"build": build, "capture_command": command,
        "environment": env, "runtime": runtime,
        "sources": [contract.file_identity(REPO / name) for name in
                    ("tests/drawlist_checkpoint_capture.cpp", "tools/drawlist-color-readback.h",
                     "script/run_drawlist_checkpoint_integration.py")]})
    checked_run(command, env, capture_out, args.timeout)
    producer = contract.read_json(capture_out / "capture.json")
    contract.require(producer.get("schema") == "pvrgpu.checkpoint-fixture.v1" and producer.get("draws") == 3 and
                     producer.get("depth") is (args.depth or args.depth_renderbuffer or args.depth_low_codes) and
                     producer.get("depth_renderbuffer") is args.depth_renderbuffer and
                     producer.get("depth_low_codes") is args.depth_low_codes and
                     producer.get("buffer_target_alias") is args.buffer_target_alias and
                     producer.get("rgb9e5") is args.rgb9e5 and
                     producer.get("width") == (7 if args.depth_renderbuffer or args.depth_low_codes else 8) and
                     producer.get("height") == (5 if args.depth_renderbuffer or args.depth_low_codes else 8) and
                     producer.get("readback_state_restored") is True and
                     producer.get("hidden_mip") is args.hidden_mip and producer.get("tail_poisoned_prefix_texture") is True and
                     producer.get("readback_negative_checks") == 4 and
                     producer.get("renderer", "").lower().startswith("llvmpipe"),
                     "unexpected producer receipt")
    if args.depth_renderbuffer:
        contract.require(type(producer.get("depth_renderbuffer_name")) is int and
                         producer["depth_renderbuffer_name"] > 0 and
                         producer.get("depth_renderbuffer_internal_format") == 0x81A5 and
                         producer.get("depth_renderbuffer_samples") == 0 and
                         producer.get("depth_alias_fbos") == 2 and producer.get("depth_alias_verified") is True and
                         producer.get("tail_poisoned_depth") is True,
                         "unverified shared single-sample D16 renderbuffer fixture")
    if args.depth_low_codes:
        contract.require(type(producer.get("depth_texture_name")) is int and
                         producer["depth_texture_name"] > 0 and producer.get("depth_alias_fbos") == 2 and
                         producer.get("depth_alias_verified") is True and producer.get("tail_poisoned_depth") is True,
                         "unverified shared low-code D24 texture fixture")
    if args.buffer_target_alias:
        contract.require(type(producer.get("buffer_target_alias_name")) is int and
                         producer["buffer_target_alias_name"] > 0 and
                         producer.get("buffer_target_alias_size") == 32 and
                         producer.get("buffer_target_alias_verified") is True and
                         producer.get("buffer_target_alias_targets") == [0x8F3F, 0x90D2, 0x8F3F, 0x8F36],
                         "unverified same-object mutable buffer target fixture")
    if args.rgb9e5:
        names = producer.get("rgb9e5_texture_names")
        contract.require(isinstance(names, list) and len(names) == 2 and
                         all(type(name) is int and name > 0 for name in names) and names[0] != names[1] and
                         producer.get("rgb9e5_levels") == 2 and producer.get("rgb9e5_sampled_mip") == 1 and
                         producer.get("rgb9e5_tail_poisoned") is True, "unverified RGB9E5 fixture storage/dependency")
    capture = contract.safe_path(producer["capture"])
    contract.require(capture.parent == capture_out and capture.suffix == ".rdc", "capture escaped output directory")
    capture_id = contract.file_identity(capture)
    expected = expected_colors(args.depth, args.depth_renderbuffer, args.depth_low_codes, args.rgb9e5)
    producer_comparisons = [compare_color(capture_out / f"draw{i}.rgba", expected[i]) for i in range(3)]
    contract.require(all(row["equal"] for row in producer_comparisons), "producer GL result disagrees with CPU oracle")
    depth_comparisons = []
    packed_input_comparisons = []
    if args.rgb9e5:
        for image in range(2):
            for level in range(2):
                row = compare_color(capture_out / f"rgb9e5-input{image}-mip{level}.u32", expected_rgb9e5_words(image, level))
                contract.require(row["equal"], "recorded RGB9E5 input differs from independent packed-word oracle")
                packed_input_comparisons.append(dict(row, source="recorded-upload-input", image=image, level=level))
    if args.depth_low_codes:
        metadata = producer.get("depth_codes")
        contract.require(isinstance(metadata, list) and len(metadata) == 3,
                         "missing actual producer D24 boundary observations")
        for index, expected_depth in enumerate(expected_low_depth_codes()):
            row = metadata[index]
            contract.require(row.get("format") == "D24_GL_UNSIGNED_INT_LE" and
                             row.get("origin") == "bottom-left" and row.get("source") == "completed-producer-depth" and
                             row.get("width") == 7 and row.get("height") == 5 and row.get("size_bytes") == 140,
                             "invalid producer D24 observation metadata")
            comparison = compare_color(capture_out / f"draw{index}.depth-u32", expected_depth)
            contract.require(comparison["path"] == row.get("path") and comparison["sha256"] == row.get("sha256") and
                             comparison["equal"], "producer D24 codes disagree with independent integer oracle")
            depth_comparisons.append(comparison)
    for key in contract.RUNTIME_FILES:
        if runtime[key] is not None:
            contract.require(contract.file_identity(Path(runtime[key]["path"])) == runtime[key],
                             f"capture runtime changed: {key}")
    report = {"schema": "pvrgpu.checkpoint-integration.v1", "capture": capture_id,
              "depth": args.depth or args.depth_renderbuffer or args.depth_low_codes,
              "depth_renderbuffer": args.depth_renderbuffer, "depth_low_codes": args.depth_low_codes,
              "buffer_target_alias": args.buffer_target_alias, "rgb9e5": args.rgb9e5,
              "hidden_mip": args.hidden_mip, "capture_runtime": runtime, "producer": producer,
              "producer_comparisons": producer_comparisons,
              "producer_depth_code_comparisons": depth_comparisons,
              "rgb9e5_upload_input_comparisons": packed_input_comparisons,
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
                "--through-draw", str(draw), "--outdir", str(run_out), "--timeout", str(args.timeout),
                "--gles-version", "3.1", "--verify-color"]
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
        packed_comparison = verify_rgb9e5_checkpoints(out, capture_id["sha256"]) if args.rgb9e5 else None
        report.update(status="PASS", checkpoint_verified=True, runs=runs,
                      uninterrupted_resumed_raw_equal=True, cold_cache=True,
                      complete_state_comparison=state_comparison,
                      rgb9e5_packed_checkpoint_comparison=packed_comparison,
                      comparison_scope="CPU-oracle RGBA8 at all boundaries and exact final complete resource/context archive")
    contract.atomic_json(out / "integration.json", report)
    return out / "integration.json", report["checkpoint_verified"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesa-prefix", required=True, help="llvmpipe capture prefix")
    parser.add_argument("--renderdoc-source", required=True, help="headers for explicit capture RenderDoc library")
    parser.add_argument("--renderdoc-lib", required=True, help="capture RenderDoc EGL/GLES interception library")
    parser.add_argument("--outdir", required=True, help="new external output directory")
    depth_mode = parser.add_mutually_exclusive_group()
    depth_mode.add_argument("--depth", action="store_true", help="also consume a shared depth texture across draws")
    depth_mode.add_argument("--depth-renderbuffer", action="store_true",
                            help="7x5 D16 renderbuffer shared by two FBO aliases, with two depth-dependent suffix draws")
    depth_mode.add_argument("--depth-low-codes", action="store_true",
                            help="7x5 shared D24 texture with exact small integer depth codes and raw boundary observations")
    depth_mode.add_argument("--buffer-target-alias", action="store_true",
                            help="same physical UBO changes generic bind targets while remaining a live indexed shader input")
    depth_mode.add_argument("--rgb9e5", action="store_true",
                            help="two sampler-only RGB9E5 textures with exact packed mip preservation and dependent suffix outputs")
    parser.add_argument("--hidden-mip", action="store_true", help="draw into a hidden physical mip, then expose and consume it after restore")
    parser.add_argument("--cxx", default="/usr/local/opt/llvm/bin/clang++" if sys.platform == "darwin" else "c++")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--player", help="snapshot-capable player; omit to generate capture only")
    parser.add_argument("--replay-renderdoc-lib", help="snapshot extension library linked by --player; defaults to capture library")
    parser.add_argument("--replay-mesa-prefix", help="replay Mesa prefix; defaults to capture prefix")
    parser.add_argument("--backend", choices=("pvrgpu", "llvmpipe"), default="llvmpipe")
    parser.add_argument("--bridge", help="required for native replay")
    args = parser.parse_args(argv)
    if args.depth_renderbuffer and args.hidden_mip:
        parser.error("--depth-renderbuffer cannot be combined with --hidden-mip")
    if args.depth_low_codes and args.hidden_mip:
        parser.error("--depth-low-codes cannot be combined with --hidden-mip")
    if args.buffer_target_alias and args.hidden_mip:
        parser.error("--buffer-target-alias cannot be combined with --hidden-mip")
    if args.rgb9e5 and args.hidden_mip:
        parser.error("--rgb9e5 cannot be combined with --hidden-mip")
    return args


def main() -> int:
    try:
        path, verified = execute(parse_args())
    except (contract.ReplayError, OSError, ValueError, TypeError, KeyError, subprocess.TimeoutExpired) as error:
        print(f"checkpoint integration: {error}", file=sys.stderr)
        return 1
    print(f"{'PASS checkpoint continuation' if verified else 'CAPTURE ONLY; checkpoint not yet tested'}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
