#!/usr/bin/env python3
"""Save/resume a completed DrawList's full GL state, never a single-draw capsule.

The player owns GL synchronization, native auditing and the versioned RenderDoc
snapshot extension. This wrapper verifies its completion receipt and freezes the
input/runtime identities. Cold-cache resumes preserve functional state, not the
prefix's cache residency, elapsed GPU time or counters. No replay is built here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
from typing import Any

SNAPSHOT_SCHEMA = "pvrgpu.drawlist-snapshot.v1"
REPLAY_SCHEMA = "pvrgpu.drawlist-replay.v1"
RUNTIME_SCHEMA = "pvrgpu.drawlist-runtime.v1"
API_VERSION = 1
SEMANTICS = "functional-state-cold-cache; prefix timing/counters not restored"
RUNTIME_FILES = ("player", "renderdoc", "gallium", "dri_loader", "egl", "gles", "bridge")


class ReplayError(Exception):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayError(message)


def integer(value: Any, name: str, minimum: int = 0) -> int:
    require(type(value) is int and minimum <= value <= (1 << 63) - 1,
            f"{name} must be an integer >= {minimum}")
    return value


def digest_string(value: Any, name: str) -> str:
    require(isinstance(value, str) and len(value) == 64 and
            all(c in "0123456789abcdef" for c in value), f"invalid {name}")
    return value


def safe_path(value: str | Path, *, symlinks: bool = False) -> Path:
    """Reject lexical traversal before resolving; data paths never follow links.

    Mesa's usual unversioned library links are permitted only by the runtime
    resolver. Capture, bridge, snapshot and output paths use the strict default.
    """
    path = Path(value).expanduser()
    require(".." not in path.parts, f"path traversal is not allowed: {value}")
    path = Path(os.path.abspath(path))
    if not symlinks:
        for part in (path, *path.parents):
            require(not part.is_symlink(), f"symlink is not allowed: {part}")
    return path.resolve(strict=False) if symlinks else path


def file_identity(path: Path, *, single_link: bool = False) -> dict[str, Any]:
    path = safe_path(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode), f"not a regular file: {path}")
        require(before.st_size > 0, f"empty file: {path}")
        require(not single_link or before.st_nlink == 1,
                f"snapshot artifacts must not be hard linked: {path}")
        digest = hashlib.sha256()
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
        after = os.fstat(stream.fileno())
        require((before.st_size, before.st_mtime_ns, before.st_ino, before.st_dev) ==
                (after.st_size, after.st_mtime_ns, after.st_ino, after.st_dev),
                f"file changed while hashing: {path}")
    return {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": before.st_size}


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON field: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> dict[str, Any]:
    identity = file_identity(path, single_link=True)
    require(identity["size_bytes"] <= 4 * 1024 * 1024, f"JSON artifact too large: {path}")
    with path.open(encoding="utf-8") as stream:
        result = json.load(stream, object_pairs_hook=unique_object,
                           parse_constant=lambda value: (_ for _ in ()).throw(
                               ReplayError(f"non-finite JSON number: {value}")))
    require(isinstance(result, dict), f"JSON root must be an object: {path}")
    return result


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def atomic_json(path: Path, value: Any) -> None:
    safe_path(path)
    require(not path.exists(), f"refusing to overwrite artifact: {path}")
    fd, temporary = tempfile.mkstemp(prefix=".manifest-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        # link is atomic AND fails if a player created the destination. A rename
        # alone could silently overwrite that destination between checks.
        os.link(temporary, path, follow_symlinks=False)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(temporary)


def library(prefix: Path, names: tuple[str, ...]) -> Path:
    candidates = {safe_path(prefix / "lib" / name, symlinks=True)
                  for name in names if (prefix / "lib" / name).is_file()}
    require(len(candidates) == 1, f"missing or ambiguous Mesa library: {' / '.join(names)}")
    return candidates.pop()


def runtime(args: argparse.Namespace) -> dict[str, Any]:
    prefix = safe_path(args.mesa_prefix)
    require(prefix.is_dir(), f"Mesa prefix is not a directory: {prefix}")
    gallium = {safe_path(path, symlinks=True) for pattern in
               ("libgallium*.dylib", "libgallium*.so*", "gallium*.dll")
               for path in (prefix / "lib").glob(pattern) if path.is_file()}
    require(len(gallium) == 1, "missing or ambiguous actual Gallium implementation in Mesa prefix")
    implementation = gallium.pop()
    direct = {safe_path(path, symlinks=True) for suffix in ("dylib", "so", "dll")
              if (path := prefix / "lib" / "dri" / f"swrast_dri.{suffix}").is_file()}
    require(len(direct) <= 1, "ambiguous swrast DRI loader in Mesa prefix")
    # Current Mesa can either expose __driDriverGetExtensions_swrast directly
    # from libgallium, or use a small libdril_dri shim. The latter is NOT the
    # implementation linked by EGL/GLES: hash both, even when the shim exists.
    loader = direct.pop() if direct else implementation
    player = safe_path(args.player, symlinks=True)
    require(os.access(player, os.X_OK), f"player is not executable: {player}")
    require(args.backend != "pvrgpu" or args.bridge is not None,
            "pvrgpu requires --bridge pointing to a real bridge file")
    require(args.backend != "llvmpipe" or args.bridge is None,
            "llvmpipe must not receive --bridge")
    require(not args.allow_model_change or args.backend == "pvrgpu",
            "--allow-model-change is only valid with pvrgpu")
    return {
        "schema": RUNTIME_SCHEMA,
        "mesa_prefix": str(prefix),
        "player": file_identity(player),
        "renderdoc": file_identity(safe_path(args.renderdoc_lib, symlinks=True)),
        "gallium": file_identity(implementation),
        "dri_loader": file_identity(loader),
        "egl": file_identity(library(prefix, ("libEGL.dylib", "libEGL.1.dylib", "libEGL.so", "libEGL.so.1"))),
        "gles": file_identity(library(prefix, ("libGLESv2.dylib", "libGLESv2.2.dylib", "libGLESv2.so", "libGLESv2.so.2"))),
        "bridge": file_identity(safe_path(args.bridge)) if args.bridge else None,
    }


def validate_boundary(value: Any) -> dict[str, Any]:
    require(isinstance(value, dict), "missing snapshot boundary")
    draw = integer(value.get("after_draw"), "after_draw")
    event = integer(value.get("after_event"), "after_event")
    last = integer(value.get("capture_last_event"), "capture_last_event", 1)
    count = integer(value.get("trace_draw_actions"), "trace_draw_actions", 1)
    require(draw < count and event <= last, "boundary exceeds capture metadata")
    following = value.get("next_event")
    if draw == count - 1:
        require(event == last and following is None, "last draw must include trailing capture events")
    else:
        integer(following, "next_event", 1)
        require(event < last and following == event + 1, "invalid next-event boundary")
    return {key: value[key] for key in
            ("after_draw", "after_event", "next_event", "capture_last_event", "trace_draw_actions")}


def validate_receipt(receipt: Any, *, backend: str, capture: dict[str, Any],
                     state: dict[str, Any], draw: int, resume_event: int | None,
                     resume_boundary: dict[str, Any] | None = None) -> dict[str, Any]:
    require(isinstance(receipt, dict), "missing engine receipt")
    expected = {
        "schema": REPLAY_SCHEMA, "status": "snapshot_saved", "backend": backend,
        "capture_path": capture["path"], "state_path": state["path"],
        "source_capture_sha256": capture["sha256"],
        "snapshot_state_sha256": state["sha256"],
    }
    for key, value in expected.items():
        require(receipt.get(key) == value, f"engine receipt mismatch: {key}")
    require(type(receipt.get("snapshot_api_version")) is int and
            receipt["snapshot_api_version"] == API_VERSION,
            "unsupported snapshot library/API version (required version 1)")
    for key, value in (("context_finished", True), ("cold_cache", True),
                       ("native_prefix_replayed", False),
                       ("snapshot_restore_verified", resume_event is not None)):
        require(receipt.get(key) is value, f"engine receipt mismatch: {key}")
    require(integer(receipt.get("api_errors"), "api_errors") == 0, "engine recorded API errors")
    require("resumed_from_event" in receipt, "engine receipt missing resumed_from_event")
    actual_resume = receipt["resumed_from_event"]
    if resume_event is not None:
        integer(actual_resume, "resumed_from_event")
    require(actual_resume == resume_event, "engine resumed from the wrong event")
    boundary = validate_boundary(receipt)
    require(boundary["after_draw"] == draw, "engine stopped at the wrong DrawList")
    if resume_boundary is not None:
        require(boundary["after_event"] > resume_boundary["after_event"] and
                boundary["capture_last_event"] == resume_boundary["capture_last_event"] and
                boundary["trace_draw_actions"] == resume_boundary["trace_draw_actions"],
                "engine changed capture metadata or failed to advance")
    return boundary


def identity_shape(value: Any, label: str) -> None:
    require(isinstance(value, dict), f"missing {label} identity")
    digest_string(value.get("sha256"), f"{label} sha256")
    integer(value.get("size_bytes"), f"{label} size_bytes", 1)
    require(isinstance(value.get("path"), str) and Path(value["path"]).is_absolute() and
            ".." not in Path(value["path"]).parts, f"invalid {label} provenance path")


def verify_color(receipt: dict[str, Any], path: Path) -> dict[str, Any]:
    color = receipt.get("color_output")
    require(isinstance(color, dict), "missing requested color verification receipt")
    identity = file_identity(path, single_link=True)
    for key, value in identity.items():
        require(color.get(key) == value and
                (key != "size_bytes" or type(color[key]) is int), f"color receipt mismatch: {key}")
    width = integer(color.get("width"), "color width", 1)
    height = integer(color.get("height"), "color height", 1)
    mip = integer(color.get("mip"), "color mip")
    require(mip < 32 and identity["size_bytes"] == width * height * 4, "invalid color extent/size")
    for key, value in (("format", "RGBA8"), ("origin", "bottom-left"),
                       ("source", "completed-replay-color0")):
        require(color.get(key) == value, f"unsupported color verification {key}")
    return color


def load_resume(args: argparse.Namespace, capture: dict[str, Any], current: dict[str, Any]
                ) -> tuple[dict[str, Any] | None, Path | None, bool]:
    if args.resume is None:
        require(not args.allow_model_change, "--allow-model-change requires --resume")
        return None, None, False
    directory = safe_path(args.resume)
    require(directory.is_dir(), f"snapshot directory does not exist: {directory}")
    manifest = read_json(directory / "manifest.json")
    require(manifest.get("schema") == SNAPSHOT_SCHEMA, "unsupported snapshot manifest schema")
    require(type(manifest.get("snapshot_api_version")) is int and
            manifest["snapshot_api_version"] == API_VERSION, "unsupported snapshot manifest ABI")
    require(manifest.get("backend") == args.backend, "snapshot backend mismatch")
    require(manifest.get("cold_cache") is True and manifest.get("semantics") == SEMANTICS and
            manifest.get("native_prefix_replayed") is False, "unsupported snapshot execution semantics")
    source = manifest.get("source_capture")
    identity_shape(source, "capture")
    require((source["sha256"], source["size_bytes"]) ==
            (capture["sha256"], capture["size_bytes"]), "snapshot capture hash/size mismatch")
    boundary = validate_boundary(manifest.get("boundary"))
    require(boundary["next_event"] is not None, "terminal snapshot has no next event to resume")
    require(args.through_draw > boundary["after_draw"], "resume --through-draw must advance beyond saved DrawList")
    require(args.through_draw < boundary["trace_draw_actions"], "requested DrawList exceeds snapshot capture")
    state = manifest.get("state")
    require(isinstance(state, dict) and state.get("path") == "state.bin",
            "snapshot state path must be exactly state.bin")
    actual_state = file_identity(directory / "state.bin", single_link=True)
    require(type(state.get("size_bytes")) is int and state["size_bytes"] == actual_state["size_bytes"] and
            state.get("sha256") == actual_state["sha256"], "snapshot state hash/size mismatch")
    saved_runtime = manifest.get("runtime")
    require(isinstance(saved_runtime, dict), "missing snapshot runtime")
    require(saved_runtime.get("schema") == RUNTIME_SCHEMA,
            "unsupported snapshot runtime identity schema; both Gallium and DRI loader must be pinned")
    changed_bridge = False
    for key in RUNTIME_FILES:
        saved = saved_runtime.get(key)
        now = current[key]
        if key == "bridge" and args.backend == "llvmpipe":
            require(saved is None, "llvmpipe snapshot contains a native bridge")
            continue
        identity_shape(saved, key)
        if (saved["sha256"], saved["size_bytes"]) != (now["sha256"], now["size_bytes"]):
            require(key == "bridge" and args.allow_model_change,
                    f"snapshot runtime changed: {key}; only bridge changes may be acknowledged")
            changed_bridge = True
    # Embed the original completion receipt so relocated snapshot directories do
    # not depend on an old output directory. Its paths are provenance, never I/O.
    receipt = manifest.get("engine_receipt")
    require(isinstance(receipt, dict), "missing snapshot engine receipt")
    receipt_state = dict(actual_state, path=receipt.get("state_path"))
    identity_shape(receipt_state, "original state")
    verified = validate_receipt(receipt, backend=args.backend, capture=source,
                                state=receipt_state, draw=boundary["after_draw"],
                                resume_event=manifest.get("resumed_from_event"))
    require(verified == boundary, "manifest boundary does not match saved engine receipt")
    digest_string(manifest.get("engine_receipt_sha256"), "engine receipt sha256")
    require(hashlib.sha256(json_bytes(receipt)).hexdigest() == manifest["engine_receipt_sha256"],
            "saved engine receipt checksum mismatch")
    return manifest, directory, changed_bridge


def child_environment(args: argparse.Namespace, current: dict[str, Any], out: Path) -> dict[str, str]:
    # A small allowlist prevents inherited case selectors, guessed action gates,
    # oracle paths, old reports, preload shims or model overrides reaching replay.
    env = {key: os.environ[key] for key in
           ("PATH", "HOME", "USER", "LOGNAME", "SYSTEMROOT", "WINDIR") if key in os.environ}
    prefix = Path(current["mesa_prefix"])
    driver = Path(current["dri_loader"]["path"])
    suffix = ".dylib" if ".dylib" in driver.name else ".dll" if driver.suffix == ".dll" else ".so"
    (out / "dri" / ("swrast_dri" + suffix)).symlink_to(driver)
    env.update({
        "LC_ALL": "C", "EGL_PLATFORM": "surfaceless", "LIBGL_ALWAYS_SOFTWARE": "1",
        "GALLIUM_DRIVER": args.backend, "MESA_LOADER_DRIVER_OVERRIDE": "swrast",
        "MESA_SHADER_CACHE_DISABLE": "true", "MESA_GLES_VERSION_OVERRIDE": "3.1",
        "DYLD_LIBRARY_PATH": str(prefix / "lib"), "LD_LIBRARY_PATH": str(prefix / "lib"),
        "LIBGL_DRIVERS_PATH": str(out / "dri"), "TMPDIR": str(out / "tmp"),
        "XDG_CACHE_HOME": str(out / "cache"),
        "RENDERDOC_MESA_EGL_PATH": current["egl"]["path"],
        "RENDERDOC_MESA_GLES_PATH": current["gles"]["path"],
        "PVRGPU_RENDERDOC_LIB": current["renderdoc"]["path"],
    })
    if args.backend == "pvrgpu":
        env.update({
            "PVRGPU_SYSTEMC_API_LIB": current["bridge"]["path"],
            "PVRGPU_SYSTEMC_JSONL_OUT": str(out / "model.jsonl"),
            "PVRGPU_SYSTEMC_STDERR_OUT": str(out / "model.stderr.log"),
            "PVRGPU_SYSTEMC_OUTDIR": str(out / "model"),
            "PVRGPU_DRIVER_COMMAND_OUT": str(out / "driver-command.txt"),
            "PVRGPU_DRIVER_COUNTER_OUT": str(out / "driver-counter.txt"),
        })
    # Engine, not this wrapper, isolates these native outputs around OpenCapture
    # and snapshot import. They are restored only for the ordered actual suffix.
    return env


def run_child(command: list[str], env: dict[str, str], out: Path, timeout: float) -> None:
    with (out / "stdout.log").open("xb") as stdout, (out / "stderr.log").open("xb") as stderr:
        process = subprocess.Popen(command, env=env, cwd=out, stdout=stdout, stderr=stderr,
                                   start_new_session=os.name == "posix")
        try:
            code = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()
            raise ReplayError("player timed out/interrupted; no snapshot committed") from error
    require(code == 0, f"player exited {code}; inspect {out / 'stderr.log'} (unsupported snapshot library must fail)")


def execute(args: argparse.Namespace) -> Path:
    integer(args.through_draw, "--through-draw")
    require(math.isfinite(args.timeout) and args.timeout > 0, "--timeout must be finite and positive")
    capture = file_identity(safe_path(args.capture))
    current = runtime(args)
    previous, resume, bridge_changed = load_resume(args, capture, current)
    out = safe_path(args.outdir)
    require(not out.exists(), f"--outdir must be fresh (must not exist): {out}")
    require(out.parent.is_dir(), f"--outdir parent must already exist: {out.parent}")
    out.mkdir(mode=0o700)
    for name in ("snapshot", "model", "dri", "tmp", "cache"):
        (out / name).mkdir(mode=0o700)
    state_path = out / "snapshot" / "state.bin"
    receipt_path = out / "replay.json"
    command = [current["player"]["path"], capture["path"], "--stop-after-draw", str(args.through_draw),
               "--state-out", str(state_path), "--receipt-out", str(receipt_path)]
    if args.verify_color:
        command += ["--color-out", str(out / "color.rgba")]
    resume_event = previous["boundary"]["after_event"] if previous else None
    if resume is not None:
        command += ["--state-in", str(resume / "state.bin"), "--expected-resume-event", str(resume_event)]
    env = child_environment(args, current, out)
    atomic_json(out / "invocation.json", {
        "schema": "pvrgpu.drawlist-invocation.v1", "argv": command, "runtime": current,
        "source_capture": capture, "backend": args.backend, "environment": env,
        "timeout_seconds": args.timeout, "cold_cache": True, "semantics": SEMANTICS,
        "allow_model_change": args.allow_model_change, "bridge_changed": bridge_changed,
    })
    run_child(command, env, out, args.timeout)
    # Do not accept a child which modified the inputs it claims to have replayed.
    require(file_identity(Path(capture["path"])) == capture, "capture changed during replay")
    for key in RUNTIME_FILES:
        if current[key] is not None:
            require(file_identity(Path(current[key]["path"])) == current[key], f"{key} changed during replay")
    if resume is not None:
        require(read_json(resume / "manifest.json") == previous, "resume manifest changed during replay")
        saved_state = file_identity(resume / "state.bin", single_link=True)
        require(saved_state["sha256"] == previous["state"]["sha256"] and
                saved_state["size_bytes"] == previous["state"]["size_bytes"], "resume state changed during replay")
    state = file_identity(state_path, single_link=True)
    receipt = read_json(receipt_path)
    boundary = validate_receipt(receipt, backend=args.backend, capture=capture, state=state,
                                draw=args.through_draw, resume_event=resume_event,
                                resume_boundary=previous["boundary"] if previous else None)
    color = verify_color(receipt, out / "color.rgba") if args.verify_color else None
    manifest = {
        "schema": SNAPSHOT_SCHEMA, "snapshot_api_version": API_VERSION, "backend": args.backend,
        "source_capture": capture, "runtime": current, "boundary": boundary,
        "state": dict(state, path="state.bin"), "resumed_from_event": resume_event,
        "cold_cache": True, "semantics": SEMANTICS, "native_prefix_replayed": False,
        "engine_receipt": receipt, "engine_receipt_sha256": hashlib.sha256(json_bytes(receipt)).hexdigest(),
        "verification_color": color,
        "provenance": {
            "source_snapshot": file_identity(resume / "manifest.json", single_link=True) if resume else None,
            "allow_model_change": args.allow_model_change, "bridge_changed": bridge_changed,
            "previous_bridge": previous["runtime"]["bridge"] if previous else None,
            "model_change_contract": "caller affirms saved prefix functional state remains valid",
        },
    }
    # The manifest is the commit marker; failed/incomplete runs retain evidence
    # but never become usable checkpoints merely because state.bin exists.
    atomic_json(out / "snapshot" / "manifest.json", manifest)
    return out / "snapshot"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", metavar="CAPTURE")
    parser.add_argument("--player", required=True, help="snapshot-capable player executable (no automatic build)")
    parser.add_argument("--renderdoc-lib", required=True, help="exact snapshot-extension RenderDoc library")
    parser.add_argument("--mesa-prefix", required=True)
    parser.add_argument("--backend", required=True, choices=("pvrgpu", "llvmpipe"))
    parser.add_argument("--bridge", help="required real PvrGPU bridge file; forbidden for llvmpipe")
    parser.add_argument("--through-draw", required=True, type=int, metavar="N", help="inclusive zero-based DrawList")
    parser.add_argument("--resume", metavar="SNAPSHOT_DIR", help="directory containing manifest.json and state.bin")
    parser.add_argument("--outdir", required=True, help="fresh, non-existing output directory")
    parser.add_argument("--timeout", type=float, default=300.0, help="child wall-clock timeout in seconds (default: 300)")
    parser.add_argument("--allow-model-change", action="store_true", help="acknowledge bridge-only change on resume; prefix validity is your responsibility")
    parser.add_argument("--verify-color", action="store_true", help="also read/verify live color0 RGBA8 bytes (diagnostic output, not snapshot state)")
    args = parser.parse_args(argv)
    try:
        snapshot = execute(args)
    except (ReplayError, OSError, ValueError, TypeError, KeyError) as error:
        print(f"run_drawlist_replay.py: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "snapshot_saved", "snapshot": str(snapshot),
                      "after_draw": args.through_draw, "cold_cache": True}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
