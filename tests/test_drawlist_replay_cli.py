"""CLI/manifest process contracts, not renderer or image correctness tests.

The fake executable emits tiny opaque state and receipts. No Mesa, RenderDoc,
SystemC, real capture, native draw or expected GPU counters are used.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "script" / "run_drawlist_replay.py"
SPEC = importlib.util.spec_from_file_location("drawlist_cli", CLI)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

FAKE_PLAYER = r'''#!/usr/bin/env python3
import argparse, hashlib, json, os, pathlib, sys, time
p = pathlib.Path
root = p(__file__).parent
mode = json.loads((root / 'control.json').read_text())
with (root / 'calls.jsonl').open('a') as f:
    f.write(json.dumps({'argv': sys.argv[1:], 'env': dict(os.environ)}) + '\n')
a = argparse.ArgumentParser()
a.add_argument('capture')
a.add_argument('--stop-after-draw', type=int, required=True)
a.add_argument('--state-out', required=True)
a.add_argument('--receipt-out', required=True)
a.add_argument('--state-in')
a.add_argument('--color-out')
a.add_argument('--expected-resume-event', type=int)
v = a.parse_args()
if mode.get('timeout'):
    print('started', flush=True)
    time.sleep(60)
if mode.get('exit'):
    print('unsupported snapshot library version', file=sys.stderr)
    sys.exit(mode['exit'])
if mode.get('stdout_only'):
    print('{"status":"snapshot_saved"}')
    sys.exit(0)
capture = p(v.capture)
state = p(v.state_out)
receipt_path = p(v.receipt_out)
previous = p(v.state_in).read_bytes() if v.state_in else b'initial'
version = mode.get('state_api_version', mode.get('receipt', {}).get('snapshot_api_version', 1))
magic = {1: b'RDGLSN01', 2: b'RDGLSN02'}.get(version, b'RDGLSN99')
payload = magic + b'fake opaque full-state test data\0' + previous + str(v.stop_after_draw).encode()
if mode.get('state_symlink'):
    state.symlink_to(capture)
elif mode.get('state_hardlink'):
    os.link(capture, state)
elif not mode.get('missing_state'):
    state.write_bytes(b'' if mode.get('empty_state') else payload)
draw = v.stop_after_draw
after = 3000 if draw == 230 else 10 * (draw + 1)
receipt = dict(schema='pvrgpu.drawlist-replay.v1', status='snapshot_saved',
    backend=os.environ['GALLIUM_DRIVER'], capture_path=str(capture), state_path=str(state),
    source_capture_sha256=hashlib.sha256(capture.read_bytes()).hexdigest(),
    snapshot_state_sha256=hashlib.sha256(state.read_bytes()).hexdigest() if state.exists() else '0'*64,
    snapshot_api_version=1, after_draw=draw, after_event=after,
    next_event=None if draw == 230 else after+1, capture_last_event=3000, trace_draw_actions=231,
    resumed_from_event=v.expected_resume_event, native_prefix_replayed=False,
    context_finished=True, api_errors=0, snapshot_restore_verified=v.state_in is not None,
    cold_cache=True)
receipt.update(mode.get('receipt', {}))
if v.color_out:
    color = p(v.color_out)
    if not mode.get('missing_color'): color.write_bytes(bytes(range(16)))
    receipt['color_output'] = dict(path=str(color), sha256=hashlib.sha256(bytes(range(16))).hexdigest(),
        width=2, height=2, size_bytes=16, mip=0, format='RGBA8', origin='bottom-left',
        source='completed-replay-color0')
    receipt['color_output'].update(mode.get('color', {}))
for field in mode.get('remove', []): receipt.pop(field, None)
if mode.get('corrupt_state') and state.exists(): state.write_bytes(state.read_bytes() + b'corrupt')
if mode.get('mutate_capture'): capture.write_bytes(b'mutated capture')
if mode.get('mutate_player'): p(__file__).write_text(p(__file__).read_text() + '\n# changed\n')
if mode.get('mutate_resume') and v.state_in: p(v.state_in).write_bytes(b'mutated resume')
if mode.get('early_manifest'): (state.parent / 'manifest.json').write_text('{}')
if mode.get('receipt_symlink'):
    receipt_path.symlink_to(capture)
elif not mode.get('missing_receipt'):
    text = json.dumps(receipt)
    if mode.get('duplicate'): text = text[:-1] + ',"status":"snapshot_saved"}'
    if mode.get('truncated'): text = text[:-5]
    receipt_path.write_text(text)
sys.exit(mode.get('late_exit', 0))
'''


@unittest.skipIf(os.name != "posix", "POSIX fake executable/process-group fixture")
class DrawListReplayCliTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="drawlist-cli-")
        # macOS /var commonly aliases /private/var; exercise strict canonical
        # data paths, without accidentally using the OS temporary-directory link.
        self.root = Path(self.temporary.name).resolve()
        self.capture = self.root / "unrelated capture name.rdc"
        self.capture.write_bytes(b"not-a-real-capture-test-only")
        self.player = self.root / "fake-player"
        self.player.write_text(FAKE_PLAYER)
        self.player.chmod(0o755)
        self.renderdoc = self.root / "renderdoc.dylib"
        self.renderdoc.write_bytes(b"fake-renderdoc")
        self.bridge = self.root / "bridge.dylib"
        self.bridge.write_bytes(b"fake-bridge")
        self.mesa = self.root / "mesa"
        (self.mesa / "lib").mkdir(parents=True)
        for name in ("libEGL.1.dylib", "libGLESv2.2.dylib", "libgallium-test.dylib"):
            (self.mesa / "lib" / name).write_bytes(name.encode())
        (self.mesa / "lib" / "libEGL.dylib").symlink_to("libEGL.1.dylib")
        (self.mesa / "lib" / "libGLESv2.dylib").symlink_to("libGLESv2.2.dylib")
        self.control({})
        self.sequence = 0

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def control(self, value: dict) -> None:
        (self.root / "control.json").write_text(json.dumps(value))

    def run_cli(self, draw: int = 147, *, resume: Path | None = None,
                backend: str = "pvrgpu", extra: tuple[str, ...] = (),
                capture: Path | None = None, output: Path | None = None,
                env: dict | None = None) -> tuple[subprocess.CompletedProcess, Path]:
        self.sequence += 1
        out = output or self.root / f"out-{self.sequence}"
        command = [sys.executable, str(CLI), str(capture or self.capture),
                   "--player", str(self.player), "--renderdoc-lib", str(self.renderdoc),
                   "--mesa-prefix", str(self.mesa), "--backend", backend,
                   "--through-draw", str(draw), "--outdir", str(out)]
        if backend == "pvrgpu":
            command += ["--bridge", str(self.bridge)]
        if resume is not None:
            command += ["--resume", str(resume)]
        command += list(extra)
        return subprocess.run(command, capture_output=True, text=True,
                              env=env, timeout=15), out

    def calls(self) -> list[dict]:
        path = self.root / "calls.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def save(self, draw: int = 147, **kwargs) -> tuple[Path, dict]:
        result, out = self.run_cli(draw, **kwargs)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "snapshot_saved")
        snapshot = out / "snapshot"
        return snapshot, json.loads((snapshot / "manifest.json").read_text())

    def failed(self, result: subprocess.CompletedProcess, out: Path,
               text: str | None = None) -> None:
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertNotIn('"status": "snapshot_saved"', result.stdout)
        if not (out / "snapshot" / "manifest.json").is_file():
            self.assertFalse((out / "snapshot" / "manifest.json").exists())
        if text:
            self.assertIn(text, result.stderr)

    def test_three_fresh_processes_resume_chain_and_capture_rename(self) -> None:
        first, a = self.save()
        renamed = self.root / "renamed.rdc"
        renamed.write_bytes(self.capture.read_bytes())
        second, b = self.save(148, resume=first, capture=renamed)
        third, c = self.save(149, resume=second, capture=renamed)
        self.assertEqual([a["boundary"]["after_draw"], b["boundary"]["after_draw"],
                          c["boundary"]["after_draw"]], [147, 148, 149])
        self.assertEqual(c["resumed_from_event"], b["boundary"]["after_event"])
        self.assertFalse(c["native_prefix_replayed"])
        self.assertTrue(c["engine_receipt"]["snapshot_restore_verified"])
        self.assertEqual(c["state"]["sha256"], hashlib.sha256((third / "state.bin").read_bytes()).hexdigest())
        self.assertEqual(c["runtime"]["gallium"]["path"], str(self.mesa / "lib" / "libgallium-test.dylib"))
        self.assertEqual(c["runtime"]["schema"], MODULE.RUNTIME_SCHEMA)
        self.assertEqual(c["runtime"]["dri_loader"], c["runtime"]["gallium"])
        self.assertEqual(c["provenance"]["source_snapshot"]["sha256"],
                         hashlib.sha256((second / "manifest.json").read_bytes()).hexdigest())
        calls = self.calls()
        self.assertEqual(len(calls), 3)
        self.assertNotIn("--state-in", calls[0]["argv"])
        self.assertEqual(calls[2]["argv"][-4:], ["--state-in", str(second / "state.bin"),
                                               "--expected-resume-event", str(b["boundary"]["after_event"])])

    def test_clean_environment_has_no_oracle_case_or_inherited_native_scope(self) -> None:
        dirty = dict(os.environ, PVRGPU_RDC_CASE_NAME="old-case", PVRGPU_SYSTEMC_BRIDGE="old-shim",
                     PVRGPU_RDC_TRACE_DRAW_ACTIONS="1", PVRGPU_DIAGNOSTIC_COMPILE_ONLY_INIT="1",
                     PVRGPU_SYSTEMC_JSONL_OUT="old-json", MESA_COUNTER_REPORT_PATH="oracle",
                     PVRGPU_ORACLE="reference")
        first, manifest = self.save(env=dirty)
        env = self.calls()[-1]["env"]
        for key in ("PVRGPU_RDC_CASE_NAME", "PVRGPU_SYSTEMC_BRIDGE", "PVRGPU_RDC_TRACE_DRAW_ACTIONS",
                    "PVRGPU_DIAGNOSTIC_COMPILE_ONLY_INIT", "MESA_COUNTER_REPORT_PATH", "PVRGPU_ORACLE",
                    "LD_PRELOAD", "DYLD_INSERT_LIBRARIES"):
            self.assertNotIn(key, env)
        self.assertEqual(env["PVRGPU_SYSTEMC_API_LIB"], str(self.bridge))
        self.assertEqual(env["PVRGPU_RENDERDOC_LIB"], str(self.renderdoc))
        self.assertEqual(env["PVRGPU_SYSTEMC_JSONL_OUT"], str(first.parent / "model.jsonl"))
        # A nonexistent preload prevents Python itself from starting, before
        # this CLI can sanitize anything. Test those loader variables in-process.
        env_out = self.root / "environment-only"
        (env_out / "dri").mkdir(parents=True)
        with mock.patch.dict(os.environ, dict(LD_PRELOAD="/missing", DYLD_INSERT_LIBRARIES="/missing")):
            cleaned = MODULE.child_environment(type("Args", (), {"backend": "pvrgpu"})(),
                                               manifest["runtime"], env_out)
        self.assertNotIn("LD_PRELOAD", cleaned)
        self.assertNotIn("DYLD_INSERT_LIBRARIES", cleaned)
        self.save(backend="llvmpipe")
        env = self.calls()[-1]["env"]
        self.assertFalse(any(key.startswith("PVRGPU_SYSTEMC_") or key.startswith("PVRGPU_DRIVER_") for key in env))

    def test_receipt_failures_never_commit_snapshot(self) -> None:
        cases = [
            {"missing_receipt": True}, {"missing_state": True}, {"empty_state": True},
            {"stdout_only": True}, {"exit": 6}, {"late_exit": 7}, {"truncated": True},
            {"duplicate": True}, {"corrupt_state": True}, {"receipt_symlink": True},
            {"state_symlink": True}, {"state_hardlink": True},
        ]
        fields = {
            "schema": "pvrgpu.drawlist-replay.v99", "status": "PASS", "backend": "llvmpipe",
            "capture_path": "/other.rdc", "state_path": "/outside/state.bin",
            "source_capture_sha256": "0" * 64, "snapshot_state_sha256": "0" * 64,
            "snapshot_api_version": 3, "after_draw": 148, "after_event": -1,
            "next_event": None, "capture_last_event": 1, "trace_draw_actions": 1,
            "resumed_from_event": 10, "native_prefix_replayed": True,
            "context_finished": False, "api_errors": 1, "snapshot_restore_verified": True,
            "cold_cache": False,
        }
        cases += [{"receipt": {key: value}} for key, value in fields.items()]
        cases += [{"receipt": {key: True}} for key in ("snapshot_api_version", "after_draw", "api_errors")]
        cases += [{"remove": [key]} for key in fields]
        for control in cases:
            with self.subTest(control=control):
                self.control(control)
                result, out = self.run_cli()
                self.failed(result, out)
                self.assertFalse((out / "snapshot" / "manifest.json").exists())

    def test_timeout_stops_child_and_preserves_logs_without_manifest(self) -> None:
        self.control({"timeout": True})
        result, out = self.run_cli(extra=("--timeout", "2"))
        self.failed(result, out, "timed out")
        self.assertIn("started", (out / "stdout.log").read_text())
        self.assertTrue((out / "invocation.json").is_file())
        self.assertFalse((out / "snapshot" / "manifest.json").exists())

    def test_manifest_and_state_corruption_rejected_before_child(self) -> None:
        snapshot, original = self.save()
        original_bytes = (snapshot / "state.bin").read_bytes()
        mutations = [
            lambda m: m.update(schema="unknown"), lambda m: m.update(snapshot_api_version=2),
            lambda m: m.update(backend="llvmpipe"), lambda m: m.update(cold_cache=False),
            lambda m: m["state"].update(path="../state.bin"),
            lambda m: m["state"].update(path="/tmp/state.bin"),
            lambda m: m["state"].update(sha256="0" * 64),
            lambda m: m["source_capture"].update(sha256="0" * 64),
            lambda m: m["boundary"].update(after_event=1400, next_event=1401),
            lambda m: m.update(engine_receipt_sha256="0" * 64),
            lambda m: m["engine_receipt"].update(context_finished=False),
            lambda m: m.pop("runtime"), lambda m: m["runtime"].pop("bridge"),
            lambda m: m["runtime"].pop("schema"), lambda m: m["runtime"].pop("dri_loader"),
            lambda m: m["runtime"].update(schema="pvrgpu.drawlist-runtime.v99"),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                data = json.loads(json.dumps(original))
                mutate(data)
                (snapshot / "manifest.json").write_text(json.dumps(data))
                result, out = self.run_cli(148, resume=snapshot)
                self.failed(result, out)
                self.assertEqual(len(self.calls()), 1)
        (snapshot / "manifest.json").write_text(json.dumps(original))
        (snapshot / "state.bin").write_bytes(original_bytes + b"bad")
        result, out = self.run_cli(148, resume=snapshot)
        self.failed(result, out, "state hash/size mismatch")
        self.assertEqual(len(self.calls()), 1)

    def test_runtime_changes_require_bridge_only_acknowledgement(self) -> None:
        snapshot, _ = self.save()
        self.bridge.write_bytes(b"new bridge")
        result, out = self.run_cli(148, resume=snapshot)
        self.failed(result, out, "runtime changed: bridge")
        _, manifest = self.save(148, resume=snapshot, extra=("--allow-model-change",))
        self.assertTrue(manifest["provenance"]["bridge_changed"])
        self.assertTrue(manifest["provenance"]["allow_model_change"])
        for artifact in (self.player, self.renderdoc, self.mesa / "lib" / "libgallium-test.dylib",
                         self.mesa / "lib" / "libEGL.1.dylib", self.mesa / "lib" / "libGLESv2.2.dylib"):
            before = artifact.read_bytes()
            artifact.write_bytes(before + b"changed")
            result, out = self.run_cli(148, resume=snapshot, extra=("--allow-model-change",))
            self.failed(result, out, "runtime changed:")
            artifact.write_bytes(before)
        self.assertEqual(len(self.calls()), 2)

    def test_v2_abi_is_preserved_through_fresh_process_chain(self) -> None:
        self.control({"receipt": {"snapshot_api_version": 2}})
        first, a = self.save()
        second, b = self.save(148, resume=first)
        _, c = self.save(149, resume=second)
        for manifest in (a, b, c):
            self.assertEqual(manifest["snapshot_api_version"], 2)
            self.assertEqual(manifest["engine_receipt"]["snapshot_api_version"], 2)
        self.assertEqual(c["resumed_from_event"], b["boundary"]["after_event"])

    def test_resume_cannot_switch_codec_abi_even_with_bridge_acknowledgement(self) -> None:
        for saved, changed in ((1, 2), (2, 1)):
            with self.subTest(saved=saved, changed=changed):
                self.control({"receipt": {"snapshot_api_version": saved}})
                first, _ = self.save()
                self.control({"receipt": {"snapshot_api_version": changed}})
                result, out = self.run_cli(148, resume=first,
                                          extra=("--allow-model-change",))
                self.failed(result, out, "snapshot ABI changed")
                self.assertFalse((out / "snapshot" / "manifest.json").exists())

    def test_manifest_cannot_relabel_legacy_receipt_as_v2(self) -> None:
        first, manifest = self.save()
        manifest["snapshot_api_version"] = 2
        (first / "manifest.json").write_text(json.dumps(manifest))
        result, out = self.run_cli(148, resume=first)
        self.failed(result, out, "archive/manifest ABI mismatch")
        self.assertEqual(len(self.calls()), 1)

    def test_archive_magic_must_match_reported_abi(self) -> None:
        for reported, actual in ((1, 2), (2, 1)):
            with self.subTest(reported=reported, actual=actual):
                self.control({"receipt": {"snapshot_api_version": reported},
                              "state_api_version": actual})
                result, out = self.run_cli()
                self.failed(result, out, "archive/receipt ABI mismatch")
                self.assertFalse((out / "snapshot" / "manifest.json").exists())

    def test_matching_manifest_magic_cannot_disguise_legacy_engine_receipt(self) -> None:
        first, manifest = self.save()
        state = b"RDGLSN02" + (first / "state.bin").read_bytes()[8:]
        (first / "state.bin").write_bytes(state)
        manifest["snapshot_api_version"] = 2
        manifest["state"]["sha256"] = hashlib.sha256(state).hexdigest()
        manifest["engine_receipt"]["snapshot_state_sha256"] = manifest["state"]["sha256"]
        manifest["engine_receipt_sha256"] = hashlib.sha256(
            MODULE.json_bytes(manifest["engine_receipt"])).hexdigest()
        (first / "manifest.json").write_text(json.dumps(manifest))
        result, out = self.run_cli(148, resume=first)
        self.failed(result, out, "snapshot ABI changed")
        self.assertEqual(len(self.calls()), 1)

    def test_gles_version_is_explicit_and_pinned_across_resume(self) -> None:
        dirty = dict(os.environ, MESA_GLES_VERSION_OVERRIDE="9.9")
        for backend in ("pvrgpu", "llvmpipe"):
            with self.subTest(backend=backend):
                snapshot, manifest = self.save(backend=backend, env=dirty,
                                               extra=("--gles-version", "3.2"))
                self.assertEqual(manifest["runtime"]["gles_version_override"], "3.2")
                self.assertEqual(self.calls()[-1]["env"]["MESA_GLES_VERSION_OVERRIDE"], "3.2")
                calls_before = len(self.calls())
                extra = ("--allow-model-change",) if backend == "pvrgpu" else ()
                result, out = self.run_cli(148, resume=snapshot, backend=backend, extra=extra)
                self.failed(result, out, "runtime changed: GLES version")
                self.assertEqual(len(self.calls()), calls_before)
                self.save(148, resume=snapshot, backend=backend, extra=("--gles-version", "3.2"))
        _, default = self.save(env=dirty)
        self.assertEqual(default["runtime"]["gles_version_override"], "3.1")
        self.assertEqual(self.calls()[-1]["env"]["MESA_GLES_VERSION_OVERRIDE"], "3.1")

    def test_legacy_runtime_is_only_compatible_with_gles31(self) -> None:
        snapshot, manifest = self.save()
        manifest["runtime"]["schema"] = MODULE.LEGACY_RUNTIME_SCHEMA
        manifest["runtime"].pop("gles_version_override")
        (snapshot / "manifest.json").write_text(json.dumps(manifest))
        _, resumed = self.save(148, resume=snapshot)
        self.assertEqual(resumed["runtime"]["schema"], MODULE.RUNTIME_SCHEMA)
        self.assertEqual(resumed["runtime"]["gles_version_override"], "3.1")
        result, out = self.run_cli(148, resume=snapshot, extra=("--gles-version", "3.2",
                                                              "--allow-model-change"))
        self.failed(result, out, "runtime changed: GLES version")
        manifest["runtime"]["gles_version_override"] = "3.2"
        (snapshot / "manifest.json").write_text(json.dumps(manifest))
        result, out = self.run_cli(148, resume=snapshot, extra=("--gles-version", "3.2"))
        self.failed(result, out, "legacy runtime must use its fixed GLES 3.1 contract")
        self.assertEqual(len(self.calls()), 2)

    def test_new_runtime_requires_a_valid_gles_version_before_child(self) -> None:
        snapshot, manifest = self.save()
        for value in (None, "", "3.0", "3.20", 3.1, True, {}, []):
            with self.subTest(value=value):
                manifest["runtime"]["gles_version_override"] = value
                (snapshot / "manifest.json").write_text(json.dumps(manifest))
                result, out = self.run_cli(148, resume=snapshot)
                self.failed(result, out, "invalid snapshot GLES version")
        manifest["runtime"].pop("gles_version_override")
        (snapshot / "manifest.json").write_text(json.dumps(manifest))
        result, out = self.run_cli(148, resume=snapshot)
        self.failed(result, out, "invalid snapshot GLES version")
        self.assertEqual(len(self.calls()), 1)

    def test_actual_gallium_and_separate_dri_loader_are_both_pinned(self) -> None:
        dri = self.mesa / "lib" / "dri"; dri.mkdir()
        loader = dri / "libdril_dri.dylib"; loader.write_bytes(b"small standalone DRI shim")
        (dri / "swrast_dri.dylib").symlink_to(loader.name)
        snapshot, manifest = self.save()
        runtime = manifest["runtime"]
        implementation = self.mesa / "lib" / "libgallium-test.dylib"
        self.assertEqual(runtime["gallium"]["path"], str(implementation))
        self.assertEqual(runtime["dri_loader"]["path"], str(loader))
        staged_loader = Path(self.calls()[-1]["env"]["LIBGL_DRIVERS_PATH"]) / "swrast_dri.dylib"
        self.assertEqual(staged_loader.resolve(), loader)
        for path, field in ((implementation, "gallium"), (loader, "dri_loader")):
            before = path.read_bytes()
            path.write_bytes(before + b"implementation changed, other file unchanged")
            result, out = self.run_cli(148, resume=snapshot, extra=("--allow-model-change",))
            self.failed(result, out, "runtime changed: " + field)
            path.write_bytes(before)
        self.assertEqual(len(self.calls()), 1)
        self.save(148, resume=snapshot)
        self.assertEqual(len(self.calls()), 2)

    def test_standalone_loader_without_implementation_is_not_a_runtime(self) -> None:
        dri = self.mesa / "lib" / "dri"; dri.mkdir()
        (dri / "swrast_dri.dylib").write_bytes(b"loader only, not an implementation")
        (self.mesa / "lib" / "libgallium-test.dylib").unlink()
        result, out = self.run_cli()
        self.failed(result, out, "actual Gallium implementation")
        self.assertEqual(self.calls(), [])

    def test_boundary_noop_terminal_and_capture_mismatch(self) -> None:
        snapshot, _ = self.save()
        for draw in (0, 147, 231):
            result, out = self.run_cli(draw, resume=snapshot)
            self.failed(result, out)
        self.assertEqual(len(self.calls()), 1)
        terminal, manifest = self.save(230)
        self.assertEqual(manifest["boundary"]["after_event"], 3000)
        self.assertIsNone(manifest["boundary"]["next_event"])
        result, out = self.run_cli(231, resume=terminal)
        self.failed(result, out, "terminal snapshot")
        other = self.root / "different.rdc"
        other.write_bytes(b"different capture")
        result, out = self.run_cli(148, resume=snapshot, capture=other)
        self.failed(result, out, "capture hash/size mismatch")
        result, out = self.run_cli(148, resume=snapshot, backend="llvmpipe")
        self.failed(result, out, "backend mismatch")

    def test_resume_receipt_must_verify_restore_and_advance_metadata(self) -> None:
        snapshot, _ = self.save()
        for receipt in ({"snapshot_restore_verified": False}, {"resumed_from_event": 1},
                        {"after_event": 1480, "next_event": 1481},
                        {"capture_last_event": 4000}, {"trace_draw_actions": 232}):
            self.control({"receipt": receipt})
            result, out = self.run_cli(148, resume=snapshot)
            self.failed(result, out)
            self.assertFalse((out / "snapshot" / "manifest.json").exists())

    def test_relocated_snapshot_and_identical_runtime_bytes_are_allowed(self) -> None:
        snapshot, _ = self.save()
        moved = self.root / "moved snapshot"
        shutil.copytree(snapshot, moved)
        old_bridge = self.bridge
        self.bridge = self.root / "same-bridge-new-location.dylib"
        self.bridge.write_bytes(old_bridge.read_bytes())
        _, manifest = self.save(148, resume=moved)
        self.assertFalse(manifest["provenance"]["bridge_changed"])
        self.assertEqual(manifest["runtime"]["bridge"]["path"], str(self.bridge))

    def test_symlinks_traversal_and_existing_outputs_are_refused(self) -> None:
        snapshot, _ = self.save()
        linked = self.root / "capture-link.rdc"
        linked.symlink_to(self.capture)
        result, out = self.run_cli(capture=linked)
        self.failed(result, out, "symlink")
        result, out = self.run_cli(capture=self.root / "mesa" / ".." / self.capture.name)
        self.failed(result, out, "traversal")
        parent_link = self.root / "directory-link"
        parent_link.symlink_to(self.root, target_is_directory=True)
        result, out = self.run_cli(output=parent_link / "new-output")
        self.failed(result, out, "symlink")
        result, out = self.run_cli(output=snapshot.parent)
        self.failed(result, out, "fresh")
        (snapshot / "state.bin").unlink()
        (snapshot / "state.bin").symlink_to(self.capture)
        result, out = self.run_cli(148, resume=snapshot)
        self.failed(result, out, "symlink")
        self.assertEqual(len(self.calls()), 1)

    def test_child_cannot_change_inputs_or_precreate_commit_marker(self) -> None:
        for mode in ("mutate_capture", "mutate_player", "early_manifest"):
            old_capture = self.capture.read_bytes()
            old_player = self.player.read_bytes()
            self.control({mode: True})
            result, out = self.run_cli()
            self.failed(result, out)
            if mode == "early_manifest":
                self.assertEqual((out / "snapshot" / "manifest.json").read_text(), "{}")
            else:
                self.assertFalse((out / "snapshot" / "manifest.json").exists())
            self.capture.write_bytes(old_capture)
            self.player.write_bytes(old_player)
        self.control({})
        snapshot, _ = self.save()
        self.control({"mutate_resume": True})
        result, out = self.run_cli(148, resume=snapshot)
        self.failed(result, out, "resume state changed")

    def test_invalid_options_missing_runtime_and_ambiguous_gallium(self) -> None:
        for extra in (("--timeout", "nan"), ("--timeout", "0"), ("--allow-model-change",),
                      ("--gles-version", "3.0"), ("--gles-version", "9.9")):
            result, out = self.run_cli(extra=extra)
            self.failed(result, out)
        result, out = self.run_cli(-1)
        self.failed(result, out)
        old_bridge = self.bridge
        self.bridge = self.root / "missing-bridge"
        result, out = self.run_cli()
        self.failed(result, out)
        self.bridge = old_bridge
        (self.mesa / "lib" / "libgallium-other.dylib").write_bytes(b"ambiguous")
        result, out = self.run_cli()
        self.failed(result, out, "ambiguous")
        self.assertEqual(self.calls(), [])

    def test_optional_color_output_is_verified_not_substituted_for_state(self) -> None:
        snapshot, manifest = self.save(extra=("--verify-color",))
        self.assertEqual((snapshot.parent / "color.rgba").read_bytes(), bytes(range(16)))
        self.assertEqual(manifest["verification_color"]["size_bytes"], 16)
        self.assertIn("--color-out", self.calls()[-1]["argv"])
        for mode in ({"missing_color": True}, {"color": {"path": "/other.rgba"}},
                     {"color": {"sha256": "0" * 64}}, {"color": {"width": 3}},
                     {"color": {"source": "intermediate-model-png"}}, {"color": {"origin": "top-left"}},
                     {"color": {"format": "RGB565"}}, {"color": {"mip": 32}}):
            self.control(mode)
            result, out = self.run_cli(extra=("--verify-color",))
            self.failed(result, out)
            self.assertFalse((out / "snapshot" / "manifest.json").exists())


class ManifestUtilitiesTest(unittest.TestCase):
    def test_boolean_is_not_event_or_abi_integer(self) -> None:
        for value in (True, False, None, 1.0, "1", -1):
            with self.subTest(value=value), self.assertRaises(MODULE.ReplayError):
                MODULE.integer(value, "event")

    def test_duplicate_json_keys_rejected(self) -> None:
        with self.assertRaises(MODULE.ReplayError):
            json.loads('{"schema":"one","schema":"two"}', object_pairs_hook=MODULE.unique_object)


if __name__ == "__main__":
    unittest.main()
