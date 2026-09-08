"""Process-contract tests. The fake player is deliberately not a renderer.

CTest passes the freshly built runner; standalone unittest runs may opt in with
PVRGPU_NATIVE_RUNNER_TEST_BIN. No Mesa or SystemC installation is loaded.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from rdc.write_counter_txt import counters_from_pvrgpu_jsonl

FAKE_PLAYER = r'''#!/usr/bin/env python3
import json, os, pathlib, struct, sys, zlib
p = pathlib.Path
mode = os.environ['PVRGPU_TEST_MODE']
calls = p(os.environ['PVRGPU_TEST_CALLS'])
with calls.open('a') as stream: stream.write('player\n')
for name in ('PVRGPU_RDC_TRACE_DRAW_ACTIONS', 'PVRGPU_RDC_OUTPUT_WIDTH', 'PVRGPU_RDC_OUTPUT_HEIGHT'):
    assert name not in os.environ, name
if mode == 'stale': sys.exit(0)
out = p(sys.argv[2]); out.parent.mkdir(parents=True, exist_ok=True)
def png(width, height, color):
    def chunk(kind, data):
        return struct.pack('!I', len(data)) + kind + data + struct.pack('!I', zlib.crc32(kind + data))
    pixels = b''.join(b'\0' + bytes(color) * width for _ in range(height))
    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', width, height, 8, 6, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(pixels)) + chunk(b'IEND', b'')
fields = 'ia_vertices ia_primitives vs_invocations gs_invocations gs_primitives c_invocations c_primitives ps_invocations hs_invocations ds_invocations cs_invocations ts_invocations ms_invocations ms_primitives drawlists setup_triangles texel_fetches'.split()
common = dict(protocol='pvrgpu-jsonl', version=1, schema='pvrgpu.counter.v1', backend='pvrgpu')
model = []
events = []
def event(name, details=''):
    events.append('schema=pvrgpu.driver-counter.v1 producer=pvrgpu-gallium-driver event=' + name + ' ' + details)
if mode != 'compute_only':
    for amount in (2, 5):
        event('systemc_api_submit'); event('systemc_api_done')
        values = {k: amount for k in fields}; values['cs_invocations'] = 0
        model.extend([dict(common, type='hello'), dict(common, type='counter', frame=1, source='pvrgpu-systemc', provenance='modeled', counters=values), dict(common, type='done', frames=1, pool_leaks=0)])
event('compute_api_submit', 'grid=999x999x999')
if mode != 'missing_compute_done':
    event('compute_api_done', 'workgroups=2 invocations=37 alu_instructions=111 memory_instructions=72 atomic_instructions=1 load_instructions=35 store_instructions=36 dram_read_bytes=100 dram_write_bytes=110 direct_read_bytes=0 direct_write_bytes=0 readback_bytes=144 pool_allocations=7 pool_releases=7 texture_requests=6 texel_fetches=17')
if mode == 'driver_failure': event('compute_launch_unsupported', 'reason=deliberate_fixture_failure')
if mode == 'truncated_model': model.pop()
p(os.environ['PVRGPU_DRIVER_COUNTER_OUT']).write_text('\n'.join(events) + '\n')
p(os.environ['PVRGPU_SYSTEMC_JSONL_OUT']).write_text(''.join(json.dumps(m) + '\n' for m in model))
intermediate = p(os.environ['PVRGPU_SYSTEMC_OUTDIR']) / 'intermediate.png'
intermediate.write_bytes(png(1, 1, [255, 0, 0, 255]))
color = mode not in ('no_color', 'compute_only')
if color: out.write_bytes(png(2, 3, [11, 22, 33, 255]))
receipt = dict(schema='pvrgpu.rdc-final-output.v2', backend='pvrgpu', status='PASS', rdc_path=str(p(sys.argv[1]).resolve()), initial_native_isolated=True, replay_completed=True, replay_context_finished=True, api_errors=0, replay_begin_event=1, replay_end_event=91, trace_draw_actions=0, color_output=color, api_error_capture='synchronous-gl-debug-callback', debug_callback_verified=True)
if color: receipt.update(source='completed-replay-attachment', resource_id='123', mip=2, layer=1, sample=0, width=2, height=3, format='RGBA8_UNORM', png_path=str(out.resolve()))
if mode == 'wrong_png': receipt['png_path'] = str(intermediate)
if mode == 'wrong_rdc': receipt['rdc_path'] = str(out)
if mode == 'unfinished': receipt['replay_context_finished'] = False
if mode == 'api_error': receipt['api_errors'] = 1282
if mode == 'png_extent': receipt['width'] = 99
if mode != 'missing_receipt': p(os.environ['PVRGPU_RDC_FINAL_OUTPUT_RECEIPT']).write_text(json.dumps(receipt))
'''


@unittest.skipUnless(os.environ.get("PVRGPU_NATIVE_RUNNER_TEST_BIN"), "requires freshly built native runner")
@unittest.skipIf(os.name == "nt", "POSIX fake-player executable fixture; protocol tests are portable")
class NativeRunnerProcessTests(unittest.TestCase):
    def test_actual_process_contract(self) -> None:
        binary = Path(os.environ["PVRGPU_NATIVE_RUNNER_TEST_BIN"]).resolve()
        with tempfile.TemporaryDirectory(prefix="rdc-runner-contract-") as directory:
            root = Path(directory)
            mesa = root / "mesa"
            (mesa / "lib" / "dri").mkdir(parents=True)
            for name in ("libEGL.dylib", "libGLESv2.dylib", "dri/swrast_dri.dylib"):
                (mesa / "lib" / name).touch()
            bridge = root / "fake-bridge.dylib"
            bridge.touch()
            player = root / "fake-player"
            player.write_text(FAKE_PLAYER)
            player.chmod(0o755)
            rdc = root / "capture with spaces.rdc"
            rdc.write_bytes(b"process-contract-fixture-not-a-real-capture")
            for mode in ("pass", "no_color", "compute_only", "missing_receipt", "wrong_png", "wrong_rdc",
                         "unfinished", "api_error", "png_extent", "missing_compute_done", "driver_failure",
                         "truncated_model", "stale"):
                with self.subTest(mode=mode):
                    output = root / mode
                    output.mkdir()
                    # Deliberately valid-looking old outputs must be removed.
                    (output / "frame.png").write_bytes(b"stale")
                    (output / "player-final-output.json").write_text("{}")
                    (output / "driver-counter.txt").write_text("stale")
                    calls = root / (mode + "-calls")
                    env = dict(os.environ, PVRGPU_TEST_MODE=mode, PVRGPU_TEST_CALLS=str(calls),
                               PVRGPU_SYSTEMC_API_LIB=str(bridge), PVRGPU_RDC_TRACE_DRAW_ACTIONS="999999",
                               PVRGPU_RDC_OUTPUT_WIDTH="800", PVRGPU_RDC_OUTPUT_HEIGHT="600")
                    run = subprocess.run([str(binary), str(rdc), "--outdir", str(output),
                                          "--project-root", str(root), "--mesa-prefix", str(mesa),
                                          "--player", str(player), "--trace-draw-actions", "123"],
                                         capture_output=True, text=True, env=env, timeout=30)
                    self.assertEqual(calls.read_text().splitlines(), ["player"], "no trace-probe replay")
                    result = json.loads((output / "backend-result.json").read_text())
                    if mode in ("pass", "no_color", "compute_only"):
                        self.assertEqual(run.returncode, 0, run.stderr)
                        self.assertEqual(result["status"], "PASS")
                        report = output / "capture-report.jsonl"
                        counters = counters_from_pvrgpu_jsonl(report, expected_rdc_sha256=hashlib.sha256(rdc.read_bytes()).hexdigest())
                        self.assertEqual(counters["cs_invocations"], 37)
                        self.assertEqual(counters["texel_fetches"], 17 if mode == "compute_only" else 24)
                        self.assertEqual(counters["drawlists"], 0 if mode == "compute_only" else 7)
                        self.assertEqual(run.stdout, report.read_text())
                        raw = [json.loads(line) for line in (output / "model.stdout.jsonl").read_text().splitlines()]
                        self.assertEqual(len(raw), 0 if mode == "compute_only" else 6)
                        if mode == "pass":
                            self.assertEqual((output / "frame.png").read_bytes(), (output / "player-png" / "capture with spaces_replay.png").read_bytes())
                            self.assertNotEqual((output / "frame.png").read_bytes(), (output / "png" / "intermediate.png").read_bytes())
                            self.assertIn("trace_draw_actions=0", (output / "runner.txt").read_text())
                        else:
                            self.assertFalse((output / "frame.png").exists())
                    else:
                        self.assertNotEqual(run.returncode, 0)
                        self.assertEqual(result["status"], "FAIL")
                        self.assertFalse((output / "capture-report.jsonl").exists())
                        self.assertFalse((output / "frame.png").exists())


if __name__ == "__main__":
    unittest.main()
