"""Independent oracle controls, plus opt-in actual EGL/RDC integration.

For the real test set PVRGPU_DRAWLIST_INTEGRATION_CONFIG to a JSON file with
mesa_prefix, renderdoc_source, renderdoc_lib, player, replay_renderdoc_lib and
output_parent (all explicit paths). Optional replay_mesa_prefix/backend/bridge
select native replay. Color, depth and hidden-mip cases retain all artifacts.
Without this configuration the real test is explicitly SKIPPED, never PASS.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "script"))
from run_drawlist_checkpoint_integration import (compare_checkpoint_states, compare_color,
                                                  expected_colors, native_fixture_work)
from run_drawlist_replay import ReplayError


class CheckpointOracleTest(unittest.TestCase):
    def test_independent_integer_math_and_dependency(self) -> None:
        a, b, c = expected_colors()
        self.assertEqual([len(v) for v in (a, b, c)], [256, 256, 256])
        self.assertEqual(a[:4], bytes([8, 24, 16, 255]))
        self.assertEqual(b[:4], bytes([36, 28, 16, 255]))
        self.assertEqual(c[:4], bytes([219, 227, 239, 255]))
        for pixel in range(64):
            start = 4 * pixel
            for channel, bias in enumerate((32, 16, 8)):
                self.assertEqual(b[start + channel], a[start + channel] // 2 + bias)
                self.assertEqual(c[start + channel], 255 - b[start + channel])
        self.assertNotEqual(c, a)
        self.assertNotEqual(c, b)

    def test_shared_depth_changes_only_expected_half(self) -> None:
        plain = expected_colors()
        depth = expected_colors(True)
        self.assertEqual(plain[0], depth[0])
        for y in range(8):
            for x in range(8):
                start = 4 * (8 * y + x)
                if x < 4:
                    self.assertEqual(depth[1][start:start + 4], bytes([12, 20, 28, 255]))
                    self.assertEqual(depth[2][start:start + 4], bytes([243, 235, 227, 255]))
                else:
                    self.assertEqual(plain[1][start:start + 4], depth[1][start:start + 4])
                    self.assertEqual(plain[2][start:start + 4], depth[2][start:start + 4])
        self.assertNotEqual(plain[2], depth[2], "losing saved depth must be observable")

    def test_comparator_rejects_stale_zero_partial_and_wrong_depth(self) -> None:
        expected = expected_colors()[2]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / "actual.rgba"
            path.write_bytes(expected)
            self.assertTrue(compare_color(path, expected)["equal"])
            for wrong in (bytes(256), expected[:-1], expected + b"\0", expected_colors()[1], expected_colors(True)[2]):
                path.write_bytes(wrong)
                result = compare_color(path, expected)
                self.assertFalse(result["equal"])
                self.assertGreater(result["differing_bytes"], 0)

    def test_fixture_native_count_and_helper_exclusion_controls(self) -> None:
        prefix = "schema=pvrgpu.driver-counter.v1 producer=pvrgpu-gallium-driver event="
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary).resolve()
            actual = out / "driver-counter.txt"
            helper = out / "replay.json.restore-driver.txt"
            (out / "replay.json.save-driver.txt").write_text(prefix + "flush\n")
            helper.write_text(prefix + "texture_subdata\n")
            valid = "".join(prefix + event + "\n" for event in
                            ("draw_array_primitive_recorded", "systemc_api_submit", "systemc_api_done"))
            actual.write_text(valid)
            self.assertEqual(native_fixture_work(out, 1)["actual_accepted_draws"], 1)
            with self.assertRaises(ReplayError):
                native_fixture_work(out, 3)
            actual.write_text(valid + prefix + "draw_array_primitive_recorded\n")
            with self.assertRaises(ReplayError):
                native_fixture_work(out, 1)
            actual.write_text(valid)
            for event in ("draw_vbo", "systemc_api_submit", "compute_api_done", "launch_grid"):
                helper.write_text(prefix + event + "\n")
                with self.assertRaises(ReplayError):
                    native_fixture_work(out, 1)

    def test_complete_state_comparator_detects_noncolor_changes(self) -> None:
        # Opaque byte-comparison controls, not a fabricated codec/GL fixture.
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary).resolve()
            a, b = out / "uninterrupted.bin", out / "resumed.bin"
            original = b"opaque context/resource comparison control\0" * 40000
            a.write_bytes(original); b.write_bytes(original)
            result = compare_checkpoint_states(a, b)
            self.assertTrue(result["byte_equal"])
            self.assertEqual(result["uninterrupted"]["sha256"], result["resumed"]["sha256"])
            for wrong in (original[:-1], original + b"x", original[:-1] + b"x", b"x" + original[1:]):
                b.write_bytes(wrong)
                self.assertFalse(compare_checkpoint_states(a, b)["byte_equal"])
            b.write_bytes(b"")
            with self.assertRaises(ReplayError):
                compare_checkpoint_states(a, b)
            b.unlink(); b.symlink_to(a)
            with self.assertRaises(ReplayError):
                compare_checkpoint_states(a, b)


@unittest.skipUnless(os.environ.get("PVRGPU_DRAWLIST_INTEGRATION_CONFIG"),
                     "requires explicit real snapshot player/runtime configuration")
class RealCheckpointIntegrationTest(unittest.TestCase):
    def test_complete_capture_uninterrupted_vs_resumed(self) -> None:
        config = json.loads(Path(os.environ["PVRGPU_DRAWLIST_INTEGRATION_CONFIG"]).read_text())
        required = ("mesa_prefix", "renderdoc_source", "renderdoc_lib", "player", "replay_renderdoc_lib", "output_parent")
        for key in required:
            self.assertIn(key, config)
            self.assertTrue(Path(config[key]).is_absolute(), key)
        output = Path(tempfile.mkdtemp(prefix="checkpoint-integration-test-", dir=config["output_parent"])).resolve()
        for label, flag in (("color", None), ("depth", "--depth"), ("hidden-mip", "--hidden-mip")):
            with self.subTest(fixture=label):
                command = [sys.executable, str(ROOT / "script/run_drawlist_checkpoint_integration.py"),
                           "--outdir", str(output / label)]
                for key in ("mesa_prefix", "renderdoc_source", "renderdoc_lib", "player", "replay_renderdoc_lib",
                            "replay_mesa_prefix", "backend", "bridge", "timeout", "cxx"):
                    if key in config:
                        command += ["--" + key.replace("_", "-"), str(config[key])]
                if flag:
                    command.append(flag)
                # Inner capture/engine wrappers own their child groups and
                # timeout cleanup; do not kill an outer Python process alone.
                result = subprocess.run(command, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                report = json.loads((output / label / "integration.json").read_text())
                self.assertIs(report["checkpoint_verified"], True)
                self.assertEqual(report["status"], "PASS")
                self.assertIs(report["uninterrupted_resumed_raw_equal"], True)
                self.assertIs(report["complete_state_comparison"]["byte_equal"], True)


if __name__ == "__main__":
    unittest.main()
