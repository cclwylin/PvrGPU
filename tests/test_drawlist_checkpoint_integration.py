"""Independent oracle controls, plus opt-in actual EGL/RDC integration.

For the real test set PVRGPU_DRAWLIST_INTEGRATION_CONFIG to a JSON file with
mesa_prefix, renderdoc_source, renderdoc_lib, player, replay_renderdoc_lib and
output_parent (all explicit paths). Optional replay_mesa_prefix/backend/bridge
select native replay. Color, depth texture, D16 renderbuffer and hidden-mip
cases retain all artifacts.
Without this configuration the real test is explicitly SKIPPED, never PASS.
"""
from __future__ import annotations

import json
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "script"))
from run_drawlist_checkpoint_integration import (compare_checkpoint_states, compare_color,
                                                  expected_colors, native_fixture_work,
                                                  capture_runtime, parse_args, expected_low_depth_codes,
                                                  rgb9_codes, expected_rgb9e5_words, rgb9e5_mips)
from run_drawlist_replay import ReplayError


class CheckpointOracleTest(unittest.TestCase):
    def test_cli_depth_renderbuffer_and_legacy_modes(self) -> None:
        required = ["--mesa-prefix", "/not-opened/mesa", "--renderdoc-source", "/not-opened/source",
                    "--renderdoc-lib", "/not-opened/renderdoc", "--outdir", "/not-created/output"]
        plain = parse_args(required)
        self.assertFalse(plain.depth)
        self.assertFalse(plain.depth_renderbuffer)
        self.assertFalse(plain.depth_low_codes)
        self.assertFalse(plain.buffer_target_alias)
        self.assertFalse(plain.rgb9e5)
        self.assertFalse(plain.hidden_mip)
        rb = parse_args(required + ["--depth-renderbuffer"])
        self.assertTrue(rb.depth_renderbuffer)
        self.assertFalse(rb.depth)
        low = parse_args(required + ["--depth-low-codes"])
        self.assertTrue(low.depth_low_codes)
        self.assertFalse(low.depth or low.depth_renderbuffer or low.hidden_mip)
        alias = parse_args(required + ["--buffer-target-alias"])
        self.assertTrue(alias.buffer_target_alias)
        self.assertFalse(alias.depth or alias.depth_renderbuffer or alias.depth_low_codes or alias.hidden_mip)
        self.assertTrue(parse_args(required + ["--rgb9e5"]).rgb9e5)
        legacy = parse_args(required + ["--depth", "--hidden-mip"])
        self.assertTrue(legacy.depth and legacy.hidden_mip)
        for invalid in (["--depth", "--depth-renderbuffer"],
                        ["--hidden-mip", "--depth-renderbuffer"],
                        ["--depth-low-codes", "--depth"],
                        ["--depth-low-codes", "--depth-renderbuffer"],
                        ["--depth-low-codes", "--hidden-mip"],
                        *(["--buffer-target-alias", flag] for flag in
                          ("--depth", "--depth-renderbuffer", "--depth-low-codes", "--hidden-mip")),
                        *(["--rgb9e5", flag] for flag in
                          ("--depth", "--depth-renderbuffer", "--depth-low-codes", "--hidden-mip", "--buffer-target-alias"))):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                parse_args(required + invalid)
            self.assertEqual(error.exception.code, 2)
        with mock.patch("run_drawlist_checkpoint_integration.contract.runtime", return_value={}) as runtime:
            self.assertEqual(capture_runtime(rb), {})
            passed = runtime.call_args.args[0]
            self.assertEqual(passed.gles_version, "3.1")
            self.assertEqual(passed.backend, "llvmpipe")
            self.assertIsNone(passed.bridge)
            self.assertFalse(passed.allow_model_change)

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

    def test_renderbuffer_alias_and_updated_depth_oracle(self) -> None:
        a, b, c = expected_colors(depth_renderbuffer=True)
        self.assertEqual([len(v) for v in (a, b, c)], [7 * 5 * 4] * 3)
        classes = set()
        stale_depth_c = bytearray(c)
        for y in range(5):
            for x in range(7):
                pixel = 4 * (7 * y + x)
                cell = (3 * x + 5 * y) % 4
                classes.add(cell)
                seed = a[pixel:pixel + 3]
                fresh = bytes(channel // 2 + bias for channel, bias in zip(seed, (32, 16, 8)))
                self.assertEqual(b[pixel:pixel + 4],
                                 (fresh if cell >= 2 else bytes([12, 20, 28])) + b"\xff")
                if cell in (1, 2):
                    self.assertEqual(c[pixel:pixel + 3], bytes(255 - channel for channel in b[pixel:pixel + 3]))
                else:
                    self.assertEqual(c[pixel:pixel + 4], bytes([12, 20, 28, 255]))
                if cell == 3:
                    # Restoring draw0 depth rather than draw1's newly written
                    # depth, or cloning the FBO aliases, wrongly admits this.
                    stale_depth_c[pixel:pixel + 3] = bytes(255 - channel for channel in b[pixel:pixel + 3])
        self.assertEqual(classes, {0, 1, 2, 3})
        self.assertNotEqual(bytes(stale_depth_c), c)
        self.assertNotEqual(b, bytes([12, 20, 28, 255]) * 35,
                            "losing saved depth to tail poison cannot pass")
        self.assertNotEqual(c, b)
        with self.assertRaises(ReplayError):
            expected_colors(depth=True, depth_renderbuffer=True)

    def test_low_depth_integer_codes_and_boundary_propagation(self) -> None:
        raw = expected_low_depth_codes()
        self.assertEqual([len(row) for row in raw], [140] * 3)
        expected_by_class = ((1, 159, 256, 65535), (1, 159, 160, 160), (1, 100, 100, 160))
        for draw, data in enumerate(raw):
            for y in range(5):
                for x in range(7):
                    cell = (3*x + 5*y) % 4
                    offset = 4*(7*y + x)
                    self.assertEqual(int.from_bytes(data[offset:offset+4], "little"),
                                     expected_by_class[draw][cell] << 8)
        self.assertIn((159 << 8).to_bytes(4, "little"), raw[0])
        self.assertNotEqual(raw[0], raw[1])
        self.assertNotEqual(raw[1], raw[2])
        self.assertEqual(expected_colors(depth_low_codes=True), expected_colors(depth_renderbuffer=True))
        for kwargs in ({"depth": True}, {"depth_renderbuffer": True}):
            with self.assertRaises(ReplayError):
                expected_colors(depth_low_codes=True, **kwargs)

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

    def test_rgb9e5_values_noncanonical_bits_and_dependencies(self) -> None:
        a, b, c = expected_colors(rgb9e5=True)
        exponents = set()
        for seed in range(2):
            for level in range(2):
                data = expected_rgb9e5_words(seed, level)
                extent = 16 >> level
                self.assertEqual(len(data), extent**2*4)
                for y in range(extent):
                    for x in range(extent):
                        offset = (y*extent+x)*4
                        word = int.from_bytes(data[offset:offset+4], "little")
                        exponent = word >> 27
                        exponents.add(exponent)
                        decoded = tuple(((word >> (9*channel)) & 511) * 2.0**(exponent-24) for channel in range(3))
                        self.assertEqual(decoded, tuple(k/8 for k in rgb9_codes(seed, x, y)))
        self.assertEqual(exponents, {15, 16, 17, 18, 31})
        for y in range(8):
            for x in range(8):
                offset = (y*8+x)*4
                k0, k1 = rgb9_codes(0, x, y), rgb9_codes(1, x, y)
                self.assertEqual(a[offset:offset+4], bytes([*(32*k for k in k0), 255]))
                self.assertEqual(b[offset:offset+4], bytes([*(24*k for k in k0), 255]))
                self.assertEqual(c[offset:offset+4], bytes([*(12*k+8*v for k, v in zip(k0, k1)), 255]))
        self.assertNotEqual(a, b)
        self.assertNotEqual(b, c)
        self.assertNotEqual(c, bytes([0, 0, 0, 255])*64)
        with self.assertRaises(ReplayError):
            expected_colors(depth=True, rgb9e5=True)

    def test_rgb9e5_locked_parser_refuses_bad_shape_lengths_padding(self) -> None:
        # Synthetic serializer-shaped bytes exercise parser guards only; this
        # is not presented as a GPU/capture or Save/Load positive control.
        q = lambda value: value.to_bytes(8, "little")
        d = lambda value: value.to_bytes(4, "little")
        signature = b"".join(q(v) for v in (2, 0x0de1, 0x8c3d, 16, 16, 1, 0, 2, 0, 0, 3, 2,
                                            16, 16, 1, 0x8c3d, 8, 8, 1, 0x8c3d))
        native = bytearray(d(3) + d(0) + q(42) + d(2) + d(0x8c3d) + b"\0" +
                           b"".join(d(v) for v in (16, 16, 1, 0, 2, 0x0de1, 2)) + bytes(44))
        for size in (4, 3, 4): native.extend(q(size) + bytes(size*4))
        native.extend(bytes(24))
        self.assertEqual(len(native), 189)
        for level in range(2):
            raw = expected_rgb9e5_words(0, level)
            native.extend(q(len(raw)))
            native.extend(bytes((-len(native)) % 64))
            native.extend(raw)
        native[4:8] = d(len(native)-8)
        native.extend(bytes((-len(native)) % 64))
        self.assertEqual(rgb9e5_mips(42, signature, bytes(native), 0x24),
                         [expected_rgb9e5_words(0, level) for level in range(2)])
        bad_padding = bytearray(native); bad_padding[197] = 1
        bad_size = bytearray(native); bad_size[189:197] = q(1)
        for bad in (bytes(native[:-1]), bytes(native)+b"\0", bytes(bad_padding), bytes(bad_size)):
            with self.assertRaises((ReplayError, ValueError)):
                rgb9e5_mips(42, signature, bad, 0x24)
        for identity, sig, section in ((41, signature, 0x24), (42, signature[:-8], 0x24), (42, signature, 0x22)):
            with self.assertRaises(ReplayError):
                rgb9e5_mips(identity, sig, bytes(native), section)

    def test_fixture_native_count_and_helper_exclusion_controls(self) -> None:
        prefix = "schema=pvrgpu.driver-counter.v1 producer=pvrgpu-gallium-driver event="
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary).resolve()
            actual = out / "driver-counter.txt"
            helper = out / "replay.json.restore-driver.txt"
            model = out / "model.jsonl"
            model.write_text(json.dumps({"type": "counter", "source": "pvrgpu-systemc",
                                         "counters": {"drawlists": 1}}) + "\n")
            (out / "replay.json.save-driver.txt").write_text(prefix + "flush\n")
            helper.write_text(prefix + "texture_subdata\n")
            valid = "".join(prefix + event + "\n" for event in
                            ("draw_array_primitive_recorded", "systemc_api_submit", "systemc_api_done"))
            actual.write_text(valid)
            work = native_fixture_work(out, 1)
            self.assertEqual(work["actual_accepted_draws"], 1)
            self.assertEqual(work["actual_physical_draws"], 1)
            self.assertEqual(work["capture_markers"], 0)
            valid_counter = model.read_text()
            marker = "@CAPTURE: driver_pco_triangles sample=1 png=driver_pco_triangles_sample_000001.png\n"
            model.write_text(valid_counter + marker)
            self.assertEqual(native_fixture_work(out, 1)["capture_markers"], 1)
            for invalid in ("not JSON\n", "@CAPTURE: unknown\n", marker[:-5] + "\n",
                            marker.replace("sample=1", "sample=2"),
                            marker.replace("sample=1", "sample=0"),
                            "[]\n", "null\n", '{"type":"unknown"}\n',
                            '{"type":"counter","source":"pvrgpu-systemc","counters":3}\n'):
                model.write_text(valid_counter + invalid)
                with self.subTest(invalid_model_line=invalid), self.assertRaises(ReplayError):
                    native_fixture_work(out, 1)
            model.write_text(valid_counter)
            with self.assertRaises(ReplayError):
                native_fixture_work(out, 3)
            actual.write_text(valid + prefix + "draw_array_primitive_recorded\n")
            with self.assertRaises(ReplayError):
                native_fixture_work(out, 1)
            actual.write_text(valid)
            for wrong in (0, 2, True, None):
                model.write_text(json.dumps({"type": "counter", "source": "pvrgpu-systemc",
                                             "counters": {"drawlists": wrong}}) + "\n")
                with self.assertRaises(ReplayError):
                    native_fixture_work(out, 1)
            model.write_text(json.dumps({"type": "counter", "source": "pvrgpu-systemc",
                                         "counters": {"drawlists": 1}}) + "\n")
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
        for label, flag in (("color", None), ("depth", "--depth"), ("hidden-mip", "--hidden-mip"),
                            ("depth-renderbuffer", "--depth-renderbuffer"), ("depth-low-codes", "--depth-low-codes"),
                            ("buffer-target-alias", "--buffer-target-alias"), ("rgb9e5", "--rgb9e5")):
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
