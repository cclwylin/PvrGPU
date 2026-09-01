from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from counter_protocol import STANDARD_COUNTER_FIELDS  # noqa: E402
from rdc_counter_report import build_argument_parser  # noqa: E402
from rdc_counter_report import DirectoryCounterRun  # noqa: E402
from rdc_counter_report import DiscoveredRdc  # noqa: E402
from rdc_counter_report import decode_rgba8_png  # noqa: E402
from rdc_counter_report import EventSink  # noqa: E402
from rdc_counter_report import ManifestEntry  # noqa: E402
from rdc_counter_report import natural_path_sort_key  # noqa: E402
from rdc_counter_report import runner_exit_failure_message  # noqa: E402


WORKER = TOOLS_DIR / "rdc_counter_report.py"


class RdcCounterReportTests(unittest.TestCase):
    def test_runner_failure_uses_backend_stage_and_reason(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory)
            (artifact_dir / "backend-result.json").write_text(
                json.dumps(
                    {
                        "status": "FAIL",
                        "stage": "driver-support",
                        "reason": "unsupported draw",
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                runner_exit_failure_message(artifact_dir, 1),
                "Runner exited with code 1 at driver-support: unsupported draw",
            )

    def test_default_rdc_directory_comes_from_work_root_sibling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            work_root = Path(directory) / "PvrGPU"
            environment = {
                "PVRGPU_WORK_ROOT": str(work_root),
                "PVRGPU_BUILD_DIR": str(work_root / "build"),
            }
            with mock.patch.dict(os.environ, environment, clear=False):
                os.environ.pop("PVRGPU_RDC_ROOT", None)
                options = build_argument_parser().parse_args([])

            self.assertEqual(
                options.rdc_dir,
                work_root.parent / "GPU_TestPatterns" / "1.GLBench",
            )

    def test_default_rdc_directory_accepts_environment_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rdc_root = Path(directory) / "2.dEQP_100Frames"
            with mock.patch.dict(
                os.environ, {"PVRGPU_RDC_ROOT": str(rdc_root)}, clear=False
            ):
                options = build_argument_parser().parse_args([])

            self.assertEqual(options.rdc_dir, rdc_root)

    def test_default_runners_come_from_the_native_build_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build_root = Path(directory) / "native-build"
            with mock.patch.dict(
                os.environ, {"PVRGPU_BUILD_DIR": str(build_root)}, clear=False
            ):
                os.environ.pop("PVRGPU_RDC_GOLDEN_RUNNER", None)
                os.environ.pop("PVRGPU_RDC_PVRGPU_RUNNER", None)
                options = build_argument_parser().parse_args(
                    ["--rdc-dir", "/tmp/rdcs"]
                )

            suffix = ".exe" if os.name == "nt" else ""
            self.assertEqual(
                options.golden_runner, build_root / "bin" / f"llvmpipe{suffix}"
            )
            self.assertEqual(
                options.pvrgpu_runner, build_root / "bin" / f"pvrgpu{suffix}"
            )

    def test_natural_path_sort_key_keeps_drawlists_in_frame_order(self) -> None:
        names = ["drawlist101-104.rdc", "drawlist4-8.rdc", "drawlist0-3.rdc"]
        self.assertEqual(
            sorted(names, key=natural_path_sort_key),
            ["drawlist0-3.rdc", "drawlist4-8.rdc", "drawlist101-104.rdc"],
        )

    def test_staging_copies_when_links_are_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.rdc"
            source.write_bytes(b"rdc-payload")
            case_root = root / "case"
            case_root.mkdir()
            entry = ManifestEntry(1, "case", "canonical.rdc", "0" * 64, 1, 1)
            item = DiscoveredRdc(source, source.name, "0" * 64, entry)
            run = DirectoryCounterRun(
                input_root=root,
                output_root=root / "output",
                manifest_path=root / "manifest.tsv",
                golden_runner=root / "llvmpipe",
                pvrgpu_runner=root / "pvrgpu",
                timeout_seconds=0,
                require_manifest=False,
                reuse_golden_cache=False,
                events=EventSink(False),
            )

            with mock.patch.object(
                Path, "symlink_to", side_effect=OSError("no symlinks")
            ), mock.patch("rdc_counter_report.os.link", side_effect=OSError("no links")):
                staged = run._stage_input(item, case_root, entry)

            self.assertEqual(staged.name, "canonical.rdc")
            self.assertEqual(staged.read_bytes(), b"rdc-payload")

    def _write_fake_runners(self, root: Path) -> tuple[Path, Path]:
        fields = repr(tuple(STANDARD_COUNTER_FIELDS))
        common = f"""
            #!/usr/bin/env python3
            import argparse
            import hashlib
            import json
            from pathlib import Path
            import struct
            import sys
            import time
            import zlib

            FIELDS = {fields}

            def write_rgba8_png(path, width, height, pixel):
                path.parent.mkdir(parents=True, exist_ok=True)
                raw = b''.join(b'\\x00' + bytes(pixel) * width for _ in range(height))

                def chunk(kind, payload):
                    crc = zlib.crc32(kind)
                    crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
                    return struct.pack('>I', len(payload)) + kind + payload + struct.pack('>I', crc)

                ihdr = struct.pack('>IIBBBBB', width, height, 8, 6, 0, 0, 0)
                path.write_bytes(
                    b'\\x89PNG\\r\\n\\x1a\\n'
                    + chunk(b'IHDR', ihdr)
                    + chunk(b'IDAT', zlib.compress(raw))
                    + chunk(b'IEND', b'')
                )

            parser = argparse.ArgumentParser()
            parser.add_argument('--rdc', type=Path, required=True)
            parser.add_argument('--case', required=True)
            parser.add_argument('--width', required=True)
            parser.add_argument('--height', required=True)
            parser.add_argument('--outdir', type=Path, required=True)
            args = parser.parse_args()
            mode = args.rdc.read_text(encoding='utf-8').strip()
            args.outdir.mkdir(parents=True, exist_ok=True)
            values = {{field: index + 10 for index, field in enumerate(FIELDS)}}
            if mode == 'png-no-color-write':
                values['drawlists'] = 1
                values['ps_invocations'] = 0
        """
        golden = root / "fake_golden.py"
        golden.write_text(
            textwrap.dedent(common).lstrip()
            + textwrap.dedent(
                """
                if mode == 'golden-error':
                    print('intentional golden failure', file=sys.stderr)
                    raise SystemExit(7)
                if mode == 'slow-golden':
                    time.sleep(2)
                headers = ['Frame', 'Marker', *FIELDS]
                row = ['1', args.case, *(str(values[field]) for field in FIELDS)]
                report = [
                    '# Fake Golden report',
                    '',
                    '| ' + ' | '.join(headers) + ' |',
                    '| ' + ' | '.join('---:' for _ in headers) + ' |',
                    '| ' + ' | '.join(row) + ' |',
                    '',
                ]
                (args.outdir / 'Report.md').write_text('\\n'.join(report), encoding='utf-8')
                if mode == 'png-no-selected-color':
                    (args.outdir / 'player-wrapper.stdout.log').write_text(
                        'Color output: none\\n'
                        'PNG: none (selected replay range has no color output)\\n',
                        encoding='utf-8',
                    )
                if mode not in {
                    'png-both-missing',
                    'png-pvrgpu-only',
                    'png-no-selected-color',
                }:
                    trace_stem = args.rdc.stem
                    png_width = 32 if mode == 'png-size-from-golden' else int(args.width)
                    png_height = 48 if mode == 'png-size-from-golden' else int(args.height)
                    write_rgba8_png(
                        args.outdir / 'player-output' / trace_stem / 'png' / f'{trace_stem}_replay.png',
                        png_width,
                        png_height,
                        (1, 2, 3, 255),
                    )
                """
            ),
            encoding="utf-8",
        )
        pvrgpu = root / "fake_pvrgpu.py"
        pvrgpu.write_text(
            textwrap.dedent(common).lstrip()
            + textwrap.dedent(
                """
                if mode == 'pvrgpu-error':
                    print('intentional PvrGPU failure', file=sys.stderr)
                    raise SystemExit(8)
                if mode == 'mismatch':
                    values[FIELDS[-1]] += 1
                artifact_png = None
                if mode not in {'png-both-missing', 'png-golden-only'}:
                    artifact_png = args.outdir / 'png' / 'driver_indexed_quad_sample_000001.png'
                    pixel = (
                        (9, 2, 3, 255)
                        if mode == 'png-mismatch'
                        else (1, 2, 3, 255)
                    )
                    write_rgba8_png(artifact_png, int(args.width), int(args.height), pixel)
                digest = hashlib.sha256(args.rdc.read_bytes()).hexdigest()
                counter_message = {
                    'schema': 'pvrgpu.counter.v1',
                    'type': 'counter',
                    'frame': 1,
                    'source': 'fake-pvrgpu',
                    'provenance': 'modeled',
                    'counters': values,
                }
                if artifact_png is not None:
                    counter_message['artifact_png'] = str(artifact_png)
                messages = [
                    {
                        'schema': 'pvrgpu.counter.v1',
                        'type': 'hello',
                        'backend': 'pvrgpu',
                        'rdc_sha256': digest,
                    },
                    counter_message,
                    {
                        'schema': 'pvrgpu.counter.v1',
                        'type': 'done',
                        'frames': 1,
                        'pool_leaks': 0,
                    },
                ]
                for message in messages:
                    print(json.dumps(message), flush=True)
                """
            ),
            encoding="utf-8",
        )
        golden.chmod(0o755)
        pvrgpu.chmod(0o755)
        return golden, pvrgpu

    def _write_manifest(self, path: Path, cases: list[tuple[str, Path]]) -> None:
        fieldnames = ("index", "case", "rdc_path", "rdc_sha256", "width", "height")
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            for index, (case, rdc) in enumerate(cases, start=1):
                writer.writerow(
                    {
                        "index": index,
                        "case": case,
                        "rdc_path": f"{case}/recorder/trace/{case}_capture_1.rdc",
                        "rdc_sha256": hashlib.sha256(rdc.read_bytes()).hexdigest(),
                        "width": 64,
                        "height": 64,
                    }
                )

    def _run(
        self,
        *,
        input_root: Path,
        output_root: Path,
        manifest: Path,
        golden: Path,
        pvrgpu: Path,
        timeout: float = 0,
        require_manifest: bool = False,
        reuse_golden_cache: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        arguments = [
            sys.executable,
            str(WORKER),
            "--rdc-dir",
            str(input_root),
            "--output-root",
            str(output_root),
            "--manifest",
            str(manifest),
            "--golden-runner",
            str(golden),
            "--pvrgpu-runner",
            str(pvrgpu),
            "--timeout-seconds",
            str(timeout),
            "--json",
        ]
        if require_manifest:
            arguments.append("--require-manifest")
        if reuse_golden_cache:
            arguments.append("--reuse-golden-cache")
        return subprocess.run(
            arguments,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=20,
        )

    @staticmethod
    def _single_run_root(output_root: Path) -> Path:
        roots = list(output_root.glob("rdc-counter-*"))
        if len(roots) != 1:
            raise AssertionError(f"expected one run root, found {roots}")
        return roots[0]

    @staticmethod
    def _latest_run_root(output_root: Path) -> Path:
        roots = sorted(output_root.glob("rdc-counter-*"))
        if not roots:
            raise AssertionError("expected at least one run root")
        return roots[-1]

    def test_recursive_all_pass_writes_two_counters_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs"
            nested = inputs / "nested"
            nested.mkdir(parents=True)
            first = inputs / "a.rdc"
            second = nested / "pipe| 名稱.RDC"
            first.write_text("pass-a", encoding="utf-8")
            second.write_text("pass-b", encoding="utf-8")
            (inputs / "ignored.txt").write_text("ignored", encoding="utf-8")
            (inputs / "wrong-extension.drc").write_text("ignored", encoding="utf-8")
            manifest = root / "manifest.tsv"
            self._write_manifest(manifest, [("case_a", first), ("case_b", second)])
            golden, pvrgpu = self._write_fake_runners(root)
            output = root / "output"

            completed = self._run(
                input_root=inputs,
                output_root=output,
                manifest=manifest,
                golden=golden,
                pvrgpu=pvrgpu,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            events = [json.loads(line) for line in completed.stdout.splitlines()]
            summary = next(event for event in events if event["type"] == "summary")
            self.assertEqual((summary["total"], summary["passed"], summary["failed"]), (2, 2, 0))
            run_root = self._single_run_root(output)
            run_data = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(run_data["status"], "PASS")
            self.assertEqual(
                [result["rdc"] for result in run_data["results"]],
                ["a.rdc", "nested/pipe| 名稱.RDC"],
            )
            for result in run_data["results"]:
                case_root = run_root / result["artifact_dir"]
                golden_text = (case_root / "counter_golden.txt").read_text(encoding="utf-8")
                pvrgpu_text = (case_root / "counter_pvrgpu.txt").read_text(encoding="utf-8")
                self.assertEqual(golden_text, pvrgpu_text)
                self.assertEqual(len(golden_text.splitlines()), 17)
            report = (run_root / "report.md").read_text(encoding="utf-8")
            self.assertIn("Overall: **PASS**", report)
            self.assertIn("nested/pipe\\| 名稱.RDC", report)

    def test_mixed_failures_continue_and_are_all_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs"
            inputs.mkdir()
            cases: list[tuple[str, Path]] = []
            for case, mode in (
                ("a_pass", "pass"),
                ("b_mismatch", "mismatch"),
                ("c_golden_error", "golden-error"),
                ("d_pvrgpu_error", "pvrgpu-error"),
                ("e_after_errors", "pass-after"),
            ):
                rdc = inputs / f"{case}.rdc"
                rdc.write_text(mode, encoding="utf-8")
                cases.append((case, rdc))
            unknown = inputs / "z_unknown.rdc"
            unknown.write_text("not-in-manifest", encoding="utf-8")
            manifest = root / "manifest.tsv"
            self._write_manifest(manifest, cases)
            golden, pvrgpu = self._write_fake_runners(root)
            output = root / "output"

            completed = self._run(
                input_root=inputs,
                output_root=output,
                manifest=manifest,
                golden=golden,
                pvrgpu=pvrgpu,
            )

            self.assertEqual(completed.returncode, 1, completed.stderr)
            run_root = self._single_run_root(output)
            run_data = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
            self.assertEqual((run_data["total"], run_data["passed"], run_data["failed"]), (6, 3, 3))
            results = {result["case"]: result for result in run_data["results"]}
            self.assertEqual(results["a_pass"]["status"], "PASS")
            self.assertEqual(results["b_mismatch"]["stage"], "compare")
            mismatch_root = run_root / results["b_mismatch"]["artifact_dir"]
            self.assertTrue((mismatch_root / "counter_diff.txt").is_file())
            self.assertEqual(results["c_golden_error"]["stage"], "golden")
            self.assertEqual(results["c_golden_error"]["pvrgpu"], "SKIP")
            self.assertEqual(results["d_pvrgpu_error"]["stage"], "pvrgpu")
            self.assertEqual(results["e_after_errors"]["status"], "PASS")
            self.assertEqual(results["z_unknown"]["status"], "PASS")
            report = (run_root / "report.md").read_text(encoding="utf-8")
            self.assertIn("Overall: **FAIL**", report)
            self.assertIn("17-counter exact comparison mismatch", report)

    def test_png_compare_is_required_when_framebuffer_pngs_exist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs"
            inputs.mkdir()
            same = inputs / "png_same.rdc"
            mismatch = inputs / "png_mismatch.rdc"
            same.write_text("png-same", encoding="utf-8")
            mismatch.write_text("png-mismatch", encoding="utf-8")
            manifest = root / "manifest.tsv"
            self._write_manifest(
                manifest,
                [("png_same", same), ("png_mismatch", mismatch)],
            )
            golden, pvrgpu = self._write_fake_runners(root)
            output = root / "output"

            completed = self._run(
                input_root=inputs,
                output_root=output,
                manifest=manifest,
                golden=golden,
                pvrgpu=pvrgpu,
            )

            self.assertEqual(completed.returncode, 1, completed.stderr)
            run_root = self._single_run_root(output)
            run_data = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
            self.assertEqual((run_data["total"], run_data["passed"], run_data["failed"]), (2, 1, 1))
            results = {result["case"]: result for result in run_data["results"]}
            self.assertEqual(results["png_same"]["status"], "PASS")
            self.assertEqual(results["png_same"]["compare"], "PASS")
            self.assertEqual(results["png_same"]["png"], "PASS")
            self.assertTrue(results["png_same"]["golden_png"].endswith("golden.png"))
            self.assertIn("/png/", results["png_same"]["pvrgpu_png"])

            failed = results["png_mismatch"]
            self.assertEqual(failed["status"], "FAIL")
            self.assertEqual(failed["compare"], "PASS")
            self.assertEqual(failed["png"], "FAIL")
            self.assertIn("RGBA_MISMATCH", failed["reason"])
            failed_root = run_root / failed["artifact_dir"]
            self.assertTrue((failed_root / "png_diff.txt").is_file())
            report = (run_root / "report.md").read_text(encoding="utf-8")
            self.assertIn("PNG comparison", report)
            self.assertIn("PNG | Result", report)

    def test_missing_pngs_fail_without_explicit_no_color_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs"
            inputs.mkdir()
            rdc = inputs / "png_both_missing.rdc"
            rdc.write_text("png-both-missing", encoding="utf-8")
            manifest = root / "manifest.tsv"
            self._write_manifest(manifest, [("png_both_missing", rdc)])
            golden, pvrgpu = self._write_fake_runners(root)
            output = root / "output"

            completed = self._run(
                input_root=inputs,
                output_root=output,
                manifest=manifest,
                golden=golden,
                pvrgpu=pvrgpu,
            )

            self.assertEqual(completed.returncode, 1, completed.stderr)
            run_root = self._single_run_root(output)
            run_data = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
            result = run_data["results"][0]
            self.assertEqual(result["status"], "FAIL")
            self.assertEqual(result["compare"], "PASS")
            self.assertEqual(result["png"], "FAIL")
            self.assertIn("both missing", result["reason"])

    def test_png_compare_is_skipped_for_draw_counter_with_no_pixel_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs"
            inputs.mkdir()
            rdc = inputs / "png_no_color_write.rdc"
            rdc.write_text("png-no-color-write", encoding="utf-8")
            manifest = root / "manifest.tsv"
            self._write_manifest(manifest, [("png_no_color_write", rdc)])
            golden, pvrgpu = self._write_fake_runners(root)
            output = root / "output"

            completed = self._run(
                input_root=inputs,
                output_root=output,
                manifest=manifest,
                golden=golden,
                pvrgpu=pvrgpu,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            run_root = self._single_run_root(output)
            run_data = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
            result = run_data["results"][0]
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["compare"], "PASS")
            self.assertEqual(result["png"], "SKIP")
            self.assertEqual(result["png_diff"], "png_skip.txt")
            self.assertIn("ps_invocations=0", (run_root / result["artifact_dir"] / "png_skip.txt").read_text(encoding="utf-8"))

    def test_png_compare_is_skipped_when_selected_action_has_no_color_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs"
            inputs.mkdir()
            rdc = inputs / "png_no_selected_color.rdc"
            rdc.write_text("png-no-selected-color", encoding="utf-8")
            manifest = root / "manifest.tsv"
            self._write_manifest(manifest, [("png_no_selected_color", rdc)])
            golden, pvrgpu = self._write_fake_runners(root)
            output = root / "output"

            completed = self._run(
                input_root=inputs,
                output_root=output,
                manifest=manifest,
                golden=golden,
                pvrgpu=pvrgpu,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            run_root = self._single_run_root(output)
            run_data = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
            result = run_data["results"][0]
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["compare"], "PASS")
            self.assertEqual(result["png"], "SKIP")
            self.assertEqual(result["golden_png"], "")
            self.assertIn("/png/", result["pvrgpu_png"])
            case_root = run_root / result["artifact_dir"]
            skip_text = (case_root / "png_skip.txt").read_text(encoding="utf-8")
            self.assertIn("no color output", skip_text)
            self.assertNotIn("ps_invocations=0", skip_text)
            digest = hashlib.sha256(rdc.read_bytes()).hexdigest()
            metadata = json.loads(
                (output / "golden-cache" / digest / "metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertIs(metadata["no_color_output"], True)

    def test_pvrgpu_runner_receives_actual_golden_png_extent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs"
            inputs.mkdir()
            rdc = inputs / "png_size_from_golden.rdc"
            rdc.write_text("png-size-from-golden", encoding="utf-8")
            manifest = root / "manifest.tsv"
            self._write_manifest(manifest, [("png_size_from_golden", rdc)])
            golden, pvrgpu = self._write_fake_runners(root)
            output = root / "output"

            completed = self._run(
                input_root=inputs,
                output_root=output,
                manifest=manifest,
                golden=golden,
                pvrgpu=pvrgpu,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            run_root = self._single_run_root(output)
            run_data = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
            result = run_data["results"][0]
            self.assertEqual(result["status"], "PASS")
            pvrgpu_png = run_root / result["artifact_dir"] / result["pvrgpu_png"]
            self.assertEqual(decode_rgba8_png(pvrgpu_png)[:2], (32, 48))

    def test_require_manifest_fails_unmapped_rdc_before_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs"
            inputs.mkdir()
            known = inputs / "known.rdc"
            known.write_text("pass", encoding="utf-8")
            unknown = inputs / "unknown.rdc"
            unknown.write_text("pass-unknown", encoding="utf-8")
            manifest = root / "manifest.tsv"
            self._write_manifest(manifest, [("known", known)])
            golden, pvrgpu = self._write_fake_runners(root)
            output = root / "output"

            completed = self._run(
                input_root=inputs,
                output_root=output,
                manifest=manifest,
                golden=golden,
                pvrgpu=pvrgpu,
                require_manifest=True,
            )

            self.assertEqual(completed.returncode, 1, completed.stderr)
            run_root = self._single_run_root(output)
            run_data = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
            self.assertEqual((run_data["total"], run_data["passed"], run_data["failed"]), (2, 1, 1))
            results = {result["rdc"]: result for result in run_data["results"]}
            self.assertEqual(results["known.rdc"]["status"], "PASS")
            self.assertEqual(results["unknown.rdc"]["stage"], "manifest-map")
            self.assertEqual(results["unknown.rdc"]["golden"], "SKIP")
            self.assertIn("not present in the frozen manifest", results["unknown.rdc"]["reason"])

    def test_existing_golden_counter_cache_skips_second_golden_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs"
            inputs.mkdir()
            rdc = inputs / "cached.rdc"
            rdc.write_text("pass", encoding="utf-8")
            manifest = root / "manifest.tsv"
            self._write_manifest(manifest, [("cached_case", rdc)])
            golden, pvrgpu = self._write_fake_runners(root)
            output = root / "output"

            first = self._run(
                input_root=inputs,
                output_root=output,
                manifest=manifest,
                golden=golden,
                pvrgpu=pvrgpu,
                reuse_golden_cache=True,
            )
            self.assertEqual(first.returncode, 0, first.stderr)
            digest = hashlib.sha256(rdc.read_bytes()).hexdigest()
            self.assertTrue(
                (output / "golden-cache" / digest / "counter_golden.txt").is_file()
            )

            golden.write_text(
                "#!/usr/bin/env python3\n"
                "import sys\n"
                "print('golden should have been cached', file=sys.stderr)\n"
                "raise SystemExit(77)\n",
                encoding="utf-8",
            )
            golden.chmod(0o755)

            second = self._run(
                input_root=inputs,
                output_root=output,
                manifest=manifest,
                golden=golden,
                pvrgpu=pvrgpu,
                reuse_golden_cache=True,
            )
            self.assertEqual(second.returncode, 0, second.stderr)
            events = [json.loads(line) for line in second.stdout.splitlines()]
            cached_stage = next(
                event
                for event in events
                if event["type"] == "stage"
                and event["stage"] == "golden"
                and event["status"] == "CACHED"
            )
            self.assertEqual(cached_stage["status"], "CACHED")
            run_data = json.loads(
                (self._latest_run_root(output) / "run.json").read_text(encoding="utf-8")
            )
            result = run_data["results"][0]
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["golden"], "CACHED")
            self.assertTrue(result["golden_cache"].endswith("counter_golden.txt"))
            case_root = self._latest_run_root(output) / result["artifact_dir"]
            self.assertTrue((case_root / "golden" / "cache-hit.txt").is_file())
            self.assertFalse((case_root / "golden" / "stderr.log").exists())

    def test_golden_cache_is_not_reused_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs"
            inputs.mkdir()
            rdc = inputs / "fresh.rdc"
            rdc.write_text("pass", encoding="utf-8")
            manifest = root / "manifest.tsv"
            self._write_manifest(manifest, [("fresh_case", rdc)])
            golden, pvrgpu = self._write_fake_runners(root)
            output = root / "output"

            first = self._run(
                input_root=inputs,
                output_root=output,
                manifest=manifest,
                golden=golden,
                pvrgpu=pvrgpu,
            )
            self.assertEqual(first.returncode, 0, first.stderr)

            golden.write_text(
                "#!/usr/bin/env python3\n"
                "import sys\n"
                "print('fresh golden failure', file=sys.stderr)\n"
                "raise SystemExit(77)\n",
                encoding="utf-8",
            )
            golden.chmod(0o755)

            second = self._run(
                input_root=inputs,
                output_root=output,
                manifest=manifest,
                golden=golden,
                pvrgpu=pvrgpu,
            )
            self.assertEqual(second.returncode, 1, second.stderr)
            run_data = json.loads(
                (self._latest_run_root(output) / "run.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(run_data["results"][0]["golden"], "FAIL")
            self.assertIn("code 77", run_data["results"][0]["reason"])

    def test_runner_timeout_does_not_prevent_later_rdc(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs"
            inputs.mkdir()
            slow = inputs / "a_slow.rdc"
            later = inputs / "b_later.rdc"
            slow.write_text("slow-golden", encoding="utf-8")
            later.write_text("pass", encoding="utf-8")
            manifest = root / "manifest.tsv"
            self._write_manifest(manifest, [("slow", slow), ("later", later)])
            golden, pvrgpu = self._write_fake_runners(root)
            output = root / "output"

            completed = self._run(
                input_root=inputs,
                output_root=output,
                manifest=manifest,
                golden=golden,
                pvrgpu=pvrgpu,
                timeout=1,
            )

            self.assertEqual(completed.returncode, 1, completed.stderr)
            run_data = json.loads(
                (self._single_run_root(output) / "run.json").read_text(encoding="utf-8")
            )
            self.assertIn("timeout", run_data["results"][0]["reason"])
            self.assertEqual(run_data["results"][1]["status"], "PASS")

    def test_empty_directory_still_writes_failure_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs"
            inputs.mkdir()
            manifest_rdc = root / "manifest-source.rdc"
            manifest_rdc.write_text("manifest-only", encoding="utf-8")
            manifest = root / "manifest.tsv"
            self._write_manifest(manifest, [("case", manifest_rdc)])
            golden, pvrgpu = self._write_fake_runners(root)
            output = root / "output"

            completed = self._run(
                input_root=inputs,
                output_root=output,
                manifest=manifest,
                golden=golden,
                pvrgpu=pvrgpu,
            )

            self.assertEqual(completed.returncode, 1, completed.stderr)
            run_root = self._single_run_root(output)
            report = (run_root / "report.md").read_text(encoding="utf-8")
            self.assertIn("No .rdc files were found", report)
            self.assertIn("Overall: **FAIL**", report)


if __name__ == "__main__":
    unittest.main()
