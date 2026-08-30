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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from counter_protocol import STANDARD_COUNTER_FIELDS  # noqa: E402


WORKER = TOOLS_DIR / "rdc_counter_report.py"


class RdcCounterReportTests(unittest.TestCase):
    def _write_fake_runners(self, root: Path) -> tuple[Path, Path]:
        fields = repr(tuple(STANDARD_COUNTER_FIELDS))
        common = f"""
            #!/usr/bin/env python3
            import argparse
            import hashlib
            import json
            from pathlib import Path
            import sys
            import time

            FIELDS = {fields}
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
                digest = hashlib.sha256(args.rdc.read_bytes()).hexdigest()
                messages = [
                    {
                        'schema': 'pvrgpu.counter.v1',
                        'type': 'hello',
                        'backend': 'pvrgpu',
                        'mesa_command_ingest': True,
                        'command_source': 'renderdoc-mesa-gallium-trace-poc',
                        'mesa_command_schema': 'pvrgpu.mesa-poc-command.v1',
                        'rdc_sha256': digest,
                        'api_trace_sha256': 'a' * 64,
                        'gallium_trace_sha256': 'f' * 64,
                    },
                    {
                        'schema': 'pvrgpu.counter.v1',
                        'type': 'counter',
                        'frame': 1,
                        'source': 'fake-pvrgpu',
                        'provenance': 'modeled',
                        'counters': values,
                    },
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
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
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
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=20,
        )

    @staticmethod
    def _single_run_root(output_root: Path) -> Path:
        roots = list(output_root.iterdir())
        if len(roots) != 1:
            raise AssertionError(f"expected one run root, found {roots}")
        return roots[0]

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
            self.assertEqual((run_data["total"], run_data["passed"], run_data["failed"]), (6, 2, 4))
            results = {result["case"]: result for result in run_data["results"]}
            self.assertEqual(results["a_pass"]["status"], "PASS")
            self.assertEqual(results["b_mismatch"]["stage"], "compare")
            mismatch_root = run_root / results["b_mismatch"]["artifact_dir"]
            self.assertTrue((mismatch_root / "counter_diff.txt").is_file())
            self.assertEqual(results["c_golden_error"]["stage"], "golden")
            self.assertEqual(results["c_golden_error"]["pvrgpu"], "SKIP")
            self.assertEqual(results["d_pvrgpu_error"]["stage"], "pvrgpu")
            self.assertEqual(results["e_after_errors"]["status"], "PASS")
            self.assertEqual(results["UNMAPPED"]["stage"], "manifest-map")
            report = (run_root / "report.md").read_text(encoding="utf-8")
            self.assertIn("Overall: **FAIL**", report)
            self.assertIn("17-counter exact comparison mismatch", report)
            self.assertIn("not present in the frozen manifest", report)

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
