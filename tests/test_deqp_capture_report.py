from __future__ import annotations

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

from deqp_capture_report import build_argument_parser  # noqa: E402
from deqp_capture_report import classify_phase, parse_phase_selector  # noqa: E402


WORKER = TOOLS_DIR / "deqp_capture_report.py"


class DeqpCaptureReportTests(unittest.TestCase):
    def test_default_runners_come_from_the_native_build_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build_root = Path(directory) / "native-build"
            with mock.patch.dict(
                os.environ, {"PVRGPU_BUILD_DIR": str(build_root)}, clear=False
            ):
                os.environ.pop("PVRGPU_RDC_GOLDEN_RUNNER", None)
                os.environ.pop("PVRGPU_RDC_PVRGPU_RUNNER", None)
                options = build_argument_parser().parse_args([])

            suffix = ".exe" if os.name == "nt" else ""
            self.assertEqual(
                options.golden_runner, build_root / "bin" / f"llvmpipe{suffix}"
            )
            self.assertEqual(
                options.pvrgpu_runner, build_root / "bin" / f"pvrgpu{suffix}"
            )

    def _write_capture(
        self,
        root: Path,
        *,
        suite: str,
        group: str,
        filename: str,
        case_name: str,
        status: str = "OK",
    ) -> Path:
        recorder = root / suite / f"{group}." / "recorder"
        trace_dir = recorder / "trace"
        trace_dir.mkdir(parents=True)
        rdc = trace_dir / filename
        rdc.write_bytes(f"fake-rdc:{case_name}".encode("utf-8"))
        (recorder / "manifest.txt").write_text(
            f"1 | {case_name} | {filename} | {status}\n",
            encoding="utf-8",
        )
        return rdc

    @staticmethod
    def _single_run_root(output_root: Path) -> Path:
        roots = list(output_root.iterdir())
        if len(roots) != 1:
            raise AssertionError(f"expected one run root, found {roots}")
        return roots[0]

    def test_phase_contract_reaches_phase6(self) -> None:
        self.assertEqual(
            classify_phase(
                "dEQP-GLES31.functional.compute.basic",
                "dEQP-GLES31.functional.compute",
            ).number,
            6,
        )
        self.assertEqual(
            classify_phase(
                "dEQP-GLES3.functional.color_clear.single_rgb",
                "dEQP-GLES3.functional.color_clear",
            ).number,
            1,
        )
        self.assertEqual(parse_phase_selector("phase6-advanced-large"), 6)

    def test_catalog_phase_max_writes_ready_without_deqp_binary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs"
            output = root / "output"
            self._write_capture(
                inputs,
                suite="gles3",
                group="dEQP-GLES3.functional.color_clear",
                filename="0001_color_clear_single_rgb_capture.rdc",
                case_name="dEQP-GLES3.functional.color_clear.single_rgb",
            )
            self._write_capture(
                inputs,
                suite="gles31",
                group="dEQP-GLES31.functional.compute",
                filename="0001_basic_capture.rdc",
                case_name="dEQP-GLES31.functional.compute.basic",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(WORKER),
                    "--rdc-dir",
                    str(inputs),
                    "--output-root",
                    str(output),
                    "--phase-max",
                    "6",
                    "--json",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=20,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            run_root = self._single_run_root(output)
            run_data = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
            self.assertFalse(run_data["uses_deqp_binary"])
            self.assertEqual(run_data["status"], "PASS")
            self.assertEqual([result["status"] for result in run_data["results"]], ["READY", "READY"])
            self.assertEqual(
                [result["phase"] for result in run_data["results"]],
                [1, 6],
            )

    def test_native_counter_artifacts_compare_the_same_rdc(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs"
            output = root / "output"
            rdc = self._write_capture(
                inputs,
                suite="gles3",
                group="dEQP-GLES3.functional.color_clear",
                filename="0001_color_clear_single_rgb_capture.rdc",
                case_name="dEQP-GLES3.functional.color_clear.single_rgb",
            )
            runner = root / "fake_native_runner.py"
            runner.write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env python3
                    import argparse
                    import hashlib
                    from pathlib import Path

                    parser = argparse.ArgumentParser()
                    parser.add_argument("--rdc", type=Path, required=True)
                    parser.add_argument("--case", required=True)
                    parser.add_argument("--outdir", type=Path, required=True)
                    options = parser.parse_args()
                    options.outdir.mkdir(parents=True, exist_ok=True)
                    (options.outdir / "counter.txt").write_text(
                        "drawlists=1\\nps_invocations=3\\n", encoding="utf-8"
                    )
                    (options.outdir / "input.sha256").write_text(
                        hashlib.sha256(options.rdc.read_bytes()).hexdigest() + "\\n",
                        encoding="utf-8",
                    )
                    """
                ).lstrip(),
                encoding="utf-8",
            )
            runner.chmod(0o755)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(WORKER),
                    "--rdc-dir",
                    str(inputs),
                    "--output-root",
                    str(output),
                    "--run-golden",
                    "--run-pvrgpu",
                    "--golden-runner",
                    str(runner),
                    "--pvrgpu-runner",
                    str(runner),
                    "--json",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=20,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            run_root = self._single_run_root(output)
            run_data = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
            [result] = run_data["results"]
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["golden_counter"], "golden/counter.txt")
            self.assertEqual(result["pvrgpu_counter"], "pvrgpu/counter.txt")
            digest = hashlib.sha256(rdc.read_bytes()).hexdigest()
            case_root = run_root / result["artifact_dir"]
            self.assertEqual(
                (case_root / "golden" / "input.sha256").read_text().strip(), digest
            )
            self.assertEqual(
                (case_root / "pvrgpu" / "input.sha256").read_text().strip(), digest
            )

    def test_pvrgpu_exit_code_3_is_unsupported_without_fake_counter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs"
            output = root / "output"
            self._write_capture(
                inputs,
                suite="gles31",
                group="dEQP-GLES31.functional.compute",
                filename="0001_basic_capture.rdc",
                case_name="dEQP-GLES31.functional.compute.basic",
            )
            pvrgpu_runner = root / "fake_pvrgpu_unsupported.py"
            pvrgpu_runner.write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env python3
                    import argparse
                    from pathlib import Path

                    parser = argparse.ArgumentParser()
                    parser.add_argument("--outdir", type=Path, required=True)
                    parser.add_argument("--case", required=True)
                    options, _unknown = parser.parse_known_args()
                    options.outdir.mkdir(parents=True, exist_ok=True)
                    (options.outdir / "unsupported.txt").write_text(
                        f"unsupported phase6-advanced-large for {options.case}\\n",
                        encoding="utf-8",
                    )
                    raise SystemExit(3)
                    """
                ).lstrip(),
                encoding="utf-8",
            )
            pvrgpu_runner.chmod(0o755)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(WORKER),
                    "--rdc-dir",
                    str(inputs),
                    "--output-root",
                    str(output),
                    "--phase",
                    "6",
                    "--run-pvrgpu",
                    "--pvrgpu-runner",
                    str(pvrgpu_runner),
                    "--json",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=20,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            run_root = self._single_run_root(output)
            run_data = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(run_data["status"], "PASS")
            [result] = run_data["results"]
            self.assertEqual(result["status"], "UNSUPPORTED")
            self.assertEqual(result["pvrgpu"], "UNSUPPORTED")
            self.assertIn("phase6-advanced-large", result["reason"])
            self.assertFalse(
                (run_root / result["artifact_dir"] / "pvrgpu" / "counter.txt").exists()
            )


if __name__ == "__main__":
    unittest.main()
