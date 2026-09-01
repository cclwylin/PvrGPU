from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GROUP_RUNNER = PROJECT_ROOT / "tools" / "deqp_group_runner.py"


FAKE_RUNNER = r'''#!/usr/bin/env python3
from pathlib import Path
import json
import re
import sys

arguments = sys.argv[1:]

def value(prefix):
    for argument in arguments:
        if argument.startswith(prefix):
            return argument.split("=", 1)[1]
    return ""

output_dir = Path(value("--pvrgpu-output-dir="))
output_dir.mkdir(parents=True, exist_ok=True)
invocation_log = Path(__file__).with_suffix(".jsonl")
with invocation_log.open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(arguments) + "\n")

if value("--deqp-runmode=") == "txt-caselist":
    export_file = Path(value("--deqp-caselist-export-file="))
    export_file.write_text(
        "GROUP: dEQP-EGL.functional.create_context\n"
        "TEST: dEQP-EGL.functional.create_context.alpha\n"
        "TEST: dEQP-EGL.functional.image.unrelated\n"
        "TEST: dEQP-EGL.functional.create_context.beta\n",
        encoding="utf-8",
    )
    print("fake discovery complete")
    raise SystemExit(0)

case_name = value("--deqp-case=")
if not case_name or "*" in case_name or "," in case_name:
    raise SystemExit("execution requires one exact case")
safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", case_name)
case_dir = output_dir / "cases" / safe_name
(case_dir / "systemc").mkdir(parents=True, exist_ok=True)
(case_dir / "driver-command.txt").write_text("case=" + case_name + "\n")
(case_dir / "driver-counter.txt").write_text("jobs=1\n")
(case_dir / "systemc.jsonl").write_text('{"type":"done"}\n')
(case_dir / "systemc" / "frame.png").write_bytes(b"PNG")
status = "NotSupported" if case_name.endswith("beta") else "Pass"
(output_dir / "results.qpa").write_text(
    '#sessionInfo vendor "PvrGPU"\n'
    '#sessionInfo renderer "fake grouped runner"\n\n'
    '#beginSession\n\n'
    '#beginTestCaseResult ' + case_name + '\n'
    '<TestCaseResult CasePath="' + case_name + '">\n'
    '<Result StatusCode="' + status + '">' + status + '</Result>\n'
    '</TestCaseResult>\n\n'
    '#endTestCaseResult\n\n#endSession\n',
    encoding="utf-8",
)
print("fake exact " + case_name)
'''


class DeqpGroupRunnerTests(unittest.TestCase):
    def _fake_runner(self, root: Path) -> Path:
        runner = root / "fake-pvrgpu-deqp"
        runner.write_text(textwrap.dedent(FAKE_RUNNER), encoding="utf-8")
        runner.chmod(0o755)
        return runner

    def test_group_is_discovered_then_executed_one_exact_case_per_process(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pvrgpu-group-runner-") as directory:
            root = Path(directory)
            runner = self._fake_runner(root)
            output = root / "output"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(GROUP_RUNNER),
                    f"--runner={runner}",
                    f"--output-dir={output}",
                    "--group-id=egl-create-context",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stdout)
            self.assertIn('"event": "discovery_finished"', completed.stdout)
            self.assertIn('"event": "batch_finished"', completed.stdout)
            case_list = (output / "case-list.txt").read_text(encoding="utf-8").splitlines()
            self.assertEqual(
                case_list,
                [
                    "dEQP-EGL.functional.create_context.alpha",
                    "dEQP-EGL.functional.create_context.beta",
                ],
            )
            qpa = (output / "results.qpa").read_text(encoding="utf-8")
            self.assertEqual(qpa.count("#beginTestCaseResult"), 2)
            self.assertIn('StatusCode="Pass"', qpa)
            self.assertIn('StatusCode="NotSupported"', qpa)
            summary = json.loads((output / "batch-summary.json").read_text())
            self.assertEqual(summary["total"], 2)
            self.assertEqual(summary["completed"], 2)
            self.assertEqual(summary["passed"], 1)
            self.assertEqual(summary["skipped"], 1)
            self.assertEqual(summary["failed"], 0)

            invocations = [
                json.loads(line)
                for line in runner.with_suffix(".jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(invocations), 3)
            execution_cases = [
                next(value for value in call if value.startswith("--deqp-case="))
                for call in invocations[1:]
            ]
            self.assertEqual(
                execution_cases,
                [
                    "--deqp-case=dEQP-EGL.functional.create_context.alpha",
                    "--deqp-case=dEQP-EGL.functional.create_context.beta",
                ],
            )
            self.assertTrue(all("*" not in value for value in execution_cases))

    def test_unavailable_group_can_be_listed_but_not_executed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pvrgpu-group-list-") as directory:
            root = Path(directory)
            runner = self._fake_runner(root)
            output = root / "output"
            listed = subprocess.run(
                [
                    sys.executable,
                    str(GROUP_RUNNER),
                    f"--runner={runner}",
                    f"--output-dir={output}",
                    "--group-id=gles3-color-clear",
                    "--list-only",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            self.assertEqual(listed.returncode, 0, listed.stdout)
            self.assertTrue((output / "case-list.txt").is_file())


if __name__ == "__main__":
    unittest.main()
