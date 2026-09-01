from __future__ import annotations

import ast
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UI_SOURCE = PROJECT_ROOT / "tools" / "deqp_live_ui.py"
UI_SMOKE = PROJECT_ROOT / "tests" / "deqp_live_ui_smoke.py"
README = PROJECT_ROOT / "README.md"
CONFIG_EXAMPLE = PROJECT_ROOT / "config" / "local.env.example"


class LiveDeqpUiSourceTests(unittest.TestCase):
    def test_ui_source_is_valid_and_keeps_native_code_out_of_process(self) -> None:
        source = UI_SOURCE.read_text(encoding="utf-8")
        ast.parse(source, filename=str(UI_SOURCE))

        self.assertIn("QProcess(self)", source)
        self.assertIn("QSettings(\"PvrGPU\", \"LiveDeqp\")", source)
        self.assertIn("ProcessChannelMode.MergedChannels", source)
        self.assertIn("self.process.terminate()", source)
        self.assertIn("self.process.kill()", source)
        self.assertNotIn("ctypes.CDLL", source)

    def test_ui_owns_one_exact_case_and_fresh_external_output(self) -> None:
        source = UI_SOURCE.read_text(encoding="utf-8")

        for option in (
            "--pvrgpu-output-dir=",
            "--deqp-case=",
            "--deqp-surface-type=",
            "--deqp-watchdog=",
            "--deqp-log-images=",
            "--deqp-gl-config-name=",
        ):
            with self.subTest(option=option):
                self.assertIn(option, source)
        self.assertIn("EXACT_CASE_RE.fullmatch(case_name)", source)
        self.assertIn("Wildcards, comma lists", source)
        self.assertIn("PROJECT_ROOT in output_root.parents", source)
        self.assertNotIn("extra_args_edit", source)

    def test_ui_reports_qpa_process_and_artifact_states_separately(self) -> None:
        source = UI_SOURCE.read_text(encoding="utf-8")

        for state in (
            '"PASS"',
            '"SKIPPED"',
            '"WARNING"',
            '"FAIL"',
            '"NO MATCH"',
            '"CRASHED"',
            '"INCOMPLETE"',
            '"CANCELLED"',
        ):
            with self.subTest(state=state):
                self.assertIn(state, source)
        self.assertIn("#endTestCaseResult", source)
        self.assertIn("driver-command.txt", source)
        self.assertIn("driver-counter.txt", source)
        self.assertIn("systemc.jsonl", source)
        self.assertIn("QPixmap", source)

    def test_only_currently_runnable_suites_are_enabled(self) -> None:
        source = UI_SOURCE.read_text(encoding="utf-8")

        self.assertIn('"dEQP-GLES3":', source)
        self.assertIn('"dEQP-GLES31":', source)
        self.assertIn("item.setEnabled(False)", source)
        self.assertIn('"dEQP-GLES2"', source)
        self.assertIn('"dEQP-EGL"', source)

    def test_group_mode_expands_24_groups_into_isolated_exact_processes(self) -> None:
        source = UI_SOURCE.read_text(encoding="utf-8")
        group_source = (PROJECT_ROOT / "tools" / "deqp_groups.py").read_text(
            encoding="utf-8"
        )
        runner_source = (PROJECT_ROOT / "tools" / "deqp_group_runner.py").read_text(
            encoding="utf-8"
        )

        self.assertIn('self.mode_combo.addItem("24-group batch", "group")', source)
        self.assertIn("for group in GROUP_SPECS", source)
        self.assertIn("exact_case_belongs_to_group", source)
        self.assertIn("len(GROUP_SPECS) != 24", group_source)
        self.assertIn("filter_exact_cases(group, discovered)", runner_source)
        self.assertIn("_run_exact_case(args, case_name", runner_source)
        self.assertIn("batch-summary.json", runner_source)

    def test_launch_and_offscreen_smoke_are_documented(self) -> None:
        readme = README.read_text(encoding="utf-8")
        config = CONFIG_EXAMPLE.read_text(encoding="utf-8")

        self.assertIn("python3 tools/deqp_live_ui.py", readme)
        self.assertIn("PVRGPU_DEQP_UI_OUTPUT_ROOT=", config)
        self.assertTrue(UI_SMOKE.is_file())


if __name__ == "__main__":
    unittest.main()
