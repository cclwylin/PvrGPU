#!/usr/bin/env python3
"""Offscreen end-to-end smoke test for the PvrGPU live dEQP UI."""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import sys
import tempfile
import textwrap

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

from PySide6.QtCore import QEventLoop, QProcess, QSettings, QTimer
from PySide6.QtWidgets import QApplication

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from deqp_live_ui import MainWindow


EXACT_CASE = "dEQP-GLES2.functional.prerequisite.clear_color"
PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _write_fake_runner(path: Path) -> None:
    body = f'''#!/usr/bin/env python3
import base64
from pathlib import Path
import re
import sys
import time

arguments = sys.argv[1:]

def value(prefix):
    for argument in arguments:
        if argument.startswith(prefix):
            return argument.split("=", 1)[1]
    raise SystemExit("missing argument: " + prefix)

output_dir = Path(value("--pvrgpu-output-dir="))
case_name = value("--deqp-case=")
if "--deqp-runmode=txt-caselist" in arguments:
    export_file = Path(value("--deqp-caselist-export-file="))
    export_file.write_text(
        "TEST: dEQP-EGL.functional.create_context.alpha\\n"
        "TEST: dEQP-EGL.functional.image.unrelated\\n"
        "TEST: dEQP-EGL.functional.create_context.beta\\n",
        encoding="utf-8",
    )
    print("fake pvrgpu-deqp discovery complete", flush=True)
    raise SystemExit(0)
safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", case_name) or "unnamed"
case_dir = output_dir / "cases" / safe_name
systemc_dir = case_dir / "systemc"
systemc_dir.mkdir(parents=True, exist_ok=True)

print("fake pvrgpu-deqp started", flush=True)
time.sleep(0.25)

(output_dir / "runner-arguments.txt").write_text(
    "\\n".join(arguments) + "\\n", encoding="utf-8"
)
(case_dir / "driver-command.txt").write_text(
    "case=" + case_name + "\\nop=clear_color\\n", encoding="utf-8"
)
(case_dir / "driver-counter.txt").write_text(
    "virtual_gpu_cycles=128\\nfragment_jobs=1\\n", encoding="utf-8"
)
(case_dir / "systemc.jsonl").write_text(
    '{{"type":"hello","schema":"pvrgpu.systemc.v1"}}\\n'
    '{{"type":"done","status":"ok"}}\\n',
    encoding="utf-8",
)
(systemc_dir / "framebuffer-0001.png").write_bytes(base64.b64decode({PNG_BASE64!r}))
(output_dir / "results.qpa").write_text(
    '#sessionInfo vendor "Imagination Technologies"\\n'
    '#sessionInfo renderer "PvrGPU Mesa Gallium (SystemC)"\\n'
    '#beginTestCaseResult ' + case_name + '\\n'
    '<Number Name="TestDuration" Description="Test duration in microseconds" '
    'Unit="us" Tag="Time">1234</Number>\\n'
    '<Result StatusCode="Pass">Pass</Result>\\n'
    '#endTestCaseResult\\n',
    encoding="utf-8",
)
print("fake pvrgpu-deqp PASS", flush=True)
'''
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    path.chmod(0o755)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _exercise_terminal_states(root: Path, runner: Path) -> None:
    scenarios = (
        ("no-match", "", QProcess.ExitStatus.NormalExit, "NO MATCH"),
        (
            "unsupported",
            '#beginTestCaseResult ' + EXACT_CASE + '\n'
            '<Result StatusCode="NotSupported">unsupported</Result>\n'
            '#endTestCaseResult\n',
            QProcess.ExitStatus.NormalExit,
            "SKIPPED",
        ),
        (
            "incomplete",
            '#beginTestCaseResult ' + EXACT_CASE + '\n',
            QProcess.ExitStatus.NormalExit,
            "INCOMPLETE",
        ),
        ("crashed", "", QProcess.ExitStatus.CrashExit, "CRASHED"),
    )
    for name, qpa_text, exit_status, expected_state in scenarios:
        run_dir = root / f"status-{name}"
        run_dir.mkdir()
        if qpa_text:
            (run_dir / "results.qpa").write_text(qpa_text, encoding="utf-8")
        settings = QSettings(
            str(root / f"settings-{name}.ini"), QSettings.Format.IniFormat
        )
        settings.setFallbacksEnabled(False)
        window = MainWindow(runner_path=runner, settings=settings)
        window.case_edit.setText(EXACT_CASE)
        window.run_dir = run_dir
        window._refresh_results(0, exit_status)
        _require(
            window.state_badge.text() == expected_state,
            f"{name} produced {window.state_badge.text()}, expected {expected_state}",
        )
        if name == "unsupported":
            _require(window.driver_value.text() == "N/A · no GPU submit",
                     "NotSupported incorrectly required driver artifacts")
        window.close()


def _exercise_group_mode(root: Path, runner: Path) -> None:
    output_root = root / "group-runs"
    settings = QSettings(str(root / "settings-group.ini"), QSettings.Format.IniFormat)
    settings.setFallbacksEnabled(False)
    window = MainWindow(runner_path=runner, settings=settings)
    window.runner_edit.setText(str(runner))
    window.output_root_edit.setText(str(output_root))
    window.mode_combo.setCurrentIndex(window.mode_combo.findData("group"))
    window.group_combo.setCurrentIndex(window.group_combo.findData("egl-create-context"))
    loop = QEventLoop()
    timed_out = {"value": False}

    def timeout() -> None:
        timed_out["value"] = True
        if window.process.state() != QProcess.ProcessState.NotRunning:
            window.process.kill()
        loop.quit()

    window.process.finished.connect(loop.quit)
    window._start()
    _require(window.process.state() != QProcess.ProcessState.NotRunning,
             "group worker did not start")
    QTimer.singleShot(10_000, timeout)
    loop.exec()

    _require(not timed_out["value"], "timed out waiting for group worker")
    _require(window.state_badge.text() == "PASS",
             f"group run produced {window.state_badge.text()}")
    _require(window.results_table.rowCount() == 2,
             "group results were not aggregated")
    _require(window.batch_progress.maximum() == 2 and window.batch_progress.value() == 2,
             "group progress did not reach 2/2")
    _require(window.run_dir is not None, "group output was not created")
    _require((window.run_dir / "batch-summary.json").is_file(),
             "group summary is missing")
    _require((window.run_dir / "results.qpa").read_text().count("#beginTestCaseResult") == 2,
             "group QPA does not contain both exact cases")
    _require(any(value == "--group-id=egl-create-context" for value in window.last_command),
             "group id was not passed to the batch helper")
    window.close()


def main() -> int:
    app = QApplication([])
    result = {"error": None, "finished": False}

    with tempfile.TemporaryDirectory(prefix="pvrgpu-deqp-live-ui-") as directory:
        root = Path(directory)
        runner = root / "fake-pvrgpu-deqp"
        output_root = root / "runs"
        _write_fake_runner(runner)

        settings = QSettings(str(root / "settings.ini"), QSettings.Format.IniFormat)
        settings.setFallbacksEnabled(False)
        window = MainWindow(
            runner_path=runner,
            settings=settings,
            now_provider=lambda: datetime(2026, 9, 1, 12, 34, 56, 789000),
        )
        _require(window.group_combo.count() == 24, "UI does not expose all 24 groups")
        blocked_group = window.group_combo.findData("gles3-fbo")
        _require(blocked_group >= 0, "GLES3 FBO group is missing")
        _require(window.group_combo.model().item(blocked_group).isEnabled(),
                 "blocked group cannot be selected to inspect its reason")
        exact_mode = window.mode_combo.findData("exact")
        window.mode_combo.setCurrentIndex(exact_mode)
        window.suite_combo.setCurrentIndex(window.suite_combo.findData("dEQP-GLES2"))
        window.runner_edit.setText(str(runner))
        window.output_root_edit.setText(str(output_root))
        window.bridge_edit.clear()
        window._case_preset_activated(1)
        _require(window.case_edit.text() == "dEQP-GLES2.info.vendor",
                 "preset label was passed as a case instead of its exact path")
        suite_model = window.suite_combo.model()
        _require(not suite_model.item(1).isEnabled() and not suite_model.item(2).isEnabled(),
                 "currently unavailable GLES3 suites remained selectable")
        window.case_edit.setText(EXACT_CASE)
        window.show()

        def fail(error: BaseException) -> None:
            result["error"] = error
            app.quit()

        def verify_running() -> None:
            try:
                window._start()
                expected_output_arg = f"--pvrgpu-output-dir={window.run_dir}"
                expected_args = {
                    expected_output_arg,
                    f"--deqp-case={EXACT_CASE}",
                    "--deqp-surface-type=pbuffer",
                    "--deqp-watchdog=disable",
                    "--deqp-log-images=disable",
                    "--deqp-gl-config-name=rgba8888d24s8ms0",
                }
                _require(window.run_dir is not None, "run directory was not created")
                _require(window.process.state() != QProcess.ProcessState.NotRunning,
                         "fake runner did not remain active long enough to observe RUNNING")
                _require(window.state_badge.text() == "RUNNING", "state badge is not RUNNING")
                _require(not window.start_button.isEnabled(), "Run remained enabled while running")
                _require(window.stop_button.isEnabled(), "Stop was not enabled while running")
                _require(not window.suite_combo.isEnabled(), "suite remained editable while running")
                _require(not window.case_combo.isEnabled(), "case remained editable while running")
                _require(not window.output_root_edit.isEnabled(),
                         "output root remained editable while running")
                _require(not window.open_output_button.isEnabled(),
                         "Open Output was enabled before completion")
                _require(not window.open_qpa_button.isEnabled(),
                         "Open results.qpa was enabled before completion")
                _require(window.last_command[0] == str(runner.resolve()),
                         f"wrong runner executable: {window.last_command[0]}")
                _require(expected_args.issubset(set(window.last_command[1:])),
                         f"runner arguments are incomplete: {window.last_command}")
            except BaseException as error:
                fail(error)

        def verify_finished(*_args: object) -> None:
            if result["error"] is not None:
                return
            try:
                _require(window.process.state() == QProcess.ProcessState.NotRunning,
                         "runner finished signal arrived while still running")
                _require(window.state_badge.text() == "PASS", "successful run was not PASS")
                _require(window.start_button.isEnabled(), "Run was not restored after completion")
                _require(not window.stop_button.isEnabled(), "Stop remained enabled after completion")
                _require(window.suite_combo.isEnabled(), "suite was not restored after completion")
                _require(window.case_combo.isEnabled(), "case was not restored after completion")
                _require(window.output_root_edit.isEnabled(),
                         "output root was not restored after completion")
                _require(window.open_output_button.isEnabled(),
                         "Open Output was not enabled after completion")
                _require(window.open_qpa_button.isEnabled(),
                         "Open results.qpa was not enabled after completion")

                _require(window.deqp_value.text() == "1 pass · 0 skip · 0 warn · 0 fail",
                         f"unexpected dEQP card: {window.deqp_value.text()}")
                _require(window.driver_value.text() == "1 / 1 ready",
                         f"unexpected driver card: {window.driver_value.text()}")
                _require(window.systemc_value.text() == "1 / 1 complete",
                         f"unexpected SystemC card: {window.systemc_value.text()}")
                _require(window.renderer_value.text() == "PvrGPU Mesa Gallium (SystemC)",
                         f"unexpected renderer: {window.renderer_value.text()}")
                _require(window.results_table.rowCount() == 1,
                         "result table does not contain the exact case")
                expected_row = (
                    EXACT_CASE,
                    "PASS",
                    "READY",
                    "READY",
                    "1.2 ms",
                )
                actual_row = tuple(
                    window.results_table.item(0, column).text() for column in range(5)
                )
                _require(actual_row == expected_row,
                         f"unexpected result table row: {actual_row}")

                artifact_names = {
                    window.artifacts_table.item(row, 0).text()
                    for row in range(window.artifacts_table.rowCount())
                }
                safe_case = EXACT_CASE
                expected_artifacts = {
                    "results.qpa",
                    "runner-arguments.txt",
                    f"cases/{safe_case}/driver-command.txt",
                    f"cases/{safe_case}/driver-counter.txt",
                    f"cases/{safe_case}/systemc.jsonl",
                    f"cases/{safe_case}/systemc/framebuffer-0001.png",
                }
                _require(expected_artifacts.issubset(artifact_names),
                         f"artifact table is incomplete: {sorted(artifact_names)}")
                _require(window.latest_png is not None and window.latest_png.is_file(),
                         "SystemC PNG was not discovered")
                _require(not window.image_preview.pixmap().isNull(),
                         "valid PNG was not rendered in the artifact preview")
                _require("fake pvrgpu-deqp started" in window.log.toPlainText()
                         and "fake pvrgpu-deqp PASS" in window.log.toPlainText(),
                         "live process output was not captured")

                argument_log = (window.run_dir / "runner-arguments.txt").read_text(
                    encoding="utf-8"
                )
                _require(f"--deqp-case={EXACT_CASE}" in argument_log,
                         "fake runner did not receive the exact case")
                result["finished"] = True
            except BaseException as error:
                result["error"] = error
            finally:
                app.quit()

        def timed_out() -> None:
            if not result["finished"] and result["error"] is None:
                fail(RuntimeError("timed out waiting for the fake dEQP runner"))

        window.process.finished.connect(verify_finished)
        QTimer.singleShot(0, verify_running)
        QTimer.singleShot(10_000, timed_out)
        app.exec()
        window.close()

        if result["error"] is not None:
            raise result["error"]  # type: ignore[misc]
        _require(bool(result["finished"]), "finished verification was not reached")
        _exercise_group_mode(root, runner)
        _exercise_terminal_states(root, runner)

    print("DEQP_LIVE_UI_SMOKE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
