#!/usr/bin/env python3
"""Offscreen QProcess smoke test for the standalone RDC counter UI."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import textwrap
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QProcess, QSettings
from PySide6.QtWidgets import QApplication

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from rdc_counter_ui import MainWindow


def _pump_until(app: QApplication, predicate: object, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if bool(predicate()):  # type: ignore[operator]
            return
        time.sleep(0.01)
    raise RuntimeError("timed out waiting for the UI process")


def _write_worker(path: Path, *, slow: bool) -> None:
    if slow:
        body = r'''#!/usr/bin/env python3
import argparse
import json
import signal
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--rdc-dir", required=True)
parser.add_argument("--output-root", required=True)
parser.add_argument("--pvrgpu-runner", required=True)
parser.add_argument("--json", action="store_true")
options = parser.parse_args()

def emit(event_type, **values):
    print(json.dumps({"schema": "pvrgpu.rdc-counter-run.v1", "type": event_type, **values}), flush=True)

signal.signal(signal.SIGTERM, lambda *_args: sys.exit(143))
emit("run_started", input_root=options.rdc_dir, run_root=options.output_root + "/slow-run")
emit("scan_complete", total=1)
emit("rdc_started", index=1, total=1, rdc="nested/slow.rdc", case="fill_solid")
emit("stage", index=1, stage="golden", status="RUNNING")
while True:
    time.sleep(1)
'''
    else:
        body = r'''#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--rdc-dir", required=True)
parser.add_argument("--output-root", required=True)
parser.add_argument("--pvrgpu-runner", required=True)
parser.add_argument("--json", action="store_true")
options = parser.parse_args()
run_root = Path(options.output_root) / "fake-run"
run_root.mkdir(parents=True, exist_ok=True)

def emit(event_type, **values):
    print(json.dumps({"schema": "pvrgpu.rdc-counter-run.v1", "type": event_type, **values}), flush=True)

emit("run_started", input_root=options.rdc_dir, run_root=str(run_root))
emit("scan_complete", total=3)
time.sleep(0.12)
results = (("a.rdc", "fill_solid", "PASS", "CACHED"), ("nested/b.rdc", "triangle_setup", "FAIL", "PASS"), ("nested/c.rdc", "fill_tex_nearest", "PASS", "PASS"))
for index, (rdc, case, result, golden_status) in enumerate(results, start=1):
    emit("rdc_started", index=index, total=3, rdc=rdc, case=case)
    emit("stage", index=index, stage="golden", status="RUNNING")
    emit("stage", index=index, stage="golden", status=golden_status)
    emit("stage", index=index, stage="pvrgpu", status="RUNNING")
    emit("stage", index=index, stage="pvrgpu", status="PASS")
    emit("stage", index=index, stage="compare", status="RUNNING")
    emit("stage", index=index, stage="compare", status=result)
    emit("stage", index=index, stage="png", status="RUNNING")
    emit("stage", index=index, stage="png", status="SKIP")
    emit("rdc_result", index=index, rdc=rdc, case=case, status=result, png="SKIP", reason="counter mismatch" if result == "FAIL" else "exact match")
report = run_root / "report.md"
report.write_text("# Fake RDC report\n", encoding="utf-8")
print("hidden worker diagnostic", file=sys.stderr, flush=True)
emit("report_written", path=str(report))
emit("summary", total=3, passed=2, failed=1, status="FAIL", report=str(report))
raise SystemExit(1)
'''
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    path.chmod(0o755)


def main() -> int:
    app = QApplication([])
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        input_root = root / "input"
        output_root = root / "output"
        input_root.mkdir()
        (input_root / "nested").mkdir()
        for relative in ("a.rdc", "nested/b.rdc", "nested/c.rdc"):
            (input_root / relative).write_bytes(b"fake-rdc")

        worker = root / "fake_worker.py"
        _write_worker(worker, slow=False)
        pvrgpu_runner = root / "fake_pvrgpu_runner.sh"
        pvrgpu_runner.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        settings = QSettings(
            str(root / "settings.ini"), QSettings.Format.IniFormat
        )
        window = MainWindow(worker_path=worker, settings=settings)
        if window.quit_button.text() != "Quit" or not window.quit_button.isEnabled():
            raise RuntimeError("Quit button is missing or disabled")
        window.input_dir_edit.setText(str(input_root))
        window.output_dir_edit.setText(str(output_root))
        window.pvrgpu_runner_edit.setText(str(pvrgpu_runner))
        window._start()
        if window.start_button.isEnabled() or not window.cancel_button.isEnabled():
            raise RuntimeError("running controls are not locked correctly")
        _pump_until(
            app,
            lambda: window.process.state() == QProcess.ProcessState.NotRunning,
        )
        if not window.summary_received or window.state_badge.text() != "FAIL":
            raise RuntimeError("non-zero comparison result was treated as a crash")
        if window.results_table.rowCount() != 3:
            raise RuntimeError("result table does not contain every RDC")
        statuses = [
            window.results_table.item(row, 4).text() for row in range(3)
        ]
        if statuses != ["PASS", "FAIL", "PASS"]:
            raise RuntimeError(f"unexpected table results: {statuses}")
        png_statuses = [
            window.results_table.item(row, 3).text() for row in range(3)
        ]
        if png_statuses != ["SKIP", "SKIP", "SKIP"]:
            raise RuntimeError(f"unexpected PNG statuses: {png_statuses}")
        golden_statuses = [
            window.results_table.item(row, 1).text() for row in range(3)
        ]
        if golden_statuses != ["CACHED", "PASS", "PASS"]:
            raise RuntimeError(f"unexpected golden statuses: {golden_statuses}")
        if window.result_filter != "FAIL" or not window.filter_buttons["FAIL"].isChecked():
            raise RuntimeError("default result filter is not Fail")
        fail_visibility = [
            not window.results_table.isRowHidden(row) for row in range(3)
        ]
        if fail_visibility != [False, True, False]:
            raise RuntimeError(f"Fail filter visibility is wrong: {fail_visibility}")
        window.filter_buttons["PASS"].setChecked(True)
        app.processEvents()
        pass_visibility = [
            not window.results_table.isRowHidden(row) for row in range(3)
        ]
        if pass_visibility != [True, False, True]:
            raise RuntimeError(f"Pass filter visibility is wrong: {pass_visibility}")
        window.filter_buttons["ALL"].setChecked(True)
        app.processEvents()
        all_visibility = [
            not window.results_table.isRowHidden(row) for row in range(3)
        ]
        if all_visibility != [True, True, True]:
            raise RuntimeError(f"All filter visibility is wrong: {all_visibility}")
        if window.progress_bar.maximum() != 3 or window.progress_bar.value() != 3:
            raise RuntimeError("progress did not reach 3 / 3")
        if "Total 3 · Pass 2 · Fail 1" != window.summary_label.text():
            raise RuntimeError("summary counts were not rendered")
        if window.report_path is None or window.report_path.name != "report.md":
            raise RuntimeError("final lowercase report.md path was not captured")
        if not window.open_report_button.isEnabled():
            raise RuntimeError("Open report.md was not enabled")
        if "hidden worker diagnostic" not in window.stderr_tail:
            raise RuntimeError("stderr tail was not retained privately")
        if not window.start_button.isEnabled() or window.cancel_button.isEnabled():
            raise RuntimeError("controls were not restored after completion")
        window.close()

        slow_worker = root / "slow_worker.py"
        _write_worker(slow_worker, slow=True)
        cancel_settings = QSettings(
            str(root / "cancel-settings.ini"), QSettings.Format.IniFormat
        )
        cancel_window = MainWindow(worker_path=slow_worker, settings=cancel_settings)
        cancel_window.input_dir_edit.setText(str(input_root))
        cancel_window.output_dir_edit.setText(str(output_root))
        cancel_window.pvrgpu_runner_edit.setText(str(pvrgpu_runner))
        cancel_window._start()
        _pump_until(app, lambda: cancel_window.results_table.rowCount() == 1)
        cancel_window._cancel()
        _pump_until(
            app,
            lambda: cancel_window.process.state() == QProcess.ProcessState.NotRunning,
        )
        if cancel_window.state_badge.text() != "CANCELLED":
            raise RuntimeError("cancelled worker did not reach CANCELLED state")
        if not cancel_window.start_button.isEnabled():
            raise RuntimeError("Start was not restored after cancellation")
        cancel_window.close()

    print("RDC_COUNTER_UI_SMOKE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
