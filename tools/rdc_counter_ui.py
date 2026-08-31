#!/usr/bin/env python3
"""Standalone Qt front end for the recursive RDC counter report worker."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

from PySide6.QtCore import QProcess, QProcessEnvironment, QSettings, QTimer, QUrl, Qt
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


SCHEMA = "pvrgpu.rdc-counter-run.v1"
PROJECT_ROOT = Path(
    os.environ.get("PVRGPU_PROJECT_ROOT", Path(__file__).resolve().parents[1])
).resolve()
WORK_ROOT = Path(
    os.environ.get(
        "PVRGPU_WORK_ROOT",
        str(Path.home() / "Downloads" / "_Codex" / "Working" / "PvrGPU"),
    )
).expanduser().resolve()
BUILD_ROOT = Path(
    os.environ.get("PVRGPU_BUILD_DIR", str(WORK_ROOT / "build"))
).expanduser().resolve()
NATIVE_EXECUTABLE_SUFFIX = ".exe" if os.name == "nt" else ""
DEFAULT_WORKER = PROJECT_ROOT / "tools" / "rdc_counter_report.py"
DEFAULT_LLVMPIPE_RUNNER = BUILD_ROOT / "bin" / f"llvmpipe{NATIVE_EXECUTABLE_SUFFIX}"
DEFAULT_PVRGPU_RUNNER = BUILD_ROOT / "bin" / f"pvrgpu{NATIVE_EXECUTABLE_SUFFIX}"
STDERR_TAIL_LIMIT = 12_000


def _resolved_path(value: str) -> Path:
    path = Path(value.strip()).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _migrate_native_runner_setting(value: object, default: Path) -> str:
    """Discard saved paths that point at the removed shell adapters."""
    saved = str(value or "").strip()
    if not saved:
        return str(default)
    candidate = Path(saved).expanduser()
    legacy_parts = {part.casefold() for part in candidate.parts}
    if candidate.suffix.casefold() in {".sh", ".bash"} or "scripts" in legacy_parts:
        return str(default)
    return saved


class MainWindow(QMainWindow):
    """Small status-only UI; detailed diagnostics remain in worker artifacts."""

    def __init__(
        self,
        *,
        worker_path: str | Path | None = None,
        settings: QSettings | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("PvrGPU RDC Counter Report")
        self.resize(980, 650)

        self.worker_path = Path(worker_path or DEFAULT_WORKER).expanduser().resolve()
        self.settings = settings or QSettings("PvrGPU", "RdcCounterReport")
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        self.process.readyReadStandardOutput.connect(self._read_stdout)
        self.process.readyReadStandardError.connect(self._read_stderr)
        self.process.finished.connect(self._process_finished)
        self.process.errorOccurred.connect(self._process_error)

        self.stdout_buffer = ""
        self.stderr_tail = ""
        self.protocol_errors: list[str] = []
        self.completed_indices: set[int] = set()
        self.input_root: Path | None = None
        self.run_root: Path | None = None
        self.report_path: Path | None = None
        self.total = 0
        self.summary_received = False
        self.summary_status = ""
        self.cancel_requested = False
        self.run_generation = 0
        self._fatal_shown = False
        self.result_filter = "FAIL"

        self._build_ui()
        self._restore_settings()
        self._set_state("READY")

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        header = QHBoxLayout()
        heading = QVBoxLayout()
        title = QLabel("RDC Counter Pass / Fail")
        title.setObjectName("title")
        subtitle = QLabel(
            "Recursive RenderDoc replay · Golden and explicit PvrGPU 17-counter comparison"
        )
        subtitle.setObjectName("subtitle")
        heading.addWidget(title)
        heading.addWidget(subtitle)
        header.addLayout(heading)
        header.addStretch()
        self.state_badge = QLabel("READY")
        self.state_badge.setObjectName("stateBadge")
        header.addWidget(self.state_badge)
        root.addLayout(header)

        run_box = QGroupBox("Run")
        run_grid = QGridLayout(run_box)
        run_grid.setColumnMinimumWidth(0, 92)
        run_grid.setColumnStretch(1, 1)

        self.input_dir_edit = QLineEdit()
        self.input_dir_edit.setPlaceholderText("Directory to scan recursively for *.rdc")
        self.input_browse_button = QPushButton("Browse")
        self.input_browse_button.clicked.connect(self._browse_input)
        run_grid.addWidget(QLabel("Input dir"), 0, 0)
        run_grid.addWidget(self.input_dir_edit, 0, 1)
        run_grid.addWidget(self.input_browse_button, 0, 2)

        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("Artifact and report output root")
        self.output_browse_button = QPushButton("Browse")
        self.output_browse_button.clicked.connect(self._browse_output)
        run_grid.addWidget(QLabel("Output dir"), 1, 0)
        run_grid.addWidget(self.output_dir_edit, 1, 1)
        run_grid.addWidget(self.output_browse_button, 1, 2)

        self.llvmpipe_runner_edit = QLineEdit()
        self.llvmpipe_runner_edit.setPlaceholderText(
            "Single-RDC llvmpipe executable"
        )
        self.llvmpipe_runner_browse_button = QPushButton("Browse")
        self.llvmpipe_runner_browse_button.clicked.connect(
            self._browse_llvmpipe_runner
        )
        run_grid.addWidget(QLabel("llvmpipe runner"), 2, 0)
        run_grid.addWidget(self.llvmpipe_runner_edit, 2, 1)
        run_grid.addWidget(self.llvmpipe_runner_browse_button, 2, 2)

        self.pvrgpu_runner_edit = QLineEdit()
        self.pvrgpu_runner_edit.setPlaceholderText(
            "Single-RDC pvrgpu executable"
        )
        self.pvrgpu_runner_browse_button = QPushButton("Browse")
        self.pvrgpu_runner_browse_button.clicked.connect(self._browse_pvrgpu_runner)
        run_grid.addWidget(QLabel("PvrGPU runner"), 3, 0)
        run_grid.addWidget(self.pvrgpu_runner_edit, 3, 1)
        run_grid.addWidget(self.pvrgpu_runner_browse_button, 3, 2)

        actions = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.start_button.setObjectName("primaryButton")
        self.start_button.clicked.connect(self._start)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel)
        self.open_report_button = QPushButton("Open report.md")
        self.open_report_button.setEnabled(False)
        self.open_report_button.clicked.connect(self._open_report)
        self.quit_button = QPushButton("Quit")
        self.quit_button.clicked.connect(self.close)
        actions.addWidget(self.start_button)
        actions.addWidget(self.cancel_button)
        actions.addWidget(self.open_report_button)
        actions.addStretch()
        actions.addWidget(self.quit_button)
        run_grid.addLayout(actions, 4, 0, 1, 3)
        root.addWidget(run_box)

        progress_row = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("0 / 0")
        self.summary_label = QLabel("No run started")
        self.summary_label.setObjectName("summary")
        progress_row.addWidget(self.progress_bar, 3)
        progress_row.addWidget(self.summary_label, 2)
        progress_row.addStretch()
        progress_row.addWidget(QLabel("Filter"))
        self.filter_button_group = QButtonGroup(self)
        self.filter_buttons: dict[str, QRadioButton] = {}
        for filter_key, label in (("FAIL", "Fail"), ("PASS", "Pass"), ("ALL", "All")):
            button = QRadioButton(label)
            button.toggled.connect(
                lambda checked, key=filter_key: checked
                and self._set_result_filter(key)
            )
            self.filter_button_group.addButton(button)
            self.filter_buttons[filter_key] = button
            progress_row.addWidget(button)
        root.addLayout(progress_row)

        self.results_table = QTableWidget(0, 5)
        self.results_table.setHorizontalHeaderLabels(
            ("RDC", "Golden", "PvrGPU", "PNG", "Result")
        )
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.results_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.results_table.verticalHeader().setVisible(False)
        header_view = self.results_table.horizontalHeader()
        header_view.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in (1, 2, 3, 4):
            header_view.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        root.addWidget(self.results_table, 1)

        # Short aliases make offscreen integration tests less coupled to wording.
        self.input_edit = self.input_dir_edit
        self.output_edit = self.output_dir_edit
        self.llvmpipe_runner_edit_alias = self.llvmpipe_runner_edit
        self.pvrgpu_runner_edit_alias = self.pvrgpu_runner_edit
        self.progress = self.progress_bar
        self.table = self.results_table

        self.filter_buttons[self.result_filter].setChecked(True)
        self._set_style()

    def _restore_settings(self) -> None:
        default_input = os.environ.get("PVRGPU_RDC_ROOT") or str(
            WORK_ROOT.parent / "deqp"
        )
        default_output = os.environ.get(
            "PVRGPU_RDC_COUNTER_OUTPUT",
            str(WORK_ROOT / "out" / "rdc-counter-report"),
        )
        default_pvrgpu_runner = os.environ.get(
            "PVRGPU_RDC_PVRGPU_RUNNER",
            str(DEFAULT_PVRGPU_RUNNER),
        )
        default_llvmpipe_runner = os.environ.get(
            "PVRGPU_RDC_GOLDEN_RUNNER",
            str(DEFAULT_LLVMPIPE_RUNNER),
        )
        self.input_dir_edit.setText(
            str(self.settings.value("input_dir", default_input))
        )
        self.output_dir_edit.setText(
            str(self.settings.value("output_dir", default_output))
        )
        saved_llvmpipe_runner = _migrate_native_runner_setting(
            self.settings.value("llvmpipe_runner", default_llvmpipe_runner),
            Path(default_llvmpipe_runner),
        )
        saved_pvrgpu_runner = _migrate_native_runner_setting(
            self.settings.value("pvrgpu_runner", default_pvrgpu_runner),
            Path(default_pvrgpu_runner),
        )
        self.llvmpipe_runner_edit.setText(saved_llvmpipe_runner)
        self.pvrgpu_runner_edit.setText(saved_pvrgpu_runner)
        self.settings.setValue("llvmpipe_runner", saved_llvmpipe_runner)
        self.settings.setValue("pvrgpu_runner", saved_pvrgpu_runner)
        self.settings.sync()

    def _save_settings(self) -> None:
        self.settings.setValue("input_dir", self.input_dir_edit.text().strip())
        self.settings.setValue("output_dir", self.output_dir_edit.text().strip())
        self.settings.setValue(
            "llvmpipe_runner", self.llvmpipe_runner_edit.text().strip()
        )
        self.settings.setValue(
            "pvrgpu_runner", self.pvrgpu_runner_edit.text().strip()
        )
        self.settings.sync()

    def _browse_input(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Select RDC input directory", self.input_dir_edit.text()
        )
        if path:
            self.input_dir_edit.setText(path)

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Select report output directory", self.output_dir_edit.text()
        )
        if path:
            self.output_dir_edit.setText(path)

    def _browse_pvrgpu_runner(self) -> None:
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Select PvrGPU runner",
            self.pvrgpu_runner_edit.text() or str(BUILD_ROOT / "bin"),
        )
        if path:
            self.pvrgpu_runner_edit.setText(path)

    def _browse_llvmpipe_runner(self) -> None:
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Select llvmpipe runner",
            self.llvmpipe_runner_edit.text() or str(BUILD_ROOT / "bin"),
        )
        if path:
            self.llvmpipe_runner_edit.setText(path)

    def _start(self) -> None:
        if self.process.state() != QProcess.ProcessState.NotRunning:
            return

        if not self.input_dir_edit.text().strip():
            QMessageBox.warning(self, "Input directory required", "Select an input directory.")
            return
        if not self.output_dir_edit.text().strip():
            QMessageBox.warning(self, "Output directory required", "Select an output directory.")
            return
        if not self.llvmpipe_runner_edit.text().strip():
            QMessageBox.warning(
                self,
                "llvmpipe runner required",
                "Select the single-RDC llvmpipe executable used as Golden.",
            )
            return
        if not self.pvrgpu_runner_edit.text().strip():
            QMessageBox.warning(
                self,
                "PvrGPU runner required",
                "Select the single-RDC pvrgpu executable to compare against llvmpipe.",
            )
            return

        input_root = _resolved_path(self.input_dir_edit.text())
        output_root = _resolved_path(self.output_dir_edit.text())
        llvmpipe_runner = _resolved_path(self.llvmpipe_runner_edit.text())
        pvrgpu_runner = _resolved_path(self.pvrgpu_runner_edit.text())
        if not input_root.is_dir():
            QMessageBox.critical(
                self, "Input directory missing", f"Input directory was not found:\n{input_root}"
            )
            return
        if output_root == input_root or input_root in output_root.parents:
            QMessageBox.critical(
                self,
                "Output directory overlaps input",
                "Choose an output directory outside the RDC input directory.",
            )
            return
        if not llvmpipe_runner.is_file():
            QMessageBox.critical(
                self,
                "llvmpipe runner missing",
                f"llvmpipe runner was not found:\n{llvmpipe_runner}",
            )
            return
        if os.name != "nt" and not os.access(llvmpipe_runner, os.X_OK):
            QMessageBox.critical(
                self,
                "llvmpipe runner is not executable",
                f"llvmpipe runner must be executable:\n{llvmpipe_runner}",
            )
            return
        if not pvrgpu_runner.is_file():
            QMessageBox.critical(
                self,
                "PvrGPU runner missing",
                f"PvrGPU runner was not found:\n{pvrgpu_runner}",
            )
            return
        if os.name != "nt" and not os.access(pvrgpu_runner, os.X_OK):
            QMessageBox.critical(
                self,
                "PvrGPU runner is not executable",
                f"PvrGPU runner must be executable:\n{pvrgpu_runner}",
            )
            return
        if not self.worker_path.is_file():
            QMessageBox.critical(
                self, "Worker missing", f"RDC report worker was not found:\n{self.worker_path}"
            )
            return
        try:
            output_root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            QMessageBox.critical(
                self, "Output directory unavailable", f"Could not create {output_root}:\n{exc}"
            )
            return

        self._reset_run_state(input_root)
        self.run_generation += 1
        self._save_settings()
        self._set_running_controls(True)
        self._set_state("STARTING")

        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONDONTWRITEBYTECODE", "1")
        environment.insert("PVRGPU_PROJECT_ROOT", str(PROJECT_ROOT))
        self.process.setProcessEnvironment(environment)
        self.process.setWorkingDirectory(str(PROJECT_ROOT))

        worker_arguments = [
            "--rdc-dir",
            str(input_root),
            "--output-root",
            str(output_root),
            "--golden-runner",
            str(llvmpipe_runner),
            "--pvrgpu-runner",
            str(pvrgpu_runner),
            "--json",
        ]
        program = sys.executable
        arguments = [str(self.worker_path), *worker_arguments]

        self.process.start(program, arguments)
        if not self.process.waitForStarted(3000):
            message = self.process.errorString() or "The worker could not be started."
            self._set_running_controls(False)
            self._set_state("FAILED")
            self._show_fatal("Launch failed", message)
            return
        self._set_state("RUNNING")

    def _reset_run_state(self, input_root: Path) -> None:
        self.stdout_buffer = ""
        self.stderr_tail = ""
        self.protocol_errors.clear()
        self.completed_indices.clear()
        self.input_root = input_root
        self.run_root = None
        self.report_path = None
        self.total = 0
        self.summary_received = False
        self.summary_status = ""
        self.cancel_requested = False
        self._fatal_shown = False
        self.results_table.setRowCount(0)
        self._apply_result_filter()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("0 / 0")
        self.summary_label.setText("Scanning for .rdc files…")
        self.open_report_button.setEnabled(False)

    def _set_running_controls(self, running: bool) -> None:
        self.input_dir_edit.setEnabled(not running)
        self.input_browse_button.setEnabled(not running)
        self.output_dir_edit.setEnabled(not running)
        self.output_browse_button.setEnabled(not running)
        self.llvmpipe_runner_edit.setEnabled(not running)
        self.llvmpipe_runner_browse_button.setEnabled(not running)
        self.pvrgpu_runner_edit.setEnabled(not running)
        self.pvrgpu_runner_browse_button.setEnabled(not running)
        self.start_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)

    def _cancel(self) -> None:
        if self.process.state() == QProcess.ProcessState.NotRunning:
            return
        self.cancel_requested = True
        self.cancel_button.setEnabled(False)
        self._set_state("CANCELLING")
        self.summary_label.setText("Cancelling current replay…")
        self.process.terminate()
        generation = self.run_generation
        QTimer.singleShot(2000, lambda: self._kill_if_running(generation))

    def _kill_if_running(self, generation: int) -> None:
        if (
            generation == self.run_generation
            and self.process.state() != QProcess.ProcessState.NotRunning
        ):
            self.process.kill()

    def _read_stdout(self) -> None:
        chunk = bytes(self.process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        self.stdout_buffer += chunk
        while "\n" in self.stdout_buffer:
            line, self.stdout_buffer = self.stdout_buffer.split("\n", 1)
            self._handle_protocol_line(line.rstrip("\r"))

    def _read_stderr(self) -> None:
        chunk = bytes(self.process.readAllStandardError()).decode(
            "utf-8", errors="replace"
        )
        if chunk:
            self.stderr_tail = (self.stderr_tail + chunk)[-STDERR_TAIL_LIMIT:]

    def _handle_protocol_line(self, line: str) -> None:
        if not line.strip():
            return
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            self.protocol_errors.append(f"Invalid JSONL event: {exc}")
            return
        if not isinstance(event, dict):
            self.protocol_errors.append("JSONL event must be an object")
            return
        if event.get("schema") != SCHEMA:
            self.protocol_errors.append(
                f"Unexpected event schema: {event.get('schema')!r}"
            )
            return
        try:
            self._handle_event(event)
        except (KeyError, TypeError, ValueError) as exc:
            self.protocol_errors.append(
                f"Invalid {event.get('type', 'unknown')} event: {exc}"
            )

    def _handle_event(self, event: dict[str, Any]) -> None:
        event_type = str(event["type"])
        if event_type == "run_started":
            self.input_root = Path(str(event["input_root"]))
            self.run_root = Path(str(event["run_root"]))
            self.summary_label.setText("Scanning for .rdc files…")
            return

        if event_type == "scan_complete":
            total = int(event["total"])
            if total < 0:
                raise ValueError("total must be non-negative")
            self.total = total
            self.results_table.setRowCount(total)
            self._apply_result_filter()
            self.progress_bar.setRange(0, max(1, total))
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat(f"%v / {total}")
            self.summary_label.setText(f"Found {total} RDC file{'s' if total != 1 else ''}")
            return

        if event_type == "rdc_started":
            index = self._event_index(event)
            row = self._ensure_row(index)
            rdc = str(event["rdc"])
            case = str(event.get("case", ""))
            self._set_cell(row, 0, rdc)
            if case:
                self.results_table.item(row, 0).setToolTip(f"case={case}\n{rdc}")
            self._set_cell(row, 1, "PENDING", "PENDING")
            self._set_cell(row, 2, "PENDING", "PENDING")
            self._set_cell(row, 3, "PENDING", "PENDING")
            self._set_cell(row, 4, "PENDING", "PENDING")
            self.summary_label.setText(f"Running {index} of {self.total or event.get('total', '?')}")
            return

        if event_type == "stage":
            index = self._event_index(event)
            row = self._ensure_row(index)
            stage = str(event["stage"])
            status = str(event["status"])
            columns = {"golden": 1, "pvrgpu": 2, "png": 3, "compare": 4}
            if stage not in columns:
                raise ValueError(f"unsupported stage {stage!r}")
            if status not in {"RUNNING", "PASS", "FAIL", "SKIP", "CACHED"}:
                raise ValueError(f"unsupported stage status {status!r}")
            self._set_cell(row, columns[stage], status, status)
            return

        if event_type == "rdc_result":
            index = self._event_index(event)
            row = self._ensure_row(index)
            rdc = str(event["rdc"])
            case = str(event.get("case", ""))
            status = str(event["status"])
            if status not in {"PASS", "FAIL"}:
                raise ValueError(f"unsupported result status {status!r}")
            self._set_cell(row, 0, rdc)
            if case:
                self.results_table.item(row, 0).setToolTip(f"case={case}\n{rdc}")
            self._set_cell(row, 4, status, status)
            reason = str(event.get("reason", "")).strip()
            if reason:
                self.results_table.item(row, 4).setToolTip(reason)
            for field, column in (("golden", 1), ("pvrgpu", 2), ("png", 3)):
                stage_status = str(event.get(field, ""))
                if stage_status in {"PASS", "FAIL", "SKIP", "CACHED"}:
                    self._set_cell(row, column, stage_status, stage_status)
            self._set_cell(row, 4, status, status)
            if reason:
                self.results_table.item(row, 4).setToolTip(reason)
            self.completed_indices.add(index)
            self.progress_bar.setValue(min(len(self.completed_indices), max(1, self.total)))
            return

        if event_type == "report_written":
            self._set_report_path(event["path"])
            return

        if event_type == "summary":
            total = int(event["total"])
            passed = int(event["passed"])
            failed = int(event["failed"])
            status = str(event["status"])
            if status not in {"PASS", "FAIL", "CANCELLED"}:
                raise ValueError(f"unsupported summary status {status!r}")
            self.total = total
            self.summary_received = True
            self.summary_status = status
            report = str(event.get("report", "")).strip()
            if report:
                self._set_report_path(report)
            self.progress_bar.setRange(0, max(1, total))
            self.progress_bar.setValue(min(passed + failed, max(1, total)))
            self.progress_bar.setFormat(f"%v / {total}")
            self.summary_label.setText(
                f"Total {total} · Pass {passed} · Fail {failed}"
            )
            self._set_state(status)
            return

        if event_type == "fatal":
            message = str(event.get("message", "Worker setup failed"))
            self.protocol_errors.append(message)
            self.summary_label.setText("Worker setup failed")
            return

        raise ValueError(f"unsupported event type {event_type!r}")

    def _event_index(self, event: dict[str, Any]) -> int:
        index = int(event["index"])
        if index < 1:
            raise ValueError("index must be one-based")
        return index

    def _ensure_row(self, index: int) -> int:
        row = index - 1
        if row >= self.results_table.rowCount():
            previous_count = self.results_table.rowCount()
            self.results_table.setRowCount(row + 1)
            for new_row in range(previous_count, row + 1):
                self._update_row_visibility(new_row)
        return row

    def _set_cell(self, row: int, column: int, text: str, status: str = "") -> None:
        item = self.results_table.item(row, column)
        if item is None:
            item = QTableWidgetItem()
            self.results_table.setItem(row, column, item)
        item.setText(text)
        colors = {
            "RUNNING": QColor("#1d4ed8"),
            "PASS": QColor("#166534"),
            "FAIL": QColor("#b91c1c"),
            "CACHED": QColor("#9333ea"),
            "SKIP": QColor("#64748b"),
            "PENDING": QColor("#64748b"),
        }
        if status in colors:
            item.setForeground(colors[status])
        if column > 0:
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        if column == 4:
            self._update_row_visibility(row)

    def _set_result_filter(self, result_filter: str) -> None:
        if result_filter not in {"FAIL", "PASS", "ALL"}:
            raise ValueError(f"unsupported result filter {result_filter!r}")
        self.result_filter = result_filter
        self._apply_result_filter()

    def _apply_result_filter(self) -> None:
        for row in range(self.results_table.rowCount()):
            self._update_row_visibility(row)

    def _update_row_visibility(self, row: int) -> None:
        if row < 0 or row >= self.results_table.rowCount():
            return
        result_item = self.results_table.item(row, 4)
        result = result_item.text() if result_item is not None else ""
        visible = self.result_filter == "ALL" or result == self.result_filter
        self.results_table.setRowHidden(row, not visible)

    def _set_report_path(self, value: object) -> None:
        path = Path(str(value)).expanduser()
        self.report_path = path if path.is_absolute() else (PROJECT_ROOT / path).resolve()
        self.open_report_button.setEnabled(self.report_path.is_file())
        self.open_report_button.setToolTip(str(self.report_path))

    def _process_finished(self, exit_code: int, _exit_status: object) -> None:
        self._read_stdout()
        self._read_stderr()
        if self.stdout_buffer.strip():
            self._handle_protocol_line(self.stdout_buffer.rstrip("\r"))
        self.stdout_buffer = ""
        self._set_running_controls(False)

        if self.summary_received:
            # A comparison failure intentionally uses a non-zero exit code. The
            # structured summary, not the process code, is authoritative.
            self._set_state(self.summary_status)
            return
        if self.cancel_requested:
            self.summary_status = "CANCELLED"
            self._set_state("CANCELLED")
            self.summary_label.setText("Run cancelled")
            return

        self._set_state("FAILED")
        detail_parts = [f"Worker exited with code {exit_code} without a summary event."]
        if self.protocol_errors:
            detail_parts.append(self.protocol_errors[-1])
        if self.stderr_tail.strip():
            detail_parts.append(self.stderr_tail.strip()[-3000:])
        self.summary_label.setText("Worker failed before writing a summary")
        self._show_fatal("RDC counter worker failed", "\n\n".join(detail_parts))

    def _process_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self._set_running_controls(False)
            self._set_state("FAILED")
            self._show_fatal("Launch failed", self.process.errorString())

    def _show_fatal(self, title: str, message: str) -> None:
        if self._fatal_shown:
            return
        self._fatal_shown = True
        QMessageBox.critical(self, title, message)

    def _open_report(self) -> None:
        if self.report_path and self.report_path.is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.report_path)))
            return
        QMessageBox.warning(self, "Report unavailable", "report.md has not been written yet.")

    def _set_state(self, state: str) -> None:
        self.state_badge.setText(state)
        self.state_badge.setProperty("state", state.lower())
        self.state_badge.style().unpolish(self.state_badge)
        self.state_badge.style().polish(self.state_badge)

    def closeEvent(self, event: object) -> None:
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.cancel_requested = True
            self.process.terminate()
            if not self.process.waitForFinished(2000):
                self.process.kill()
                self.process.waitForFinished(1000)
        self._save_settings()
        super().closeEvent(event)  # type: ignore[arg-type]

    def _set_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #f4f6f8; color: #172033; }
            QLabel#title { font-size: 24px; font-weight: 700; }
            QLabel#subtitle, QLabel#summary { color: #5b667a; }
            QLabel#stateBadge {
                background: #e0e7ff; color: #3730a3; border-radius: 10px;
                padding: 5px 10px; font-weight: 700;
            }
            QLabel#stateBadge[state="running"], QLabel#stateBadge[state="starting"] {
                background: #dbeafe; color: #1d4ed8;
            }
            QLabel#stateBadge[state="pass"] { background: #dcfce7; color: #166534; }
            QLabel#stateBadge[state="fail"], QLabel#stateBadge[state="failed"] {
                background: #fee2e2; color: #991b1b;
            }
            QLabel#stateBadge[state="cancelled"], QLabel#stateBadge[state="cancelling"] {
                background: #fef3c7; color: #92400e;
            }
            QGroupBox {
                background: white; border: 1px solid #d9dee7; border-radius: 8px;
                margin-top: 10px; padding-top: 12px; font-weight: 600;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; }
            QLineEdit { background: white; border: 1px solid #cbd5e1; border-radius: 4px; padding: 6px; }
            QPushButton { background: white; border: 1px solid #cbd5e1; border-radius: 5px; padding: 7px 13px; }
            QPushButton:hover { background: #eef2ff; }
            QPushButton#primaryButton { background: #4f46e5; color: white; border: none; font-weight: 700; }
            QPushButton:disabled { color: #9ca3af; background: #e5e7eb; }
            QProgressBar { background: white; border: 1px solid #cbd5e1; border-radius: 5px; text-align: center; }
            QProgressBar::chunk { background: #4f46e5; border-radius: 4px; }
            QTableWidget { background: white; alternate-background-color: #f8fafc; gridline-color: #e2e8f0; }
            """
        )


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("PvrGPU RDC Counter Report")
    window = MainWindow()
    window.show()
    if "--smoke-test" in sys.argv:
        QTimer.singleShot(150, app.quit)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
