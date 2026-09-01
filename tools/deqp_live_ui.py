#!/usr/bin/env python3
"""Desktop front end for the directly linked PvrGPU live dEQP runner."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import json
import re
import shlex
import sys
from typing import Callable


def _ensure_ui_python() -> None:
    """Re-exec direct launches with the configured PySide6 interpreter."""
    try:
        import PySide6  # noqa: F401
    except ModuleNotFoundError as error:
        if error.name != "PySide6":
            raise
    else:
        return

    work_root = Path(
        os.environ.get(
            "PVRGPU_WORK_ROOT",
            str(Path.home() / "Downloads" / "_Codex" / "Working" / "PvrGPU"),
        )
    ).expanduser()
    configured_python = os.environ.get("PVRGPU_UI_PYTHON", "").strip()
    configured_venv = Path(
        os.environ.get("PVRGPU_UI_VENV", str(work_root / "venv"))
    ).expanduser()
    candidates = [
        Path(configured_python).expanduser() if configured_python else None,
        configured_venv / "bin" / "python",
    ]
    current_python = Path(os.path.abspath(sys.executable))
    for candidate in candidates:
        if candidate is None or not candidate.is_file() or not os.access(candidate, os.X_OK):
            continue
        candidate = Path(os.path.abspath(candidate))
        if candidate == current_python:
            continue
        os.execv(
            str(candidate),
            [str(candidate), str(Path(__file__).resolve()), *sys.argv[1:]],
        )

    raise SystemExit(
        "PySide6 is not installed for the current python3 and no usable UI "
        "interpreter was found. Set PVRGPU_UI_PYTHON or install "
        "requirements-ui.txt into the project venv."
    )


if __name__ == "__main__":
    _ensure_ui_python()


from PySide6.QtCore import QProcess, QProcessEnvironment, QSettings, QTimer, QUrl, Qt
from PySide6.QtGui import QColor, QDesktopServices, QFontDatabase, QPixmap, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from deqp_groups import GROUP_SPECS, exact_case_belongs_to_group, get_group


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
DEFAULT_RUNNER = BUILD_ROOT / "bin" / f"pvrgpu-deqp{NATIVE_EXECUTABLE_SUFFIX}"
DEFAULT_OUTPUT_ROOT = WORK_ROOT / "out" / "deqp-live-ui"
GROUP_RUNNER = Path(__file__).resolve().with_name("deqp_group_runner.py")
RESULT_ROW_LIMIT = 2_000
ARTIFACT_ROW_LIMIT = 5_000
EVENT_PREFIX = "PVRGPU_BATCH "

SUITE_CASES: dict[str, tuple[tuple[str, str], ...]] = {
    "dEQP-GLES2": (
        (
            "Verified · Clear color smoke",
            "dEQP-GLES2.functional.prerequisite.clear_color",
        ),
        ("Info · Vendor", "dEQP-GLES2.info.vendor"),
        ("Info · Renderer", "dEQP-GLES2.info.renderer"),
        ("Info · Version", "dEQP-GLES2.info.version"),
        ("Color clear · Single RGB", "dEQP-GLES2.functional.color_clear.single_rgb"),
    ),
    "dEQP-GLES3": (
        ("Info · Vendor", "dEQP-GLES3.info.vendor"),
        ("Info · Renderer", "dEQP-GLES3.info.renderer"),
        ("Info · Version", "dEQP-GLES3.info.version"),
        ("Custom GLES3 case…", "dEQP-GLES3."),
    ),
    "dEQP-GLES31": (
        ("Info · Vendor", "dEQP-GLES31.info.vendor"),
        ("Info · Renderer", "dEQP-GLES31.info.renderer"),
        ("Info · Version", "dEQP-GLES31.info.version"),
        ("Custom GLES31 case…", "dEQP-GLES31."),
    ),
    "dEQP-EGL": (
        ("Info · Version", "dEQP-EGL.info.version"),
        ("Info · Vendor", "dEQP-EGL.info.vendor"),
        ("Info · Client APIs", "dEQP-EGL.info.client_apis"),
        ("Info · Extensions", "dEQP-EGL.info.extensions"),
        ("Info · Configs", "dEQP-EGL.info.configs"),
    ),
}

UNAVAILABLE_SUITES = {
    "dEQP-GLES3": (
        "The package is compiled into the runner, but every current PvrGPU EGL "
        "config is ES2-only. Missing ES3 format/query/restart prerequisites keep "
        "EGL_OPENGL_ES3_BIT_KHR disabled."
    ),
    "dEQP-GLES31": (
        "The package is compiled into the runner, but the EGL config has no ES3 "
        "bit. GLES31 also requires compute, SSBO, shader-image, and atomic support "
        "that the current PvrGPU driver does not implement."
    ),
}

PASS_STATUSES = {"pass"}
SKIP_STATUSES = {"notsupported", "waiver"}
WARNING_STATUSES = {"qualitywarning", "compatibilitywarning"}
NON_FAILURE_STATUSES = PASS_STATUSES | SKIP_STATUSES | WARNING_STATUSES
EXACT_CASE_RE = re.compile(r"^dEQP-(?:EGL|GLES2|GLES3|GLES31)\.[A-Za-z0-9_.-]+$")

CASE_RESULT_RE = re.compile(
    r"#beginTestCaseResult\s+([^\r\n]+)(.*?)#endTestCaseResult",
    re.DOTALL,
)
RESULT_RE = re.compile(r'<Result\s+StatusCode="([^"]+)">([^<]*)</Result>')
DURATION_RE = re.compile(r'<Number\s+Name="TestDuration"[^>]*>(\d+)</Number>')
SESSION_RE_TEMPLATE = r'^#sessionInfo\s+{name}\s+"([^"]*)"'
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


@dataclass(frozen=True)
class QpaCaseResult:
    case_name: str
    status: str
    message: str
    duration_us: int | None


@dataclass(frozen=True)
class QpaSummary:
    vendor: str
    renderer: str
    cases: tuple[QpaCaseResult, ...]
    incomplete_count: int = 0


def safe_case_name(case_name: str) -> str:
    """Mirror the native runner's per-case artifact directory mapping."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", case_name) or "unnamed"


def parse_qpa(path: Path) -> QpaSummary:
    if not path.is_file():
        return QpaSummary("", "", (), 0)
    text = path.read_text(encoding="utf-8", errors="replace")

    def session_value(name: str) -> str:
        match = re.search(
            SESSION_RE_TEMPLATE.format(name=re.escape(name)),
            text,
            re.MULTILINE,
        )
        return match.group(1) if match else ""

    cases: list[QpaCaseResult] = []
    for case_match in CASE_RESULT_RE.finditer(text):
        case_name = case_match.group(1).strip()
        body = case_match.group(2)
        result = RESULT_RE.search(body)
        status = result.group(1).strip() if result else "Incomplete"
        message = result.group(2).strip() if result else "No terminal result"
        duration_match = DURATION_RE.search(body)
        duration_us = int(duration_match.group(1)) if duration_match else None
        cases.append(QpaCaseResult(case_name, status, message, duration_us))
    begin_count = len(re.findall(r"^#beginTestCaseResult\b", text, re.MULTILINE))
    incomplete_count = max(0, begin_count - len(cases))
    return QpaSummary(
        session_value("vendor"),
        session_value("renderer"),
        tuple(cases),
        incomplete_count,
    )


def format_bytes(size: int) -> str:
    value = float(size)
    for suffix in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or suffix == "GiB":
            return f"{value:.0f} {suffix}" if suffix == "B" else f"{value:.1f} {suffix}"
        value /= 1024.0
    return f"{size} B"


def has_complete_systemc(case_dir: Path) -> bool:
    jsonl = case_dir / "systemc.jsonl"
    if not jsonl.is_file() or not any((case_dir / "systemc").glob("*.png")):
        return False
    try:
        return '"type":"done"' in jsonl.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


def expects_gpu_artifacts(case_name: str) -> bool:
    """Return whether a case is expected to submit a PvrGPU command."""
    return case_name.startswith("dEQP-GLES") and ".info." not in case_name


class MainWindow(QMainWindow):
    """Run one live dEQP selection and summarize every produced layer."""

    def __init__(
        self,
        *,
        runner_path: str | Path | None = None,
        settings: QSettings | None = None,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("PvrGPU Live dEQP")
        self.resize(1120, 790)

        self.default_runner = Path(runner_path or DEFAULT_RUNNER).expanduser().resolve()
        self.settings = settings or QSettings("PvrGPU", "LiveDeqp")
        self.now_provider = now_provider or datetime.now
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_process_output)
        self.process.finished.connect(self._process_finished)
        self.process.errorOccurred.connect(self._process_error)

        self.run_dir: Path | None = None
        self.qpa_summary = QpaSummary("", "", ())
        self.cancel_requested = False
        self.process_error_seen = False
        self.last_command: list[str] = []
        self.latest_png: Path | None = None
        self.active_mode = "exact"
        self.active_group_id = ""
        self.process_line_buffer = ""
        self.batch_counts = {"passed": 0, "skipped": 0, "warnings": 0, "failed": 0}

        self._build_ui()
        self._restore_settings()
        self._suite_changed()
        self._set_state("READY")
        self._update_case_note()
        self._update_command_preview()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(11)

        header = QHBoxLayout()
        heading = QVBoxLayout()
        title = QLabel("PvrGPU Live dEQP")
        title.setObjectName("title")
        subtitle = QLabel("dEQP → Mesa EGL/GLES → PvrGPU Gallium → SystemC")
        subtitle.setObjectName("subtitle")
        heading.addWidget(title)
        heading.addWidget(subtitle)
        header.addLayout(heading)
        header.addStretch()
        self.state_badge = QLabel("READY")
        self.state_badge.setObjectName("stateBadge")
        header.addWidget(self.state_badge)
        root.addLayout(header)

        run_box = QGroupBox("Test selection")
        run_grid = QGridLayout(run_box)
        run_grid.setColumnMinimumWidth(0, 96)
        run_grid.setColumnStretch(1, 1)
        run_grid.setColumnMinimumWidth(2, 96)
        run_grid.setColumnStretch(3, 1)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Exact case", "exact")
        self.mode_combo.addItem("24-group batch", "group")
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        run_grid.addWidget(QLabel("Run mode"), 0, 0)
        run_grid.addWidget(self.mode_combo, 0, 1)

        self.group_combo = QComboBox()
        for group in GROUP_SPECS:
            state = " · blocked" if not group.available else ""
            self.group_combo.addItem(
                f"{group.label} · {group.locked_case_count:,} cases{state}",
                group.id,
            )
            tooltip = "\n".join(group.selectors)
            if group.availability_reason:
                tooltip += "\n\nBlocked: " + group.availability_reason
            self.group_combo.setItemData(
                self.group_combo.count() - 1,
                tooltip,
                Qt.ItemDataRole.ToolTipRole,
            )
        self.group_combo.currentIndexChanged.connect(self._group_changed)
        run_grid.addWidget(QLabel("Test group"), 0, 2)
        run_grid.addWidget(self.group_combo, 0, 3)

        self.suite_combo = QComboBox()
        self.suite_combo.addItem("OpenGL ES 2.0", "dEQP-GLES2")
        self.suite_combo.addItem("OpenGL ES 3.0 · unavailable", "dEQP-GLES3")
        self.suite_combo.addItem("OpenGL ES 3.1 · unavailable", "dEQP-GLES31")
        self.suite_combo.addItem("EGL", "dEQP-EGL")
        suite_model = self.suite_combo.model()
        for suite, reason in UNAVAILABLE_SUITES.items():
            index = self.suite_combo.findData(suite)
            item = suite_model.item(index) if index >= 0 and hasattr(suite_model, "item") else None
            if item is not None:
                item.setEnabled(False)
                item.setToolTip(reason)
        self.suite_combo.currentIndexChanged.connect(self._suite_changed)
        run_grid.addWidget(QLabel("Suite"), 1, 0)
        run_grid.addWidget(self.suite_combo, 1, 1)

        self.case_combo = QComboBox()
        self.case_combo.setEditable(True)
        self.case_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.case_combo.currentTextChanged.connect(self._case_changed)
        self.case_combo.activated.connect(self._case_preset_activated)
        run_grid.addWidget(QLabel("Exact case"), 1, 2)
        run_grid.addWidget(self.case_combo, 1, 3)

        self.gl_config_combo = QComboBox()
        self.gl_config_combo.addItem("RGBA8 · Depth24 · Stencil8 (verified)", "rgba8888d24s8ms0")
        self.gl_config_combo.addItem("RGBA8 · no depth/stencil", "rgba8888d0s0ms0")
        self.gl_config_combo.addItem("Automatic", "")
        self.gl_config_combo.currentIndexChanged.connect(self._update_command_preview)
        run_grid.addWidget(QLabel("GL config"), 2, 0)
        run_grid.addWidget(self.gl_config_combo, 2, 1)

        self.surface_combo = QComboBox()
        self.surface_combo.addItem("Pbuffer (recommended)", "pbuffer")
        self.surface_combo.addItem("FBO (experimental)", "fbo")
        self.surface_combo.addItem("Window (experimental)", "window")
        self.surface_combo.currentIndexChanged.connect(self._update_command_preview)
        self.surface_combo.currentIndexChanged.connect(self._update_case_note)
        run_grid.addWidget(QLabel("Surface"), 2, 2)
        run_grid.addWidget(self.surface_combo, 2, 3)

        self.watchdog_combo = QComboBox()
        self.watchdog_combo.addItem("Disabled (debug friendly)", "disable")
        self.watchdog_combo.addItem("Enabled", "enable")
        self.watchdog_combo.currentIndexChanged.connect(self._update_command_preview)
        run_grid.addWidget(QLabel("Watchdog"), 3, 0)
        run_grid.addWidget(self.watchdog_combo, 3, 1)

        option_row = QHBoxLayout()
        self.log_images_check = QCheckBox("Store dEQP log images")
        self.log_images_check.setChecked(False)
        self.log_images_check.toggled.connect(self._update_command_preview)
        option_row.addWidget(self.log_images_check)
        option_row.addStretch()
        run_grid.addWidget(QLabel("Logging"), 3, 2)
        run_grid.addLayout(option_row, 3, 3)

        self.case_note = QLabel()
        self.case_note.setObjectName("caseNote")
        self.case_note.setWordWrap(True)
        run_grid.addWidget(self.case_note, 4, 0, 1, 4)

        self.batch_progress = QProgressBar()
        self.batch_progress.setRange(0, 1)
        self.batch_progress.setValue(0)
        self.batch_progress.setFormat("Waiting for group discovery")
        self.batch_progress.setVisible(False)
        run_grid.addWidget(self.batch_progress, 5, 0, 1, 4)
        root.addWidget(run_box)

        paths_box = QGroupBox("Runtime and output")
        paths_grid = QGridLayout(paths_box)
        paths_grid.setColumnMinimumWidth(0, 96)
        paths_grid.setColumnStretch(1, 1)

        self.runner_edit = QLineEdit()
        self.runner_browse = QPushButton("Browse")
        self.runner_browse.clicked.connect(self._browse_runner)
        paths_grid.addWidget(QLabel("Runner"), 0, 0)
        paths_grid.addWidget(self.runner_edit, 0, 1)
        paths_grid.addWidget(self.runner_browse, 0, 2)

        self.output_edit = QLineEdit()
        self.output_browse = QPushButton("Browse")
        self.output_browse.clicked.connect(self._browse_output)
        paths_grid.addWidget(QLabel("Output root"), 1, 0)
        paths_grid.addWidget(self.output_edit, 1, 1)
        paths_grid.addWidget(self.output_browse, 1, 2)

        self.bridge_edit = QLineEdit()
        self.bridge_edit.setPlaceholderText("Optional override; leave empty to use linked bridge")
        self.bridge_browse = QPushButton("Browse")
        self.bridge_browse.clicked.connect(self._browse_bridge)
        paths_grid.addWidget(QLabel("SystemC bridge"), 2, 0)
        paths_grid.addWidget(self.bridge_edit, 2, 1)
        paths_grid.addWidget(self.bridge_browse, 2, 2)

        self.command_preview = QLineEdit()
        self.command_preview.setReadOnly(True)
        fixed_font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        self.command_preview.setFont(fixed_font)
        paths_grid.addWidget(QLabel("Command"), 3, 0)
        paths_grid.addWidget(self.command_preview, 3, 1, 1, 2)

        actions = QHBoxLayout()
        self.start_button = QPushButton("Run dEQP")
        self.start_button.setObjectName("primaryButton")
        self.start_button.clicked.connect(self._start)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop)
        self.open_output_button = QPushButton("Open Output")
        self.open_output_button.setEnabled(False)
        self.open_output_button.clicked.connect(self._open_output)
        self.open_qpa_button = QPushButton("Open results.qpa")
        self.open_qpa_button.setEnabled(False)
        self.open_qpa_button.clicked.connect(self._open_qpa)
        self.quit_button = QPushButton("Quit")
        self.quit_button.clicked.connect(self.close)
        actions.addWidget(self.start_button)
        actions.addWidget(self.stop_button)
        actions.addWidget(self.open_output_button)
        actions.addWidget(self.open_qpa_button)
        actions.addStretch()
        actions.addWidget(self.quit_button)
        paths_grid.addLayout(actions, 4, 0, 1, 3)
        root.addWidget(paths_box)

        cards = QHBoxLayout()
        self.deqp_value = self._card(cards, "dEQP", "—")
        self.driver_value = self._card(cards, "Driver artifacts", "—")
        self.systemc_value = self._card(cards, "SystemC", "—")
        self.renderer_value = self._card(cards, "Renderer", "—")
        root.addLayout(cards)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)

        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)
        self.summary_label = QLabel("Choose a case and run it through PvrGPU")
        self.summary_label.setObjectName("summary")
        results_layout.addWidget(self.summary_label)
        self.results_table = QTableWidget(0, 5)
        self.results_table.setHorizontalHeaderLabels(
            ("Case", "dEQP", "Driver", "SystemC", "Duration")
        )
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.results_table.verticalHeader().setVisible(False)
        results_header = self.results_table.horizontalHeader()
        results_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in (1, 2, 3, 4):
            results_header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        results_layout.addWidget(self.results_table)
        self.tabs.addTab(results_tab, "Results")

        artifacts_tab = QWidget()
        artifacts_layout = QVBoxLayout(artifacts_tab)
        artifacts_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.artifacts_table = QTableWidget(0, 3)
        self.artifacts_table.setHorizontalHeaderLabels(("Artifact", "Type", "Size"))
        self.artifacts_table.setAlternatingRowColors(True)
        self.artifacts_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.artifacts_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.artifacts_table.verticalHeader().setVisible(False)
        self.artifacts_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.artifacts_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.artifacts_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.artifacts_table.itemSelectionChanged.connect(self._artifact_selected)
        self.artifacts_table.itemDoubleClicked.connect(self._open_artifact_item)
        artifacts_splitter.addWidget(self.artifacts_table)
        preview_frame = QFrame()
        preview_frame.setObjectName("previewFrame")
        preview_layout = QVBoxLayout(preview_frame)
        self.image_preview = QLabel("Select a PNG artifact to preview")
        self.image_preview.setObjectName("imagePreview")
        self.image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_preview.setMinimumSize(300, 220)
        self.image_preview.setWordWrap(True)
        preview_layout.addWidget(self.image_preview)
        artifacts_splitter.addWidget(preview_frame)
        artifacts_splitter.setSizes((650, 390))
        artifacts_layout.addWidget(artifacts_splitter)
        self.tabs.addTab(artifacts_tab, "Artifacts")

        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(12_000)
        self.log.setFont(fixed_font)
        log_layout.addWidget(self.log)
        self.tabs.addTab(log_tab, "Live log")

        # Stable aliases used by the offscreen integration smoke.
        self.case_edit = self.case_combo.lineEdit()
        self.output_root_edit = self.output_edit
        self.table = self.results_table

        self._config_widgets = (
            self.mode_combo,
            self.group_combo,
            self.suite_combo,
            self.case_combo,
            self.gl_config_combo,
            self.surface_combo,
            self.watchdog_combo,
            self.log_images_check,
            self.runner_edit,
            self.runner_browse,
            self.output_edit,
            self.output_browse,
            self.bridge_edit,
            self.bridge_browse,
        )
        self._set_style()

    def _card(self, row: QHBoxLayout, title: str, value: str) -> QLabel:
        frame = QFrame()
        frame.setObjectName("metricCard")
        layout = QVBoxLayout(frame)
        title_label = QLabel(title)
        title_label.setObjectName("metricTitle")
        value_label = QLabel(value)
        value_label.setObjectName("metricValue")
        value_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        value_label.setWordWrap(True)
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        row.addWidget(frame, 1)
        return value_label

    def _restore_settings(self) -> None:
        default_runner = os.environ.get("PVRGPU_DEQP_RUNNER", str(self.default_runner))
        default_output = os.environ.get(
            "PVRGPU_DEQP_UI_OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT)
        )
        self.runner_edit.setText(str(self.settings.value("runner", default_runner)))
        self.output_edit.setText(str(self.settings.value("output_root", default_output)))
        self.bridge_edit.setText(
            str(self.settings.value("systemc_bridge", os.environ.get("PVRGPU_SYSTEMC_API_LIB", "")))
        )
        suite = str(self.settings.value("suite", "dEQP-GLES2"))
        if suite in UNAVAILABLE_SUITES:
            suite = "dEQP-GLES2"
        suite_index = self.suite_combo.findData(suite)
        if suite_index >= 0:
            self.suite_combo.setCurrentIndex(suite_index)
        self._populate_cases(suite)
        saved_case = str(
            self.settings.value(
                "case", "dEQP-GLES2.functional.prerequisite.clear_color"
            )
        )
        self.case_combo.setCurrentText(saved_case)

        for widget, key, default in (
            (self.gl_config_combo, "gl_config", "rgba8888d24s8ms0"),
            (self.surface_combo, "surface", "pbuffer"),
            (self.watchdog_combo, "watchdog", "disable"),
        ):
            index = widget.findData(str(self.settings.value(key, default)))
            if index >= 0:
                widget.setCurrentIndex(index)
        log_images = str(self.settings.value("log_images", "false")).lower()
        self.log_images_check.setChecked(log_images in {"1", "true", "yes"})
        group_id = str(self.settings.value("group", GROUP_SPECS[0].id))
        group_index = self.group_combo.findData(group_id)
        if group_index >= 0:
            self.group_combo.setCurrentIndex(group_index)
        mode = str(self.settings.value("mode", "group"))
        mode_index = self.mode_combo.findData(mode)
        if mode_index >= 0:
            self.mode_combo.setCurrentIndex(mode_index)
        self._mode_changed()

    def _save_settings(self) -> None:
        self.settings.setValue("runner", self.runner_edit.text().strip())
        self.settings.setValue("output_root", self.output_edit.text().strip())
        self.settings.setValue("systemc_bridge", self.bridge_edit.text().strip())
        self.settings.setValue("suite", self.suite_combo.currentData())
        self.settings.setValue("case", self.case_combo.currentText().strip())
        self.settings.setValue("mode", self.mode_combo.currentData())
        self.settings.setValue("group", self.group_combo.currentData())
        self.settings.setValue("gl_config", self.gl_config_combo.currentData())
        self.settings.setValue("surface", self.surface_combo.currentData())
        self.settings.setValue("watchdog", self.watchdog_combo.currentData())
        self.settings.setValue("log_images", self.log_images_check.isChecked())
        self.settings.sync()

    def _is_group_mode(self) -> bool:
        return str(self.mode_combo.currentData() or "exact") == "group"

    def _mode_changed(self, _value: object = None) -> None:
        group_mode = self._is_group_mode()
        self.group_combo.setEnabled(group_mode)
        self.suite_combo.setEnabled(not group_mode)
        self.case_combo.setEnabled(not group_mode)
        self.batch_progress.setVisible(group_mode)
        self.start_button.setText("Run group" if group_mode else "Run exact case")
        if group_mode:
            self._group_changed()
        else:
            self._update_case_note()
            self._update_command_preview()

    def _group_changed(self, _value: object = None) -> None:
        if not hasattr(self, "group_combo"):
            return
        try:
            group = get_group(str(self.group_combo.currentData()))
        except KeyError:
            return
        suite_index = self.suite_combo.findData(group.suite)
        if suite_index >= 0:
            self.suite_combo.setCurrentIndex(suite_index)
        if group.id == "gles3-stress-shaders":
            watchdog_index = self.watchdog_combo.findData("enable")
            if watchdog_index >= 0:
                self.watchdog_combo.setCurrentIndex(watchdog_index)
        self.batch_progress.setRange(0, max(1, group.locked_case_count))
        self.batch_progress.setValue(0)
        self.batch_progress.setFormat(
            f"0 / {group.locked_case_count:,} cases · discovered again at run time"
        )
        self._update_case_note()
        self._update_command_preview()

    def _suite_changed(self) -> None:
        suite = str(self.suite_combo.currentData() or "dEQP-GLES2")
        current_case = self.case_combo.currentText().strip()
        self._populate_cases(suite)
        if current_case.startswith(suite + "."):
            self.case_combo.setCurrentText(current_case)
        if suite == "dEQP-EGL" and self.gl_config_combo.currentData():
            automatic_index = self.gl_config_combo.findData("")
            if automatic_index >= 0:
                self.gl_config_combo.setCurrentIndex(automatic_index)
        elif suite != "dEQP-EGL" and not self.gl_config_combo.currentData():
            verified_index = self.gl_config_combo.findData("rgba8888d24s8ms0")
            if verified_index >= 0:
                self.gl_config_combo.setCurrentIndex(verified_index)
        self._update_case_note()
        self._update_command_preview()

    def _populate_cases(self, suite: str) -> None:
        self.case_combo.blockSignals(True)
        self.case_combo.clear()
        for label, case_name in SUITE_CASES.get(suite, ()):
            self.case_combo.addItem(label, case_name)
            self.case_combo.setItemData(
                self.case_combo.count() - 1,
                f"{label}\n{case_name}",
                Qt.ItemDataRole.ToolTipRole,
            )
        if self.case_combo.count():
            self.case_combo.setCurrentIndex(0)
            self.case_combo.setEditText(str(self.case_combo.itemData(0)))
        self.case_combo.blockSignals(False)

    def _case_changed(self, _text: str) -> None:
        self._update_case_note()
        self._update_command_preview()

    def _case_preset_activated(self, index: int) -> None:
        case_name = self.case_combo.itemData(index)
        if case_name:
            self.case_combo.setEditText(str(case_name))

    def _update_case_note(self, _value: object = None) -> None:
        if self._is_group_mode():
            group = get_group(str(self.group_combo.currentData()))
            if group.availability_reason:
                self.case_note.setText(
                    f"{group.label}: {group.locked_case_count:,} cases in the locked CTS. "
                    f"BLOCKED — {group.availability_reason}"
                )
                self.case_note.setProperty("warning", True)
            else:
                self.case_note.setText(
                    f"{group.label}: {group.locked_case_count:,} cases in the locked CTS. "
                    "The group is expanded first, then every exact case runs in its own "
                    "process so dEQP, driver, and SystemC evidence stay paired."
                )
                self.case_note.setProperty("warning", False)
            self.case_note.style().unpolish(self.case_note)
            self.case_note.style().polish(self.case_note)
            return
        case_name = self.case_combo.currentText().strip()
        surface = str(self.surface_combo.currentData() or "pbuffer")
        if surface != "pbuffer":
            self.case_note.setText(
                f"{surface.upper()} is experimental with the current PvrGPU EGL runtime. "
                "Use Pbuffer for the verified live pipeline."
            )
            self.case_note.setProperty("warning", True)
        elif case_name.startswith("dEQP-EGL.") or ".info." in case_name:
            self.case_note.setText(
                "Runtime-info case: validates dEQP and the linked Mesa runtime; it may not "
                "submit a GPU command, so driver/SystemC artifacts can be N/A."
            )
            self.case_note.setProperty("warning", False)
        else:
            self.case_note.setText(
                "Exact-case pipeline mode: one process produces dEQP, driver, and SystemC "
                "evidence. Wildcards and comma lists are intentionally rejected."
            )
            self.case_note.setProperty("warning", False)
        self.case_note.style().unpolish(self.case_note)
        self.case_note.style().polish(self.case_note)

    def _browse_runner(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select pvrgpu-deqp", self.runner_edit.text() or str(BUILD_ROOT / "bin")
        )
        if path:
            self.runner_edit.setText(path)
            self._update_command_preview()

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Select live dEQP output root", self.output_edit.text()
        )
        if path:
            self.output_edit.setText(path)
            self._update_command_preview()

    def _browse_bridge(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select PvrGPU SystemC bridge", self.bridge_edit.text() or str(BUILD_ROOT / "lib")
        )
        if path:
            self.bridge_edit.setText(path)
            self._update_command_preview()

    def _runner_arguments(self, output_dir: str) -> list[str]:
        arguments = [
            f"--pvrgpu-output-dir={output_dir}",
            f"--deqp-case={self.case_combo.currentText().strip()}",
            f"--deqp-surface-type={self.surface_combo.currentData()}",
            f"--deqp-watchdog={self.watchdog_combo.currentData()}",
            "--deqp-log-images=" + ("enable" if self.log_images_check.isChecked() else "disable"),
        ]
        gl_config = str(self.gl_config_combo.currentData() or "")
        if gl_config:
            arguments.append(f"--deqp-gl-config-name={gl_config}")
        bridge = self.bridge_edit.text().strip()
        if bridge:
            arguments.append(f"--pvrgpu-systemc-api-lib={bridge}")
        return arguments

    def _group_arguments(self, runner: str, output_dir: str) -> list[str]:
        arguments = [
            str(GROUP_RUNNER),
            f"--runner={runner}",
            f"--output-dir={output_dir}",
            f"--group-id={self.group_combo.currentData()}",
            f"--surface={self.surface_combo.currentData()}",
            f"--watchdog={self.watchdog_combo.currentData()}",
            "--log-images=" + ("enable" if self.log_images_check.isChecked() else "disable"),
        ]
        gl_config = str(self.gl_config_combo.currentData() or "")
        if gl_config:
            arguments.append(f"--gl-config={gl_config}")
        bridge = self.bridge_edit.text().strip()
        if bridge:
            arguments.append(f"--systemc-api-lib={bridge}")
        return arguments

    def _command_for_output(self, output_dir: str) -> tuple[str, list[str]]:
        runner = self.runner_edit.text().strip() or str(self.default_runner)
        if self._is_group_mode():
            return sys.executable, self._group_arguments(runner, output_dir)
        return runner, self._runner_arguments(output_dir)

    def _update_command_preview(self) -> None:
        if not hasattr(self, "command_preview"):
            return
        try:
            program, arguments = self._command_for_output("<timestamped-output>")
            command = shlex.join([program, *arguments])
        except ValueError as error:
            command = f"Invalid extra arguments: {error}"
        self.command_preview.setText(command)
        self.command_preview.setCursorPosition(0)

    def _resolved_path(self, value: str) -> Path:
        path = Path(value.strip()).expanduser()
        return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()

    def _validate_start(self) -> tuple[Path, Path] | None:
        if self._is_group_mode():
            try:
                group = get_group(str(self.group_combo.currentData()))
            except KeyError as error:
                QMessageBox.critical(self, "Group required", str(error))
                return None
            if not group.available:
                QMessageBox.critical(
                    self,
                    "Group blocked by driver capability",
                    group.availability_reason or "This group is unavailable.",
                )
                return None
            if not group.locked_case_count:
                QMessageBox.critical(
                    self,
                    "Empty group",
                    "This group contains no exact cases in the locked CTS tree.",
                )
                return None
            if not GROUP_RUNNER.is_file():
                QMessageBox.critical(
                    self, "Group runner missing", f"Batch helper was not found:\n{GROUP_RUNNER}"
                )
                return None
        else:
            case_name = self.case_combo.currentText().strip()
            if not case_name:
                QMessageBox.warning(self, "Case required", "Choose or enter a dEQP case.")
                return None
            suite = str(self.suite_combo.currentData() or "")
            if suite in UNAVAILABLE_SUITES:
                QMessageBox.critical(self, "Suite unavailable", UNAVAILABLE_SUITES[suite])
                return None
            if not EXACT_CASE_RE.fullmatch(case_name) or not case_name.startswith(suite + "."):
                QMessageBox.critical(
                    self,
                    "Exact case required",
                    f"Enter one exact {suite} case. Wildcards, comma lists, and other suites "
                    "are not accepted because the SystemC bridge finalizes one command per process.",
                )
                return None
        runner = self._resolved_path(self.runner_edit.text())
        if not runner.is_file():
            QMessageBox.critical(self, "Runner missing", f"pvrgpu-deqp was not found:\n{runner}")
            return None
        if os.name != "nt" and not os.access(runner, os.X_OK):
            QMessageBox.critical(self, "Runner is not executable", str(runner))
            return None

        output_root = self._resolved_path(self.output_edit.text())
        if output_root == PROJECT_ROOT or PROJECT_ROOT in output_root.parents:
            QMessageBox.critical(
                self,
                "Output inside source tree",
                "Choose an output root outside the PvrGPU source/iCloud tree.",
            )
            return None
        bridge = self.bridge_edit.text().strip()
        if bridge and not self._resolved_path(bridge).is_file():
            QMessageBox.critical(
                self,
                "SystemC bridge missing",
                f"The selected bridge was not found:\n{self._resolved_path(bridge)}",
            )
            return None
        return runner, output_root

    def _start(self) -> None:
        if self.process.state() != QProcess.ProcessState.NotRunning:
            return
        validated = self._validate_start()
        if validated is None:
            return
        runner, output_root = validated
        try:
            output_root.mkdir(parents=True, exist_ok=True)
            timestamp = self.now_provider().strftime("%Y%m%d-%H%M%S-%f")[:-3]
            run_tail = (
                str(self.group_combo.currentData())
                if self._is_group_mode()
                else safe_case_name(self.case_combo.currentText().strip())[-96:]
            )
            run_dir = output_root / f"{timestamp}-{run_tail}"
            run_dir.mkdir(parents=False, exist_ok=False)
        except OSError as error:
            QMessageBox.critical(self, "Output unavailable", str(error))
            return

        self.run_dir = run_dir
        self.cancel_requested = False
        self.process_error_seen = False
        self.active_mode = "group" if self._is_group_mode() else "exact"
        self.active_group_id = (
            str(self.group_combo.currentData()) if self._is_group_mode() else ""
        )
        self.process_line_buffer = ""
        self.batch_counts = {"passed": 0, "skipped": 0, "warnings": 0, "failed": 0}
        self.qpa_summary = QpaSummary("", "", ())
        self.latest_png = None
        self.results_table.setRowCount(0)
        self.artifacts_table.setRowCount(0)
        self.image_preview.setText("Waiting for SystemC framebuffer PNG…")
        self.image_preview.setPixmap(QPixmap())
        self.log.clear()
        self.deqp_value.setText("RUNNING")
        self.driver_value.setText("Waiting")
        self.systemc_value.setText("Waiting")
        self.renderer_value.setText("—")
        if self._is_group_mode():
            group = get_group(self.active_group_id)
            self.summary_label.setText(f"Discovering {group.label}")
            self.batch_progress.setRange(0, max(1, group.locked_case_count))
            self.batch_progress.setValue(0)
        else:
            self.summary_label.setText(f"Running {self.case_combo.currentText().strip()}")
        self.open_output_button.setEnabled(False)
        self.open_qpa_button.setEnabled(False)
        self._save_settings()
        self._set_running_controls(True)
        self._set_state("RUNNING")
        self.tabs.setCurrentIndex(2)

        program, arguments = self._command_for_output(str(run_dir))
        if self._is_group_mode():
            arguments[1] = f"--runner={runner}"
        else:
            program = str(runner)
        self.last_command = [str(program), *arguments]
        self.log.appendPlainText("$ " + shlex.join(self.last_command) + "\n")

        environment = QProcessEnvironment.systemEnvironment()
        environment.remove("MESA_GLES_VERSION_OVERRIDE")
        environment.insert("PVRGPU_PROJECT_ROOT", str(PROJECT_ROOT))
        self.process.setProcessEnvironment(environment)
        self.process.setWorkingDirectory(str(run_dir))
        self.process.start(str(program), arguments)
        if not self.process.waitForStarted(3000):
            self._set_running_controls(False)
            self._set_state("FAILED")
            self.deqp_value.setText("LAUNCH ERROR")
            self.summary_label.setText(self.process.errorString() or "Runner failed to start")

    def _set_running_controls(self, running: bool) -> None:
        for widget in self._config_widgets:
            widget.setEnabled(not running)
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        if not running:
            group_mode = self._is_group_mode()
            self.group_combo.setEnabled(group_mode)
            self.suite_combo.setEnabled(not group_mode)
            self.case_combo.setEnabled(not group_mode)
            self.start_button.setText("Run group" if group_mode else "Run exact case")

    def _stop(self) -> None:
        if self.process.state() == QProcess.ProcessState.NotRunning:
            return
        self.cancel_requested = True
        self.stop_button.setEnabled(False)
        self._set_state("CANCELLING")
        self.summary_label.setText(
            "Stopping group after the current case…"
            if self.active_mode == "group"
            else "Stopping dEQP…"
        )
        self.process.terminate()
        QTimer.singleShot(2000, self._kill_if_running)

    def _kill_if_running(self) -> None:
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.kill()

    def _read_process_output(self) -> None:
        chunk = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        chunk = ANSI_ESCAPE_RE.sub("", chunk)
        if not chunk:
            return
        cursor = self.log.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(chunk)
        self.log.setTextCursor(cursor)
        self.log.ensureCursorVisible()
        if self.active_mode == "group":
            self.process_line_buffer += chunk
            while "\n" in self.process_line_buffer:
                line, self.process_line_buffer = self.process_line_buffer.split("\n", 1)
                self._parse_batch_event(line)

    def _parse_batch_event(self, line: str) -> None:
        if not line.startswith(EVENT_PREFIX):
            return
        try:
            payload = json.loads(line[len(EVENT_PREFIX) :])
        except (json.JSONDecodeError, TypeError):
            return
        event = str(payload.get("event", ""))
        if event in {"discovery_finished", "batch_started"}:
            total = int(payload.get("selected", payload.get("total", 0)) or 0)
            self.batch_progress.setRange(0, max(1, total))
            self.batch_progress.setValue(0)
            self.batch_progress.setFormat(f"0 / {total:,} cases")
            self.summary_label.setText(f"Discovered {total:,} exact cases")
        elif event == "case_started":
            index = int(payload.get("index", 0) or 0)
            total = int(payload.get("total", 0) or 0)
            case_name = str(payload.get("case", ""))
            self.summary_label.setText(f"Running {index:,}/{total:,} · {case_name}")
        elif event == "case_finished":
            index = int(payload.get("index", 0) or 0)
            total = int(payload.get("total", 0) or 0)
            status = str(payload.get("status", "")).casefold()
            if status in PASS_STATUSES:
                bucket = "passed"
            elif status in SKIP_STATUSES:
                bucket = "skipped"
            elif status in WARNING_STATUSES:
                bucket = "warnings"
            else:
                bucket = "failed"
            self.batch_counts[bucket] += 1
            self.batch_progress.setRange(0, max(1, total))
            self.batch_progress.setValue(index)
            self.batch_progress.setFormat(
                f"{index:,} / {total:,} · "
                f"{self.batch_counts['passed']:,} pass · "
                f"{self.batch_counts['failed']:,} fail"
            )
            self.deqp_value.setText(
                f"{self.batch_counts['passed']:,} pass · "
                f"{self.batch_counts['skipped']:,} skip · "
                f"{self.batch_counts['warnings']:,} warn · "
                f"{self.batch_counts['failed']:,} fail"
            )
        elif event == "error":
            self.summary_label.setText(str(payload.get("message", "Group runner failed")))

    def _process_error(self, error: QProcess.ProcessError) -> None:
        self.process_error_seen = True
        if error == QProcess.ProcessError.FailedToStart:
            self._set_state("FAILED")
            self.summary_label.setText(self.process.errorString() or "Runner failed to start")

    def _process_finished(
        self,
        exit_code: int,
        exit_status: QProcess.ExitStatus,
    ) -> None:
        self._read_process_output()
        if self.process_line_buffer:
            self._parse_batch_event(self.process_line_buffer)
            self.process_line_buffer = ""
        self._set_running_controls(False)
        self.stop_button.setEnabled(False)
        self._refresh_results(exit_code, exit_status)
        if self.cancel_requested:
            self._set_state("CANCELLED")
            self.deqp_value.setText("CANCELLED")
            partial = len(self.qpa_summary.cases)
            incomplete = self.qpa_summary.incomplete_count
            self.summary_label.setText(
                f"Run cancelled · {partial} completed · {incomplete} incomplete · "
                "partial artifacts retained"
            )
        self.open_output_button.setEnabled(self.run_dir is not None and self.run_dir.is_dir())
        self.open_qpa_button.setEnabled(
            self.run_dir is not None and (self.run_dir / "results.qpa").is_file()
        )

    def _refresh_results(
        self,
        exit_code: int,
        exit_status: QProcess.ExitStatus,
    ) -> None:
        if self.run_dir is None:
            return
        self.qpa_summary = parse_qpa(self.run_dir / "results.qpa")
        case_results = self.qpa_summary.cases
        requested_case = self.case_combo.currentText().strip()
        passed = sum(result.status.casefold() in PASS_STATUSES for result in case_results)
        skipped = sum(result.status.casefold() in SKIP_STATUSES for result in case_results)
        warned = sum(result.status.casefold() in WARNING_STATUSES for result in case_results)
        failed = sum(result.status.casefold() not in NON_FAILURE_STATUSES for result in case_results)
        if self.active_mode == "group" and self.active_group_id:
            group = get_group(self.active_group_id)
            mismatched = sum(
                not exact_case_belongs_to_group(group, result.case_name)
                for result in case_results
            )
        else:
            mismatched = sum(result.case_name != requested_case for result in case_results)

        artifact_results = tuple(
            result
            for result in case_results
            if expects_gpu_artifacts(result.case_name)
            and result.status.casefold() in (PASS_STATUSES | WARNING_STATUSES)
        )

        def driver_is_ready(result: QpaCaseResult) -> bool:
            case_dir = self.run_dir / "cases" / safe_case_name(result.case_name)
            return (
                (case_dir / "driver-command.txt").is_file()
                and (case_dir / "driver-counter.txt").is_file()
            )

        driver_ready = sum(driver_is_ready(result) for result in artifact_results)
        systemc_ready = sum(
            has_complete_systemc(self.run_dir / "cases" / safe_case_name(result.case_name))
            for result in artifact_results
        )
        displayed_results = case_results[:RESULT_ROW_LIMIT]
        self.results_table.setRowCount(len(displayed_results))
        for row, result in enumerate(displayed_results):
            case_dir = self.run_dir / "cases" / safe_case_name(result.case_name)
            needs_artifacts = (
                expects_gpu_artifacts(result.case_name)
                and result.status.casefold() in (PASS_STATUSES | WARNING_STATUSES)
            )
            if needs_artifacts:
                driver_status = "READY" if driver_is_ready(result) else "MISSING"
                systemc_status = "READY" if has_complete_systemc(case_dir) else "MISSING"
            else:
                driver_status = "N/A"
                systemc_status = "N/A"
            duration = "—" if result.duration_us is None else f"{result.duration_us / 1000.0:.1f} ms"
            values = (result.case_name, result.status.upper(), driver_status, systemc_status, duration)
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column in (1, 2, 3):
                    self._color_status_item(item, value)
                if column == 1 and result.message:
                    item.setToolTip(result.message)
                self.results_table.setItem(row, column, item)

        self.deqp_value.setText(
            f"{passed} pass · {skipped} skip · {warned} warn · {failed} fail"
            if case_results
            else f"EXIT {exit_code} · 0 results"
        )
        if artifact_results:
            self.driver_value.setText(f"{driver_ready} / {len(artifact_results)} ready")
            self.systemc_value.setText(f"{systemc_ready} / {len(artifact_results)} complete")
        else:
            self.driver_value.setText("N/A · no GPU submit")
            self.systemc_value.setText("N/A · no GPU submit")
        self.renderer_value.setText(self.qpa_summary.renderer or "—")

        if not self.cancel_requested:
            crashed = exit_status == QProcess.ExitStatus.CrashExit
            artifacts_complete = (
                driver_ready == len(artifact_results)
                and systemc_ready == len(artifact_results)
            )
            if crashed:
                self._set_state("CRASHED")
                self.summary_label.setText(
                    f"Runner crashed · exit {exit_code} · {len(case_results)} closed results"
                )
                self.tabs.setCurrentIndex(2)
            elif exit_code != 0:
                state = "FAIL" if failed else "ERROR"
                self._set_state(state)
                self.summary_label.setText(
                    f"{state} · exit {exit_code} · {passed} pass · {failed} fail"
                )
                self.tabs.setCurrentIndex(0 if case_results else 2)
            elif self.qpa_summary.incomplete_count:
                self._set_state("INCOMPLETE")
                self.summary_label.setText(
                    f"INCOMPLETE · {self.qpa_summary.incomplete_count} unclosed QPA result"
                )
                self.tabs.setCurrentIndex(0)
            elif not case_results or mismatched:
                self._set_state("NO MATCH")
                self.summary_label.setText(
                    "NO MATCH · one or more results do not belong to the requested selection"
                )
                self.tabs.setCurrentIndex(2)
            elif failed:
                self._set_state("FAIL")
                self.summary_label.setText(
                    f"FAIL · {passed} pass · {skipped} skip · {warned} warn · {failed} fail"
                )
                self.tabs.setCurrentIndex(0)
            elif artifact_results and not artifacts_complete:
                self._set_state("INCOMPLETE")
                self.summary_label.setText(
                    f"dEQP passed, but linked pipeline evidence is incomplete · "
                    f"driver {driver_ready}/{len(artifact_results)} · "
                    f"SystemC {systemc_ready}/{len(artifact_results)}"
                )
                self.tabs.setCurrentIndex(0)
            elif warned:
                self._set_state("WARNING")
                self.summary_label.setText(
                    f"WARNING · {warned} warning result · {skipped} skipped"
                )
                self.tabs.setCurrentIndex(0)
            elif passed:
                self._set_state("PASS")
                self.summary_label.setText(
                    f"PASS · {passed}/{len(case_results)} dEQP · "
                    + (
                        f"driver {driver_ready}/{len(artifact_results)} · "
                        f"SystemC {systemc_ready}/{len(artifact_results)}"
                        if artifact_results
                        else "runtime-info case · no GPU submission expected"
                    )
                )
                self.tabs.setCurrentIndex(0)
            else:
                self._set_state("SKIPPED")
                self.summary_label.setText(
                    f"SKIPPED · {skipped}/{len(case_results)} unsupported or waived"
                )
                self.tabs.setCurrentIndex(0)
            if len(case_results) > len(displayed_results):
                self.summary_label.setText(
                    self.summary_label.text()
                    + f" · table shows first {len(displayed_results):,}/{len(case_results):,}"
                )
        self._refresh_artifacts()

    def _refresh_artifacts(self) -> None:
        if self.run_dir is None or not self.run_dir.is_dir():
            return
        files: list[Path] = []
        truncated = False
        for path in self.run_dir.rglob("*"):
            if not path.is_file():
                continue
            if len(files) >= ARTIFACT_ROW_LIMIT:
                truncated = True
                break
            files.append(path)
        files.sort()
        self.artifacts_table.setRowCount(len(files))
        self.latest_png = None
        for row, path in enumerate(files):
            relative = path.relative_to(self.run_dir)
            type_name = path.suffix.lstrip(".").upper() or "FILE"
            name_item = QTableWidgetItem(str(relative))
            name_item.setData(Qt.ItemDataRole.UserRole, str(path))
            self.artifacts_table.setItem(row, 0, name_item)
            self.artifacts_table.setItem(row, 1, QTableWidgetItem(type_name))
            try:
                size = format_bytes(path.stat().st_size)
            except OSError:
                size = "—"
            self.artifacts_table.setItem(row, 2, QTableWidgetItem(size))
            if path.suffix.casefold() == ".png":
                self.latest_png = path
        if self.latest_png is not None:
            self._show_image(self.latest_png)
        elif truncated:
            self.image_preview.setText(
                f"Artifact table is capped at {ARTIFACT_ROW_LIMIT:,} files. "
                "Open the output directory for the complete batch."
            )

    def _color_status_item(self, item: QTableWidgetItem, status: str) -> None:
        normalized = status.upper()
        if normalized in {"PASS", "READY"}:
            item.setForeground(QColor("#166534"))
            item.setBackground(QColor("#dcfce7"))
        elif normalized in {
            "NOTSUPPORTED",
            "WAIVER",
            "QUALITYWARNING",
            "COMPATIBILITYWARNING",
        }:
            item.setForeground(QColor("#92400e"))
            item.setBackground(QColor("#fef3c7"))
        elif normalized in {"FAIL", "MISSING", "ERROR", "INCOMPLETE"}:
            item.setForeground(QColor("#991b1b"))
            item.setBackground(QColor("#fee2e2"))
        elif normalized == "N/A":
            item.setForeground(QColor("#64748b"))

    def _artifact_selected(self) -> None:
        row = self.artifacts_table.currentRow()
        if row < 0:
            return
        item = self.artifacts_table.item(row, 0)
        if item is None:
            return
        path_text = item.data(Qt.ItemDataRole.UserRole)
        if not path_text:
            return
        path = Path(str(path_text))
        if path.suffix.casefold() == ".png":
            self._show_image(path)
        else:
            self.image_preview.setPixmap(QPixmap())
            self.image_preview.setText(f"{path.name}\n\nDouble-click to open")

    def _show_image(self, path: Path) -> None:
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self.image_preview.setPixmap(QPixmap())
            self.image_preview.setText(f"Could not preview {path.name}")
            return
        scaled = pixmap.scaled(
            max(1, self.image_preview.width() - 18),
            max(1, self.image_preview.height() - 18),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_preview.setText("")
        self.image_preview.setPixmap(scaled)

    def _open_artifact_item(self, item: QTableWidgetItem) -> None:
        path_item = self.artifacts_table.item(item.row(), 0)
        if path_item is None:
            return
        path_text = path_item.data(Qt.ItemDataRole.UserRole)
        if path_text:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path_text)))

    def _open_output(self) -> None:
        if self.run_dir is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.run_dir)))

    def _open_qpa(self) -> None:
        if self.run_dir is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.run_dir / "results.qpa")))

    def _set_state(self, state: str) -> None:
        self.state_badge.setText(state)
        self.state_badge.setProperty("state", state.casefold())
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
            QLabel#title { font-size: 25px; font-weight: 750; }
            QLabel#subtitle, QLabel#summary { color: #5b667a; }
            QLabel#stateBadge {
                background: #e0e7ff; color: #3730a3; border-radius: 11px;
                padding: 6px 12px; font-weight: 750;
            }
            QLabel#stateBadge[state="running"] { background: #dbeafe; color: #1d4ed8; }
            QLabel#stateBadge[state="pass"] {
                background: #dcfce7; color: #166534;
            }
            QLabel#stateBadge[state="fail"], QLabel#stateBadge[state="failed"],
            QLabel#stateBadge[state="error"], QLabel#stateBadge[state="crashed"],
            QLabel#stateBadge[state="incomplete"], QLabel#stateBadge[state="no match"] {
                background: #fee2e2; color: #991b1b;
            }
            QLabel#stateBadge[state="cancelled"], QLabel#stateBadge[state="cancelling"],
            QLabel#stateBadge[state="warning"], QLabel#stateBadge[state="skipped"] {
                background: #fef3c7; color: #92400e;
            }
            QLabel#caseNote {
                background: #eff6ff; color: #1e40af; border-radius: 5px; padding: 7px;
            }
            QLabel#caseNote[warning="true"] { background: #fff7ed; color: #9a3412; }
            QGroupBox {
                background: white; border: 1px solid #d9dee7; border-radius: 8px;
                margin-top: 10px; padding-top: 12px; font-weight: 650;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; }
            QLineEdit, QComboBox, QPlainTextEdit {
                background: white; border: 1px solid #cbd5e1; border-radius: 5px; padding: 6px;
            }
            QComboBox QAbstractItemView { background: white; selection-background-color: #e0e7ff; }
            QPushButton {
                background: white; border: 1px solid #cbd5e1; border-radius: 5px;
                padding: 7px 13px;
            }
            QPushButton:hover { background: #eef2ff; }
            QPushButton#primaryButton {
                background: #4f46e5; color: white; border: none; font-weight: 750;
            }
            QPushButton:disabled { color: #9ca3af; background: #e5e7eb; }
            QFrame#metricCard {
                background: white; border: 1px solid #d9dee7; border-radius: 8px;
            }
            QLabel#metricTitle { color: #64748b; font-size: 11px; font-weight: 650; }
            QLabel#metricValue { font-size: 15px; font-weight: 750; }
            QTableWidget {
                background: white; alternate-background-color: #f8fafc;
                gridline-color: #e2e8f0;
            }
            QHeaderView::section {
                background: #eef2f7; color: #334155; padding: 6px;
                border: none; border-right: 1px solid #dbe2ea; font-weight: 650;
            }
            QTabWidget::pane { border: 1px solid #d9dee7; background: white; }
            QTabBar::tab { padding: 8px 16px; background: #e9edf3; }
            QTabBar::tab:selected { background: white; color: #4338ca; font-weight: 700; }
            QFrame#previewFrame { background: #111827; border-radius: 6px; }
            QLabel#imagePreview { color: #cbd5e1; background: #111827; }
            """
        )


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("PvrGPU Live dEQP")
    window = MainWindow()
    window.show()
    if "--smoke-test" in sys.argv:
        QTimer.singleShot(250, app.quit)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
