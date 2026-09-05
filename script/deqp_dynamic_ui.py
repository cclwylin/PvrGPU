#!/usr/bin/env python3
"""Desktop front end for the dynamically linked PvrGPU dEQP runner.

This drives ``script/run_deqp_dynamic.sh`` -- the shell script stays the single
source of truth for the dynamic wiring (stock deqp-<module> binary ->
DYLD_LIBRARY_PATH -> PCO driver Mesa prefix -> dlopen'd PvrGPU SystemC bridge).
The UI only selects what to run, shows live status, and summarizes the run.

It is the dynamic-link counterpart of ``tools/deqp_live_ui.py``, which drives
the statically linked ``pvrgpu-deqp`` executable instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import re
import sys


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
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QColor,
    QDesktopServices,
    QFont,
    QPalette,
)
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
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

# ------------------------------------------------------------------------------
# Project layout
# ------------------------------------------------------------------------------
REPO_ROOT = Path(
    os.environ.get("PVRGPU_PROJECT_ROOT", Path(__file__).resolve().parents[1])
).resolve()
RUNNER_SCRIPT = REPO_ROOT / "script" / "run_deqp_dynamic.sh"
TOOLS_DIR = REPO_ROOT / "tools"
if TOOLS_DIR.is_dir() and str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

try:  # the 24-group catalog is shared with tools/deqp_live_ui.py
    from deqp_groups import GROUP_SPECS, filter_exact_cases, get_group

    GROUPS_AVAILABLE = True
except Exception:  # pragma: no cover - catalog missing or unreadable
    GROUP_SPECS = ()
    GROUPS_AVAILABLE = False

    def filter_exact_cases(group, case_names):  # type: ignore[misc]
        return tuple(case_names)

    def get_group(group):  # type: ignore[misc]
        raise RuntimeError("deqp_groups is unavailable")


EVENT_PREFIX = "PVRGPU_DYN "
LOG_LINE_LIMIT = 20_000
# Diagnostics report bounds.  The report is meant to be pasted into a
# conversation, so it stays readable rather than complete; the run
# directory beside it holds everything that was trimmed.
DIAGNOSTICS_MAX_CASES = 12
DIAGNOSTICS_LOG_LINES = 200
ARTIFACT_ROW_LIMIT = 5_000

PASS_STATUSES = {"pass"}
SKIP_STATUSES = {"notsupported", "waiver"}
WARNING_STATUSES = {"qualitywarning", "compatibilitywarning"}

EXACT_CASE_RE = re.compile(r"^dEQP-(?:EGL|GLES2|GLES3|GLES31)\.[A-Za-z0-9_.-]+$")

SUITE_TO_MODULE = {
    "dEQP-EGL": "egl",
    "dEQP-GLES2": "gles2",
    "dEQP-GLES3": "gles3",
    "dEQP-GLES31": "gles31",
}

PRESET_CASES: tuple[tuple[str, str], ...] = (
    ("EGL · Create context · rgb565", "dEQP-EGL.functional.create_context.rgb565_no_depth_no_stencil"),
    ("EGL · Create context · rgb888 depth", "dEQP-EGL.functional.create_context.rgb888_depth_no_stencil"),
    ("EGL · Info · Configs", "dEQP-EGL.info.configs"),
    ("EGL · Info · Client APIs", "dEQP-EGL.info.client_apis"),
    ("EGL · Info · Version", "dEQP-EGL.info.version"),
    ("GLES2 · Clear color smoke", "dEQP-GLES2.functional.prerequisite.clear_color"),
    ("GLES2 · Info · Renderer", "dEQP-GLES2.info.renderer"),
    ("GLES2 · Color clear · single_rgb", "dEQP-GLES2.functional.color_clear.single_rgb"),
    (
        "GLES31 · draw_indirect smoke (dEQP 專案預設 case)",
        "dEQP-GLES31.functional.draw_indirect.draw_arrays_indirect.triangles.single_attribute",
    ),
)

MODE_GROUP = "Group 目錄（24 組）"
MODE_PRESET = "常用單一 case"
MODE_CUSTOM = "自訂 case"
MODE_CASELIST = "Caselist 檔案"

ENV_LINE_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


def load_local_env(path: Path) -> dict[str, str]:
    """Parse config/local.env well enough to seed the path fields."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values

    def expand(text: str) -> str:
        return re.sub(
            r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}",
            lambda match: values.get(
                match.group(1), os.environ.get(match.group(1), "")
            ),
            text,
        )

    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = ENV_LINE_RE.match(line)
        if not match:
            continue
        name, value = match.group(1), match.group(2).strip()
        if len(value) >= 2 and value[0] == value[-1] == "'":
            value = value[1:-1]
        else:
            if len(value) >= 2 and value[0] == value[-1] == '"':
                value = value[1:-1]
            value = expand(value)
        values[name] = value
    return values


def status_bucket(status: str) -> str:
    lowered = status.strip().lower()
    if lowered in PASS_STATUSES:
        return "pass"
    if lowered in SKIP_STATUSES:
        return "skip"
    if lowered in WARNING_STATUSES:
        return "warn"
    return "fail"


def format_bytes(size: int) -> str:
    value = float(size)
    for suffix in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or suffix == "GiB":
            return f"{value:.0f} {suffix}" if suffix == "B" else f"{value:.1f} {suffix}"
        value /= 1024.0
    return f"{size} B"


def format_duration(milliseconds: int) -> str:
    seconds = milliseconds / 1000.0
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, seconds = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m {seconds:02d}s"


@dataclass
class CaseRow:
    index: int
    case_name: str
    status: str = "Running"
    exit_code: int | None = None
    duration_ms: int = 0
    case_dir: Path | None = None
    qpa: Path | None = None
    log: Path | None = None

    @property
    def bucket(self) -> str:
        return status_bucket(self.status)

    def artifact_summary(self) -> str:
        if self.case_dir is None or not self.case_dir.is_dir():
            return "-"
        marks: list[str] = []
        jsonl = self.case_dir / "systemc.jsonl"
        if jsonl.is_file() and jsonl.stat().st_size > 0:
            try:
                done = '"type":"done"' in jsonl.read_text(
                    encoding="utf-8", errors="replace"
                )
            except OSError:
                done = False
            marks.append("systemc✓" if done else "systemc…")
        if any((self.case_dir / "systemc").glob("*.png")):
            marks.append("png")
        command = self.case_dir / "driver-command.txt"
        if command.is_file() and command.stat().st_size > 0:
            marks.append("cmd")
        if (self.case_dir / "link.txt").is_file():
            marks.append("link")
        return " · ".join(marks) if marks else "-"


@dataclass
class RunState:
    output_root: Path | None = None
    summary_path: Path | None = None
    module: str = ""
    runner: str = ""
    archive_dir: str = ""
    mesa_prefix: str = ""
    systemc_lib: str = ""
    host_arch: str = ""
    total: int = 0
    started_at: datetime | None = None
    duration_ms: int = 0
    rows: list[CaseRow] = field(default_factory=list)


@dataclass(frozen=True)
class Theme:
    """One coherent color set, so the window works in light and dark mode."""

    window: str
    surface: str
    field: str
    subtle: str
    border: str
    text: str
    muted: str
    accent: str
    accent_text: str
    ok: str
    bad: str
    warn: str
    skip: str
    ok_bg: str
    bad_bg: str
    warn_bg: str
    log_bg: str
    log_text: str


LIGHT_THEME = Theme(
    window="#f4f6f9", surface="#ffffff", field="#ffffff", subtle="#eef1f6",
    border="#d9dee7", text="#1f2430", muted="#5b6472",
    accent="#2f6feb", accent_text="#ffffff",
    ok="#1b7f3b", bad="#b3261e", warn="#a06000", skip="#5b6472",
    ok_bg="#e8f5ec", bad_bg="#fdecea", warn_bg="#fdf4e3",
    log_bg="#11151c", log_text="#dfe6f1",
)

DARK_THEME = Theme(
    window="#1b1e24", surface="#242830", field="#1a1d24", subtle="#2b303a",
    border="#39404c", text="#e7ebf3", muted="#9aa4b5",
    accent="#4d8bff", accent_text="#0b0e13",
    ok="#5cc07f", bad="#ff7168", warn="#f0b357", skip="#9aa4b5",
    ok_bg="#1e3327", bad_bg="#3a2321", warn_bg="#38301d",
    log_bg="#11151c", log_text="#dfe6f1",
)


def detect_theme(application) -> Theme:
    """Follow the desktop appearance instead of forcing one palette."""
    try:
        window_color = application.palette().color(QPalette.ColorRole.Window)
        if window_color.lightness() < 128:
            return DARK_THEME
    except Exception:  # pragma: no cover - defensive
        pass
    return LIGHT_THEME


DEFAULT_APPEARANCE = "light"


def resolve_theme(preference: str, system_theme: Theme) -> Theme:
    """Map a stored appearance preference onto a concrete theme."""
    if preference == "light":
        return LIGHT_THEME
    if preference == "dark":
        return DARK_THEME
    return system_theme


def apply_palette(application, theme: Theme) -> None:
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(theme.window))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(theme.text))
    palette.setColor(QPalette.ColorRole.Base, QColor(theme.field))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(theme.subtle))
    palette.setColor(QPalette.ColorRole.Text, QColor(theme.text))
    palette.setColor(QPalette.ColorRole.Button, QColor(theme.surface))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(theme.text))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(theme.surface))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(theme.text))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(theme.muted))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(theme.accent))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(theme.accent_text))
    application.setPalette(palette)


class MainWindow(QMainWindow):
    """Pick a preset, watch it run, then read the dashboard."""

    def __init__(
        self,
        theme: Theme | None = None,
        *,
        system_theme: Theme | None = None,
        appearance: str = DEFAULT_APPEARANCE,
    ) -> None:
        super().__init__()
        self.system_theme = system_theme or theme or LIGHT_THEME
        self.appearance = appearance
        self.theme = theme or LIGHT_THEME
        self.setWindowTitle("PvrGPU dEQP · Dynamic Link")
        self.resize(1440, 900)
        self.setMinimumSize(1100, 700)

        self.settings = QSettings("PvrGPU", "deqp-dynamic-ui")
        self.local_env = load_local_env(REPO_ROOT / "config" / "local.env")

        self.process: QProcess | None = None
        self.phase = "idle"  # idle | check | discover | run
        self.state = RunState()
        self.pending_cases: list[str] = []
        self.run_dir: Path | None = None
        self.last_exit_code: int | None = None
        self.last_phase: str = ""
        self.current_group = None
        self.caselist_path: Path | None = None
        self.stdout_buffer = ""
        self.log_lines = 0

        self.elapsed_timer = QTimer(self)
        self.elapsed_timer.setInterval(500)
        self.elapsed_timer.timeout.connect(self._tick_elapsed)

        self._build_ui()
        self._build_menu()
        self._restore_settings()
        self._update_mode_visibility()
        self._apply_style()

    # --------------------------------------------------------------------------
    # Construction
    # --------------------------------------------------------------------------
    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setChildrenCollapsible(False)
        splitter.setSizes([440, 1000])
        self.setCentralWidget(splitter)
        self.statusBar().showMessage("Ready")

    def _compact_combo(self, combo: QComboBox) -> None:
        """Keep a long item list from dictating the side panel's width."""
        combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        combo.setMinimumContentsLength(12)
        combo.view().setMinimumWidth(360)

    def _caption(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("caption")
        return label

    def _field(self, caption: str, widget: QWidget, tooltip: str = "") -> QWidget:
        """One stacked caption + control, so nothing fights for column width."""
        holder = QWidget()
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._caption(caption))
        layout.addWidget(widget)
        if tooltip:
            widget.setToolTip(tooltip)
        return holder

    def _side_by_side(self, left: QWidget, right: QWidget) -> QWidget:
        holder = QWidget()
        layout = QHBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(left, 1)
        layout.addWidget(right, 1)
        return holder

    def _build_menu(self) -> None:
        """A small appearance menu, so the window is not stuck on one theme."""
        menu = self.menuBar().addMenu("外觀")
        group = QActionGroup(self)
        group.setExclusive(True)
        self.theme_actions: dict[str, QAction] = {}
        for key, text in (
            ("light", "淺色"),
            ("dark", "深色"),
            ("system", "跟隨系統"),
        ):
            action = QAction(text, self)
            action.setCheckable(True)
            action.setChecked(key == self.appearance)
            action.triggered.connect(
                lambda _checked=False, chosen=key: self.set_appearance(chosen)
            )
            group.addAction(action)
            menu.addAction(action)
            self.theme_actions[key] = action

    def set_appearance(self, preference: str) -> None:
        self.appearance = preference
        self.settings.setValue("appearance/theme", preference)
        self.theme = resolve_theme(preference, self.system_theme)
        application = QApplication.instance()
        if application is not None:
            apply_palette(application, self.theme)
        self._apply_style()
        for row in self.state.rows:
            if row.exit_code is not None:
                self._update_result_row(row)
        action = self.theme_actions.get(preference)
        if action is not None and not action.isChecked():
            action.setChecked(True)

    def _build_left_panel(self) -> QWidget:
        """Scrollable settings with the Run/Cancel row pinned below them."""
        panel = QWidget()
        panel.setObjectName("sidePanel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(14, 14, 10, 4)
        panel_layout.setSpacing(12)
        panel_layout.addWidget(self._build_selection_group())
        panel_layout.addWidget(self._build_paths_group())
        panel_layout.addWidget(self._build_parameters_group())
        panel_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setObjectName("sideScroll")
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        container.setObjectName("sidePanel")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(scroll, 1)

        actions = QWidget()
        actions.setObjectName("sideActions")
        action_layout = QHBoxLayout(actions)
        action_layout.setContentsMargins(14, 10, 10, 12)
        action_layout.setSpacing(8)
        self.run_button = QPushButton("Run")
        self.run_button.setObjectName("primary")
        self.run_button.setMinimumHeight(36)
        self.run_button.clicked.connect(self.start_run)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setMinimumHeight(36)
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_run)
        self.quit_button = QPushButton("Quit")
        self.quit_button.setObjectName("quit")
        self.quit_button.setMinimumHeight(36)
        self.quit_button.setToolTip("關閉視窗；若還在跑會先問要不要中止")
        self.quit_button.clicked.connect(self.close)
        action_layout.addWidget(self.run_button, 2)
        action_layout.addWidget(self.cancel_button, 1)
        action_layout.addWidget(self.quit_button, 1)
        layout.addWidget(actions)

        container.setMinimumWidth(360)
        container.setMaximumWidth(560)
        return container

    def _build_selection_group(self) -> QGroupBox:
        box = QGroupBox("選擇要跑的項目")
        layout = QVBoxLayout(box)
        layout.setSpacing(10)

        self.mode_combo = QComboBox()
        modes = [MODE_GROUP, MODE_PRESET, MODE_CUSTOM, MODE_CASELIST]
        if not GROUPS_AVAILABLE:
            modes.remove(MODE_GROUP)
        self.mode_combo.addItems(modes)
        self.mode_combo.currentTextChanged.connect(self._update_mode_visibility)
        layout.addWidget(self._field("模式", self.mode_combo))

        self.group_combo = QComboBox()
        for spec in GROUP_SPECS:
            marker = "" if spec.available else "⛔ "
            self.group_combo.addItem(
                f"{marker}{spec.label} · {spec.locked_case_count}", spec.id
            )
        self.group_combo.currentIndexChanged.connect(self._update_group_note)
        self._compact_combo(self.group_combo)
        self.group_field = self._field("Group 目錄", self.group_combo)
        layout.addWidget(self.group_field)

        self.group_note = QLabel("")
        self.group_note.setObjectName("note")
        self.group_note.setWordWrap(True)
        self.group_note.setMinimumWidth(1)
        layout.addWidget(self.group_note)

        self.preset_combo = QComboBox()
        for label, case_name in PRESET_CASES:
            self.preset_combo.addItem(label, case_name)
        self.preset_combo.currentIndexChanged.connect(self._sync_custom_from_preset)
        self._compact_combo(self.preset_combo)
        self.preset_field = self._field("常用 case", self.preset_combo)
        layout.addWidget(self.preset_field)

        self.custom_case_edit = QLineEdit(PRESET_CASES[0][1])
        self.custom_case_edit.setMinimumWidth(150)
        self.custom_field = self._field("Case（一個 exact case）", self.custom_case_edit)
        layout.addWidget(self.custom_field)

        caselist_row = QWidget()
        caselist_layout = QHBoxLayout(caselist_row)
        caselist_layout.setContentsMargins(0, 0, 0, 0)
        caselist_layout.setSpacing(6)
        self.caselist_edit = QLineEdit()
        self.caselist_edit.setMinimumWidth(120)
        caselist_button = QPushButton("…")
        caselist_button.setFixedWidth(34)
        caselist_button.clicked.connect(self._browse_caselist)
        caselist_layout.addWidget(self.caselist_edit, 1)
        caselist_layout.addWidget(caselist_button)
        self.caselist_field = self._field("Caselist 檔案", caselist_row)
        layout.addWidget(self.caselist_field)

        self.max_cases_spin = QSpinBox()
        self.max_cases_spin.setRange(0, 100_000)
        self.max_cases_spin.setValue(20)
        self.max_cases_spin.setSpecialValueText("全部")
        self.max_cases_field = self._field(
            "最多跑幾個 case", self.max_cases_spin, "0 = 跑完整個 group"
        )
        layout.addWidget(self.max_cases_field)

        self._update_group_note()
        return box

    def _path_field(
        self, caption: str, value: str, *, directory: bool, tooltip: str = ""
    ) -> tuple[QWidget, QLineEdit]:
        edit = QLineEdit(value)
        edit.setMinimumWidth(120)
        edit.setCursorPosition(0)
        if tooltip:
            edit.setToolTip(tooltip)
        button = QPushButton("…")
        button.setFixedWidth(34)

        def browse() -> None:
            start = edit.text().strip() or str(Path.home())
            if directory:
                chosen = QFileDialog.getExistingDirectory(self, caption, start)
            else:
                chosen, _ = QFileDialog.getOpenFileName(self, caption, start)
            if chosen:
                edit.setText(chosen)
                edit.setCursorPosition(0)

        button.clicked.connect(browse)
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)
        row_layout.addWidget(edit, 1)
        row_layout.addWidget(button)
        return self._field(caption, row, tooltip), edit

    def _build_paths_group(self) -> QGroupBox:
        box = QGroupBox("路徑與環境")
        layout = QVBoxLayout(box)
        layout.setSpacing(10)

        hint = QLabel("留空則沿用 config/local.env")
        hint.setObjectName("note")
        layout.addWidget(hint)

        field, self.mesa_edit = self._path_field(
            "PCO driver（Mesa prefix）",
            self.local_env.get("PVRGPU_MESA_PVRGPU_PREFIX", ""),
            directory=True,
            tooltip="含 gallium pvrgpu 驅動與 libEGL / libGLESv2 的 Mesa prefix",
        )
        layout.addWidget(field)

        field, self.bridge_edit = self._path_field(
            "SystemC bridge（.dylib）",
            self.local_env.get("PVRGPU_SYSTEMC_API_LIB", ""),
            directory=False,
            tooltip="libpvrgpu_systemc_bridge.dylib，由驅動在執行時 dlopen",
        )
        layout.addWidget(field)

        field, self.deqp_binary_edit = self._path_field(
            "dEQP binary（可留空）",
            "",
            directory=False,
            tooltip="留空則自動由 dEQP 專案的 out/deqp-build-<arch>.env 找 deqp-<module>",
        )
        layout.addWidget(field)

        field, self.deqp_build_edit = self._path_field(
            "dEQP build dir",
            self.local_env.get("PVRGPU_DEQP_BUILD_DIR", ""),
            directory=True,
            tooltip="dEQP 的 CMake build 目錄（modules/<module>/deqp-<module>）",
        )
        layout.addWidget(field)

        field, self.output_edit = self._path_field(
            "Output root",
            self.local_env.get("PVRGPU_OUTPUT_ROOT", str(REPO_ROOT / "outputs")),
            directory=True,
        )
        layout.addWidget(field)

        button_row = QHBoxLayout()
        button_row.setSpacing(8)
        self.check_button = QPushButton("Preflight")
        self.check_button.clicked.connect(self.start_check)
        self.open_output_button = QPushButton("開啟輸出資料夾")
        self.open_output_button.clicked.connect(self._open_output_root)
        button_row.addWidget(self.check_button)
        button_row.addWidget(self.open_output_button)
        layout.addLayout(button_row)

        self.check_label = QLabel("尚未驗證")
        self.check_label.setObjectName("note")
        self.check_label.setWordWrap(True)
        layout.addWidget(self.check_label)
        return box

    def _build_parameters_group(self) -> QGroupBox:
        box = QGroupBox("執行參數")
        layout = QVBoxLayout(box)
        layout.setSpacing(10)

        self.gl_config_combo = QComboBox()
        self.gl_config_combo.setEditable(True)
        self.gl_config_combo.addItems(
            ["rgba8888d24s8ms0", "rgba8888d24s8ms4", "rgb565d16s0ms0", ""]
        )
        layout.addWidget(self._field("GL config", self.gl_config_combo))

        self.surface_combo = QComboBox()
        self.surface_combo.addItems(["pbuffer", "fbo", "window"])
        self.size_combo = QComboBox()
        self.size_combo.setEditable(True)
        self.size_combo.addItems(["256x256", "128x128", "64x64", "512x512"])
        layout.addWidget(
            self._side_by_side(
                self._field("Surface", self.surface_combo),
                self._field("Size", self.size_combo),
            )
        )

        self.log_images_combo = QComboBox()
        self.log_images_combo.addItems(["disable", "enable"])
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(0, 86_400)
        self.timeout_spin.setSuffix(" s")
        self.timeout_spin.setSpecialValueText("無 timeout")
        layout.addWidget(
            self._side_by_side(
                self._field("Log images", self.log_images_combo),
                self._field("Per-case timeout", self.timeout_spin),
            )
        )

        self.verify_link_check = QCheckBox("--verify-link")
        self.verify_link_check.setToolTip(
            "用 DYLD_PRINT_LIBRARIES 確認真的走到你 build 的 PCO driver"
        )
        verify_note = QLabel("記錄實際載入的 libEGL / 驅動 / bridge")
        verify_note.setObjectName("note")
        verify_note.setWordWrap(True)

        self.keep_going_check = QCheckBox("--keep-going")
        self.keep_going_check.setChecked(True)
        keep_note = QLabel("某個 case 失敗後仍然跑完清單")
        keep_note.setObjectName("note")
        keep_note.setWordWrap(True)

        layout.addWidget(self.verify_link_check)
        layout.addWidget(verify_note)
        layout.addWidget(self.keep_going_check)
        layout.addWidget(keep_note)
        return box

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 12, 12, 12)
        layout.setSpacing(8)

        header = QFrame()
        header.setObjectName("header")
        header_layout = QGridLayout(header)
        header_layout.setContentsMargins(12, 8, 12, 8)

        self.current_case_label = QLabel("尚未執行")
        self.current_case_label.setObjectName("currentCase")
        header_layout.addWidget(self.current_case_label, 0, 0, 1, 5)

        self.copy_diagnostics_button = QPushButton("複製診斷資訊")
        self.copy_diagnostics_button.setToolTip(
            "把這次執行的設定、解析後的路徑、每個 case 的結果、失敗 case 的\n"
            "命令與 log 尾巴收成一份純文字，複製到剪貼簿,\n"
            "同時寫成 run 目錄裡的 diagnostics.txt。"
        )
        self.copy_diagnostics_button.clicked.connect(self.copy_diagnostics)
        header_layout.addWidget(self.copy_diagnostics_button, 0, 5)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        header_layout.addWidget(self.progress, 1, 0, 1, 6)

        self.counter_labels: dict[str, QLabel] = {}
        for column, (key, text) in enumerate(
            (
                ("total", "Total"),
                ("pass", "Pass"),
                ("fail", "Fail"),
                ("skip", "Skip"),
                ("warn", "Warn"),
                ("elapsed", "Elapsed"),
            )
        ):
            tile = QLabel(f"{text}\n0")
            tile.setObjectName(f"tile-{key}")
            tile.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tile.setProperty("tile", True)
            tile.setMinimumHeight(52)
            self.counter_labels[key] = tile
            header_layout.addWidget(tile, 2, column)
        layout.addWidget(header)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_status_tab(), "Run status")
        self.tabs.addTab(self._build_log_tab(), "Log")
        self.tabs.addTab(self._build_dashboard_tab(), "Dashboard")
        self.tabs.addTab(self._build_artifacts_tab(), "Artifacts")
        layout.addWidget(self.tabs, 1)
        return panel

    def _build_status_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 8, 0, 0)
        self.results_table = QTableWidget(0, 6)
        self.results_table.setHorizontalHeaderLabels(
            ["#", "Case", "Status", "Exit", "Duration", "Artifacts"]
        )
        self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.verticalHeader().setDefaultSectionSize(26)
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for column in range(2, 6):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.results_table.itemDoubleClicked.connect(self._open_case_item)
        layout.addWidget(self.results_table)
        return widget

    def _build_log_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 8, 0, 0)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(LOG_LINE_LIMIT)
        self.log_view.setFont(QFont("Menlo", 11))
        layout.addWidget(self.log_view)
        return widget

    def _build_dashboard_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 12, 4, 4)
        layout.setSpacing(12)

        self.dashboard_title = QLabel("跑完之後這裡會出現彙總")
        self.dashboard_title.setObjectName("dashboardTitle")
        layout.addWidget(self.dashboard_title)

        distribution = QGroupBox("結果分佈")
        distribution_layout = QGridLayout(distribution)
        self.distribution_bars: dict[str, tuple[QLabel, QProgressBar, QLabel]] = {}
        for row, (key, text) in enumerate(
            (("pass", "Pass"), ("fail", "Fail"), ("skip", "Skip"), ("warn", "Warning"))
        ):
            name = QLabel(text)
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setObjectName(f"bar-{key}")
            count = QLabel("0")
            count.setMinimumWidth(90)
            distribution_layout.addWidget(name, row, 0)
            distribution_layout.addWidget(bar, row, 1)
            distribution_layout.addWidget(count, row, 2)
            self.distribution_bars[key] = (name, bar, count)
        layout.addWidget(distribution)

        wiring = QGroupBox("這次實際串到的東西")
        wiring_layout = QVBoxLayout(wiring)
        self.wiring_table = QTableWidget(0, 2)
        self.wiring_table.setHorizontalHeaderLabels(["項目", "值"])
        self.wiring_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.wiring_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.wiring_table.verticalHeader().setVisible(False)
        self.wiring_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.wiring_table.setMaximumHeight(230)
        wiring_layout.addWidget(self.wiring_table)
        layout.addWidget(wiring)

        slowest = QGroupBox("最慢的 case")
        slowest_layout = QVBoxLayout(slowest)
        self.slowest_table = QTableWidget(0, 3)
        self.slowest_table.setHorizontalHeaderLabels(["Case", "Status", "Duration"])
        self.slowest_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        for column in (1, 2):
            self.slowest_table.horizontalHeader().setSectionResizeMode(
                column, QHeaderView.ResizeMode.ResizeToContents
            )
        self.slowest_table.verticalHeader().setVisible(False)
        self.slowest_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.slowest_table.setMaximumHeight(220)
        slowest_layout.addWidget(self.slowest_table)
        layout.addWidget(slowest)

        self.failure_label = QLabel("")
        self.failure_label.setObjectName("note")
        self.failure_label.setWordWrap(True)
        layout.addWidget(self.failure_label)

        button_row = QHBoxLayout()
        self.open_summary_button = QPushButton("開啟 summary.tsv")
        self.open_summary_button.clicked.connect(self._open_summary)
        self.open_run_button = QPushButton("開啟這次的輸出目錄")
        self.open_run_button.clicked.connect(self._open_run_dir)
        button_row.addWidget(self.open_summary_button)
        button_row.addWidget(self.open_run_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        return scroll

    def _build_artifacts_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 8, 0, 0)
        self.artifacts_table = QTableWidget(0, 3)
        self.artifacts_table.setHorizontalHeaderLabels(["檔案", "類型", "大小"])
        self.artifacts_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.artifacts_table.verticalHeader().setVisible(False)
        self.artifacts_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.artifacts_table.itemDoubleClicked.connect(self._open_artifact_item)
        layout.addWidget(self.artifacts_table)
        refresh = QPushButton("重新掃描")
        refresh.clicked.connect(self._refresh_artifacts)
        layout.addWidget(refresh)
        return widget

    # --------------------------------------------------------------------------
    # Selection helpers
    # --------------------------------------------------------------------------
    def _update_mode_visibility(self) -> None:
        mode = self.mode_combo.currentText()
        self.group_field.setVisible(mode == MODE_GROUP)
        self.group_note.setVisible(mode == MODE_GROUP)
        self.preset_field.setVisible(mode == MODE_PRESET)
        self.custom_field.setVisible(mode in (MODE_PRESET, MODE_CUSTOM))
        self.caselist_field.setVisible(mode == MODE_CASELIST)
        self.max_cases_field.setVisible(mode in (MODE_GROUP, MODE_CASELIST))

    def _update_group_note(self) -> None:
        if not GROUPS_AVAILABLE or self.group_combo.count() == 0:
            self.group_note.setText("tools/deqp_groups.py 不可用，group 模式已停用。")
            return
        spec = get_group(self.group_combo.currentData())
        if spec.available:
            note = (
                f"{spec.suite} · {spec.locked_case_count} cases（locked CTS）。"
                " 會先 discovery 展開，再一個 case 一個 process 跑。"
            )
            if spec.suite in ("dEQP-GLES3", "dEQP-GLES31"):
                note += (
                    " 注意：bridge 在 process 結束才模擬、結果不會回寫給 glReadPixels，"
                    "所以影像比對類的 case 目前一律回報 Fail；請看 case 目錄裡的"
                    " systemc PNG 判斷模型輸出。"
                )
            self.group_note.setText(note)
        else:
            self.group_note.setText(f"⛔ {spec.availability_reason}")

    def _sync_custom_from_preset(self) -> None:
        case_name = self.preset_combo.currentData()
        if case_name:
            self.custom_case_edit.setText(case_name)

    def _browse_caselist(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(
            self, "Caselist", self.caselist_edit.text() or str(REPO_ROOT)
        )
        if chosen:
            self.caselist_edit.setText(chosen)

    # --------------------------------------------------------------------------
    # Command assembly
    # --------------------------------------------------------------------------
    def _common_arguments(self) -> list[str]:
        arguments: list[str] = ["--emit-events"]
        if self.mesa_edit.text().strip():
            arguments += ["--mesa-prefix", self.mesa_edit.text().strip()]
        if self.bridge_edit.text().strip():
            arguments += ["--systemc-lib", self.bridge_edit.text().strip()]
        if self.deqp_binary_edit.text().strip():
            arguments += ["--deqp-binary", self.deqp_binary_edit.text().strip()]
        elif self.deqp_build_edit.text().strip():
            arguments += ["--deqp-build-dir", self.deqp_build_edit.text().strip()]
        return arguments

    def _run_arguments(self) -> list[str]:
        arguments = self._common_arguments()
        arguments += ["--surface-type", self.surface_combo.currentText()]
        size = self.size_combo.currentText().strip()
        if re.fullmatch(r"[1-9][0-9]*x[1-9][0-9]*", size):
            arguments += ["--size", size]
        arguments += ["--gl-config", self.gl_config_combo.currentText().strip()]
        arguments += ["--log-images", self.log_images_combo.currentText()]
        if self.timeout_spin.value() > 0:
            arguments += ["--timeout", str(self.timeout_spin.value())]
        if self.verify_link_check.isChecked():
            arguments.append("--verify-link")
        if self.keep_going_check.isChecked():
            arguments.append("--keep-going")
        return arguments

    def _new_run_dir(self) -> Path:
        base = Path(
            self.output_edit.text().strip()
            or self.local_env.get("PVRGPU_OUTPUT_ROOT", str(REPO_ROOT / "outputs"))
        ).expanduser()
        run_dir = base / "deqp_dynamic" / datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _selected_case(self) -> str:
        return self.custom_case_edit.text().strip()

    # --------------------------------------------------------------------------
    # Process control
    # --------------------------------------------------------------------------
    def _start_process(self, arguments: list[str], phase: str) -> None:
        if self.process is not None:
            QMessageBox.warning(self, "PvrGPU", "已經有一個執行中的工作。")
            return
        if not RUNNER_SCRIPT.is_file():
            QMessageBox.critical(
                self, "PvrGPU", f"找不到執行腳本：\n{RUNNER_SCRIPT}"
            )
            return

        self.phase = phase
        process = QProcess(self)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        process.setProcessEnvironment(environment)
        process.setWorkingDirectory(str(REPO_ROOT))
        process.readyReadStandardOutput.connect(self._read_process_output)
        process.finished.connect(self._process_finished)
        process.setProgram(str(RUNNER_SCRIPT))
        process.setArguments(arguments)
        self.process = process

        self._append_log(f"$ {RUNNER_SCRIPT.name} {' '.join(arguments)}")
        self.run_button.setEnabled(False)
        self.check_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        process.start()

    def start_check(self) -> None:
        self.log_view.clear()
        self.log_lines = 0
        arguments = self._common_arguments() + ["--check", "--print-env"]
        mode = self.mode_combo.currentText()
        if mode == MODE_GROUP and GROUPS_AVAILABLE:
            spec = get_group(self.group_combo.currentData())
            arguments += ["--module", SUITE_TO_MODULE[spec.suite]]
        elif mode == MODE_CASELIST:
            if self.caselist_edit.text().strip():
                arguments += ["--caselist", self.caselist_edit.text().strip()]
        else:
            case_name = self._selected_case()
            if case_name:
                arguments += ["--case", case_name]
        self.check_label.setText("驗證中 …")
        self.tabs.setCurrentIndex(1)
        self._start_process(arguments, "check")

    def start_run(self) -> None:
        mode = self.mode_combo.currentText()
        self.run_dir = self._new_run_dir()
        self._reset_run_view()

        if mode == MODE_GROUP:
            if not GROUPS_AVAILABLE:
                QMessageBox.warning(self, "PvrGPU", "group 目錄不可用。")
                return
            spec = get_group(self.group_combo.currentData())
            if not spec.available:
                answer = QMessageBox.question(
                    self,
                    "PvrGPU",
                    f"{spec.label} 目前是 blocked：\n\n{spec.availability_reason}\n\n仍要嘗試嗎？",
                )
                if answer != QMessageBox.StandardButton.Yes:
                    return
            self.current_group = spec
            self.caselist_path = self.run_dir / "caselist.txt"
            arguments = self._common_arguments() + [
                "--module", SUITE_TO_MODULE[spec.suite],
                "--discover",
                "--caselist-out", str(self.run_dir / "discovered.txt"),
                "--output-dir", str(self.run_dir),
            ]
            self.current_case_label.setText(f"Discovering {spec.label} …")
            self.statusBar().showMessage("Discovering cases")
            self._start_process(arguments, "discover")
            return

        if mode == MODE_CASELIST:
            caselist = self.caselist_edit.text().strip()
            if not caselist or not Path(caselist).is_file():
                QMessageBox.warning(self, "PvrGPU", "請先選一個 caselist 檔案。")
                return
            names = [
                line.split("#", 1)[0].strip()
                for line in Path(caselist).read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
            ]
            names = [name for name in names if name]
            self._launch_cases(names)
            return

        case_name = self._selected_case()
        if not EXACT_CASE_RE.match(case_name):
            QMessageBox.warning(
                self, "PvrGPU", f"不是一個合法的 exact case：\n{case_name}"
            )
            return
        self._launch_cases([case_name])

    def _launch_cases(self, names: list[str]) -> None:
        limit = self.max_cases_spin.value()
        if limit > 0 and self.mode_combo.currentText() in (MODE_GROUP, MODE_CASELIST):
            names = names[:limit]
        names = [name for name in names if EXACT_CASE_RE.match(name)]
        if not names:
            QMessageBox.warning(self, "PvrGPU", "沒有可執行的 exact case。")
            self._finish_idle()
            return

        assert self.run_dir is not None
        self.caselist_path = self.run_dir / "caselist.txt"
        self.caselist_path.write_text("\n".join(names) + "\n", encoding="utf-8")

        self.state = RunState(total=len(names), started_at=datetime.now())
        self.progress.setRange(0, len(names))
        self.progress.setValue(0)
        self.elapsed_timer.start()
        self.tabs.setCurrentIndex(0)

        arguments = self._run_arguments() + [
            "--caselist", str(self.caselist_path),
            "--output-dir", str(self.run_dir),
        ]
        self.statusBar().showMessage(f"Running {len(names)} case(s)")
        self._start_process(arguments, "run")

    def cancel_run(self) -> None:
        if self.process is None:
            return
        self._append_log("--- cancel requested ---")
        self.process.kill()

    # --------------------------------------------------------------------------
    # Process output
    # --------------------------------------------------------------------------
    def _read_process_output(self) -> None:
        if self.process is None:
            return
        chunk = bytes(self.process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        self.stdout_buffer += chunk
        while "\n" in self.stdout_buffer:
            line, self.stdout_buffer = self.stdout_buffer.split("\n", 1)
            self._handle_line(line.rstrip("\r"))

    def _handle_line(self, line: str) -> None:
        if line.startswith(EVENT_PREFIX):
            try:
                payload = json.loads(line[len(EVENT_PREFIX) :])
            except json.JSONDecodeError:
                self._append_log(line)
                return
            self._handle_event(payload)
            return
        self._append_log(line)

    def _handle_event(self, payload: dict) -> None:
        event = payload.get("event", "")
        if event == "resolved":
            self.state.module = payload.get("module", "")
            self.state.runner = payload.get("runner", "")
            self.state.archive_dir = payload.get("archive_dir", "")
            self.state.mesa_prefix = payload.get("mesa_prefix", "")
            self.state.systemc_lib = payload.get("systemc_lib", "")
            self.state.host_arch = payload.get("host_arch", "")
            output_root = payload.get("output_root", "")
            if output_root:
                self.state.output_root = Path(output_root)
            self._refresh_wiring_table()
        elif event == "check_ok":
            self.check_label.setText("✅ Preflight 通過：三個產物都在，架構一致。")
        elif event == "discovery_started":
            self._append_log(f"--- discovery: {payload.get('suite','')} ---")
        elif event == "discovery_finished":
            self._append_log(
                f"--- discovered {payload.get('discovered', 0)} case(s) ---"
            )
        elif event == "run_start":
            self.state.total = int(payload.get("total", 0))
            summary = payload.get("summary", "")
            self.state.summary_path = Path(summary) if summary else None
            self.progress.setRange(0, max(1, self.state.total))
        elif event == "case_start":
            self._on_case_start(payload)
        elif event == "case_end":
            self._on_case_end(payload)
        elif event == "run_end":
            self.state.duration_ms = int(payload.get("duration_ms", 0))

    def _on_case_start(self, payload: dict) -> None:
        row = CaseRow(
            index=int(payload.get("index", len(self.state.rows) + 1)),
            case_name=payload.get("case", ""),
            case_dir=Path(payload["case_dir"]) if payload.get("case_dir") else None,
        )
        self.state.rows.append(row)
        self.current_case_label.setText(
            f"[{row.index}/{payload.get('total', self.state.total)}] {row.case_name}"
        )
        self._append_result_row(row)

    def _on_case_end(self, payload: dict) -> None:
        case_name = payload.get("case", "")
        row = next(
            (item for item in reversed(self.state.rows) if item.case_name == case_name),
            None,
        )
        if row is None:
            row = CaseRow(index=len(self.state.rows) + 1, case_name=case_name)
            self.state.rows.append(row)
            self._append_result_row(row)
        row.status = payload.get("status", "Unknown")
        row.exit_code = int(payload.get("exit_code", 0))
        row.duration_ms = int(payload.get("duration_ms", 0))
        for key in ("case_dir", "qpa", "log"):
            value = payload.get(key)
            if value:
                setattr(row, key, Path(value))
        self._update_result_row(row)
        self.progress.setValue(min(row.index, self.progress.maximum()))
        self._refresh_counters()

    # --------------------------------------------------------------------------
    # Views
    # --------------------------------------------------------------------------
    def _reset_run_view(self) -> None:
        self.state = RunState()
        self.results_table.setRowCount(0)
        self.log_view.clear()
        self.log_lines = 0
        self.progress.setValue(0)
        self.current_case_label.setText("準備中 …")
        self.failure_label.setText("")
        self._refresh_counters()

    def _append_log(self, text: str) -> None:
        if not text:
            return
        self.log_view.appendPlainText(text)
        self.log_lines += 1

    def _append_result_row(self, row: CaseRow) -> None:
        index = self.results_table.rowCount()
        self.results_table.insertRow(index)
        for column, value in enumerate(
            (str(row.index), row.case_name, row.status, "", "", "")
        ):
            item = QTableWidgetItem(value)
            if column == 1:
                item.setData(Qt.ItemDataRole.UserRole, str(row.case_dir or ""))
            self.results_table.setItem(index, column, item)
        self.results_table.scrollToBottom()

    def _row_position(self, row: CaseRow) -> int | None:
        for position in range(self.results_table.rowCount() - 1, -1, -1):
            item = self.results_table.item(position, 1)
            if item is not None and item.text() == row.case_name:
                return position
        return None

    def _update_result_row(self, row: CaseRow) -> None:
        position = self._row_position(row)
        if position is None:
            return
        values = (
            str(row.index),
            row.case_name,
            row.status,
            "-" if row.exit_code is None else str(row.exit_code),
            f"{row.duration_ms} ms",
            row.artifact_summary(),
        )
        for column, value in enumerate(values):
            item = self.results_table.item(position, column)
            if item is None:
                item = QTableWidgetItem()
                self.results_table.setItem(position, column, item)
            item.setText(value)
            if column == 1:
                item.setData(Qt.ItemDataRole.UserRole, str(row.case_dir or ""))
        status_item = self.results_table.item(position, 2)
        colors = {
            "pass": QColor(self.theme.ok),
            "fail": QColor(self.theme.bad),
            "skip": QColor(self.theme.skip),
            "warn": QColor(self.theme.warn),
        }
        status_item.setForeground(colors.get(row.bucket, QColor(self.theme.text)))

    def _refresh_counters(self) -> None:
        buckets = {"pass": 0, "fail": 0, "skip": 0, "warn": 0}
        for row in self.state.rows:
            if row.exit_code is None:
                continue
            buckets[row.bucket] += 1
        self.counter_labels["total"].setText(f"Total\n{self.state.total}")
        for key in buckets:
            self.counter_labels[key].setText(f"{key.capitalize()}\n{buckets[key]}")
        self._tick_elapsed()

    def _tick_elapsed(self) -> None:
        if self.state.started_at is None:
            self.counter_labels["elapsed"].setText("Elapsed\n-")
            return
        elapsed = (datetime.now() - self.state.started_at).total_seconds()
        self.counter_labels["elapsed"].setText(
            f"Elapsed\n{format_duration(int(elapsed * 1000))}"
        )

    def _refresh_wiring_table(self) -> None:
        entries = [
            ("Module", self.state.module),
            ("dEQP runner", self.state.runner),
            ("Archive dir", self.state.archive_dir),
            ("PCO driver (Mesa)", self.state.mesa_prefix),
            ("SystemC bridge", self.state.systemc_lib),
            ("Host arch", self.state.host_arch),
            ("Output", str(self.state.output_root or self.run_dir or "")),
        ]
        self.wiring_table.setRowCount(len(entries))
        for row, (name, value) in enumerate(entries):
            self.wiring_table.setItem(row, 0, QTableWidgetItem(name))
            self.wiring_table.setItem(row, 1, QTableWidgetItem(value))

    def _refresh_dashboard(self) -> None:
        finished = [row for row in self.state.rows if row.exit_code is not None]
        total = len(finished)
        buckets = {"pass": 0, "fail": 0, "skip": 0, "warn": 0}
        for row in finished:
            buckets[row.bucket] += 1

        stamp = (
            self.state.started_at.strftime("%Y-%m-%d %H:%M:%S")
            if self.state.started_at
            else "-"
        )
        duration = (
            format_duration(self.state.duration_ms)
            if self.state.duration_ms
            else "-"
        )
        self.dashboard_title.setText(
            f"{stamp} · {self.state.module or '-'} · {total}/{self.state.total} cases"
            f" · {duration}"
        )

        for key, (_, bar, count) in self.distribution_bars.items():
            value = buckets[key]
            percent = int(round(100.0 * value / total)) if total else 0
            bar.setValue(percent)
            count.setText(f"{value}  ({percent}%)")

        slowest = sorted(finished, key=lambda item: item.duration_ms, reverse=True)[:10]
        self.slowest_table.setRowCount(len(slowest))
        for row, item in enumerate(slowest):
            self.slowest_table.setItem(row, 0, QTableWidgetItem(item.case_name))
            self.slowest_table.setItem(row, 1, QTableWidgetItem(item.status))
            self.slowest_table.setItem(
                row, 2, QTableWidgetItem(format_duration(item.duration_ms))
            )

        failures = [row for row in finished if row.bucket == "fail"]
        if failures:
            listed = "\n".join(
                f"  · {item.case_name} → {item.status} (exit {item.exit_code})"
                for item in failures[:12]
            )
            more = "" if len(failures) <= 12 else f"\n  … 另外 {len(failures) - 12} 個"
            self.failure_label.setText(f"未通過的 case：\n{listed}{more}")
        else:
            self.failure_label.setText("沒有未通過的 case。")

        self._refresh_wiring_table()
        self._refresh_artifacts()

    def _refresh_artifacts(self) -> None:
        root = self.state.output_root or self.run_dir
        self.artifacts_table.setRowCount(0)
        if root is None or not Path(root).is_dir():
            return
        root = Path(root)
        rows = 0
        for path in sorted(root.rglob("*")):
            if rows >= ARTIFACT_ROW_LIMIT:
                break
            if not path.is_file():
                continue
            self.artifacts_table.insertRow(rows)
            name_item = QTableWidgetItem(str(path.relative_to(root)))
            name_item.setData(Qt.ItemDataRole.UserRole, str(path))
            self.artifacts_table.setItem(rows, 0, name_item)
            self.artifacts_table.setItem(
                rows, 1, QTableWidgetItem(path.suffix.lstrip(".") or "file")
            )
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            self.artifacts_table.setItem(rows, 2, QTableWidgetItem(format_bytes(size)))
            rows += 1

    # --------------------------------------------------------------------------
    # Completion
    # --------------------------------------------------------------------------
    def _process_finished(self, exit_code: int, _status) -> None:
        if self.stdout_buffer:
            self._handle_line(self.stdout_buffer)
            self.stdout_buffer = ""
        phase = self.phase
        self.last_phase = phase
        self.last_exit_code = exit_code
        self.process = None
        self.phase = "idle"

        if phase == "check":
            if exit_code == 0:
                self.check_label.setText("✅ Preflight 通過：三個產物都在，架構一致。")
            else:
                self.check_label.setText(
                    "❌ Preflight 失敗，詳見 Log 分頁最後幾行。"
                )
            self._finish_idle()
            return

        if phase == "discover":
            self._finish_idle()
            if exit_code != 0:
                QMessageBox.critical(
                    self, "PvrGPU", "Case discovery 失敗，詳見 Log 分頁。"
                )
                return
            assert self.run_dir is not None
            discovered_path = self.run_dir / "discovered.txt"
            if not discovered_path.is_file():
                QMessageBox.critical(self, "PvrGPU", "Discovery 沒有產生 case 清單。")
                return
            discovered = [
                line.strip()
                for line in discovered_path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
                if line.strip()
            ]
            selected = list(filter_exact_cases(self.current_group, discovered))
            self._append_log(
                f"--- {len(discovered)} discovered, {len(selected)} in group ---"
            )
            if not selected:
                QMessageBox.warning(
                    self, "PvrGPU", "這個 group 在目前的 CTS 裡沒有對應的 case。"
                )
                return
            self._launch_cases(selected)
            return

        self.elapsed_timer.stop()
        self._refresh_counters()
        self._refresh_dashboard()
        self.tabs.setCurrentIndex(2)
        finished = len([row for row in self.state.rows if row.exit_code is not None])
        failures = len(
            [
                row
                for row in self.state.rows
                if row.exit_code is not None and row.bucket == "fail"
            ]
        )
        self.current_case_label.setText(
            f"完成：{finished}/{self.state.total} cases，{failures} 個未通過"
        )
        self.statusBar().showMessage(
            f"Run finished (exit {exit_code}) · {self.state.output_root or self.run_dir}"
        )
        self._finish_idle()

    def _finish_idle(self) -> None:
        self.run_button.setEnabled(True)
        self.check_button.setEnabled(True)
        self.cancel_button.setEnabled(False)

    # --------------------------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------------------------
    @staticmethod
    def _tail(text: str, lines: int) -> str:
        """The last `lines` lines, marked when something was dropped."""
        rows = text.splitlines()
        if len(rows) <= lines:
            return "\n".join(rows)
        return "\n".join([f"... ({len(rows) - lines} earlier lines omitted)"] + rows[-lines:])

    @staticmethod
    def _read_tail(path: Path, lines: int, byte_cap: int = 262144) -> str:
        """Read the end of a file without pulling a huge log into memory."""
        try:
            size = path.stat().st_size
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                if size > byte_cap:
                    handle.seek(size - byte_cap)
                    handle.readline()
                return MainWindow._tail(handle.read(), lines)
        except OSError as error:
            return f"<unreadable: {error}>"

    def _case_detail(self, row: CaseRow) -> list[str]:
        """Everything about one case that a reader would otherwise have to
        open four files to find."""
        out = [f"-- [{row.index}] {row.case_name}",
               f"   status={row.status} exit={row.exit_code} "
               f"duration={format_duration(row.duration_ms)}"]
        if row.case_dir is None or not row.case_dir.is_dir():
            out.append("   case dir: <missing>")
            return out
        out.append(f"   case dir: {row.case_dir}")
        command = row.case_dir / "command.txt"
        if not command.is_file():
            command = row.case_dir / "driver-command.txt"
        if command.is_file():
            out.append("   command:")
            out += [f"     {line}" for line in
                    self._read_tail(command, 12).splitlines()]
        jsonl = row.case_dir / "systemc.jsonl"
        if jsonl.is_file():
            interesting: list[str] = []
            for line in self._read_tail(jsonl, 400).splitlines():
                if any(mark in line for mark in
                       ('"type":"done"', '"type":"error"', '"error"', '"warning"')):
                    interesting.append(line[:600])
            if interesting:
                out.append("   systemc.jsonl (done/error lines):")
                out += [f"     {line}" for line in interesting[-6:]]
            else:
                out.append("   systemc.jsonl: no done/error line")
        for name in ("stderr.log", "stdout.log", "case.log", "run.log"):
            candidate = row.case_dir / name
            if candidate.is_file() and candidate.stat().st_size:
                out.append(f"   {name} (tail):")
                out += [f"     {line}" for line in
                        self._read_tail(candidate, 25).splitlines()]
        pngs = sorted((row.case_dir / "systemc").glob("*.png"))
        if pngs:
            out.append(f"   systemc png: {len(pngs)} file(s), e.g. {pngs[0].name}")
        return out

    def build_diagnostics(self) -> str:
        """One self-contained plain-text report: what was asked for, what the
        wiring resolved to, what every case did, and the detail behind each
        failure.  Written so it can be pasted somewhere and read on its own."""
        now = datetime.now().astimezone()
        lines: list[str] = []
        add = lines.append

        add("=== PvrGPU dEQP · Dynamic Link — diagnostics ===")
        add(f"generated : {now.isoformat(timespec='seconds')}")
        add(f"ui        : {Path(__file__).resolve()}")
        add(f"host      : {platform.platform()} ({platform.machine()})")
        add(f"python    : {sys.version.split()[0]}  Qt/PySide6: "
            f"{getattr(__import__('PySide6'), '__version__', '?')}")
        add(f"repo      : {REPO_ROOT}")
        add("")

        mode = self.mode_combo.currentText()
        add("[selection]")
        add(f"mode          : {mode}")
        # Only the fields the active mode actually uses: the other combos keep
        # their last value and would read as though they were in play.
        if mode == MODE_GROUP and hasattr(self, "group_combo"):
            add(f"group         : {self.group_combo.currentText()}")
        elif mode == MODE_PRESET and hasattr(self, "preset_combo"):
            add(f"preset        : {self.preset_combo.currentText()}")
        elif mode == MODE_CUSTOM:
            add(f"custom case   : {self._selected_case() or '<blank>'}")
        if mode in (MODE_GROUP, MODE_CASELIST) and hasattr(self, "max_cases_spin"):
            limit = self.max_cases_spin.value()
            add(f"case limit    : {limit if limit else 'all'}")
        if getattr(self, "caselist_path", None):
            add(f"caselist      : {self.caselist_path}")
        add("")

        add("[paths as typed in the UI]  (blank = taken from config/local.env)")
        for caption, widget in (
            ("PCO driver (Mesa prefix)", getattr(self, "mesa_edit", None)),
            ("SystemC bridge", getattr(self, "bridge_edit", None)),
            ("dEQP binary", getattr(self, "deqp_binary_edit", None)),
            ("dEQP build dir", getattr(self, "deqp_build_edit", None)),
            ("Output root", getattr(self, "output_edit", None)),
        ):
            if widget is not None:
                add(f"{caption:<26}: {widget.text().strip() or '<blank>'}")
        add("")

        add("[wiring the runner reported]")
        for caption, value in (
            ("Module", self.state.module),
            ("dEQP runner", self.state.runner),
            ("Archive dir", self.state.archive_dir),
            ("PCO driver (Mesa)", self.state.mesa_prefix),
            ("SystemC bridge", self.state.systemc_lib),
            ("Host arch", self.state.host_arch),
        ):
            add(f"{caption:<26}: {value or '<not reported>'}")
        add("")

        add("[run parameters]")
        add(f"arguments     : run_deqp_dynamic.sh {' '.join(self._run_arguments())}")
        add(f"run dir       : {self.run_dir or '<none>'}")
        add(f"output root   : {self.state.output_root or '<none>'}")
        add(f"summary       : {self.state.summary_path or '<none>'}")
        started = self.state.started_at
        add(f"started       : {started.isoformat(timespec='seconds') if started else '<none>'}")
        add(f"elapsed       : {format_duration(self.state.duration_ms)}")
        add(f"last phase    : {self.last_phase or '<none>'}  "
            f"exit={self.last_exit_code}")
        add("")

        finished = [row for row in self.state.rows if row.exit_code is not None]
        buckets = {"pass": 0, "fail": 0, "skip": 0, "warn": 0}
        for row in finished:
            buckets[row.bucket] += 1
        add("[totals]")
        add(f"total={self.state.total} finished={len(finished)} "
            f"pass={buckets['pass']} fail={buckets['fail']} "
            f"skip={buckets['skip']} warn={buckets['warn']}")
        add("")

        add("[cases]")
        add(f"{'#':>3}  {'status':<13} {'exit':>4} {'duration':>9}  "
            f"{'artifacts':<24} case")
        for row in self.state.rows:
            add(f"{row.index:>3}  {row.status:<13} "
                f"{('' if row.exit_code is None else row.exit_code):>4} "
                f"{format_duration(row.duration_ms):>9}  "
                f"{row.artifact_summary():<24} {row.case_name}")
        add("")

        failing = [row for row in self.state.rows if row.bucket in ("fail", "warn")]
        add(f"[failing cases in detail]  ({len(failing)} of {len(self.state.rows)};"
            f" first {min(len(failing), DIAGNOSTICS_MAX_CASES)} shown)")
        if not failing:
            add("(none)")
        for row in failing[:DIAGNOSTICS_MAX_CASES]:
            lines.extend(self._case_detail(row))
            add("")

        add("[UI log tail]")
        add(self._tail(self.log_view.toPlainText(), DIAGNOSTICS_LOG_LINES))
        add("")
        add("=== end of diagnostics ===")
        return "\n".join(lines)

    def copy_diagnostics(self) -> None:
        """Copy the report to the clipboard, and drop the same text in the run
        directory so it can be read from disk instead of pasted."""
        try:
            report = self.build_diagnostics()
        except Exception as error:  # noqa: BLE001 - a diagnostic must not crash the UI
            QMessageBox.critical(self, "PvrGPU", f"產生診斷資訊失敗：{error}")
            return

        saved: Path | None = None
        target_dir = self.run_dir or self.state.output_root
        if target_dir:
            candidate = Path(target_dir) / "diagnostics.txt"
            try:
                candidate.parent.mkdir(parents=True, exist_ok=True)
                candidate.write_text(report, encoding="utf-8")
                saved = candidate
            except OSError:
                saved = None

        if saved is not None:
            report = f"diagnostics file: {saved}\n\n{report}"
        QApplication.clipboard().setText(report)

        size = len(report.encode("utf-8"))
        message = f"診斷資訊已複製（{format_bytes(size)}）"
        if saved is not None:
            message += f" · 也寫到 {saved}"
        self.statusBar().showMessage(message, 8000)

    # --------------------------------------------------------------------------
    # Opening things
    # --------------------------------------------------------------------------
    def _open_path(self, path: Path | None) -> None:
        if path is None or not Path(path).exists():
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _open_output_root(self) -> None:
        self._open_path(Path(self.output_edit.text().strip() or REPO_ROOT / "outputs"))

    def _open_run_dir(self) -> None:
        self._open_path(self.state.output_root or self.run_dir)

    def _open_summary(self) -> None:
        self._open_path(self.state.summary_path)

    def _open_case_item(self, item: QTableWidgetItem) -> None:
        row = item.row()
        name_item = self.results_table.item(row, 1)
        if name_item is None:
            return
        directory = name_item.data(Qt.ItemDataRole.UserRole)
        if directory:
            self._open_path(Path(directory))

    def _open_artifact_item(self, item: QTableWidgetItem) -> None:
        path = self.artifacts_table.item(item.row(), 0)
        if path is None:
            return
        value = path.data(Qt.ItemDataRole.UserRole)
        if value:
            self._open_path(Path(value))

    # --------------------------------------------------------------------------
    # Settings and style
    # --------------------------------------------------------------------------
    def _restore_settings(self) -> None:
        mapping = {
            "mesa": self.mesa_edit,
            "bridge": self.bridge_edit,
            "deqp_binary": self.deqp_binary_edit,
            "deqp_build": self.deqp_build_edit,
            "output": self.output_edit,
            "caselist": self.caselist_edit,
        }
        for key, widget in mapping.items():
            stored = self.settings.value(f"paths/{key}", "", str)
            if stored:
                widget.setText(stored)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self.process is not None:
            answer = QMessageBox.question(
                self,
                "PvrGPU",
                "還有一個工作正在執行，要中止並關閉嗎？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        mapping = {
            "mesa": self.mesa_edit,
            "bridge": self.bridge_edit,
            "deqp_binary": self.deqp_binary_edit,
            "deqp_build": self.deqp_build_edit,
            "output": self.output_edit,
            "caselist": self.caselist_edit,
        }
        for key, widget in mapping.items():
            self.settings.setValue(f"paths/{key}", widget.text())
        if self.process is not None:
            self.process.kill()
            self.process.waitForFinished(2000)
        super().closeEvent(event)

    def _apply_style(self) -> None:
        theme = self.theme
        self.setStyleSheet(
            f"""
            QWidget {{ color: {theme.text}; font-size: 13px; }}
            QMainWindow, QScrollArea#sideScroll, QWidget#sidePanel,
            QWidget#sideActions {{
                background: {theme.window};
            }}
            QWidget#sideActions {{ border-top: 1px solid {theme.border}; }}
            QGroupBox {{
                border: 1px solid {theme.border};
                border-radius: 10px;
                margin-top: 16px;
                padding: 14px 12px 12px 12px;
                background: {theme.surface};
                font-weight: 600;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 12px;
                padding: 0 6px;
                background: {theme.surface};
                color: {theme.muted};
            }}
            QLabel#caption {{ color: {theme.muted}; font-size: 12px; }}
            QLabel#note {{ color: {theme.muted}; font-size: 12px; }}
            QFrame#header {{
                background: {theme.surface};
                border: 1px solid {theme.border};
                border-radius: 10px;
            }}
            QLabel#currentCase, QLabel#dashboardTitle {{
                font-size: 15px; font-weight: 600; color: {theme.text};
            }}
            QLabel[tile="true"] {{
                background: {theme.subtle};
                border: 1px solid {theme.border};
                border-radius: 8px;
                padding: 8px 4px;
                font-weight: 600;
                color: {theme.text};
            }}
            QLabel#tile-pass {{ background: {theme.ok_bg}; color: {theme.ok}; }}
            QLabel#tile-fail {{ background: {theme.bad_bg}; color: {theme.bad}; }}
            QLabel#tile-warn {{ background: {theme.warn_bg}; color: {theme.warn}; }}
            QLineEdit, QComboBox, QSpinBox, QAbstractSpinBox {{
                background: {theme.field};
                border: 1px solid {theme.border};
                border-radius: 6px;
                padding: 5px 8px;
                min-height: 20px;
                color: {theme.text};
                selection-background-color: {theme.accent};
                selection-color: {theme.accent_text};
            }}
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus {{
                border-color: {theme.accent};
            }}
            QComboBox::drop-down {{ border: none; width: 20px; }}
            QComboBox QAbstractItemView {{
                background: {theme.surface};
                color: {theme.text};
                border: 1px solid {theme.border};
                selection-background-color: {theme.accent};
                selection-color: {theme.accent_text};
            }}
            QCheckBox {{ color: {theme.text}; spacing: 7px; }}
            QPushButton {{
                padding: 6px 12px;
                border: 1px solid {theme.border};
                border-radius: 7px;
                background: {theme.surface};
                color: {theme.text};
            }}
            QPushButton:hover {{ background: {theme.subtle}; }}
            QPushButton#primary {{
                background: {theme.accent};
                border-color: {theme.accent};
                color: {theme.accent_text};
                font-weight: 600;
            }}
            QPushButton#quit {{ color: {theme.muted}; }}
            QPushButton#quit:hover {{
                color: {theme.bad}; border-color: {theme.bad};
            }}
            QPushButton:disabled {{ color: {theme.muted}; background: {theme.subtle}; }}
            QTableWidget {{
                background: {theme.surface};
                alternate-background-color: {theme.subtle};
                border: 1px solid {theme.border};
                border-radius: 8px;
                gridline-color: {theme.border};
                color: {theme.text};
            }}
            QHeaderView::section {{
                background: {theme.subtle};
                border: none;
                border-bottom: 1px solid {theme.border};
                padding: 6px;
                font-weight: 600;
                color: {theme.muted};
            }}
            QTabWidget::pane {{
                border: 1px solid {theme.border};
                background: {theme.surface};
                border-radius: 8px;
            }}
            QTabBar::tab {{
                padding: 6px 14px;
                margin-right: 2px;
                border: 1px solid transparent;
                border-radius: 7px;
                color: {theme.muted};
            }}
            QTabBar::tab:selected {{
                background: {theme.accent};
                color: {theme.accent_text};
                font-weight: 600;
            }}
            QProgressBar {{
                border: 1px solid {theme.border};
                border-radius: 7px;
                background: {theme.subtle};
                height: 18px;
                text-align: center;
                color: {theme.text};
            }}
            QProgressBar::chunk {{ background: {theme.accent}; border-radius: 6px; }}
            QProgressBar#bar-pass::chunk {{ background: {theme.ok}; }}
            QProgressBar#bar-fail::chunk {{ background: {theme.bad}; }}
            QProgressBar#bar-skip::chunk {{ background: {theme.skip}; }}
            QProgressBar#bar-warn::chunk {{ background: {theme.warn}; }}
            QPlainTextEdit {{
                background: {theme.log_bg};
                color: {theme.log_text};
                border: 1px solid {theme.border};
                border-radius: 8px;
            }}
            QSplitter::handle {{ background: {theme.border}; width: 1px; }}
            QStatusBar {{ color: {theme.muted}; }}
            QMenuBar {{ background: {theme.window}; color: {theme.text}; }}
            QMenuBar::item:selected {{
                background: {theme.accent}; color: {theme.accent_text};
            }}
            QMenu {{
                background: {theme.surface}; color: {theme.text};
                border: 1px solid {theme.border};
            }}
            QMenu::item:selected {{
                background: {theme.accent}; color: {theme.accent_text};
            }}
            """
        )


def main() -> int:
    application = QApplication(sys.argv)
    application.setApplicationName("PvrGPU dEQP Dynamic Link")

    system_theme = detect_theme(application)
    # Light is the default look; "跟隨系統" is one menu click away.
    preference = QSettings("PvrGPU", "deqp-dynamic-ui").value(
        "appearance/theme", DEFAULT_APPEARANCE, str
    )
    if preference not in ("system", "light", "dark"):
        preference = DEFAULT_APPEARANCE
    theme = resolve_theme(preference, system_theme)

    application.setStyle("Fusion")
    apply_palette(application, theme)
    window = MainWindow(theme, system_theme=system_theme, appearance=preference)
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
