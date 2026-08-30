#!/usr/bin/env python3
"""Qt control panel for GLBench -> Mesa -> llvmpipe/PvrGPU.

The llvmpipe path reads the counter-enabled Mesa Report.md. Until the native
PvrGPU Mesa driver exists, completed GLBench raster gates can run through the
functional, event-driven SystemC model and produce modeled counters plus PNG
frames.
"""

from __future__ import annotations

from datetime import datetime
import math
import os
from pathlib import Path
import re
import sys
from typing import Iterable

from PySide6.QtCore import QProcess, QProcessEnvironment, QSettings, Qt, QTimer, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QFont, QPainter, QPen, QPixmap, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from counter_protocol import (
    ALL_COUNTER_FIELDS,
    COUNTER_INFO,
    CounterProtocolError,
    CounterRecord,
    counter_record_from_message,
    human_number,
    parse_jsonl_line,
    parse_markdown_report,
)


PROJECT_ROOT = Path(os.environ.get("PVRGPU_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
WORK_ROOT = Path(
    os.environ.get(
        "PVRGPU_WORK_ROOT",
        str(Path.home() / "Downloads" / "_Codex" / "Working" / "PvrGPU"),
    )
).expanduser().resolve()

GLBENCH_CASES: tuple[tuple[str, str, str], ...] = (
    ("Fill · Solid", "fill_rate", "fill_solid"),
    ("Fill · Solid blended", "fill_rate", "fill_solid_blended"),
    ("Fill · Depth not equal", "fill_rate", "fill_solid_depth_neq"),
    ("Fill · Depth never", "fill_rate", "fill_solid_depth_never"),
    ("Fill · Texture nearest", "fill_rate", "fill_tex_nearest"),
    ("Fill · Texture bilinear", "fill_rate", "fill_tex_bilinear"),
    ("Fill · Trilinear 0.5", "fill_rate", "fill_tex_trilinear_linear_05"),
    ("Fill · Trilinear 0.4", "fill_rate", "fill_tex_trilinear_linear_04"),
    ("Fill · Trilinear 0.1", "fill_rate", "fill_tex_trilinear_linear_01"),
    ("Triangle setup", "triangle_setup", "triangle_setup"),
    ("Triangle setup · All culled", "triangle_setup", "triangle_setup_all_culled"),
    ("Triangle setup · Half culled", "triangle_setup", "triangle_setup_half_culled"),
    ("Attribute fetch · 1", "attribute_fetch_shader", "attribute_fetch_shader"),
    ("Attribute fetch · 2", "attribute_fetch_shader", "attribute_fetch_shader_2_attr"),
    ("Attribute fetch · 4", "attribute_fetch_shader", "attribute_fetch_shader_4_attr"),
    ("Attribute fetch · 8", "attribute_fetch_shader", "attribute_fetch_shader_8_attr"),
    ("Varyings / DDX · 1", "varyings_ddx_shader", "varyings_shader_1"),
    ("Varyings / DDX · 2", "varyings_ddx_shader", "varyings_shader_2"),
    ("Varyings / DDX · 4", "varyings_ddx_shader", "varyings_shader_4"),
    ("Varyings / DDX · 8", "varyings_ddx_shader", "varyings_shader_8"),
)

SYSTEMC_FUNCTIONAL_CASES = frozenset(
    {
        "fill_solid",
        "fill_solid_blended",
        "fill_solid_depth_neq",
        "fill_solid_depth_never",
        "fill_tex_nearest",
        "fill_tex_bilinear",
        "fill_tex_trilinear_linear_01",
        "fill_tex_trilinear_linear_04",
        "fill_tex_trilinear_linear_05",
        "triangle_setup",
        "triangle_setup_all_culled",
        "triangle_setup_half_culled",
        "attribute_fetch_shader",
        "attribute_fetch_shader_2_attr",
        "attribute_fetch_shader_4_attr",
        "attribute_fetch_shader_8_attr",
        "varyings_shader_1",
        "varyings_shader_2",
        "varyings_shader_4",
        "varyings_shader_8",
    }
)


def _resolved_path(value: str, default: Path | None = None) -> Path:
    text = value.strip()
    if not text:
        return default or Path()
    path = Path(text).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "run"


class CounterPlot(QWidget):
    """Small dependency-free timeline plot using QPainter."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.records: list[CounterRecord] = []
        self.field = "ps_invocations"
        self.setMinimumHeight(230)

    def set_series(self, records: Iterable[CounterRecord], field: str) -> None:
        self.records = list(records)[-200:]
        self.field = field
        self.update()

    def paintEvent(self, _event: object) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#111827"))

        left, top, right, bottom = 72, 28, 22, 42
        plot_w = max(1, self.width() - left - right)
        plot_h = max(1, self.height() - top - bottom)
        values = [float(record.values.get(self.field, 0)) for record in self.records]
        label, unit, _description = COUNTER_INFO.get(self.field, (self.field, "", ""))

        painter.setPen(QColor("#dbeafe"))
        title_font = painter.font()
        title_font.setPointSize(11)
        title_font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(title_font)
        painter.drawText(left, 19, f"{label} · {unit}")

        painter.setPen(QPen(QColor("#334155"), 1))
        for step in range(5):
            y = top + plot_h * step / 4
            painter.drawLine(left, int(y), left + plot_w, int(y))

        if not values:
            painter.setPen(QColor("#94a3b8"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Run a workload to see counters")
            return

        maximum = max(values)
        scale_max = math.log10(maximum + 1.0) if maximum > 0 else 1.0
        points = []
        for index, value in enumerate(values):
            x = left + (plot_w * index / max(1, len(values) - 1))
            normalized = math.log10(max(0.0, value) + 1.0) / scale_max
            y = top + plot_h * (1.0 - normalized)
            points.append((int(x), int(y)))

        painter.setPen(QPen(QColor("#60a5fa"), 2.5))
        for first, second in zip(points, points[1:]):
            painter.drawLine(first[0], first[1], second[0], second[1])
        painter.setBrush(QColor("#93c5fd"))
        painter.setPen(Qt.PenStyle.NoPen)
        for x, y in points:
            painter.drawEllipse(x - 3, y - 3, 6, 6)

        painter.setPen(QColor("#94a3b8"))
        axis_font = painter.font()
        axis_font.setPointSize(9)
        axis_font.setWeight(QFont.Weight.Normal)
        painter.setFont(axis_font)
        painter.drawText(4, top + 8, human_number(maximum))
        painter.drawText(4, top + plot_h, "0")
        painter.drawText(left, self.height() - 14, f"records: {len(values)}")
        painter.drawText(
            left + plot_w - 190,
            self.height() - 14,
            190,
            16,
            Qt.AlignmentFlag.AlignRight,
            f"latest: {human_number(values[-1])}",
        )


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PvrGPU Control · GLBench / Mesa")
        self.resize(1240, 820)

        self.settings = QSettings("PvrGPU", "Control")
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_process_output)
        self.process.finished.connect(self._process_finished)
        self.process.errorOccurred.connect(self._process_error)

        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(250)
        self.poll_timer.timeout.connect(self._poll_artifacts)

        self.stdout_buffer = ""
        self.records: list[CounterRecord] = []
        self.metadata: dict[str, str] = {}
        self.run_dir: Path | None = None
        self.report_mtime_ns = -1
        self.latest_image: Path | None = None
        self.renderer_verified = False
        self.active_mode = ""

        self._build_ui()
        self._restore_settings()
        self._backend_changed()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel("PvrGPU Control")
        title.setObjectName("title")
        subtitle = QLabel("App (GLBench) → Mesa → llvmpipe / pvrgpu")
        subtitle.setObjectName("subtitle")
        title_box = QVBoxLayout()
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        header.addStretch()
        self.state_badge = QLabel("READY")
        self.state_badge.setObjectName("stateBadge")
        header.addWidget(self.state_badge)
        root_layout.addLayout(header)

        controls = QGroupBox("Run")
        controls_grid = QGridLayout(controls)
        controls_grid.setColumnMinimumWidth(0, 120)
        controls_grid.setColumnMinimumWidth(2, 80)
        controls_grid.setColumnMinimumWidth(4, 96)
        controls_grid.setColumnStretch(1, 2)
        controls_grid.setColumnStretch(3, 1)
        self.backend_combo = QComboBox()
        self.backend_combo.addItem("llvmpipe · Mesa software renderer", "llvmpipe")
        self.backend_combo.addItem(
            "pvrgpu · native or SystemC functional model", "pvrgpu"
        )
        self.backend_combo.currentIndexChanged.connect(self._backend_changed)
        controls_grid.addWidget(QLabel("Backend"), 0, 0)
        controls_grid.addWidget(self.backend_combo, 0, 1, 1, 4)

        self.case_combo = QComboBox()
        for label, group, case_name in GLBENCH_CASES:
            self.case_combo.addItem(label, (group, case_name))
        self.case_combo.currentIndexChanged.connect(self._refresh_runtime_label)
        controls_grid.addWidget(QLabel("GLBench case"), 1, 0)
        controls_grid.addWidget(self.case_combo, 1, 1, 1, 4)

        self.samples_spin = QSpinBox()
        self.samples_spin.setRange(1, 1000)
        self.samples_spin.setValue(5)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(16, 8192)
        self.width_spin.setValue(512)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(16, 8192)
        self.height_spin.setValue(512)
        controls_grid.addWidget(QLabel("Samples"), 2, 0)
        controls_grid.addWidget(self.samples_spin, 2, 1)
        controls_grid.addWidget(QLabel("Surface"), 2, 2)
        size_layout = QHBoxLayout()
        size_layout.addWidget(self.width_spin)
        size_layout.addWidget(QLabel("×"))
        size_layout.addWidget(self.height_spin)
        controls_grid.addLayout(size_layout, 2, 3, 1, 2)

        self.cache_bypass_label = QLabel("Cache bypass")
        self.cache_bypass_combo = QComboBox()
        self.cache_bypass_combo.addItem("Off · cache model enabled", "off")
        self.cache_bypass_combo.addItem("On · fast bypass", "on")
        self.cache_bypass_combo.setToolTip(
            "Off runs the complete cache model (default). On bypasses cache "
            "lookup and allocation to speed up functional simulation; DRAM "
            "access remains active."
        )
        self.cache_bypass_combo.currentIndexChanged.connect(
            self._refresh_runtime_label
        )
        controls_grid.addWidget(self.cache_bypass_label, 3, 0)
        controls_grid.addWidget(self.cache_bypass_combo, 3, 1, 1, 4)

        self.runtime_label = QLabel()
        self.runtime_label.setWordWrap(True)
        self.runtime_label.setObjectName("runtimeLabel")
        controls_grid.addWidget(self.runtime_label, 4, 0, 1, 5)

        self.runner_edit = QLineEdit()
        self.runner_browse = QPushButton("Browse")
        self.runner_browse.clicked.connect(self._browse_runner)
        controls_grid.addWidget(QLabel("Runner"), 5, 0)
        controls_grid.addWidget(self.runner_edit, 5, 1, 1, 3)
        controls_grid.addWidget(self.runner_browse, 5, 4)

        self.mesa_edit = QLineEdit()
        self.mesa_browse = QPushButton("Browse")
        self.mesa_browse.clicked.connect(self._browse_mesa)
        controls_grid.addWidget(QLabel("Mesa prefix"), 6, 0)
        controls_grid.addWidget(self.mesa_edit, 6, 1, 1, 3)
        controls_grid.addWidget(self.mesa_browse, 6, 4)

        self.output_edit = QLineEdit(
            os.environ.get("PVRGPU_OUTPUT_ROOT", str(WORK_ROOT / "out" / "runs"))
        )
        output_browse = QPushButton("Browse")
        output_browse.clicked.connect(self._browse_output)
        controls_grid.addWidget(QLabel("Output root"), 7, 0)
        controls_grid.addWidget(self.output_edit, 7, 1, 1, 3)
        controls_grid.addWidget(output_browse, 7, 4)

        button_row = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.start_button.setObjectName("primaryButton")
        self.start_button.clicked.connect(self._start)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop)
        self.open_button = QPushButton("Open Output")
        self.open_button.setEnabled(False)
        self.open_button.clicked.connect(self._open_output)
        self.quit_button = QPushButton("Quit")
        self.quit_button.setObjectName("quitButton")
        self.quit_button.clicked.connect(self.close)
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.stop_button)
        button_row.addWidget(self.open_button)
        button_row.addStretch()
        button_row.addWidget(self.quit_button)
        controls_grid.addLayout(button_row, 8, 0, 1, 5)
        root_layout.addWidget(controls)

        cards = QHBoxLayout()
        self.renderer_value = self._card(cards, "Renderer", "—")
        self.frame_value = self._card(cards, "Record", "0")
        self.draw_value = self._card(cards, "DrawLists", "0")
        self.pixel_value = self._card(cards, "FS invocations", "0")
        self.cycle_value = self._card(cards, "Virtual cycles", "—")
        root_layout.addLayout(cards)

        tabs = QTabWidget()
        root_layout.addWidget(tabs, 1)

        counter_tab = QWidget()
        counter_layout = QVBoxLayout(counter_tab)
        chart_bar = QHBoxLayout()
        chart_bar.addWidget(QLabel("Plot"))
        self.counter_combo = QComboBox()
        for field in ALL_COUNTER_FIELDS:
            label, unit, _ = COUNTER_INFO[field]
            self.counter_combo.addItem(f"{label} · {unit}", field)
        self.counter_combo.setCurrentIndex(7)
        self.counter_combo.currentIndexChanged.connect(self._refresh_plot)
        chart_bar.addWidget(self.counter_combo)
        chart_bar.addStretch()
        self.provenance_badge = QLabel("—")
        self.provenance_badge.setObjectName("provenanceBadge")
        chart_bar.addWidget(self.provenance_badge)
        counter_layout.addLayout(chart_bar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.plot = CounterPlot()
        self.counter_table = QTableWidget(0, 4)
        self.counter_table.setHorizontalHeaderLabels(("Counter", "Latest", "Run total", "Origin"))
        self.counter_table.setAlternatingRowColors(True)
        self.counter_table.verticalHeader().setVisible(False)
        self.counter_table.horizontalHeader().setStretchLastSection(True)
        splitter.addWidget(self.plot)
        splitter.addWidget(self.counter_table)
        splitter.setSizes((560, 620))
        counter_layout.addWidget(splitter, 1)
        tabs.addTab(counter_tab, "Counters")

        drawlist_tab = QWidget()
        drawlist_layout = QVBoxLayout(drawlist_tab)
        self.drawlist_note = QLabel(
            "Latest frame · Static = decoded program composition; Executed = "
            "repeat-expanded instructions × shader invocations."
        )
        self.drawlist_note.setWordWrap(True)
        drawlist_layout.addWidget(self.drawlist_note)
        self.drawlist_table = QTableWidget(0, 12)
        self.drawlist_table.setHorizontalHeaderLabels(
            (
                "DrawList",
                "Draw ID",
                "Stage",
                "Invocations",
                "Groups",
                "Static inst",
                "Static ALU",
                "Static Tex",
                "Static Memory",
                "Executed ALU",
                "Executed Tex",
                "Executed Memory",
            )
        )
        self.drawlist_table.setAlternatingRowColors(True)
        self.drawlist_table.verticalHeader().setVisible(False)
        self.drawlist_table.horizontalHeader().setStretchLastSection(True)
        drawlist_layout.addWidget(self.drawlist_table, 1)
        tabs.addTab(drawlist_tab, "DrawLists")

        image_tab = QWidget()
        image_layout = QVBoxLayout(image_tab)
        self.image_label = QLabel("GLBench output image will appear here")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(320, 260)
        self.image_label.setObjectName("imagePreview")
        image_layout.addWidget(self.image_label)
        tabs.addTab(image_tab, "Frame")

        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(10000)
        log_layout.addWidget(self.log)
        tabs.addTab(log_tab, "Log")

        self._set_style()

    def _card(self, row: QHBoxLayout, title: str, value: str) -> QLabel:
        frame = QFrame()
        frame.setObjectName("metricCard")
        layout = QVBoxLayout(frame)
        label = QLabel(title)
        label.setObjectName("metricTitle")
        number = QLabel(value)
        number.setObjectName("metricValue")
        number.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(label)
        layout.addWidget(number)
        row.addWidget(frame, 1)
        return number

    def _restore_settings(self) -> None:
        backend = str(self.settings.value("backend", "llvmpipe"))
        index = self.backend_combo.findData(backend)
        if index >= 0:
            self.backend_combo.setCurrentIndex(index)
        self.samples_spin.setValue(int(self.settings.value("samples", 5)))
        self.width_spin.setValue(int(self.settings.value("width", 512)))
        self.height_spin.setValue(int(self.settings.value("height", 512)))
        cache_bypass = str(self.settings.value("cache_bypass", "off")).lower()
        cache_index = self.cache_bypass_combo.findData(cache_bypass)
        self.cache_bypass_combo.setCurrentIndex(max(0, cache_index))
        case_name = str(self.settings.value("case", "fill_solid"))
        for index in range(self.case_combo.count()):
            if self.case_combo.itemData(index)[1] == case_name:
                self.case_combo.setCurrentIndex(index)
                break

    def _save_settings(self) -> None:
        self.settings.setValue("backend", self.backend_combo.currentData())
        self.settings.setValue("samples", self.samples_spin.value())
        self.settings.setValue("width", self.width_spin.value())
        self.settings.setValue("height", self.height_spin.value())
        self.settings.setValue(
            "cache_bypass", self.cache_bypass_combo.currentData()
        )
        self.settings.setValue("case", self.case_combo.currentData()[1])

    def _backend_changed(self) -> None:
        backend = str(self.backend_combo.currentData())
        cache_control_enabled = backend == "pvrgpu"
        self.cache_bypass_label.setEnabled(cache_control_enabled)
        self.cache_bypass_combo.setEnabled(cache_control_enabled)
        if backend == "llvmpipe":
            runner = os.environ.get("PVRGPU_GLBENCH_RUNNER", "")
            mesa = os.environ.get("PVRGPU_LLVMPIPE_MESA_PREFIX", "")
            self.runner_edit.setText(runner)
            self.mesa_edit.setText(mesa)
            self.mesa_edit.setEnabled(True)
            self.mesa_browse.setEnabled(True)
        else:
            native_runner = os.environ.get("PVRGPU_NATIVE_GLBENCH_RUNNER", "")
            native_mesa = os.environ.get("PVRGPU_NATIVE_MESA_PREFIX", "")
            if _resolved_path(native_runner).is_file() and _resolved_path(native_mesa).is_dir():
                self.runner_edit.setText(native_runner)
                self.mesa_edit.setText(native_mesa)
                self.mesa_edit.setEnabled(True)
                self.mesa_browse.setEnabled(True)
            else:
                stub = _resolved_path(
                    os.environ.get(
                        "PVRGPU_MODEL_STUB", str(WORK_ROOT / "build" / "bin" / "pvrgpu-model-stub")
                    )
                )
                self.runner_edit.setText(str(stub))
                self.mesa_edit.clear()
                self.mesa_edit.setEnabled(False)
                self.mesa_browse.setEnabled(False)
        self._refresh_runtime_label()

    def _refresh_runtime_label(self) -> None:
        backend = str(self.backend_combo.currentData())
        cache_bypass = str(self.cache_bypass_combo.currentData() or "off")
        cache_mode = (
            "cache bypass ON (fast simulation)"
            if cache_bypass == "on"
            else "cache bypass OFF (cache model enabled)"
        )
        runner_ready = _resolved_path(self.runner_edit.text()).is_file()
        mesa_text = self.mesa_edit.text().strip()
        mesa_ready = bool(mesa_text) and _resolved_path(mesa_text).is_dir()

        if backend == "llvmpipe":
            self.runtime_label.setText(
                "REAL · GLBench → Mesa → llvmpipe · 17 reported frame counters"
                if runner_ready and mesa_ready
                else "UNAVAILABLE · configure PVRGPU_GLBENCH_RUNNER and PVRGPU_LLVMPIPE_MESA_PREFIX"
            )
            return

        if runner_ready and mesa_ready:
            self.runtime_label.setText(
                f"NATIVE · GLBench → Mesa → pvrgpu · {cache_mode}"
            )
            return

        case_data = self.case_combo.currentData()
        case_name = case_data[1] if case_data else ""
        if case_name in SYSTEMC_FUNCTIONAL_CASES:
            self.runtime_label.setText(
                "MOD · SystemC functional raster rendering with "
                f"modelled counters and DRAM-readback PNG · {cache_mode}. "
                "Configure native runner/prefix to enable GLBench → Mesa → pvrgpu."
            )
        else:
            self.runtime_label.setText(
                "UNSUPPORTED · the SystemC functional fallback currently supports "
                "the completed Fill.Solid and Triangle setup state cases plus "
                "Texture nearest/bilinear/trilinear 0.1/0.4, "
                "Attribute fetch · 1/2/4/8 and "
                "Varyings / DDX · 1/2/4/8 only. "
                "Configure native runner/prefix for this "
                "GLBench case."
            )

    def _browse_runner(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select runner", self.runner_edit.text())
        if path:
            self.runner_edit.setText(path)

    def _browse_mesa(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Mesa prefix", self.mesa_edit.text())
        if path:
            self.mesa_edit.setText(path)

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output root", self.output_edit.text())
        if path:
            self.output_edit.setText(path)

    def _start(self) -> None:
        if self.process.state() != QProcess.ProcessState.NotRunning:
            return
        backend = str(self.backend_combo.currentData())
        runner = _resolved_path(self.runner_edit.text())
        mesa_prefix = _resolved_path(self.mesa_edit.text()) if self.mesa_edit.text().strip() else Path()
        group, case_name = self.case_combo.currentData()

        native_pvrgpu = backend == "pvrgpu" and bool(self.mesa_edit.text().strip())
        if (
            backend == "pvrgpu"
            and not native_pvrgpu
            and case_name not in SYSTEMC_FUNCTIONAL_CASES
        ):
            self._refresh_runtime_label()
            QMessageBox.warning(
                self,
                "Unsupported SystemC case",
                "The SystemC functional fallback currently supports the completed "
                "Fill.Solid and Triangle setup state cases plus Texture "
                "nearest/bilinear/trilinear 0.1/0.4, Attribute fetch · 1/2/4/8 and "
                "Varyings / DDX · 1/2/4/8. "
                "Configure a native PvrGPU Mesa runner/prefix to run this case.",
            )
            return

        if not runner.is_file():
            QMessageBox.critical(self, "Runner missing", f"Runner was not found:\n{runner}")
            return

        if (backend == "llvmpipe" or native_pvrgpu) and (
            not self.mesa_edit.text().strip() or not mesa_prefix.is_dir()
        ):
            QMessageBox.critical(self, "Mesa prefix missing", f"Mesa prefix was not found:\n{mesa_prefix}")
            return

        output_root = _resolved_path(
            self.output_edit.text(), WORK_ROOT / "out" / "runs"
        )
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        mode_name = (
            backend
            if backend == "llvmpipe" or native_pvrgpu
            else "pvrgpu-systemc-functional"
        )
        self.run_dir = output_root / f"{stamp}-{_safe_name(mode_name)}-{_safe_name(case_name)}"
        png_dir = self.run_dir / "png"
        png_dir.mkdir(parents=True, exist_ok=False)

        self.records.clear()
        self.metadata.clear()
        self.stdout_buffer = ""
        self.report_mtime_ns = -1
        self.latest_image = None
        self.renderer_verified = False
        self.active_mode = mode_name
        self.log.clear()
        self.image_label.setText("Waiting for framebuffer artifact…")
        self._refresh_counters()

        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PVRGPU_BACKEND", backend)
        cache_bypass = str(self.cache_bypass_combo.currentData() or "off")
        if backend == "pvrgpu":
            environment.insert("PVRGPU_CACHE_BYPASS", cache_bypass)
        else:
            environment.remove("PVRGPU_CACHE_BYPASS")
        if backend == "llvmpipe" or native_pvrgpu:
            driver = "llvmpipe" if backend == "llvmpipe" else "pvrgpu"
            environment.insert("EGL_PLATFORM", "surfaceless")
            environment.insert("LIBGL_ALWAYS_SOFTWARE", "1")
            environment.insert("MESA_LOADER_DRIVER_OVERRIDE", "swrast")
            environment.insert("GALLIUM_DRIVER", driver)
            environment.insert("LIBGL_DRIVERS_PATH", str(mesa_prefix / "lib" / "dri"))
            old_dyld = environment.value("DYLD_LIBRARY_PATH")
            dyld = str(mesa_prefix / "lib") + (f":{old_dyld}" if old_dyld else "")
            environment.insert("DYLD_LIBRARY_PATH", dyld)
            environment.insert("MESA_SHADER_CACHE_DISABLE", "true")
            environment.insert("MESA_COUNTER_REPORT_PATH", str(self.run_dir / "Report.md"))
            environment.insert(
                "MESA_COUNTER_FRAME_SELECTION_MS",
                f"PvrGPU {driver} {case_name} samples 1-{self.samples_spin.value()}",
            )
            arguments = [
                "--test", group,
                "--case", case_name,
                "--sample", str(self.samples_spin.value()),
                "--size", f"{self.width_spin.value()}x{self.height_spin.value()}",
                "--outdir", str(png_dir),
            ]
        else:
            arguments = [
                "--frames", str(self.samples_spin.value()),
                "--width", str(self.width_spin.value()),
                "--height", str(self.height_spin.value()),
                "--case", case_name,
                "--outdir", str(png_dir),
                "--cache-bypass", cache_bypass,
            ]

        self.process.setProcessEnvironment(environment)
        WORK_ROOT.mkdir(parents=True, exist_ok=True)
        self.process.setWorkingDirectory(str(WORK_ROOT))
        self._append_log(f"[run] {runner} {' '.join(arguments)}\n")
        self._append_log(f"[output] {self.run_dir}\n")
        if mode_name == "pvrgpu-systemc-functional":
            self._append_log(
                "[MOD] SystemC functional raster render with modelled "
                "counters and DRAM-readback PNG output.\n"
            )

        self.process.start(str(runner), arguments)
        if not self.process.waitForStarted(3000):
            QMessageBox.critical(self, "Launch failed", self.process.errorString())
            return

        self._save_settings()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.open_button.setEnabled(True)
        self._set_state("RUNNING")
        self.poll_timer.start()

    def _stop(self) -> None:
        if self.process.state() == QProcess.ProcessState.NotRunning:
            return
        self._set_state("STOPPING")
        self.process.terminate()
        QTimer.singleShot(2000, self._kill_if_running)

    def _kill_if_running(self) -> None:
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.kill()

    def _read_process_output(self) -> None:
        chunk = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self.stdout_buffer += chunk
        while "\n" in self.stdout_buffer:
            line, self.stdout_buffer = self.stdout_buffer.split("\n", 1)
            self._handle_line(line.rstrip("\r"))

    def _handle_line(self, line: str) -> None:
        self._append_log(line + "\n")
        stripped = line.strip()
        if stripped.startswith("{"):
            try:
                message = parse_jsonl_line(stripped)
            except CounterProtocolError as exc:
                self._append_log(f"[counter protocol error] {exc}\n")
                return
            message_type = message.get("type")
            if message_type == "hello":
                self.renderer_value.setText(str(message.get("renderer", "—")))
                warning = message.get("warning")
                if self.active_mode == "pvrgpu-systemc-functional":
                    self.provenance_badge.setText("MOD · modelled")
                elif warning:
                    self.provenance_badge.setText("MOCK · counter-only")
            elif message_type in {"counter", "counter_sample"}:
                try:
                    self.records.append(counter_record_from_message(message))
                except CounterProtocolError as exc:
                    self._append_log(f"[counter sample error] {exc}\n")
                self._refresh_counters()
            return

        board = re.search(r"# board_id:\s*(.+?)\s+-\s+(.+)$", stripped)
        if board:
            renderer = board.group(2).strip()
            self.renderer_value.setText(renderer)
            if self.active_mode == "llvmpipe":
                self.renderer_verified = "llvmpipe" in renderer.lower()
                if not self.renderer_verified:
                    self._append_log("[ERROR] Requested llvmpipe but renderer verification failed.\n")
        capture = re.search(r"@CAPTURE:.*sample=(\d+)", stripped)
        if capture:
            self.frame_value.setText(capture.group(1))

    def _poll_artifacts(self) -> None:
        if not self.run_dir:
            return
        report_path = self.run_dir / "Report.md"
        if report_path.is_file():
            try:
                mtime = report_path.stat().st_mtime_ns
                if mtime != self.report_mtime_ns:
                    report = parse_markdown_report(report_path)
                    self.report_mtime_ns = mtime
                    self.metadata = dict(report.metadata)
                    self.records = list(report.records)
                    renderer = self.metadata.get("Renderer", "")
                    if renderer:
                        self.renderer_value.setText(renderer)
                        if self.active_mode == "llvmpipe":
                            self.renderer_verified = "llvmpipe" in renderer.lower()
                    self._refresh_counters()
            except (OSError, CounterProtocolError):
                pass

        pngs = sorted((self.run_dir / "png").glob("*.png"))
        if pngs and pngs[-1] != self.latest_image:
            self.latest_image = pngs[-1]
            self._show_image(self.latest_image)

    def _show_image(self, path: Path) -> None:
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            return
        scaled = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)
        self.image_label.setToolTip(str(path))

    def resizeEvent(self, event: object) -> None:
        super().resizeEvent(event)  # type: ignore[arg-type]
        if self.latest_image:
            self._show_image(self.latest_image)

    def _refresh_counters(self) -> None:
        latest = self.records[-1] if self.records else None
        self.frame_value.setText(str(latest.frame if latest else 0))
        self.draw_value.setText(human_number(latest.values.get("drawlists") if latest else 0))
        self.pixel_value.setText(human_number(latest.values.get("ps_invocations") if latest else 0))
        self.cycle_value.setText(human_number(latest.values.get("virtual_gpu_cycles") if latest else None))

        present = [
            field
            for field in ALL_COUNTER_FIELDS
            if any(field in record.values for record in self.records)
        ]
        self.counter_table.setRowCount(len(present))
        totals = {
            field: sum(record.values.get(field, 0) for record in self.records)
            for field in present
        }
        for row, field in enumerate(present):
            label, unit, description = COUNTER_INFO[field]
            current = latest.values.get(field) if latest else None
            origin = self._origin_label(latest) if latest else "—"
            values = (
                f"{label} · {unit}",
                human_number(current),
                human_number(totals[field]),
                origin,
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(description)
                if column in (1, 2):
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight
                        | Qt.AlignmentFlag.AlignVCenter
                    )
                self.counter_table.setItem(row, column, item)
        self.counter_table.resizeColumnsToContents()
        if latest:
            self.provenance_badge.setText(self._origin_label(latest))
        self._refresh_drawlists(latest)
        self._refresh_plot()

    def _refresh_drawlists(self, latest: CounterRecord | None) -> None:
        drawlists = latest.drawlist_stats if latest else ()
        self.drawlist_table.setRowCount(len(drawlists) * 2)
        if drawlists:
            self.drawlist_note.setText(
                f"Frame {latest.frame} · Static = decoded program composition; "
                "Executed = repeat-expanded instructions × shader invocations. "
                "Memory includes shader export instructions, not bytes."
            )
        else:
            self.drawlist_note.setText(
                "This backend has not reported per-DrawList shader instruction statistics."
            )

        row = 0
        for drawlist in drawlists:
            for stage_name, stats in (
                ("VS", drawlist.vertex),
                ("FS", drawlist.fragment),
            ):
                values = (
                    drawlist.drawlist,
                    drawlist.draw_id,
                    stage_name,
                    stats.invocations,
                    stats.program_groups,
                    stats.program_instructions,
                    stats.program_alu_instructions,
                    stats.program_tex_instructions,
                    stats.program_memory_instructions,
                    stats.executed_alu_instructions,
                    stats.executed_tex_instructions,
                    stats.executed_memory_instructions,
                )
                for column, value in enumerate(values):
                    item = QTableWidgetItem(
                        str(value) if column == 2 else human_number(value)
                    )
                    if column != 2:
                        item.setTextAlignment(
                            Qt.AlignmentFlag.AlignRight
                            | Qt.AlignmentFlag.AlignVCenter
                        )
                    self.drawlist_table.setItem(row, column, item)
                row += 1
        self.drawlist_table.resizeColumnsToContents()

    def _origin_label(self, record: CounterRecord | None) -> str:
        if not record:
            return "—"
        if record.provenance == "reported":
            return "REP · llvmpipe"
        if record.provenance in {"mock", "mock-derived"}:
            return "MOCK · counter-only"
        if record.provenance in {"modeled", "modelled"}:
            return "MOD · modelled"
        return f"{record.provenance} · {record.source}"

    def _refresh_plot(self) -> None:
        field = str(self.counter_combo.currentData() or "ps_invocations")
        self.plot.set_series(self.records, field)

    def _process_finished(self, exit_code: int, _status: object) -> None:
        self._read_process_output()
        if self.stdout_buffer:
            self._handle_line(self.stdout_buffer)
            self.stdout_buffer = ""
        self._poll_artifacts()
        self.poll_timer.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

        failed = exit_code != 0
        if self.active_mode == "llvmpipe" and not self.renderer_verified:
            failed = True
            self._append_log("[failed] llvmpipe renderer was not verified.\n")
        self._set_state("FAILED" if failed else "FINISHED")
        self._append_log(f"[exit] code={exit_code}\n")

    def _process_error(self, _error: object) -> None:
        self._append_log(f"[process error] {self.process.errorString()}\n")

    def _append_log(self, text: str) -> None:
        self.log.moveCursor(QTextCursor.MoveOperation.End)
        self.log.insertPlainText(text)
        self.log.moveCursor(QTextCursor.MoveOperation.End)

    def _set_state(self, state: str) -> None:
        self.state_badge.setText(state)
        self.state_badge.setProperty("state", state.lower())
        self.state_badge.style().unpolish(self.state_badge)
        self.state_badge.style().polish(self.state_badge)

    def _open_output(self) -> None:
        if self.run_dir:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.run_dir)))

    def closeEvent(self, event: object) -> None:
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.kill()
            self.process.waitForFinished(1000)
        self._save_settings()
        super().closeEvent(event)  # type: ignore[arg-type]

    def _set_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #f4f6f8; color: #172033; }
            QLabel#title { font-size: 24px; font-weight: 700; color: #172033; }
            QLabel#subtitle { color: #5b667a; }
            QLabel#stateBadge, QLabel#provenanceBadge {
                background: #e0e7ff; color: #3730a3; border-radius: 10px;
                padding: 5px 10px; font-weight: 700;
            }
            QLabel#stateBadge[state="running"] { background: #dcfce7; color: #166534; }
            QLabel#stateBadge[state="failed"] { background: #fee2e2; color: #991b1b; }
            QLabel#stateBadge[state="finished"] { background: #dbeafe; color: #1e40af; }
            QLabel#runtimeLabel { background: #fff7ed; color: #9a3412; padding: 8px; border-radius: 5px; }
            QGroupBox { background: white; border: 1px solid #d9dee7; border-radius: 8px;
                        margin-top: 10px; padding-top: 12px; font-weight: 600; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; }
            QFrame#metricCard { background: white; border: 1px solid #d9dee7; border-radius: 8px; }
            QLabel#metricTitle { color: #64748b; font-size: 11px; }
            QLabel#metricValue { color: #0f172a; font-size: 18px; font-weight: 700; }
            QPushButton { background: white; border: 1px solid #cbd5e1; border-radius: 5px; padding: 7px 13px; }
            QPushButton:hover { background: #eef2ff; }
            QPushButton#primaryButton { background: #4f46e5; color: white; border: none; font-weight: 700; }
            QPushButton#quitButton { background: #fff1f2; color: #9f1239; border-color: #fecdd3; font-weight: 700; }
            QPushButton#quitButton:hover { background: #ffe4e6; }
            QPushButton:disabled { color: #9ca3af; background: #e5e7eb; }
            QLineEdit, QComboBox, QSpinBox { background: white; border: 1px solid #cbd5e1;
                                            border-radius: 4px; padding: 5px; }
            QTabWidget::pane { border: 1px solid #d9dee7; background: white; }
            QTabBar::tab { background: #e8ecf2; padding: 8px 16px; }
            QTabBar::tab:selected { background: white; color: #3730a3; font-weight: 700; }
            QTableWidget { background: white; alternate-background-color: #f8fafc; gridline-color: #e2e8f0; }
            QPlainTextEdit { background: #111827; color: #d1fae5; font-family: Menlo, monospace; }
            QLabel#imagePreview { background: #111827; color: #94a3b8; border-radius: 5px; }
            """
        )


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("PvrGPU Control")
    window = MainWindow()
    if "--smoke-test" in sys.argv:
        window.show()
        QTimer.singleShot(150, app.quit)
    else:
        window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
