from __future__ import annotations

import json
import sys
import tempfile
import threading
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtGui, QtWidgets

from osa_module.controller import MeasurementSettings, OSAController

from ..calibration import CalibrationFit, fit_calibration, load_calibration_csv
from ..calibration.calibration_new import (
    CalibrationAborted,
    CalibrationProgress,
    CalibrationResult,
    find_min_max_intensity_levels,
    intensity_calibration,
    load_calibration_result,
    load_wavelength_map_csv,
    save_calibration_result,
    wavelength_calibration,
    write_intensity_calibration_csv,
)
from ..controller import ScanParams, ScanResult, SLMController
from ..detector import Detector, SimulatedDetector
from ..generator import (
    MAX_LEVEL,
    equal_x_segment_edges,
    make_equal_x_segments,
    make_vertical_window,
    make_x_segments,
    write_santec_csv,
)
from ..keepalive import SLMKeepAlive
from .style import DARK_STYLESHEET


# a calibration progress callback marshalled onto the GUI thread via a signal
ProgressEmit = Callable[[CalibrationProgress], None]


class WorkerSignals(QtCore.QObject):
    finished = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)


class FunctionWorker(QtCore.QRunnable):
    def __init__(self, func: Callable[[], Any]):
        super().__init__()
        self.func = func
        self.signals = WorkerSignals()

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            self.signals.finished.emit(self.func())
        except Exception:
            self.signals.error.emit(traceback.format_exc())


class CalibrationProgressDialog(QtWidgets.QDialog):
    """Live view of an OSA calibration run: phase, progress bar, log and plot.

    update_progress() is called on the GUI thread for every measured step; the
    plot itself is redrawn on a timer so a fast stream of points cannot flood
    the event loop. finish() freezes the view and enables Close.
    """

    _PHASES = {
        "min_max": ("Step 1 / 3 · Min/Max level sweep", "Level", "Output power (W)"),
        "wavelength": (
            "Step 2 / 3 · Wavelength mapping",
            "x coordinate (px)",
            "Wavelength (nm)",
        ),
        "intensity": (
            "Step 3 / 3 · Intensity vs level",
            "Level",
            "Normalized intensity",
        ),
    }

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        on_stop: Callable[[], None] | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Calibration Progress")
        self.setModal(False)
        self.resize(760, 600)
        self._on_stop = on_stop
        self._running = True
        self._phase: str | None = None
        self._xs: list[float] = []
        self._ys: list[float] = []
        self._dirty = False

        layout = QtWidgets.QVBoxLayout(self)

        self.phase_label = QtWidgets.QLabel("Preparing…")
        self.phase_label.setObjectName("PageSubtitle")
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%v / %m  (%p%)")
        self.status_label = QtWidgets.QLabel("\N{EN DASH}")
        self.status_label.setWordWrap(True)

        self.figure = Figure(figsize=(6, 3.2), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.add_subplot(111)
        self._style_axes()

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setObjectName("LogBox")
        self.log.setMaximumHeight(140)

        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.setProperty("variant", "danger")
        self.close_button = QtWidgets.QPushButton("Close")
        self.close_button.setEnabled(False)
        self.stop_button.clicked.connect(self._handle_stop)
        self.close_button.clicked.connect(self.close)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.stop_button)
        buttons.addWidget(self.close_button)

        layout.addWidget(self.phase_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.log)
        layout.addLayout(buttons)

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._redraw)
        self._timer.start(120)

    def _style_axes(self) -> None:
        self.figure.patch.set_facecolor("#101820")
        axes = self.axes
        axes.set_facecolor("#101820")
        axes.grid(True, color="#2b3a42", linewidth=0.7)
        axes.tick_params(colors="#d8dee9")
        axes.xaxis.label.set_color("#d8dee9")
        axes.yaxis.label.set_color("#d8dee9")
        for spine in axes.spines.values():
            spine.set_color("#41515c")

    def update_progress(self, progress: CalibrationProgress) -> None:
        if progress.phase != self._phase:
            self._enter_phase(progress.phase)
        total = max(int(progress.total), 1)
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(min(int(progress.step) + 1, total))
        self.status_label.setText(progress.message)
        if progress.x is not None and progress.y is not None:
            self._xs.append(float(progress.x))
            self._ys.append(float(progress.y))
            self._dirty = True

    def _enter_phase(self, phase: str) -> None:
        self._phase = phase
        self._xs.clear()
        self._ys.clear()
        title, _xlabel, _ylabel = self._PHASES.get(phase, (phase, "x", "y"))
        self.phase_label.setText(title)
        self.log.appendPlainText(f"\N{BLACK RIGHT-POINTING TRIANGLE} {title}")
        self._dirty = True

    def _redraw(self) -> None:
        if not self._dirty:
            return
        self._dirty = False
        self.axes.clear()
        self._style_axes()
        phase = self._phase or ""
        _title, xlabel, ylabel = self._PHASES.get(phase, (phase, "x", "y"))
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        if self._xs:
            self.axes.plot(
                self._xs,
                self._ys,
                color="#47b8e0",
                marker="o",
                markersize=3,
                linewidth=1.0,
            )
        self.canvas.draw_idle()

    def finish(self, success: bool, message: str) -> None:
        self._running = False
        self._timer.stop()
        self._dirty = True
        self._redraw()
        self.status_label.setText(message)
        self.log.appendPlainText(message)
        self.stop_button.setEnabled(False)
        self.close_button.setEnabled(True)
        if success:
            self.progress_bar.setValue(self.progress_bar.maximum())

    def _handle_stop(self) -> None:
        self.stop_button.setEnabled(False)
        self.status_label.setText("Stopping…")
        if self._on_stop is not None:
            self._on_stop()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        # closing mid-run requests a stop but still lets the window close
        if self._running and self._on_stop is not None:
            self._on_stop()
        self._timer.stop()
        super().closeEvent(event)


class MainWindow(QtWidgets.QMainWindow):
    scan_progress = QtCore.pyqtSignal(int, int, str)
    scan_started = QtCore.pyqtSignal(int, int, int, int)
    scan_sample = QtCore.pyqtSignal(float, float)
    keepalive_status = QtCore.pyqtSignal(bool, str)
    calibration_progress = QtCore.pyqtSignal(object)

    def __init__(
        self,
        controller_factory: Callable[..., SLMController] = SLMController,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.controller_factory = controller_factory
        self.controller: SLMController | None = None
        self.controller_display_no: int | None = None
        self.thread_pool = QtCore.QThreadPool.globalInstance()
        self._workers: set[FunctionWorker] = set()
        self.slm_size = (1920, 1200)
        self.calibration_fits: dict[float, CalibrationFit] = {}
        self.osa_controller: OSAController | None = None
        self.calibration_result: CalibrationResult | None = None
        self.calibration_stop_event: threading.Event | None = None
        self.calibration_dialog: CalibrationProgressDialog | None = None
        self.scan_stop_event: threading.Event | None = None
        self.scan_pause_event: threading.Event | None = None
        self.scan_params: ScanParams | None = None
        self.keepalive: SLMKeepAlive | None = None
        self._slm_tasks_active = 0
        self._scan_x_range: tuple[int, int] = (0, 0)
        self._segments_updating = False

        self.setWindowTitle("Santec SLM Control")
        self.resize(1280, 840)
        self._build_ui()
        self._apply_style()
        self.scan_progress.connect(self._on_scan_progress)
        self.scan_started.connect(self._on_scan_started)
        self.scan_sample.connect(self._on_scan_sample)
        self.keepalive_status.connect(self._on_keepalive_status)
        self.calibration_progress.connect(self._on_calibration_progress)

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        sidebar = QtWidgets.QWidget()
        sidebar.setObjectName("Navigation")
        sidebar.setFixedWidth(220)
        sidebar_layout = QtWidgets.QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        brand = QtWidgets.QLabel("Santec SLM-200")
        brand.setObjectName("AppBrand")
        brand_sub = QtWidgets.QLabel("Control Suite")
        brand_sub.setObjectName("AppBrandSub")
        sidebar_layout.addWidget(brand)
        sidebar_layout.addWidget(brand_sub)

        self.nav = QtWidgets.QListWidget()
        self.nav.setObjectName("Navigation")
        self.nav.setFrameShape(QtWidgets.QFrame.NoFrame)
        nav_items = (
            ("\N{LINK SYMBOL}  SLM Control", "Connection and basic display"),
            ("\N{CHART WITH UPWARDS TREND}  Calibration", "Intensity curve fitting"),
            ("\N{LEFT RIGHT ARROW}  Center Scan", "Sweep a window across x"),
            ("\N{TRIGRAM FOR HEAVEN}  Phase Segments", "Piecewise phase along x"),
            ("\N{HIGH VOLTAGE SIGN}  TPA Encoding", "Not implemented yet"),
        )
        for label, tooltip in nav_items:
            item = QtWidgets.QListWidgetItem(label)
            item.setSizeHint(QtCore.QSize(180, 48))
            item.setToolTip(tooltip)
            self.nav.addItem(item)
        sidebar_layout.addWidget(self.nav, 1)

        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self._build_control_page())
        self.stack.addWidget(self._build_calibration_page())
        self.stack.addWidget(self._build_scan_page())
        self.stack.addWidget(self._build_segments_page())
        self.stack.addWidget(self._build_tpa_page())

        layout.addWidget(sidebar)
        layout.addWidget(self.stack, 1)
        self.setCentralWidget(central)
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav.setCurrentRow(0)

    def _build_control_page(self) -> QtWidgets.QWidget:
        page = self._page_shell("SLM Control")

        connection = self._panel("Connection")
        connection_layout = QtWidgets.QGridLayout(connection)
        self.display_no_spin = QtWidgets.QSpinBox()
        self.display_no_spin.setRange(1, 8)
        self.display_no_spin.setValue(1)
        self.display_no_spin.valueChanged.connect(self._reset_controller)
        self.rate120_check = QtWidgets.QCheckBox("120 Hz model")
        self.rate120_check.toggled.connect(self._reset_controller)
        self.conn_status_label = QtWidgets.QLabel("Status: closed")
        self.info_label = QtWidgets.QLabel("Size: unknown")

        detect_button = QtWidgets.QPushButton("Detect SLM")
        open_button = QtWidgets.QPushButton("Open")
        close_button = QtWidgets.QPushButton("Close")
        info_button = QtWidgets.QPushButton("Read Info")
        detect_button.clicked.connect(self._detect_slm)
        open_button.clicked.connect(self._open_slm)
        close_button.clicked.connect(self._close_slm)
        info_button.clicked.connect(self._read_slm_info)

        self.usb_slm_no_spin = QtWidgets.QSpinBox()
        self.usb_slm_no_spin.setRange(1, 8)
        self.usb_slm_no_spin.setValue(1)
        dvi_mode_button = QtWidgets.QPushButton("Switch to DVI Mode")
        dvi_mode_button.setToolTip(
            "Set the SLM video interface to DVI over USB "
            "(required before using the display functions)"
        )
        dvi_mode_button.clicked.connect(self._switch_to_dvi_mode)

        self.keepalive_check = QtWidgets.QCheckBox("DVI keep-alive")
        self.keepalive_check.setToolTip(
            "Re-send the current pattern over DVI at a fixed interval so the "
            "display link stays active and the SLM does not shut down or error"
        )
        self.keepalive_check.toggled.connect(self._toggle_keepalive)
        self.keepalive_interval_spin = QtWidgets.QDoubleSpinBox()
        self.keepalive_interval_spin.setRange(0.5, 30.0)
        self.keepalive_interval_spin.setDecimals(1)
        self.keepalive_interval_spin.setSingleStep(0.5)
        self.keepalive_interval_spin.setValue(15.0)
        self.keepalive_interval_spin.setSuffix(" s")
        self.keepalive_interval_spin.valueChanged.connect(self._on_keepalive_interval)
        self.keepalive_status_label = QtWidgets.QLabel("Keep-alive: off")
        self._set_status(self.keepalive_status_label, "Keep-alive: off", "off")

        connection_layout.addWidget(QtWidgets.QLabel("Display"), 0, 0)
        connection_layout.addWidget(self.display_no_spin, 0, 1)
        connection_layout.addWidget(detect_button, 0, 2)
        connection_layout.addWidget(open_button, 0, 3)
        connection_layout.addWidget(close_button, 0, 4)
        connection_layout.addWidget(info_button, 0, 5)
        connection_layout.addWidget(QtWidgets.QLabel("USB SLM"), 1, 0)
        connection_layout.addWidget(self.usb_slm_no_spin, 1, 1)
        connection_layout.addWidget(dvi_mode_button, 1, 2)
        connection_layout.addWidget(self.rate120_check, 1, 3, 1, 2)
        connection_layout.addWidget(self.keepalive_check, 2, 0, 1, 2)
        connection_layout.addWidget(QtWidgets.QLabel("Interval"), 2, 2)
        connection_layout.addWidget(self.keepalive_interval_spin, 2, 3)
        connection_layout.addWidget(self.keepalive_status_label, 2, 4, 1, 2)
        connection_layout.addWidget(self.conn_status_label, 3, 0, 1, 3)
        connection_layout.addWidget(self.info_label, 3, 3, 1, 3)

        grayscale = self._panel("Grayscale")
        grayscale_layout = QtWidgets.QGridLayout(grayscale)
        self.gray_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.gray_slider.setRange(0, 1023)
        self.gray_slider.setValue(0)
        self.gray_spin = QtWidgets.QSpinBox()
        self.gray_spin.setRange(0, 1023)
        self.gray_slider.valueChanged.connect(self.gray_spin.setValue)
        self.gray_spin.valueChanged.connect(self.gray_slider.setValue)
        gray_button = QtWidgets.QPushButton("Display Level")
        gray_button.clicked.connect(self._display_grayscale)

        grayscale_layout.addWidget(self.gray_slider, 0, 0)
        grayscale_layout.addWidget(self.gray_spin, 0, 1)
        grayscale_layout.addWidget(gray_button, 0, 2)

        csv_panel = self._panel("CSV Display")
        csv_layout = QtWidgets.QGridLayout(csv_panel)
        self.csv_path_edit = QtWidgets.QLineEdit()
        csv_browse = QtWidgets.QPushButton("Browse")
        csv_display = QtWidgets.QPushButton("Display CSV")
        csv_browse.clicked.connect(self._browse_display_csv)
        csv_display.clicked.connect(self._display_csv)
        csv_layout.addWidget(self.csv_path_edit, 0, 0)
        csv_layout.addWidget(csv_browse, 0, 1)
        csv_layout.addWidget(csv_display, 0, 2)

        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setObjectName("LogBox")

        page.layout().addWidget(connection)
        page.layout().addWidget(grayscale)
        page.layout().addWidget(csv_panel)
        page.layout().addWidget(self._panel_with_widget("Status", self.log_box), 1)
        return page

    def _build_calibration_page(self) -> QtWidgets.QWidget:
        page = self._page_shell("Calibration")

        acquisition = self._build_acquisition_panel()

        controls = self._panel("Fit from CSV")
        controls_layout = QtWidgets.QGridLayout(controls)
        self.calibration_path_edit = QtWidgets.QLineEdit()
        browse_button = QtWidgets.QPushButton("Browse")
        fit_button = QtWidgets.QPushButton("Run Fit")
        self.save_fit_button = QtWidgets.QPushButton("Save Result")
        self.save_fit_button.setEnabled(False)
        browse_button.clicked.connect(self._browse_calibration_csv)
        fit_button.clicked.connect(self._run_calibration_fit)
        self.save_fit_button.clicked.connect(self._save_calibration_result)

        self.wavelength_combo = QtWidgets.QComboBox()
        self.wavelength_combo.currentIndexChanged.connect(self._update_calibration_view)

        controls_layout.addWidget(self.calibration_path_edit, 0, 0)
        controls_layout.addWidget(browse_button, 0, 1)
        controls_layout.addWidget(fit_button, 0, 2)
        controls_layout.addWidget(self.save_fit_button, 0, 3)
        controls_layout.addWidget(QtWidgets.QLabel("Wavelength"), 1, 0)
        controls_layout.addWidget(self.wavelength_combo, 1, 1, 1, 3)

        self.fit_table = QtWidgets.QTableWidget(0, 2)
        self.fit_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.fit_table.horizontalHeader().setStretchLastSection(True)
        self.fit_table.verticalHeader().setVisible(False)

        self.figure = Figure(figsize=(6, 4), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        plot_panel = self._panel("Fit Curve")
        plot_layout = QtWidgets.QVBoxLayout(plot_panel)
        plot_layout.addWidget(self.canvas)

        # second canvas: the acquired intensity maps (normalized vs raw W)
        self.map_figure = Figure(figsize=(6, 4), tight_layout=True)
        self.map_canvas = FigureCanvas(self.map_figure)
        map_panel = self._panel("Intensity Map")
        map_layout = QtWidgets.QVBoxLayout(map_panel)
        map_controls = QtWidgets.QHBoxLayout()
        self.map_kind_combo = QtWidgets.QComboBox()
        self.map_kind_combo.addItems(["Normalized", "Raw (W)"])
        self.map_kind_combo.currentIndexChanged.connect(self._update_intensity_map)
        map_controls.addWidget(QtWidgets.QLabel("Map"))
        map_controls.addWidget(self.map_kind_combo)
        map_controls.addStretch(1)
        map_layout.addLayout(map_controls)
        map_layout.addWidget(self.map_canvas)

        right_tabs = QtWidgets.QTabWidget()
        right_tabs.addTab(plot_panel, "Fit Curve")
        right_tabs.addTab(map_panel, "Intensity Map")

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.addWidget(self._panel_with_widget("Fit Parameters", self.fit_table))
        split.addWidget(right_tabs)
        split.setSizes([360, 720])

        page.layout().addWidget(acquisition)
        page.layout().addWidget(controls)
        page.layout().addWidget(split, 1)
        return page

    def _build_acquisition_panel(self) -> QtWidgets.QGroupBox:
        """OSA-driven calibration: connect, then run the 3 steps independently."""
        panel = self._panel("Acquisition (OSA + SLM)")
        outer = QtWidgets.QVBoxLayout(panel)

        # --- shared OSA connection row ---
        conn = QtWidgets.QGridLayout()
        self.osa_host_edit = QtWidgets.QLineEdit("192.168.1.11")
        self.osa_host_edit.setPlaceholderText("OSA host / IP")
        self.osa_port_spin = self._spin(1, 65535, 10001)
        self.osa_connect_button = QtWidgets.QPushButton("Connect OSA")
        self.osa_disconnect_button = QtWidgets.QPushButton("Disconnect")
        self.osa_disconnect_button.setProperty("variant", "ghost")
        self.osa_disconnect_button.setEnabled(False)
        self.osa_status_label = QtWidgets.QLabel("OSA: closed")
        self._set_status(self.osa_status_label, "OSA: closed", "off")
        self.osa_connect_button.clicked.connect(self._connect_osa)
        self.osa_disconnect_button.clicked.connect(self._disconnect_osa)
        conn.addWidget(QtWidgets.QLabel("OSA Host"), 0, 0)
        conn.addWidget(self.osa_host_edit, 0, 1)
        conn.addWidget(QtWidgets.QLabel("Port"), 0, 2)
        conn.addWidget(self.osa_port_spin, 0, 3)
        conn.addWidget(self.osa_connect_button, 0, 4)
        conn.addWidget(self.osa_disconnect_button, 0, 5)
        conn.addWidget(self.osa_status_label, 0, 6)
        conn.setColumnStretch(1, 1)
        outer.addLayout(conn)

        # per-step widget registry: self.step_widgets[step][key]
        self.step_widgets: dict[int, dict[str, Any]] = {1: {}, 2: {}, 3: {}}

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._build_step1_tab(), "Step 1 · Min/Max")
        tabs.addTab(self._build_step2_tab(), "Step 2 · Wavelength")
        tabs.addTab(self._build_step3_tab(), "Step 3 · Intensity")
        outer.addWidget(tabs)

        # --- shared bottom row ---
        bottom = QtWidgets.QHBoxLayout()
        self.run_all_button = QtWidgets.QPushButton("Run All (1→2→3)")
        self.run_all_button.setEnabled(False)
        self.run_all_button.clicked.connect(self._run_all)
        self.stop_cal_button = QtWidgets.QPushButton("Stop")
        self.stop_cal_button.setProperty("variant", "danger")
        self.stop_cal_button.setEnabled(False)
        self.stop_cal_button.clicked.connect(self._stop_full_calibration)
        bottom.addStretch(1)
        bottom.addWidget(self.run_all_button)
        bottom.addWidget(self.stop_cal_button)
        outer.addLayout(bottom)

        # every Run button, toggled together by _set_calibration_running
        self.calibration_run_buttons = [
            self.step_widgets[1]["run"],
            self.step_widgets[2]["run"],
            self.step_widgets[3]["run"],
            self.run_all_button,
        ]
        return panel

    def _build_measurement_group(self, step: int, defaults: dict[str, str]) -> QtWidgets.QGroupBox:
        """OSA measurement settings (center λ / span / sensitivity / ref) for a step."""
        box = QtWidgets.QGroupBox("OSA settings")
        grid = QtWidgets.QGridLayout(box)
        widgets = self.step_widgets[step]
        widgets["center_wl"] = QtWidgets.QLineEdit(defaults.get("center_wl", "778nm"))
        widgets["span"] = QtWidgets.QLineEdit(defaults.get("span", "8nm"))
        widgets["sensitivity"] = QtWidgets.QComboBox()
        widgets["sensitivity"].addItems(["NORM", "MID", "HIGH1", "HIGH2", "HIGH3"])
        widgets["sensitivity"].setCurrentText(defaults.get("sensitivity", "HIGH2"))
        widgets["ref_level"] = QtWidgets.QLineEdit(defaults.get("ref_level", "10uW"))
        grid.addWidget(QtWidgets.QLabel("Center λ"), 0, 0)
        grid.addWidget(widgets["center_wl"], 0, 1)
        grid.addWidget(QtWidgets.QLabel("Span"), 0, 2)
        grid.addWidget(widgets["span"], 0, 3)
        grid.addWidget(QtWidgets.QLabel("Sensitivity"), 1, 0)
        grid.addWidget(widgets["sensitivity"], 1, 1)
        grid.addWidget(QtWidgets.QLabel("Ref level"), 1, 2)
        grid.addWidget(widgets["ref_level"], 1, 3)
        return box

    def _level_sweep_row(self, step: int, *, stop: int = 1023, stepv: int = 64) -> QtWidgets.QWidget:
        """A 'Levels start / stop / step' row stored on self.step_widgets[step]."""
        row = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        widgets = self.step_widgets[step]
        widgets["level_start"] = self._spin(0, 1023, 0)
        widgets["level_stop"] = self._spin(0, 1023, stop)
        widgets["level_step"] = self._spin(1, 1023, stepv)
        layout.addWidget(QtWidgets.QLabel("Levels"))
        layout.addWidget(widgets["level_start"])
        layout.addWidget(QtWidgets.QLabel("→"))
        layout.addWidget(widgets["level_stop"])
        layout.addWidget(QtWidgets.QLabel("step"))
        layout.addWidget(widgets["level_step"])
        layout.addStretch(1)
        return row

    def _output_row(self, step: int, key: str, label: str, default_name: str, is_csv: bool) -> QtWidgets.QWidget:
        """An output path edit + Browse, stored under self.step_widgets[step][key]."""
        row = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        edit = QtWidgets.QLineEdit()
        edit.setPlaceholderText(f"{label} (blank = temp file)")
        button = QtWidgets.QPushButton("Browse")
        filt = "CSV Files (*.csv)" if is_csv else "JSON Files (*.json)"
        button.clicked.connect(lambda: self._browse_save_into(edit, default_name, filt))
        self.step_widgets[step][key] = edit
        layout.addWidget(QtWidgets.QLabel(label))
        layout.addWidget(edit, 1)
        layout.addWidget(button)
        return row

    def _input_file_row(self, step: int, caption: str, filt: str) -> QtWidgets.QWidget:
        """An input path edit + Browse for a step, stored under [step]['in_path']."""
        row = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        edit = QtWidgets.QLineEdit()
        button = QtWidgets.QPushButton("Browse")
        button.clicked.connect(lambda: self._browse_open_into(edit, caption, filt))
        self.step_widgets[step]["in_path"] = edit
        layout.addWidget(QtWidgets.QLabel("Input file"))
        layout.addWidget(edit, 1)
        layout.addWidget(button)
        return row

    def _min_max_row(self, step: int, label: str) -> QtWidgets.QWidget:
        """A manual min/max level pair stored under [step]['min'] / [step]['max']."""
        row = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        widgets = self.step_widgets[step]
        widgets["min"] = self._spin(0, 1023, 0)
        widgets["max"] = self._spin(0, 1023, 1023)
        layout.addWidget(QtWidgets.QLabel(label))
        layout.addWidget(QtWidgets.QLabel("min"))
        layout.addWidget(widgets["min"])
        layout.addWidget(QtWidgets.QLabel("max"))
        layout.addWidget(widgets["max"])
        layout.addStretch(1)
        return row

    def _region_row(self, step: int) -> QtWidgets.QWidget:
        """A 'Limit region x start→end' toggle stored on self.step_widgets[step]."""
        row = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        widgets = self.step_widgets[step]
        check = QtWidgets.QCheckBox("Limit region")
        check.setToolTip(
            "Only sweep/calibrate this band of SLM columns (x). Off = full width "
            "(or, for a loaded map, its whole range)."
        )
        start = self._spin(0, 8191, 0)
        end = self._spin(0, 8191, 1919)
        start.setEnabled(False)
        end.setEnabled(False)
        check.toggled.connect(start.setEnabled)
        check.toggled.connect(end.setEnabled)
        widgets["region_check"] = check
        widgets["region_start"] = start
        widgets["region_end"] = end
        layout.addWidget(check)
        layout.addWidget(QtWidgets.QLabel("x"))
        layout.addWidget(start)
        layout.addWidget(QtWidgets.QLabel("→"))
        layout.addWidget(end)
        layout.addStretch(1)
        return row

    def _run_row(self, step: int, run_text: str, slot: Callable[[], None]) -> QtWidgets.QWidget:
        """A status label + Run button row, stored under [step]['status'] / ['run']."""
        row = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        status = QtWidgets.QLabel("\N{EN DASH}")
        button = QtWidgets.QPushButton(run_text)
        button.setEnabled(False)
        button.clicked.connect(slot)
        self.step_widgets[step]["status"] = status
        self.step_widgets[step]["run"] = button
        layout.addWidget(status, 1)
        layout.addWidget(button)
        return row

    def _build_step1_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.addWidget(
            self._caption("Sweep full-screen levels to find the darkest/brightest levels.")
        )
        layout.addWidget(self._build_measurement_group(1, {}))
        layout.addWidget(self._level_sweep_row(1, stop=1023, stepv=64))
        layout.addWidget(self._output_row(1, "out", "Output JSON", "calib_step1.json", False))
        layout.addWidget(self._run_row(1, "Run Step 1", self._run_step1))
        layout.addStretch(1)
        return page

    def _build_step2_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.addWidget(
            self._caption("Map x→wavelength with a bright window. Needs min/max levels.")
        )
        layout.addWidget(self._build_measurement_group(2, {}))

        cfg = QtWidgets.QHBoxLayout()
        widgets = self.step_widgets[2]
        widgets["window"] = self._spin(1, 8191, 8)
        widgets["peak_nm"] = self._double_spin(0.0, 50.0, 0.2, " nm", 3)
        widgets["peak_nm"].setToolTip("Centroid half-window around the peak, in nm")
        cfg.addWidget(QtWidgets.QLabel("Window px"))
        cfg.addWidget(widgets["window"])
        cfg.addWidget(QtWidgets.QLabel("Peak ± window"))
        cfg.addWidget(widgets["peak_nm"])
        cfg.addStretch(1)
        layout.addLayout(cfg)
        layout.addWidget(self._region_row(2))

        # input source
        src_row = QtWidgets.QHBoxLayout()
        widgets["source"] = QtWidgets.QComboBox()
        widgets["source"].addItems(
            ["Step 1 result (memory)", "From file…", "Manual min/max"]
        )
        widgets["source"].currentIndexChanged.connect(self._toggle_step2_source)
        src_row.addWidget(QtWidgets.QLabel("Min/max source"))
        src_row.addWidget(widgets["source"])
        src_row.addStretch(1)
        layout.addLayout(src_row)

        widgets["in_row"] = self._input_file_row(
            2, "Open Step 1/2 result", "JSON Files (*.json)"
        )
        layout.addWidget(widgets["in_row"])
        widgets["manual_row"] = self._min_max_row(2, "Manual levels")
        layout.addWidget(widgets["manual_row"])

        layout.addWidget(self._output_row(2, "out", "Output JSON", "calib_step2.json", False))
        layout.addWidget(self._run_row(2, "Run Step 2", self._run_step2))
        layout.addStretch(1)
        self._toggle_step2_source()
        return page

    def _build_step3_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.addWidget(
            self._caption(
                "Sweep levels at each calibrated wavelength. Narrow window + higher "
                "sensitivity = less noise."
            )
        )
        # higher precision defaults: HIGH3 + narrower span
        layout.addWidget(
            self._build_measurement_group(3, {"sensitivity": "HIGH3", "span": "4nm"})
        )

        cfg = QtWidgets.QHBoxLayout()
        widgets = self.step_widgets[3]
        widgets["window"] = self._spin(1, 8191, 3)
        widgets["avg_nm"] = self._double_spin(0.0, 50.0, 0.1, " nm", 3)
        widgets["avg_nm"].setToolTip("Averaging window around each wavelength, in nm")
        cfg.addWidget(QtWidgets.QLabel("Window px"))
        cfg.addWidget(widgets["window"])
        cfg.addWidget(QtWidgets.QLabel("Avg ± window"))
        cfg.addWidget(widgets["avg_nm"])
        cfg.addStretch(1)
        layout.addLayout(cfg)
        layout.addWidget(self._level_sweep_row(3, stop=1023, stepv=32))
        layout.addWidget(self._region_row(3))

        # wavelength source
        src_row = QtWidgets.QHBoxLayout()
        widgets["source"] = QtWidgets.QComboBox()
        widgets["source"].addItems(["Step 2 result (memory)", "From file…"])
        widgets["source"].currentIndexChanged.connect(self._toggle_step3_source)
        src_row.addWidget(QtWidgets.QLabel("Wavelength source"))
        src_row.addWidget(widgets["source"])
        src_row.addStretch(1)
        layout.addLayout(src_row)

        widgets["in_row"] = self._input_file_row(
            3, "Open Step 2 result or λ-map CSV", "Calibration (*.json *.csv)"
        )
        layout.addWidget(widgets["in_row"])
        widgets["manual_row"] = self._min_max_row(3, "min/max for CSV source")
        layout.addWidget(widgets["manual_row"])

        layout.addWidget(self._output_row(3, "out", "Output JSON", "calib_step3.json", False))
        layout.addWidget(self._output_row(3, "out_csv", "Output CSV", "calibration.csv", True))
        layout.addWidget(self._run_row(3, "Run Step 3", self._run_step3))
        layout.addStretch(1)
        self._toggle_step3_source()
        return page

    def _caption(self, text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setObjectName("PageSubtitle")
        label.setWordWrap(True)
        return label

    def _double_spin(
        self, minimum: float, maximum: float, value: float, suffix: str, decimals: int
    ) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setSingleStep(0.1)
        spin.setValue(value)
        spin.setSuffix(suffix)
        return spin

    def _build_scan_page(self) -> QtWidgets.QWidget:
        page = self._page_shell("Center Scan")

        controls = self._panel("Pattern")
        form = QtWidgets.QGridLayout(controls)
        self.scan_level_spin = self._spin(0, 1023, 512)
        self.bg_level_spin = self._spin(0, 1023, 0)
        self.bg_level_spin.setToolTip(
            "Grayscale level applied to every column outside the scan window"
        )
        self.window_px_spin = self._spin(1, 256, 5)
        self.step_px_spin = self._spin(1, 1024, 5)
        self.start_x_spin = self._spin(0, 8191, 0)
        self.end_x_spin = self._spin(0, 8191, 1919)
        self.dwell_spin = QtWidgets.QDoubleSpinBox()
        self.dwell_spin.setRange(0.01, 60.0)
        self.dwell_spin.setSingleStep(0.05)
        self.dwell_spin.setValue(0.2)
        self.dwell_spin.setSuffix(" s")

        self.detector_combo = QtWidgets.QComboBox()
        self.detector_combo.addItems(["None", "Simulated"])
        self.detector_combo.setToolTip(
            "Detector sampled at each scan position for center detection; "
            "real hardware can be plugged in via the Detector interface"
        )

        fields = [
            ("Level", self.scan_level_spin),
            ("Background", self.bg_level_spin),
            ("Window", self.window_px_spin),
            ("Step", self.step_px_spin),
            ("Start x", self.start_x_spin),
            ("End x", self.end_x_spin),
            ("Dwell", self.dwell_spin),
            ("Detector", self.detector_combo),
        ]
        for index, (label, widget) in enumerate(fields):
            row = index // 3
            col = (index % 3) * 2
            form.addWidget(QtWidgets.QLabel(label), row, col)
            form.addWidget(widget, row, col + 1)

        for widget in (
            self.scan_level_spin,
            self.bg_level_spin,
            self.window_px_spin,
            self.step_px_spin,
            self.start_x_spin,
            self.end_x_spin,
        ):
            widget.valueChanged.connect(self._update_scan_preview)

        # level/window/step/dwell can be adjusted while a scan runs;
        # changes take effect on the next frame
        self.scan_level_spin.valueChanged.connect(
            lambda value: self._on_scan_param_changed(level=value)
        )
        self.bg_level_spin.valueChanged.connect(
            lambda value: self._on_scan_param_changed(background_level=value)
        )
        self.window_px_spin.valueChanged.connect(
            lambda value: self._on_scan_param_changed(window_px=value)
        )
        self.step_px_spin.valueChanged.connect(
            lambda value: self._on_scan_param_changed(step_px=value)
        )
        self.dwell_spin.valueChanged.connect(
            lambda value: self._on_scan_param_changed(dwell_seconds=value)
        )

        output = self._panel("Output")
        output_layout = QtWidgets.QGridLayout(output)
        self.scan_output_edit = QtWidgets.QLineEdit()
        output_browse = QtWidgets.QPushButton("Browse")
        self.start_scan_button = QtWidgets.QPushButton("Start Scan")
        self.pause_scan_button = QtWidgets.QPushButton("Pause")
        self.pause_scan_button.setProperty("variant", "ghost")
        self.pause_scan_button.setEnabled(False)
        self.stop_scan_button = QtWidgets.QPushButton("Stop")
        self.stop_scan_button.setProperty("variant", "danger")
        self.stop_scan_button.setEnabled(False)
        output_browse.clicked.connect(self._browse_scan_output)
        self.start_scan_button.clicked.connect(self._start_center_scan)
        self.pause_scan_button.clicked.connect(self._toggle_scan_pause)
        self.stop_scan_button.clicked.connect(self._stop_center_scan)
        output_layout.addWidget(self.scan_output_edit, 0, 0)
        output_layout.addWidget(output_browse, 0, 1)
        output_layout.addWidget(self.start_scan_button, 0, 2)
        output_layout.addWidget(self.pause_scan_button, 0, 3)
        output_layout.addWidget(self.stop_scan_button, 0, 4)

        self.scan_size_label = QtWidgets.QLabel("Using preview size 1920 x 1200")
        self.scan_progress_bar = QtWidgets.QProgressBar()
        self.scan_progress_bar.setValue(0)
        status_row = QtWidgets.QHBoxLayout()
        self.scan_signal_label = QtWidgets.QLabel("Signal: \N{EN DASH}")
        self.scan_center_label = QtWidgets.QLabel("Center: \N{EN DASH}")
        self._set_status(self.scan_center_label, "Center: \N{EN DASH}", "off")
        status_row.addWidget(self.scan_size_label)
        status_row.addStretch(1)
        status_row.addWidget(self.scan_signal_label)
        status_row.addWidget(self.scan_center_label)

        self.preview_label = QtWidgets.QLabel()
        self.preview_label.setMinimumHeight(280)
        self.preview_label.setAlignment(QtCore.Qt.AlignCenter)
        self.preview_label.setObjectName("Preview")

        page.layout().addWidget(controls)
        page.layout().addWidget(output)
        page.layout().addLayout(status_row)
        page.layout().addWidget(self.scan_progress_bar)
        page.layout().addWidget(self.preview_label, 1)
        self._update_scan_preview()
        return page

    def _build_segments_page(self) -> QtWidgets.QWidget:
        page = self._page_shell("Phase Segments")
        subtitle = QtWidgets.QLabel(
            "Divide the x axis into vertical bands and assign a phase level "
            "to each (constant along y)."
        )
        subtitle.setObjectName("PageSubtitle")
        page.layout().addWidget(subtitle)

        controls = self._panel("Segments")
        controls_layout = QtWidgets.QGridLayout(controls)
        self.segment_mode_combo = QtWidgets.QComboBox()
        self.segment_mode_combo.addItems(["Equal division", "Explicit segments"])
        self.segment_count_spin = self._spin(1, 256, 4)
        self.segment_fill_spin = self._spin(0, MAX_LEVEL, 512)
        fill_button = QtWidgets.QPushButton("Set All Levels")
        fill_button.setProperty("variant", "ghost")
        add_row_button = QtWidgets.QPushButton("Add Row")
        add_row_button.setProperty("variant", "ghost")
        remove_row_button = QtWidgets.QPushButton("Remove Row")
        remove_row_button.setProperty("variant", "ghost")

        controls_layout.addWidget(QtWidgets.QLabel("Mode"), 0, 0)
        controls_layout.addWidget(self.segment_mode_combo, 0, 1)
        controls_layout.addWidget(QtWidgets.QLabel("Parts"), 0, 2)
        controls_layout.addWidget(self.segment_count_spin, 0, 3)
        controls_layout.addWidget(self.segment_fill_spin, 0, 4)
        controls_layout.addWidget(fill_button, 0, 5)
        controls_layout.addWidget(add_row_button, 0, 6)
        controls_layout.addWidget(remove_row_button, 0, 7)

        self.segments_table = QtWidgets.QTableWidget(0, 3)
        self.segments_table.setHorizontalHeaderLabels(["x start", "x end", "Level"])
        self.segments_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch
        )
        self.segments_table.verticalHeader().setVisible(False)
        self.segments_table.setAlternatingRowColors(True)
        self.segments_table.setMaximumHeight(220)

        actions = self._panel("Actions")
        actions_layout = QtWidgets.QGridLayout(actions)
        display_button = QtWidgets.QPushButton("Display on SLM")
        export_button = QtWidgets.QPushButton("Export CSV")
        export_button.setProperty("variant", "ghost")
        self.segment_status_label = QtWidgets.QLabel("")
        actions_layout.addWidget(display_button, 0, 0)
        actions_layout.addWidget(export_button, 0, 1)
        actions_layout.addWidget(self.segment_status_label, 0, 2)
        actions_layout.setColumnStretch(2, 1)

        self.segment_preview_label = QtWidgets.QLabel()
        self.segment_preview_label.setMinimumHeight(240)
        self.segment_preview_label.setAlignment(QtCore.Qt.AlignCenter)
        self.segment_preview_label.setObjectName("Preview")

        page.layout().addWidget(controls)
        page.layout().addWidget(self._panel_with_widget("Definition", self.segments_table))
        page.layout().addWidget(actions)
        page.layout().addWidget(self.segment_preview_label, 1)

        self.segment_mode_combo.currentIndexChanged.connect(self._on_segment_mode_changed)
        self.segment_count_spin.valueChanged.connect(self._rebuild_equal_segment_rows)
        fill_button.clicked.connect(self._fill_segment_levels)
        add_row_button.clicked.connect(self._add_segment_row)
        remove_row_button.clicked.connect(self._remove_segment_row)
        self.segments_table.itemChanged.connect(self._on_segment_item_changed)
        display_button.clicked.connect(self._display_segments)
        export_button.clicked.connect(self._export_segments_csv)

        self._segment_add_button = add_row_button
        self._segment_remove_button = remove_row_button
        self._rebuild_equal_segment_rows()
        self._on_segment_mode_changed()
        return page

    def _build_tpa_page(self) -> QtWidgets.QWidget:
        page = self._page_shell("TPA Encoding")
        panel = self._panel("Encoding")
        panel.setEnabled(False)
        layout = QtWidgets.QGridLayout(panel)
        layout.addWidget(QtWidgets.QLabel("Strategy"), 0, 0)
        layout.addWidget(QtWidgets.QLineEdit("TPA Multiplication"), 0, 1)
        layout.addWidget(QtWidgets.QPushButton("Encode"), 1, 1)
        page.layout().addWidget(panel)
        page.layout().addStretch(1)
        return page

    def _page_shell(self, title: str) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(26, 24, 26, 24)
        layout.setSpacing(18)
        heading = QtWidgets.QLabel(title)
        heading.setObjectName("PageTitle")
        layout.addWidget(heading)
        return page

    def _panel(self, title: str) -> QtWidgets.QGroupBox:
        panel = QtWidgets.QGroupBox(title)
        panel.setObjectName("Panel")
        return panel

    def _panel_with_widget(self, title: str, widget: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
        panel = self._panel(title)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.addWidget(widget)
        return panel

    def _spin(self, minimum: int, maximum: int, value: int) -> QtWidgets.QSpinBox:
        spin = QtWidgets.QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        return spin

    def _set_status(self, label: QtWidgets.QLabel, text: str, status: str) -> None:
        """Update a status pill label (status in: ok, error, off)."""
        label.setText(text)
        if label.property("status") != status:
            label.setProperty("status", status)
            label.style().unpolish(label)
            label.style().polish(label)

    def _controller(self) -> SLMController:
        display_no = self.display_no_spin.value()
        rate120 = self.rate120_check.isChecked()
        if self.controller is None or self.controller_display_no != display_no:
            try:
                controller = self.controller_factory(display_no, rate120=rate120)
            except TypeError:
                controller = self.controller_factory(display_no)
            self.controller = controller
            self.controller_display_no = display_no
        return self.controller

    def _reset_controller(self) -> None:
        old_controller = self.controller
        self.controller = None
        self.controller_display_no = None
        self._stop_keepalive()
        self._set_status(self.conn_status_label, "Status: closed", "off")
        if old_controller is not None and getattr(old_controller, "is_open", False):
            self._run_slm_task("Close previous SLM", old_controller.close_slm)

    def _run_task(
        self,
        label: str,
        func: Callable[[], Any],
        on_success: Callable[[Any], None] | None = None,
        on_error: Callable[[str], None] | None = None,
    ) -> FunctionWorker:
        self._log(f"{label} started")
        worker = FunctionWorker(func)
        self._workers.add(worker)

        def finish(result: Any) -> None:
            self._workers.discard(worker)
            self._finish_task(label, result, on_success)

        def fail(error: str) -> None:
            self._workers.discard(worker)
            self._fail_task(label, error, on_error)

        worker.signals.finished.connect(finish)
        worker.signals.error.connect(fail)
        self.thread_pool.start(worker)
        return worker

    def _run_slm_task(
        self,
        label: str,
        func: Callable[[], Any],
        on_success: Callable[[Any], None] | None = None,
        on_error: Callable[[str], None] | None = None,
    ) -> FunctionWorker:
        self._slm_tasks_active += 1
        self._sync_keepalive_state()

        def finish_slm_task() -> None:
            self._slm_tasks_active = max(0, self._slm_tasks_active - 1)
            self._sync_keepalive_state()

        def finish(result: Any) -> None:
            try:
                if on_success is not None:
                    on_success(result)
            finally:
                finish_slm_task()

        def fail(error: str) -> None:
            try:
                if on_error is not None:
                    on_error(error)
            finally:
                finish_slm_task()

        return self._run_task(label, func, finish, fail)

    def _finish_task(
        self,
        label: str,
        result: Any,
        on_success: Callable[[Any], None] | None,
    ) -> None:
        self._log(f"{label} complete")
        if on_success is not None:
            on_success(result)
        self._refresh_conn_status()

    def _refresh_conn_status(self) -> None:
        is_open = self.controller is not None and getattr(self.controller, "is_open", False)
        if is_open:
            self._set_status(self.conn_status_label, "Status: open", "ok")
        else:
            self._set_status(self.conn_status_label, "Status: closed", "off")

    def _fail_task(
        self,
        label: str,
        error: str,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        self._log(f"{label} failed")
        self._log(error)
        if on_error is not None:
            on_error(error)
        QtWidgets.QMessageBox.critical(self, label, error)

    def _log(self, message: str) -> None:
        if hasattr(self, "log_box"):
            self.log_box.appendPlainText(message.rstrip())
        self.statusBar().showMessage(message.splitlines()[0], 6000)

    def _open_slm(self) -> None:
        controller = self._controller()
        self._run_slm_task("Open SLM", controller.open_slm)

    def _close_slm(self) -> None:
        self._stop_keepalive()
        controller = self._controller()
        self._run_slm_task("Close SLM", controller.close_slm)

    def _detect_slm(self) -> None:
        controller = self._controller()
        self._run_slm_task(
            "Detect SLM",
            controller.detect_displays,
            self._on_detect,
        )

    def _on_detect(self, displays: list[tuple[int, int, int, str]]) -> None:
        if not displays:
            self._log("No displays found")
            return
        slm_no = None
        for display_no, width, height, name in displays:
            self._log(f"Display {display_no}: {width} x {height} ({name})")
            if slm_no is None and name.startswith("LCOS-SLM"):
                slm_no = display_no
        if slm_no is None:
            self._log("No LCOS-SLM display found; check connection and mode")
            return
        self._log(f"LCOS-SLM found on display {slm_no}")
        self.display_no_spin.setValue(slm_no)

    def _switch_to_dvi_mode(self) -> None:
        slm_number = self.usb_slm_no_spin.value()
        controller = self._controller()
        self._run_slm_task(
            "Switch to DVI mode",
            lambda: controller.set_dvi_mode(slm_number),
        )

    def _read_slm_info(self) -> None:
        controller = self._controller()
        self._run_slm_task(
            "Read SLM info",
            controller.get_slm_info,
            self._on_info_read,
        )

    def _toggle_keepalive(self, checked: bool) -> None:
        if checked:
            self._start_keepalive()
        else:
            self._stop_keepalive()

    def _start_keepalive(self) -> None:
        if self.keepalive is not None and self.keepalive.is_running:
            return
        # capture the controller on the GUI thread; the heartbeat thread
        # must not touch widgets
        controller = self._controller()
        interval = self.keepalive_interval_spin.value()
        self.keepalive = SLMKeepAlive(
            # re-send the last displayed pattern so the DVI link stays active
            ping=lambda: controller.refresh_display(),
            interval_seconds=interval,
            on_status=lambda ok, message: self.keepalive_status.emit(ok, message),
        )
        self.keepalive.start()
        self._sync_keepalive_state()
        self._set_status(
            self.keepalive_status_label,
            f"Keep-alive: every {self._format_seconds(interval)}",
            "ok",
        )
        self._log(
            f"DVI keep-alive started (re-send pattern every "
            f"{self._format_seconds(interval)})"
        )

    def _stop_keepalive(self) -> None:
        if self.keepalive is not None:
            stopped = self.keepalive.stop()
            if stopped:
                self.keepalive = None
                self._log("Keep-alive stopped")
            else:
                self._log("Keep-alive stop requested; worker is still finishing")
        if hasattr(self, "keepalive_status_label"):
            self._set_status(self.keepalive_status_label, "Keep-alive: off", "off")
        if hasattr(self, "keepalive_check") and self.keepalive_check.isChecked():
            self.keepalive_check.blockSignals(True)
            self.keepalive_check.setChecked(False)
            self.keepalive_check.blockSignals(False)

    def _on_keepalive_interval(self, value: float) -> None:
        if self.keepalive is not None and self.keepalive.is_running:
            self.keepalive.set_interval(value)
            self._set_status(
                self.keepalive_status_label,
                f"Keep-alive: every {self._format_seconds(value)}",
                "ok",
            )

    def _format_seconds(self, seconds: float) -> str:
        return f"{seconds:g} s"

    def _sync_keepalive_state(self) -> None:
        if self.keepalive is None:
            return
        scan_active = self.scan_stop_event is not None
        scan_paused = (
            self.scan_pause_event is not None and self.scan_pause_event.is_set()
        )
        if self._slm_tasks_active > 0 or (scan_active and not scan_paused):
            self.keepalive.suspend()
        else:
            self.keepalive.resume()

    def _on_keepalive_status(self, ok: bool, message: str) -> None:
        timestamp = QtCore.QTime.currentTime().toString("HH:mm:ss")
        if ok:
            self._set_status(
                self.keepalive_status_label, f"Keep-alive: ok {timestamp}", "ok"
            )
        else:
            self._set_status(
                self.keepalive_status_label, f"Keep-alive: error {timestamp}", "error"
            )
            self._log(f"Keep-alive refresh failed: {message}")

    def _on_info_read(self, result: tuple[int, int]) -> None:
        width, height = result
        self.slm_size = (int(width), int(height))
        self.info_label.setText(f"Size: {width} x {height}")
        self.scan_size_label.setText(f"Using SLM size {width} x {height}")
        self.start_x_spin.setMaximum(width - 1)
        self.end_x_spin.setMaximum(width - 1)
        self.end_x_spin.setValue(width - 1)
        # keep the calibration region spinners bounded to the real SLM width
        for step in (2, 3):
            widgets = getattr(self, "step_widgets", {}).get(step, {})
            if "region_end" in widgets:
                widgets["region_start"].setMaximum(width - 1)
                widgets["region_end"].setMaximum(width - 1)
                if not widgets["region_check"].isChecked():
                    widgets["region_end"].setValue(width - 1)
        self._update_scan_preview()
        if self._segment_mode_is_equal():
            self._rebuild_equal_segment_rows()
        else:
            self._update_segment_preview()

    def _display_grayscale(self) -> None:
        value = self.gray_spin.value()
        controller = self._controller()
        self._run_slm_task(
            "Display grayscale",
            lambda: controller.display_grayscale(value),
        )

    def _browse_display_csv(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select SLM CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        if path:
            self.csv_path_edit.setText(path)

    def _display_csv(self) -> None:
        path = self.csv_path_edit.text().strip()
        if not path:
            self._log("Select a CSV file first")
            return
        controller = self._controller()
        self._run_slm_task("Display CSV", lambda: controller.display_csv(path))

    def _browse_calibration_csv(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Calibration CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        if path:
            self.calibration_path_edit.setText(path)

    def _run_calibration_fit(self) -> None:
        path = self.calibration_path_edit.text().strip()
        if not path:
            self._log("Select a calibration CSV first")
            return

        def fit_file() -> dict[float, CalibrationFit]:
            points = load_calibration_csv(path)
            return fit_calibration(points)

        self._run_task("Calibration fit", fit_file, self._on_calibration_fit)

    def _on_calibration_fit(self, fits: dict[float, CalibrationFit]) -> None:
        self.calibration_fits = fits
        self.wavelength_combo.blockSignals(True)
        self.wavelength_combo.clear()
        for wavelength in fits:
            self.wavelength_combo.addItem(f"{wavelength:g} nm", wavelength)
        self.wavelength_combo.blockSignals(False)
        self.save_fit_button.setEnabled(True)
        self._update_calibration_view()

    def _update_calibration_view(self) -> None:
        if not self.calibration_fits or self.wavelength_combo.count() == 0:
            return
        wavelength = float(self.wavelength_combo.currentData())
        fit = self.calibration_fits[wavelength]

        rows = [
            ("wavelength_nm", fit.wavelength_nm),
            ("I0", fit.i0),
            ("phase_slope", fit.phase_slope),
            ("phase_offset", fit.phase_offset),
            ("RMSE", fit.rmse),
            ("R2", fit.r_squared),
        ]
        self.fit_table.setRowCount(len(rows))
        for row, (name, value) in enumerate(rows):
            self.fit_table.setItem(row, 0, QtWidgets.QTableWidgetItem(name))
            self.fit_table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{value:.8g}"))
        self.fit_table.resizeColumnsToContents()

        self.figure.clear()
        axes = self.figure.add_subplot(111)
        axes.set_facecolor("#101820")
        axes.scatter(fit.levels, fit.intensities, color="#47b8e0", label="Measured", s=32)
        axes.plot(fit.levels, fit.fitted_intensities, color="#f5c542", label="Fit", linewidth=2)
        axes.set_xlabel("Level")
        axes.set_ylabel("Intensity")
        axes.grid(True, color="#2b3a42", linewidth=0.7)
        axes.legend()
        self.figure.patch.set_facecolor("#101820")
        axes.tick_params(colors="#d8dee9")
        axes.xaxis.label.set_color("#d8dee9")
        axes.yaxis.label.set_color("#d8dee9")
        for spine in axes.spines.values():
            spine.set_color("#41515c")
        self.canvas.draw_idle()

    def _save_calibration_result(self) -> None:
        if not self.calibration_fits:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Calibration Result", "calibration_fit.json", "JSON Files (*.json)"
        )
        if not path:
            return
        payload = {
            f"{wavelength:g}": fit.to_dict()
            for wavelength, fit in self.calibration_fits.items()
        }
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
        self._log(f"Saved calibration result: {path}")

    # ----- OSA-driven acquisition -----
    def _browse_save_into(self, edit: QtWidgets.QLineEdit, default_name: str, filt: str) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Select output", default_name, filt
        )
        if path:
            edit.setText(path)

    def _browse_open_into(self, edit: QtWidgets.QLineEdit, caption: str, filt: str) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, caption, "", filt)
        if path:
            edit.setText(path)

    def _toggle_step2_source(self) -> None:
        index = self.step_widgets[2]["source"].currentIndex()
        self.step_widgets[2]["in_row"].setVisible(index == 1)
        self.step_widgets[2]["manual_row"].setVisible(index == 2)

    def _toggle_step3_source(self) -> None:
        index = self.step_widgets[3]["source"].currentIndex()
        self.step_widgets[3]["in_row"].setVisible(index == 1)
        # manual min/max only matter for a bare wavelength-map CSV source
        self.step_widgets[3]["manual_row"].setVisible(index == 1)

    def _connect_osa(self) -> None:
        host = self.osa_host_edit.text().strip()
        if not host:
            self._log("Enter the OSA host first")
            return
        port = self.osa_port_spin.value()
        self.osa_connect_button.setEnabled(False)

        def connect() -> tuple[OSAController, str]:
            osa = OSAController(host=host, port=port)
            osa.connect()
            return osa, osa.identify()

        self._run_task("Connect OSA", connect, self._on_osa_connected, self._on_osa_error)

    def _on_osa_connected(self, payload: tuple[OSAController, str]) -> None:
        osa, identity = payload
        self.osa_controller = osa
        self._set_status(self.osa_status_label, "OSA: open", "ok")
        self._set_calibration_running(False)
        self._log(f"OSA connected: {identity.strip()}")

    def _on_osa_error(self, _error: str) -> None:
        self._set_status(self.osa_status_label, "OSA: error", "error")
        self._set_calibration_running(False)

    def _disconnect_osa(self) -> None:
        osa = self.osa_controller
        self.osa_controller = None
        self._set_status(self.osa_status_label, "OSA: closed", "off")
        self._set_calibration_running(False)
        if osa is not None:
            self._run_task("Disconnect OSA", osa.disconnect)

    def _set_calibration_running(self, running: bool) -> None:
        connected = self.osa_controller is not None
        for button in getattr(self, "calibration_run_buttons", []):
            button.setEnabled(connected and not running)
        self.stop_cal_button.setEnabled(running)
        self.osa_connect_button.setEnabled(not running and not connected)
        self.osa_disconnect_button.setEnabled(not running and connected)

    # ----- per-step config readers (GUI thread) -----
    def _step_settings(self, step: int) -> MeasurementSettings:
        widgets = self.step_widgets[step]
        return MeasurementSettings(
            center_wl=widgets["center_wl"].text().strip() or "778nm",
            span=widgets["span"].text().strip() or "8nm",
            sensitivity=widgets["sensitivity"].currentText(),
            reference_level=widgets["ref_level"].text().strip() or "10uW",
            y_unit="LINear",
        )

    def _step_levels(self, step: int) -> list[int]:
        widgets = self.step_widgets[step]
        start = widgets["level_start"].value()
        stop = widgets["level_stop"].value()
        step_size = widgets["level_step"].value()
        if stop < start:
            raise ValueError("level stop must be >= level start")
        levels = list(range(start, stop + 1, step_size))
        if not levels:
            levels = [start]
        if levels[-1] != stop:
            levels.append(stop)
        return levels

    def _step_region(self, step: int) -> tuple[int, int] | None:
        widgets = self.step_widgets[step]
        if not widgets["region_check"].isChecked():
            return None
        start = widgets["region_start"].value()
        end = widgets["region_end"].value()
        if end < start:
            raise ValueError("region end must be >= region start")
        return (start, end)

    def _resolve_output_path(self, text: str, default_name: str) -> Path:
        text = text.strip()
        if text:
            return Path(text)
        suffix = Path(default_name).suffix or ".json"
        handle = tempfile.NamedTemporaryFile(
            mode="w", suffix=suffix, prefix="santec_calib_", delete=False
        )
        handle.close()
        return Path(handle.name)

    def _resolve_step_input(self, step: int) -> CalibrationResult:
        widgets = self.step_widgets[step]
        index = widgets["source"].currentIndex()
        if step == 2:
            if index == 2:  # manual min/max
                low = widgets["min"].value()
                high = widgets["max"].value()
                if high < low:
                    raise ValueError("max level must be >= min level")
                return CalibrationResult(
                    wavelength=np.asarray([]),
                    coordinates=np.asarray([]),
                    max_level=high,
                    min_level=low,
                    level_range=np.asarray([], dtype=int),
                )
            result = self._load_input_result(
                index,
                widgets["in_path"].text().strip(),
                "run Step 1 first, or choose a file / manual min/max",
                "choose a Step 1/2 result file",
            )
            self._require_levels(result)
            return result

        # step 3 wavelength source
        if index == 1:  # from file (JSON snapshot or coordinate-wavelength CSV)
            path = widgets["in_path"].text().strip()
            if not path:
                raise ValueError("choose a Step 2 result or wavelength-map CSV")
            if path.lower().endswith(".csv"):
                result = load_wavelength_map_csv(
                    path,
                    min_level=widgets["min"].value(),
                    max_level=widgets["max"].value(),
                )
            else:
                result = load_calibration_result(path)
        else:  # memory
            result = self.calibration_result
            if result is None:
                raise ValueError("run Step 2 first, or choose a file")
        if (
            np.asarray(result.coordinates).size == 0
            or np.asarray(result.wavelength).size == 0
        ):
            raise ValueError("the wavelength source has no coordinate -> wavelength map")
        self._require_levels(result)
        return result

    def _load_input_result(
        self, index: int, path: str, empty_msg: str, no_path_msg: str
    ) -> CalibrationResult:
        if index == 1:  # from file
            if not path:
                raise ValueError(no_path_msg)
            return load_calibration_result(path)
        result = self.calibration_result  # in memory
        if result is None:
            raise ValueError(empty_msg)
        return result

    def _require_levels(self, result: CalibrationResult) -> None:
        try:
            int(np.asarray(result.min_level).flat[0])
            int(np.asarray(result.max_level).flat[0])
        except (ValueError, IndexError, TypeError):
            raise ValueError("min/max levels are missing from the input")

    def _reject_calibration(self, exc: Exception) -> None:
        self._log(f"Calibration input rejected: {exc}")
        QtWidgets.QMessageBox.warning(self, "Calibration", str(exc))

    # ----- per-step run handlers -----
    def _osa_ready(self) -> OSAController | None:
        osa = self.osa_controller
        if osa is None or not osa.is_connected:
            self._log("Connect to the OSA first")
            return None
        return osa

    def _run_step1(self) -> None:
        osa = self._osa_ready()
        if osa is None:
            return
        try:
            settings = self._step_settings(1)
            levels = self._step_levels(1)
        except ValueError as exc:
            return self._reject_calibration(exc)
        out_path = self._resolve_output_path(
            self.step_widgets[1]["out"].text(), "calib_step1.json"
        )
        controller = self._controller()
        self._log(f"Step 1 started: {len(levels)} levels")

        def work(report: ProgressEmit, stop_event: threading.Event) -> dict[str, Any]:
            _mn, _mx, min_level, max_level, _rec = find_min_max_intensity_levels(
                osa, controller, levels, settings,
                stop_event=stop_event, progress_callback=report,
            )
            result = CalibrationResult(
                wavelength=np.asarray([]),
                coordinates=np.asarray([]),
                max_level=max_level,
                min_level=min_level,
                level_range=np.asarray(levels, dtype=int),
            )
            save_calibration_result(result, out_path)
            return {
                "status": "ok", "step": 1, "result": result, "saved": out_path,
                "summary": f"min level {min_level}, max level {max_level}",
            }

        self._launch_calibration("Run step 1", work)

    def _run_step2(self) -> None:
        osa = self._osa_ready()
        if osa is None:
            return
        try:
            settings = self._step_settings(2)
            seed = self._resolve_step_input(2)
            window = self.step_widgets[2]["window"].value()
            peak_nm = self.step_widgets[2]["peak_nm"].value() or None
            region = self._step_region(2)
        except ValueError as exc:
            return self._reject_calibration(exc)
        out_path = self._resolve_output_path(
            self.step_widgets[2]["out"].text(), "calib_step2.json"
        )
        controller = self._controller()
        self._log(f"Step 2 started: window {window} px")

        def work(report: ProgressEmit, stop_event: threading.Event) -> dict[str, Any]:
            result = wavelength_calibration(
                osa, controller, [], settings, seed,
                window_size=window, peak_half_window_nm=peak_nm, region=region,
                stop_event=stop_event, progress_callback=report,
            )
            save_calibration_result(result, out_path)
            return {
                "status": "ok", "step": 2, "result": result, "saved": out_path,
                "summary": f"{result.coordinates.size} coordinates",
            }

        self._launch_calibration("Run step 2", work)

    def _run_step3(self) -> None:
        osa = self._osa_ready()
        if osa is None:
            return
        try:
            settings = self._step_settings(3)
            mapping = self._resolve_step_input(3)
            levels = self._step_levels(3)
            window = self.step_widgets[3]["window"].value()
            avg_nm = self.step_widgets[3]["avg_nm"].value() or None
            region = self._step_region(3)
        except ValueError as exc:
            return self._reject_calibration(exc)
        out_json = self._resolve_output_path(
            self.step_widgets[3]["out"].text(), "calib_step3.json"
        )
        out_csv = self._resolve_output_path(
            self.step_widgets[3]["out_csv"].text(), "calibration.csv"
        )
        controller = self._controller()
        self._log(f"Step 3 started: {len(levels)} levels, window {window} px")

        def work(report: ProgressEmit, stop_event: threading.Event) -> dict[str, Any]:
            result = intensity_calibration(
                osa, controller, levels, settings, mapping,
                window_size=window, wavelength_window_nm=avg_nm, region=region,
                stop_event=stop_event, progress_callback=report,
            )
            save_calibration_result(result, out_json)
            csv_path = write_intensity_calibration_csv(result, out_csv)
            return {
                "status": "ok", "step": 3, "result": result, "saved": out_json,
                "csv": csv_path, "summary": f"{result.coordinates.size} coordinates",
            }

        self._launch_calibration("Run step 3", work)

    def _run_all(self) -> None:
        osa = self._osa_ready()
        if osa is None:
            return
        try:
            s1 = self._step_settings(1)
            levels1 = self._step_levels(1)
            s2 = self._step_settings(2)
            window2 = self.step_widgets[2]["window"].value()
            peak_nm = self.step_widgets[2]["peak_nm"].value() or None
            region2 = self._step_region(2)
            s3 = self._step_settings(3)
            levels3 = self._step_levels(3)
            window3 = self.step_widgets[3]["window"].value()
            avg_nm = self.step_widgets[3]["avg_nm"].value() or None
            region3 = self._step_region(3)
        except ValueError as exc:
            return self._reject_calibration(exc)
        out1 = self._resolve_output_path(self.step_widgets[1]["out"].text(), "calib_step1.json")
        out2 = self._resolve_output_path(self.step_widgets[2]["out"].text(), "calib_step2.json")
        out3 = self._resolve_output_path(self.step_widgets[3]["out"].text(), "calib_step3.json")
        out_csv = self._resolve_output_path(
            self.step_widgets[3]["out_csv"].text(), "calibration.csv"
        )
        controller = self._controller()
        self._log("Run all started (steps 1 -> 2 -> 3)")

        def work(report: ProgressEmit, stop_event: threading.Event) -> dict[str, Any]:
            _mn, _mx, min_level, max_level, _rec = find_min_max_intensity_levels(
                osa, controller, levels1, s1,
                stop_event=stop_event, progress_callback=report,
            )
            seed = CalibrationResult(
                wavelength=np.asarray([]), coordinates=np.asarray([]),
                max_level=max_level, min_level=min_level,
                level_range=np.asarray(levels1, dtype=int),
            )
            save_calibration_result(seed, out1)
            wl_result = wavelength_calibration(
                osa, controller, [], s2, seed,
                window_size=window2, peak_half_window_nm=peak_nm, region=region2,
                stop_event=stop_event, progress_callback=report,
            )
            save_calibration_result(wl_result, out2)
            final = intensity_calibration(
                osa, controller, levels3, s3, wl_result,
                window_size=window3, wavelength_window_nm=avg_nm, region=region3,
                stop_event=stop_event, progress_callback=report,
            )
            save_calibration_result(final, out3)
            csv_path = write_intensity_calibration_csv(final, out_csv)
            return {
                "status": "ok", "step": "all", "result": final, "saved": out3,
                "csv": csv_path,
                "summary": (
                    f"min {min_level}, max {max_level}, "
                    f"{final.coordinates.size} coordinates"
                ),
            }

        self._launch_calibration("Run all", work)

    def _launch_calibration(
        self,
        label: str,
        work: Callable[[ProgressEmit, threading.Event], dict[str, Any]],
    ) -> None:
        stop_event = threading.Event()
        self.calibration_stop_event = stop_event
        self._set_calibration_running(True)
        self._open_calibration_dialog()

        # the callback runs on the worker thread, so hop to the GUI thread
        def report(progress: CalibrationProgress) -> None:
            self.calibration_progress.emit(progress)

        def run() -> dict[str, Any]:
            try:
                return work(report, stop_event)
            except CalibrationAborted:
                # report as an ordinary result so no error dialog is shown
                return {"status": "aborted"}

        # treat acquisition as an SLM task so the DVI keep-alive is suspended
        self._run_slm_task(label, run, self._on_step_finished, self._on_step_error)

    def _open_calibration_dialog(self) -> None:
        if self.calibration_dialog is not None:
            self.calibration_dialog.close()
        dialog = CalibrationProgressDialog(self, on_stop=self._stop_full_calibration)
        dialog.setStyleSheet(DARK_STYLESHEET)
        dialog.finished.connect(self._on_calibration_dialog_closed)
        self.calibration_dialog = dialog
        dialog.show()

    def _on_calibration_dialog_closed(self, _result: int) -> None:
        self.calibration_dialog = None

    def _on_calibration_progress(self, progress: CalibrationProgress) -> None:
        if self.calibration_dialog is not None:
            self.calibration_dialog.update_progress(progress)

    def _stop_full_calibration(self) -> None:
        if self.calibration_stop_event is not None:
            self.calibration_stop_event.set()
            self._log("Calibration stop requested")

    def _on_step_finished(self, payload: dict[str, Any]) -> None:
        self.calibration_stop_event = None
        self._set_calibration_running(False)
        if payload.get("status") == "aborted":
            self._log("Calibration stopped")
            if self.calibration_dialog is not None:
                self.calibration_dialog.finish(False, "Calibration stopped")
            return

        step = payload["step"]
        result = payload["result"]
        summary = payload.get("summary", "")
        saved = payload.get("saved")
        self.calibration_result = result

        if step in (1, 2, 3):
            self.step_widgets[step]["status"].setText(f"Done \N{MIDDLE DOT} {summary}")
            out_edit = self.step_widgets[step]["out"]
            if saved is not None and not out_edit.text().strip():
                out_edit.setText(str(saved))
        if saved is not None:
            self._log(f"Saved {saved}")

        label = "Run all" if step == "all" else f"Step {step}"
        self._log(f"{label} done: {summary}")

        csv_path = payload.get("csv")
        if csv_path is not None:
            self._log(f"Calibration CSV saved: {csv_path}")
            self.calibration_path_edit.setText(str(csv_path))
            self.map_kind_combo.setCurrentIndex(0)
            self._update_intensity_map()
            # feed the freshly written CSV into the existing fit + plot flow
            self._run_calibration_fit()

        if self.calibration_dialog is not None:
            self.calibration_dialog.finish(True, f"{label} done \N{MIDDLE DOT} {summary}")

    def _on_step_error(self, _error: str) -> None:
        # _fail_task already logged the traceback and showed a dialog
        self.calibration_stop_event = None
        self._set_calibration_running(False)
        if self.calibration_dialog is not None:
            self.calibration_dialog.finish(False, "Calibration failed")

    def _style_dark_axes(self, axes: Any) -> None:
        axes.set_facecolor("#101820")
        axes.grid(True, color="#2b3a42", linewidth=0.7)
        axes.tick_params(colors="#d8dee9")
        axes.xaxis.label.set_color("#d8dee9")
        axes.yaxis.label.set_color("#d8dee9")
        for spine in axes.spines.values():
            spine.set_color("#41515c")

    def _update_intensity_map(self) -> None:
        if not hasattr(self, "map_canvas"):
            return
        self.map_figure.clear()
        self.map_figure.patch.set_facecolor("#101820")
        axes = self.map_figure.add_subplot(111)
        self._style_dark_axes(axes)

        result = self.calibration_result
        if result is None or result.intensity_levels is None:
            axes.text(
                0.5,
                0.5,
                "Run a calibration to see the intensity map",
                ha="center",
                va="center",
                color="#d8dee9",
                transform=axes.transAxes,
            )
            self.map_canvas.draw_idle()
            return

        raw = self.map_kind_combo.currentText().startswith("Raw")
        data = result.raw_intensity_levels if raw else result.intensity_levels
        if data is None:
            axes.text(
                0.5,
                0.5,
                "Raw intensity map is not available",
                ha="center",
                va="center",
                color="#d8dee9",
                transform=axes.transAxes,
            )
            self.map_canvas.draw_idle()
            return

        data = np.asarray(data, dtype=float)
        levels = np.asarray(result.level_range, dtype=float)
        wavelengths = np.asarray(result.wavelength, dtype=float)
        extent = [
            float(levels.min()),
            float(levels.max()),
            float(wavelengths.min()),
            float(wavelengths.max()),
        ]
        if extent[0] == extent[1]:
            extent[1] += 1.0
        if extent[2] == extent[3]:
            extent[3] += 1.0
        image = axes.imshow(
            data,
            aspect="auto",
            origin="lower",
            extent=(extent[0], extent[1], extent[2], extent[3]),
            cmap="viridis",
        )
        axes.set_xlabel("Level")
        axes.set_ylabel("Wavelength (nm)")
        colorbar = self.map_figure.colorbar(image, ax=axes)
        colorbar.set_label("Intensity (W)" if raw else "Normalized intensity")
        colorbar.ax.yaxis.set_tick_params(color="#d8dee9")
        colorbar.ax.yaxis.label.set_color("#d8dee9")
        for label in colorbar.ax.get_yticklabels():
            label.set_color("#d8dee9")
        self.map_canvas.draw_idle()

    def _browse_scan_output(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if path:
            self.scan_output_edit.setText(path)

    def _make_detector(self, start_x: int, end_x: int) -> Detector | None:
        """Build the selected detector; extend here for real hardware."""
        choice = self.detector_combo.currentText()
        if choice == "Simulated":
            span = max(end_x - start_x, 1)
            return SimulatedDetector(
                center_x=(start_x + end_x) / 2.0,
                sigma_px=max(span / 8.0, 1.0),
            )
        return None

    def _start_center_scan(self) -> None:
        start_x = self.start_x_spin.value()
        end_x = self.end_x_spin.value()
        output_dir = self.scan_output_edit.text().strip() or None

        try:
            params = ScanParams(
                self.scan_level_spin.value(),
                window_px=self.window_px_spin.value(),
                step_px=self.step_px_spin.value(),
                dwell_seconds=self.dwell_spin.value(),
                background_level=self.bg_level_spin.value(),
            )
        except ValueError as exc:
            self._log(f"Invalid scan parameters: {exc}")
            return

        detector = self._make_detector(start_x, end_x)

        self.scan_progress_bar.setValue(0)
        self.scan_signal_label.setText("Signal: \N{EN DASH}")
        self._set_status(self.scan_center_label, "Center: \N{EN DASH}", "off")
        self.scan_params = params
        self.scan_stop_event = threading.Event()
        self.scan_pause_event = threading.Event()
        self.start_scan_button.setEnabled(False)
        self.pause_scan_button.setEnabled(True)
        self.pause_scan_button.setText("Pause")
        self.stop_scan_button.setEnabled(True)
        self._sync_keepalive_state()

        stop_event = self.scan_stop_event
        pause_event = self.scan_pause_event
        controller = self._controller()

        def run_scan() -> ScanResult:
            width, height = controller.get_slm_info()
            clamped_start = min(start_x, width - 1)
            clamped_end = min(end_x, width - 1)
            self.scan_started.emit(clamped_start, clamped_end, width, height)
            return controller.run_center_scan(
                params,
                start_x=clamped_start,
                end_x=clamped_end,
                output_dir=output_dir,
                stop_event=stop_event,
                pause_event=pause_event,
                detector=detector,
                progress_callback=lambda index, x, path: self.scan_progress.emit(
                    index, x, str(path)
                ),
                sample_callback=lambda x, signal: self.scan_sample.emit(x, signal),
            )

        self._run_task("Center scan", run_scan, self._on_scan_finished, self._on_scan_error)

    def _stop_center_scan(self) -> None:
        if self.scan_stop_event is not None:
            self.scan_stop_event.set()
            self._log("Center scan stop requested")

    def _toggle_scan_pause(self) -> None:
        if self.scan_pause_event is None:
            return
        if self.scan_pause_event.is_set():
            self.scan_pause_event.clear()
            self.pause_scan_button.setText("Pause")
            # the scan streams frames again, so the heartbeat can rest
            self._sync_keepalive_state()
            self._log("Center scan resumed")
        else:
            self.scan_pause_event.set()
            self.pause_scan_button.setText("Resume")
            # no frames flow while paused; let the heartbeat keep DVI active
            self._sync_keepalive_state()
            self._log("Center scan paused")

    def _on_scan_param_changed(self, **kwargs: Any) -> None:
        params = self.scan_params
        if params is None:
            return
        try:
            params.update(**kwargs)
        except ValueError as exc:
            self._log(f"Scan parameter rejected: {exc}")
            return
        name, value = next(iter(kwargs.items()))
        self._log(f"Scan parameter updated for next frame: {name} = {value}")

    def _on_scan_started(self, start_x: int, end_x: int, width: int, height: int) -> None:
        self._scan_x_range = (start_x, end_x)
        # progress tracks the x position, which stays correct when the step
        # size is changed mid-scan
        self.scan_progress_bar.setMaximum(max(end_x - start_x + 1, 1))
        self.slm_size = (width, height)
        self.scan_size_label.setText(f"Using SLM size {width} x {height}")

    def _on_scan_progress(self, index: int, x: int, path: str) -> None:
        start_x, _end_x = self._scan_x_range
        self.scan_progress_bar.setValue(max(x - start_x + 1, 0))
        self._log(f"Displayed frame {index + 1} at x={x} ({Path(path).name})")

    def _on_scan_sample(self, x: float, signal: float) -> None:
        self.scan_signal_label.setText(f"Signal: {signal:.4g} at x={x:.1f}")

    def _finish_scan_ui(self) -> None:
        self.start_scan_button.setEnabled(True)
        self.pause_scan_button.setEnabled(False)
        self.pause_scan_button.setText("Pause")
        self.stop_scan_button.setEnabled(False)
        self.scan_stop_event = None
        self.scan_pause_event = None
        self.scan_params = None
        self._sync_keepalive_state()

    def _on_scan_finished(self, result: ScanResult) -> None:
        self._finish_scan_ui()
        self.scan_progress_bar.setValue(self.scan_progress_bar.maximum())
        self._log(f"Center scan frames displayed: {len(result.frames)}")
        if result.center is not None:
            center = result.center
            self._set_status(
                self.scan_center_label,
                f"Center: peak x={center.peak_x:.0f}, centroid x={center.centroid_x:.1f}",
                "ok",
            )
            self._log(
                f"Center detected: peak x={center.peak_x:.1f} "
                f"(signal {center.peak_signal:.4g}), centroid x={center.centroid_x:.1f}"
            )
        elif result.samples:
            self._set_status(self.scan_center_label, "Center: not enough samples", "error")
        else:
            self._set_status(self.scan_center_label, "Center: no detector", "off")
        if result.samples_path is not None:
            self._log(f"Detector samples saved: {result.samples_path}")

    def _on_scan_error(self, _error: str) -> None:
        self._finish_scan_ui()

    def _render_pattern_preview(self, label: QtWidgets.QLabel, data: np.ndarray) -> None:
        # render the real grayscale levels (0..1023) as display brightness
        preview = (data.astype(np.float32) / MAX_LEVEL * 217.0 + 18.0).astype(np.uint8)
        image = QtGui.QImage(
            preview.data,
            preview.shape[1],
            preview.shape[0],
            preview.shape[1],
            QtGui.QImage.Format_Grayscale8,
        ).copy()
        pixmap = QtGui.QPixmap.fromImage(image).scaled(
            label.size().expandedTo(QtCore.QSize(760, 240)),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        label.setPixmap(pixmap)

    def _update_scan_preview(self) -> None:
        width, height = self.slm_size
        try:
            data = make_vertical_window(
                width,
                height,
                min(self.start_x_spin.value(), width - 1),
                self.scan_level_spin.value(),
                self.window_px_spin.value(),
                self.bg_level_spin.value(),
            )
        except ValueError as exc:
            self.preview_label.setText(str(exc))
            return
        self._render_pattern_preview(self.preview_label, data)

    def _segment_mode_is_equal(self) -> bool:
        return self.segment_mode_combo.currentIndex() == 0

    def _on_segment_mode_changed(self) -> None:
        equal = self._segment_mode_is_equal()
        self.segment_count_spin.setEnabled(equal)
        self._segment_add_button.setEnabled(not equal)
        self._segment_remove_button.setEnabled(not equal)
        if equal:
            self._rebuild_equal_segment_rows()
        else:
            self._make_segment_x_cells_editable()
            self._update_segment_preview()

    def _segment_table_item(self, value: int, editable: bool) -> QtWidgets.QTableWidgetItem:
        item = QtWidgets.QTableWidgetItem(str(value))
        if not editable:
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
        return item

    def _rebuild_equal_segment_rows(self) -> None:
        if not self._segment_mode_is_equal():
            return
        width, _height = self.slm_size
        count = min(self.segment_count_spin.value(), width)
        edges = equal_x_segment_edges(width, count)

        previous_levels = []
        for row in range(self.segments_table.rowCount()):
            item = self.segments_table.item(row, 2)
            previous_levels.append(item.text() if item is not None else "0")

        self._segments_updating = True
        try:
            self.segments_table.setRowCount(count)
            for row in range(count):
                level = previous_levels[row] if row < len(previous_levels) else "0"
                self.segments_table.setItem(
                    row, 0, self._segment_table_item(edges[row], editable=False)
                )
                self.segments_table.setItem(
                    row, 1, self._segment_table_item(edges[row + 1], editable=False)
                )
                level_item = QtWidgets.QTableWidgetItem(level)
                self.segments_table.setItem(row, 2, level_item)
        finally:
            self._segments_updating = False
        self._update_segment_preview()

    def _make_segment_x_cells_editable(self) -> None:
        self._segments_updating = True
        try:
            for row in range(self.segments_table.rowCount()):
                for col in (0, 1):
                    item = self.segments_table.item(row, col)
                    if item is not None:
                        item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)
        finally:
            self._segments_updating = False

    def _fill_segment_levels(self) -> None:
        value = str(self.segment_fill_spin.value())
        self._segments_updating = True
        try:
            for row in range(self.segments_table.rowCount()):
                item = self.segments_table.item(row, 2)
                if item is None:
                    self.segments_table.setItem(row, 2, QtWidgets.QTableWidgetItem(value))
                else:
                    item.setText(value)
        finally:
            self._segments_updating = False
        self._update_segment_preview()

    def _add_segment_row(self) -> None:
        width, _height = self.slm_size
        row = self.segments_table.rowCount()
        previous_end = 0
        if row > 0:
            item = self.segments_table.item(row - 1, 1)
            try:
                previous_end = int(item.text()) if item is not None else 0
            except ValueError:
                previous_end = 0
        self._segments_updating = True
        try:
            self.segments_table.insertRow(row)
            self.segments_table.setItem(
                row, 0, QtWidgets.QTableWidgetItem(str(min(previous_end, width - 1)))
            )
            self.segments_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(width)))
            self.segments_table.setItem(row, 2, QtWidgets.QTableWidgetItem("0"))
        finally:
            self._segments_updating = False
        self._update_segment_preview()

    def _remove_segment_row(self) -> None:
        row = self.segments_table.currentRow()
        if row < 0:
            row = self.segments_table.rowCount() - 1
        if row >= 0:
            self.segments_table.removeRow(row)
            self._update_segment_preview()

    def _on_segment_item_changed(self, _item: QtWidgets.QTableWidgetItem) -> None:
        if not self._segments_updating:
            self._update_segment_preview()

    def _segment_pattern_data(self) -> np.ndarray:
        width, height = self.slm_size
        rows = self.segments_table.rowCount()
        if rows == 0:
            raise ValueError("define at least one segment")

        def cell(row: int, col: int, name: str) -> int:
            item = self.segments_table.item(row, col)
            text = item.text().strip() if item is not None else ""
            try:
                return int(text)
            except ValueError as exc:
                raise ValueError(f"row {row + 1}: {name} must be an integer") from exc

        if self._segment_mode_is_equal():
            levels = [cell(row, 2, "level") for row in range(rows)]
            return make_equal_x_segments(width, height, levels)
        segments = [
            (cell(row, 0, "x start"), cell(row, 1, "x end"), cell(row, 2, "level"))
            for row in range(rows)
        ]
        return make_x_segments(width, height, segments)

    def _update_segment_preview(self) -> None:
        if not hasattr(self, "segment_preview_label"):
            return
        try:
            data = self._segment_pattern_data()
        except ValueError as exc:
            self.segment_preview_label.setText(str(exc))
            self.segment_status_label.setText(str(exc))
            return
        self.segment_status_label.setText("")
        self._render_pattern_preview(self.segment_preview_label, data)

    def _display_segments(self) -> None:
        try:
            data = self._segment_pattern_data()
        except ValueError as exc:
            self._log(f"Invalid segments: {exc}")
            QtWidgets.QMessageBox.warning(self, "Phase Segments", str(exc))
            return
        controller = self._controller()
        self._run_slm_task(
            "Display segments",
            lambda: controller.display_mask_csv(data),
        )

    def _export_segments_csv(self) -> None:
        try:
            data = self._segment_pattern_data()
        except ValueError as exc:
            self._log(f"Invalid segments: {exc}")
            QtWidgets.QMessageBox.warning(self, "Phase Segments", str(exc))
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Segments CSV", "phase_segments.csv", "CSV Files (*.csv)"
        )
        if not path:
            return
        self._run_task(
            "Export segments CSV",
            lambda: write_santec_csv(data, path),
            lambda saved: self._log(f"Segments CSV saved: {saved}"),
        )

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if hasattr(self, "preview_label"):
            self._update_scan_preview()
        if hasattr(self, "segment_preview_label"):
            self._update_segment_preview()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.keepalive is not None:
            self.keepalive.stop()
            self.keepalive = None
        if self.scan_stop_event is not None:
            self.scan_stop_event.set()
        if self.scan_pause_event is not None:
            # wake a paused scan so the worker can observe the stop event
            self.scan_pause_event.clear()
        if self.calibration_stop_event is not None:
            self.calibration_stop_event.set()
        self.thread_pool.waitForDone(3000)
        if self.controller is not None and getattr(self.controller, "is_open", False):
            try:
                self.controller.close_slm()
            except Exception:
                pass
        if self.osa_controller is not None:
            try:
                self.osa_controller.disconnect()
            except Exception:
                pass
            self.osa_controller = None
        super().closeEvent(event)

    def _apply_style(self) -> None:
        self.setStyleSheet(DARK_STYLESHEET)


def main(argv: list[str] | None = None) -> int:
    app = QtWidgets.QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("Santec SLM Control")
    window = MainWindow()
    window.show()
    return app.exec_()
