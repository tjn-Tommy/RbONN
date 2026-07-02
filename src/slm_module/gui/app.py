from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
import matplotlib
from PyQt5 import QtCore, QtGui, QtWidgets

from osa_module.controller import MeasurementSettings, OSAController
from scope_module.controller import (
    MonitorSample,
    MonitorSettings,
    ScopeController,
    ScopeSettings,
    Waveform,
)

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
from ..analysis import (
    AnalysisAborted,
    AnalysisProgress,
    EncodingGain,
    ModulationErrorResult,
    encoding_gain,
    measure_channel_spectra,
    write_analysis_csv,
    write_gain_csv,
)
from ..encoding import (
    ChannelLayout,
    build_channel_layout,
    encode_to_pattern,
    optimize_from_osa,
)
from ..optimization import (
    OSAOptimizationConfig,
    OptimizationAborted,
    OptimizationProgress,
    OptimizationResult,
    amplitudes_to_intensity_commands,
    load_optimization_result,
)
from ..scope_background import (
    BackgroundAborted,
    BackgroundProgress,
    BackgroundResult,
    load_background_result,
    measure_scatter_background,
    save_background_json,
    save_background_npz,
    write_background_csv,
)
from ..scope_tpa import (
    TPAAborted,
    TPAProgress,
    TPAResult,
    load_tpa_result,
    measure_tpa_efficiency,
    save_tpa_json,
    save_tpa_npz,
    write_tpa_csv,
)
from ..encoding import ChannelLayout, build_channel_layout, encode_to_pattern
from ..keepalive import SLMKeepAlive
from .style import DARK_STYLESHEET


# a calibration progress callback marshalled onto the GUI thread via a signal
ProgressEmit = Callable[[CalibrationProgress], None]


def _pattern_to_qimage(data: np.ndarray) -> QtGui.QImage:
    """Render a 0..1023 grayscale grid as an 8-bit QImage for preview.

    Levels are mapped onto 18..235 so even level 0 is visible against a black
    background while full scale stays near white.
    """
    array = np.asarray(data, dtype=np.float32)
    preview = (array / MAX_LEVEL * 217.0 + 18.0).clip(0, 255).astype(np.uint8)
    preview = np.ascontiguousarray(preview)
    height, width = preview.shape
    image = QtGui.QImage(
        preview.data, width, height, width, QtGui.QImage.Format_Grayscale8
    )
    return image.copy()


_CMAP_LUT = None


def _cmap_lut() -> np.ndarray:
    """Lazy 0..MAX_LEVEL -> RGB lookup table for the viridis colormap."""
    global _CMAP_LUT
    if _CMAP_LUT is None:
        colours = matplotlib.colormaps["viridis"](np.linspace(0.0, 1.0, MAX_LEVEL + 1))
        _CMAP_LUT = (colours[:, :3] * 255.0).astype(np.uint8)
    return _CMAP_LUT


def _pattern_to_qimage_color(data: np.ndarray) -> QtGui.QImage:
    """Render a 0..1023 level grid as a colour (viridis) QImage."""
    idx = np.clip(np.asarray(data), 0, MAX_LEVEL).astype(np.int32)
    rgb = np.ascontiguousarray(_cmap_lut()[idx])          # H x W x 3, uint8
    height, width = idx.shape
    image = QtGui.QImage(
        rgb.data, width, height, 3 * width, QtGui.QImage.Format_RGB888
    )
    return image.copy()


def _format_duration(seconds: float) -> str:
    """Format a duration as m:ss, or h:mm:ss once it passes an hour."""
    if not np.isfinite(seconds) or seconds < 0:
        return "—"
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


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


class WheelSpinBox(QtWidgets.QDoubleSpinBox):
    """Double spin box with an independent (large) mouse-wheel step.

    The wheel changes the value by ``wheel_step`` regardless of the small
    ``singleStep`` used by the arrows/keyboard, so a couple of scrolls can span
    the whole 0..1 range while typed/arrow entry stays fine-grained.
    """

    def __init__(self, wheel_step: float = 0.2, parent=None):
        super().__init__(parent)
        self.wheel_step = float(wheel_step)

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        if delta:
            self.setValue(self.value() + (self.wheel_step if delta > 0 else -self.wheel_step))
            event.accept()
        else:
            super().wheelEvent(event)


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
        self._phase_start: float | None = None
        self._xs: list[float] = []
        self._ys: list[float] = []
        self._dirty = False

        layout = QtWidgets.QVBoxLayout(self)

        self.phase_label = QtWidgets.QLabel("Preparing…")
        self.phase_label.setObjectName("PageSubtitle")
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%v / %m  (%p%)")
        self.eta_label = QtWidgets.QLabel("Elapsed 0:00 · ETA —")
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
        layout.addWidget(self.eta_label)
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
        done = min(int(progress.step) + 1, total)
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(done)
        self.status_label.setText(progress.message)
        self._update_eta(done, total)
        if progress.x is not None and progress.y is not None:
            self._xs.append(float(progress.x))
            self._ys.append(float(progress.y))
            self._dirty = True

    def _enter_phase(self, phase: str) -> None:
        self._phase = phase
        self._phase_start = time.perf_counter()
        self._xs.clear()
        self._ys.clear()
        title, _xlabel, _ylabel = self._PHASES.get(phase, (phase, "x", "y"))
        self.phase_label.setText(title)
        self.log.appendPlainText(f"\N{BLACK RIGHT-POINTING TRIANGLE} {title}")
        self.eta_label.setText("Elapsed 0:00 · ETA —")
        self._dirty = True

    def _update_eta(self, done: int, total: int) -> None:
        """Estimate time remaining from the average pace of this phase so far."""
        if self._phase_start is None:
            return
        elapsed = time.perf_counter() - self._phase_start
        if done <= 0:
            self.eta_label.setText(f"Elapsed {_format_duration(elapsed)} · ETA —")
            return
        remaining = (elapsed / done) * max(total - done, 0)
        self.eta_label.setText(
            f"Elapsed {_format_duration(elapsed)} · ETA {_format_duration(remaining)}"
        )

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


class SLMMonitorView(QtWidgets.QWidget):
    """An embeddable live view of the exact pattern currently on the SLM.

    It does not talk to hardware directly: it polls ``get_pattern`` (which
    returns a copy of the controller's last displayed grid) on a timer and
    renders both the 2D image and a column-averaged level-vs-x profile, so the
    user can watch the SLM while operating other pages. ``describe`` returns a
    short string for the source (grayscale level / CSV path).
    """

    def __init__(
        self,
        get_pattern: Callable[[], np.ndarray | None],
        describe: Callable[[], str | None],
        parent: QtWidgets.QWidget | None = None,
        *,
        image_min_height: int = 300,
        profile_height: int = 200,
        show_profile: bool = True,
    ):
        super().__init__(parent)
        self._get_pattern = get_pattern
        self._describe = describe
        self._last_shape: tuple[int, int] | None = None
        self._show_profile = show_profile
        self._preview = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        controls = QtWidgets.QHBoxLayout()
        self.live_check = QtWidgets.QCheckBox("Live")
        self.live_check.setChecked(True)
        self.live_check.toggled.connect(self._on_live_toggled)
        self.interval_spin = QtWidgets.QDoubleSpinBox()
        self.interval_spin.setRange(0.1, 10.0)
        self.interval_spin.setSingleStep(0.1)
        self.interval_spin.setDecimals(1)
        self.interval_spin.setValue(0.5)
        self.interval_spin.setSuffix(" s")
        self.interval_spin.valueChanged.connect(self._on_interval_changed)
        self.refresh_button = QtWidgets.QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh)
        self.save_button = QtWidgets.QPushButton("Save PNG…")
        self.save_button.setProperty("variant", "ghost")
        self.save_button.clicked.connect(self._save_png)
        controls.addWidget(self.live_check)
        controls.addWidget(QtWidgets.QLabel("Every"))
        controls.addWidget(self.interval_spin)
        controls.addStretch(1)
        controls.addWidget(self.refresh_button)
        controls.addWidget(self.save_button)
        layout.addLayout(controls)

        self.info_label = QtWidgets.QLabel("\N{EN DASH}")
        self.info_label.setObjectName("PageSubtitle")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        self.image_label = QtWidgets.QLabel()
        self.image_label.setMinimumHeight(image_min_height)
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setObjectName("Preview")
        layout.addWidget(self.image_label, 1)

        if show_profile:
            self.figure = Figure(figsize=(6, 2.2), tight_layout=True)
            self.canvas = FigureCanvas(self.figure)
            self.canvas.setMaximumHeight(profile_height)
            self.axes = self.figure.add_subplot(111)
            self._style_axes()
            layout.addWidget(self.canvas)
        else:
            self.figure = None
            self.canvas = None
            self.axes = None

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(int(self.interval_spin.value() * 1000))
        self.refresh()

    def _style_axes(self) -> None:
        self.figure.patch.set_facecolor("#101820")
        axes = self.axes
        axes.set_facecolor("#101820")
        axes.grid(True, color="#2b3a42", linewidth=0.7)
        axes.tick_params(colors="#d8dee9", labelsize=8)
        axes.xaxis.label.set_color("#d8dee9")
        axes.yaxis.label.set_color("#d8dee9")
        for spine in axes.spines.values():
            spine.set_color("#41515c")
        axes.set_xlabel("x column (px)")
        axes.set_ylabel("mean level")

    def _on_live_toggled(self, checked: bool) -> None:
        if checked:
            self._timer.start(int(self.interval_spin.value() * 1000))
            self.refresh()
        else:
            self._timer.stop()

    def _on_interval_changed(self, value: float) -> None:
        if self.live_check.isChecked():
            self._timer.start(int(value * 1000))

    def set_preview(self, on: bool) -> None:
        """Dim the view to signal an un-sent preview (vs. what's on the SLM)."""
        self._preview = bool(on)
        self.refresh()

    def refresh(self) -> None:
        pattern = None
        try:
            pattern = self._get_pattern()
        except Exception as exc:  # never let a poll error kill the timer
            self.info_label.setText(f"Monitor error: {exc}")
            return
        if pattern is None:
            self.info_label.setText(
                "Nothing displayed yet (open the SLM and show a pattern)."
            )
            self.image_label.setText("\N{EN DASH}")
            return

        source = None
        try:
            source = self._describe()
        except Exception:
            source = None
        height, width = pattern.shape
        unique = int(np.unique(pattern).size)
        prefix = f"{source}  ·  " if source else ""
        preview_tag = "  ·  PREVIEW (not sent)" if self._preview else ""
        self.info_label.setText(
            f"{prefix}{width} x {height} px  ·  level "
            f"{int(pattern.min())}–{int(pattern.max())}  ·  {unique} distinct{preview_tag}"
        )

        image = _pattern_to_qimage_color(pattern)
        pixmap = QtGui.QPixmap.fromImage(image).scaled(
            self.image_label.size().expandedTo(QtCore.QSize(760, 280)),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        if self._preview:
            pixmap = self._dim_pixmap(pixmap)
        self.image_label.setPixmap(pixmap)
        if self._show_profile:
            self._draw_profile(pattern)
        self._last_shape = (width, height)

    @staticmethod
    def _dim_pixmap(pixmap: QtGui.QPixmap) -> QtGui.QPixmap:
        """Overlay a translucent dark veil to mark an un-sent preview."""
        out = QtGui.QPixmap(pixmap)
        painter = QtGui.QPainter(out)
        painter.fillRect(out.rect(), QtGui.QColor(15, 20, 25, 150))
        painter.end()
        return out

    def _draw_profile(self, pattern: np.ndarray) -> None:
        profile = pattern.astype(np.float32).mean(axis=0)
        xs = np.arange(profile.size)
        reset = self._last_shape != (pattern.shape[1], pattern.shape[0])
        self.axes.clear()
        self._style_axes()
        self.axes.plot(xs, profile, color="#47b8e0", linewidth=1.0)
        self.axes.set_ylim(-20, MAX_LEVEL + 20)
        if reset and profile.size:
            self.axes.set_xlim(0, profile.size - 1)
        self.canvas.draw_idle()

    def _save_png(self) -> None:
        pattern = None
        try:
            pattern = self._get_pattern()
        except Exception:
            pattern = None
        if pattern is None:
            QtWidgets.QMessageBox.information(
                self, "SLM Monitor", "There is no pattern to save yet."
            )
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save SLM Pattern", "slm_pattern.png", "PNG Image (*.png)"
        )
        if not path:
            return
        _pattern_to_qimage_color(pattern).save(path, "PNG")

    def stop(self) -> None:
        self._timer.stop()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._timer.stop()
        super().closeEvent(event)


class MainWindow(QtWidgets.QMainWindow):
    scan_progress = QtCore.pyqtSignal(int, int, str)
    scan_started = QtCore.pyqtSignal(int, int, int, int)
    scan_sample = QtCore.pyqtSignal(float, float)
    keepalive_status = QtCore.pyqtSignal(bool, str)
    calibration_progress = QtCore.pyqtSignal(object)
    analysis_progress = QtCore.pyqtSignal(object)
    background_progress = QtCore.pyqtSignal(object)
    tpa_progress = QtCore.pyqtSignal(object)
    edge_gain_progress = QtCore.pyqtSignal(int, int, str)
    edge_optimization_progress = QtCore.pyqtSignal(object)
    monitor_sample = QtCore.pyqtSignal(object)
    hold_progress = QtCore.pyqtSignal(int, int)

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
        self._scan_start_time: float | None = None
        self._segments_updating = False
        self.encoding_layout: ChannelLayout | None = None
        self._encoding_pattern: np.ndarray | None = None
        self.enc_col_ratio: np.ndarray | None = None  # per-column edge-ratio profile
        self._edge_gain: EncodingGain | None = None
        self._edge_optimization_result: OptimizationResult | None = None
        self.edge_gain_stop_event: threading.Event | None = None
        self._enc_wheel_step = 0.2   # scroll sensitivity for channel value cells
        self._enc_calib_override: CalibrationResult | None = None
        self.analysis_result: ModulationErrorResult | None = None
        self.analysis_stop_event: threading.Event | None = None
        self._ana_capture_dir: str | None = None
        self.background_result: BackgroundResult | None = None
        self.background_stop_event: threading.Event | None = None
        self._bg_capture_dir: str | None = None
        self._bg_live: list[tuple[float, float]] = []
        self.tpa_result: TPAResult | None = None
        self.tpa_stop_event: threading.Event | None = None
        self._tpa_capture_dir: str | None = None
        self._tpa_live: list[tuple[float, float]] = []
        self.scope_controller: ScopeController | None = None
        self.scope_stop_event: threading.Event | None = None
        self.monitor_stop_event: threading.Event | None = None
        self._monitor_values: list[float] = []
        self.hold_stop_event: threading.Event | None = None

        self.setWindowTitle("Santec SLM Control")
        self.resize(1280, 840)
        self._build_ui()
        self._apply_style()
        self.scan_progress.connect(self._on_scan_progress)
        self.scan_started.connect(self._on_scan_started)
        self.scan_sample.connect(self._on_scan_sample)
        self.keepalive_status.connect(self._on_keepalive_status)
        self.calibration_progress.connect(self._on_calibration_progress)
        self.analysis_progress.connect(self._on_analysis_progress)
        self.background_progress.connect(self._on_background_progress)
        self.tpa_progress.connect(self._on_tpa_progress)
        self.edge_gain_progress.connect(self._edge_gain_progress)
        self.edge_optimization_progress.connect(self._edge_optimization_progress)
        self.monitor_sample.connect(self._on_monitor_sample)
        self.hold_progress.connect(self._on_hold_progress)

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
            ("\N{ELECTRIC PLUG}  Connections", "Connect SLM, OSA and scope"),
            ("\N{LINK SYMBOL}  SLM Control", "Grayscale and CSV display"),
            ("\N{CHART WITH UPWARDS TREND}  Calibration", "Intensity, mod error, scope holding"),
            ("\N{LEFT RIGHT ARROW}  Center Scan", "Sweep a window across x"),
            ("\N{TRIGRAM FOR HEAVEN}  Phase Segments", "Piecewise phase along x"),
            ("\N{HIGH VOLTAGE SIGN}  TPA Encoding", "Channel grid encoding + scope readout"),
            ("\N{DOWNWARDS ARROW WITH TIP RIGHTWARDS}  Edge Ratio",
             "Per-column edge-taper ratio + OSA optimisation hook"),
        )
        for label, tooltip in nav_items:
            item = QtWidgets.QListWidgetItem(label)
            item.setSizeHint(QtCore.QSize(180, 48))
            item.setToolTip(tooltip)
            self.nav.addItem(item)
        sidebar_layout.addWidget(self.nav, 1)

        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self._build_connection_page())
        self.stack.addWidget(self._build_control_page())
        self.stack.addWidget(self._build_calibration_page())
        self.stack.addWidget(self._build_scan_page())
        self.stack.addWidget(self._build_segments_page())
        self.stack.addWidget(self._build_tpa_page())
        self.stack.addWidget(self._build_edge_ratio_page())

        layout.addWidget(sidebar)
        layout.addWidget(self.stack, 1)
        self.setCentralWidget(central)
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav.setCurrentRow(0)

    def _build_connection_page(self) -> QtWidgets.QWidget:
        """First page: connect the SLM, OSA and scope, with a shared status log."""
        page = self._page_shell("Connections")

        # ---- SLM ----
        slm = self._panel("SLM (Santec)")
        sl = QtWidgets.QGridLayout(slm)
        self.display_no_spin = QtWidgets.QSpinBox()
        self.display_no_spin.setRange(1, 8)
        self.display_no_spin.setValue(1)
        self.display_no_spin.valueChanged.connect(self._reset_controller)
        self.rate120_check = QtWidgets.QCheckBox("120 Hz model")
        self.rate120_check.toggled.connect(self._reset_controller)
        self.conn_status_label = QtWidgets.QLabel("Status: closed")
        self._set_status(self.conn_status_label, "Status: closed", "off")
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
        self.keepalive_interval_spin.setValue(0.5)
        self.keepalive_interval_spin.setSuffix(" s")
        self.keepalive_interval_spin.valueChanged.connect(self._on_keepalive_interval)
        self.keepalive_status_label = QtWidgets.QLabel("Keep-alive: off")
        self._set_status(self.keepalive_status_label, "Keep-alive: off", "off")

        sl.addWidget(QtWidgets.QLabel("Display"), 0, 0)
        sl.addWidget(self.display_no_spin, 0, 1)
        sl.addWidget(detect_button, 0, 2)
        sl.addWidget(open_button, 0, 3)
        sl.addWidget(close_button, 0, 4)
        sl.addWidget(info_button, 0, 5)
        sl.addWidget(QtWidgets.QLabel("USB SLM"), 1, 0)
        sl.addWidget(self.usb_slm_no_spin, 1, 1)
        sl.addWidget(dvi_mode_button, 1, 2)
        sl.addWidget(self.rate120_check, 1, 3, 1, 2)
        sl.addWidget(self.keepalive_check, 2, 0, 1, 2)
        sl.addWidget(QtWidgets.QLabel("Interval"), 2, 2)
        sl.addWidget(self.keepalive_interval_spin, 2, 3)
        sl.addWidget(self.keepalive_status_label, 2, 4, 1, 2)
        sl.addWidget(self.conn_status_label, 3, 0, 1, 3)
        sl.addWidget(self.info_label, 3, 3, 1, 3)
        page.layout().addWidget(slm)

        # ---- OSA ----
        osa = self._panel("OSA (Yokogawa AQ637X)")
        ol = QtWidgets.QGridLayout(osa)
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
        ol.addWidget(QtWidgets.QLabel("OSA Host"), 0, 0)
        ol.addWidget(self.osa_host_edit, 0, 1)
        ol.addWidget(QtWidgets.QLabel("Port"), 0, 2)
        ol.addWidget(self.osa_port_spin, 0, 3)
        ol.addWidget(self.osa_connect_button, 0, 4)
        ol.addWidget(self.osa_disconnect_button, 0, 5)
        ol.addWidget(self.osa_status_label, 0, 6)
        ol.setColumnStretch(1, 1)
        page.layout().addWidget(osa)

        # ---- Scope ----
        scope = self._panel("Oscilloscope (R&S RTO6)")
        scl = QtWidgets.QGridLayout(scope)
        self.scope_host_edit = QtWidgets.QLineEdit("192.168.1.2")
        self.scope_host_edit.setPlaceholderText("RTO6 host / IP")
        self.scope_connect_button = QtWidgets.QPushButton("Connect Scope")
        self.scope_connect_button.clicked.connect(self._connect_scope)
        self.scope_disconnect_button = QtWidgets.QPushButton("Disconnect")
        self.scope_disconnect_button.setProperty("variant", "ghost")
        self.scope_disconnect_button.setEnabled(False)
        self.scope_disconnect_button.clicked.connect(self._disconnect_scope)
        self.scope_status_label = QtWidgets.QLabel("Scope: closed")
        self._set_status(self.scope_status_label, "Scope: closed", "off")
        scl.addWidget(QtWidgets.QLabel("Scope Host"), 0, 0)
        scl.addWidget(self.scope_host_edit, 0, 1)
        scl.addWidget(self.scope_connect_button, 0, 2)
        scl.addWidget(self.scope_disconnect_button, 0, 3)
        scl.addWidget(self.scope_status_label, 0, 4)
        scl.setColumnStretch(1, 1)
        page.layout().addWidget(scope)

        # ---- shared status log ----
        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setObjectName("LogBox")
        page.layout().addWidget(self._panel_with_widget("Status", self.log_box), 1)
        return page

    def _build_control_page(self) -> QtWidgets.QWidget:
        page = self._page_shell("SLM Control")

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

        self.slm_monitor_view = SLMMonitorView(
            get_pattern=self._current_slm_pattern,
            describe=self._describe_slm_pattern,
        )

        page.layout().addWidget(grayscale)
        page.layout().addWidget(csv_panel)
        page.layout().addWidget(
            self._panel_with_widget("SLM Pattern Monitor", self.slm_monitor_view), 1
        )
        return page

    def _build_calibration_page(self) -> QtWidgets.QWidget:
        """Top-level step tabs; each step (incl. Mod Error / Holding) uses the full page."""
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)

        # per-step widget registry: self.step_widgets[step][key]
        self.step_widgets: dict[int, dict[str, Any]] = {1: {}, 2: {}, 3: {}}

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._build_step1_tab(), "Step 1 · Min/Max")
        tabs.addTab(self._build_step2_tab(), "Step 2 · Wavelength")
        tabs.addTab(self._build_step3_page(), "Step 3 · Intensity")
        tabs.addTab(self._build_analysis_page(), "Step 4 · Mod Error")
        tabs.addTab(self._build_scope_holding_tab(), "Step 5 · Holding")
        tabs.addTab(self._build_background_tab(), "Step 6 · BG Scatter")
        tabs.addTab(self._build_tpa_tab(), "Step 7 · TPA Efficiency")
        lay.addWidget(tabs)

        # every Run button, toggled together by _set_calibration_running
        self.calibration_run_buttons = [
            self.step_widgets[1]["run"],
            self.step_widgets[2]["run"],
            self.step_widgets[3]["run"],
            self.run_all_button,
        ]
        return page

    def _build_step3_page(self) -> QtWidgets.QWidget:
        """Step 3 (intensity) config + Run-All + the calibration fit/plots (full page)."""
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(18, 14, 18, 14)
        lay.addWidget(self._build_step3_tab())

        # run all three steps in sequence
        self.run_all_button = QtWidgets.QPushButton("Run All (1→2→3)")
        self.run_all_button.setEnabled(False)
        self.run_all_button.clicked.connect(self._run_all)
        self.stop_cal_button = QtWidgets.QPushButton("Stop")
        self.stop_cal_button.setProperty("variant", "danger")
        self.stop_cal_button.setEnabled(False)
        self.stop_cal_button.clicked.connect(self._stop_full_calibration)
        run_row = QtWidgets.QHBoxLayout()
        run_row.addStretch(1)
        run_row.addWidget(self.run_all_button)
        run_row.addWidget(self.stop_cal_button)
        lay.addLayout(run_row)

        # --- fit from a saved calibration CSV ---
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
        lay.addWidget(controls)

        # --- results: fit parameters + fit curve / intensity map ---
        self.fit_table = QtWidgets.QTableWidget(0, 2)
        self.fit_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.fit_table.horizontalHeader().setStretchLastSection(True)
        self.fit_table.verticalHeader().setVisible(False)

        self.figure = Figure(figsize=(6, 4), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        plot_panel = self._panel("Fit Curve")
        plot_layout = QtWidgets.QVBoxLayout(plot_panel)
        plot_layout.addWidget(self.canvas)

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
        lay.addWidget(split, 1)
        return page

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
        widgets["sweep_nm"] = self._double_spin(0.0, 50.0, 0.5, " nm", 3)
        widgets["sweep_nm"].setToolTip(
            "OSA span per coordinate, re-centered on the Step 2 wavelength. "
            "Narrower = faster. 0 = use the full span above."
        )
        widgets["stride"] = self._spin(1, 8191, 1)
        widgets["stride"].setToolTip(
            "Measure only every Nth calibrated coordinate (1 = every coordinate)."
        )
        widgets["refine"] = QtWidgets.QCheckBox("Refine λ")
        widgets["refine"].setChecked(True)
        widgets["refine"].setToolTip(
            "Re-calibrate each coordinate's wavelength from the narrow high-res "
            "sweep (needs a sweep span above)."
        )
        cfg.addWidget(QtWidgets.QLabel("Window px"))
        cfg.addWidget(widgets["window"])
        cfg.addWidget(QtWidgets.QLabel("Avg ± window"))
        cfg.addWidget(widgets["avg_nm"])
        cfg.addWidget(QtWidgets.QLabel("Sweep span"))
        cfg.addWidget(widgets["sweep_nm"])
        cfg.addWidget(QtWidgets.QLabel("Stride"))
        cfg.addWidget(widgets["stride"])
        cfg.addWidget(widgets["refine"])
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

    def _build_scope_holding_tab(self) -> QtWidgets.QWidget:
        page = self._page_shell("Scope Holding Time")
        subtitle = QtWidgets.QLabel(
            "Measure the SLM settling (hold) time on a single PC-scripted timeline: "
            "record → hold A (pre-switch) → switch A→B → hold B (post-switch) → stop. "
            "Each repeat is aligned on the A→B edge detected in its own trace, so the "
            "settle time is referenced to the real optical transition and is immune "
            "to the scope-vs-PC clock offset (reported separately)."
        )
        subtitle.setObjectName("PageSubtitle")
        subtitle.setWordWrap(True)
        page.layout().addWidget(subtitle)
        page.layout().addWidget(self._build_scope_holding_controls())
        self.hold_fig = Figure(figsize=(7, 3.4), tight_layout=True)
        self.hold_canvas = FigureCanvas(self.hold_fig)
        page.layout().addWidget(self._panel_with_widget("Averaged transient", self.hold_canvas), 1)
        self._hold_result = QtWidgets.QLabel("\N{EN DASH}")
        page.layout().addWidget(self._hold_result)
        return page

    def _build_scope_holding_controls(self) -> QtWidgets.QGroupBox:
        panel = self._panel("Settling measurement")
        grid = QtWidgets.QGridLayout(panel)
        self.hold_channel = QtWidgets.QComboBox(); self.hold_channel.addItems(["1", "2", "3", "4"])
        self.hold_gray_a = self._spin(0, 1023, 880)
        self.hold_gray_b = self._spin(0, 1023, 420)
        self.hold_averages = self._spin(1, 1000, 60)
        self.hold_window = self._double_spin(0.05, 5.0, 0.8, " s", 2)
        self.hold_window.setToolTip("Time B is held/captured after the A→B switch")
        self.hold_settle = self._double_spin(0.1, 5.0, 0.6, " s", 2)
        self.hold_settle.setToolTip("Settle at A before recording starts (not measured)")
        self.hold_baseline = self._double_spin(0.02, 2.0, 0.30, " s", 2)
        self.hold_baseline.setToolTip("Time A is held after recording starts, before the switch")
        grid.addWidget(QtWidgets.QLabel("Channel"), 0, 0); grid.addWidget(self.hold_channel, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Gray A (start)"), 0, 2); grid.addWidget(self.hold_gray_a, 0, 3)
        grid.addWidget(QtWidgets.QLabel("Gray B (switch to)"), 0, 4); grid.addWidget(self.hold_gray_b, 0, 5)
        grid.addWidget(QtWidgets.QLabel("Averages"), 1, 0); grid.addWidget(self.hold_averages, 1, 1)
        grid.addWidget(QtWidgets.QLabel("Post-switch hold (B)"), 1, 2); grid.addWidget(self.hold_window, 1, 3)
        grid.addWidget(QtWidgets.QLabel("Pre-settle"), 1, 4); grid.addWidget(self.hold_settle, 1, 5)
        grid.addWidget(QtWidgets.QLabel("Pre-switch hold (A)"), 2, 0); grid.addWidget(self.hold_baseline, 2, 1)
        self.hold_status = QtWidgets.QLabel("\N{EN DASH}")
        self.hold_start_button = QtWidgets.QPushButton("Run")
        self.hold_start_button.clicked.connect(self._hold_start)
        self.hold_stop_button = QtWidgets.QPushButton("Stop")
        self.hold_stop_button.setProperty("variant", "danger")
        self.hold_stop_button.setEnabled(False)
        self.hold_stop_button.clicked.connect(self._hold_stop)
        grid.addWidget(self.hold_status, 3, 0, 1, 3)
        grid.addWidget(self.hold_start_button, 3, 4)
        grid.addWidget(self.hold_stop_button, 3, 5)
        return panel

    def _hold_set_running(self, running: bool) -> None:
        self.hold_start_button.setEnabled(not running)
        self.hold_stop_button.setEnabled(running)

    def _hold_start(self) -> None:
        scope = self.scope_controller
        if scope is None or not scope.is_connected:
            self.hold_status.setText("Connect the scope on the Connections page first.")
            return
        controller = self._controller()
        if not getattr(controller, "is_open", False):
            self.hold_status.setText("Open the SLM on the Connections page first.")
            return
        ch = int(self.hold_channel.currentText())
        ga, gb = self.hold_gray_a.value(), self.hold_gray_b.value()
        n = self.hold_averages.value()
        pre_hold = self.hold_baseline.value()      # A held after record start, before switch
        post_hold = self.hold_window.value()       # captured after the switch
        settle = self.hold_settle.value()          # pre-settle at A before recording
        total = pre_hold + post_hold + 0.1         # scope record span (+ margin)
        rl = max(1000, int(total * 100_000))       # ~100 kSa/s
        align_pre = min(0.1, pre_hold * 0.5)       # kept before the edge in the aligned avg
        stop_event = threading.Event()
        self.hold_stop_event = stop_event
        self.hold_progress.emit(0, n)
        self._hold_set_running(True)

        def work() -> dict[str, Any]:
            drv = scope.driver
            drv.configure_channel(ch, state=True, scale="0.02", offset="0", coupling="DCLimit")
            drv.set_decimation(ch, "HRESolution")
            drv.set_time_range(str(total)); drv.set_record_length(rl)
            drv.set_post_trigger_window(); drv.write("TRIGger1:MODE AUTO")

            # center the vertical range on the gray-A level
            controller.display_grayscale(ga, interval=0.0); time.sleep(settle)
            drv.single_acquisition()
            dl = time.monotonic() + total + 4
            while time.monotonic() < dl and not drv.is_acquisition_complete():
                time.sleep(0.02)
            y0 = drv.read_waveform(ch)
            mid = float(np.mean(y0)); pkpk = float(np.ptp(y0))
            scale = min(max(pkpk * 1.5 / 8.0, 0.002), 0.5)
            drv.configure_channel(ch, state=True, scale=f"{scale:.4f}",
                                  offset=f"{mid:.4f}", coupling="DCLimit")

            raws: list[np.ndarray] = []; onsets: list[int] = []
            switch_rels: list[float] = []; t_axis = None; dt = None
            for i in range(n):
                if stop_event.is_set():
                    return {"status": "aborted"}
                # ---- one PC-scripted timeline (a single clock drives the sequence) ----
                controller.display_grayscale(ga, interval=0.0); time.sleep(settle)
                drv.single_acquisition(); t0 = time.monotonic()   # start recording
                time.sleep(pre_hold)                              # hold A (pre-switch)
                controller.display_grayscale(gb, interval=0.0)    # A -> B switch
                switch_rels.append(time.monotonic() - t0)         # PC time of the switch
                dl = time.monotonic() + total + 4
                while time.monotonic() < dl and not drv.is_acquisition_complete():
                    if stop_event.is_set():
                        return {"status": "aborted"}
                    time.sleep(0.02)
                xs, xe, npts, vps = drv.read_waveform_header(ch)
                y = np.asarray(drv.read_waveform(ch), dtype=float)
                if t_axis is None:
                    t_axis = np.linspace(xs, xe, y.size)
                    dt = (xe - xs) / max(y.size - 1, 1)
                raws.append(y)
                onsets.append(self._hold_edge_onset(y, dt))
                self.hold_progress.emit(i + 1, n)

            return self._hold_reduce(raws, onsets, switch_rels, t_axis, dt,
                                     align_pre, post_hold, pre_hold, n)

        self._run_task("Scope holding", work, self._hold_finished, self._hold_error)

    @staticmethod
    def _hold_edge_onset(y: np.ndarray, dt: float) -> int:
        """Index where the trace first leaves its initial baseline (A→B edge onset).

        Referenced to the signal itself, so it is independent of the scope-vs-PC
        clock offset. Returns -1 if no clear edge is found.
        """
        n5 = max(10, y.size // 20)
        initial = float(np.median(y[:n5]))
        final = float(np.median(y[-n5:]))
        noise = float(np.std(y[:n5]))
        step = final - initial
        thr = max(0.1 * abs(step), 5.0 * noise, 1e-9)
        guard = min(y.size - 1, int(0.03 / dt) if dt and dt > 0 else 0)
        hit = np.where(np.abs(y[guard:] - initial) > thr)[0]
        return int(hit[0] + guard) if hit.size else -1

    @staticmethod
    def _hold_reduce(raws, onsets, switch_rels, t_axis, dt, align_pre,
                     post_hold, pre_hold, n_req) -> dict[str, Any]:
        """Edge-align the per-repeat traces, average, and derive settle metrics.

        Each repeat is cropped to a common [-align_pre, +post_hold] window around
        its own detected edge, so averaging sharpens the transition instead of
        smearing it with the trigger jitter. Time is returned with t=0 at the edge.
        """
        pre_n = max(1, int(align_pre / dt))
        post_n = max(1, int(post_hold / dt))
        subs: list[np.ndarray] = []; onset_times: list[float] = []
        for y, on in zip(raws, onsets):
            if on < 0 or on - pre_n < 0 or on + post_n > y.size:
                continue
            subs.append(y[on - pre_n: on + post_n])
            onset_times.append(float(t_axis[on]))

        used = len(subs)
        if used:
            avg = np.mean(np.vstack(subs), axis=0)
            first = subs[0]
            t_rel = (np.arange(avg.size) - pre_n) * dt
            edge_scope = float(np.median(onset_times))
        else:
            # fallback: no detectable edge — average unaligned, reference to the mean
            m = min(y.size for y in raws)
            avg = np.mean(np.vstack([y[:m] for y in raws]), axis=0)
            first = raws[0][:m]
            on = MainWindow._hold_edge_onset(avg, dt)
            ref = on if on >= 0 else 0
            t_rel = (np.arange(avg.size) - ref) * dt
            edge_scope = float(t_axis[on]) if on >= 0 else pre_hold

        n5 = max(10, avg.size // 20)
        pre_mask = t_rel < -0.01
        initial = (float(np.median(avg[pre_mask])) if pre_mask.any()
                   else float(np.median(avg[:n5])))
        final = float(np.median(avg[t_rel > t_rel[-1] - 0.1]))
        step = final - initial
        resid = (float(np.std(avg[pre_mask])) if pre_mask.sum() > 1
                 else float(np.std(avg[:n5])))
        # settle-to-2% on a lightly smoothed trace so the metric is not limited by
        # the averaged residual noise (2% of a small step can sit below the noise)
        win = max(1, int(0.002 / dt))          # ~2 ms boxcar
        if win > 1:
            pad = win // 2
            kern = np.ones(win) / win
            sm = np.convolve(np.pad(avg, pad, mode="edge"), kern, mode="valid")[:avg.size]
        else:
            sm = avg
        band = 0.02 * abs(step)
        post_mask = t_rel >= 0.0
        outside = np.where(post_mask & (np.abs(sm - final) > band))[0]
        settle = float(t_rel[outside[-1]]) if outside.size else 0.0
        # the clock offset the old command-referenced marker suffered from:
        # scope-time of the real edge minus PC-time of the issued switch
        pc_switch = float(np.median(switch_rels)) if switch_rels else pre_hold
        offset = edge_scope - pc_switch
        return {"status": "ok", "t": t_rel, "avg": avg, "first": first,
                "initial": initial, "final": final, "step": step, "resid": resid,
                "settle": settle, "cmd_rel": -offset, "offset": offset,
                "n": used, "n_req": n_req}

    def _hold_stop(self) -> None:
        if self.hold_stop_event is not None:
            self.hold_stop_event.set()
            self.hold_status.setText("Stopping…")

    def _on_hold_progress(self, done: int, total: int) -> None:
        self.hold_status.setText(f"Averaging {done}/{total} transients…")

    def _hold_finished(self, payload: dict[str, Any]) -> None:
        self.hold_stop_event = None
        self._hold_set_running(False)
        if payload.get("status") == "aborted":
            self.hold_status.setText("Stopped.")
            return
        self._hold_draw(payload)
        sig = abs(payload["step"]) / max(payload["resid"], 1e-9)
        self.hold_status.setText(
            f"Done · {payload['n']}/{payload['n_req']} repeats edge-aligned"
        )
        self._hold_result.setText(
            f"Settle to 2%: {payload['settle']*1000:.0f} ms after edge  ·  "
            f"step {payload['step']*1000:.2f} mV  ·  residual noise "
            f"{payload['resid']*1000:.2f} mV  ·  step/noise {sig:.1f}  ·  "
            f"PC↔scope offset {payload['offset']*1000:+.0f} ms"
            + ("  \N{WARNING SIGN} step not significant (use higher-contrast patterns)"
               if sig < 3 else "")
        )

    def _hold_error(self, _error: str) -> None:
        self.hold_stop_event = None
        self._hold_set_running(False)
        self.hold_status.setText("Measurement failed (see Status log)")

    def _hold_draw(self, p: dict[str, Any]) -> None:
        self.hold_fig.clear()
        self.hold_fig.patch.set_facecolor("#101820")
        ax = self.hold_fig.add_subplot(111)
        self._style_dark_axes(ax)
        ax.set_xlabel("time after A→B edge (ms)"); ax.set_ylabel("CH (mV)")
        t = p["t"] * 1000.0
        if p.get("first") is not None:
            ax.plot(t, p["first"] * 1000.0, lw=0.5, color="#556", label="single raw")
        ax.plot(t, p["avg"] * 1000.0, lw=1.4, color="#47b8e0", label=f"avg N={p['n']}")
        ax.axvline(0.0, color="#8fd14f", ls="-", lw=1.0, label="A→B edge (detected)")
        ax.axvline(p["cmd_rel"] * 1000.0, color="#f0a3a3", ls="--", lw=1.2,
                   label="PC switch cmd")
        if p.get("settle"):
            ax.axvline(p["settle"] * 1000.0, color="#e0a447", ls=":", lw=1.0,
                       label="settled (2%)")
        ax.axhline(p["final"] * 1000.0, color="#8fd6a0", ls=":", lw=1.0)
        ax.legend(loc="upper right", fontsize=8)
        self.hold_canvas.draw_idle()

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
        self.scan_eta_label = QtWidgets.QLabel("Elapsed 0:00 · ETA —")
        self.scan_center_label = QtWidgets.QLabel("Center: \N{EN DASH}")
        self._set_status(self.scan_center_label, "Center: \N{EN DASH}", "off")
        status_row.addWidget(self.scan_size_label)
        status_row.addStretch(1)
        status_row.addWidget(self.scan_signal_label)
        status_row.addWidget(self.scan_eta_label)
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

        # --- Layout config panel ---
        cfg_panel = self._panel("Channel Layout")
        cfg_grid = QtWidgets.QGridLayout(cfg_panel)

        self.enc_center_wl_spin = self._double_spin(700.0, 900.0, 778.0, " nm", 2)
        self.enc_width_spin = self._spin(1, 256, 15)
        self.enc_pad_spin   = self._spin(0, 64, 5)

        self.enc_calib_label = QtWidgets.QLabel("Calibration: (none loaded)")
        self.enc_calib_label.setObjectName("PageSubtitle")
        enc_reload = QtWidgets.QPushButton("Load other…")
        enc_reload.setProperty("variant", "ghost")
        enc_reload.setToolTip("Override the local calibration with another result file")
        enc_reload.clicked.connect(self._enc_browse_calib)

        self.enc_build_button = QtWidgets.QPushButton("Build Layout")
        self.enc_build_button.clicked.connect(self._enc_build_layout)

        self.enc_layout_status = QtWidgets.QLabel("Configure parameters and click Build Layout")
        self.enc_layout_status.setWordWrap(True)

        self.enc_width_spin.valueChanged.connect(self._enc_update_channel_count)
        self.enc_pad_spin.valueChanged.connect(self._enc_update_channel_count)
        self.enc_center_wl_spin.valueChanged.connect(self._enc_update_channel_count)

        cfg_grid.addWidget(QtWidgets.QLabel("Centre λ"),      0, 0)
        cfg_grid.addWidget(self.enc_center_wl_spin,           0, 1)
        cfg_grid.addWidget(QtWidgets.QLabel("Channel width"),  0, 2)
        cfg_grid.addWidget(self.enc_width_spin,               0, 3)
        cfg_grid.addWidget(QtWidgets.QLabel("px   Padding"),  0, 4)
        cfg_grid.addWidget(self.enc_pad_spin,                 0, 5)
        cfg_grid.addWidget(QtWidgets.QLabel("px"),            0, 6)
        cfg_grid.addWidget(self.enc_calib_label,             1, 0, 1, 5)
        cfg_grid.addWidget(enc_reload,                        1, 5)
        cfg_grid.addWidget(self.enc_build_button,             1, 6)
        cfg_grid.addWidget(self.enc_layout_status,            2, 0, 1, 7)

        # --- Channel values table ---
        # Columns: # | x λ (nm) | x value [0-1] | w λ (nm) | w value [0-1]
        self.enc_val_table = QtWidgets.QTableWidget(0, 5)
        self.enc_val_table.setHorizontalHeaderLabels(
            ["#", "x  λ (nm)", "x value [0–1]", "w  λ (nm)", "w value [0–1]"]
        )
        hdr = self.enc_val_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        hdr.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        hdr.setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)
        hdr.setSectionResizeMode(4, QtWidgets.QHeaderView.Stretch)
        self.enc_val_table.verticalHeader().setVisible(False)
        self.enc_val_table.setAlternatingRowColors(True)
        # keep at least ~3 data rows (plus header) visible even when the splitter
        # is dragged small
        self.enc_val_table.setMinimumHeight(170)

        val_buttons = QtWidgets.QHBoxLayout()
        enc_zeros = QtWidgets.QPushButton("All Zeros")
        enc_zeros.setProperty("variant", "ghost")
        enc_zeros.clicked.connect(lambda: self._enc_fill_values(0.0))
        enc_ones = QtWidgets.QPushButton("All Ones")
        enc_ones.setProperty("variant", "ghost")
        enc_ones.clicked.connect(lambda: self._enc_fill_values(1.0))
        enc_randomize = QtWidgets.QPushButton("Randomize")
        enc_randomize.setProperty("variant", "ghost")
        enc_randomize.clicked.connect(self._enc_randomize)
        self.enc_wheel_step_spin = QtWidgets.QDoubleSpinBox()
        self.enc_wheel_step_spin.setRange(0.01, 1.0)
        self.enc_wheel_step_spin.setDecimals(2)
        self.enc_wheel_step_spin.setSingleStep(0.05)
        self.enc_wheel_step_spin.setValue(self._enc_wheel_step)
        self.enc_wheel_step_spin.setToolTip(
            "Mouse-wheel step for the channel value cells "
            "(a few scrolls span 0→1 at higher values)"
        )
        self.enc_wheel_step_spin.valueChanged.connect(self._enc_set_wheel_step)
        val_buttons.addWidget(enc_zeros)
        val_buttons.addWidget(enc_ones)
        val_buttons.addWidget(enc_randomize)
        self.enc_use_optimized_lut = QtWidgets.QCheckBox(
            "Values are amplitudes (use optimized LUT)"
        )
        self.enc_use_optimized_lut.setEnabled(False)
        self.enc_use_optimized_lut.setToolTip(
            "Convert each target amplitude through the nearest measured final "
            "channel LUT before applying the 15-pixel intensity profile."
        )
        val_buttons.addWidget(self.enc_use_optimized_lut)
        val_buttons.addStretch(1)
        val_buttons.addWidget(QtWidgets.QLabel("Scroll step"))
        val_buttons.addWidget(self.enc_wheel_step_spin)

        val_panel = self._panel("Channel Values  [0 = off · 1 = on]")
        val_layout = QtWidgets.QVBoxLayout(val_panel)
        val_layout.addWidget(self.enc_val_table, 1)
        val_layout.addLayout(val_buttons)

        # --- Controls row ---
        ctrl_row = QtWidgets.QHBoxLayout()
        self.enc_generate_button = QtWidgets.QPushButton("Generate & Preview")
        self.enc_generate_button.setEnabled(False)
        self.enc_generate_button.clicked.connect(self._enc_generate)
        self.enc_send_button = QtWidgets.QPushButton("Send to SLM")
        self.enc_send_button.setEnabled(False)
        self.enc_send_button.clicked.connect(self._enc_send)
        self.enc_status_label = QtWidgets.QLabel("\N{EN DASH}")
        ctrl_row.addWidget(self.enc_status_label, 1)
        ctrl_row.addWidget(self.enc_generate_button)
        ctrl_row.addWidget(self.enc_send_button)

        # --- live SLM pattern monitor (colour) replaces the static preview ---
        # short-and-wide monitor: the pattern is already wide (1920x1200), so a
        # low image band + a compact profile keeps most of the height for the
        # channel-value table below
        self.enc_monitor_view = SLMMonitorView(
            get_pattern=lambda: self._encoding_pattern,
            describe=lambda: "generated encoding pattern",
            image_min_height=110,
            show_profile=False,
        )
        monitor_panel = self._panel_with_widget("Pattern Monitor", self.enc_monitor_view)

        left_split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        left_split.addWidget(val_panel)
        left_split.addWidget(monitor_panel)
        left_split.setStretchFactor(0, 3)   # value table gets the bulk of the height
        left_split.setStretchFactor(1, 1)
        left_split.setSizes([460, 230])

        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(left_split, 1)
        left_layout.addLayout(ctrl_row)

        # scope readout merged in at ~1/3 of the width
        main_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_split.addWidget(left)
        main_split.addWidget(self._build_scope_monitor_widget())
        main_split.setStretchFactor(0, 2)
        main_split.setStretchFactor(1, 1)
        main_split.setSizes([840, 420])

        # --- single Feedback log for both SLM and scope, spanning full width ---
        self.enc_log = QtWidgets.QPlainTextEdit()
        self.enc_log.setReadOnly(True)
        self.enc_log.setObjectName("LogBox")
        self.enc_log.setMaximumHeight(120)
        log_panel = self._panel_with_widget("Feedback", self.enc_log)

        page.layout().addWidget(cfg_panel)
        page.layout().addWidget(main_split, 1)
        page.layout().addWidget(log_panel)

        # auto-load the local calibration and build a default layout so the
        # value table is populated and ready for manual input immediately
        QtCore.QTimer.singleShot(0, self._enc_autostart)
        return page

    def _enc_autostart(self) -> None:
        """Load local calibration and build the default layout on first show."""
        calib = self._enc_get_calib()
        if calib is None or calib.intensity_levels is None:
            self._enc_log(
                "No calibration found. Run Step 3 on the Calibration page, or use "
                "'Load other…' to pick a result file."
            )
            return
        self._enc_build_layout()

    # ------------------------------------------------------------------
    # Encoding page handlers
    # ------------------------------------------------------------------

    def _enc_log(self, message: str) -> None:
        """Append a timestamped hint/action line to the encoding feedback box."""
        stamp = time.strftime("%H:%M:%S")
        self.enc_log.appendPlainText(f"[{stamp}] {message}")

    def _mon_status(self, message: str) -> None:
        """Scope-monitor feedback shares the encoder's merged Feedback log."""
        self._enc_log(f"[scope] {message}")

    # calibration results are named calib_step*.json (calib_step3.json,
    # calib_step33.json, ...); the encoder needs a step-3 intensity result.
    _CALIB_RE = re.compile(r"^calib_step.*\.json$", re.IGNORECASE)

    def _enc_local_calib_path(self) -> Path | None:
        """Locate the newest usable project-local calibration result.

        Scans the working dir and project root for calib_step*.json files and
        returns the most recently modified one that loads as a valid intensity
        (step-3) result. Files without intensity_levels (step-1/step-2 outputs)
        are skipped so the encoder never auto-loads an unusable calibration.
        """
        search_dirs = [Path.cwd(), Path(__file__).resolve().parents[3]]
        matches: dict[Path, float] = {}
        for directory in search_dirs:
            try:
                for entry in directory.iterdir():
                    if entry.is_file() and self._CALIB_RE.match(entry.name):
                        matches.setdefault(entry.resolve(), entry.stat().st_mtime)
            except OSError:
                continue
        for path in sorted(matches, key=matches.get, reverse=True):
            try:
                calib = load_calibration_result(str(path))
            except Exception:
                continue
            if calib.intensity_levels is not None:
                return path
        return None

    def _enc_get_calib(self) -> CalibrationResult | None:
        """Calibration source: explicit override → in-memory result → local file."""
        if self._enc_calib_override is not None:
            return self._enc_calib_override
        if self.calibration_result is not None and self.calibration_result.intensity_levels is not None:
            self.enc_calib_label.setText("Calibration: in-memory (from Step 3 / loaded fit)")
            return self.calibration_result
        local = self._enc_local_calib_path()
        if local is not None:
            try:
                calib = load_calibration_result(str(local))
                self.enc_calib_label.setText(f"Calibration: {local.name} (local)")
                return calib
            except Exception as exc:
                self._enc_log(f"Failed to read {local.name}: {exc}")
                return None
        return None

    def _enc_update_channel_count(self) -> None:
        pitch = self.enc_width_spin.value() + self.enc_pad_spin.value()
        calib = self._enc_get_calib()
        if calib is None or calib.intensity_levels is None:
            self.enc_layout_status.setText(
                f"Pitch = {pitch} px  —  no calibration available"
            )
            return
        coords = np.asarray(calib.coordinates, dtype=float)
        wls    = np.asarray(calib.wavelength,  dtype=float)
        a, b   = np.polyfit(coords, wls, 1)
        cx     = (self.enc_center_wl_spin.value() - b) / a
        max_ch = int(min(cx - coords.min(), coords.max() - cx) / pitch)
        self.enc_layout_status.setText(
            f"Pitch = {pitch} px  |  max {max_ch} channels per side  "
            f"(calib x = {int(coords.min())}–{int(coords.max())} px, "
            f"centre ≈ {int(cx)} px)"
        )

    def _enc_browse_calib(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Calibration Result", "", "JSON Files (*.json)"
        )
        if not path:
            return
        try:
            self._enc_calib_override = load_calibration_result(path)
        except Exception as exc:
            self._enc_log(f"Failed to load {Path(path).name}: {exc}")
            return
        self.enc_calib_label.setText(f"Calibration: {Path(path).name} (override)")
        self._enc_log(f"Loaded calibration override: {path}")
        self._enc_build_layout()

    def _enc_build_layout(self) -> None:
        calib = self._enc_get_calib()
        if calib is None or calib.intensity_levels is None:
            self.enc_layout_status.setText(
                "No calibration available. Run Step 3 or load a result file."
            )
            return

        # compute max channels from calibrated range
        coords = np.asarray(calib.coordinates, dtype=float)
        wls    = np.asarray(calib.wavelength,  dtype=float)
        a, b   = np.polyfit(coords, wls, 1)
        cx     = (self.enc_center_wl_spin.value() - b) / a
        pitch  = self.enc_width_spin.value() + self.enc_pad_spin.value()
        n_ch   = int(min(cx - coords.min(), coords.max() - cx) / pitch)
        if n_ch < 1:
            self.enc_layout_status.setText(
                "Pitch too large — no channels fit on both sides of the centre wavelength."
            )
            return

        try:
            layout = build_channel_layout(
                calib,
                n_channels=n_ch,
                channel_width_px=self.enc_width_spin.value(),
                gap_px=self.enc_pad_spin.value(),
                center_wl=self.enc_center_wl_spin.value(),
            )
        except Exception as exc:
            self.enc_layout_status.setText(f"Layout error: {exc}")
            return

        self.encoding_layout = layout
        self._edge_optimization_result = None
        self.enc_use_optimized_lut.setChecked(False)
        self.enc_use_optimized_lut.setEnabled(False)
        self._enc_populate_val_table(layout)
        self._edge_sync_layout(layout)
        self.enc_layout_status.setText(
            f"{n_ch} channels per side  |  "
            f"x: {layout.x_channels[-1].wavelength_nm:.3f}–{layout.x_channels[0].wavelength_nm:.3f} nm  |  "
            f"w: {layout.w_channels[0].wavelength_nm:.3f}–{layout.w_channels[-1].wavelength_nm:.3f} nm  |  "
            f"pitch {layout.pitch_px} px  |  padding {layout.pitch_px - layout.channel_width_px} px (nearest off level)"
        )
        self.enc_generate_button.setEnabled(True)
        self._enc_log(
            f"Layout built: {n_ch} channels/side, width "
            f"{layout.channel_width_px} px, padding {layout.pitch_px - layout.channel_width_px} px. "
            "Edit values in the table, then Generate & Preview."
        )

    def _enc_populate_val_table(self, layout: ChannelLayout) -> None:
        n = layout.n_channels
        self.enc_val_table.setRowCount(n)
        for i in range(n):
            xch = layout.x_channels[i]
            wch = layout.w_channels[i]

            idx_item = QtWidgets.QTableWidgetItem(str(i))
            idx_item.setTextAlignment(QtCore.Qt.AlignCenter)
            idx_item.setFlags(idx_item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.enc_val_table.setItem(i, 0, idx_item)

            for col, text in [(1, f"{xch.wavelength_nm:.4f}"), (3, f"{wch.wavelength_nm:.4f}")]:
                item = QtWidgets.QTableWidgetItem(text)
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                self.enc_val_table.setItem(i, col, item)

            for col in (2, 4):
                spin = WheelSpinBox(wheel_step=self._enc_wheel_step)
                spin.setRange(0.0, 1.0)
                spin.setSingleStep(0.01)     # fine step for arrows / typing
                spin.setDecimals(3)
                spin.setValue(0.0)
                spin.setFrame(False)
                self.enc_val_table.setCellWidget(i, col, spin)

        self.enc_val_table.resizeRowsToContents()

    def _enc_set_wheel_step(self, step: float) -> None:
        self._enc_wheel_step = float(step)
        for i in range(self.enc_val_table.rowCount()):
            for col in (2, 4):
                w = self.enc_val_table.cellWidget(i, col)
                if isinstance(w, WheelSpinBox):
                    w.wheel_step = self._enc_wheel_step

    def _enc_fill_values(self, value: float) -> None:
        if self.encoding_layout is None:
            return
        for i in range(self.encoding_layout.n_channels):
            for col in (2, 4):
                w = self.enc_val_table.cellWidget(i, col)
                if w:
                    w.setValue(value)

    def _enc_randomize(self) -> None:
        if self.encoding_layout is None:
            return
        rng = np.random.default_rng()
        vals = rng.uniform(0.0, 1.0, (self.encoding_layout.n_channels, 2))
        for i in range(self.encoding_layout.n_channels):
            for j, col in enumerate((2, 4)):
                w = self.enc_val_table.cellWidget(i, col)
                if w:
                    w.setValue(float(vals[i, j]))

    def _enc_get_values(self) -> tuple[np.ndarray, np.ndarray] | None:
        layout = self.encoding_layout
        if layout is None:
            return None
        n = layout.n_channels
        x_vals = np.zeros(n)
        w_vals = np.zeros(n)
        for i in range(n):
            xw = self.enc_val_table.cellWidget(i, 2)
            ww = self.enc_val_table.cellWidget(i, 4)
            if xw:
                x_vals[i] = xw.value()
            if ww:
                w_vals[i] = ww.value()
        return x_vals, w_vals

    def _enc_generate(self) -> None:
        layout = self.encoding_layout
        if layout is None:
            return
        parsed = self._enc_get_values()
        if parsed is None:
            return
        x_vals, w_vals = parsed
        use_amplitude_lut = self.enc_use_optimized_lut.isChecked()
        if use_amplitude_lut:
            result = self._edge_optimization_result
            ratio = self._edge_get_ratio()
            if result is None:
                self.enc_status_label.setText("No optimized amplitude LUT is loaded")
                return
            if ratio is None or not np.allclose(
                ratio, result.final_profile, atol=5e-4
            ):
                self.enc_status_label.setText(
                    "Optimized LUT is invalid because the intensity profile changed"
                )
                self._enc_log(
                    "Re-run OSA optimisation before using amplitude-mode encoding."
                )
                return
            try:
                x_vals, w_vals = amplitudes_to_intensity_commands(
                    x_vals, w_vals, layout, result.final_luts
                )
            except Exception as exc:
                self.enc_status_label.setText(f"Amplitude LUT error: {exc}")
                return
        slm_w, slm_h = self.slm_size
        try:
            pattern = encode_to_pattern(
                x_vals, w_vals, layout, slm_w, slm_h,
                col_ratio=self._edge_get_ratio(),
            )
        except Exception as exc:
            self.enc_status_label.setText(f"Encoding error: {exc}")
            self._enc_log(f"Encoding error: {exc}")
            return
        self._encoding_pattern = pattern
        self.enc_send_button.setEnabled(True)
        self.enc_status_label.setText(
            f"Pattern ready  |  SLM levels {int(pattern.min())}–{int(pattern.max())}"
        )
        self._enc_log(
            f"Pattern generated ({slm_w}x{slm_h}, levels "
            f"{int(pattern.min())}–{int(pattern.max())}). Open the SLM and click "
            "Send to SLM to display it."
        )
        # dimmed preview: what's shown is generated but not yet on the SLM
        self.enc_monitor_view.set_preview(True)

    def _enc_send(self) -> None:
        pattern = self._encoding_pattern
        if pattern is None:
            self._enc_log("Nothing to send — click Generate & Preview first.")
            return
        controller = self._controller()
        if not getattr(controller, "is_open", False):
            self._enc_log(
                "SLM is not open. Open it on the Connections page first, then "
                "click Send to SLM again."
            )
            return
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        tmp.close()
        write_santec_csv(pattern, tmp.name)
        self._enc_log("Sending encoding pattern to SLM… (full-res transfer, a few seconds)")
        self.enc_send_button.setEnabled(False)

        def _cleanup() -> None:
            try:
                Path(tmp.name).unlink(missing_ok=True)
            except OSError:
                pass

        def done(_result: Any) -> None:
            self._enc_log("\N{CHECK MARK} Pattern received and displayed on the SLM.")
            # pattern is now live on the SLM: clear the dim preview veil
            self.enc_monitor_view.set_preview(False)
            _cleanup()
            if self._enc_should_read_scope():
                self._enc_read_scope_after_send()   # keeps Send disabled until read done
            else:
                self.enc_send_button.setEnabled(True)

        def failed(_error: str) -> None:
            self._enc_log("\N{CROSS MARK} Send failed (see the Status log on the Connections page).")
            self.enc_send_button.setEnabled(True)
            _cleanup()

        self._run_slm_task(
            "Send encoding pattern",
            lambda: controller.display_csv(tmp.name),
            done, failed,
        )

    def _enc_should_read_scope(self) -> bool:
        """Take a scope reading after a send only if it's safe/enabled."""
        scope = self.scope_controller
        return (
            scope is not None and scope.is_connected
            and self.mon_read_on_send.isChecked()
            and self.monitor_stop_event is None   # not already running the trigger loop
        )

    def _enc_read_scope_after_send(self) -> None:
        """Path C: after the pattern is displayed, wait the hold then read one
        software-triggered averaged value and append it to the scope monitor."""
        scope = self.scope_controller
        # AUTO free-run with no armed edge: the SINGle self-triggers and completes
        # right away (the earlier timeout was a stale ACQuire:COUNt, now forced to
        # 1 in configure_monitor). This is the Path-C software read.
        settings = self._monitor_settings(trigger_mode="AUTO")
        self._mon_status("Reading scope after send…")

        def work() -> MonitorSample | None:
            scope.configure_monitor(settings)
            time.sleep(settings.hold)             # settle after the SLM pattern change
            return scope.monitor_cycle(
                index=len(self._monitor_values),
                timeout=max(30.0, settings.duration * 3.0 + 10.0),
            )

        def ok(sample: MonitorSample | None) -> None:
            if sample is not None:
                self._on_monitor_sample(sample)
            else:
                self._mon_status("Scope read returned nothing.")
            self.enc_send_button.setEnabled(True)

        def err(_error: str) -> None:
            self._mon_status("Scope read failed (see Status log).")
            self.enc_send_button.setEnabled(True)

        self._run_task("Scope read on send", work, ok, err)

    # ==================================================================
    # Edge Ratio page: per-column edge-taper ratio + OSA optimisation hook
    # ==================================================================

    def _build_edge_ratio_page(self) -> QtWidgets.QWidget:
        page = self._page_shell("Edge Ratio")

        subtitle = QtWidgets.QLabel(
            "Shape the profile inside each channel with a per-column intensity ratio. "
            "Column j encodes level_for(intensity command × ratio[j]), i.e. "
            "edge = ratio × (max − min) + min, where min is the channel's measured "
            "background. All-1.0 reproduces the flat band. Build a layout on the "
            "TPA Encoding page first (channel width sets the number of columns)."
        )
        subtitle.setObjectName("PageSubtitle")
        subtitle.setWordWrap(True)
        page.layout().addWidget(subtitle)

        # --- per-column ratio table (1 row, channel_width_px columns) ---
        self._edge_spins: list[WheelSpinBox] = []
        self.edge_width_label = QtWidgets.QLabel("Channel width: (no layout built)")
        self.edge_width_label.setObjectName("PageSubtitle")

        self.edge_table = QtWidgets.QTableWidget(1, 0)
        self.edge_table.verticalHeader().setVisible(False)
        self.edge_table.setMaximumHeight(90)
        self.edge_table.setToolTip(
            "Ratio per column across the channel width (left→right). 1.0 = full "
            "value, 0.0 = channel's measured background."
        )

        edge_buttons = QtWidgets.QHBoxLayout()
        edge_all1 = QtWidgets.QPushButton("All 1.0")
        edge_all1.setProperty("variant", "ghost")
        edge_all1.clicked.connect(lambda: self._edge_set_all(1.0))
        edge_cos = QtWidgets.QPushButton("Cosine taper…")
        edge_cos.setProperty("variant", "ghost")
        edge_cos.setToolTip("Fill a raised-cosine taper over the outer k columns of each edge")
        edge_cos.clicked.connect(self._edge_apply_cosine)
        edge_mirror = QtWidgets.QPushButton("Mirror L→R")
        edge_mirror.setProperty("variant", "ghost")
        edge_mirror.setToolTip("Copy the left half onto the right half (symmetric profile)")
        edge_mirror.clicked.connect(self._edge_mirror)
        edge_buttons.addWidget(self.edge_width_label, 1)
        edge_buttons.addWidget(edge_all1)
        edge_buttons.addWidget(edge_cos)
        edge_buttons.addWidget(edge_mirror)

        ratio_panel = self._panel("Per-column Ratio  [0 = background · 1 = full value]")
        ratio_layout = QtWidgets.QVBoxLayout(ratio_panel)
        ratio_layout.addWidget(self.edge_table)
        ratio_layout.addLayout(edge_buttons)

        # --- preview (matplotlib) ---
        ref_row = QtWidgets.QHBoxLayout()
        ref_row.addWidget(QtWidgets.QLabel("Preview at scalar intensity command"))
        self.edge_ref_val = self._double_spin(0.0, 1.0, 1.0, "", 3)
        self.edge_ref_val.setSingleStep(0.05)
        self.edge_ref_val.valueChanged.connect(lambda _=None: self._edge_draw_preview())
        ref_row.addWidget(self.edge_ref_val)
        ref_row.addStretch(1)

        self.edge_figure = Figure(figsize=(10, 2.4), tight_layout=True)
        self.edge_canvas = FigureCanvas(self.edge_figure)
        self.edge_canvas.setMinimumHeight(180)
        preview_panel = self._panel("Profile Preview")
        preview_layout = QtWidgets.QVBoxLayout(preview_panel)
        preview_layout.addLayout(ref_row)
        preview_layout.addWidget(self.edge_canvas, 1)

        # --- A/B encoding gain via Modulation Error (chains the two features) ---
        self.edge_gain_button = QtWidgets.QPushButton("Measure encoding gain")
        self.edge_gain_button.setToolTip(
            "Run the Modulation Error sweep twice — flat baseline then the current "
            "edge-ratio profile — and report the per-channel change in neighbour "
            "leakage and in-band fraction. Needs OSA + SLM connected and a layout "
            "built. Uses the sweep settings on Calibration ▸ Step 4."
        )
        self.edge_gain_button.clicked.connect(self._edge_measure_gain)
        self.edge_gain_stop_button = QtWidgets.QPushButton("Stop")
        self.edge_gain_stop_button.setProperty("variant", "danger")
        self.edge_gain_stop_button.setEnabled(False)
        self.edge_gain_stop_button.clicked.connect(self._edge_gain_stop)
        self.edge_gain_save_button = QtWidgets.QPushButton("Save gain CSV…")
        self.edge_gain_save_button.setProperty("variant", "ghost")
        self.edge_gain_save_button.setEnabled(False)
        self.edge_gain_save_button.clicked.connect(self._edge_gain_save)
        self.edge_osa_button = QtWidgets.QPushButton("Optimize from OSA")
        self.edge_osa_button.setToolTip(
            "Run the two-stage live optimisation: one-hot crosstalk search, "
            "coarse amplitude LUT, then fixed-LUT modulation-fidelity search. "
            "The first 8 table values are treated as symmetric intensity ratios."
        )
        self.edge_osa_button.clicked.connect(self._edge_optimize_osa)
        self.edge_load_optimization_button = QtWidgets.QPushButton(
            "Load optimized result…"
        )
        self.edge_load_optimization_button.setProperty("variant", "ghost")
        self.edge_load_optimization_button.setToolTip(
            "Load an accepted final_result.json and restore its intensity "
            "profile plus amplitude LUTs."
        )
        self.edge_load_optimization_button.clicked.connect(
            self._edge_load_optimization_result
        )

        self.edge_gain_bar = QtWidgets.QProgressBar()
        self.edge_gain_bar.setValue(0)
        self.edge_gain_status = QtWidgets.QLabel("\N{EN DASH}")
        self.edge_gain_status.setWordWrap(True)

        self.edge_gain_table = QtWidgets.QTableWidget(0, 6)
        self.edge_gain_table.setHorizontalHeaderLabels(
            ["Ch", "λ (nm)", "Δ Leak (pp)", "Δ In-band (pp)", "Win loss %", "Tot loss %"]
        )
        self.edge_gain_table.setToolTip(
            "Δ Leak / Δ In-band: crosstalk benefit (leak down, in-band up = good). "
            "Win/Tot loss: intensity lost in the encoding window / whole channel "
            "vs the trivial rectangular encoding (the taper's cost)."
        )
        self.edge_gain_table.verticalHeader().setVisible(False)
        self.edge_gain_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.edge_gain_table.setAlternatingRowColors(True)
        ghdr = self.edge_gain_table.horizontalHeader()
        ghdr.setSectionResizeMode(QtWidgets.QHeaderView.Stretch)

        self.edge_log = QtWidgets.QPlainTextEdit()
        self.edge_log.setReadOnly(True)
        self.edge_log.setObjectName("LogBox")
        self.edge_log.setMaximumHeight(90)

        gain_panel = self._panel("Encoding Gain  (Modulation Error A/B: flat vs taper)")
        gain_layout = QtWidgets.QVBoxLayout(gain_panel)
        gain_row = QtWidgets.QHBoxLayout()
        gain_row.addWidget(self.edge_gain_button)
        gain_row.addWidget(self.edge_gain_stop_button)
        gain_row.addWidget(self.edge_gain_save_button)
        gain_row.addStretch(1)
        gain_row.addWidget(self.edge_load_optimization_button)
        gain_row.addWidget(self.edge_osa_button)
        gain_layout.addLayout(gain_row)
        gain_layout.addWidget(self.edge_gain_bar)
        gain_layout.addWidget(self.edge_gain_status)
        gain_layout.addWidget(self.edge_gain_table, 1)
        gain_layout.addWidget(self.edge_log)

        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        split.addWidget(ratio_panel)
        split.addWidget(preview_panel)
        split.addWidget(gain_panel)
        split.setSizes([150, 220, 320])
        page.layout().addWidget(split, 1)

        self._edge_draw_preview()
        return page

    def _edge_log(self, message: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.edge_log.appendPlainText(f"[{stamp}] {message}")

    def _edge_sync_layout(self, layout: ChannelLayout) -> None:
        """Rebuild the ratio table to the layout's channel width (default 1.0).

        Overlapping columns keep their previous values so tweaking the layout
        does not silently discard a tuned profile.
        """
        width = int(layout.channel_width_px)
        prev = self._edge_get_ratio()
        self.edge_table.clear()
        self.edge_table.setColumnCount(width)
        self.edge_table.setHorizontalHeaderLabels([str(j) for j in range(width)])
        self._edge_spins = []
        for j in range(width):
            spin = WheelSpinBox(wheel_step=self._enc_wheel_step)
            spin.setRange(0.0, 1.0)
            spin.setSingleStep(0.01)
            spin.setDecimals(3)
            value = float(prev[j]) if prev is not None and j < len(prev) else 1.0
            spin.setValue(value)
            spin.setFrame(False)
            spin.valueChanged.connect(lambda _=None: self._edge_draw_preview())
            self.edge_table.setCellWidget(0, j, spin)
            self._edge_spins.append(spin)
        self.edge_table.resizeColumnsToContents()
        self.edge_width_label.setText(
            f"Channel width: {width} px  ({layout.n_channels} channels/side)"
        )
        self._edge_draw_preview()

    def _edge_get_ratio(self) -> np.ndarray | None:
        """Current per-column ratio profile, or None when no layout is built."""
        spins = getattr(self, "_edge_spins", None)
        if not spins:
            return None
        return np.array([s.value() for s in spins], dtype=float)

    def _edge_set_all(self, value: float) -> None:
        for spin in getattr(self, "_edge_spins", []):
            spin.setValue(float(value))

    def _edge_apply_cosine(self) -> None:
        spins = getattr(self, "_edge_spins", [])
        width = len(spins)
        if width == 0:
            return
        k, ok = QtWidgets.QInputDialog.getInt(
            self, "Cosine taper", "Edge columns to taper (per side):",
            min(2, width), 1, width, 1
        )
        if not ok:
            return
        j = np.arange(width, dtype=float)
        d = np.minimum(j + 0.5, width - j - 0.5)
        ratios = np.ones(width, dtype=float)
        taper = d < k
        ratios[taper] = 0.5 - 0.5 * np.cos(np.pi * d[taper] / k)
        for spin, r in zip(spins, ratios):
            spin.setValue(float(r))

    def _edge_mirror(self) -> None:
        spins = getattr(self, "_edge_spins", [])
        width = len(spins)
        if width == 0:
            return
        for i in range(width // 2):
            spins[width - 1 - i].setValue(spins[i].value())

    def _edge_draw_preview(self) -> None:
        ratios = self._edge_get_ratio()
        self.edge_figure.clear()
        ax = self.edge_figure.add_subplot(111)
        if ratios is None or len(ratios) == 0:
            ax.text(0.5, 0.5, "Build a layout on the TPA Encoding page",
                    ha="center", va="center", color="#d8dee9", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        else:
            cols = np.arange(len(ratios))
            ax.step(cols, ratios, where="mid", color="#88c0d0", linewidth=1.5,
                    marker="o", markersize=4, label="ratio")
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel("column within channel", color="#d8dee9", fontsize=8)
            ax.set_ylabel("ratio", color="#88c0d0", fontsize=8)
            layout = self.encoding_layout
            if layout is not None and layout.all_channels:
                ch = layout.x_channels[0]
                val = float(self.edge_ref_val.value())
                levels = np.array([ch.level_for(val * float(r)) for r in ratios])
                ax2 = ax.twinx()
                ax2.step(cols, levels, where="mid", color="#ebcb8b", linewidth=1.2,
                         linestyle="--", marker="s", markersize=3, label="SLM level")
                ax2.set_ylabel(f"SLM level @ value {val:g}", color="#ebcb8b", fontsize=8)
                ax2.tick_params(colors="#ebcb8b", labelsize=7)
                for spine in ax2.spines.values():
                    spine.set_color("#41515c")
            ax.tick_params(colors="#d8dee9", labelsize=7)
            for spine in ax.spines.values():
                spine.set_color("#41515c")
        self.edge_figure.patch.set_facecolor("#101820")
        ax.set_facecolor("#101820")
        self.edge_canvas.draw_idle()

    def _edge_optimize_osa(self) -> None:
        """Start the live two-stage symmetric intensity-profile optimisation."""
        layout = self.encoding_layout
        if layout is None:
            self._edge_log("Build a layout on the TPA Encoding page first.")
            return
        if layout.channel_width_px != 15:
            self._edge_log("OSA optimisation currently requires a 15 px channel width.")
            return
        osa = self._osa_ready()
        if osa is None:
            self._edge_log("Connect the OSA first.")
            return
        controller = self._controller()
        if not getattr(controller, "is_open", False):
            self._edge_log("Open the SLM first.")
            return
        ratio = self._edge_get_ratio()
        if ratio is None or ratio.size != 15:
            self._edge_log("A 15-value intensity profile is required.")
            return

        answer = QtWidgets.QMessageBox.question(
            self,
            "Start OSA optimisation",
            "This performs hundreds of live OSA sweeps and may run overnight. "
            "The first 8 intensity values will be mirrored to 15 pixels. Continue?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return

        ana = self._ana_settings()
        settings = MeasurementSettings(
            center_wl="778nm",
            span="0.8nm",
            sensitivity=ana.sensitivity,
            sampling_points="1001",
            y_unit=ana.y_unit,
            reference_level=ana.reference_level,
            trace_id=ana.trace_id,
            trace_mode=ana.trace_mode,
        )
        config = OSAOptimizationConfig(settings=settings)
        initial_l = ratio[:8].copy()
        stop_event = threading.Event()
        self.edge_gain_stop_event = stop_event
        self._edge_gain_running(True)
        self.edge_gain_bar.setRange(0, 0)
        self.edge_gain_status.setText("Preparing fixed OSA bins and references…")
        self._edge_log(
            "OSA optimisation started with 8 intensity ratios: "
            + np.array2string(initial_l, precision=4)
        )

        def report(progress: OptimizationProgress) -> None:
            self.edge_optimization_progress.emit(progress)

        def work() -> dict[str, Any]:
            try:
                result = optimize_from_osa(
                    layout,
                    osa=osa,
                    slm=controller,
                    initial_l=initial_l,
                    config=config,
                    stop_event=stop_event,
                    progress_callback=report,
                )
            except OptimizationAborted:
                return {"status": "aborted"}
            return {"status": "ok", "result": result}

        self._run_slm_task(
            "OSA intensity-profile optimisation",
            work,
            self._edge_optimization_finished,
            self._edge_optimization_error,
        )

    def _edge_optimization_progress(self, progress: OptimizationProgress) -> None:
        if progress.total > 0:
            self.edge_gain_bar.setRange(0, progress.total)
            self.edge_gain_bar.setValue(min(progress.step, progress.total))
            counter = f"[{progress.step}/{progress.total}] "
        else:
            self.edge_gain_bar.setRange(0, 0)
            counter = ""
        best = "" if progress.best_loss is None else f" · best {progress.best_loss:.5g}"
        self.edge_gain_status.setText(
            f"{counter}{progress.stage}: {progress.message}{best}"
        )

    def _edge_optimization_finished(self, payload: dict[str, Any]) -> None:
        self.edge_gain_stop_event = None
        self._edge_gain_running(False)
        self.edge_gain_bar.setRange(0, 100)
        if payload.get("status") == "aborted":
            self.edge_gain_bar.setValue(0)
            self.edge_gain_status.setText("Stopped — completed candidates remain on disk.")
            self._edge_log("OSA optimisation stopped; saved candidate data were retained.")
            return
        result: OptimizationResult = payload["result"]
        self._edge_optimization_result = result
        if result.accepted:
            for spin, value in zip(self._edge_spins, result.final_profile):
                spin.setValue(float(value))
            self.enc_col_ratio = result.final_profile.copy()
            self.enc_use_optimized_lut.setEnabled(True)
            self.enc_use_optimized_lut.setChecked(True)
        else:
            self.enc_use_optimized_lut.setChecked(False)
            self.enc_use_optimized_lut.setEnabled(False)
        self.edge_gain_bar.setValue(100)
        verdict = "accepted" if result.accepted else "saved, acceptance checks failed"
        self.edge_gain_status.setText(
            f"Complete ({verdict}) · results saved in {result.run_dir}"
        )
        self._edge_log(
            "OSA optimisation complete. Final intensity ratios: "
            + np.array2string(result.final_l, precision=5)
        )
        for issue in result.acceptance_issues:
            self._edge_log(f"Acceptance: {issue}")
        if not result.accepted:
            self._edge_log(
                "The failed candidate was saved for inspection but was not applied "
                "to the active intensity profile."
            )

    def _edge_optimization_error(self, _error: str) -> None:
        self.edge_gain_stop_event = None
        self._edge_gain_running(False)
        self.edge_gain_bar.setRange(0, 100)
        self.edge_gain_bar.setValue(0)
        self.edge_gain_status.setText("OSA optimisation failed (see Status log).")

    def _edge_load_optimization_result(self) -> None:
        layout = self.encoding_layout
        if layout is None:
            self._edge_log("Build the matching 15 px channel layout first.")
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load optimized result",
            str(Path("data/osa_optimization").resolve()),
            "Optimization result (final_result.json);;JSON files (*.json)",
        )
        if not path:
            return
        try:
            result = load_optimization_result(path)
        except Exception as exc:
            self._edge_log(f"Could not load optimization result: {exc}")
            return
        if result.final_profile.shape != (layout.channel_width_px,):
            self._edge_log(
                "Loaded profile width does not match the current channel layout."
            )
            return
        self._edge_optimization_result = result
        if not result.accepted:
            self.enc_use_optimized_lut.setChecked(False)
            self.enc_use_optimized_lut.setEnabled(False)
            self._edge_log(
                "Loaded result failed its acceptance checks and was not applied."
            )
            for issue in result.acceptance_issues:
                self._edge_log(f"Acceptance: {issue}")
            return
        for spin, value in zip(self._edge_spins, result.final_profile):
            spin.setValue(float(value))
        self.enc_col_ratio = result.final_profile.copy()
        self.enc_use_optimized_lut.setEnabled(True)
        self.enc_use_optimized_lut.setChecked(True)
        self._edge_log(f"Loaded accepted optimization result from {result.run_dir}")

    # --- A/B encoding gain: flat baseline vs current taper -------------

    def _edge_gain_running(self, running: bool) -> None:
        self.edge_gain_button.setEnabled(not running)
        self.edge_gain_stop_button.setEnabled(running)
        self.edge_osa_button.setEnabled(not running)
        self.edge_load_optimization_button.setEnabled(not running)
        self.edge_gain_save_button.setEnabled(not running and self._edge_gain is not None)

    def _edge_measure_gain(self) -> None:
        layout = self.encoding_layout
        if layout is None:
            self._edge_log("Build a layout on the TPA Encoding page first.")
            return
        osa = self._osa_ready()
        if osa is None:
            self._edge_log("Connect the OSA (Connections page) first.")
            return
        controller = self._controller()
        if not getattr(controller, "is_open", False):
            self._edge_log("Open the SLM (Connections page) first.")
            return
        ratio = self._edge_get_ratio()
        if ratio is None or np.allclose(ratio, 1.0):
            self._edge_log(
                "Edge profile is flat (all 1.0) — the gain would be ~0. Set a "
                "taper (e.g. Cosine taper…) before measuring."
            )
            return

        settings = self._ana_settings()
        averages = self.ana_averages.value()
        stride = self.ana_stride.value()
        subtract_bg = self.ana_bg_check.isChecked()
        n_targets = 2 * len(range(0, layout.n_channels, max(1, stride)))
        total = 2 * n_targets  # two sweeps (baseline + taper)

        self.edge_gain_bar.setMaximum(total)
        self.edge_gain_bar.setValue(0)
        self.edge_gain_status.setText("Starting baseline (flat) sweep…")
        self._edge_gain_running(True)
        stop_event = threading.Event()
        self.edge_gain_stop_event = stop_event

        def make_cb(offset: int, phase: str):
            def cb(progress: AnalysisProgress) -> None:
                self.edge_gain_progress.emit(
                    offset + progress.step + 1, total, f"{phase} {progress.message}"
                )
            return cb

        def work() -> dict[str, Any]:
            try:
                baseline = measure_channel_spectra(
                    osa, controller, layout, settings,
                    averages=averages, stride=stride, subtract_background=subtract_bg,
                    stop_event=stop_event, progress_callback=make_cb(0, "[flat]"),
                    col_ratio=None,
                )
                tuned = measure_channel_spectra(
                    osa, controller, layout, settings,
                    averages=averages, stride=stride, subtract_background=subtract_bg,
                    stop_event=stop_event, progress_callback=make_cb(n_targets, "[taper]"),
                    col_ratio=ratio,
                )
            except AnalysisAborted:
                return {"status": "aborted"}
            return {"status": "ok", "baseline": baseline, "tuned": tuned}

        self._run_slm_task("Encoding gain (A/B)", work,
                           self._edge_gain_finished, self._edge_gain_error)

    def _edge_gain_stop(self) -> None:
        if self.edge_gain_stop_event is not None:
            self.edge_gain_stop_event.set()
            self.edge_gain_status.setText("Stopping…")

    def _edge_gain_progress(self, done: int, total: int, message: str) -> None:
        self.edge_gain_bar.setValue(done)
        self.edge_gain_status.setText(f"[{done}/{total}] {message}")

    def _edge_gain_finished(self, payload: dict[str, Any]) -> None:
        self.edge_gain_stop_event = None
        self._edge_gain_running(False)
        if payload.get("status") == "aborted":
            self.edge_gain_status.setText("Stopped — partial sweep discarded.")
            self._edge_log("Gain measurement stopped.")
            return
        gain = encoding_gain(payload["baseline"], payload["tuned"])
        self._edge_gain = gain
        self._edge_gain_populate(gain)

    def _edge_gain_error(self, _error: str) -> None:
        self.edge_gain_stop_event = None
        self._edge_gain_running(False)
        self.edge_gain_status.setText("Gain measurement failed (see Status log).")

    def _edge_gain_populate(self, gain: EncodingGain) -> None:
        self.edge_gain_table.setRowCount(gain.n)
        for row, c in enumerate(gain.channels):
            cells = [
                f"{c.side}{c.index}",
                f"{c.nominal_wl_nm:.3f}",
                f"{c.d_leak * 100:+.2f}",
                f"{c.d_in_band * 100:+.2f}",
                f"{c.loss_window * 100:.2f}",
                f"{c.loss_total * 100:.2f}",
            ]
            for col, text in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(text)
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.edge_gain_table.setItem(row, col, item)
        summary = (
            f"{gain.n} channels  ·  leak {gain.mean_leak_before * 100:.2f}% → "
            f"{gain.mean_leak_after * 100:.2f}% (Δ {gain.mean_d_leak * 100:+.2f} pp)  ·  "
            f"in-band {gain.mean_in_band_before * 100:.1f}% → "
            f"{gain.mean_in_band_after * 100:.1f}% (Δ {gain.mean_d_in_band * 100:+.2f} pp)  ·  "
            f"loss: window {gain.mean_loss_window * 100:.1f}%, "
            f"total {gain.mean_loss_total * 100:.1f}%"
        )
        self.edge_gain_status.setText(summary)
        self.edge_gain_save_button.setEnabled(gain.n > 0)
        verdict = "improves" if gain.mean_d_leak < 0 else "does not reduce"
        self._edge_log(f"Gain: taper {verdict} crosstalk — {summary}")

    def _edge_gain_save(self) -> None:
        if self._edge_gain is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save encoding gain", "encoding_gain.csv", "CSV (*.csv)")
        if not path:
            return
        try:
            write_gain_csv(self._edge_gain, path)
            self._edge_log(f"Saved gain table → {path}")
        except Exception as exc:
            self._edge_log(f"Save failed: {exc}")

    # ==================================================================
    # Modulation Error Analysis page (B1: single-channel spectral shape)
    # ==================================================================

    def _build_analysis_page(self) -> QtWidgets.QWidget:
        page = self._page_shell("Modulation Error Analysis")
        subtitle = QtWidgets.QLabel(
            "Turn on each channel of the encoder grid in isolation, sweep the "
            "OSA across the spectrum, and quantify each single-channel lineshape "
            "vs an ideal rectangular passband."
        )
        subtitle.setObjectName("PageSubtitle")
        subtitle.setWordWrap(True)
        page.layout().addWidget(subtitle)

        # --- OSA measurement settings ---
        cfg = self._panel("OSA Sweep Settings")
        grid = QtWidgets.QGridLayout(cfg)
        # OSA re-centres on each channel; only the (narrow) span is set here
        self.ana_span = QtWidgets.QLineEdit("0.8nm")
        self.ana_span.setToolTip("OSA span, re-centred on each channel's wavelength")
        self.ana_sensitivity = QtWidgets.QComboBox()
        self.ana_sensitivity.addItems(["NORM", "MID", "HIGH1", "HIGH2", "HIGH3"])
        self.ana_sensitivity.setCurrentText("HIGH3")
        self.ana_ref_level = QtWidgets.QLineEdit("10uW")
        self.ana_yunit = QtWidgets.QComboBox()
        self.ana_yunit.addItems(["LOG (dBm)", "LIN (W)"])
        self.ana_yunit.setToolTip(
            "OSA acquisition Y unit. LOG resolves weak crosstalk tails far below "
            "the peak; LIN compresses them near the noise floor. Saved data is "
            "always converted to watts."
        )
        self.ana_averages = self._spin(1, 20, 1)
        self.ana_stride = self._spin(1, 64, 1)
        self.ana_stride.setToolTip("Measure only every Nth channel per side (1 = all)")
        self.ana_bg_check = QtWidgets.QCheckBox("Subtract background")
        self.ana_bg_check.setChecked(True)
        self.ana_bg_check.setToolTip(
            "Take an all-off trace at each channel's centre and subtract it "
            "(2x sweeps, cleaner low-level crosstalk floor)"
        )
        grid.addWidget(QtWidgets.QLabel("Span / channel"), 0, 0)
        grid.addWidget(self.ana_span, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Sensitivity"), 0, 2)
        grid.addWidget(self.ana_sensitivity, 0, 3)
        grid.addWidget(QtWidgets.QLabel("Ref level"), 0, 4)
        grid.addWidget(self.ana_ref_level, 0, 5)
        grid.addWidget(QtWidgets.QLabel("Y unit"), 1, 0)
        grid.addWidget(self.ana_yunit, 1, 1)
        grid.addWidget(QtWidgets.QLabel("Averages"), 1, 2)
        grid.addWidget(self.ana_averages, 1, 3)
        grid.addWidget(QtWidgets.QLabel("Stride"), 1, 4)
        grid.addWidget(self.ana_stride, 1, 5)
        grid.addWidget(self.ana_bg_check, 2, 0, 1, 3)
        page.layout().addWidget(cfg)

        self.ana_layout_label = QtWidgets.QLabel("Grid: (build a layout on the TPA Encoding page)")
        self.ana_layout_label.setObjectName("PageSubtitle")
        self.ana_layout_label.setWordWrap(True)
        page.layout().addWidget(self.ana_layout_label)

        # --- results: table + plots ---
        self.ana_table = QtWidgets.QTableWidget(0, 8)
        self.ana_table.setHorizontalHeaderLabels(
            ["Ch", "λ (nm)", "Peak λ", "FWHM (nm)",
             "Window (W)", "Channel (W)", "In-band %", "Leak %"]
        )
        self.ana_table.verticalHeader().setVisible(False)
        self.ana_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.ana_table.setAlternatingRowColors(True)
        self.ana_table.itemSelectionChanged.connect(self._ana_on_row_selected)

        self.ana_spectra_fig = Figure(figsize=(6, 3), tight_layout=True)
        self.ana_spectra_canvas = FigureCanvas(self.ana_spectra_fig)
        self.ana_metrics_fig = Figure(figsize=(6, 3), tight_layout=True)
        self.ana_metrics_canvas = FigureCanvas(self.ana_metrics_fig)

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._panel_with_widget("Spectra", self.ana_spectra_canvas), "Spectra")
        tabs.addTab(self._panel_with_widget("Metrics vs λ", self.ana_metrics_canvas), "Metrics vs λ")

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.addWidget(self._panel_with_widget("Per-channel metrics", self.ana_table))
        split.addWidget(tabs)
        split.setSizes([430, 650])
        page.layout().addWidget(split, 1)

        # --- controls ---
        self.ana_progress_bar = QtWidgets.QProgressBar()
        self.ana_progress_bar.setValue(0)
        self.ana_status = QtWidgets.QLabel("\N{EN DASH}")
        self.ana_run_button = QtWidgets.QPushButton("Run Analysis")
        self.ana_run_button.clicked.connect(self._ana_run)
        self.ana_stop_button = QtWidgets.QPushButton("Stop")
        self.ana_stop_button.setProperty("variant", "danger")
        self.ana_stop_button.setEnabled(False)
        self.ana_stop_button.clicked.connect(self._ana_stop)
        self.ana_save_button = QtWidgets.QPushButton("Save CSV…")
        self.ana_save_button.setProperty("variant", "ghost")
        self.ana_save_button.setEnabled(False)
        self.ana_save_button.clicked.connect(self._ana_save)
        ctrl = QtWidgets.QHBoxLayout()
        ctrl.addWidget(self.ana_status, 1)
        ctrl.addWidget(self.ana_save_button)
        ctrl.addWidget(self.ana_run_button)
        ctrl.addWidget(self.ana_stop_button)
        page.layout().addWidget(self.ana_progress_bar)
        page.layout().addLayout(ctrl)

        self._ana_live_wl: list[float] = []
        self._ana_live_metric: list[float] = []
        return page

    def _ana_settings(self) -> MeasurementSettings:
        # center_wl is a placeholder; measure_channel_spectra re-centres per channel
        y_unit = "LOGarithmic" if self.ana_yunit.currentText().startswith("LOG") else "LINear"
        return MeasurementSettings(
            center_wl="778nm",
            span=self.ana_span.text().strip() or "0.8nm",
            sensitivity=self.ana_sensitivity.currentText(),
            reference_level=self.ana_ref_level.text().strip() or "10uW",
            y_unit=y_unit,
        )

    def _ana_set_running(self, running: bool) -> None:
        self.ana_run_button.setEnabled(not running)
        self.ana_stop_button.setEnabled(running)
        self.ana_save_button.setEnabled(not running and self.analysis_result is not None)

    def _ana_run(self) -> None:
        layout = self.encoding_layout
        if layout is None:
            self.ana_status.setText("No channel grid — open the TPA Encoding page to build one.")
            return
        osa = self._osa_ready()
        if osa is None:
            self.ana_status.setText("Connect the OSA on the Calibration page first.")
            return
        controller = self._controller()
        if not getattr(controller, "is_open", False):
            self.ana_status.setText("Open the SLM on the SLM Control page first.")
            return

        settings = self._ana_settings()
        averages = self.ana_averages.value()
        stride = self.ana_stride.value()
        subtract_bg = self.ana_bg_check.isChecked()
        n_targets = 2 * len(range(0, layout.n_channels, max(1, stride)))
        capture_dir = tempfile.mkdtemp(prefix="mod_err_")
        self._ana_capture_dir = capture_dir

        self.ana_layout_label.setText(
            f"Grid: {layout.n_channels} ch/side, width {layout.channel_width_px} px "
            f"({layout.channel_width_px * layout.nm_per_px:.4f} nm), centre "
            f"{layout.center_wl:.2f} nm  ·  measuring {n_targets} channels"
        )
        self._ana_live_wl = []
        self._ana_live_metric = []
        self.ana_progress_bar.setMaximum(n_targets)
        self.ana_progress_bar.setValue(0)
        self.ana_status.setText("Starting…")
        self._ana_set_running(True)

        stop_event = threading.Event()
        self.analysis_stop_event = stop_event

        def report(progress: AnalysisProgress) -> None:
            self.analysis_progress.emit(progress)

        def work() -> dict[str, Any]:
            try:
                result = measure_channel_spectra(
                    osa, controller, layout, settings,
                    averages=averages, stride=stride,
                    subtract_background=subtract_bg, capture_dir=capture_dir,
                    stop_event=stop_event, progress_callback=report,
                )
            except AnalysisAborted:
                return {"status": "aborted"}
            return {"status": "ok", "result": result}

        self._run_slm_task("Modulation error analysis", work,
                           self._ana_finished, self._ana_error)

    def _ana_stop(self) -> None:
        if self.analysis_stop_event is not None:
            self.analysis_stop_event.set()
            self.ana_status.setText("Stopping…")

    def _on_analysis_progress(self, progress: AnalysisProgress) -> None:
        done = min(progress.step + 1, progress.total)
        self.ana_progress_bar.setMaximum(max(progress.total, 1))
        self.ana_progress_bar.setValue(done)
        self.ana_status.setText(progress.message)
        if progress.wl is not None and progress.metric is not None:
            self._ana_live_wl.append(progress.wl)
            self._ana_live_metric.append(progress.metric)
            self._ana_draw_live()

    def _ana_draw_live(self) -> None:
        self.ana_metrics_fig.clear()
        ax = self.ana_metrics_fig.add_subplot(111)
        self._style_dark_axes(ax)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("In-band fraction")
        ax.set_ylim(0, 1.02)
        ax.scatter(self._ana_live_wl, self._ana_live_metric, s=12, color="#47b8e0")
        self.ana_metrics_fig.patch.set_facecolor("#101820")
        self.ana_metrics_canvas.draw_idle()

    def _ana_finished(self, payload: dict[str, Any]) -> None:
        self.analysis_stop_event = None
        self._ana_set_running(False)
        if payload.get("status") == "aborted":
            self.ana_status.setText(
                f"Analysis stopped · partial captures in {self._ana_capture_dir}"
            )
            return
        result = payload["result"]
        self.analysis_result = result
        self.ana_save_button.setEnabled(True)
        self._ana_populate_table(result)
        self._ana_draw_spectra(result)
        self._ana_draw_metrics(result)
        n = len(result.channels)
        mean_inband = float(np.mean([c.in_band_fraction for c in result.channels])) if n else 0.0
        mean_leak = float(np.mean([c.neighbor_leakage for c in result.channels])) if n else 0.0
        npz = result.raw_npz_path or "(none)"
        self.ana_status.setText(
            f"Done · {n} channels · mean in-band {mean_inband*100:.1f}% · "
            f"mean leak {mean_leak*100:.1f}% · raw NPZ: {npz}"
        )

    def _ana_error(self, _error: str) -> None:
        self.analysis_stop_event = None
        self._ana_set_running(False)
        self.ana_status.setText("Analysis failed (see Status log)")

    def _ana_populate_table(self, result: ModulationErrorResult) -> None:
        self.ana_table.setRowCount(len(result.channels))
        for r, ch in enumerate(result.channels):
            cells = [
                f"{ch.side}[{ch.index}]",
                f"{ch.nominal_wl_nm:.4f}",
                f"{ch.peak_wl_nm:.4f}",
                f"{ch.fwhm_nm:.4f}",
                f"{ch.window_power_w:.3e}",
                f"{ch.channel_power_w:.3e}",
                f"{ch.in_band_fraction*100:.1f}",
                f"{ch.neighbor_leakage*100:.1f}",
            ]
            for c, text in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(text)
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.ana_table.setItem(r, c, item)
        self.ana_table.resizeColumnsToContents()

    def _ana_draw_spectra(self, result: ModulationErrorResult, highlight: int | None = None) -> None:
        self.ana_spectra_fig.clear()
        ax = self.ana_spectra_fig.add_subplot(111)
        self._style_dark_axes(ax)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Power (W)")
        for i, ch in enumerate(result.channels):
            if ch.wavelengths_nm.size == 0:
                continue
            if highlight is not None and i != highlight:
                ax.plot(ch.wavelengths_nm, ch.signal_w, color="#3a4a54", linewidth=0.6)
        for i, ch in enumerate(result.channels):
            if ch.wavelengths_nm.size == 0:
                continue
            if highlight is None:
                ax.plot(ch.wavelengths_nm, ch.signal_w, linewidth=0.8)
            elif i == highlight:
                ax.plot(ch.wavelengths_nm, ch.signal_w, color="#47b8e0", linewidth=1.4)
                half = ch.nominal_bw_nm / 2.0
                ax.axvspan(ch.nominal_wl_nm - half, ch.nominal_wl_nm + half,
                           color="#47b8e0", alpha=0.15)
        self.ana_spectra_fig.patch.set_facecolor("#101820")
        self.ana_spectra_canvas.draw_idle()

    def _ana_draw_metrics(self, result: ModulationErrorResult) -> None:
        self.ana_metrics_fig.clear()
        ax = self.ana_metrics_fig.add_subplot(111)
        self._style_dark_axes(ax)
        wl = [c.nominal_wl_nm for c in result.channels]
        inband = [c.in_band_fraction for c in result.channels]
        leak = [c.neighbor_leakage for c in result.channels]
        ax.scatter(wl, inband, s=14, color="#47b8e0", label="in-band fraction")
        ax.scatter(wl, leak, s=14, color="#e0735a", label="neighbour leakage")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Fraction")
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=7, facecolor="#101820", edgecolor="#41515c", labelcolor="#d8dee9")
        self.ana_metrics_fig.patch.set_facecolor("#101820")
        self.ana_metrics_canvas.draw_idle()

    def _ana_on_row_selected(self) -> None:
        if self.analysis_result is None:
            return
        rows = self.ana_table.selectionModel().selectedRows()
        if not rows:
            return
        self._ana_draw_spectra(self.analysis_result, highlight=rows[0].row())

    def _ana_save(self) -> None:
        if self.analysis_result is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Analysis CSV", "modulation_error.csv", "CSV Files (*.csv)"
        )
        if not path:
            return
        out = write_analysis_csv(self.analysis_result, path)
        # copy the consolidated raw NPZ next to the metrics CSV
        npz_src = self.analysis_result.raw_npz_path
        msg = f"Saved {out}"
        if npz_src and Path(npz_src).is_file():
            npz_dst = Path(path).with_suffix(".npz")
            try:
                shutil.copyfile(npz_src, npz_dst)
                msg += f"  +  raw spectra {npz_dst}"
            except OSError as exc:
                msg += f"  (raw NPZ copy failed: {exc})"
        self.ana_status.setText(msg)

    # ==================================================================
    # Background scatter calibration (Step 6): dark + per-channel 780 scatter
    # ==================================================================

    def _build_background_tab(self) -> QtWidgets.QWidget:
        page = self._page_shell("Background Scatter Calibration")
        subtitle = QtWidgets.QLabel(
            "Automated send-pattern → record-scope-mean sweep. First measures the "
            "dark/baseline with all SLM channels off, then turns on each channel "
            "in isolation and sweeps its level to build a linear 780-scatter curve "
            "per channel. Save the result to subtract this background from the 420 "
            "TPA signal later. Build a channel grid on the TPA Encoding page first."
        )
        subtitle.setObjectName("PageSubtitle")
        subtitle.setWordWrap(True)
        page.layout().addWidget(subtitle)

        # --- sweep settings ---
        cfg = self._panel("Sweep & Scope Settings")
        grid = QtWidgets.QGridLayout(cfg)
        self.bg_channel = QtWidgets.QComboBox()
        self.bg_channel.addItems(["1", "2", "3", "4"])
        self.bg_duration = QtWidgets.QDoubleSpinBox()
        self.bg_duration.setRange(0.001, 10.0); self.bg_duration.setDecimals(3)
        self.bg_duration.setValue(1.0); self.bg_duration.setSuffix(" s")
        self.bg_duration.setToolTip("Scope MEAN averaging window per reading")
        self.bg_hold = QtWidgets.QDoubleSpinBox()
        self.bg_hold.setRange(0.0, 10000.0); self.bg_hold.setValue(150.0)
        self.bg_hold.setSuffix(" ms")
        self.bg_hold.setToolTip("Settle time after each SLM pattern change")
        self.bg_levels = self._spin(2, 21, 5)
        self.bg_levels.setToolTip("Number of levels swept 0→1 per channel (incl. 0)")
        self.bg_dark_samples = self._spin(1, 100, 5)
        self.bg_dark_samples.setToolTip("Repeated dark/baseline readings to average")
        self.bg_repeats = self._spin(1, 20, 1)
        self.bg_repeats.setToolTip("Repeated scope readings averaged per level point")
        self.bg_stride = self._spin(1, 64, 1)
        self.bg_stride.setToolTip("Measure only every Nth channel per side (1 = all)")
        self.bg_decimation = QtWidgets.QComboBox()
        self.bg_decimation.addItems(["HRESolution", "SAMPle"])
        pairs = [
            ("Channel", self.bg_channel), ("Average for", self.bg_duration),
            ("Hold", self.bg_hold), ("Levels", self.bg_levels),
            ("Dark samples", self.bg_dark_samples), ("Repeats", self.bg_repeats),
            ("Stride", self.bg_stride), ("Decimation", self.bg_decimation),
        ]
        for i, (label, widget) in enumerate(pairs):
            r, c = i // 2, (i % 2) * 2
            grid.addWidget(QtWidgets.QLabel(label), r, c)
            grid.addWidget(widget, r, c + 1)
        page.layout().addWidget(cfg)

        self.bg_layout_label = QtWidgets.QLabel(
            "Grid: (build a layout on the TPA Encoding page)"
        )
        self.bg_layout_label.setObjectName("PageSubtitle")
        self.bg_layout_label.setWordWrap(True)
        page.layout().addWidget(self.bg_layout_label)

        # --- results: fitted-lines plot (left) + raw scatter plot (right) ---
        # left: one line per channel, x = normalised intensity, dashed dark line
        self.bg_fits_fig = Figure(figsize=(5, 3.4), tight_layout=True)
        self.bg_fits_canvas = FigureCanvas(self.bg_fits_fig)
        # right: raw scatter (wavelength vs scatter above dark), incl. big jumps
        self.bg_fig = Figure(figsize=(5, 3.4), tight_layout=True)
        self.bg_canvas = FigureCanvas(self.bg_fig)

        plot_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        plot_split.addWidget(self._panel_with_widget("Per-channel fits", self.bg_fits_canvas))
        plot_split.addWidget(self._panel_with_widget("Raw scatter", self.bg_canvas))
        plot_split.setSizes([560, 520])
        page.layout().addWidget(plot_split, 1)

        # --- controls ---
        self.bg_progress_bar = QtWidgets.QProgressBar()
        self.bg_progress_bar.setValue(0)
        self.bg_status = QtWidgets.QLabel("\N{EN DASH}")
        self.bg_run_button = QtWidgets.QPushButton("Run Sweep")
        self.bg_run_button.clicked.connect(self._bg_run)
        self.bg_stop_button = QtWidgets.QPushButton("Stop")
        self.bg_stop_button.setProperty("variant", "danger")
        self.bg_stop_button.setEnabled(False)
        self.bg_stop_button.clicked.connect(self._bg_stop)
        self.bg_save_button = QtWidgets.QPushButton("Save…")
        self.bg_save_button.setProperty("variant", "ghost")
        self.bg_save_button.setEnabled(False)
        self.bg_save_button.clicked.connect(self._bg_save)
        self.bg_load_button = QtWidgets.QPushButton("Load…")
        self.bg_load_button.setProperty("variant", "ghost")
        self.bg_load_button.setToolTip(
            "Load a saved background NPZ; big jumps are re-eliminated and each "
            "channel re-fit before plotting."
        )
        self.bg_load_button.clicked.connect(self._bg_load)
        ctrl = QtWidgets.QHBoxLayout()
        ctrl.addWidget(self.bg_status, 1)
        ctrl.addWidget(self.bg_load_button)
        ctrl.addWidget(self.bg_save_button)
        ctrl.addWidget(self.bg_run_button)
        ctrl.addWidget(self.bg_stop_button)
        page.layout().addWidget(self.bg_progress_bar)
        page.layout().addLayout(ctrl)
        return page

    def _bg_monitor_settings(self) -> MonitorSettings:
        # free-run AUTO software read (same recipe as the encoder auto-read)
        return MonitorSettings(
            channel=int(self.bg_channel.currentText()),
            trigger_mode="AUTO",
            hold=self.bg_hold.value() / 1000.0,     # ms -> s
            duration=self.bg_duration.value(),
            decimation=self.bg_decimation.currentText(),
        )

    def _bg_set_running(self, running: bool) -> None:
        self.bg_run_button.setEnabled(not running)
        self.bg_stop_button.setEnabled(running)
        self.bg_load_button.setEnabled(not running)
        self.bg_save_button.setEnabled(
            not running and self.background_result is not None
        )

    def _bg_run(self) -> None:
        layout = self.encoding_layout
        if layout is None:
            self.bg_status.setText(
                "No channel grid — open the TPA Encoding page to build one."
            )
            return
        scope = self.scope_controller
        if scope is None or not scope.is_connected:
            self.bg_status.setText("Connect the scope on the Scope page first.")
            return
        controller = self._controller()
        if not getattr(controller, "is_open", False):
            self.bg_status.setText("Open the SLM on the SLM Control page first.")
            return

        settings = self._bg_monitor_settings()
        n_levels = self.bg_levels.value()
        levels = np.linspace(0.0, 1.0, n_levels)
        dark_samples = self.bg_dark_samples.value()
        repeats = self.bg_repeats.value()
        stride = self.bg_stride.value()
        capture_dir = tempfile.mkdtemp(prefix="bg_scatter_")
        self._bg_capture_dir = capture_dir

        n_targets = 2 * len(range(0, layout.n_channels, max(1, stride)))
        self.bg_layout_label.setText(
            f"Grid: {layout.n_channels} ch/side, width {layout.channel_width_px} px, "
            f"centre {layout.center_wl:.2f} nm  ·  {n_targets} channels × "
            f"{n_levels} levels + dark"
        )
        self._bg_live = []
        self.bg_progress_bar.setMaximum(n_targets * n_levels + 1)
        self.bg_progress_bar.setValue(0)
        self.bg_status.setText("Starting…")
        self._bg_set_running(True)

        stop_event = threading.Event()
        self.background_stop_event = stop_event

        def report(progress: BackgroundProgress) -> None:
            self.background_progress.emit(progress)

        def work() -> dict[str, Any]:
            try:
                result = measure_scatter_background(
                    scope, controller, layout, settings,
                    levels=levels, dark_samples=dark_samples, repeats=repeats,
                    stride=stride, capture_dir=capture_dir,
                    stop_event=stop_event, progress_callback=report,
                )
            except BackgroundAborted:
                return {"status": "aborted"}
            return {"status": "ok", "result": result}

        self._run_slm_task("Background scatter sweep", work,
                           self._bg_finished, self._bg_error)

    def _bg_stop(self) -> None:
        if self.background_stop_event is not None:
            self.background_stop_event.set()
            self.bg_status.setText("Stopping…")

    def _on_background_progress(self, progress: BackgroundProgress) -> None:
        self.bg_progress_bar.setMaximum(max(progress.total, 1))
        self.bg_progress_bar.setValue(min(progress.step + 1, progress.total))
        self.bg_status.setText(progress.message)
        if progress.wl is not None and progress.metric is not None:
            self._bg_live.append((progress.wl, progress.metric))
            self._bg_draw_live()

    def _bg_draw_live(self) -> None:
        self.bg_fig.clear()
        self.bg_fig.patch.set_facecolor("#101820")
        ax = self.bg_fig.add_subplot(111)
        self._style_dark_axes(ax)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Scatter above dark (mV)")
        if self._bg_live:
            wl = [p[0] for p in self._bg_live]
            mv = [p[1] * 1000 for p in self._bg_live]
            ax.scatter(wl, mv, s=12, color="#47b8e0")
        self.bg_canvas.draw_idle()

    def _bg_finished(self, payload: dict[str, Any]) -> None:
        self.background_stop_event = None
        self._bg_set_running(False)
        if payload.get("status") == "aborted":
            self.bg_status.setText(
                f"Sweep stopped · partial captures in {self._bg_capture_dir}"
            )
            return
        result = payload["result"]
        self.background_result = result
        self.bg_save_button.setEnabled(True)
        self._bg_show_result(result)
        n_out = sum(
            int(np.size(c.inlier_mask) - np.count_nonzero(c.inlier_mask))
            for c in result.channels if c.inlier_mask is not None
        )
        self.bg_status.setText(
            f"Done · dark {result.dark_mean_v*1000:.4f} mV "
            f"(±{result.dark_std_v*1000:.4f}) · {len(result.channels)} channels · "
            f"{n_out} big-jump point(s) removed · raw NPZ: {result.raw_npz_path or '(none)'}"
        )

    def _bg_error(self, _error: str) -> None:
        self.background_stop_event = None
        self._bg_set_running(False)
        self.bg_status.setText("Background sweep failed (see Status log)")

    def _bg_show_result(self, result: BackgroundResult) -> None:
        """Refresh both plots from a finished/loaded result."""
        self._bg_draw_fits(result)
        self._bg_draw_raw(result)

    def _bg_draw_fits(self, result: BackgroundResult) -> None:
        """Left plot: one linear-fit line per channel vs normalised intensity."""
        self.bg_fits_fig.clear()
        self.bg_fits_fig.patch.set_facecolor("#101820")
        ax = self.bg_fits_fig.add_subplot(111)
        self._style_dark_axes(ax)
        ax.set_xlabel("Normalised channel intensity")
        ax.set_ylabel("Scope mean (mV)")
        xs = np.array([0.0, 1.0])
        for ch in result.channels:
            ys = (ch.slope_v * xs + ch.intercept_v) * 1000.0
            ax.plot(xs, ys, linewidth=0.7)
        ax.axhline(result.dark_mean_v * 1000, color="#e0a447", ls="--",
                   linewidth=1.2, label=f"dark {result.dark_mean_v*1000:.3f} mV")
        ax.legend(loc="upper left", fontsize=7)
        self.bg_fits_canvas.draw_idle()

    def _bg_draw_raw(self, result: BackgroundResult) -> None:
        """Right plot: raw scatter (λ vs scatter above dark); big jumps in red."""
        self.bg_fig.clear()
        self.bg_fig.patch.set_facecolor("#101820")
        ax = self.bg_fig.add_subplot(111)
        self._style_dark_axes(ax)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Scatter above dark (mV)")
        kept_wl, kept_y, out_wl, out_y = [], [], [], []
        for ch in result.channels:
            above = (ch.means_v - result.dark_mean_v) * 1000.0
            mask = ch.inlier_mask
            for k in range(ch.means_v.size):
                keep = True if mask is None else bool(mask[k])
                (kept_wl if keep else out_wl).append(ch.nominal_wl_nm)
                (kept_y if keep else out_y).append(above[k])
        ax.scatter(kept_wl, kept_y, s=12, color="#47b8e0", label="kept")
        if out_wl:
            ax.scatter(out_wl, out_y, s=28, color="#e05a5a", marker="x",
                       label="big jump (removed)")
        ax.axhline(0.0, color="#e0a447", ls="--", linewidth=1.0)
        ax.legend(loc="upper right", fontsize=7)
        self.bg_canvas.draw_idle()

    def _bg_load(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Background Result", "", "NumPy (*.npz)"
        )
        if not path:
            return
        try:
            result = load_background_result(path)
        except Exception as exc:
            self.bg_status.setText(f"Load failed: {exc}")
            return
        self.background_result = result
        self.bg_save_button.setEnabled(True)
        self._bg_show_result(result)
        self.bg_status.setText(
            f"Loaded {Path(path).name} · dark {result.dark_mean_v*1000:.4f} mV · "
            f"{len(result.channels)} channels (re-fit with big-jump removal)"
        )

    def _bg_save(self) -> None:
        if self.background_result is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Background Result", "bg_scatter.npz",
            "NumPy (*.npz);;JSON (*.json);;CSV (*.csv)"
        )
        if not path:
            return
        base = Path(path).with_suffix("")
        npz = save_background_npz(self.background_result, base.with_suffix(".npz"))
        js = save_background_json(self.background_result, base.with_suffix(".json"))
        csv_path = write_background_csv(self.background_result, base.with_suffix(".csv"))
        self.bg_status.setText(f"Saved {npz}  +  {Path(js).name}  +  {Path(csv_path).name}")

    # ===================== TPA efficiency tab ========================
    def _build_tpa_tab(self) -> QtWidgets.QWidget:
        page = self._page_shell("Channel TPA Efficiency Calibration")
        subtitle = QtWidgets.QLabel(
            "Automated per-channel-pair sweep. For each pair the commanded "
            "product x·w is swept 0→1 (driving x=w=√u), the 780 scatter + dark "
            "background is subtracted using the Step 6 result, and the remaining "
            "420 TPA signal is fit to a_i·(x·w)². Run Step 6 (BG Scatter) first "
            "or load a saved background NPZ below."
        )
        subtitle.setObjectName("PageSubtitle")
        subtitle.setWordWrap(True)
        page.layout().addWidget(subtitle)

        # --- sweep settings ---
        cfg = self._panel("Sweep & Scope Settings")
        grid = QtWidgets.QGridLayout(cfg)
        self.tpa_channel = QtWidgets.QComboBox()
        self.tpa_channel.addItems(["1", "2", "3", "4"])
        self.tpa_duration = QtWidgets.QDoubleSpinBox()
        self.tpa_duration.setRange(0.001, 10.0); self.tpa_duration.setDecimals(3)
        self.tpa_duration.setValue(1.0); self.tpa_duration.setSuffix(" s")
        self.tpa_duration.setToolTip("Scope MEAN averaging window per reading")
        self.tpa_hold = QtWidgets.QDoubleSpinBox()
        self.tpa_hold.setRange(0.0, 10000.0); self.tpa_hold.setValue(150.0)
        self.tpa_hold.setSuffix(" ms")
        self.tpa_hold.setToolTip("Settle time after each SLM pattern change")
        self.tpa_levels = self._spin(2, 21, 5)
        self.tpa_levels.setToolTip("Number of product levels swept 0→1 per pair (incl. 0)")
        self.tpa_repeats = self._spin(1, 20, 1)
        self.tpa_repeats.setToolTip("Repeated scope readings averaged per product point")
        self.tpa_stride = self._spin(1, 64, 1)
        self.tpa_stride.setToolTip("Measure only every Nth pair (1 = all)")
        self.tpa_decimation = QtWidgets.QComboBox()
        self.tpa_decimation.addItems(["HRESolution", "SAMPle"])
        pairs = [
            ("Channel", self.tpa_channel), ("Average for", self.tpa_duration),
            ("Hold", self.tpa_hold), ("Product levels", self.tpa_levels),
            ("Repeats", self.tpa_repeats), ("Stride", self.tpa_stride),
            ("Decimation", self.tpa_decimation),
        ]
        for i, (label, widget) in enumerate(pairs):
            r, c = i // 2, (i % 2) * 2
            grid.addWidget(QtWidgets.QLabel(label), r, c)
            grid.addWidget(widget, r, c + 1)
        page.layout().addWidget(cfg)

        self.tpa_bg_label = QtWidgets.QLabel(
            "Background: (run Step 6 or load a background NPZ)"
        )
        self.tpa_bg_label.setObjectName("PageSubtitle")
        self.tpa_bg_label.setWordWrap(True)
        page.layout().addWidget(self.tpa_bg_label)

        # --- results: quadratic-fit plot (left) + raw TPA signal plot (right) ---
        self.tpa_fits_fig = Figure(figsize=(5, 3.4), tight_layout=True)
        self.tpa_fits_canvas = FigureCanvas(self.tpa_fits_fig)
        self.tpa_fig = Figure(figsize=(5, 3.4), tight_layout=True)
        self.tpa_canvas = FigureCanvas(self.tpa_fig)

        plot_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        plot_split.addWidget(self._panel_with_widget("Per-pair fits", self.tpa_fits_canvas))
        plot_split.addWidget(self._panel_with_widget("Raw TPA signal", self.tpa_canvas))
        plot_split.setSizes([560, 520])
        page.layout().addWidget(plot_split, 1)

        # --- controls ---
        self.tpa_progress_bar = QtWidgets.QProgressBar()
        self.tpa_progress_bar.setValue(0)
        self.tpa_status = QtWidgets.QLabel("\N{EN DASH}")
        self.tpa_run_button = QtWidgets.QPushButton("Run Sweep")
        self.tpa_run_button.clicked.connect(self._tpa_run)
        self.tpa_stop_button = QtWidgets.QPushButton("Stop")
        self.tpa_stop_button.setProperty("variant", "danger")
        self.tpa_stop_button.setEnabled(False)
        self.tpa_stop_button.clicked.connect(self._tpa_stop)
        self.tpa_loadbg_button = QtWidgets.QPushButton("Load BG…")
        self.tpa_loadbg_button.setProperty("variant", "ghost")
        self.tpa_loadbg_button.setToolTip("Load a saved Step 6 background NPZ to subtract")
        self.tpa_loadbg_button.clicked.connect(self._tpa_load_bg)
        self.tpa_save_button = QtWidgets.QPushButton("Save…")
        self.tpa_save_button.setProperty("variant", "ghost")
        self.tpa_save_button.setEnabled(False)
        self.tpa_save_button.clicked.connect(self._tpa_save)
        self.tpa_load_button = QtWidgets.QPushButton("Load…")
        self.tpa_load_button.setProperty("variant", "ghost")
        self.tpa_load_button.setToolTip(
            "Load a saved TPA NPZ; big jumps are re-eliminated and each pair re-fit."
        )
        self.tpa_load_button.clicked.connect(self._tpa_load)
        ctrl = QtWidgets.QHBoxLayout()
        ctrl.addWidget(self.tpa_status, 1)
        ctrl.addWidget(self.tpa_loadbg_button)
        ctrl.addWidget(self.tpa_load_button)
        ctrl.addWidget(self.tpa_save_button)
        ctrl.addWidget(self.tpa_run_button)
        ctrl.addWidget(self.tpa_stop_button)
        page.layout().addWidget(self.tpa_progress_bar)
        page.layout().addLayout(ctrl)
        return page

    def _tpa_monitor_settings(self) -> MonitorSettings:
        return MonitorSettings(
            channel=int(self.tpa_channel.currentText()),
            trigger_mode="AUTO",
            hold=self.tpa_hold.value() / 1000.0,     # ms -> s
            duration=self.tpa_duration.value(),
            decimation=self.tpa_decimation.currentText(),
        )

    def _tpa_set_running(self, running: bool) -> None:
        self.tpa_run_button.setEnabled(not running)
        self.tpa_stop_button.setEnabled(running)
        self.tpa_load_button.setEnabled(not running)
        self.tpa_loadbg_button.setEnabled(not running)
        self.tpa_save_button.setEnabled(not running and self.tpa_result is not None)

    def _tpa_run(self) -> None:
        layout = self.encoding_layout
        if layout is None:
            self.tpa_status.setText(
                "No channel grid — open the TPA Encoding page to build one."
            )
            return
        if self.background_result is None:
            self.tpa_status.setText(
                "No background — run Step 6 (BG Scatter) or Load BG… first."
            )
            return
        scope = self.scope_controller
        if scope is None or not scope.is_connected:
            self.tpa_status.setText("Connect the scope on the Scope page first.")
            return
        controller = self._controller()
        if not getattr(controller, "is_open", False):
            self.tpa_status.setText("Open the SLM on the SLM Control page first.")
            return

        settings = self._tpa_monitor_settings()
        n_levels = self.tpa_levels.value()
        products = np.linspace(0.0, 1.0, n_levels)
        repeats = self.tpa_repeats.value()
        stride = self.tpa_stride.value()
        background = self.background_result
        capture_dir = tempfile.mkdtemp(prefix="tpa_eff_")
        self._tpa_capture_dir = capture_dir

        n_pairs = len(range(0, layout.n_channels, max(1, stride)))
        self.tpa_bg_label.setText(
            f"Grid: {layout.n_channels} pairs, centre {layout.center_wl:.2f} nm "
            f"· background dark {background.dark_mean_v*1000:.3f} mV, "
            f"{len(background.channels)} scatter channels  ·  {n_pairs} pairs × "
            f"{n_levels} products"
        )
        self._tpa_live = []
        self.tpa_progress_bar.setMaximum(max(n_pairs * n_levels, 1))
        self.tpa_progress_bar.setValue(0)
        self.tpa_status.setText("Starting…")
        self._tpa_set_running(True)

        stop_event = threading.Event()
        self.tpa_stop_event = stop_event

        def report(progress: TPAProgress) -> None:
            self.tpa_progress.emit(progress)

        def work() -> dict[str, Any]:
            try:
                result = measure_tpa_efficiency(
                    scope, controller, layout, settings, background,
                    products=products, repeats=repeats, stride=stride,
                    capture_dir=capture_dir,
                    stop_event=stop_event, progress_callback=report,
                )
            except TPAAborted:
                return {"status": "aborted"}
            return {"status": "ok", "result": result}

        self._run_slm_task("TPA efficiency sweep", work,
                           self._tpa_finished, self._tpa_error)

    def _tpa_stop(self) -> None:
        if self.tpa_stop_event is not None:
            self.tpa_stop_event.set()
            self.tpa_status.setText("Stopping…")

    def _on_tpa_progress(self, progress: TPAProgress) -> None:
        self.tpa_progress_bar.setMaximum(max(progress.total, 1))
        self.tpa_progress_bar.setValue(min(progress.step, progress.total))
        self.tpa_status.setText(progress.message)
        if progress.wl is not None and progress.metric is not None:
            self._tpa_live.append((progress.wl, progress.metric))
            self._tpa_draw_live()

    def _tpa_draw_live(self) -> None:
        self.tpa_fig.clear()
        self.tpa_fig.patch.set_facecolor("#101820")
        ax = self.tpa_fig.add_subplot(111)
        self._style_dark_axes(ax)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("TPA signal above background (mV)")
        if self._tpa_live:
            wl = [p[0] for p in self._tpa_live]
            mv = [p[1] * 1000 for p in self._tpa_live]
            ax.scatter(wl, mv, s=12, color="#8fd14f")
        self.tpa_canvas.draw_idle()

    def _tpa_finished(self, payload: dict[str, Any]) -> None:
        self.tpa_stop_event = None
        self._tpa_set_running(False)
        if payload.get("status") == "aborted":
            self.tpa_status.setText(
                f"Sweep stopped · partial captures in {self._tpa_capture_dir}"
            )
            return
        result = payload["result"]
        self.tpa_result = result
        self.tpa_save_button.setEnabled(True)
        self._tpa_show_result(result)
        n_out = sum(
            int(np.size(c.inlier_mask) - np.count_nonzero(c.inlier_mask))
            for c in result.channels if c.inlier_mask is not None
        )
        self.tpa_status.setText(
            f"Done · {len(result.channels)} pairs · {n_out} big-jump point(s) "
            f"removed · raw NPZ: {result.raw_npz_path or '(none)'}"
        )

    def _tpa_error(self, _error: str) -> None:
        self.tpa_stop_event = None
        self._tpa_set_running(False)
        self.tpa_status.setText("TPA sweep failed (see Status log)")

    def _tpa_show_result(self, result: TPAResult) -> None:
        """Refresh both plots from a finished/loaded result."""
        self._tpa_draw_fits(result)
        self._tpa_draw_raw(result)

    def _tpa_draw_fits(self, result: TPAResult) -> None:
        """Left plot: one quadratic-fit curve per pair vs commanded product x·w."""
        self.tpa_fits_fig.clear()
        self.tpa_fits_fig.patch.set_facecolor("#101820")
        ax = self.tpa_fits_fig.add_subplot(111)
        self._style_dark_axes(ax)
        ax.set_xlabel("Commanded product x·w")
        ax.set_ylabel("TPA signal (mV)")
        us = np.linspace(0.0, 1.0, 50)
        for ch in result.channels:
            ys = ch.coeff_a * us * us * 1000.0
            ax.plot(us, ys, linewidth=0.7)
        ax.axhline(0.0, color="#e0a447", ls="--", linewidth=1.0)
        self.tpa_fits_canvas.draw_idle()

    def _tpa_draw_raw(self, result: TPAResult) -> None:
        """Right plot: raw background-removed TPA signal (λ vs signal); jumps red."""
        self.tpa_fig.clear()
        self.tpa_fig.patch.set_facecolor("#101820")
        ax = self.tpa_fig.add_subplot(111)
        self._style_dark_axes(ax)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("TPA signal above background (mV)")
        kept_wl, kept_y, out_wl, out_y = [], [], [], []
        for ch in result.channels:
            sig = ch.signal_v * 1000.0
            mask = ch.inlier_mask
            for k in range(ch.signal_v.size):
                keep = True if mask is None else bool(mask[k])
                (kept_wl if keep else out_wl).append(ch.nominal_wl_nm)
                (kept_y if keep else out_y).append(sig[k])
        ax.scatter(kept_wl, kept_y, s=12, color="#8fd14f", label="kept")
        if out_wl:
            ax.scatter(out_wl, out_y, s=28, color="#e05a5a", marker="x",
                       label="big jump (removed)")
        ax.axhline(0.0, color="#e0a447", ls="--", linewidth=1.0)
        ax.legend(loc="upper right", fontsize=7)
        self.tpa_canvas.draw_idle()

    def _tpa_load_bg(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Background Result", "", "NumPy (*.npz)"
        )
        if not path:
            return
        try:
            result = load_background_result(path)
        except Exception as exc:
            self.tpa_status.setText(f"Background load failed: {exc}")
            return
        self.background_result = result
        self.tpa_bg_label.setText(
            f"Background: {Path(path).name} · dark {result.dark_mean_v*1000:.4f} mV · "
            f"{len(result.channels)} scatter channels"
        )
        self.tpa_status.setText("Background loaded — ready to run the TPA sweep.")

    def _tpa_load(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load TPA Result", "", "NumPy (*.npz)"
        )
        if not path:
            return
        try:
            result = load_tpa_result(path)
        except Exception as exc:
            self.tpa_status.setText(f"Load failed: {exc}")
            return
        self.tpa_result = result
        self.tpa_save_button.setEnabled(True)
        self._tpa_show_result(result)
        self.tpa_status.setText(
            f"Loaded {Path(path).name} · {len(result.channels)} pairs "
            f"(re-fit with big-jump removal)"
        )

    def _tpa_save(self) -> None:
        if self.tpa_result is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save TPA Result", "tpa_efficiency.npz",
            "NumPy (*.npz);;JSON (*.json);;CSV (*.csv)"
        )
        if not path:
            return
        base = Path(path).with_suffix("")
        npz = save_tpa_npz(self.tpa_result, base.with_suffix(".npz"))
        js = save_tpa_json(self.tpa_result, base.with_suffix(".json"))
        csv_path = write_tpa_csv(self.tpa_result, base.with_suffix(".csv"))
        self.tpa_status.setText(f"Saved {npz}  +  {Path(js).name}  +  {Path(csv_path).name}")

    # ===================== Scope (RTO6) page =========================
    def _connect_scope(self) -> None:
        host = self.scope_host_edit.text().strip()
        if not host:
            self._log("Enter the scope host first")
            return
        self.scope_connect_button.setEnabled(False)

        def connect() -> tuple[ScopeController, str]:
            scope = ScopeController(host=host)
            scope.connect()
            return scope, scope.identify()

        self._run_task("Connect scope", connect, self._on_scope_connected, self._on_scope_error)

    def _on_scope_connected(self, payload: tuple[ScopeController, str]) -> None:
        scope, identity = payload
        self.scope_controller = scope
        self._set_status(self.scope_status_label, "Scope: open", "ok")
        self.scope_connect_button.setEnabled(False)
        self.scope_disconnect_button.setEnabled(True)
        self._log(f"Scope connected: {identity.strip()}")

    def _on_scope_error(self, _error: str) -> None:
        self._set_status(self.scope_status_label, "Scope: error", "error")
        self.scope_connect_button.setEnabled(True)

    def _disconnect_scope(self) -> None:
        scope = self.scope_controller
        self.scope_controller = None
        self._set_status(self.scope_status_label, "Scope: closed", "off")
        self.scope_connect_button.setEnabled(True)
        self.scope_disconnect_button.setEnabled(False)
        if scope is not None:
            self._run_task("Disconnect scope", scope.disconnect)

    # ===================== Scope Monitor page ========================
    _TRIG_SOURCES = [("CH1", "CHANnel1"), ("CH2", "CHANnel2"), ("CH3", "CHANnel3"),
                     ("CH4", "CHANnel4"), ("EXT", "EXTernanalog")]

    def _build_scope_monitor_widget(self) -> QtWidgets.QWidget:
        """Embeddable triggered scope readout (lives inside the TPA encoder page)."""
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)

        cfg = self._panel("Scope Monitor · trigger & averaging")
        grid = QtWidgets.QGridLayout(cfg)
        self.mon_channel = QtWidgets.QComboBox(); self.mon_channel.addItems(["1", "2", "3", "4"])
        self.mon_trig_source = QtWidgets.QComboBox()
        for label, _tok in self._TRIG_SOURCES:
            self.mon_trig_source.addItem(label)
        self.mon_trig_source.setCurrentIndex(2)
        self.mon_trig_level = QtWidgets.QDoubleSpinBox()
        self.mon_trig_level.setRange(-5.0, 5.0); self.mon_trig_level.setSingleStep(0.1)
        self.mon_trig_level.setValue(1.5); self.mon_trig_level.setSuffix(" V")
        self.mon_hold = QtWidgets.QDoubleSpinBox()
        self.mon_hold.setRange(0.0, 10000.0); self.mon_hold.setValue(100.0); self.mon_hold.setSuffix(" ms")
        self.mon_duration = QtWidgets.QDoubleSpinBox()
        self.mon_duration.setRange(0.001, 10.0); self.mon_duration.setDecimals(3)
        self.mon_duration.setValue(1.0); self.mon_duration.setSuffix(" s")
        self.mon_decimation = QtWidgets.QComboBox(); self.mon_decimation.addItems(["HRESolution", "SAMPle"])
        self.mon_bandwidth = QtWidgets.QComboBox(); self.mon_bandwidth.addItems(["(keep)", "FULL", "B800", "B200", "B20"])
        self.mon_digfilter = QtWidgets.QLineEdit(""); self.mon_digfilter.setPlaceholderText("off")
        pairs = [("Channel", self.mon_channel), ("Trigger src", self.mon_trig_source),
                 ("Level", self.mon_trig_level), ("Hold", self.mon_hold),
                 ("Average for", self.mon_duration), ("Decimation", self.mon_decimation),
                 ("BW limit", self.mon_bandwidth), ("Digital LP", self.mon_digfilter)]
        for i, (label, widget) in enumerate(pairs):
            r, c = i // 2, (i % 2) * 2
            grid.addWidget(QtWidgets.QLabel(label), r, c)
            grid.addWidget(widget, r, c + 1)
        v.addWidget(cfg)

        self.mon_count_label = QtWidgets.QLabel("0 patterns")
        self.mon_count_label.setObjectName("PageSubtitle")
        self.mon_count_label.setAlignment(QtCore.Qt.AlignCenter)
        v.addWidget(self.mon_count_label)

        self.mon_fig = Figure(figsize=(4, 2.4), tight_layout=True)
        self.mon_canvas = FigureCanvas(self.mon_fig)
        v.addWidget(self._panel_with_widget("Average per pattern", self.mon_canvas), 1)

        # This readout is a behaviour recorder, not a live monitor: each pattern
        # you send appends one (pattern #, average) point via auto-read-on-send.
        self.mon_clear_button = QtWidgets.QPushButton("Clear")
        self.mon_clear_button.setProperty("variant", "ghost")
        self.mon_clear_button.clicked.connect(self._monitor_clear)
        self.mon_save_button = QtWidgets.QPushButton("Save CSV…")
        self.mon_save_button.setProperty("variant", "ghost")
        self.mon_save_button.clicked.connect(self._monitor_save)
        self.mon_read_on_send = QtWidgets.QCheckBox("Auto-read on SLM send")
        self.mon_read_on_send.setChecked(True)
        self.mon_read_on_send.setToolTip(
            "After a pattern is sent from this page, take one free-run averaged "
            "scope reading and append it to the record."
        )
        v.addWidget(self.mon_read_on_send)
        row = QtWidgets.QHBoxLayout(); row.addWidget(self.mon_clear_button); row.addWidget(self.mon_save_button)
        v.addLayout(row)
        return w

    def _monitor_settings(self, trigger_mode: str = "NORMal") -> MonitorSettings:
        cutoff_text = self.mon_digfilter.text().strip()
        try:
            cutoff = float(cutoff_text) if cutoff_text else None
        except ValueError:
            cutoff = None
        bw = self.mon_bandwidth.currentText()
        return MonitorSettings(
            channel=int(self.mon_channel.currentText()),
            trigger_mode=trigger_mode,
            trigger_source=self._TRIG_SOURCES[self.mon_trig_source.currentIndex()][1],
            trigger_level=self.mon_trig_level.value(),
            trigger_slope="POSitive",
            hold=self.mon_hold.value() / 1000.0,      # ms -> s
            duration=self.mon_duration.value(),
            decimation=self.mon_decimation.currentText(),
            bandwidth_limit=None if bw == "(keep)" else bw,
            digital_filter_cutoff=cutoff,
        )

    def _on_monitor_sample(self, sample: MonitorSample) -> None:
        self._monitor_values.append(sample.value)
        self.mon_count_label.setText(f"{len(self._monitor_values)} patterns")
        self._mon_status(f"pattern #{len(self._monitor_values)}: {sample.value*1000:.4f} mV")
        self._monitor_draw()

    def _monitor_draw(self) -> None:
        self.mon_fig.clear()
        self.mon_fig.patch.set_facecolor("#101820")
        ax = self.mon_fig.add_subplot(111)
        self._style_dark_axes(ax)
        ax.set_xlabel("Pattern # (send order)")
        ax.set_ylabel("Average (mV)")
        if self._monitor_values:
            n = len(self._monitor_values)
            xs = range(1, n + 1)
            ax.plot(xs, [v * 1000 for v in self._monitor_values],
                    marker="o", ms=3, color="#47b8e0", linewidth=0.8)
            # integer-only ticks on the pattern axis (1, 2, 3, …)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_xlim(0.5, n + 0.5)
        self.mon_canvas.draw_idle()

    def _monitor_clear(self) -> None:
        self._monitor_values = []
        self.mon_count_label.setText("0 patterns")
        self._monitor_draw()

    def _monitor_save(self) -> None:
        if not self._monitor_values:
            self._mon_status("No readings to save.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save pattern readings", "scope_readings.csv", "CSV (*.csv)")
        if not path:
            return
        import csv as _csv

        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = _csv.writer(f)
            writer.writerow(["pattern", "mean_V"])
            for i, v in enumerate(self._monitor_values, start=1):
                writer.writerow([i, v])
        self._log(f"Pattern readings saved: {path}")

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

    def _current_slm_pattern(self) -> np.ndarray | None:
        controller = self.controller
        if controller is None:
            return None
        try:
            return controller.current_pattern()
        except Exception:
            return None

    def _describe_slm_pattern(self) -> str | None:
        controller = self.controller
        if controller is None:
            return None
        try:
            return controller.describe_last_display()
        except Exception:
            return None

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
            sweep_nm = self.step_widgets[3]["sweep_nm"].value() or None
            stride = self.step_widgets[3]["stride"].value()
            refine = self.step_widgets[3]["refine"].isChecked()
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
                window_size=window, wavelength_window_nm=avg_nm,
                sweep_span_nm=sweep_nm, coordinate_stride=stride,
                refine_wavelength=refine, region=region,
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
            sweep_nm = self.step_widgets[3]["sweep_nm"].value() or None
            stride = self.step_widgets[3]["stride"].value()
            refine = self.step_widgets[3]["refine"].isChecked()
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
                window_size=window3, wavelength_window_nm=avg_nm,
                sweep_span_nm=sweep_nm, coordinate_stride=stride,
                refine_wavelength=refine, region=region3,
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
        self.scan_eta_label.setText("Elapsed 0:00 · ETA —")
        self._scan_start_time = time.perf_counter()
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
        start_x, end_x = self._scan_x_range
        done = max(x - start_x + 1, 0)
        self.scan_progress_bar.setValue(done)
        if self._scan_start_time is not None and done > 0:
            elapsed = time.perf_counter() - self._scan_start_time
            total = max(end_x - start_x + 1, 1)
            remaining = (elapsed / done) * max(total - done, 0)
            self.scan_eta_label.setText(
                f"Elapsed {_format_duration(elapsed)} · ETA {_format_duration(remaining)}"
            )
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
        image = _pattern_to_qimage(data)
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
        if hasattr(self, "slm_monitor_view"):
            self.slm_monitor_view.stop()
        if hasattr(self, "enc_monitor_view"):
            self.enc_monitor_view.stop()
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
        if self.scope_stop_event is not None:
            self.scope_stop_event.set()
        if self.monitor_stop_event is not None:
            self.monitor_stop_event.set()
        if self.scope_controller is not None:
            try:
                self.scope_controller.disconnect()
            except Exception:
                pass
            self.scope_controller = None
        super().closeEvent(event)

    def _apply_style(self) -> None:
        self.setStyleSheet(DARK_STYLESHEET)


def main(argv: list[str] | None = None) -> int:
    app = QtWidgets.QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("Santec SLM Control")
    window = MainWindow()
    window.show()
    return app.exec_()
