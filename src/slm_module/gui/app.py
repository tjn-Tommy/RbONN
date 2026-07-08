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

from daq_module.controller import DAQController, DAQMonitorSettings
from osa_module.controller import MeasurementSettings, OSAController
from osa_module.driver import OSAError
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
    batch_intensity_calibration,
    build_channel_calibration_grid,
    find_min_max_intensity_levels,
    intensity_calibration,
    load_calibration_result,
    load_wavelength_map_csv,
    refine_center_coordinate_with_osa,
    restrict_to_measured_intensity_range,
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
    ChannelSpectrum,
    EncodingGain,
    ModulationErrorResult,
    encoding_gain,
    measure_channel_spectra,
    measure_one_channel,
    write_analysis_csv,
    write_gain_csv,
)
from ..encoding import (
    ChannelLayout,
    build_channel_layout,
    build_single_anchor_layout,
    encode_to_pattern,
    interpolate_coordinate_for_wavelength,
    optimize_from_osa,
)
from ..optimization import (
    OPTIMIZED_ENCODING_SHAPE,
    OSAOptimizationConfig,
    OptimizationAborted,
    OptimizationProgress,
    OptimizationResult,
    independent_intensity_profile,
    amplitudes_to_intensity_commands,
    load_optimization_result,
    mirror_intensity_profile,
    validate_independent_profile,
)
from ..tpa_pair import (
    TPAPairAborted,
    TPAPairProgress,
    TPAPairResult,
    build_sweep,
    load_tpa_pair_csv,
    measure_pair_grids,
    save_tpa_pair_json,
    write_tpa_pair_csv,
)
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
        "fast_center": (
            "Fast channel setup - 778 nm center",
            "x coordinate (px)",
            "Wavelength (nm)",
        ),
        "batch_intensity": (
            "Fast channel calibration",
            "Level",
            "Mean normalized intensity",
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
    tpa_progress = QtCore.pyqtSignal(object)
    edge_gain_progress = QtCore.pyqtSignal(int, int, str)
    edge_optimization_progress = QtCore.pyqtSignal(object)
    qt_test_progress = QtCore.pyqtSignal(int, int, str)
    osa_trace_ready = QtCore.pyqtSignal(object)
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
        self._calibration_is_running = False
        self._active_calibration_label: str | None = None
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
        self._pipeline_optimization_active = False
        self.edge_gain_stop_event: threading.Event | None = None
        self._qt_test: dict[str, ChannelSpectrum] | None = None  # quick-test A/B result
        self.qt_test_stop_event: threading.Event | None = None
        self.qt_layout: ChannelLayout | None = None  # quick-test layout from picked calib
        self.osa_view_trace = None                # last OSA viewer trace (for save)
        self.osa_view_stop_event: threading.Event | None = None
        self._enc_wheel_step = 0.2   # scroll sensitivity for channel value cells
        self._enc_calib_override: CalibrationResult | None = None
        self.analysis_result: ModulationErrorResult | None = None
        self.analysis_stop_event: threading.Event | None = None
        self._ana_capture_dir: str | None = None
        self.tpa_result: TPAPairResult | None = None
        self.tpa_stop_event: threading.Event | None = None
        self.scope_controller: ScopeController | None = None
        self.scope_stop_event: threading.Event | None = None
        self.daq_controller: DAQController | None = None
        self.monitor_stop_event: threading.Event | None = None
        self._monitor_values: list[float] = []
        self._monitor_stds: list[float] = []
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
        self.tpa_progress.connect(self._on_tpa_progress)
        self.edge_gain_progress.connect(self._edge_gain_progress)
        self.edge_optimization_progress.connect(self._edge_optimization_progress)
        self.qt_test_progress.connect(self._qt_test_progress)
        self.osa_trace_ready.connect(self._osa_view_on_trace)
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
            ("\N{ELECTRIC PLUG}  Connections", "Connect SLM, OSA, scope and DAQ"),
            ("\N{LINK SYMBOL}  SLM Control", "Grayscale and CSV display"),
            ("\N{CHART WITH UPWARDS TREND}  Calibration", "Intensity, mod error, scope holding"),
            ("\N{LEFT RIGHT ARROW}  Center Scan", "Sweep a window across x"),
            ("\N{TRIGRAM FOR HEAVEN}  Phase Segments", "Piecewise phase along x"),
            ("\N{HIGH VOLTAGE SIGN}  TPA Encoding", "Channel grid encoding + scope/DAQ readout"),
            ("\N{DOWNWARDS ARROW WITH TIP RIGHTWARDS}  Shape",
             "Global per-column encoding shape (applied to all encoding "
             "+ calibration) + OSA optimisation hook"),
            ("\N{WHITE HEAVY CHECK MARK}  Quick Test",
             "A/B crosstalk test: flat vs optimised encoding shape from OSA data"),
            ("\N{SATELLITE ANTENNA}  OSA Viewer",
             "Live OSA spectrum viewer: single / continuous sweeps with settings"),
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
        self.stack.addWidget(self._build_quick_test_page())
        self.stack.addWidget(self._build_osa_viewer_page())

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

        # ---- DAQ ----
        # Scope and DAQ both read the same PMT signal on the TPA encoder page,
        # so only one may be connected at a time (see _on_scope_connected /
        # _on_daq_connected).
        daq = self._panel("DAQ (NI-DAQmx)")
        dql = QtWidgets.QGridLayout(daq)
        self.daq_device_edit = QtWidgets.QLineEdit("Dev1")
        self.daq_device_edit.setPlaceholderText("NI-DAQ device name")
        self.daq_connect_button = QtWidgets.QPushButton("Connect DAQ")
        self.daq_connect_button.clicked.connect(self._connect_daq)
        self.daq_disconnect_button = QtWidgets.QPushButton("Disconnect")
        self.daq_disconnect_button.setProperty("variant", "ghost")
        self.daq_disconnect_button.setEnabled(False)
        self.daq_disconnect_button.clicked.connect(self._disconnect_daq)
        self.daq_status_label = QtWidgets.QLabel("DAQ: closed")
        self._set_status(self.daq_status_label, "DAQ: closed", "off")
        dql.addWidget(QtWidgets.QLabel("Device"), 0, 0)
        dql.addWidget(self.daq_device_edit, 0, 1)
        dql.addWidget(self.daq_connect_button, 0, 2)
        dql.addWidget(self.daq_disconnect_button, 0, 3)
        dql.addWidget(self.daq_status_label, 0, 4)
        dql.setColumnStretch(1, 1)
        page.layout().addWidget(daq)

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
        tabs.addTab(self._build_fast_channel_calibration_page(), "Step 3b - Fast Channels")
        tabs.addTab(self._build_analysis_page(), "Step 4 · Mod Error")
        tabs.addTab(self._build_scope_holding_tab(), "Step 5 · Holding")
        tabs.addTab(self._build_tpa_tab(), "Step 6 · TPA Efficiency")
        tabs.insertTab(0, self._build_pipeline_page(), "Pipeline")
        tabs.addTab(self._build_stage3_reopt_page(), "Step 4b - Stage3 Reopt")
        self.calibration_tabs = tabs
        lay.addWidget(tabs)

        # every Run button, toggled together by _set_calibration_running
        self.calibration_run_buttons = [
            self.step_widgets[1]["run"],
            self.step_widgets[2]["run"],
            self.step_widgets[3]["run"],
            self.run_all_button,
            self.fast_channel_run_button,
            self.pipeline_run_button,
            self.stage3_reopt_run_button,
        ]
        return page

    def _build_pipeline_page(self) -> QtWidgets.QWidget:
        """Build the file-only calibration and encoding-optimisation runner."""
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.addWidget(
            self._caption(
                "Select any stages to run in dependency order. Every "
                "handoff is saved and reloaded from a file; no in-memory result is "
                "used as a step input. Step settings come from the corresponding tabs."
            )
        )

        flow = self._panel("Pipeline steps and files")
        grid = QtWidgets.QGridLayout(flow)
        grid.addWidget(QtWidgets.QLabel("Run"), 0, 0)
        grid.addWidget(QtWidgets.QLabel("Step"), 0, 1)
        grid.addWidget(QtWidgets.QLabel("Input"), 0, 2, 1, 2)
        grid.addWidget(QtWidgets.QLabel("Output JSON"), 0, 4, 1, 2)

        self.pipeline_checks: dict[int, QtWidgets.QCheckBox] = {}
        self.pipeline_input_edits: dict[int, QtWidgets.QLineEdit] = {}
        self.pipeline_input_buttons: dict[int, QtWidgets.QPushButton] = {}
        self.pipeline_source_labels: dict[int, QtWidgets.QLabel] = {}
        self.pipeline_output_edits: dict[int, QtWidgets.QLineEdit] = {}
        self.pipeline_output_buttons: dict[int, QtWidgets.QPushButton] = {}

        step_names = {
            1: "Step 1 - Min/Max",
            2: "Step 2 - Wavelength",
            3: "Step 3 - Intensity",
        }
        output_names = {step: self._default_calib_name(step) for step in (1, 2, 3)}
        for row, step in enumerate((1, 2, 3), start=1):
            check = QtWidgets.QCheckBox()
            check.setChecked(True)
            check.toggled.connect(self._refresh_pipeline_ui)
            self.pipeline_checks[step] = check
            grid.addWidget(check, row, 0, QtCore.Qt.AlignCenter)
            grid.addWidget(QtWidgets.QLabel(step_names[step]), row, 1)

            if step == 1:
                no_input = QtWidgets.QLabel("No input")
                no_input.setObjectName("PageSubtitle")
                grid.addWidget(no_input, row, 2, 1, 2)
            else:
                input_edit = QtWidgets.QLineEdit()
                input_edit.setPlaceholderText(
                    "Required when the preceding step is not selected"
                )
                input_button = QtWidgets.QPushButton("Browse")
                input_filter = (
                    "JSON Files (*.json)"
                    if step == 2
                    else "Calibration Files (*.json *.csv)"
                )
                input_button.clicked.connect(
                    lambda _checked=False, edit=input_edit, number=step,
                    filt=input_filter: self._browse_open_into(
                        edit, f"Select Step {number} input", filt
                    )
                )
                self.pipeline_input_edits[step] = input_edit
                self.pipeline_input_buttons[step] = input_button
                grid.addWidget(input_edit, row, 2)
                grid.addWidget(input_button, row, 3)

            output_edit = QtWidgets.QLineEdit(output_names[step])
            output_button = QtWidgets.QPushButton("Browse")
            output_button.clicked.connect(
                lambda _checked=False, edit=output_edit, name=output_names[step]:
                self._browse_save_into(edit, name, "JSON Files (*.json)")
            )
            self.pipeline_output_edits[step] = output_edit
            self.pipeline_output_buttons[step] = output_button
            grid.addWidget(output_edit, row, 4)
            grid.addWidget(output_button, row, 5)

        for row, step in enumerate((2, 3), start=4):
            source_label = QtWidgets.QLabel()
            source_label.setObjectName("PageSubtitle")
            self.pipeline_source_labels[step] = source_label
            grid.addWidget(source_label, row, 2, 1, 4)

        self.pipeline_csv_edit = QtWidgets.QLineEdit("calibration.csv")
        self.pipeline_csv_button = QtWidgets.QPushButton("Browse")
        self.pipeline_csv_button.clicked.connect(
            lambda: self._browse_save_into(
                self.pipeline_csv_edit, "calibration.csv", "CSV Files (*.csv)"
            )
        )
        grid.addWidget(QtWidgets.QLabel("Step 3 output CSV"), 6, 1)
        grid.addWidget(self.pipeline_csv_edit, 6, 4)
        grid.addWidget(self.pipeline_csv_button, 6, 5)
        grid.setColumnStretch(2, 1)
        grid.setColumnStretch(4, 1)
        layout.addWidget(flow)

        optimization = self._panel("Encoding Optimization")
        opt_grid = QtWidgets.QGridLayout(optimization)
        self.pipeline_checks[4] = QtWidgets.QCheckBox("Run Encoding Optimization")
        self.pipeline_checks[4].setChecked(False)
        self.pipeline_checks[4].toggled.connect(self._refresh_pipeline_ui)
        opt_grid.addWidget(self.pipeline_checks[4], 0, 0, 1, 3)

        self.pipeline_quick_optimization_check = QtWidgets.QCheckBox(
            "Fast single-channel mode at the Encoding centre wavelength"
        )
        self.pipeline_quick_optimization_check.setToolTip(
            "Use a Step 2 wavelength map to interpolate the centre pixel, run "
            "a local intensity calibration there, and optimise only that one "
            "OSA anchor. Full three-anchor mode remains the default."
        )
        self.pipeline_quick_optimization_check.toggled.connect(
            self._refresh_pipeline_ui
        )
        opt_grid.addWidget(self.pipeline_quick_optimization_check, 1, 0, 1, 3)

        self.pipeline_input_edits[4] = QtWidgets.QLineEdit()
        self.pipeline_input_edits[4].setPlaceholderText(
            "External Step 3 calibration JSON when Step 3 is not selected"
        )
        self.pipeline_input_buttons[4] = QtWidgets.QPushButton("Browse")
        self.pipeline_input_buttons[4].clicked.connect(
            lambda: self._browse_open_into(
                self.pipeline_input_edits[4],
                "Select encoding calibration",
                "JSON Files (*.json)",
            )
        )
        self.pipeline_optimization_input_label = QtWidgets.QLabel(
            "Calibration input"
        )
        opt_grid.addWidget(self.pipeline_optimization_input_label, 2, 0)
        opt_grid.addWidget(self.pipeline_input_edits[4], 2, 1)
        opt_grid.addWidget(self.pipeline_input_buttons[4], 2, 2)

        self.pipeline_source_labels[4] = QtWidgets.QLabel()
        self.pipeline_source_labels[4].setObjectName("PageSubtitle")
        opt_grid.addWidget(self.pipeline_source_labels[4], 3, 1, 1, 2)

        self.pipeline_profile_source_combo = QtWidgets.QComboBox()
        self.pipeline_profile_source_combo.addItems(["Direct values", "From file"])
        self.pipeline_profile_source_combo.currentIndexChanged.connect(
            self._refresh_pipeline_ui
        )
        opt_grid.addWidget(QtWidgets.QLabel("Initial profile source"), 4, 0)
        opt_grid.addWidget(self.pipeline_profile_source_combo, 4, 1, 1, 2)

        self.pipeline_profile_values_edit = QtWidgets.QLineEdit(
            "1, 1, 1, 1, 1, 1, 1, 1"
        )
        self.pipeline_profile_values_edit.setPlaceholderText(
            "8 values, or a symmetric 15-value profile"
        )
        self.pipeline_profile_values_edit.setToolTip(
            "Normalised intensity ratios in [0, 1], separated by commas or spaces."
        )
        opt_grid.addWidget(QtWidgets.QLabel("Initial profile values"), 5, 0)
        opt_grid.addWidget(self.pipeline_profile_values_edit, 5, 1, 1, 2)

        self.pipeline_profile_edit = QtWidgets.QLineEdit()
        self.pipeline_profile_edit.setPlaceholderText(
            "8-value l_init or symmetric 15-value profile"
        )
        self.pipeline_profile_edit.setToolTip(
            "JSON may be an array or use l_init, initial_l, final_l, final_profile, "
            "or profile. CSV/TXT may contain 8 numeric values or a symmetric 15-value "
            "intensity profile."
        )
        self.pipeline_profile_button = QtWidgets.QPushButton("Browse")
        self.pipeline_profile_button.clicked.connect(
            lambda: self._browse_open_into(
                self.pipeline_profile_edit,
                "Select initial encoding profile",
                "Profile Files (*.json *.csv *.txt)",
            )
        )
        opt_grid.addWidget(QtWidgets.QLabel("Initial profile file"), 6, 0)
        opt_grid.addWidget(self.pipeline_profile_edit, 6, 1)
        opt_grid.addWidget(self.pipeline_profile_button, 6, 2)

        self.pipeline_optimization_root_edit = QtWidgets.QLineEdit(
            "data/osa_optimization"
        )
        self.pipeline_optimization_root_button = QtWidgets.QPushButton("Browse")
        self.pipeline_optimization_root_button.clicked.connect(
            self._browse_pipeline_optimization_root
        )
        opt_grid.addWidget(QtWidgets.QLabel("Output root"), 7, 0)
        opt_grid.addWidget(self.pipeline_optimization_root_edit, 7, 1)
        opt_grid.addWidget(self.pipeline_optimization_root_button, 7, 2)

        self.pipeline_optimization_name_edit = QtWidgets.QLineEdit()
        self.pipeline_optimization_name_edit.setPlaceholderText(
            "Optional; blank uses a timestamped run directory"
        )
        opt_grid.addWidget(QtWidgets.QLabel("Run name"), 8, 0)
        opt_grid.addWidget(self.pipeline_optimization_name_edit, 8, 1, 1, 2)

        self.pipeline_quick_levels_edit = QtWidgets.QLineEdit("420~870+50")
        self.pipeline_quick_levels_edit.setToolTip(
            "SLM levels for the interpolated centre pixel, formatted as "
            "min~max+stride, for example 420~870+50. The maximum level is "
            "included even when the stride does not land on it exactly. After "
            "acquisition, the measured minimum and maximum define the "
            "off-to-on range used by encoding and optimisation."
        )
        self.pipeline_quick_levels_label = QtWidgets.QLabel(
            "Quick SLM range"
        )
        opt_grid.addWidget(self.pipeline_quick_levels_label, 9, 0)
        opt_grid.addWidget(self.pipeline_quick_levels_edit, 9, 1, 1, 2)

        self.pipeline_quick_calibration_edit = QtWidgets.QLineEdit(
            "calib_quick_center.json"
        )
        self.pipeline_quick_calibration_button = QtWidgets.QPushButton("Browse")
        self.pipeline_quick_calibration_button.clicked.connect(
            lambda: self._browse_save_into(
                self.pipeline_quick_calibration_edit,
                "calib_quick_center.json",
                "JSON Files (*.json)",
            )
        )
        self.pipeline_quick_calibration_label = QtWidgets.QLabel(
            "Quick calibration JSON"
        )
        opt_grid.addWidget(self.pipeline_quick_calibration_label, 10, 0)
        opt_grid.addWidget(self.pipeline_quick_calibration_edit, 10, 1)
        opt_grid.addWidget(self.pipeline_quick_calibration_button, 10, 2)
        opt_grid.setColumnStretch(1, 1)
        layout.addWidget(optimization)

        reopt = self._panel("Stage 3 Re-optimization")
        reopt_grid = QtWidgets.QGridLayout(reopt)
        self.pipeline_stage3_only_check = QtWidgets.QCheckBox(
            "Use supplied Stage 1 level/profile data and skip Stage 1 search"
        )
        self.pipeline_stage3_only_check.setToolTip(
            "Start from an existing 8-value Stage 1 level/profile JSON, rebuild "
            "the amplitude LUT, and run Stage 3 with the scan settings below."
        )
        self.pipeline_stage3_only_check.toggled.connect(self._refresh_pipeline_ui)
        reopt_grid.addWidget(self.pipeline_stage3_only_check, 0, 0, 1, 4)

        self.pipeline_reopt_profile_edit = QtWidgets.QLineEdit()
        self.pipeline_reopt_profile_edit.setPlaceholderText(
            "stage1_result.json, final_result.json, or an 8-value profile file"
        )
        self.pipeline_reopt_profile_edit.setToolTip(
            "JSON may be an array or use l, l_init, initial_l, final_l, "
            "final_profile, or profile."
        )
        self.pipeline_reopt_profile_button = QtWidgets.QPushButton("Browse")
        self.pipeline_reopt_profile_button.clicked.connect(
            lambda: self._browse_open_into(
                self.pipeline_reopt_profile_edit,
                "Select Stage 1 level/profile data",
                "Profile Files (*.json *.csv *.txt)",
            )
        )
        reopt_grid.addWidget(QtWidgets.QLabel("Stage 1 level/profile"), 1, 0)
        reopt_grid.addWidget(self.pipeline_reopt_profile_edit, 1, 1, 1, 2)
        reopt_grid.addWidget(self.pipeline_reopt_profile_button, 1, 3)

        self.pipeline_reopt_calibration_edit = QtWidgets.QLineEdit()
        self.pipeline_reopt_calibration_edit.setPlaceholderText(
            "Existing quick intensity calibration JSON"
        )
        self.pipeline_reopt_calibration_edit.setToolTip(
            "Fast single-channel re-optimization can reuse a saved one-pixel "
            "quick intensity calibration instead of measuring the SLM range again."
        )
        self.pipeline_reopt_calibration_button = QtWidgets.QPushButton("Browse")
        self.pipeline_reopt_calibration_button.clicked.connect(
            lambda: self._browse_open_into(
                self.pipeline_reopt_calibration_edit,
                "Select quick intensity calibration",
                "JSON Files (*.json)",
            )
        )
        self.pipeline_reopt_calibration_label = QtWidgets.QLabel(
            "Quick calibration input"
        )
        reopt_grid.addWidget(self.pipeline_reopt_calibration_label, 2, 0)
        reopt_grid.addWidget(self.pipeline_reopt_calibration_edit, 2, 1, 1, 2)
        reopt_grid.addWidget(self.pipeline_reopt_calibration_button, 2, 3)

        self.pipeline_reopt_sensitivity_combo = QtWidgets.QComboBox()
        self.pipeline_reopt_sensitivity_combo.addItems(
            ["NORM", "MID", "HIGH1", "HIGH2", "HIGH3"]
        )
        self.pipeline_reopt_sensitivity_combo.setCurrentText("HIGH1")
        self.pipeline_reopt_averages_spin = self._spin(1, 20, 1)
        self.pipeline_reopt_stage2_repeats_spin = self._spin(1, 20, 3)
        self.pipeline_reopt_stage3_maxfev_spin = self._spin(1, 500, 100)
        self.pipeline_reopt_rerank_averages_spin = self._spin(1, 20, 3)
        reopt_grid.addWidget(QtWidgets.QLabel("Sensitivity"), 3, 0)
        reopt_grid.addWidget(self.pipeline_reopt_sensitivity_combo, 3, 1)
        reopt_grid.addWidget(QtWidgets.QLabel("OSA averages"), 3, 2)
        reopt_grid.addWidget(self.pipeline_reopt_averages_spin, 3, 3)
        reopt_grid.addWidget(QtWidgets.QLabel("Stage3 baseline repeats"), 4, 0)
        reopt_grid.addWidget(self.pipeline_reopt_stage2_repeats_spin, 4, 1)
        reopt_grid.addWidget(QtWidgets.QLabel("Stage3 max evals"), 4, 2)
        reopt_grid.addWidget(self.pipeline_reopt_stage3_maxfev_spin, 4, 3)
        reopt_grid.addWidget(QtWidgets.QLabel("Rerank averages"), 5, 0)
        reopt_grid.addWidget(self.pipeline_reopt_rerank_averages_spin, 5, 1)
        reopt_grid.setColumnStretch(1, 1)

        layout.addWidget(
            self._caption(
                "A selected prerequisite supplies its output file. A skipped "
                "prerequisite must be supplied as an external file. The Encoding "
                "Optimization initial profile may be entered directly or loaded "
                "from a file. Fast single-channel mode uses the Step 2 map to "
                "interpolate the Encoding centre wavelength (778 nm by default), "
                "measures the min~max+stride SLM range at that physical pixel, then "
                "uses its measured min-to-max range for the single-anchor optimisation."
            )
        )

        self.pipeline_log = QtWidgets.QPlainTextEdit()
        self.pipeline_log.setReadOnly(True)
        self.pipeline_log.setObjectName("LogBox")
        self.pipeline_log.setPlaceholderText("Pipeline activity will appear here.")
        layout.addWidget(self._panel_with_widget("Pipeline log", self.pipeline_log), 1)

        action_row = QtWidgets.QHBoxLayout()
        self.pipeline_status_label = QtWidgets.QLabel("Ready")
        self.pipeline_run_button = QtWidgets.QPushButton("Run Pipeline")
        self.pipeline_run_button.setEnabled(False)
        self.pipeline_run_button.clicked.connect(self._run_pipeline)
        self.pipeline_stop_button = QtWidgets.QPushButton("Stop")
        self.pipeline_stop_button.setProperty("variant", "danger")
        self.pipeline_stop_button.setEnabled(False)
        self.pipeline_stop_button.clicked.connect(self._stop_full_calibration)
        action_row.addWidget(self.pipeline_status_label, 1)
        action_row.addWidget(self.pipeline_run_button)
        action_row.addWidget(self.pipeline_stop_button)
        layout.addLayout(action_row)
        self._refresh_pipeline_ui()
        return page

    def _refresh_pipeline_ui(self) -> None:
        """Update file-source hints and controls after the selection changes."""
        if not hasattr(self, "pipeline_checks"):
            return
        selected = {
            step for step, check in self.pipeline_checks.items() if check.isChecked()
        }
        for step in (1, 2, 3):
            enabled = step in selected
            self.pipeline_output_edits[step].setEnabled(enabled)
            self.pipeline_output_buttons[step].setEnabled(enabled)
        for step in (2, 3):
            predecessor = step - 1
            chained = step in selected and predecessor in selected
            external = step in selected and not chained
            self.pipeline_input_edits[step].setEnabled(external)
            self.pipeline_input_buttons[step].setEnabled(external)
            if chained:
                source = (
                    f"Step {step} input: Step {predecessor} output file "
                    "(saved, then reloaded)"
                )
            elif step in selected:
                source = f"Step {step} input: external file (required)"
            else:
                source = f"Step {step}: not selected"
            self.pipeline_source_labels[step].setText(source)

        step3_selected = 3 in selected
        self.pipeline_csv_edit.setEnabled(step3_selected)
        self.pipeline_csv_button.setEnabled(step3_selected)
        optimize_selected = 4 in selected
        self.pipeline_quick_optimization_check.setEnabled(optimize_selected)
        quick_optimization = (
            optimize_selected and self.pipeline_quick_optimization_check.isChecked()
        )
        stage3_only = (
            optimize_selected and self.pipeline_stage3_only_check.isChecked()
        )
        source_step = 2 if quick_optimization else 3
        source_step_selected = source_step in selected
        optimization_external = optimize_selected and not source_step_selected
        self.pipeline_input_edits[4].setEnabled(optimization_external)
        self.pipeline_input_buttons[4].setEnabled(optimization_external)
        if quick_optimization:
            self.pipeline_optimization_input_label.setText("Step 2 input")
            self.pipeline_input_edits[4].setPlaceholderText(
                "External Step 2 calibration JSON when Step 2 is not selected"
            )
        else:
            self.pipeline_optimization_input_label.setText("Calibration input")
            self.pipeline_input_edits[4].setPlaceholderText(
                "External Step 3 calibration JSON when Step 3 is not selected"
            )
        if optimize_selected and source_step_selected:
            optimization_source = (
                f"Optimization calibration: Step {source_step} output JSON "
                "(saved, then reloaded)"
            )
        elif optimize_selected:
            optimization_source = (
                f"Optimization calibration: external Step {source_step} JSON "
                "(required)"
            )
        else:
            optimization_source = "Encoding Optimization: not selected"
        self.pipeline_source_labels[4].setText(optimization_source)
        for widget in (
            self.pipeline_quick_levels_label,
            self.pipeline_quick_levels_edit,
            self.pipeline_quick_calibration_label,
            self.pipeline_quick_calibration_edit,
            self.pipeline_quick_calibration_button,
        ):
            widget.setEnabled(quick_optimization and not stage3_only)
        profile_from_file = self.pipeline_profile_source_combo.currentIndex() == 1
        self.pipeline_profile_source_combo.setEnabled(optimize_selected)
        self.pipeline_profile_values_edit.setEnabled(
            optimize_selected and not profile_from_file and not stage3_only
        )
        self.pipeline_profile_edit.setEnabled(
            optimize_selected and profile_from_file and not stage3_only
        )
        self.pipeline_profile_button.setEnabled(
            optimize_selected and profile_from_file and not stage3_only
        )
        self.pipeline_stage3_only_check.setEnabled(optimize_selected)
        reopt_enabled = optimize_selected and stage3_only
        for widget in (
            self.pipeline_reopt_profile_edit,
            self.pipeline_reopt_profile_button,
            self.pipeline_reopt_sensitivity_combo,
            self.pipeline_reopt_averages_spin,
            self.pipeline_reopt_stage2_repeats_spin,
            self.pipeline_reopt_stage3_maxfev_spin,
            self.pipeline_reopt_rerank_averages_spin,
        ):
            widget.setEnabled(reopt_enabled)
        reopt_quick_calibration_enabled = reopt_enabled and quick_optimization
        for widget in (
            self.pipeline_reopt_calibration_label,
            self.pipeline_reopt_calibration_edit,
            self.pipeline_reopt_calibration_button,
        ):
            widget.setEnabled(reopt_quick_calibration_enabled)
        for widget in (
            self.pipeline_optimization_root_edit,
            self.pipeline_optimization_root_button,
            self.pipeline_optimization_name_edit,
        ):
            widget.setEnabled(optimize_selected)
        self.pipeline_run_button.setEnabled(
            bool(selected)
            and self.osa_controller is not None
            and not self._calibration_is_running
        )

    def _browse_pipeline_optimization_root(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select optimization output root",
            self.pipeline_optimization_root_edit.text().strip() or ".",
        )
        if path:
            self.pipeline_optimization_root_edit.setText(path)

    def _build_stage3_reopt_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.addWidget(
            self._caption(
                "Standalone Stage 3 re-optimization. Provide a Step 2 wavelength "
                "map, an existing one-pixel quick intensity calibration, and a "
                "Stage 1 level/profile file. The run skips Stage 1 search and "
                "rebuilds the LUT + Stage 3 optimisation with the settings below."
            )
        )

        inputs = self._panel("Input files")
        grid = QtWidgets.QGridLayout(inputs)
        self.stage3_reopt_step2_edit = QtWidgets.QLineEdit()
        self.stage3_reopt_step2_edit.setPlaceholderText("calib_step2.json")
        self.stage3_reopt_step2_button = QtWidgets.QPushButton("Browse")
        self.stage3_reopt_step2_button.clicked.connect(
            lambda: self._browse_open_into(
                self.stage3_reopt_step2_edit,
                "Select Step 2 wavelength map",
                "JSON Files (*.json)",
            )
        )
        self.stage3_reopt_quick_calib_edit = QtWidgets.QLineEdit()
        self.stage3_reopt_quick_calib_edit.setPlaceholderText(
            "calib_quick_center.json"
        )
        self.stage3_reopt_quick_calib_button = QtWidgets.QPushButton("Browse")
        self.stage3_reopt_quick_calib_button.clicked.connect(
            lambda: self._browse_open_into(
                self.stage3_reopt_quick_calib_edit,
                "Select quick intensity calibration",
                "JSON Files (*.json)",
            )
        )
        self.stage3_reopt_profile_edit = QtWidgets.QLineEdit()
        self.stage3_reopt_profile_edit.setPlaceholderText(
            "stage1_result.json or another 8-value profile file"
        )
        self.stage3_reopt_profile_button = QtWidgets.QPushButton("Browse")
        self.stage3_reopt_profile_button.clicked.connect(
            lambda: self._browse_open_into(
                self.stage3_reopt_profile_edit,
                "Select Stage 1 level/profile data",
                "Profile Files (*.json *.csv *.txt)",
            )
        )
        grid.addWidget(QtWidgets.QLabel("Step 2 map"), 0, 0)
        grid.addWidget(self.stage3_reopt_step2_edit, 0, 1)
        grid.addWidget(self.stage3_reopt_step2_button, 0, 2)
        grid.addWidget(QtWidgets.QLabel("Quick calibration"), 1, 0)
        grid.addWidget(self.stage3_reopt_quick_calib_edit, 1, 1)
        grid.addWidget(self.stage3_reopt_quick_calib_button, 1, 2)
        grid.addWidget(QtWidgets.QLabel("Stage 1 level/profile"), 2, 0)
        grid.addWidget(self.stage3_reopt_profile_edit, 2, 1)
        grid.addWidget(self.stage3_reopt_profile_button, 2, 2)
        grid.setColumnStretch(1, 1)
        layout.addWidget(inputs)

        cfg = self._panel("Target and scan settings")
        cfg_grid = QtWidgets.QGridLayout(cfg)
        self.stage3_reopt_center_wl_spin = self._double_spin(
            700.0, 900.0, 778.0, " nm", 2
        )
        self.stage3_reopt_width_spin = self._spin(1, 256, 15)
        self.stage3_reopt_gap_spin = self._spin(0, 64, 5)
        self.stage3_reopt_span_edit = QtWidgets.QLineEdit("0.8nm")
        self.stage3_reopt_sensitivity_combo = QtWidgets.QComboBox()
        self.stage3_reopt_sensitivity_combo.addItems(
            ["NORM", "MID", "HIGH1", "HIGH2", "HIGH3"]
        )
        self.stage3_reopt_sensitivity_combo.setCurrentText("HIGH1")
        self.stage3_reopt_ref_level_edit = QtWidgets.QLineEdit("10uW")
        self.stage3_reopt_yunit_combo = QtWidgets.QComboBox()
        self.stage3_reopt_yunit_combo.addItems(["LOG (dBm)", "LIN (W)"])
        self.stage3_reopt_averages_spin = self._spin(1, 20, 1)
        self.stage3_reopt_baseline_repeats_spin = self._spin(1, 20, 3)
        self.stage3_reopt_maxeval_spin = self._spin(1, 500, 100)
        self.stage3_reopt_rerank_averages_spin = self._spin(1, 20, 3)
        cfg_grid.addWidget(QtWidgets.QLabel("Centre wavelength"), 0, 0)
        cfg_grid.addWidget(self.stage3_reopt_center_wl_spin, 0, 1)
        cfg_grid.addWidget(QtWidgets.QLabel("Channel width"), 0, 2)
        cfg_grid.addWidget(self.stage3_reopt_width_spin, 0, 3)
        cfg_grid.addWidget(QtWidgets.QLabel("Gap px"), 0, 4)
        cfg_grid.addWidget(self.stage3_reopt_gap_spin, 0, 5)
        cfg_grid.addWidget(QtWidgets.QLabel("Span"), 1, 0)
        cfg_grid.addWidget(self.stage3_reopt_span_edit, 1, 1)
        cfg_grid.addWidget(QtWidgets.QLabel("Sensitivity"), 1, 2)
        cfg_grid.addWidget(self.stage3_reopt_sensitivity_combo, 1, 3)
        cfg_grid.addWidget(QtWidgets.QLabel("Ref level"), 1, 4)
        cfg_grid.addWidget(self.stage3_reopt_ref_level_edit, 1, 5)
        cfg_grid.addWidget(QtWidgets.QLabel("Y unit"), 2, 0)
        cfg_grid.addWidget(self.stage3_reopt_yunit_combo, 2, 1)
        cfg_grid.addWidget(QtWidgets.QLabel("OSA averages"), 2, 2)
        cfg_grid.addWidget(self.stage3_reopt_averages_spin, 2, 3)
        cfg_grid.addWidget(QtWidgets.QLabel("Stage3 baseline repeats"), 3, 0)
        cfg_grid.addWidget(self.stage3_reopt_baseline_repeats_spin, 3, 1)
        cfg_grid.addWidget(QtWidgets.QLabel("Stage3 max evals"), 3, 2)
        cfg_grid.addWidget(self.stage3_reopt_maxeval_spin, 3, 3)
        cfg_grid.addWidget(QtWidgets.QLabel("Rerank averages"), 3, 4)
        cfg_grid.addWidget(self.stage3_reopt_rerank_averages_spin, 3, 5)
        layout.addWidget(cfg)

        out = self._panel("Output")
        out_grid = QtWidgets.QGridLayout(out)
        self.stage3_reopt_root_edit = QtWidgets.QLineEdit("data/osa_optimization")
        self.stage3_reopt_root_button = QtWidgets.QPushButton("Browse")
        self.stage3_reopt_root_button.clicked.connect(
            self._browse_stage3_reopt_root
        )
        self.stage3_reopt_name_edit = QtWidgets.QLineEdit()
        self.stage3_reopt_name_edit.setPlaceholderText(
            "Optional; blank uses a timestamped run directory"
        )
        out_grid.addWidget(QtWidgets.QLabel("Output root"), 0, 0)
        out_grid.addWidget(self.stage3_reopt_root_edit, 0, 1)
        out_grid.addWidget(self.stage3_reopt_root_button, 0, 2)
        out_grid.addWidget(QtWidgets.QLabel("Run name"), 1, 0)
        out_grid.addWidget(self.stage3_reopt_name_edit, 1, 1, 1, 2)
        out_grid.setColumnStretch(1, 1)
        layout.addWidget(out)

        self.stage3_reopt_status_label = QtWidgets.QLabel("Ready")
        self.stage3_reopt_run_button = QtWidgets.QPushButton("Run Stage 3 Reopt")
        self.stage3_reopt_run_button.clicked.connect(self._run_stage3_reoptimization)
        self.stage3_reopt_stop_button = QtWidgets.QPushButton("Stop")
        self.stage3_reopt_stop_button.setProperty("variant", "danger")
        self.stage3_reopt_stop_button.setEnabled(False)
        self.stage3_reopt_stop_button.clicked.connect(self._stop_full_calibration)
        action = QtWidgets.QHBoxLayout()
        action.addWidget(self.stage3_reopt_status_label, 1)
        action.addWidget(self.stage3_reopt_run_button)
        action.addWidget(self.stage3_reopt_stop_button)
        layout.addLayout(action)
        layout.addStretch(1)
        return page

    def _browse_stage3_reopt_root(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Stage 3 re-optimization output root",
            self.stage3_reopt_root_edit.text().strip() or ".",
        )
        if path:
            self.stage3_reopt_root_edit.setText(path)

    def _build_fast_channel_calibration_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.addWidget(
            self._caption(
                "Use a Step 2 wavelength map to locate the 778 nm center, optionally "
                "fine tune that center with one OSA trace, generate 15 px + 5 px "
                "channel centers, then calibrate non-neighboring channel groups from "
                "full-span OSA sweeps."
            )
        )

        source = self._panel("Step 2 source")
        source_grid = QtWidgets.QGridLayout(source)
        self.fast_channel_source_combo = QtWidgets.QComboBox()
        self.fast_channel_source_combo.addItems(["Step 2 result (memory)", "From file"])
        self.fast_channel_source_combo.currentIndexChanged.connect(
            self._toggle_fast_channel_source
        )
        self.fast_channel_step2_edit = QtWidgets.QLineEdit()
        self.fast_channel_step2_edit.setPlaceholderText("calib_step2.json")
        self.fast_channel_step2_button = QtWidgets.QPushButton("Browse")
        self.fast_channel_step2_button.clicked.connect(
            lambda: self._browse_open_into(
                self.fast_channel_step2_edit,
                "Select Step 2 wavelength map",
                "Calibration (*.json *.csv)",
            )
        )
        self.fast_channel_min_spin = self._spin(0, 1023, 0)
        self.fast_channel_max_spin = self._spin(0, 1023, 1023)
        source_grid.addWidget(QtWidgets.QLabel("Source"), 0, 0)
        source_grid.addWidget(self.fast_channel_source_combo, 0, 1)
        source_grid.addWidget(self.fast_channel_step2_edit, 1, 1)
        source_grid.addWidget(self.fast_channel_step2_button, 1, 2)
        source_grid.addWidget(QtWidgets.QLabel("CSV min/max"), 2, 0)
        source_grid.addWidget(self.fast_channel_min_spin, 2, 1)
        source_grid.addWidget(self.fast_channel_max_spin, 2, 2)
        source_grid.setColumnStretch(1, 1)
        layout.addWidget(source)

        grid_panel = self._panel("Channel grid")
        grid = QtWidgets.QGridLayout(grid_panel)
        self.fast_channel_target_spin = self._double_spin(
            700.0, 900.0, 778.0, " nm", 3
        )
        self.fast_channel_width_spin = self._spin(1, 256, 15)
        self.fast_channel_gap_spin = self._spin(0, 256, 5)
        self.fast_channel_count_spin = self._spin(1, 200, 20)
        self.fast_channel_skip_spin = self._spin(0, 20, 2)
        self.fast_channel_fine_check = QtWidgets.QCheckBox("OSA fine tune center")
        self.fast_channel_fine_check.setChecked(True)
        self.fast_channel_peak_nm_spin = self._double_spin(0.001, 50.0, 0.2, " nm", 3)
        self.fast_channel_peak_nm_spin.setToolTip(
            "Half-window around the target wavelength for center peak centroiding."
        )
        self.fast_channel_refine_check = QtWidgets.QCheckBox("Refine channel wavelengths")
        self.fast_channel_refine_check.setChecked(True)
        self.fast_channel_refine_nm_spin = self._double_spin(0.001, 50.0, 0.2, " nm", 3)
        self.fast_channel_refine_nm_spin.setToolTip(
            "Half-window around each channel for wavelength refinement."
        )
        self.fast_channel_guard_check = QtWidgets.QCheckBox("Min-level guard bands")
        self.fast_channel_guard_check.setChecked(True)
        self.fast_channel_guard_wl_edit = QtWidgets.QLineEdit("780, 776")
        self.fast_channel_guard_wl_edit.setToolTip(
            "Comma/space separated guard center wavelengths in nm."
        )
        self.fast_channel_guard_nm_spin = self._double_spin(0.001, 5.0, 0.06, " nm", 3)
        self.fast_channel_guard_nm_spin.setToolTip(
            "Half-width around each guard wavelength forced to the minimum level."
        )
        grid.addWidget(QtWidgets.QLabel("Target center"), 0, 0)
        grid.addWidget(self.fast_channel_target_spin, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Channel width"), 0, 2)
        grid.addWidget(self.fast_channel_width_spin, 0, 3)
        grid.addWidget(QtWidgets.QLabel("Gap px"), 0, 4)
        grid.addWidget(self.fast_channel_gap_spin, 0, 5)
        grid.addWidget(QtWidgets.QLabel("Channels/side"), 1, 0)
        grid.addWidget(self.fast_channel_count_spin, 1, 1)
        grid.addWidget(QtWidgets.QLabel("Skip between active"), 1, 2)
        grid.addWidget(self.fast_channel_skip_spin, 1, 3)
        grid.addWidget(self.fast_channel_fine_check, 2, 0, 1, 2)
        grid.addWidget(QtWidgets.QLabel("Center peak half-window"), 2, 2)
        grid.addWidget(self.fast_channel_peak_nm_spin, 2, 3)
        grid.addWidget(self.fast_channel_refine_check, 3, 0, 1, 2)
        grid.addWidget(QtWidgets.QLabel("Refine half-window"), 3, 2)
        grid.addWidget(self.fast_channel_refine_nm_spin, 3, 3)
        grid.addWidget(self.fast_channel_guard_check, 4, 0, 1, 2)
        grid.addWidget(QtWidgets.QLabel("Guard centers"), 4, 2)
        grid.addWidget(self.fast_channel_guard_wl_edit, 4, 3)
        grid.addWidget(QtWidgets.QLabel("Guard half-width"), 4, 4)
        grid.addWidget(self.fast_channel_guard_nm_spin, 4, 5)
        layout.addWidget(grid_panel)

        scan = self._panel("OSA and level sweep")
        scan_grid = QtWidgets.QGridLayout(scan)
        self.fast_channel_center_edit = QtWidgets.QLineEdit("778nm")
        self.fast_channel_span_edit = QtWidgets.QLineEdit("8nm")
        self.fast_channel_sensitivity_combo = QtWidgets.QComboBox()
        self.fast_channel_sensitivity_combo.addItems(
            ["NORM", "MID", "HIGH1", "HIGH2", "HIGH3"]
        )
        self.fast_channel_sensitivity_combo.setCurrentText("HIGH3")
        self.fast_channel_ref_level_edit = QtWidgets.QLineEdit("10uW")
        self.fast_channel_sampling_edit = QtWidgets.QLineEdit("AUTO")
        self.fast_channel_avg_nm_spin = self._double_spin(0.0, 50.0, 0.1, " nm", 3)
        self.fast_channel_avg_nm_spin.setToolTip(
            "Intensity averaging window around each channel wavelength. 0 uses "
            "nearest OSA samples."
        )
        self.fast_channel_level_start_spin = self._spin(0, 1023, 0)
        self.fast_channel_level_stop_spin = self._spin(0, 1023, 1023)
        self.fast_channel_level_step_spin = self._spin(1, 1023, 32)
        scan_grid.addWidget(QtWidgets.QLabel("OSA center"), 0, 0)
        scan_grid.addWidget(self.fast_channel_center_edit, 0, 1)
        scan_grid.addWidget(QtWidgets.QLabel("Span"), 0, 2)
        scan_grid.addWidget(self.fast_channel_span_edit, 0, 3)
        scan_grid.addWidget(QtWidgets.QLabel("Sensitivity"), 0, 4)
        scan_grid.addWidget(self.fast_channel_sensitivity_combo, 0, 5)
        scan_grid.addWidget(QtWidgets.QLabel("Ref level"), 1, 0)
        scan_grid.addWidget(self.fast_channel_ref_level_edit, 1, 1)
        scan_grid.addWidget(QtWidgets.QLabel("Sampling"), 1, 2)
        scan_grid.addWidget(self.fast_channel_sampling_edit, 1, 3)
        scan_grid.addWidget(QtWidgets.QLabel("Avg window"), 1, 4)
        scan_grid.addWidget(self.fast_channel_avg_nm_spin, 1, 5)
        scan_grid.addWidget(QtWidgets.QLabel("Levels"), 2, 0)
        scan_grid.addWidget(self.fast_channel_level_start_spin, 2, 1)
        scan_grid.addWidget(self.fast_channel_level_stop_spin, 2, 2)
        scan_grid.addWidget(QtWidgets.QLabel("step"), 2, 3)
        scan_grid.addWidget(self.fast_channel_level_step_spin, 2, 4)
        layout.addWidget(scan)

        out = self._panel("Output")
        out_grid = QtWidgets.QGridLayout(out)
        self.fast_channel_json_edit = QtWidgets.QLineEdit("calib_fast_channels.json")
        self.fast_channel_json_button = QtWidgets.QPushButton("Browse")
        self.fast_channel_json_button.clicked.connect(
            lambda: self._browse_save_into(
                self.fast_channel_json_edit,
                "calib_fast_channels.json",
                "JSON Files (*.json)",
            )
        )
        self.fast_channel_csv_edit = QtWidgets.QLineEdit("calibration_fast_channels.csv")
        self.fast_channel_csv_button = QtWidgets.QPushButton("Browse")
        self.fast_channel_csv_button.clicked.connect(
            lambda: self._browse_save_into(
                self.fast_channel_csv_edit,
                "calibration_fast_channels.csv",
                "CSV Files (*.csv)",
            )
        )
        out_grid.addWidget(QtWidgets.QLabel("Output JSON"), 0, 0)
        out_grid.addWidget(self.fast_channel_json_edit, 0, 1)
        out_grid.addWidget(self.fast_channel_json_button, 0, 2)
        out_grid.addWidget(QtWidgets.QLabel("Output CSV"), 1, 0)
        out_grid.addWidget(self.fast_channel_csv_edit, 1, 1)
        out_grid.addWidget(self.fast_channel_csv_button, 1, 2)
        out_grid.setColumnStretch(1, 1)
        layout.addWidget(out)

        self.fast_channel_status_label = QtWidgets.QLabel("Ready")
        self.fast_channel_run_button = QtWidgets.QPushButton("Run Fast Channel Calibration")
        self.fast_channel_run_button.setEnabled(False)
        self.fast_channel_run_button.clicked.connect(self._run_fast_channel_calibration)
        self.fast_channel_stop_button = QtWidgets.QPushButton("Stop")
        self.fast_channel_stop_button.setProperty("variant", "danger")
        self.fast_channel_stop_button.setEnabled(False)
        self.fast_channel_stop_button.clicked.connect(self._stop_full_calibration)
        action = QtWidgets.QHBoxLayout()
        action.addWidget(self.fast_channel_status_label, 1)
        action.addWidget(self.fast_channel_run_button)
        action.addWidget(self.fast_channel_stop_button)
        layout.addLayout(action)
        layout.addStretch(1)
        self._toggle_fast_channel_source()
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

    def _default_calib_name(self, step: int, suffix: str = ".json") -> str:
        """Default calibration output filename, e.g. calib_step1_0704_1530.json.

        The MMDD_HHMM timestamp keeps successive runs from overwriting each
        other. The `calib_step` prefix is preserved so the encoder's
        auto-discovery (_CALIB_RE) still matches the file.
        """
        return f"calib_step{step}_{time.strftime('%m%d_%H%M')}{suffix}"

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
        layout.addWidget(self._output_row(1, "out", "Output JSON", self._default_calib_name(1), False))
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

        layout.addWidget(self._output_row(2, "out", "Output JSON", self._default_calib_name(2), False))
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

        layout.addWidget(self._output_row(3, "out", "Output JSON", self._default_calib_name(3), False))
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
        # calibration source + the shortened channel/pitch/pad summary share one
        # row (right after the loaded-json label) so the panel needs no third row
        cfg_grid.addWidget(self.enc_calib_label,             1, 0, 1, 3)
        cfg_grid.addWidget(self.enc_layout_status,            1, 3, 1, 2)
        cfg_grid.addWidget(enc_reload,                        1, 5)
        cfg_grid.addWidget(self.enc_build_button,             1, 6)

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

        # SLM pane and the instrument monitor (scope or DAQ) split evenly
        main_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_split.addWidget(left)
        main_split.addWidget(self._build_monitor_widget())
        main_split.setStretchFactor(0, 1)
        main_split.setStretchFactor(1, 1)
        main_split.setSizes([630, 630])

        # --- single Feedback log for both SLM and the instrument monitor, spanning full width ---
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
        """Instrument-monitor feedback shares the encoder's merged Feedback log."""
        self._enc_log(f"[monitor] {message}")

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
            f"pitch {pitch} px  |  max {max_ch} ch/side"
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
            f"{n_ch} ch/side  |  pitch {layout.pitch_px} px  |  "
            f"pad {layout.pitch_px - layout.channel_width_px} px"
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
                col_ratio=self._active_col_ratio(),
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
            if self._enc_should_read_monitor():
                self._enc_read_monitor_after_send()   # keeps Send disabled until read done
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

    def _enc_active_monitor(self) -> tuple[str, ScopeController | DAQController] | None:
        """Return ('scope', ctrl) or ('daq', ctrl) for whichever is connected.

        Scope takes priority if somehow both are connected, but
        _on_scope_connected / _on_daq_connected keep that from happening.
        """
        if self.scope_controller is not None and self.scope_controller.is_connected:
            return "scope", self.scope_controller
        if self.daq_controller is not None and self.daq_controller.is_connected:
            return "daq", self.daq_controller
        return None

    def _enc_should_read_monitor(self) -> bool:
        """Take a reading after a send only if it's safe/enabled."""
        return (
            self._enc_active_monitor() is not None
            and self.mon_read_on_send.isChecked()
            and self.monitor_stop_event is None   # not already running the trigger loop
        )

    def _enc_acquire_sample(self, *, label: str, on_finish=None) -> None:
        """Read one averaged (mean+std) sample from the connected instrument and
        append it to the record.

        Shared by the manual Acquire button and the auto-read-after-send path.
        ``on_finish`` runs on the GUI thread once the read settles (ok or err) --
        e.g. to re-enable whichever button kicked it off.
        """
        def done() -> None:
            if on_finish is not None:
                on_finish()

        active = self._enc_active_monitor()
        if active is None:
            self._mon_status("No instrument connected (connect Scope or DAQ).")
            done()
            return
        kind, ctrl = active
        self._mon_status(f"Reading {kind} ({label})…")

        if kind == "scope":
            # AUTO free-run with no armed edge: the SINGle self-triggers and
            # completes right away (the earlier timeout was a stale
            # ACQuire:COUNt, now forced to 1 in configure_monitor).
            settings = self._monitor_settings(trigger_mode="AUTO")
            read_timeout = max(30.0, settings.duration * 3.0 + 10.0)

            def work() -> MonitorSample | None:
                ctrl.configure_monitor(settings)
                time.sleep(settings.hold)          # settle before the read
                return ctrl.monitor_cycle(
                    index=len(self._monitor_values), timeout=read_timeout
                )
        else:
            settings = self._daq_monitor_settings()
            read_timeout = max(30.0, settings.duration * 3.0 + 10.0)

            def work() -> MonitorSample | None:
                # DAQController.monitor_cycle() sleeps settings.hold itself.
                ctrl.configure_monitor(settings)
                return ctrl.monitor_cycle(
                    index=len(self._monitor_values), timeout=read_timeout
                )

        def ok(sample: MonitorSample | None) -> None:
            if sample is not None:
                self._on_monitor_sample(sample)
            else:
                self._mon_status(f"{kind.capitalize()} read returned nothing.")
            done()

        def err(_error: str) -> None:
            self._mon_status(f"{kind.capitalize()} read failed (see Status log).")
            done()

        self._run_task(f"{kind.capitalize()} read ({label})", work, ok, err)

    def _enc_read_monitor_after_send(self) -> None:
        """After the pattern is displayed, read one sample and keep Send disabled
        until the read finishes."""
        active = self._enc_active_monitor()
        if active is None:
            self.enc_send_button.setEnabled(True)
            return
        self._enc_acquire_sample(
            label="on send",
            on_finish=lambda: self.enc_send_button.setEnabled(True),
        )

    def _enc_acquire_clicked(self) -> None:
        """Manual Acquire: read one sample now, without needing an SLM send."""
        if self._enc_active_monitor() is None:
            self._mon_status("No instrument connected (connect Scope or DAQ).")
            return
        if self.monitor_stop_event is not None:
            self._mon_status("A monitor loop is already running.")
            return
        self.mon_acquire_button.setEnabled(False)
        # _sync_monitor_source re-enables it only if an instrument is still connected
        self._enc_acquire_sample(label="manual", on_finish=self._sync_monitor_source)

    # ==================================================================
    # Shape page: global per-column encoding shape + OSA optimisation hook
    # ==================================================================

    def _build_edge_ratio_page(self) -> QtWidgets.QWidget:
        page = self._page_shell("Encoding Shape")

        subtitle = QtWidgets.QLabel(
            "Global per-column encoding shape, applied to every encoding step — "
            "data encoding, the Modulation Error and TPA calibrations, and the "
            "Quick Test all use it, so calibration is done with the same channel "
            "shape that is deployed. Column j encodes level_for(intensity command "
            "× ratio[j]), i.e. edge = ratio × (max − min) + min, where min is the "
            "channel's measured background. A 15 px channel defaults to the learned "
            "optimised shape (tapered edges); “All 1.0” reproduces the flat band. "
            "Build a layout on the TPA Encoding page first (channel width sets the "
            "number of columns)."
        )
        subtitle.setObjectName("PageSubtitle")
        subtitle.setWordWrap(True)
        page.layout().addWidget(subtitle)

        # --- master on/off: shape vs flat band -----------------------------
        self.shape_enabled_check = QtWidgets.QCheckBox(
            "Use encoding shape  (uncheck → flat band everywhere)"
        )
        self.shape_enabled_check.setChecked(True)
        self.shape_enabled_check.setToolTip(
            "Global switch. On: every encoding step uses the per-column shape "
            "below. Off: every step uses the flat band (col_ratio = None), as if "
            "the shape were all 1.0 — the table is kept but ignored."
        )
        self.shape_enabled_check.toggled.connect(self._edge_on_toggle)
        page.layout().addWidget(self.shape_enabled_check)

        # --- per-column ratio table (1 row, channel_width_px columns) ---
        self._edge_spins: list[WheelSpinBox] = []
        self.edge_width_label = QtWidgets.QLabel("Channel width: (no layout built)")
        self.edge_width_label.setObjectName("PageSubtitle")

        self.edge_table = QtWidgets.QTableWidget(1, 0)
        self.edge_table.verticalHeader().setVisible(False)
        # one data row of spin editors: pin the row height and give the widget a
        # fixed overall height (header + row + horizontal scrollbar) so a squeezed
        # splitter can never clip the single row.
        self.edge_table.verticalHeader().setDefaultSectionSize(34)
        self.edge_table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.edge_table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.edge_table.setFixedHeight(104)
        self.edge_table.setToolTip(
            "Ratio per column across the channel width (left→right). 1.0 = full "
            "value, 0.0 = channel's measured background."
        )

        edge_buttons = QtWidgets.QHBoxLayout()
        edge_opt = QtWidgets.QPushButton("Optimized shape")
        edge_opt.setProperty("variant", "ghost")
        edge_opt.setToolTip(
            "Load the learned optimised encoding shape (from best_so_far.json, "
            ">0.99 rounded to 1.0). Defined for a 15 px channel."
        )
        edge_opt.clicked.connect(self._edge_set_optimized)
        edge_all1 = QtWidgets.QPushButton("All 1.0")
        edge_all1.setProperty("variant", "ghost")
        edge_all1.setToolTip("Flat band (the trivial rectangular encoding)")
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
        edge_buttons.addWidget(edge_opt)
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
            "encoding shape — and report the per-channel change in neighbour "
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
        # the ratio panel holds a fixed-height table + buttons; keep it from being
        # collapsed so the single spin row is always fully visible
        split.setCollapsible(0, False)
        split.setSizes([200, 220, 320])
        page.layout().addWidget(split, 1)

        self._edge_draw_preview()
        return page

    def _edge_log(self, message: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.edge_log.appendPlainText(f"[{stamp}] {message}")

    def _default_col_ratio(self, width: int) -> np.ndarray:
        """Default per-column profile for a channel ``width`` px wide.

        Returns the learned :data:`OPTIMIZED_ENCODING_SHAPE` mirrored to the full
        channel width when the width matches the 15-px channel the shape was
        trained on; any other width falls back to the flat band (all 1.0).
        """
        expected = OPTIMIZED_ENCODING_SHAPE.size * 2 - 1
        if int(width) == expected:
            return mirror_intensity_profile(OPTIMIZED_ENCODING_SHAPE, int(width))
        return np.ones(int(width), dtype=float)

    def _edge_sync_layout(self, layout: ChannelLayout) -> None:
        """Rebuild the ratio table to the layout's channel width.

        Fresh columns start at the learned optimised encoding shape (the flat
        band for widths other than 15 px). Overlapping columns keep their
        previous values so tweaking the layout does not silently discard a tuned
        profile.
        """
        width = int(layout.channel_width_px)
        prev = self._edge_get_ratio()
        default = self._default_col_ratio(width)
        self.edge_table.clear()
        self.edge_table.setColumnCount(width)
        self.edge_table.setHorizontalHeaderLabels([str(j) for j in range(width)])
        self._edge_spins = []
        for j in range(width):
            spin = WheelSpinBox(wheel_step=self._enc_wheel_step)
            spin.setRange(0.0, 1.0)
            spin.setSingleStep(0.01)
            spin.setDecimals(3)
            value = float(prev[j]) if prev is not None and j < len(prev) else float(default[j])
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
        self._qt_sync_layout(layout)

    def _edge_get_ratio(self) -> np.ndarray | None:
        """Current per-column ratio profile, or None when no layout is built."""
        spins = getattr(self, "_edge_spins", None)
        if not spins:
            return None
        return np.array([s.value() for s in spins], dtype=float)

    def _active_col_ratio(self) -> np.ndarray | None:
        """The global encoding shape applied to every encoding step.

        Single source of truth taken from the Shape page: data encoding, the
        Modulation Error and TPA calibrations, and the Quick Test all read it, so
        calibration is performed with the same channel shape that is deployed.
        Returns ``None`` (the flat band) when the shape toggle is off or no
        layout/profile has been built.
        """
        toggle = getattr(self, "shape_enabled_check", None)
        if toggle is not None and not toggle.isChecked():
            return None
        return self._edge_get_ratio()

    def _edge_on_toggle(self, checked: bool) -> None:
        """Master shape switch: grey the table when off and redraw the preview."""
        if hasattr(self, "edge_table"):
            self.edge_table.setEnabled(checked)
        self._edge_draw_preview()
        self._edge_log(
            "Encoding shape ON — applied to every encoding step."
            if checked else
            "Encoding shape OFF — flat band used everywhere (table kept, ignored)."
        )

    def _edge_set_all(self, value: float) -> None:
        for spin in getattr(self, "_edge_spins", []):
            spin.setValue(float(value))

    def _edge_set_optimized(self) -> None:
        """Fill the ratio table with the learned optimised encoding shape."""
        spins = getattr(self, "_edge_spins", [])
        width = len(spins)
        if width == 0:
            self._edge_log("Build a layout on the TPA Encoding page first.")
            return
        ratio = self._default_col_ratio(width)
        if width != OPTIMIZED_ENCODING_SHAPE.size * 2 - 1:
            self._edge_log(
                f"Optimised shape is defined for a 15 px channel; width {width} "
                "px falls back to the flat band."
            )
        for spin, r in zip(spins, ratio):
            spin.setValue(float(r))
        self._edge_log("Loaded optimised encoding shape into the per-column ratio.")

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
        toggle = getattr(self, "shape_enabled_check", None)
        if toggle is not None and not toggle.isChecked():
            ax.set_title("SHAPE OFF — flat band in use", color="#bf616a", fontsize=9)
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
    # Quick Test page: A/B crosstalk (flat vs optimised encoding shape)
    # ==================================================================

    def _build_quick_test_page(self) -> QtWidgets.QWidget:
        page = self._page_shell("Quick Test · Encoding-shape crosstalk")
        subtitle = QtWidgets.QLabel(
            "Turn on a single channel and sweep the OSA once with the flat band and "
            "once with the optimised encoding shape, then compute the crosstalk into "
            "the neighbour bands from each trace. Build a 15 px layout on the TPA "
            "Encoding page first. OSA sweep settings (span, sensitivity, Y-unit) are "
            "taken from the Modulation Error page."
        )
        subtitle.setObjectName("PageSubtitle")
        subtitle.setWordWrap(True)
        page.layout().addWidget(subtitle)

        # --- calibration source (Quick-Test-local layout) ---
        src = self._panel("Calibration source")
        src_v = QtWidgets.QVBoxLayout(src)
        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(QtWidgets.QLabel("Source"))
        self.qt_calib_mode = QtWidgets.QComboBox()
        self.qt_calib_mode.addItems([
            "TPA Encoding page (built layout)",
            "Step 3 file (complete)",
            "Step 1 + Step 2 files (coarse)",
        ])
        self.qt_calib_mode.setToolTip(
            "Where this page's channel layout comes from. A Step 3 file carries the "
            "full measured intensity curves; Step 1 + Step 2 has no intensity data, "
            "so a coarse linear curve is synthesised from Step 1's min/max levels."
        )
        self.qt_calib_mode.currentIndexChanged.connect(self._qt_calib_mode_changed)
        mode_row.addWidget(self.qt_calib_mode, 1)
        src_v.addLayout(mode_row)

        json_filt = "Calibration JSON (*.json);;All files (*)"
        self.qt_calib_s3_edit = QtWidgets.QLineEdit()
        self.qt_calib_s3_edit.setPlaceholderText("calib_step3.json  (coordinates + intensity)")
        self.qt_calib_s3_row = self._qt_file_row(
            "Step 3 file", self.qt_calib_s3_edit,
            lambda: self._browse_open_into(
                self.qt_calib_s3_edit, "Open Step 3 calibration", json_filt),
        )
        src_v.addWidget(self.qt_calib_s3_row)

        self.qt_calib_s1_edit = QtWidgets.QLineEdit()
        self.qt_calib_s1_edit.setPlaceholderText("calib_step1.json  (min/max levels)")
        self.qt_calib_s2_edit = QtWidgets.QLineEdit()
        self.qt_calib_s2_edit.setPlaceholderText("calib_step2.json  (wavelength map)")
        self.qt_calib_s12_row = QtWidgets.QWidget()
        s12v = QtWidgets.QVBoxLayout(self.qt_calib_s12_row)
        s12v.setContentsMargins(0, 0, 0, 0)
        s12v.addWidget(self._qt_file_row(
            "Step 1 (min/max)", self.qt_calib_s1_edit,
            lambda: self._browse_open_into(
                self.qt_calib_s1_edit, "Open Step 1 calibration", json_filt)))
        s12v.addWidget(self._qt_file_row(
            "Step 2 (wavelength)", self.qt_calib_s2_edit,
            lambda: self._browse_open_into(
                self.qt_calib_s2_edit, "Open Step 2 calibration", json_filt)))
        src_v.addWidget(self.qt_calib_s12_row)

        # layout geometry, used only when building from files
        self.qt_calib_params = QtWidgets.QWidget()
        pv = QtWidgets.QHBoxLayout(self.qt_calib_params)
        pv.setContentsMargins(0, 0, 0, 0)
        self.qt_calib_center = self._double_spin(700.0, 900.0, 778.0, " nm", 2)
        self.qt_calib_width = self._spin(1, 256, 15)
        self.qt_calib_pad = self._spin(0, 64, 5)
        pv.addWidget(QtWidgets.QLabel("Center λ")); pv.addWidget(self.qt_calib_center)
        pv.addWidget(QtWidgets.QLabel("Width px")); pv.addWidget(self.qt_calib_width)
        pv.addWidget(QtWidgets.QLabel("Pad px")); pv.addWidget(self.qt_calib_pad)
        pv.addStretch(1)
        src_v.addWidget(self.qt_calib_params)

        build_row = QtWidgets.QHBoxLayout()
        self.qt_calib_build_btn = QtWidgets.QPushButton("Build layout from calibration")
        self.qt_calib_build_btn.clicked.connect(self._qt_build_layout)
        build_row.addWidget(self.qt_calib_build_btn)
        build_row.addStretch(1)
        src_v.addLayout(build_row)

        self.qt_calib_label = QtWidgets.QLabel(
            "Using the layout built on the TPA Encoding page."
        )
        self.qt_calib_label.setObjectName("PageSubtitle")
        self.qt_calib_label.setWordWrap(True)
        src_v.addWidget(self.qt_calib_label)
        page.layout().addWidget(src)

        # --- test target controls ---
        cfg = self._panel("Test Target")
        grid = QtWidgets.QGridLayout(cfg)
        self.qt_channel_combo = QtWidgets.QComboBox()
        self.qt_channel_combo.setToolTip(
            "Pick which encoding channel to probe. The list is filled from the "
            "layout built on the TPA Encoding page, sorted by wavelength."
        )
        self.qt_channel_combo.setMinimumWidth(240)
        self.qt_averages = self._spin(1, 20, 1)
        self.qt_bg_check = QtWidgets.QCheckBox("Subtract background")
        self.qt_bg_check.setChecked(True)
        self.qt_bg_check.setToolTip(
            "Take an all-off trace at the channel centre and subtract it for a "
            "cleaner low-level crosstalk floor"
        )
        grid.addWidget(QtWidgets.QLabel("Channel"), 0, 0)
        grid.addWidget(self.qt_channel_combo, 0, 1, 1, 3)
        grid.addWidget(QtWidgets.QLabel("Averages"), 0, 4)
        grid.addWidget(self.qt_averages, 0, 5)
        grid.addWidget(self.qt_bg_check, 1, 0, 1, 4)
        page.layout().addWidget(cfg)

        self.qt_layout_label = QtWidgets.QLabel(
            "Grid: (build a layout on the TPA Encoding page)"
        )
        self.qt_layout_label.setObjectName("PageSubtitle")
        self.qt_layout_label.setWordWrap(True)
        page.layout().addWidget(self.qt_layout_label)

        # --- run controls ---
        run_row = QtWidgets.QHBoxLayout()
        self.qt_run_button = QtWidgets.QPushButton("Run A/B crosstalk test")
        self.qt_run_button.clicked.connect(self._qt_run)
        self.qt_stop_button = QtWidgets.QPushButton("Stop")
        self.qt_stop_button.setProperty("variant", "danger")
        self.qt_stop_button.setEnabled(False)
        self.qt_stop_button.clicked.connect(self._qt_stop)
        self.qt_save_button = QtWidgets.QPushButton("Save test CSV…")
        self.qt_save_button.setProperty("variant", "ghost")
        self.qt_save_button.setEnabled(False)
        self.qt_save_button.clicked.connect(self._qt_save)
        run_row.addWidget(self.qt_run_button)
        run_row.addWidget(self.qt_stop_button)
        run_row.addWidget(self.qt_save_button)
        run_row.addStretch(1)

        self.qt_bar = QtWidgets.QProgressBar()
        self.qt_bar.setValue(0)
        self.qt_status = QtWidgets.QLabel("\N{EN DASH}")
        self.qt_status.setWordWrap(True)

        # --- results table (metric | flat | optimised | Δ) ---
        self.qt_table = QtWidgets.QTableWidget(0, 4)
        self.qt_table.setHorizontalHeaderLabels(["Metric", "Flat", "Optimized", "Δ"])
        self.qt_table.verticalHeader().setVisible(False)
        self.qt_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.qt_table.setAlternatingRowColors(True)
        self.qt_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch
        )
        # tall enough to show every metric row without its own inner scrollbar
        self.qt_table.setMinimumHeight(300)
        self.qt_table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        # --- overlaid spectra plot ---
        self.qt_figure = Figure(figsize=(10, 2.8), tight_layout=True)
        self.qt_canvas = FigureCanvas(self.qt_figure)
        self.qt_canvas.setMinimumHeight(260)

        results = self._panel("A/B crosstalk from OSA data")
        results_layout = QtWidgets.QVBoxLayout(results)
        results_layout.addLayout(run_row)
        results_layout.addWidget(self.qt_bar)
        results_layout.addWidget(self.qt_status)
        results_layout.addWidget(self.qt_table)
        results_layout.addWidget(self.qt_canvas, 1)
        page.layout().addWidget(results, 1)

        self._qt_draw(None, None)
        self._qt_calib_mode_changed()  # set initial file-row visibility

        # The page is tall (calibration · target · run controls · table · plot), so
        # wrap it in a scroll area — on short windows the plot at the bottom stays
        # reachable instead of being squeezed to nothing.
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setWidget(page)
        return scroll

    def _qt_file_row(
        self, label_text: str, edit: QtWidgets.QLineEdit, browse_slot
    ) -> QtWidgets.QWidget:
        """label + line-edit + Browse, packed into one widget for show/hide."""
        row = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        lbl = QtWidgets.QLabel(label_text)
        lbl.setMinimumWidth(120)
        btn = QtWidgets.QPushButton("Browse")
        btn.setProperty("variant", "ghost")
        btn.clicked.connect(browse_slot)
        h.addWidget(lbl)
        h.addWidget(edit, 1)
        h.addWidget(btn)
        return row

    def _qt_active_layout(self) -> ChannelLayout | None:
        """Layout the Quick Test uses: the TPA Encoding one, or a picked-calib one."""
        if self.qt_calib_mode.currentIndex() == 0:
            return self.encoding_layout
        return self.qt_layout

    def _qt_sync_layout(self, layout: ChannelLayout) -> None:
        """Called when the TPA Encoding page builds a layout; mirror it only if
        Quick Test is set to follow that page."""
        if getattr(self, "qt_calib_mode", None) is None:
            return
        if self.qt_calib_mode.currentIndex() == 0:
            self._qt_refresh_channels()

    def _qt_refresh_channels(self) -> None:
        """Repopulate the channel picker + grid label from the active layout."""
        combo = getattr(self, "qt_channel_combo", None)
        if combo is None:
            return
        layout = self._qt_active_layout()
        prev = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        if layout is not None:
            for ch in sorted(layout.all_channels, key=lambda c: c.wavelength_nm):
                combo.addItem(
                    f"{ch.side}[{ch.index}]  ·  {ch.wavelength_nm:.3f} nm",
                    (ch.side, ch.index),
                )
            if prev is not None:
                for i in range(combo.count()):
                    if combo.itemData(i) == prev:
                        combo.setCurrentIndex(i)
                        break
        combo.blockSignals(False)
        if layout is None:
            self.qt_layout_label.setText(
                "Grid: (no layout — pick a calibration source above or build one "
                "on the TPA Encoding page)"
            )
            return
        note = "" if layout.channel_width_px == 15 else "  (not 15 px — shape is flat)"
        self.qt_layout_label.setText(
            f"Grid: {layout.n_channels} channels/side · {layout.channel_width_px} px "
            f"wide · center {layout.center_wl:.3f} nm{note}"
        )

    def _qt_calib_mode_changed(self, *_args) -> None:
        mode = self.qt_calib_mode.currentIndex()
        self.qt_calib_s3_row.setVisible(mode == 1)
        self.qt_calib_s12_row.setVisible(mode == 2)
        file_mode = mode in (1, 2)
        self.qt_calib_params.setVisible(file_mode)
        self.qt_calib_build_btn.setVisible(file_mode)
        if mode == 0:
            self.qt_calib_label.setText(
                "Using the layout built on the TPA Encoding page."
            )
        self._qt_refresh_channels()

    @staticmethod
    def _synth_calib_from_min_max(
        step1: CalibrationResult, step2: CalibrationResult
    ) -> CalibrationResult:
        """Coarse calibration from Step 1 (min/max) + Step 2 (wavelength map).

        Step 1/Step 2 carry no per-coordinate intensity curves, so every
        coordinate is given the *same* synthetic linear transfer curve rising
        from Step 1's min level to its max level. The encoder then maps power
        linearly onto SLM level (much coarser than a real Step 3 sweep).
        """
        coords = np.asarray(step2.coordinates, dtype=float)
        wls = np.asarray(step2.wavelength, dtype=float)
        if coords.size == 0 or wls.size == 0:
            raise ValueError("Step 2 file has no coordinate → wavelength map")
        try:
            lo = int(np.asarray(step1.min_level).flat[0])
            hi = int(np.asarray(step1.max_level).flat[0])
        except (ValueError, IndexError, TypeError):
            raise ValueError("Step 1 file has no min/max levels")
        if hi <= lo:
            raise ValueError("Step 1 max_level must exceed min_level")
        levels = np.arange(lo, hi + 1, dtype=int)
        ramp = np.linspace(0.0, 1.0, levels.size)
        intensity = np.tile(ramp, (coords.size, 1))
        return CalibrationResult(
            wavelength=wls, coordinates=coords,
            max_level=hi, min_level=lo,
            level_range=levels, intensity_levels=intensity,
        )

    def _layout_from_calib(
        self, calib: CalibrationResult, *,
        center_wl: float, channel_width_px: int, gap_px: int,
    ) -> ChannelLayout:
        """Build a channel layout from a calibration, sizing n_channels to fit."""
        if calib is None or calib.intensity_levels is None:
            raise ValueError("calibration has no intensity data")
        coords = np.asarray(calib.coordinates, dtype=float)
        wls = np.asarray(calib.wavelength, dtype=float)
        if coords.size == 0 or wls.size == 0:
            raise ValueError("calibration has no coordinate → wavelength map")
        a, b = np.polyfit(coords, wls, 1)
        cx = (center_wl - b) / a
        pitch = channel_width_px + gap_px
        n_ch = int(min(cx - coords.min(), coords.max() - cx) / pitch)
        if n_ch < 1:
            raise ValueError(
                "pitch too large, or centre wavelength outside the calibrated range"
            )
        return build_channel_layout(
            calib, n_channels=n_ch, channel_width_px=channel_width_px,
            gap_px=gap_px, center_wl=center_wl,
        )

    def _qt_build_layout(self) -> None:
        mode = self.qt_calib_mode.currentIndex()
        try:
            if mode == 1:
                path = self.qt_calib_s3_edit.text().strip()
                if not path:
                    raise ValueError("choose a Step 3 calibration file")
                calib = load_calibration_result(path)
                if calib.intensity_levels is None:
                    raise ValueError(
                        "that file has no intensity_levels — it is not a Step 3 result"
                    )
            elif mode == 2:
                p1 = self.qt_calib_s1_edit.text().strip()
                p2 = self.qt_calib_s2_edit.text().strip()
                if not p1 or not p2:
                    raise ValueError(
                        "choose both a Step 1 (min/max) and a Step 2 (wavelength) file"
                    )
                calib = self._synth_calib_from_min_max(
                    load_calibration_result(p1), load_calibration_result(p2)
                )
            else:
                return
            layout = self._layout_from_calib(
                calib,
                center_wl=self.qt_calib_center.value(),
                channel_width_px=self.qt_calib_width.value(),
                gap_px=self.qt_calib_pad.value(),
            )
        except Exception as exc:
            self.qt_layout = None
            self.qt_calib_label.setText(f"Layout error: {exc}")
            self._log(f"Quick test calibration load failed: {exc}")
            self._qt_refresh_channels()
            return
        self.qt_layout = layout
        src = "Step 3 file" if mode == 1 else "Step 1 + Step 2 (coarse)"
        self.qt_calib_label.setText(
            f"{src}: {layout.n_channels} ch/side · {layout.channel_width_px} px · "
            f"center {layout.center_wl:.3f} nm — layout ready."
        )
        self._log(
            f"Quick test layout built from {src.lower()}: {layout.n_channels} ch/side."
        )
        self._qt_refresh_channels()

    def _qt_set_running(self, running: bool) -> None:
        self.qt_run_button.setEnabled(not running)
        self.qt_stop_button.setEnabled(running)
        self.qt_save_button.setEnabled(not running and self._qt_test is not None)

    def _qt_run(self) -> None:
        layout = self._qt_active_layout()
        if layout is None:
            if self.qt_calib_mode.currentIndex() == 0:
                self.qt_status.setText(
                    "No channel grid — build one on the TPA Encoding page, or pick a "
                    "calibration file above."
                )
            else:
                self.qt_status.setText(
                    "No layout — click 'Build layout from calibration' above first."
                )
            return
        osa = self._osa_ready()
        if osa is None:
            self.qt_status.setText("Connect the OSA (Connections page) first.")
            return
        controller = self._controller()
        if not getattr(controller, "is_open", False):
            self.qt_status.setText("Open the SLM (Connections page) first.")
            return
        data = self.qt_channel_combo.currentData()
        if data is None:
            self.qt_status.setText(
                "No channel selected — build a layout on the TPA Encoding page."
            )
            return
        side, index = data
        if side not in ("x", "w") or index >= layout.n_channels:
            self.qt_status.setText(
                f"Channel {side}[{index}] is out of range for this layout — "
                "rebuild the grid on the TPA Encoding page."
            )
            return
        opt_ratio = self._default_col_ratio(layout.channel_width_px)
        if layout.channel_width_px != 15:
            self.qt_status.setText(
                "Layout is not 15 px wide — the optimised shape is unavailable, so "
                "both arms would be flat. Build a 15 px layout first."
            )
            return

        settings = self._ana_settings()
        averages = self.qt_averages.value()
        subtract_bg = self.qt_bg_check.isChecked()
        stop_event = threading.Event()
        self.qt_test_stop_event = stop_event
        self._qt_set_running(True)
        self.qt_bar.setRange(0, 2)
        self.qt_bar.setValue(0)
        self.qt_status.setText(f"Measuring {side}[{index}] — flat baseline…")

        def measure(col_ratio):
            return measure_one_channel(
                osa, controller, layout, settings,
                side=side, index=index, averages=averages,
                subtract_background=subtract_bg, col_ratio=col_ratio,
                stop_event=stop_event,
            )

        def work() -> dict[str, Any]:
            self.qt_test_progress.emit(0, 2, f"{side}[{index}] flat baseline…")
            flat = measure(None)
            if stop_event.is_set():
                return {"status": "aborted"}
            self.qt_test_progress.emit(1, 2, f"{side}[{index}] optimised shape…")
            optimized = measure(opt_ratio)
            if stop_event.is_set():
                return {"status": "aborted"}
            self.qt_test_progress.emit(2, 2, "computing crosstalk…")
            return {"status": "ok", "flat": flat, "optimized": optimized}

        self._run_slm_task(
            "Quick crosstalk A/B test", work, self._qt_finished, self._qt_error
        )

    def _qt_stop(self) -> None:
        if self.qt_test_stop_event is not None:
            self.qt_test_stop_event.set()
            self.qt_status.setText("Stopping…")

    def _qt_test_progress(self, done: int, total: int, message: str) -> None:
        self.qt_bar.setMaximum(total)
        self.qt_bar.setValue(done)
        self.qt_status.setText(f"[{done}/{total}] {message}")

    def _qt_finished(self, payload: dict[str, Any]) -> None:
        self.qt_test_stop_event = None
        self._qt_set_running(False)
        if payload.get("status") == "aborted":
            self.qt_bar.setValue(0)
            self.qt_status.setText("Stopped — partial test discarded.")
            return
        flat = payload["flat"]
        optimized = payload["optimized"]
        self._qt_test = {"flat": flat, "optimized": optimized}
        self.qt_bar.setValue(self.qt_bar.maximum())
        self._qt_populate(flat, optimized)
        self._qt_draw(flat, optimized)
        self.qt_save_button.setEnabled(True)

    def _qt_error(self, _error: str) -> None:
        self.qt_test_stop_event = None
        self._qt_set_running(False)
        self.qt_bar.setValue(0)
        self.qt_status.setText("Quick test failed (see Status log).")

    @staticmethod
    def _qt_metric_rows(
        flat: ChannelSpectrum, optimized: ChannelSpectrum
    ) -> list[tuple[str, str, str, str]]:
        """Comparison rows: (metric label, flat, optimised, Δ). Δ in pp for %."""
        def pp(a: float, b: float) -> str:
            return f"{(b - a) * 100:+.3f}"

        return [
            ("Peak λ (nm)", f"{flat.peak_wl_nm:.4f}", f"{optimized.peak_wl_nm:.4f}",
             f"{optimized.peak_wl_nm - flat.peak_wl_nm:+.4f}"),
            ("FWHM (nm)", f"{flat.fwhm_nm:.4f}", f"{optimized.fwhm_nm:.4f}",
             f"{optimized.fwhm_nm - flat.fwhm_nm:+.4f}"),
            ("In-band %", f"{flat.in_band_fraction * 100:.2f}",
             f"{optimized.in_band_fraction * 100:.2f}",
             pp(flat.in_band_fraction, optimized.in_band_fraction)),
            ("Neighbour leak ±1 %", f"{flat.neighbor_leakage * 100:.3f}",
             f"{optimized.neighbor_leakage * 100:.3f}",
             pp(flat.neighbor_leakage, optimized.neighbor_leakage)),
            ("Total crosstalk %", f"{flat.total_crosstalk * 100:.3f}",
             f"{optimized.total_crosstalk * 100:.3f}",
             pp(flat.total_crosstalk, optimized.total_crosstalk)),
            ("xtalk −1 %", f"{flat.crosstalk.get(-1, 0.0) * 100:.3f}",
             f"{optimized.crosstalk.get(-1, 0.0) * 100:.3f}",
             pp(flat.crosstalk.get(-1, 0.0), optimized.crosstalk.get(-1, 0.0))),
            ("xtalk +1 %", f"{flat.crosstalk.get(1, 0.0) * 100:.3f}",
             f"{optimized.crosstalk.get(1, 0.0) * 100:.3f}",
             pp(flat.crosstalk.get(1, 0.0), optimized.crosstalk.get(1, 0.0))),
            ("xtalk −2 %", f"{flat.crosstalk.get(-2, 0.0) * 100:.3f}",
             f"{optimized.crosstalk.get(-2, 0.0) * 100:.3f}",
             pp(flat.crosstalk.get(-2, 0.0), optimized.crosstalk.get(-2, 0.0))),
            ("xtalk +2 %", f"{flat.crosstalk.get(2, 0.0) * 100:.3f}",
             f"{optimized.crosstalk.get(2, 0.0) * 100:.3f}",
             pp(flat.crosstalk.get(2, 0.0), optimized.crosstalk.get(2, 0.0))),
        ]

    def _qt_populate(self, flat: ChannelSpectrum, optimized: ChannelSpectrum) -> None:
        rows = self._qt_metric_rows(flat, optimized)
        self.qt_table.setRowCount(len(rows))
        for r, cells in enumerate(rows):
            for c, text in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(text)
                if c > 0:
                    item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.qt_table.setItem(r, c, item)
        d_total = optimized.total_crosstalk - flat.total_crosstalk
        verdict = "reduces" if d_total < 0 else "does not reduce"
        summary = (
            f"{optimized.side}[{optimized.index}] @ {flat.nominal_wl_nm:.3f} nm  ·  "
            f"total crosstalk {flat.total_crosstalk * 100:.3f}% → "
            f"{optimized.total_crosstalk * 100:.3f}% (Δ {d_total * 100:+.3f} pp)  ·  "
            f"in-band {flat.in_band_fraction * 100:.1f}% → "
            f"{optimized.in_band_fraction * 100:.1f}%"
        )
        self.qt_status.setText(f"Optimised shape {verdict} crosstalk — {summary}")

    def _qt_draw(
        self, flat: ChannelSpectrum | None, optimized: ChannelSpectrum | None
    ) -> None:
        self.qt_figure.clear()
        ax = self.qt_figure.add_subplot(111)
        if flat is None or optimized is None:
            ax.text(0.5, 0.5, "Run a test to compare spectra",
                    ha="center", va="center", color="#d8dee9", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        else:
            ax.plot(flat.wavelengths_nm, flat.signal_w * 1e6,
                    color="#88c0d0", linewidth=1.4, label="flat")
            ax.plot(optimized.wavelengths_nm, optimized.signal_w * 1e6,
                    color="#ebcb8b", linewidth=1.4, label="optimized")
            pitch_nm = self.encoding_layout.pitch_px * self.encoding_layout.nm_per_px \
                if self.encoding_layout is not None else 0.0
            for offset in (-2, -1, 1, 2):
                ax.axvline(flat.peak_wl_nm + offset * pitch_nm,
                           color="#4c566a", linewidth=0.8, linestyle=":")
            ax.set_xlabel("wavelength (nm)", color="#d8dee9", fontsize=8)
            ax.set_ylabel("power (µW)", color="#d8dee9", fontsize=8)
            ax.legend(loc="upper right", fontsize=8, framealpha=0.2)
            ax.tick_params(colors="#d8dee9", labelsize=7)
            for spine in ax.spines.values():
                spine.set_color("#41515c")
        self.qt_figure.patch.set_facecolor("#101820")
        ax.set_facecolor("#101820")
        self.qt_canvas.draw_idle()

    def _qt_save(self) -> None:
        if self._qt_test is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save quick test", "quick_crosstalk_test.csv", "CSV (*.csv)")
        if not path:
            return
        flat = self._qt_test["flat"]
        optimized = self._qt_test["optimized"]
        try:
            import csv as _csv

            with open(path, "w", encoding="utf-8", newline="") as handle:
                writer = _csv.writer(handle)
                writer.writerow(
                    ["channel", f"{optimized.side}{optimized.index}",
                     "nominal_wl_nm", f"{flat.nominal_wl_nm:.5f}"]
                )
                writer.writerow(["metric", "flat", "optimized", "delta"])
                for cells in self._qt_metric_rows(flat, optimized):
                    writer.writerow(cells)
            self._log(f"Saved quick test → {path}")
        except Exception as exc:
            self._log(f"Save failed: {exc}")

    # ==================================================================
    # OSA Viewer page: live spectrum viewer (single / continuous sweeps)
    # ==================================================================

    def _build_osa_viewer_page(self) -> QtWidgets.QWidget:
        page = self._page_shell("OSA Viewer")
        subtitle = QtWidgets.QLabel(
            "Live optical-spectrum viewer. Set the sweep parameters, then take a "
            "single sweep or run continuously. Connect the OSA on the Connections "
            "page first. Values use the instrument's unit-suffixed format "
            "(e.g. 778nm, 8nm, 10uW)."
        )
        subtitle.setObjectName("PageSubtitle")
        subtitle.setWordWrap(True)
        page.layout().addWidget(subtitle)

        # --- sweep parameters ---
        cfg = self._panel("Sweep Parameters")
        grid = QtWidgets.QGridLayout(cfg)
        self.osv_center = QtWidgets.QLineEdit("778nm")
        self.osv_center.setToolTip("Center wavelength (e.g. 778nm)")
        self.osv_span = QtWidgets.QLineEdit("8nm")
        self.osv_span.setToolTip("Span (e.g. 8nm, 0.8nm)")
        self.osv_sensitivity = QtWidgets.QComboBox()
        self.osv_sensitivity.addItems(["NORM", "MID", "HIGH1", "HIGH2", "HIGH3"])
        self.osv_sensitivity.setCurrentText("HIGH2")
        self.osv_points = QtWidgets.QLineEdit("1001")
        self.osv_points.setToolTip("Sampling points: AUTO or a count like 1001")
        self.osv_ref_level = QtWidgets.QLineEdit("10uW")
        self.osv_yunit = QtWidgets.QComboBox()
        self.osv_yunit.addItems(["LIN (W)", "LOG (dBm)"])
        self.osv_averages = self._spin(1, 50, 1)
        grid.addWidget(QtWidgets.QLabel("Center"), 0, 0)
        grid.addWidget(self.osv_center, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Span"), 0, 2)
        grid.addWidget(self.osv_span, 0, 3)
        grid.addWidget(QtWidgets.QLabel("Sensitivity"), 0, 4)
        grid.addWidget(self.osv_sensitivity, 0, 5)
        grid.addWidget(QtWidgets.QLabel("Points"), 1, 0)
        grid.addWidget(self.osv_points, 1, 1)
        grid.addWidget(QtWidgets.QLabel("Ref level"), 1, 2)
        grid.addWidget(self.osv_ref_level, 1, 3)
        grid.addWidget(QtWidgets.QLabel("Y unit"), 1, 4)
        grid.addWidget(self.osv_yunit, 1, 5)
        grid.addWidget(QtWidgets.QLabel("Averages"), 2, 0)
        grid.addWidget(self.osv_averages, 2, 1)
        page.layout().addWidget(cfg)

        # --- controls ---
        ctrl = QtWidgets.QHBoxLayout()
        self.osv_single_button = QtWidgets.QPushButton("Single sweep")
        self.osv_single_button.clicked.connect(self._osa_view_single)
        self.osv_cont_button = QtWidgets.QPushButton("Continuous")
        self.osv_cont_button.setCheckable(True)
        self.osv_cont_button.setToolTip("Sweep repeatedly until stopped")
        self.osv_cont_button.clicked.connect(self._osa_view_continuous)
        self.osv_stop_button = QtWidgets.QPushButton("Stop")
        self.osv_stop_button.setProperty("variant", "danger")
        self.osv_stop_button.setEnabled(False)
        self.osv_stop_button.clicked.connect(self._osa_view_stop)
        self.osv_save_button = QtWidgets.QPushButton("Save trace…")
        self.osv_save_button.setProperty("variant", "ghost")
        self.osv_save_button.setEnabled(False)
        self.osv_save_button.clicked.connect(self._osa_view_save)
        self.osv_logy_check = QtWidgets.QCheckBox("Log Y axis")
        self.osv_logy_check.setToolTip("Plot power on a log axis (LIN data only)")
        self.osv_logy_check.toggled.connect(
            lambda _=None: self._osa_view_plot(self.osa_view_trace)
        )
        ctrl.addWidget(self.osv_single_button)
        ctrl.addWidget(self.osv_cont_button)
        ctrl.addWidget(self.osv_stop_button)
        ctrl.addWidget(self.osv_save_button)
        ctrl.addWidget(self.osv_logy_check)
        ctrl.addStretch(1)
        page.layout().addLayout(ctrl)

        self.osv_status = QtWidgets.QLabel("\N{EN DASH}")
        self.osv_status.setWordWrap(True)
        page.layout().addWidget(self.osv_status)

        # --- spectrum plot ---
        self.osv_figure = Figure(figsize=(10, 4.2), tight_layout=True)
        self.osv_canvas = FigureCanvas(self.osv_figure)
        self.osv_canvas.setMinimumHeight(280)
        plot_panel = self._panel("Spectrum")
        plot_layout = QtWidgets.QVBoxLayout(plot_panel)
        plot_layout.addWidget(self.osv_canvas, 1)
        page.layout().addWidget(plot_panel, 1)

        self._osa_view_plot(None)
        return page

    def _osa_view_settings(self) -> MeasurementSettings:
        y_unit = "LOGarithmic" if self.osv_yunit.currentText().startswith("LOG") else "LINear"
        return MeasurementSettings(
            center_wl=self.osv_center.text().strip() or "778nm",
            span=self.osv_span.text().strip() or "8nm",
            sensitivity=self.osv_sensitivity.currentText(),
            sampling_points=self.osv_points.text().strip() or "AUTO",
            y_unit=y_unit,
            reference_level=self.osv_ref_level.text().strip() or "10uW",
        )

    def _osa_view_set_running(self, running: bool, *, continuous: bool = False) -> None:
        self.osv_single_button.setEnabled(not running)
        self.osv_cont_button.setChecked(running and continuous)
        self.osv_cont_button.setEnabled(not running or continuous)
        self.osv_stop_button.setEnabled(running)
        self.osv_save_button.setEnabled(not running and self.osa_view_trace is not None)

    def _osa_view_single(self) -> None:
        osa = self._osa_ready()
        if osa is None:
            self.osv_status.setText("Connect the OSA on the Connections page first.")
            return
        settings = self._osa_view_settings()
        averages = self.osv_averages.value()
        stop_event = threading.Event()
        self.osa_view_stop_event = stop_event
        self._osa_view_set_running(True)
        self.osv_status.setText("Sweeping…")

        def work() -> dict[str, Any]:
            try:
                trace = osa.measure(settings, averages=averages, stop_event=stop_event)
            except OSAError as exc:
                if stop_event.is_set():
                    return {"status": "aborted"}
                return {"status": "error", "message": str(exc)}
            return {"status": "ok", "trace": trace}

        self._run_task(
            "OSA single sweep", work, self._osa_view_single_done, self._osa_view_error
        )

    def _osa_view_single_done(self, payload: dict[str, Any]) -> None:
        self.osa_view_stop_event = None
        self._osa_view_set_running(False)
        status = payload.get("status")
        if status == "aborted":
            self.osv_status.setText("Stopped.")
            return
        if status == "error":
            self.osv_status.setText(f"Sweep failed: {payload.get('message', '')}")
            return
        self._osa_view_on_trace(payload["trace"])

    def _osa_view_continuous(self) -> None:
        if not self.osv_cont_button.isChecked():
            # toggled off by the user's click -> treat as stop
            self._osa_view_stop()
            return
        osa = self._osa_ready()
        if osa is None:
            self.osv_cont_button.setChecked(False)
            self.osv_status.setText("Connect the OSA on the Connections page first.")
            return
        settings = self._osa_view_settings()
        averages = self.osv_averages.value()
        stop_event = threading.Event()
        self.osa_view_stop_event = stop_event
        self._osa_view_set_running(True, continuous=True)
        self.osv_status.setText("Continuous sweeping… press Stop to end.")

        def work() -> dict[str, Any]:
            try:
                while not stop_event.is_set():
                    trace = osa.measure(
                        settings, averages=averages, stop_event=stop_event
                    )
                    if stop_event.is_set():
                        break
                    self.osa_trace_ready.emit(trace)
            except OSAError as exc:
                if not stop_event.is_set():
                    return {"status": "error", "message": str(exc)}
            return {"status": "stopped"}

        self._run_task(
            "OSA continuous sweep", work,
            self._osa_view_continuous_done, self._osa_view_error,
        )

    def _osa_view_continuous_done(self, payload: dict[str, Any]) -> None:
        self.osa_view_stop_event = None
        self._osa_view_set_running(False)
        if payload.get("status") == "error":
            self.osv_status.setText(f"Sweep failed: {payload.get('message', '')}")
        else:
            self.osv_status.setText("Continuous sweep stopped.")

    def _osa_view_error(self, _error: str) -> None:
        self.osa_view_stop_event = None
        self._osa_view_set_running(False)
        self.osv_status.setText("OSA sweep failed (see Status log).")

    def _osa_view_stop(self) -> None:
        if self.osa_view_stop_event is not None:
            self.osa_view_stop_event.set()
            self.osv_status.setText("Stopping…")

    def _osa_view_on_trace(self, trace) -> None:
        """Store and plot a freshly measured trace (GUI thread via signal)."""
        self.osa_view_trace = trace
        self.osv_save_button.setEnabled(self.osa_view_stop_event is None)
        self._osa_view_plot(trace)

    def _osa_view_plot(self, trace) -> None:
        self.osv_figure.clear()
        ax = self.osv_figure.add_subplot(111)
        if trace is None:
            ax.text(0.5, 0.5, "Take a sweep to display the spectrum",
                    ha="center", va="center", color="#d8dee9", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        else:
            wl = np.asarray(trace.wavelengths_nm, dtype=float)
            is_log = trace.power_label == "power_dBm"
            if is_log:
                y = np.asarray(trace.powers, dtype=float)
                ylabel = "power (dBm)"
            else:
                y = np.asarray(trace.powers, dtype=float) * 1e6
                ylabel = "power (µW)"
            ax.plot(wl, y, color="#88c0d0", linewidth=1.2)
            if wl.size and np.any(np.isfinite(y)):
                peak = int(np.nanargmax(y))
                ax.plot(wl[peak], y[peak], "o", color="#ebcb8b", markersize=5)
                unit = "dBm" if is_log else "µW"
                ax.annotate(f"{wl[peak]:.4f} nm\n{y[peak]:.3g} {unit}",
                            (wl[peak], y[peak]), color="#ebcb8b", fontsize=8,
                            xytext=(6, -2), textcoords="offset points")
                avg = f" · avg {trace.averages}" if trace.averages > 1 else ""
                self.osv_status.setText(
                    f"peak {y[peak]:.3g} {unit} @ {wl[peak]:.4f} nm  ·  "
                    f"{wl.size} pts{avg}"
                )
            if not is_log and self.osv_logy_check.isChecked():
                ax.set_yscale("log")
            ax.set_xlabel("wavelength (nm)", color="#d8dee9", fontsize=8)
            ax.set_ylabel(ylabel, color="#d8dee9", fontsize=8)
            ax.grid(True, color="#2a3540", linewidth=0.5)
            ax.tick_params(colors="#d8dee9", labelsize=7)
            for spine in ax.spines.values():
                spine.set_color("#41515c")
        self.osv_figure.patch.set_facecolor("#101820")
        ax.set_facecolor("#101820")
        self.osv_canvas.draw_idle()

    def _osa_view_save(self) -> None:
        if self.osa_view_trace is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save OSA trace", "osa_trace.csv", "CSV (*.csv)")
        if not path:
            return
        try:
            self.osa_view_trace.to_csv(path)
            self._log(f"Saved OSA trace → {path}")
        except Exception as exc:
            self._log(f"Save failed: {exc}")

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
                    col_ratio=self._active_col_ratio(),
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

    # ===================== TPA efficiency (eta) tab ==================
    def _build_tpa_tab(self) -> QtWidgets.QWidget:
        page = self._page_shell("Channel TPA Efficiency (η) Calibration")
        subtitle = QtWidgets.QLabel(
            "For each channel pair the two sides x and w are swept "
            "independently over a grid (with the x=0 / w=0 axes), all other "
            "channels held off. The response is fit by weighted least squares to "
            "Y = η²·(x·w) + a_x·x + q_x·x² + a_w·w + q_w·w² + d, so the "
            "two-photon cross term η, the single-beam terms and the dark offset "
            "are all recovered in one model — no separate background run. Reads "
            "use whichever monitor (scope or DAQ) is connected."
        )
        subtitle.setObjectName("PageSubtitle")
        subtitle.setWordWrap(True)
        page.layout().addWidget(subtitle)

        # --- sweep settings ---
        cfg = self._panel("Sweep Settings")
        grid = QtWidgets.QGridLayout(cfg)
        self.tpa_pair_index = self._spin(0, 63, 0)
        self.tpa_pair_index.setToolTip("Which channel pair (x[i], w[i]) to calibrate")
        self.tpa_all_pairs = QtWidgets.QCheckBox("All pairs")
        self.tpa_all_pairs.setToolTip("Sweep every pair (0..n-1) instead of just the index above")
        self.tpa_sweep_min = QtWidgets.QDoubleSpinBox()
        self.tpa_sweep_min.setRange(0.0, 1.0); self.tpa_sweep_min.setSingleStep(0.05)
        self.tpa_sweep_min.setDecimals(2); self.tpa_sweep_min.setValue(0.30)
        self.tpa_sweep_min.setToolTip("Minimum commanded per-side level (>0) in the ramp")
        self.tpa_sweep_max = QtWidgets.QDoubleSpinBox()
        self.tpa_sweep_max.setRange(0.0, 1.0); self.tpa_sweep_max.setSingleStep(0.05)
        self.tpa_sweep_max.setDecimals(2); self.tpa_sweep_max.setValue(1.00)
        self.tpa_sweep_max.setToolTip("Maximum commanded per-side level in the ramp")
        self.tpa_points = self._spin(2, 15, 6)
        self.tpa_points.setToolTip(
            "Ramp points per side; the x=0 / w=0 axis is added automatically "
            "→ (points+1)² grid cells"
        )
        self.tpa_trials = self._spin(1, 500, 10)
        self.tpa_trials.setToolTip(
            "Times the whole grid is repeated (gives each cell a standard error)"
        )
        self.tpa_repeats = self._spin(1, 20, 1)
        self.tpa_repeats.setToolTip("Monitor reads averaged per grid point within a trial")
        widgets = [
            ("Pair index", self.tpa_pair_index), ("", self.tpa_all_pairs),
            ("Sweep min", self.tpa_sweep_min), ("Sweep max", self.tpa_sweep_max),
            ("Ramp points", self.tpa_points), ("Trials", self.tpa_trials),
            ("Repeats", self.tpa_repeats),
        ]
        for i, (label, widget) in enumerate(widgets):
            r, c = i // 2, (i % 2) * 2
            if label:
                grid.addWidget(QtWidgets.QLabel(label), r, c)
            grid.addWidget(widget, r, c + 1)
        page.layout().addWidget(cfg)

        # --- results: joint-fit plot (left) + pulls plot (right) ---
        self.tpa_fit_fig = Figure(figsize=(5, 3.4), tight_layout=True)
        self.tpa_fit_canvas = FigureCanvas(self.tpa_fit_fig)
        self.tpa_pulls_fig = Figure(figsize=(5, 3.4), tight_layout=True)
        self.tpa_pulls_canvas = FigureCanvas(self.tpa_pulls_fig)
        plot_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        plot_split.addWidget(
            self._panel_with_widget("Joint fit (measured vs predicted)", self.tpa_fit_canvas)
        )
        plot_split.addWidget(
            self._panel_with_widget("Pulls (residual / SEM)", self.tpa_pulls_canvas)
        )
        plot_split.setSizes([560, 520])
        page.layout().addWidget(plot_split, 1)

        # --- displayed-pair selector + eta report ---
        self.tpa_pair_combo = QtWidgets.QComboBox()
        self.tpa_pair_combo.setToolTip("Which measured pair's fit to display")
        self.tpa_pair_combo.currentIndexChanged.connect(lambda _=0: self._tpa_redraw())
        self.tpa_report = QtWidgets.QLabel("η: (run or load a sweep)")
        self.tpa_report.setObjectName("PageSubtitle")
        self.tpa_report.setWordWrap(True)
        show_row = QtWidgets.QHBoxLayout()
        show_row.addWidget(QtWidgets.QLabel("Show pair"))
        show_row.addWidget(self.tpa_pair_combo)
        show_row.addWidget(self.tpa_report, 1)
        page.layout().addLayout(show_row)

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
        self.tpa_save_button = QtWidgets.QPushButton("Save…")
        self.tpa_save_button.setProperty("variant", "ghost")
        self.tpa_save_button.setEnabled(False)
        self.tpa_save_button.clicked.connect(self._tpa_save)
        self.tpa_load_button = QtWidgets.QPushButton("Load…")
        self.tpa_load_button.setProperty("variant", "ghost")
        self.tpa_load_button.setToolTip("Load a saved pair-grid CSV; every pair is re-fit.")
        self.tpa_load_button.clicked.connect(self._tpa_load)
        ctrl = QtWidgets.QHBoxLayout()
        ctrl.addWidget(self.tpa_status, 1)
        ctrl.addWidget(self.tpa_load_button)
        ctrl.addWidget(self.tpa_save_button)
        ctrl.addWidget(self.tpa_run_button)
        ctrl.addWidget(self.tpa_stop_button)
        page.layout().addWidget(self.tpa_progress_bar)
        page.layout().addLayout(ctrl)
        self._tpa_redraw()
        return page

    def _tpa_pair_indices(self, layout) -> list[int] | None:
        """Pairs to sweep from the controls, or None if the index is out of range."""
        n = layout.n_channels
        if self.tpa_all_pairs.isChecked():
            return list(range(n))
        idx = self.tpa_pair_index.value()
        if idx >= n:
            return None
        return [idx]

    def _tpa_set_running(self, running: bool) -> None:
        self.tpa_run_button.setEnabled(not running)
        self.tpa_stop_button.setEnabled(running)
        self.tpa_load_button.setEnabled(not running)
        self.tpa_save_button.setEnabled(not running and self.tpa_result is not None)

    def _tpa_run(self) -> None:
        from dataclasses import replace
        layout = self.encoding_layout
        if layout is None:
            self.tpa_status.setText(
                "No channel grid — build a layout on the TPA Encoding page first."
            )
            return
        active = self._enc_active_monitor()
        if active is None:
            self.tpa_status.setText("Connect the scope or DAQ first (Scope / DAQ page).")
            return
        controller = self._controller()
        if not getattr(controller, "is_open", False):
            self.tpa_status.setText("Open the SLM on the SLM Control page first.")
            return
        indices = self._tpa_pair_indices(layout)
        if not indices:
            self.tpa_status.setText(
                f"Pair index out of range (layout has {layout.n_channels} pairs)."
            )
            return

        kind, monitor = active
        if kind == "scope":
            settings = self._monitor_settings(trigger_mode="AUTO")
        else:
            settings = self._daq_monitor_settings()
        settle = float(settings.hold)               # the tab's settle = monitor-page hold
        read_timeout = max(30.0, settings.duration * 3.0 + 10.0)
        settings0 = replace(settings, hold=0.0)     # the module owns the settle

        sweep = build_sweep(
            self.tpa_sweep_min.value(), self.tpa_sweep_max.value(), self.tpa_points.value()
        )
        n_trials = self.tpa_trials.value()
        repeats = self.tpa_repeats.value()
        total = max(n_trials * len(indices) * sweep.size * sweep.size, 1)

        self.tpa_progress_bar.setMaximum(total)
        self.tpa_progress_bar.setValue(0)
        self.tpa_status.setText(
            f"Starting… {len(indices)} pair(s) × {sweep.size}×{sweep.size} grid × "
            f"{n_trials} trial(s) via {kind}"
        )
        self._tpa_set_running(True)

        stop_event = threading.Event()
        self.tpa_stop_event = stop_event

        def report(progress: TPAPairProgress) -> None:
            self.tpa_progress.emit(progress)

        def work() -> dict[str, Any]:
            monitor.configure_monitor(settings0)
            try:
                result = measure_pair_grids(
                    monitor, controller, layout,
                    pair_indices=indices, sweep=sweep,
                    n_trials=n_trials, repeats=repeats, settle=settle,
                    read_timeout=read_timeout, col_ratio=self._active_col_ratio(),
                    stop_event=stop_event, progress_callback=report,
                )
            except TPAPairAborted:
                return {"status": "aborted"}
            return {"status": "ok", "result": result}

        self._run_slm_task(
            "TPA η pair-grid sweep", work, self._tpa_finished, self._tpa_error
        )

    def _tpa_stop(self) -> None:
        if self.tpa_stop_event is not None:
            self.tpa_stop_event.set()
            self.tpa_status.setText("Stopping…")

    def _on_tpa_progress(self, progress: TPAPairProgress) -> None:
        self.tpa_progress_bar.setMaximum(max(progress.total, 1))
        self.tpa_progress_bar.setValue(min(progress.step, progress.total))
        self.tpa_status.setText(progress.message)

    def _tpa_finished(self, payload: dict[str, Any]) -> None:
        self.tpa_stop_event = None
        self._tpa_set_running(False)
        if payload.get("status") == "aborted":
            self.tpa_status.setText("Sweep stopped.")
            return
        result = payload["result"]
        self.tpa_result = result
        self.tpa_save_button.setEnabled(True)
        self._tpa_populate_pairs(result)
        self._tpa_redraw()
        etas = [c.fit.eta for c in result.channels if c.fit is not None]
        if len(etas) == 1:
            summ = f"η = {etas[0]:.4g}"
        elif etas:
            summ = f"{len(etas)} pairs · η {np.nanmin(etas):.3g}–{np.nanmax(etas):.3g}"
        else:
            summ = "no fits"
        self.tpa_status.setText(f"Done · {summ}")

    def _tpa_error(self, _error: str) -> None:
        self.tpa_stop_event = None
        self._tpa_set_running(False)
        self.tpa_status.setText("TPA sweep failed (see Status log)")

    def _tpa_populate_pairs(self, result: "TPAPairResult") -> None:
        self.tpa_pair_combo.blockSignals(True)
        self.tpa_pair_combo.clear()
        for c in result.channels:
            eta = c.fit.eta if c.fit is not None else float("nan")
            wl_txt = f"{c.nominal_wl_nm:.2f} nm" if np.isfinite(c.nominal_wl_nm) else "?"
            self.tpa_pair_combo.addItem(f"pair {c.index} · {wl_txt} · η={eta:.3g}")
        self.tpa_pair_combo.blockSignals(False)
        if result.channels:
            self.tpa_pair_combo.setCurrentIndex(0)

    def _tpa_selected_pair(self):
        if self.tpa_result is None or not self.tpa_result.channels:
            return None
        i = self.tpa_pair_combo.currentIndex()
        if i < 0 or i >= len(self.tpa_result.channels):
            i = 0
        return self.tpa_result.channels[i]

    def _tpa_redraw(self) -> None:
        grid = self._tpa_selected_pair()
        self._tpa_draw_fit(grid)
        self._tpa_draw_pulls(grid)
        self._tpa_update_report(grid)

    def _tpa_update_report(self, grid) -> None:
        if grid is None or grid.fit is None:
            self.tpa_report.setText("η: (run or load a sweep)")
            return
        f = grid.fit
        p = f.params
        self.tpa_report.setText(
            f"η = {f.eta:.4g} ± {f.eta_err:.2g}   "
            f"a_x={p['a_x'][0]:.3g}  a_w={p['a_w'][0]:.3g}  "
            f"d={p['d'][0]*1e3:.3f} mV   "
            f"χ²/dof={f.chi2_red:.2f} (Birge ×{f.birge:.2f})  R²={f.r2:.4f}"
        )

    def _tpa_draw_fit(self, grid) -> None:
        """Left: measured vs predicted for the selected pair (interior colored by x·w)."""
        self.tpa_fit_fig.clear()
        self.tpa_fit_fig.patch.set_facecolor("#101820")
        ax = self.tpa_fit_fig.add_subplot(111)
        self._style_dark_axes(ax)
        ax.set_xlabel("Measured voltage (mV)")
        ax.set_ylabel("Predicted voltage (mV)")
        if grid is None or grid.fit is None:
            ax.text(0.5, 0.5, "Run or load a sweep", ha="center", va="center",
                    transform=ax.transAxes, color="#d8dee9")
            self.tpa_fit_canvas.draw_idle()
            return
        f = grid.fit
        y = f.y * 1e3
        yp = f.y_pred * 1e3
        sem = f.sem * 1e3
        axis = (f.x == 0) | (f.w == 0)
        lims = [min(y.min(), yp.min()), max(y.max(), yp.max())]
        ax.plot(lims, lims, "--", color="#e0a447", lw=1.0, label="ideal")
        ax.errorbar(y, yp, xerr=sem, fmt="none", ecolor="#41515c", elinewidth=0.8, zorder=1)
        if axis.any():
            ax.scatter(y[axis], yp[axis], marker="s", s=42, facecolor="none",
                       edgecolor="#e0a447", lw=1.3, zorder=3, label="axis")
        if (~axis).any():
            sc = ax.scatter(y[~axis], yp[~axis], c=(f.x * f.w)[~axis], cmap="viridis",
                            s=38, edgecolor="#101820", lw=0.4, zorder=2, label="interior")
            cbar = self.tpa_fit_fig.colorbar(sc, ax=ax)
            cbar.set_label("x·w", color="#d8dee9")
            cbar.ax.tick_params(colors="#d8dee9")
        ax.legend(loc="upper left", fontsize=7)
        self.tpa_fit_canvas.draw_idle()

    def _tpa_draw_pulls(self, grid) -> None:
        """Right: normalised residuals (pull = residual/SEM) vs predicted."""
        self.tpa_pulls_fig.clear()
        self.tpa_pulls_fig.patch.set_facecolor("#101820")
        ax = self.tpa_pulls_fig.add_subplot(111)
        self._style_dark_axes(ax)
        ax.set_xlabel("Predicted voltage (mV)")
        ax.set_ylabel("Pull = residual / SEM")
        if grid is None or grid.fit is None:
            self.tpa_pulls_canvas.draw_idle()
            return
        f = grid.fit
        yp = f.y_pred * 1e3
        pulls = f.residuals / f.sem
        axis = (f.x == 0) | (f.w == 0)
        ax.axhspan(-1, 1, color="#8fd14f", alpha=0.12)
        ax.axhline(0.0, color="#e0a447", ls="--", lw=1.0)
        if (~axis).any():
            ax.scatter(yp[~axis], pulls[~axis], c="#e05a5a", s=34,
                       edgecolor="#101820", lw=0.4, label="interior")
        if axis.any():
            ax.scatter(yp[axis], pulls[axis], marker="s", s=42, facecolor="none",
                       edgecolor="#e0a447", lw=1.3, label="axis")
        ax.legend(loc="upper right", fontsize=7)
        self.tpa_pulls_canvas.draw_idle()

    def _tpa_load(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load TPA Pair-Grid CSV", "", "CSV (*.csv)"
        )
        if not path:
            return
        try:
            result = load_tpa_pair_csv(path, layout=self.encoding_layout)
        except Exception as exc:
            self.tpa_status.setText(f"Load failed: {exc}")
            return
        self.tpa_result = result
        self.tpa_save_button.setEnabled(True)
        self._tpa_populate_pairs(result)
        self._tpa_redraw()
        self.tpa_status.setText(
            f"Loaded {Path(path).name} · {len(result.channels)} pair(s) re-fit"
        )

    def _tpa_save(self) -> None:
        if self.tpa_result is None:
            return
        default = f"tpa_pair_calibration_{time.strftime('%m%d_%H%M')}.csv"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save TPA Pair Calibration", default,
            "CSV (*.csv);;JSON (*.json)"
        )
        if not path:
            return
        base = Path(path).with_suffix("")
        csv_path = write_tpa_pair_csv(self.tpa_result, base.with_suffix(".csv"))
        js = save_tpa_pair_json(self.tpa_result, base.with_suffix(".json"))
        self.tpa_status.setText(f"Saved {Path(csv_path).name}  +  {Path(js).name}")

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
        if self.daq_controller is not None:
            self._disconnect_daq()
            self._log("DAQ disconnected automatically (scope connected)")
        self.scope_controller = scope
        self._set_status(self.scope_status_label, "Scope: open", "ok")
        self.scope_connect_button.setEnabled(False)
        self.scope_disconnect_button.setEnabled(True)
        self._log(f"Scope connected: {identity.strip()}")
        self._sync_monitor_source()

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
        self._sync_monitor_source()

    # ===================== DAQ (NI-DAQmx) page ========================
    def _connect_daq(self) -> None:
        device = self.daq_device_edit.text().strip()
        if not device:
            self._log("Enter the DAQ device name first")
            return
        self.daq_connect_button.setEnabled(False)

        def connect() -> tuple[DAQController, str]:
            daq = DAQController(device=device)
            daq.connect()
            return daq, daq.identify()

        self._run_task("Connect DAQ", connect, self._on_daq_connected, self._on_daq_error)

    def _on_daq_connected(self, payload: tuple[DAQController, str]) -> None:
        daq, identity = payload
        if self.scope_controller is not None:
            self._disconnect_scope()
            self._log("Scope disconnected automatically (DAQ connected)")
        self.daq_controller = daq
        self._set_status(self.daq_status_label, "DAQ: open", "ok")
        self.daq_connect_button.setEnabled(False)
        self.daq_disconnect_button.setEnabled(True)
        self._log(f"DAQ connected: {identity.strip()}")
        self._sync_monitor_source()

    def _on_daq_error(self, _error: str) -> None:
        self._set_status(self.daq_status_label, "DAQ: error", "error")
        self.daq_connect_button.setEnabled(True)

    def _disconnect_daq(self) -> None:
        daq = self.daq_controller
        self.daq_controller = None
        self._set_status(self.daq_status_label, "DAQ: closed", "off")
        self.daq_connect_button.setEnabled(True)
        self.daq_disconnect_button.setEnabled(False)
        if daq is not None:
            self._run_task("Disconnect DAQ", daq.disconnect)
        self._sync_monitor_source()

    # ===================== Scope Monitor page ========================
    _TRIG_SOURCES = [("CH1", "CHANnel1"), ("CH2", "CHANnel2"), ("CH3", "CHANnel3"),
                     ("CH4", "CHANnel4"), ("EXT", "EXTernanalog")]
    # NI-DAQmx programmable-gain input ranges (device-independent common set;
    # the driver silently snaps to the closest one the connected card supports).
    _DAQ_RANGES = [(-0.1, 0.1), (-0.2, 0.2), (-0.5, 0.5), (-1.0, 1.0),
                   (-2.0, 2.0), (-5.0, 5.0), (-10.0, 10.0)]

    def _build_monitor_widget(self) -> QtWidgets.QWidget:
        """Embeddable single-reading monitor (lives inside the TPA encoder page).

        Shows either the scope's trigger/averaging config or the DAQ's
        acquisition config, whichever instrument is connected on the
        Connections page (see _sync_monitor_source); the recorded-readings
        plot and controls below are shared between both sources. The two
        config panels are toggled with setVisible() rather than a
        QStackedWidget -- a QStackedWidget's sizeHint is the union of every
        page it has ever held, so it would keep the smaller DAQ panel
        stretched to the taller scope panel's height.
        """
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)

        self.mon_source_label = QtWidgets.QLabel("Source: (none connected)")
        self.mon_source_label.setObjectName("PageSubtitle")
        v.addWidget(self.mon_source_label)

        # config : average-per-pattern share leftover height 1:2 (only the
        # visible one of the two config panels counts, the hidden one takes no
        # space -- see _sync_monitor_source)
        self.scope_monitor_cfg = self._build_scope_monitor_config()
        self.daq_monitor_cfg = self._build_daq_monitor_config()
        v.addWidget(self.scope_monitor_cfg, 1)
        v.addWidget(self.daq_monitor_cfg, 1)

        self.mon_count_label = QtWidgets.QLabel("0 patterns")
        self.mon_count_label.setObjectName("PageSubtitle")
        self.mon_count_label.setAlignment(QtCore.Qt.AlignCenter)
        v.addWidget(self.mon_count_label)

        self.mon_fig = Figure(figsize=(4, 1.8), tight_layout=True)
        self.mon_canvas = FigureCanvas(self.mon_fig)
        v.addWidget(self._panel_with_widget("Mean \N{PLUS-MINUS SIGN} std per pattern", self.mon_canvas), 2)

        # This readout is a behaviour recorder, not a live monitor: each reading
        # appends one (pattern #, mean, std) point -- either automatically after
        # an SLM send, or on demand via the Acquire button.
        self.mon_acquire_button = QtWidgets.QPushButton("Acquire")
        self.mon_acquire_button.setToolTip(
            "Read one averaged (mean \N{PLUS-MINUS SIGN} std) sample now from the connected "
            "instrument (scope or DAQ) and append it to the record -- no SLM send needed."
        )
        self.mon_acquire_button.clicked.connect(self._enc_acquire_clicked)
        self.mon_clear_button = QtWidgets.QPushButton("Clear")
        self.mon_clear_button.setProperty("variant", "ghost")
        self.mon_clear_button.clicked.connect(self._monitor_clear)
        self.mon_save_button = QtWidgets.QPushButton("Save CSV…")
        self.mon_save_button.setProperty("variant", "ghost")
        self.mon_save_button.clicked.connect(self._monitor_save)
        self.mon_read_on_send = QtWidgets.QCheckBox("Auto-read on SLM send")
        self.mon_read_on_send.setChecked(True)
        self.mon_read_on_send.setToolTip(
            "After a pattern is sent from this page, take one averaged reading "
            "from whichever instrument (scope or DAQ) is connected, and append "
            "it to the record."
        )
        v.addWidget(self.mon_read_on_send)
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)   # push the buttons to the right at their natural width
        row.addWidget(self.mon_acquire_button)
        row.addWidget(self.mon_clear_button)
        row.addWidget(self.mon_save_button)
        v.addLayout(row)
        self._sync_monitor_source()
        return w

    def _build_scope_monitor_config(self) -> QtWidgets.QWidget:
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
        # 4 fields per row: the panel spans half the page width, so 2-per-row
        # left most of it empty and made the panel needlessly tall
        for i, (label, widget) in enumerate(pairs):
            r, c = i // 4, (i % 4) * 2
            grid.addWidget(QtWidgets.QLabel(label), r, c)
            grid.addWidget(widget, r, c + 1)
        grid.setColumnStretch(8, 1)   # absorb leftover width instead of stretching fields
        return cfg

    def _build_daq_monitor_config(self) -> QtWidgets.QWidget:
        cfg = self._panel("DAQ Monitor · acquisition")
        grid = QtWidgets.QGridLayout(cfg)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        self.daq_mon_channel = QtWidgets.QLineEdit("ai0")
        self.daq_mon_channel.setMaximumWidth(90)
        self.daq_mon_sample_rate = QtWidgets.QDoubleSpinBox()
        self.daq_mon_sample_rate.setRange(1.0, 2_000_000.0); self.daq_mon_sample_rate.setDecimals(0)
        self.daq_mon_sample_rate.setValue(100_000.0); self.daq_mon_sample_rate.setSuffix(" S/s")
        self.daq_mon_sample_rate.setMaximumWidth(120)
        self.daq_mon_hold = QtWidgets.QDoubleSpinBox()
        self.daq_mon_hold.setRange(0.0, 10000.0); self.daq_mon_hold.setValue(100.0); self.daq_mon_hold.setSuffix(" ms")
        self.daq_mon_hold.setMaximumWidth(100)
        self.daq_mon_duration = QtWidgets.QDoubleSpinBox()
        self.daq_mon_duration.setRange(0.001, 10.0); self.daq_mon_duration.setDecimals(3)
        self.daq_mon_duration.setValue(0.05); self.daq_mon_duration.setSuffix(" s")
        self.daq_mon_duration.setMaximumWidth(100)
        self.daq_mon_range = QtWidgets.QComboBox()
        self.daq_mon_range.setMaximumWidth(100)
        for lo, hi in self._DAQ_RANGES:
            self.daq_mon_range.addItem(f"\N{PLUS-MINUS SIGN}{hi:g} V", (lo, hi))
        self.daq_mon_range.setCurrentIndex(0)   # smallest / most sensitive range by default
        pairs = [("Channel", self.daq_mon_channel), ("Sample rate", self.daq_mon_sample_rate),
                 ("Hold", self.daq_mon_hold), ("Average for", self.daq_mon_duration),
                 ("Range", self.daq_mon_range)]
        # single row: the panel spans half the page width, so a 2-per-row grid
        # left most of it empty and made the panel needlessly tall
        for i, (label, widget) in enumerate(pairs):
            grid.addWidget(QtWidgets.QLabel(label), 0, i * 2)
            grid.addWidget(widget, 0, i * 2 + 1)
        grid.setColumnStretch(len(pairs) * 2, 1)   # absorb leftover width instead of stretching fields
        return cfg

    def _sync_monitor_source(self) -> None:
        """Show the config panel for whichever instrument is connected."""
        if not hasattr(self, "scope_monitor_cfg"):
            return
        connected = self._enc_active_monitor() is not None
        # manual Acquire only makes sense when an instrument is connected
        self.mon_acquire_button.setEnabled(connected)
        if self.scope_controller is not None and self.scope_controller.is_connected:
            self.scope_monitor_cfg.setVisible(True)
            self.daq_monitor_cfg.setVisible(False)
            self.mon_source_label.setText("Source: Scope (R&S RTO6)")
        elif self.daq_controller is not None and self.daq_controller.is_connected:
            self.scope_monitor_cfg.setVisible(False)
            self.daq_monitor_cfg.setVisible(True)
            self.mon_source_label.setText("Source: DAQ (NI-DAQmx)")
        else:
            self.scope_monitor_cfg.setVisible(True)
            self.daq_monitor_cfg.setVisible(False)
            self.mon_source_label.setText("Source: (none connected — connect Scope or DAQ)")

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

    def _daq_monitor_settings(self) -> DAQMonitorSettings:
        min_val, max_val = self.daq_mon_range.currentData()
        return DAQMonitorSettings(
            channel=self.daq_mon_channel.text().strip() or "ai0",
            sample_rate=self.daq_mon_sample_rate.value(),
            duration=self.daq_mon_duration.value(),
            hold=self.daq_mon_hold.value() / 1000.0,  # ms -> s
            min_val=min_val,
            max_val=max_val,
        )

    def _on_monitor_sample(self, sample: MonitorSample) -> None:
        self._monitor_values.append(sample.value)
        # std is per-window noise (None if the source doesn't report it); NaN
        # keeps the record aligned 1:1 with the mean list for plotting/saving
        self._monitor_stds.append(sample.std if sample.std is not None else float("nan"))
        self.mon_count_label.setText(f"{len(self._monitor_values)} patterns")
        n = len(self._monitor_values)
        if sample.std is not None:
            self._mon_status(
                f"pattern #{n}: {sample.value*1000:.4f} \N{PLUS-MINUS SIGN} "
                f"{sample.std*1000:.4f} mV"
            )
        else:
            self._mon_status(f"pattern #{n}: {sample.value*1000:.4f} mV")
        self._monitor_draw()

    def _monitor_draw(self) -> None:
        self.mon_fig.clear()
        self.mon_fig.patch.set_facecolor("#101820")
        ax = self.mon_fig.add_subplot(111)
        self._style_dark_axes(ax)
        ax.set_xlabel("Pattern # (send order)")
        ax.set_ylabel("Mean \N{PLUS-MINUS SIGN} std (mV)")
        if self._monitor_values:
            n = len(self._monitor_values)
            xs = list(range(1, n + 1))
            ys = [v * 1000 for v in self._monitor_values]
            # per-point std as error bars (mV); NaN entries render bar-less
            yerr = [s * 1000 for s in self._monitor_stds]
            ax.errorbar(xs, ys, yerr=yerr, marker="o", ms=3, color="#47b8e0",
                        linewidth=0.8, ecolor="#6f8ea0", elinewidth=0.8, capsize=2)
            # integer-only ticks on the pattern axis (1, 2, 3, …)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_xlim(0.5, n + 0.5)
        self.mon_canvas.draw_idle()

    def _monitor_clear(self) -> None:
        self._monitor_values = []
        self._monitor_stds = []
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
            writer.writerow(["pattern", "mean_V", "std_V"])
            for i, (v, s) in enumerate(
                zip(self._monitor_values, self._monitor_stds), start=1
            ):
                writer.writerow([i, v, "" if s != s else s])  # blank for NaN std
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

    def _toggle_fast_channel_source(self) -> None:
        if not hasattr(self, "fast_channel_source_combo"):
            return
        from_file = self.fast_channel_source_combo.currentIndex() == 1
        self.fast_channel_step2_edit.setEnabled(from_file)
        self.fast_channel_step2_button.setEnabled(from_file)
        self.fast_channel_min_spin.setEnabled(from_file)
        self.fast_channel_max_spin.setEnabled(from_file)

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
        self._calibration_is_running = running
        connected = self.osa_controller is not None
        for button in getattr(self, "calibration_run_buttons", []):
            button.setEnabled(connected and not running)
        self.stop_cal_button.setEnabled(running)
        self.pipeline_stop_button.setEnabled(running)
        if hasattr(self, "stage3_reopt_stop_button"):
            self.stage3_reopt_stop_button.setEnabled(running)
        if hasattr(self, "fast_channel_stop_button"):
            self.fast_channel_stop_button.setEnabled(running)
        self.osa_connect_button.setEnabled(not running and not connected)
        self.osa_disconnect_button.setEnabled(not running and connected)
        self._refresh_pipeline_ui()

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

    def _fast_channel_settings(self) -> MeasurementSettings:
        return MeasurementSettings(
            center_wl=self.fast_channel_center_edit.text().strip() or "778nm",
            span=self.fast_channel_span_edit.text().strip() or "8nm",
            sensitivity=self.fast_channel_sensitivity_combo.currentText(),
            sampling_points=self.fast_channel_sampling_edit.text().strip() or "AUTO",
            reference_level=self.fast_channel_ref_level_edit.text().strip() or "10uW",
            y_unit="LINear",
        )

    def _fast_channel_levels(self) -> list[int]:
        start = self.fast_channel_level_start_spin.value()
        stop = self.fast_channel_level_stop_spin.value()
        step_size = self.fast_channel_level_step_spin.value()
        if stop < start:
            raise ValueError("fast channel level stop must be >= level start")
        levels = list(range(start, stop + 1, step_size))
        if not levels:
            levels = [start]
        if levels[-1] != stop:
            levels.append(stop)
        return levels

    def _fast_channel_guard_bands(self) -> list[tuple[float, float]]:
        if not self.fast_channel_guard_check.isChecked():
            return []
        value_text = self.fast_channel_guard_wl_edit.text().strip()
        if not value_text:
            raise ValueError("guard center wavelengths are required")
        parts = [part for part in re.split(r"[\s,;]+", value_text) if part]
        if not parts:
            raise ValueError("guard center wavelengths are required")
        try:
            centers = [float(part) for part in parts]
        except ValueError as exc:
            raise ValueError("guard center wavelengths must be numbers in nm") from exc
        if not all(np.isfinite(center) for center in centers):
            raise ValueError("guard center wavelengths must be finite")
        half_width = self.fast_channel_guard_nm_spin.value()
        if half_width <= 0.0:
            raise ValueError("guard half-width must be positive")
        return [(center, half_width) for center in centers]

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

    def _resolve_fast_channel_step2_input(self) -> CalibrationResult:
        if self.fast_channel_source_combo.currentIndex() == 1:
            path = self.fast_channel_step2_edit.text().strip()
            if not path:
                raise ValueError("choose a Step 2 result or wavelength-map CSV")
            if path.lower().endswith(".csv"):
                min_level = self.fast_channel_min_spin.value()
                max_level = self.fast_channel_max_spin.value()
                if max_level < min_level:
                    raise ValueError("CSV max level must be >= min level")
                result = load_wavelength_map_csv(
                    path,
                    min_level=min_level,
                    max_level=max_level,
                )
            else:
                result = load_calibration_result(path)
        else:
            result = self.calibration_result
            if result is None:
                raise ValueError("run Step 2 first, or choose a Step 2 file")

        if (
            np.asarray(result.coordinates).size == 0
            or np.asarray(result.wavelength).size == 0
        ):
            raise ValueError("the Step 2 source has no coordinate -> wavelength map")
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

    def _run_fast_channel_calibration(self) -> None:
        osa = self._osa_ready()
        if osa is None:
            return
        try:
            settings = self._fast_channel_settings()
            step2_mapping = self._resolve_fast_channel_step2_input()
            levels = self._fast_channel_levels()
            target_wavelength = self.fast_channel_target_spin.value()
            channel_width = self.fast_channel_width_spin.value()
            gap_px = self.fast_channel_gap_spin.value()
            n_channels = self.fast_channel_count_spin.value()
            group_skip = self.fast_channel_skip_spin.value()
            fine_tune_center = self.fast_channel_fine_check.isChecked()
            peak_half_window_nm = self.fast_channel_peak_nm_spin.value()
            avg_nm = self.fast_channel_avg_nm_spin.value() or None
            refine = self.fast_channel_refine_check.isChecked()
            refine_half_window_nm = (
                self.fast_channel_refine_nm_spin.value() if refine else None
            )
            guard_bands = self._fast_channel_guard_bands()
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return self._reject_calibration(exc)

        out_json = self._resolve_output_path(
            self.fast_channel_json_edit.text(), "calib_fast_channels.json"
        )
        out_csv = self._resolve_output_path(
            self.fast_channel_csv_edit.text(), "calibration_fast_channels.csv"
        )
        controller = self._controller()
        pitch_px = channel_width + gap_px
        self.fast_channel_status_label.setText("Running fast channel calibration")
        self._log(
            "Fast channel calibration started: "
            f"{2 * n_channels} channels, pitch {pitch_px} px, "
            f"active skip {group_skip}"
        )
        if guard_bands:
            guard_text = ", ".join(f"{center:g}±{half:g} nm" for center, half in guard_bands)
            self._log(f"Fast channel guard bands: {guard_text} -> min level")

        def work(report: ProgressEmit, stop_event: threading.Event) -> dict[str, Any]:
            slm_width, _slm_height = controller.get_slm_info()
            measured_peak = None
            coarse_center = None
            refined_center = None
            if fine_tune_center:
                refined_center, measured_peak, coarse_center = (
                    refine_center_coordinate_with_osa(
                        osa,
                        controller,
                        settings,
                        step2_mapping,
                        target_wavelength_nm=target_wavelength,
                        window_size=channel_width,
                        peak_half_window_nm=peak_half_window_nm,
                        stop_event=stop_event,
                        progress_callback=report,
                    )
                )

            grid_seed, center_coordinate = build_channel_calibration_grid(
                step2_mapping,
                target_wavelength_nm=target_wavelength,
                center_coordinate=refined_center,
                n_channels_per_side=n_channels,
                channel_width_px=channel_width,
                gap_px=gap_px,
                slm_width=slm_width,
                guard_bands_nm=guard_bands,
            )
            if not fine_tune_center:
                report(
                    CalibrationProgress(
                        phase="fast_center",
                        step=0,
                        total=1,
                        message=(
                            f"Step 2 predicts {target_wavelength:.4f} nm at "
                            f"x={center_coordinate:.3f} px"
                        ),
                        x=center_coordinate,
                        y=target_wavelength,
                    )
                )

            final = batch_intensity_calibration(
                osa,
                controller,
                levels,
                settings,
                grid_seed,
                window_size=channel_width,
                wavelength_window_nm=avg_nm,
                group_skip_channels=group_skip,
                guard_bands_nm=guard_bands,
                refine_wavelength=refine,
                refine_half_window_nm=refine_half_window_nm,
                stop_event=stop_event,
                progress_callback=report,
            )
            save_calibration_result(final, out_json)
            csv_path = write_intensity_calibration_csv(final, out_csv)
            group_count = min(group_skip + 1, int(final.coordinates.size))
            return {
                "status": "ok",
                "step": "fast_channels",
                "result": final,
                "saved": out_json,
                "csv": csv_path,
                "center_coordinate": center_coordinate,
                "coarse_center": coarse_center,
                "measured_peak": measured_peak,
                "pitch_px": pitch_px,
                "group_count": group_count,
                "summary": (
                    f"{final.coordinates.size} channels, pitch {pitch_px} px, "
                    f"{group_count} channel groups"
                ),
            }

        self._launch_calibration("Fast channel calibration", work)

    def _pipeline_file_path(
        self,
        edit: QtWidgets.QLineEdit,
        label: str,
        *,
        must_exist: bool = False,
    ) -> Path:
        """Resolve a required pipeline path and reject directories/missing inputs."""
        text = edit.text().strip()
        if not text:
            raise ValueError(f"{label} is required")
        path = Path(text).expanduser().resolve()
        if must_exist and not path.is_file():
            raise ValueError(f"{label} does not exist: {path}")
        if path.exists() and path.is_dir():
            raise ValueError(f"{label} must be a file: {path}")
        return path

    def _load_pipeline_input(
        self,
        step: int,
        path: Path,
        *,
        csv_min_level: int,
        csv_max_level: int,
    ) -> CalibrationResult:
        """Load and validate one step input without consulting in-memory state."""
        if step == 3 and path.suffix.lower() == ".csv":
            result = load_wavelength_map_csv(
                path,
                min_level=csv_min_level,
                max_level=csv_max_level,
            )
        else:
            result = load_calibration_result(path)
        self._require_levels(result)
        if step == 3 and (
            np.asarray(result.coordinates).size == 0
            or np.asarray(result.wavelength).size == 0
        ):
            raise ValueError(
                "Step 3 input file has no coordinate -> wavelength mapping"
            )
        return result

    def _pipeline_directory_path(
        self, edit: QtWidgets.QLineEdit, label: str
    ) -> Path:
        text = edit.text().strip()
        if not text:
            raise ValueError(f"{label} is required")
        path = Path(text).expanduser().resolve()
        if path.exists() and not path.is_dir():
            raise ValueError(f"{label} must be a directory: {path}")
        return path

    def _validate_pipeline_initial_profile(
        self, values: Any, *, source: str
    ) -> np.ndarray:
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.size == 15:
            return independent_intensity_profile(values)
        if values.size == 8:
            return validate_independent_profile(values, width=15)
        raise ValueError(
            f"{source} must contain 8 values or a symmetric 15-value profile; "
            f"found {values.size}"
        )

    def _parse_pipeline_initial_profile(self, text: str) -> np.ndarray:
        """Parse direct comma/space-separated profile values from the UI."""
        value_text = text.strip()
        if not value_text:
            raise ValueError("Encoding Optimization initial profile values are required")
        if value_text.startswith("["):
            try:
                values = json.loads(value_text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid initial profile JSON: {exc.msg}") from exc
        else:
            parts = [part for part in re.split(r"[\s,;]+", value_text) if part]
            try:
                values = [float(part) for part in parts]
            except ValueError as exc:
                raise ValueError(
                    "initial profile values must be numbers separated by commas or spaces"
                ) from exc
        return self._validate_pipeline_initial_profile(
            values, source="initial profile input"
        )

    def _parse_pipeline_quick_levels(self, text: str) -> np.ndarray:
        """Parse the min~max+stride SLM range used by quick centre calibration."""
        value_text = text.strip()
        if not value_text:
            raise ValueError("Quick SLM range is required")
        match = re.fullmatch(r"\s*(\d+)\s*~\s*(\d+)\s*\+\s*(\d+)\s*", text)
        if match is None:
            raise ValueError(
                "Quick SLM range must use min~max+stride, for example 420~870+50"
            )
        min_level, max_level, stride = (int(part) for part in match.groups())
        if min_level >= max_level:
            raise ValueError("Quick SLM range requires min < max")
        if stride <= 0:
            raise ValueError("Quick SLM range stride must be positive")
        if min_level < 0 or max_level > MAX_LEVEL:
            raise ValueError(f"Quick SLM range must be in 0..{MAX_LEVEL}")
        levels = list(range(min_level, max_level + 1, stride))
        if levels[-1] != max_level:
            levels.append(max_level)
        return np.asarray(levels, dtype=int)

    def _load_pipeline_initial_profile(self, path: Path) -> np.ndarray:
        """Load an 8-value initial profile or a symmetric 15-value profile."""
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                for key in (
                    "l",
                    "l_init",
                    "final_l",
                    "final_profile",
                    "initial_l",
                    "profile",
                ):
                    if key in payload:
                        payload = payload[key]
                        break
                else:
                    raise ValueError(
                        "profile JSON must contain l, l_init, initial_l, final_l, "
                        "final_profile, or profile"
                    )
            values = payload
        else:
            delimiter = "," if path.suffix.lower() == ".csv" else None
            values = np.asarray(
                np.genfromtxt(path, delimiter=delimiter, dtype=float), dtype=float
            ).reshape(-1)
            values = values[np.isfinite(values)]
        return self._validate_pipeline_initial_profile(
            values, source="initial profile file"
        )

    def _load_pipeline_optimization_calibration(
        self, path: Path
    ) -> CalibrationResult:
        result = load_calibration_result(path)
        self._require_levels(result)
        if result.intensity_levels is None:
            raise ValueError("Encoding Optimization requires a Step 3 intensity result")
        if (
            np.asarray(result.coordinates).size < 2
            or np.asarray(result.wavelength).size < 2
        ):
            raise ValueError("Encoding Optimization calibration has too few coordinates")
        return result

    def _load_pipeline_wavelength_calibration(
        self, path: Path, *, target_wavelength_nm: float
    ) -> CalibrationResult:
        """Load Step 2 data and verify that the target can be interpolated."""
        result = load_calibration_result(path)
        self._require_levels(result)
        interpolate_coordinate_for_wavelength(result, target_wavelength_nm)
        return result

    def _load_pipeline_quick_intensity_calibration(
        self, path: Path
    ) -> CalibrationResult:
        result = load_calibration_result(path)
        self._require_levels(result)
        coordinates = np.asarray(result.coordinates)
        wavelengths = np.asarray(result.wavelength)
        intensity = result.intensity_levels
        if coordinates.size != 1 or wavelengths.size != 1 or intensity is None:
            raise ValueError(
                "quick optimization requires a one-coordinate intensity calibration"
            )
        return result

    def _build_pipeline_encoding_layout(
        self,
        calibration: CalibrationResult,
        *,
        center_wl: float,
        channel_width_px: int,
        gap_px: int,
    ) -> ChannelLayout:
        coords = np.asarray(calibration.coordinates, dtype=float)
        wavelengths = np.asarray(calibration.wavelength, dtype=float)
        slope, intercept = np.polyfit(coords, wavelengths, 1)
        if not np.isfinite(slope) or abs(float(slope)) < 1e-12:
            raise ValueError("calibration wavelength slope is zero or invalid")
        center_x = (center_wl - intercept) / slope
        pitch = channel_width_px + gap_px
        n_channels = int(min(center_x - coords.min(), coords.max() - center_x) / pitch)
        if n_channels < 1:
            raise ValueError("no encoding channels fit inside the calibrated range")
        return build_channel_layout(
            calibration,
            n_channels=n_channels,
            channel_width_px=channel_width_px,
            gap_px=gap_px,
            center_wl=center_wl,
        )

    def _run_stage3_reoptimization(self) -> None:
        """Standalone quick Stage-3 re-optimisation from saved files."""
        osa = self._osa_ready()
        if osa is None:
            return
        controller = self._controller()
        if not getattr(controller, "is_open", False):
            return self._reject_calibration(
                ValueError("open the SLM before running Stage 3 re-optimization")
            )
        try:
            center_wl = self.stage3_reopt_center_wl_spin.value()
            channel_width = self.stage3_reopt_width_spin.value()
            gap_px = self.stage3_reopt_gap_spin.value()
            if channel_width != 15:
                raise ValueError(
                    "Stage 3 re-optimization requires a 15 px channel width"
                )
            step2_path = self._pipeline_file_path(
                self.stage3_reopt_step2_edit,
                "Stage 3 re-optimization Step 2 map",
                must_exist=True,
            )
            quick_calibration_path = self._pipeline_file_path(
                self.stage3_reopt_quick_calib_edit,
                "Stage 3 re-optimization quick calibration",
                must_exist=True,
            )
            profile_path = self._pipeline_file_path(
                self.stage3_reopt_profile_edit,
                "Stage 3 re-optimization profile",
                must_exist=True,
            )
            initial_l = self._load_pipeline_initial_profile(profile_path)
            step2_calibration = self._load_pipeline_wavelength_calibration(
                step2_path, target_wavelength_nm=center_wl
            )
            quick_calibration = self._load_pipeline_quick_intensity_calibration(
                quick_calibration_path
            )
            optimization_layout, quick_target_coordinate = build_single_anchor_layout(
                step2_calibration,
                quick_calibration,
                target_wavelength_nm=center_wl,
                channel_width_px=channel_width,
                gap_px=gap_px,
            )
            output_root = self._pipeline_directory_path(
                self.stage3_reopt_root_edit,
                "Stage 3 re-optimization output root",
            )
            run_name = self.stage3_reopt_name_edit.text().strip() or None
            if run_name is not None and (
                Path(run_name).name != run_name or run_name in (".", "..")
            ):
                raise ValueError("Stage 3 re-optimization run name must be one directory name")
            y_unit = (
                "LOGarithmic"
                if self.stage3_reopt_yunit_combo.currentText().startswith("LOG")
                else "LINear"
            )
            optimization_settings = MeasurementSettings(
                center_wl=f"{center_wl:g}nm",
                span=self.stage3_reopt_span_edit.text().strip() or "0.8nm",
                sensitivity=self.stage3_reopt_sensitivity_combo.currentText(),
                sampling_points="1001",
                y_unit=y_unit,
                reference_level=(
                    self.stage3_reopt_ref_level_edit.text().strip() or "10uW"
                ),
            )
            optimization_config = OSAOptimizationConfig(
                settings=optimization_settings,
                anchor_offsets=(0,),
                full_validation=False,
                output_root=str(output_root),
                run_name=run_name,
                averages=self.stage3_reopt_averages_spin.value(),
                rerank_averages=self.stage3_reopt_rerank_averages_spin.value(),
                stage2_repeats=self.stage3_reopt_baseline_repeats_spin.value(),
                stage3_maxfev=self.stage3_reopt_maxeval_spin.value(),
                skip_stage1=True,
            )
            quick_measured_range = (
                int(quick_calibration.min_level),
                int(quick_calibration.max_level),
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return self._reject_calibration(exc)

        self.stage3_reopt_status_label.setText("Running Stage 3 re-optimization")
        self._edge_gain_running(True)
        self.edge_gain_bar.setRange(0, 0)
        self.edge_gain_status.setText(
            "Stage 3 re-optimization running from standalone panel"
        )

        def work(report: ProgressEmit, stop_event: threading.Event) -> dict[str, Any]:
            report(
                CalibrationProgress(
                    phase="stage3_reopt_setup",
                    step=0,
                    total=1,
                    message=(
                        f"{center_wl:g} nm -> x={quick_target_coordinate:.3f} px; "
                        "using supplied Stage 1 profile"
                    ),
                    x=quick_target_coordinate,
                    y=center_wl,
                )
            )

            def report_optimization(progress: OptimizationProgress) -> None:
                self.edge_optimization_progress.emit(progress)
                report(
                    CalibrationProgress(
                        phase=f"optimization: {progress.stage}",
                        step=progress.step,
                        total=max(progress.total, 1),
                        message=progress.message,
                        x=float(progress.step),
                        y=progress.best_loss,
                    )
                )

            try:
                optimization_result = optimize_from_osa(
                    optimization_layout,
                    osa=osa,
                    slm=controller,
                    initial_l=initial_l,
                    config=optimization_config,
                    stop_event=stop_event,
                    progress_callback=report_optimization,
                )
            except OptimizationAborted:
                return {"status": "aborted"}
            final_path = Path(
                optimization_result.run_dir, "final_result.json"
            ).resolve()
            return {
                "status": "ok",
                "step": "stage3_reopt",
                "result": quick_calibration,
                "saved": final_path,
                "optimization_result": optimization_result,
                "optimization_layout": optimization_layout,
                "quick_target_coordinate": quick_target_coordinate,
                "quick_measured_range": quick_measured_range,
                "summary": "Stage 3 re-optimization complete",
            }

        self._launch_calibration("Stage 3 re-optimization", work)
        self.edge_gain_stop_event = self.calibration_stop_event

    def _run_pipeline(self) -> None:
        """Run any selected steps, using files as the only inter-step boundary."""
        osa = self._osa_ready()
        if osa is None:
            return
        selected = tuple(
            step
            for step in (1, 2, 3, 4)
            if self.pipeline_checks[step].isChecked()
        )
        if not selected:
            return self._reject_calibration(
                ValueError("select at least one pipeline step")
            )
        quick_optimization = (
            4 in selected and self.pipeline_quick_optimization_check.isChecked()
        )
        stage3_only = (
            4 in selected and self.pipeline_stage3_only_check.isChecked()
        )

        try:
            outputs = {
                step: self._pipeline_file_path(
                    self.pipeline_output_edits[step], f"Step {step} output"
                )
                for step in selected
                if step in (1, 2, 3)
            }
            csv_output = (
                self._pipeline_file_path(
                    self.pipeline_csv_edit, "Step 3 CSV output"
                )
                if 3 in selected
                else None
            )
            external_inputs: dict[int, Path] = {}
            if 2 in selected and 1 not in selected:
                external_inputs[2] = self._pipeline_file_path(
                    self.pipeline_input_edits[2],
                    "Step 2 input",
                    must_exist=True,
                )
            if 3 in selected and 2 not in selected:
                external_inputs[3] = self._pipeline_file_path(
                    self.pipeline_input_edits[3],
                    "Step 3 input",
                    must_exist=True,
                )

            optimization_calibration_input = None
            profile_path = None
            direct_initial_profile = None
            optimization_root = None
            optimization_run_name = None
            quick_calibration_output = None
            quick_calibration_input = None
            if 4 in selected:
                optimization_source_step = 2 if quick_optimization else 3
                if optimization_source_step not in selected:
                    optimization_calibration_input = self._pipeline_file_path(
                        self.pipeline_input_edits[4],
                        f"Encoding Optimization Step {optimization_source_step} input",
                        must_exist=True,
                    )
                if quick_optimization and stage3_only:
                    quick_calibration_input = self._pipeline_file_path(
                        self.pipeline_reopt_calibration_edit,
                        "Quick centre calibration input",
                        must_exist=True,
                    )
                elif quick_optimization:
                    quick_calibration_output = self._pipeline_file_path(
                        self.pipeline_quick_calibration_edit,
                        "Quick centre calibration output",
                    )
                if stage3_only:
                    profile_path = self._pipeline_file_path(
                        self.pipeline_reopt_profile_edit,
                        "Stage 1 level/profile data",
                        must_exist=True,
                    )
                elif self.pipeline_profile_source_combo.currentIndex() == 1:
                    profile_path = self._pipeline_file_path(
                        self.pipeline_profile_edit,
                        "Encoding Optimization initial profile",
                        must_exist=True,
                    )
                else:
                    direct_initial_profile = self._parse_pipeline_initial_profile(
                        self.pipeline_profile_values_edit.text()
                    )
                optimization_root = self._pipeline_directory_path(
                    self.pipeline_optimization_root_edit,
                    "Encoding Optimization output root",
                )
                optimization_run_name = (
                    self.pipeline_optimization_name_edit.text().strip() or None
                )
                if optimization_run_name is not None and (
                    Path(optimization_run_name).name != optimization_run_name
                    or optimization_run_name in (".", "..")
                ):
                    raise ValueError("optimization run name must be one directory name")

            output_paths = list(outputs.values())
            if csv_output is not None:
                output_paths.append(csv_output)
            if quick_calibration_output is not None:
                output_paths.append(quick_calibration_output)
            if len(set(output_paths)) != len(output_paths):
                raise ValueError("every selected pipeline output must use a unique file")
            external_file_paths = list(external_inputs.values())
            if optimization_calibration_input is not None:
                external_file_paths.append(optimization_calibration_input)
            if quick_calibration_input is not None:
                external_file_paths.append(quick_calibration_input)
            if profile_path is not None:
                external_file_paths.append(profile_path)
            collisions = set(output_paths).intersection(external_file_paths)
            if collisions:
                collision = next(iter(collisions))
                raise ValueError(
                    f"external input cannot also be a pipeline output: {collision}"
                )

            # A bare wavelength-map CSV does not carry min/max levels. Reuse the
            # existing Step 3 panel values instead of maintaining duplicate state.
            csv_min_level = self.step_widgets[3]["min"].value()
            csv_max_level = self.step_widgets[3]["max"].value()
            if csv_max_level < csv_min_level:
                raise ValueError("Step 3 CSV max level must be >= min level")

            # Validate external files before starting hardware acquisition. The worker
            # reloads them again at the actual step boundary.
            for step, input_path in external_inputs.items():
                self._load_pipeline_input(
                    step,
                    input_path,
                    csv_min_level=csv_min_level,
                    csv_max_level=csv_max_level,
                )

            if 4 in selected:
                if profile_path is not None:
                    self._load_pipeline_initial_profile(profile_path)
                if quick_calibration_input is not None:
                    self._load_pipeline_quick_intensity_calibration(
                        quick_calibration_input
                    )
                if optimization_calibration_input is not None:
                    if quick_optimization:
                        self._load_pipeline_wavelength_calibration(
                            optimization_calibration_input,
                            target_wavelength_nm=self.enc_center_wl_spin.value(),
                        )
                    else:
                        self._load_pipeline_optimization_calibration(
                            optimization_calibration_input
                        )

            settings = {
                step: self._step_settings(step)
                for step in selected
                if step in (1, 2, 3)
            }
            levels1 = self._step_levels(1) if 1 in selected else None
            levels3 = self._step_levels(3) if 3 in selected else None
            quick_levels = (
                self._parse_pipeline_quick_levels(
                    self.pipeline_quick_levels_edit.text()
                )
                if quick_optimization and not stage3_only
                else None
            )
            window2 = self.step_widgets[2]["window"].value() if 2 in selected else None
            peak_nm = (
                self.step_widgets[2]["peak_nm"].value() or None
                if 2 in selected
                else None
            )
            region2 = self._step_region(2) if 2 in selected else None
            window3 = self.step_widgets[3]["window"].value() if 3 in selected else None
            avg_nm = (
                self.step_widgets[3]["avg_nm"].value() or None
                if 3 in selected
                else None
            )
            sweep_nm = (
                self.step_widgets[3]["sweep_nm"].value() or None
                if 3 in selected
                else None
            )
            stride = self.step_widgets[3]["stride"].value() if 3 in selected else None
            refine = (
                self.step_widgets[3]["refine"].isChecked()
                if 3 in selected
                else None
            )
            region3 = self._step_region(3) if 3 in selected else None
            if 4 in selected:
                center_wl = self.enc_center_wl_spin.value()
                channel_width = self.enc_width_spin.value()
                gap_px = self.enc_pad_spin.value()
                if channel_width != 15:
                    raise ValueError(
                        "Encoding Optimization requires a 15 px channel width"
                    )
                ana = self._ana_settings()
                opt_sensitivity = (
                    self.pipeline_reopt_sensitivity_combo.currentText()
                    if stage3_only
                    else ana.sensitivity
                )
                optimization_settings = MeasurementSettings(
                    center_wl=f"{center_wl:g}nm",
                    span="0.8nm",
                    sensitivity=opt_sensitivity,
                    sampling_points="1001",
                    y_unit=ana.y_unit,
                    reference_level=ana.reference_level,
                    trace_id=ana.trace_id,
                    trace_mode=ana.trace_mode,
                )
                assert optimization_root is not None
                optimization_config = OSAOptimizationConfig(
                    settings=optimization_settings,
                    anchor_offsets=(0,) if quick_optimization else (0, -10, 10),
                    full_validation=not quick_optimization,
                    output_root=str(optimization_root),
                    run_name=optimization_run_name,
                    averages=(
                        self.pipeline_reopt_averages_spin.value()
                        if stage3_only
                        else OSAOptimizationConfig.averages
                    ),
                    rerank_averages=(
                        self.pipeline_reopt_rerank_averages_spin.value()
                        if stage3_only
                        else OSAOptimizationConfig.rerank_averages
                    ),
                    stage2_repeats=(
                        self.pipeline_reopt_stage2_repeats_spin.value()
                        if stage3_only
                        else OSAOptimizationConfig.stage2_repeats
                    ),
                    stage3_maxfev=(
                        self.pipeline_reopt_stage3_maxfev_spin.value()
                        if stage3_only
                        else OSAOptimizationConfig.stage3_maxfev
                    ),
                    skip_stage1=stage3_only,
                )
                if quick_optimization:
                    quick_settings = self._step_settings(3)
                    quick_window = self.step_widgets[3]["window"].value()
                    quick_avg_nm = self.step_widgets[3]["avg_nm"].value() or None
                    quick_sweep_nm = (
                        self.step_widgets[3]["sweep_nm"].value() or None
                    )
                else:
                    quick_settings = None
                    quick_window = None
                    quick_avg_nm = None
                    quick_sweep_nm = None
            else:
                center_wl = None
                channel_width = None
                gap_px = None
                optimization_config = None
                quick_settings = None
                quick_window = None
                quick_avg_nm = None
                quick_sweep_nm = None
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return self._reject_calibration(exc)

        controller = self._controller()
        if 4 in selected and not getattr(controller, "is_open", False):
            return self._reject_calibration(
                ValueError("open the SLM before running Encoding Optimization")
            )
        stage_names = {
            1: "Step 1",
            2: "Step 2",
            3: "Step 3",
            4: (
                "Quick Stage 3 Re-optimization"
                if quick_optimization and stage3_only
                else (
                    "Stage 3 Re-optimization"
                    if stage3_only
                    else (
                        "Quick Single-Channel Optimization"
                        if quick_optimization
                        else "Encoding Optimization"
                    )
                )
            ),
        }
        sequence = " -> ".join(stage_names[step] for step in selected)
        self.pipeline_status_label.setText(f"Running steps {sequence}")
        self.pipeline_log.appendPlainText(f"Starting pipeline: {sequence}")
        self._log(f"Pipeline started (steps {sequence}, file-only handoffs)")

        def work(report: ProgressEmit, stop_event: threading.Event) -> dict[str, Any]:
            result: CalibrationResult | None = None
            optimization_result: OptimizationResult | None = None
            optimization_layout: ChannelLayout | None = None
            quick_target_coordinate: float | None = None
            quick_measured_range: tuple[int, int] | None = None
            saved_files: list[Path] = []

            if 1 in selected:
                assert levels1 is not None
                _mn, _mx, min_level, max_level, _rec = find_min_max_intensity_levels(
                    osa,
                    controller,
                    levels1,
                    settings[1],
                    stop_event=stop_event,
                    progress_callback=report,
                )
                result = CalibrationResult(
                    wavelength=np.asarray([]),
                    coordinates=np.asarray([]),
                    max_level=max_level,
                    min_level=min_level,
                    level_range=np.asarray(levels1, dtype=int),
                )
                saved_files.append(save_calibration_result(result, outputs[1]))

            if 2 in selected:
                input_path = outputs[1] if 1 in selected else external_inputs[2]
                seed = self._load_pipeline_input(
                    2,
                    input_path,
                    csv_min_level=csv_min_level,
                    csv_max_level=csv_max_level,
                )
                result = wavelength_calibration(
                    osa,
                    controller,
                    [],
                    settings[2],
                    seed,
                    window_size=window2,
                    peak_half_window_nm=peak_nm,
                    region=region2,
                    stop_event=stop_event,
                    progress_callback=report,
                )
                saved_files.append(save_calibration_result(result, outputs[2]))

            if 3 in selected:
                input_path = outputs[2] if 2 in selected else external_inputs[3]
                mapping = self._load_pipeline_input(
                    3,
                    input_path,
                    csv_min_level=csv_min_level,
                    csv_max_level=csv_max_level,
                )
                assert levels3 is not None and csv_output is not None
                result = intensity_calibration(
                    osa,
                    controller,
                    levels3,
                    settings[3],
                    mapping,
                    window_size=window3,
                    wavelength_window_nm=avg_nm,
                    sweep_span_nm=sweep_nm,
                    coordinate_stride=stride,
                    refine_wavelength=refine,
                    region=region3,
                    stop_event=stop_event,
                    progress_callback=report,
                )
                saved_files.append(save_calibration_result(result, outputs[3]))
                csv_path = write_intensity_calibration_csv(result, csv_output)
            else:
                csv_path = None

            if 4 in selected:
                assert center_wl is not None
                assert channel_width is not None
                assert gap_px is not None
                assert optimization_config is not None
                if quick_optimization:
                    step2_path = (
                        outputs[2]
                        if 2 in selected
                        else optimization_calibration_input
                    )
                    assert step2_path is not None
                    step2_calibration = self._load_pipeline_wavelength_calibration(
                        step2_path, target_wavelength_nm=center_wl
                    )
                    quick_target_coordinate = interpolate_coordinate_for_wavelength(
                        step2_calibration, center_wl
                    )
                    target_pixel = int(round(quick_target_coordinate))
                    if stage3_only:
                        assert quick_calibration_input is not None
                        report(
                            CalibrationProgress(
                                phase="quick_center",
                                step=0,
                                total=1,
                                message=(
                                    f"{center_wl:g} nm -> x={quick_target_coordinate:.3f} "
                                    f"px; reusing saved quick calibration"
                                ),
                                x=quick_target_coordinate,
                                y=center_wl,
                            )
                        )
                        optimization_calibration = (
                            self._load_pipeline_quick_intensity_calibration(
                                quick_calibration_input
                            )
                        )
                        quick_measured_range = (
                            int(optimization_calibration.min_level),
                            int(optimization_calibration.max_level),
                        )
                    else:
                        assert quick_calibration_output is not None
                        assert quick_levels is not None
                        assert quick_settings is not None
                        assert quick_window is not None
                        report(
                            CalibrationProgress(
                                phase="quick_center",
                                step=0,
                                total=1,
                                message=(
                                    f"{center_wl:g} nm -> x={quick_target_coordinate:.3f} "
                                    f"px; calibrating pixel {target_pixel}"
                                ),
                                x=quick_target_coordinate,
                                y=center_wl,
                            )
                        )
                        quick_seed = CalibrationResult(
                            wavelength=np.asarray([center_wl], dtype=float),
                            coordinates=np.asarray([target_pixel], dtype=float),
                            max_level=int(quick_levels[-1]),
                            min_level=int(quick_levels[0]),
                            level_range=np.asarray(quick_levels, dtype=int),
                            wavelength_fit_coefficients=(
                                step2_calibration.wavelength_fit_coefficients
                            ),
                        )
                        quick_result = intensity_calibration(
                            osa,
                            controller,
                            quick_levels,
                            quick_settings,
                            quick_seed,
                            window_size=quick_window,
                            wavelength_window_nm=quick_avg_nm,
                            sweep_span_nm=quick_sweep_nm,
                            coordinate_stride=1,
                            refine_wavelength=False,
                            region=None,
                            stop_event=stop_event,
                            progress_callback=report,
                        )
                        quick_result = restrict_to_measured_intensity_range(
                            quick_result
                        )
                        quick_measured_range = (
                            int(quick_result.min_level),
                            int(quick_result.max_level),
                        )
                        saved_files.append(
                            save_calibration_result(
                                quick_result, quick_calibration_output
                            )
                        )
                        optimization_calibration = (
                            self._load_pipeline_quick_intensity_calibration(
                                quick_calibration_output
                            )
                        )
                    optimization_layout, quick_target_coordinate = (
                        build_single_anchor_layout(
                            step2_calibration,
                            optimization_calibration,
                            target_wavelength_nm=center_wl,
                            channel_width_px=channel_width,
                            gap_px=gap_px,
                        )
                    )
                else:
                    calibration_path = (
                        outputs[3]
                        if 3 in selected
                        else optimization_calibration_input
                    )
                    assert calibration_path is not None
                    optimization_calibration = (
                        self._load_pipeline_optimization_calibration(calibration_path)
                    )
                    optimization_layout = self._build_pipeline_encoding_layout(
                        optimization_calibration,
                        center_wl=center_wl,
                        channel_width_px=channel_width,
                        gap_px=gap_px,
                    )
                if profile_path is not None:
                    initial_l = self._load_pipeline_initial_profile(profile_path)
                else:
                    assert direct_initial_profile is not None
                    initial_l = direct_initial_profile.copy()

                def report_optimization(progress: OptimizationProgress) -> None:
                    self.edge_optimization_progress.emit(progress)
                    report(
                        CalibrationProgress(
                            phase=f"optimization: {progress.stage}",
                            step=progress.step,
                            total=max(progress.total, 1),
                            message=progress.message,
                            x=float(progress.step),
                            y=progress.best_loss,
                        )
                    )

                try:
                    optimization_result = optimize_from_osa(
                        optimization_layout,
                        osa=osa,
                        slm=controller,
                        initial_l=initial_l,
                        config=optimization_config,
                        stop_event=stop_event,
                        progress_callback=report_optimization,
                    )
                except OptimizationAborted:
                    return {"status": "aborted"}
                result = optimization_calibration
                saved_files.append(
                    Path(optimization_result.run_dir, "final_result.json").resolve()
                )

            if result is None:  # guarded by the non-empty selection check
                raise RuntimeError("pipeline completed without a result")
            return {
                "status": "ok",
                "step": "pipeline",
                "result": result,
                "saved": saved_files[-1],
                "saved_files": saved_files,
                "csv": csv_path,
                "optimization_result": optimization_result,
                "optimization_layout": optimization_layout,
                "quick_target_coordinate": quick_target_coordinate,
                "quick_measured_range": quick_measured_range,
                "summary": f"completed steps {sequence}",
            }

        self._pipeline_optimization_active = 4 in selected
        if self._pipeline_optimization_active:
            self._edge_gain_running(True)
            self.edge_gain_bar.setRange(0, 0)
            self.edge_gain_status.setText("Encoding Optimization running from Pipeline")
        self._launch_calibration("Run pipeline", work)
        if self._pipeline_optimization_active:
            self.edge_gain_stop_event = self.calibration_stop_event

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
        self._active_calibration_label = label
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
        active_label = self._active_calibration_label
        self._active_calibration_label = None
        self.calibration_stop_event = None
        self._set_calibration_running(False)
        if payload.get("status") == "aborted":
            self._log("Calibration stopped")
            if self._pipeline_optimization_active:
                self._pipeline_optimization_active = False
                self.edge_gain_stop_event = None
                self._edge_gain_running(False)
                self.edge_gain_bar.setRange(0, 100)
                self.edge_gain_bar.setValue(0)
                self.edge_gain_status.setText(
                    "Pipeline optimization stopped; checkpoints were retained."
                )
            if active_label == "Run pipeline":
                self.pipeline_status_label.setText("Stopped")
                self.pipeline_log.appendPlainText("Pipeline stopped")
            if active_label == "Stage 3 re-optimization":
                self.stage3_reopt_status_label.setText("Stopped")
                self.edge_gain_stop_event = None
                self._edge_gain_running(False)
                self.edge_gain_bar.setRange(0, 100)
                self.edge_gain_bar.setValue(0)
                self.edge_gain_status.setText(
                    "Stage 3 re-optimization stopped; checkpoints were retained."
                )
            if active_label == "Fast channel calibration":
                self.fast_channel_status_label.setText("Stopped")
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
        elif step == "pipeline":
            self.pipeline_status_label.setText(f"Done - {summary}")
            self.pipeline_log.appendPlainText(f"Done: {summary}")
            for path in payload.get("saved_files", []):
                self.pipeline_log.appendPlainText(f"Saved: {path}")
            quick_target_coordinate = payload.get("quick_target_coordinate")
            if quick_target_coordinate is not None:
                self.pipeline_log.appendPlainText(
                    "Quick target: "
                    f"{self.enc_center_wl_spin.value():g} nm -> "
                    f"x={float(quick_target_coordinate):.3f} px "
                    f"(physical pixel {int(round(float(quick_target_coordinate)))})"
                )
            quick_measured_range = payload.get("quick_measured_range")
            if quick_measured_range is not None:
                self.pipeline_log.appendPlainText(
                    "Quick measured intensity range: "
                    f"off level {int(quick_measured_range[0])}, "
                    f"on level {int(quick_measured_range[1])}"
                )
            optimization_result = payload.get("optimization_result")
            optimization_layout = payload.get("optimization_layout")
            if optimization_result is not None and optimization_layout is not None:
                self._pipeline_optimization_active = False
                self._enc_calib_override = result
                self.encoding_layout = optimization_layout
                self._enc_populate_val_table(optimization_layout)
                self._edge_sync_layout(optimization_layout)
                self.enc_calib_label.setText("Calibration: Pipeline calibration JSON")
                self.enc_layout_status.setText(
                    f"{optimization_layout.n_channels} ch/side  |  "
                    f"pitch {optimization_layout.pitch_px} px  |  "
                    f"pad {optimization_layout.pitch_px - optimization_layout.channel_width_px} px"
                )
                self.enc_generate_button.setEnabled(True)
                self._edge_optimization_finished(
                    {"status": "ok", "result": optimization_result}
                )
                if optimization_result.accepted:
                    message = (
                        "Accepted encoding profile and LUT applied to the Encoding "
                        "and Shape pages."
                    )
                else:
                    message = (
                        "Optimization was not accepted; the layout was loaded but "
                        "the candidate profile and LUT were not applied."
                    )
                self.pipeline_log.appendPlainText(message)
        elif step == "stage3_reopt":
            self.stage3_reopt_status_label.setText(f"Done - {summary}")
            quick_target_coordinate = payload.get("quick_target_coordinate")
            quick_measured_range = payload.get("quick_measured_range")
            if quick_target_coordinate is not None:
                self._log(
                    "Stage 3 reopt target: "
                    f"x={float(quick_target_coordinate):.3f} px "
                    f"(physical pixel {int(round(float(quick_target_coordinate)))})"
                )
            if quick_measured_range is not None:
                self._log(
                    "Stage 3 reopt quick range: "
                    f"off {int(quick_measured_range[0])}, "
                    f"on {int(quick_measured_range[1])}"
                )
            optimization_result = payload.get("optimization_result")
            optimization_layout = payload.get("optimization_layout")
            if optimization_result is not None and optimization_layout is not None:
                self.edge_gain_stop_event = None
                self._enc_calib_override = result
                self.encoding_layout = optimization_layout
                self._enc_populate_val_table(optimization_layout)
                self._edge_sync_layout(optimization_layout)
                self.enc_calib_label.setText(
                    "Calibration: Stage 3 reopt quick calibration"
                )
                self.enc_layout_status.setText(
                    f"Stage 3 reopt layout: {optimization_layout.n_channels} "
                    f"channels/side, width {optimization_layout.channel_width_px} px, "
                    f"pitch {optimization_layout.pitch_px} px"
                )
                self.enc_generate_button.setEnabled(True)
                self._edge_optimization_finished(
                    {"status": "ok", "result": optimization_result}
                )
        elif step == "fast_channels":
            self.fast_channel_status_label.setText(f"Done - {summary}")
            if saved is not None and not self.fast_channel_json_edit.text().strip():
                self.fast_channel_json_edit.setText(str(saved))
            csv_saved = payload.get("csv")
            if csv_saved is not None and not self.fast_channel_csv_edit.text().strip():
                self.fast_channel_csv_edit.setText(str(csv_saved))
            center_coordinate = payload.get("center_coordinate")
            if center_coordinate is not None:
                self._log(
                    "Fast channel center: "
                    f"x={float(center_coordinate):.3f} px "
                    f"(physical pixel {int(round(float(center_coordinate)))})"
                )
            measured_peak = payload.get("measured_peak")
            coarse_center = payload.get("coarse_center")
            if measured_peak is not None and coarse_center is not None:
                self._log(
                    "Fast channel OSA fine tune: "
                    f"coarse x={float(coarse_center):.3f} px, "
                    f"measured peak {float(measured_peak):.4f} nm"
                )
        if saved is not None:
            self._log(f"Saved {saved}")

        if step == "all":
            label = "Run all"
        elif step == "pipeline":
            label = "Pipeline"
        elif step == "stage3_reopt":
            label = "Stage 3 re-optimization"
        elif step == "fast_channels":
            label = "Fast channel calibration"
        else:
            label = f"Step {step}"
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
        active_label = self._active_calibration_label
        self._active_calibration_label = None
        self.calibration_stop_event = None
        self._set_calibration_running(False)
        if active_label == "Run pipeline":
            self.pipeline_status_label.setText("Failed")
            self.pipeline_log.appendPlainText("Pipeline failed; see the main log.")
        if active_label == "Stage 3 re-optimization":
            self.stage3_reopt_status_label.setText("Failed")
            self.edge_gain_stop_event = None
            self._edge_gain_running(False)
            self.edge_gain_bar.setRange(0, 100)
            self.edge_gain_bar.setValue(0)
            self.edge_gain_status.setText("Stage 3 re-optimization failed.")
        if active_label == "Fast channel calibration":
            self.fast_channel_status_label.setText("Failed")
        if self._pipeline_optimization_active:
            self._pipeline_optimization_active = False
            self.edge_gain_stop_event = None
            self._edge_gain_running(False)
            self.edge_gain_bar.setRange(0, 100)
            self.edge_gain_bar.setValue(0)
            self.edge_gain_status.setText("Pipeline optimization failed.")
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
        if self.daq_controller is not None:
            try:
                self.daq_controller.disconnect()
            except Exception:
                pass
            self.daq_controller = None
        super().closeEvent(event)

    def _apply_style(self) -> None:
        self.setStyleSheet(DARK_STYLESHEET)


def main(argv: list[str] | None = None) -> int:
    app = QtWidgets.QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("Santec SLM Control")
    window = MainWindow()
    window.show()
    return app.exec_()
