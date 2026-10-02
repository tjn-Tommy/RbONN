"""The SLM control suite: one PyQt5 window over every instrument on the bench.

In:  the drivers -- ``slm_module``, ``osa_module``, ``daq_module``,
     ``heater_module``, ``psu_module`` -- and ``calibration_module`` for the
     sweeps and the fits.  This file drives them and defines no physics and no
     file format of its own.
Out: the window, and whatever CSV/JSON/PNG each calibration step writes through
     ``calibration_module``.  No page parses a calibration by hand: they all go
     through ``calibration_module.calib_file``, one run's ``calib.json``.

A 220 px nav list switches a stack of pages.  Connections and SLM Control first,
then the read-side instrument pages (OSA Viewer, DAQ Monitor, Heater, PMT Gain)
beside them because they are reached far more often than the Calibration page
under them, which carries the step chain as tabs: 1 Min/Max, 2 Wavelength,
3 Channels (DAQ), 6 TPA Efficiency, 7 Comb Phase, 8 Saturation.  Several pages
are BUILT BUT NOT TABBED -- the OSA step 3b, Mod Error, step 4b, step 6b -- each
saying at its own construction why; they stay built because other pages read
their widgets.

Every instrument call goes through :meth:`MainWindow._run_task`, which runs it on
a ``FunctionWorker`` thread and delivers the result on the GUI thread, so no page
blocks the event loop.  The v2 sweep drivers in ``calibration_module.measure``
take a ``progress_callback`` and a ``stop_event``, which is how one
implementation serves both this GUI's progress bar and Stop button and the
offline step scripts.

The step pages share one contract, in the ``_STEP_*`` maps on
:class:`MainWindow`: each names a ``calib.json``, declares what that file must
already contain before its Run is enabled, and describes what it found.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
import threading
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
import matplotlib
from PyQt5 import QtCore, QtGui, QtWidgets

from daq_module.controller import (
    DAQController,
    DAQMonitorSettings,
    settle_samples,
    waveform_stats,
)
from heater_module.controller import (
    HeaterCycle,
    PID_DEFAULTS as HEATER_PID_DEFAULTS,
    StaircaseSettings,
    TC300Controller,
)
from osa_module.controller import MeasurementSettings, OSAController
from osa_module.driver import OSAError
from psu_module.controller import (
    PMT_CONTROL_I_MAX,
    PMT_CONTROL_V_MAX,
    PSUController,
    PSUState,
    SET_TOL_V,
)
from calibration_module.fit.legacy_intensity import CalibrationFit, fit_calibration, load_calibration_csv
from calibration_module.fit.slm_response import (
    load_step2_map,
    CalibrationResult,
    calibration_result_from_dict,
    load_calibration_result,
    load_wavelength_map_csv,
    save_calibration_result,
    step1_section,
    step2_section,
    step3_section,
    write_intensity_calibration_csv,
    write_wavelength_map_csv,
)
from calibration_module.measure.slm_response import (
    CalibrationAborted,
    CalibrationProgress,
    run_step1,
    run_step2,
    run_step3b,
    run_step3c,
)
from slm_module.controller import ScanParams, ScanResult, SLMController
from slm_module.detector import Detector, SimulatedDetector
from slm_module.generator import (
    MAX_LEVEL,
    equal_segment_edges,
    make_equal_segments,
    make_vertical_window,
    make_segments,
    write_santec_csv,
)
from slm_module.analysis import (
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
from slm_module.encoding import (
    ChannelLayout,
    build_channel_layout,
    build_single_anchor_layout,
    channel_layout_from_calibration,
    encode_to_pattern,
    interpolate_coordinate_for_wavelength,
    optimize_from_osa,
)
from slm_module.optimization import (
    OPTIMIZED_ENCODING_SHAPE,
    OSABatchVariant,
    OSAOptimizationConfig,
    OptimizationAborted,
    OptimizationProgress,
    OptimizationResult,
    independent_intensity_profile,
    amplitudes_to_intensity_commands,
    load_optimization_result,
    mirror_intensity_profile,
    run_osa_optimization_batch,
    validate_independent_profile,
)
from calibration_module.calib_file import (
    CALIB_NAME as CALIB_FILE_NAME,
    STEP_ORDER as CALIB_STEP_ORDER,
    calib_path as calib_file_path,
    load_calib,
    write_section,
)
from calibration_module.fit.pair_v2 import (
    PairV2Config,
    PairV2Fit,
    average_levels,
    sweep_floor,
    step6_floor,
    burst_rows,
    fit_pair,
    load_meas_csv,
    save_step6_section,
    write_meas_csv,
)
from calibration_module.fit.outliers import BURST_MAX_V
from calibration_module.measure.pair_v2 import (
    DEFAULT_ACQ as BENCH_ACQ,
    PairV2Aborted,
    PairV2Acq,
    PairV2Progress,
    build_schedule,
    measure_pair,
    run_seconds,
)
from calibration_module.fit.center import TPACenterResult, average_trace_points
from calibration_module.measure.center import (
    TPACenterAborted,
    TPACenterProgress,
    measure_center_scan,
)
from calibration_module.fit.phase import (
    PairModel,
    PhaseResult,
    fit_result,
    load_pair_models,
    drop_bursts,
    guard_darks,
    load_phase_csv,
    save_step7_section,
    references_in_csv,
    targets_in_csv,
    write_meas_csv as write_phase_meas_csv,
)
from calibration_module.fit.cascade import (
    MIN_PAIR_SEP,
    Stage,
    chain_fits,
    plan_cascade,
    plan_lines,
)
from calibration_module.measure.phase_v2 import (
    PhaseV2Aborted,
    PhaseV2Config,
    PhaseV2Progress,
    build_xw_sweep,
    measure_target,
    run_seconds as phase_run_seconds,   # NOT pair_v2's: a step-7 Drive is a 4-tuple
)
from calibration_module.fit.pair_v2 import meas_csv_meta
from calibration_module.measure.pmt_temp import open_pmt
from calibration_module.fit.report import plot_fringe
from calibration_module.measure.patterns import (
    PatternsAborted,
    PatternsProgress,
    range_for,
    run_patterns,
)
from calibration_module.fit.step8_report import print_result as step8_print_result
from calibration_module.fit.verify import (
    load_forward_model as verify_load_forward_model,
    predict as verify_predict,
    write_verify_csv as verify_write_csv,
)
from calibration_module.steps import calib_step8 as STEP8
from slm_module.keepalive import SLMKeepAlive
from .common import (
    CalibrationProgressDialog,
    FunctionWorker,
    WorkerSignals,
    _format_duration,
)
from .live_plots import BatchResultsTable, LiveLossCanvas
from .osa_monitor import LiveSpectrumView, OSATraceBridge
from .style import DARK_STYLESHEET


#: Both acquisition tabs offer the same switch, so they explain it the same way.
PMT_CHECKBOX_TIP = (
    "Record the PMT thermistor (Keithley 2400) next to every row, as "
    "pmt_ohm_Ohm and pmt_c in the saved CSV.\n"
    "The 2400 is read at most once a minute and the rows in between repeat the "
    "last reading \N{EM DASH} the bead's time constant is minutes, so a "
    "per-row round trip would cost bench time and measure nothing extra.\n"
    "One GPIB bus, one 2400: do NOT run daq_pmt_monitor.py at the same time. "
    "A missing or failing 2400 is not an error, it just leaves the two columns "
    "blank."
)


def _pmt_meta_start(pmt) -> dict[str, Any]:
    """Open a sweep's CSV meta with the thermistor, seeding its cache.

    The first read is taken BEFORE the first acquisition so row 1 carries a
    temperature rather than the blank of an unprimed cache.
    """
    if pmt is None:
        return {}
    first = pmt.read()
    meta: dict[str, Any] = {"pmt_resource": pmt.resource,
                            "pmt_period_s": f"{pmt.period_s:g}"}
    meta.update({f"{k}_start": v for k, v in first.items()})
    return meta


def _pmt_meta_end(pmt, meta: dict[str, Any]) -> None:
    """Close it with a final reading, so an aborted sweep is bracketed too."""
    if pmt is not None:
        meta.update({f"{k}_end": v for k, v in pmt.read().items()})


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


class StatusLine(QtWidgets.QLabel):
    """A page's newest line of feedback, beside its title.  Never a history.

    It does not wrap.  A wrapped status is a paragraph that grows and shrinks
    under whatever it sits above, moving the rest of the page every time a
    message lands -- and two of them stacked read as a log, which this is not.
    So the text is kept whole and PAINTED elided, with all of it in the tooltip.

    ``text()`` returns the whole string rather than the part that fit: what the
    page said is not a function of how wide the window happens to be, and every
    reader of it -- the tests included -- wants the sentence.
    """

    def __init__(self, text: str = "") -> None:
        super().__init__()
        self.setObjectName("PageSubtitle")
        self.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        # Ignored: the label is as wide as the header row has left over, never
        # as wide as its longest message.
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored,
                           QtWidgets.QSizePolicy.Preferred)
        self._full = ""
        self.setText(text)

    def setText(self, text: str) -> None:
        self._full = text
        self.setToolTip(text)
        self._elide()

    def text(self) -> str:
        return self._full

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._elide()

    def _elide(self) -> None:
        shown = self.fontMetrics().elidedText(
            self._full, QtCore.Qt.ElideRight, max(0, self.width()))
        # Only on a real change: setText relayouts, a relayout can resize, and
        # a resize comes back here.
        if shown != QtWidgets.QLabel.text(self):
            super().setText(shown)


#: Every acquisition control on the step 6/7/8 pages starts from BENCH_ACQ, so
#: "the three steps ran the same way" is the default rather than a thing to
#: remember.  The boxes stay editable -- a bench needs an escape hatch -- but a
#: changed one is a deliberate departure from what the CLI will do.
BENCH_WINDOW_TIP = (
    "T_meas: the averaging window for EVERY read, the all-off dark included.\n"
    "Steps 6, 7 and 8 and the ONN loop all read at this one window, so a "
    "step-7 point and the step-6 \N{GREEK SMALL LETTER ETA} it is pinned "
    "against averaged the same noise \N{EM DASH} and a dark subtracted from a "
    "whole sweep does not carry its own, differently-averaged noise into every "
    "point of it."
)
BENCH_RANGE_TIP = (
    "Input range, for every read including the dark.\n"
    "Fixed at the amplifier's own limit: the TIA saturates at 1 V, so a read "
    "the board could only hold wider is already a flat-top and measuring it "
    "more precisely measures nothing. Board rail and amplifier rail coincide.\n"
    "Going narrower buys nothing \N{EM DASH} the board is dither-dominated even at "
    "\N{PLUS-MINUS SIGN}0.1 V, and quantization on the reported mean sits 2000x "
    "under the fit's own sigma floor."
)


class MainWindow(QtWidgets.QMainWindow):
    scan_progress = QtCore.pyqtSignal(int, int, str)
    scan_started = QtCore.pyqtSignal(int, int, int, int)
    scan_sample = QtCore.pyqtSignal(float, float)
    keepalive_status = QtCore.pyqtSignal(bool, str)
    calibration_progress = QtCore.pyqtSignal(object)
    analysis_progress = QtCore.pyqtSignal(object)
    tpa_progress = QtCore.pyqtSignal(object)
    tpa_phase_progress = QtCore.pyqtSignal(object)
    tpa_center_progress = QtCore.pyqtSignal(object)
    step8_progress = QtCore.pyqtSignal(object)
    edge_gain_progress = QtCore.pyqtSignal(int, int, str)
    edge_optimization_progress = QtCore.pyqtSignal(object)
    qt_test_progress = QtCore.pyqtSignal(int, int, str)
    heater_sample = QtCore.pyqtSignal(object)
    daq_reading = QtCore.pyqtSignal(object)
    psu_sample = QtCore.pyqtSignal(object)

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
        # Step 6 / 6b / 7 run outside _launch_calibration; each names itself here
        # while it runs, so manual OSA/DAQ reads stay locked for every calibration
        self._running_tpa_calibrations: set[str] = set()
        self._daqmon_reading = False
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
        # Step 6 keeps the raw rows beside the fits: a re-fit under a different
        # config has to start from the measurement, not from a fitted result.
        self.tpa_fits: list[PairV2Fit] = []
        self.tpa_rows: dict[int, list] = {}
        # The last sweep's PMT-thermistor meta, written into the CSV on Save.
        # Cleared by a Load, whose rows came from someone else's sweep.
        self.tpa_meta: dict[str, Any] = {}
        self.tpa_phase_meta: dict[str, Any] = {}
        self.tpa_stop_event: threading.Event | None = None
        self.tpa_phase_results: dict[int, PhaseResult] = {}
        # Every fitted pair referred back to the ROOT reference, hops and all;
        # the results dict beside it still speaks in the hop each pair was
        # actually measured over.  Keyed the same, so a row reads both.
        self.tpa_phase_chains: dict[int, Any] = {}
        self.tpa_phase_stages: list[Stage] = []
        self.tpa_phase_root: int | None = None
        self.tpa_phase_stop_event: threading.Event | None = None
        self.tpa_center_result: TPACenterResult | None = None
        self.tpa_center_stop_event: threading.Event | None = None
        self.daq_controller: DAQController | None = None
        # DAQ Monitor continuous mode: one worker loop of back-to-back
        # finite reads owns the card while it runs (see _daq_stream_toggled)
        self.daq_stream_stop_event: threading.Event | None = None
        # Heater (Thorlabs TC300B): one serial link, so at most one background
        # loop (ramp or read-only monitor) owns it at a time.
        self.heater_controller: TC300Controller | None = None
        self.heater_stop_event: threading.Event | None = None
        self._heater_disconnect_pending = False
        self._heater_times: dict[int, list[float]] = {1: [], 2: []}
        self._heater_temps: dict[int, list[float]] = {1: [], 2: []}
        self._heater_volts: dict[int, list[float]] = {1: [], 2: []}
        # PSU (Korad KWR103) driving the PMT gain-control input. Unlike the
        # heater, a live monitor does NOT own the link: the driver locks the
        # port per exchange, so setpoints can be applied while it polls.
        self.psu_controller: PSUController | None = None
        self.psu_stop_event: threading.Event | None = None
        self._psu_disconnect_pending = False
        self._psu_busy = False        # a one-shot command (apply/output) in flight
        self.psu_state: PSUState | None = None   # last full snapshot from the supply

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
        self.tpa_phase_progress.connect(self._on_tpa_phase_progress)
        self.tpa_center_progress.connect(self._on_tpa_center_progress)
        self.step8_progress.connect(self._on_step8_progress)
        self.edge_gain_progress.connect(self._edge_gain_progress)
        self.edge_optimization_progress.connect(self._edge_optimization_progress)
        self.qt_test_progress.connect(self._qt_test_progress)
        self.heater_sample.connect(self._on_heater_sample)
        self.daq_reading.connect(self._on_daq_reading)
        self.psu_sample.connect(self._on_psu_sample)

        # Live OSA monitor: measure() notifies the bridge from the worker
        # thread; the queued signal repaints the viewer page and the dock for
        # EVERY sweep, no matter which module triggered it.
        self._osa_bridge = OSATraceBridge(self)
        self._osa_bridge.trace_ready.connect(self._osa_view_on_trace)
        self._osa_bridge.trace_ready.connect(self.osa_monitor_view.set_trace)


    def _build_ui(self) -> None:
        # Dockable live OSA monitor (hidden until toggled from the OSA Viewer
        # page); built first so page builders may reference it.
        self.osa_monitor_view = LiveSpectrumView(self)
        self.osa_monitor_dock = QtWidgets.QDockWidget("OSA Monitor", self)
        self.osa_monitor_dock.setObjectName("OSAMonitorDock")
        self.osa_monitor_dock.setWidget(self.osa_monitor_view)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.osa_monitor_dock)
        self.osa_monitor_dock.hide()

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
            ("\N{ELECTRIC PLUG}  Connections",
             "Connect SLM, OSA, DAQ, heater and PSU"),
            ("\N{LINK SYMBOL}  SLM Control", "Grayscale and CSV display"),
            # instrument pages sit next to SLM Control: they are the read-side
            # counterpart to driving the panel, and are reached far more often
            # than the calibration workflow below them.
            ("\N{SATELLITE ANTENNA}  OSA Viewer",
             "Live OSA spectrum viewer: single / continuous sweeps with settings"),
            ("\N{BAR CHART}  DAQ Monitor",
             "One-shot DAQ waveform diagnostic: trace, spectrum, filtered stats"),
            ("\N{THERMOMETER}  Heater",
             "Thorlabs TC300B: staircase ramp/hold + live temperature monitor"),
            ("\N{BATTERY}  PMT Gain",
             "Korad KWR103 in constant voltage: PMT gain-control setpoint, "
             "output and live V/I"),
            ("\N{CHART WITH UPWARDS TREND}  Calibration", "Min/max, wavelength, intensity, TPA"),
            ("\N{LEFT RIGHT ARROW}  Center Scan", "Sweep a window across x"),
            ("\N{TRIGRAM FOR HEAVEN}  Phase Segments", "Piecewise phase along x"),
            ("\N{HIGH VOLTAGE SIGN}  TPA Encoding", "Channel grid encoding"),
            ("\N{DOWNWARDS ARROW WITH TIP RIGHTWARDS}  Shape",
             "Global per-column encoding shape (applied to all encoding "
             "+ calibration) + OSA optimisation hook"),
            ("\N{WHITE HEAVY CHECK MARK}  Quick Test",
             "A/B crosstalk test: flat vs optimised encoding shape from OSA data"),
        )
        for label, tooltip in nav_items:
            item = QtWidgets.QListWidgetItem(label)
            item.setSizeHint(QtCore.QSize(180, 48))
            item.setToolTip(tooltip)
            self.nav.addItem(item)
        sidebar_layout.addWidget(self.nav, 1)

        # must stay in the same order as nav_items: currentRowChanged feeds
        # setCurrentIndex directly, so a row here is a row there.
        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self._build_connection_page())
        self.stack.addWidget(self._build_control_page())
        self.stack.addWidget(self._build_osa_viewer_page())
        self.stack.addWidget(self._build_daq_monitor_page())
        self.stack.addWidget(self._build_heater_page())
        self.stack.addWidget(self._build_psu_page())
        self.stack.addWidget(self._build_calibration_page())
        self.stack.addWidget(self._build_scan_page())
        self.stack.addWidget(self._build_segments_page())
        self.stack.addWidget(self._build_tpa_page())
        self.stack.addWidget(self._build_edge_ratio_page())
        self.stack.addWidget(self._build_quick_test_page())

        layout.addWidget(sidebar)
        layout.addWidget(self.stack, 1)
        self.setCentralWidget(central)
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav.setCurrentRow(0)

    def _build_connection_page(self) -> QtWidgets.QWidget:
        """First page: connect the SLM, OSA and DAQ, with a shared status log."""
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

        # ---- DAQ ----
        # The only reader of the PMT signal: nothing else contends for it.
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

        # ---- Heater (Thorlabs TC300B) ----
        heater = self._panel("Heater (Thorlabs TC300B)")
        htl = QtWidgets.QGridLayout(heater)
        self.heater_port_edit = QtWidgets.QLineEdit("COM3")
        self.heater_port_edit.setPlaceholderText("TC300 serial port (e.g. COM3)")
        self.heater_connect_button = QtWidgets.QPushButton("Connect Heater")
        self.heater_connect_button.clicked.connect(self._connect_heater)
        self.heater_disconnect_button = QtWidgets.QPushButton("Disconnect")
        self.heater_disconnect_button.setProperty("variant", "ghost")
        self.heater_disconnect_button.setEnabled(False)
        self.heater_disconnect_button.clicked.connect(self._disconnect_heater)
        self.heater_status_label = QtWidgets.QLabel("Heater: closed")
        self._set_status(self.heater_status_label, "Heater: closed", "off")
        htl.addWidget(QtWidgets.QLabel("Port"), 0, 0)
        htl.addWidget(self.heater_port_edit, 0, 1)
        htl.addWidget(self.heater_connect_button, 0, 2)
        htl.addWidget(self.heater_disconnect_button, 0, 3)
        htl.addWidget(self.heater_status_label, 0, 4)
        htl.setColumnStretch(1, 1)
        page.layout().addWidget(heater)

        # ---- PSU (Korad KWR103 -> PMT gain control) ----
        psu = self._panel("Power supply (Korad KWR103 \N{RIGHTWARDS ARROW} PMT gain)")
        pul = QtWidgets.QGridLayout(psu)
        self.psu_port_edit = QtWidgets.QLineEdit("COM4")
        self.psu_port_edit.setPlaceholderText("KWR103 serial port (e.g. COM4)")
        self.psu_connect_button = QtWidgets.QPushButton("Connect PSU")
        self.psu_connect_button.clicked.connect(self._connect_psu)
        self.psu_disconnect_button = QtWidgets.QPushButton("Disconnect")
        self.psu_disconnect_button.setProperty("variant", "ghost")
        self.psu_disconnect_button.setEnabled(False)
        self.psu_disconnect_button.clicked.connect(self._disconnect_psu)
        self.psu_status_label = QtWidgets.QLabel("PSU: closed")
        self._set_status(self.psu_status_label, "PSU: closed", "off")
        pul.addWidget(QtWidgets.QLabel("Port"), 0, 0)
        pul.addWidget(self.psu_port_edit, 0, 1)
        pul.addWidget(self.psu_connect_button, 0, 2)
        pul.addWidget(self.psu_disconnect_button, 0, 3)
        pul.addWidget(self.psu_status_label, 0, 4)
        pul.setColumnStretch(1, 1)
        page.layout().addWidget(psu)

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
        """Top-level step tabs; each step uses the full page."""
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)

        # per-step widget registry: self.step_widgets[step][key]
        self.step_widgets: dict[int | str, dict[str, Any]] = {
            1: {}, 2: {}, "3c": {},
        }

        # tabs in step order: 1, 2, 3b, 3c, 6, 7.  4b and 6b are built but
        # not tabbed -- see below.
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._build_step1_tab(), "Step 1 · Min/Max")
        tabs.addTab(self._build_step2_tab(), "Step 2 · Wavelength")
        # Step 3a (the per-coordinate sweep) is gone: the channel-grid DAQ scan
        # covers it, and that is what the encoder and steps 6-8 read.  Its
        # "Run All (1->2->3)" button and channel-map preview went with it.  The
        # fit/plot widgets it used to host did not: _on_step_finished feeds every
        # CSV-producing step into that flow, so the container is still built,
        # just with no page to sit on.
        self._fit_backing = self._build_step3_fit_backing()
        self._fit_backing.setVisible(False)
        # The OSA-based 3b is built but NOT tabbed, the way 4b and 6b are: the
        # DAQ scan below is the step 3 this bench runs, and two tabs for one step
        # is how a sweep ends up done through the wrong one.  See the README's
        # Legacy section.
        self._fast_channel_page = self._build_fast_channel_calibration_page()
        self._fast_channel_page.setVisible(False)
        tabs.addTab(self._build_step3c_page(), "Step 3 · Channels (DAQ)")
        # Mod Error page is built but not shown as a tab: the Encoding Gain
        # (TPA Encoding page) and Quick Test sweeps read its ana_* OSA
        # sweep-settings widgets.
        self._mod_error_page = self._build_analysis_page()
        # Step 4b (Stage-3 re-optimisation) is hidden for now.  Its result is
        # not wired to anything: the shape actually in use is the frozen
        # OPTIMIZED_ENCODING_SHAPE constant in optimization.py, transcribed by
        # hand from encoding/2026-07-03_run162902/best_so_far.json.  A 4b run
        # writes a new best_so_far.json under its own output root and nothing
        # reads it, so the page can only mislead until that link is built.
        # Kept built, not deleted: _set_calibration_running gates
        # stage3_reopt_stop_button and calibration_run_buttons holds
        # stage3_reopt_run_button, and re-tabbing is a one-line change.
        self._stage3_reopt_page = self._build_stage3_reopt_page()
        self._stage3_reopt_page.setVisible(False)
        tabs.addTab(self._build_tpa_tab(), "Step 6 · TPA Efficiency")
        # Step 6b (TPA centre scan) is hidden for the same reason, and its own
        # Apply button says so out loud: the encoding layout is read verbatim
        # from the Step 3c calibration, so a fitted centre only prints "re-run
        # Step 3c with this target centre".  A page that looks like it
        # re-centres the channels and does not.  Self-contained -- nothing
        # outside its own handlers touches these widgets.
        self._tpa_center_page = self._build_tpa_center_tab()
        self._tpa_center_page.setVisible(False)
        tabs.addTab(self._build_tpa_phase_tab(), "Step 7 · Comb Phase")
        tabs.addTab(self._build_step8_tab(), "Step 8 · Saturation")
        # There is deliberately no unified "run every stage" page.  Steps 1-3
        # read the OSA and steps 6-7 read the DAQ through the TPA cell, so the
        # bench has to be re-plumbed by hand partway down the chain: an
        # unattended 1->7 run is not physically possible.  Real runs drive
        # steps 6-8 from drafts/calib_step6-8_v2.py with step 3 loaded from an
        # earlier run's JSON -- see calib_data/archive/2026-09/run_0907_1724/sequence.json.
        self.calibration_tabs = tabs
        lay.addWidget(tabs)
        # Every page's field exists now, so each can describe its file and gate
        # its own buttons.  Done once here rather than from each page's
        # constructor: a page's Run button is sometimes built after its field.
        self._calib_describe_all()

        # every OSA-gated Run button, toggled together by
        # _set_calibration_running (the unified pipeline page gates its own
        # Run button: it may legitimately run without the OSA)
        self.calibration_run_buttons = [
            self.step_widgets[1]["run"],
            self.step_widgets[2]["run"],
            self.fast_channel_run_button,
            self.stage3_reopt_run_button,
        ]
        return page

    # ---- the calib file every page names -----------------------------
    #: What a step needs to find in its calib.json before it can run.
    #:
    #: Steps 1 and 2 need nothing.  Step 1 CREATES the file, and step 2's
    #: off/on levels come either from step 1 or from the manual pair, so a
    #: calibration whose levels were typed in is complete and never has a
    #: step 1 at all -- the same rule as ``calib_step2.py --min-max``.
    _STEP_NEEDS: dict[int | str, tuple[int, ...]] = {
        "3c": (2,), 6: (3,), 7: (3, 6), 8: (3, 6, 7),
    }

    #: Pages with something better to say about their file than its chain --
    #: step 6 counts the layout's pairs, step 7 the etas it will pin to, step 8
    #: the Y_max it places its band on.  One line per page either way.
    _STEP_DESCRIBE: dict[int | str, str] = {
        6: "_tpa_describe_step3", 7: "_tpa_phase_describe_step6",
        8: "_step8_step7_changed",
    }

    #: Pages whose file description goes in the page's STATUS LINE, beside the
    #: title, rather than in a pill under the file row.
    #:
    #: A green pill under the field said the file was readable, which is the one
    #: thing already obvious from the field having a path in it -- and it said so
    #: in the loudest place on the page, every time, forever.  The status line is
    #: where a page's newest line of feedback goes, and what a file holds is
    #: feedback about loading it.  Only these three have one; steps 1-3 keep the
    #: pill, which is also the only place they report a file that is missing.
    _STEP_STATUS: dict[int | str, str] = {
        6: "tpa_status", 7: "tpa_phase_status", 8: "step8_status",
    }

    #: The per-page Run buttons each step's file gates, by attribute name.
    #:
    #: Run only.  Load is not gated by this field and must not be: it ASKS
    #: which calibration to re-fit, and looking at another night's is the usual
    #: reason to press it -- so gating it on the field meant you could not open
    #: a file until you had already named a different one.
    #:
    #: Step 3's Run is not here either: it is DAQ-gated as well, so
    #: :meth:`_refresh_step3_run_button` owns it and reads the same requirement.
    _STEP_BUTTONS: dict[int | str, tuple[str, ...]] = {
        6: ("tpa_run_button",),
        7: ("tpa_phase_run_button",),
        8: ("step8_run_button",),
    }

    def _calib_file_row(self, step: int | str, *,
                        create: bool = False) -> QtWidgets.QWidget:
        """The one input a step page takes: the calib.json it appends to.

        One field per page, because the file IS the argument -- the same
        ``--calib`` every runner takes.  A single shared field sat in the
        calibration tab's header until 2026-10-01; it kept six pages in step by
        force, which also meant looking at last night's step 6 re-pointed
        step 8.  One per page, and loading a file is loading a file.

        ``create=True`` gives step 1 a SAVE dialog: it is the step that starts a
        calibration, so it names a file that need not exist yet, and the OS asks
        before replacing one that does -- which is the whole warning, since a
        new step 1 drops every section under it anyway.

        A typed directory is accepted and means its ``calib.json``, exactly as
        the runners' :func:`~calibration_module.calib_file.calib_path` does.
        """
        if not hasattr(self, "_calib_edits"):
            self._calib_edits: dict[int | str, QtWidgets.QLineEdit] = {}
            self._calib_labels: dict[int | str, QtWidgets.QLabel] = {}

        panel = self._panel("Calibration file")
        grid = QtWidgets.QGridLayout(panel)
        grid.setColumnStretch(1, 1)
        edit = QtWidgets.QLineEdit()
        edit.setPlaceholderText(
            "required — the calib.json this step writes into"
        )
        edit.setToolTip(
            "The one file this step reads and appends to, and whose folder its "
            "CSV lands in.\n"
            "Its step3 section IS the channel layout every level is encoded "
            "through, its step6 section the etas step 7 pins to, its step7 "
            "section the forward model step 8 drives."
        )
        browse = QtWidgets.QPushButton("Browse\N{HORIZONTAL ELLIPSIS}")
        browse.setProperty("variant", "ghost")
        browse.clicked.connect(
            (lambda: self._browse_new_calib_into(edit)) if create
            else (lambda: self._browse_calib_into(edit))
        )
        # The page's own status line if it has one (see _STEP_STATUS), so
        # the description lands beside the title; otherwise a pill under the
        # field.  Registered either way, so every writer stays one call.
        label = getattr(self, self._STEP_STATUS.get(step, ""), None)
        inline = label is None
        if inline:
            label = QtWidgets.QLabel("\N{EN DASH}")
            label.setObjectName("PageSubtitle")
            label.setWordWrap(True)
        grid.addWidget(QtWidgets.QLabel("File"), 0, 0)
        grid.addWidget(edit, 0, 1)
        grid.addWidget(browse, 0, 2)
        if inline:
            grid.addWidget(label, 1, 0, 1, 3)

        self._calib_edits[step] = edit
        self._calib_labels[step] = label
        edit.textChanged.connect(lambda _="": self._calib_file_changed(step))
        return panel

    def _open_calib(self, title: str, step: int | str) -> "Path | None":
        """Ask which calib.json to look at, starting where this page's lives.

        Load ASKS rather than taking the page's field, because looking at
        another night is the usual reason to press it -- and the field is where
        a SWEEP will write, which must not move because somebody opened a file.
        Everything the fit then needs comes out of the file that was picked: its
        own sweep CSV, its own step-3 layout, its own step-6 etas.  Nothing on
        screen afterwards is a mixture of two calibrations.
        """
        here = self._calib_file_path(step)
        start = (str(here.parent) if here is not None and here.parent.is_dir()
                 else str(self._default_calib_dir()))
        name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, title, start,
            f"Calibration ({CALIB_FILE_NAME});;JSON (*.json);;All files (*)")
        return calib_file_path(name) if name else None

    def _browse_calib_into(self, edit: QtWidgets.QLineEdit) -> None:
        """Open an existing calib.json, starting in src/calib_data."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open a calibration file", str(self._default_calib_dir()),
            f"Calibration ({CALIB_FILE_NAME});;JSON (*.json);;All files (*)")
        if path:
            edit.setText(path)

    def _browse_new_calib_into(self, edit: QtWidgets.QLineEdit) -> None:
        """Choose or create the calib.json step 1 starts.

        A Save dialog, so the folder can be made here and the OS asks before
        replacing a file that already exists.  Nothing is written until the
        sweep finishes: an accepted name is a name, and an abandoned step 1
        leaves no empty calibration behind.
        """
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Choose or create a calibration file",
            str(self._default_calib_dir() / CALIB_FILE_NAME),
            f"Calibration ({CALIB_FILE_NAME});;JSON (*.json)")
        if path:
            edit.setText(path)

    @staticmethod
    def _calib_name(path: Path) -> str:
        """``run_0930/calib.json`` -- the folder is half the identity of a file
        that is called calib.json in every one of them."""
        return f"{path.parent.name}/{path.name}"

    def _calib_file_path(self, step: int | str) -> "Path | None":
        """The calib.json this page names, or None when its field is empty."""
        edit = getattr(self, "_calib_edits", {}).get(step)
        text = "" if edit is None else edit.text().strip().strip('"')
        return calib_file_path(text) if text else None

    def _require_calib_file(self, step: int | str) -> Path:
        """The page's calib file, or a ValueError the Run handlers reject on."""
        path = self._calib_file_path(step)
        if path is None:
            raise ValueError("choose a calibration file for this step first")
        return path

    def _calib_sections(self, step: int | str) -> "set[int]":
        """Which sections this page's file has; empty set if it cannot be read."""
        path = self._calib_file_path(step)
        if path is None:
            return set()
        try:
            return set(load_calib(path).steps())
        except Exception:                                   # noqa: BLE001
            return set()

    def _calib_missing(self, step: int | str) -> "list[int]":
        """Sections ``step`` needs that its file has not got, in order."""
        have = self._calib_sections(step)
        return [n for n in self._STEP_NEEDS.get(step, ()) if n not in have]

    def _calib_file_changed(self, step: int | str) -> None:
        """Re-describe this page's file, then re-gate this page's buttons."""
        hook = getattr(self, self._STEP_DESCRIBE.get(step, ""), None)
        if hook is not None:
            hook()
            self._calib_gate_step(step)
            return
        label = getattr(self, "_calib_labels", {}).get(step)
        path = self._calib_file_path(step)
        if label is not None:
            if path is None:
                self._set_status(label, "No calibration file chosen.", "off")
            elif not path.is_file():
                self._set_status(
                    label,
                    f"{path.name} does not exist yet — "
                    f"{'step 1 will create it' if step == 1 else 'nothing to read'}"
                    f" ({path.parent})",
                    "off" if step == 1 else "error")
            else:
                try:
                    chain = self._run_chain(path)
                except Exception as exc:                    # noqa: BLE001
                    self._set_status(label, f"{path.name}: {exc}", "error")
                else:
                    self._set_status(
                        label,
                        f"{self._calib_name(path)} · {chain}", "ok")
        self._calib_gate_step(step)

    def _calib_gate_step(self, step: int | str) -> None:
        """Enable this page's buttons only when its file has what it needs.

        Said before the click, not after: a step 7 driven against a file with no
        step 6 used to spend the sweep and then refuse to fit it.
        """
        if step == "3c":
            self._refresh_step3_run_button()
            return
        missing = self._calib_missing(step)
        why = ("" if not missing else
               "that file has no " + ", ".join(f"step{n}" for n in missing))
        for name in self._STEP_BUTTONS.get(step, ()):
            button = getattr(self, name, None)
            if button is None:
                continue
            # Remembered the first time, because gating overwrites the tooltip
            # and the button's own explanation has to come back afterwards.
            if button.property("_ready_tip") is None:
                button.setProperty("_ready_tip", button.toolTip())
            button.setEnabled(not missing)
            button.setToolTip(why or button.property("_ready_tip") or "")

    def _calib_describe_all(self) -> None:
        for step in getattr(self, "_calib_edits", {}):
            self._calib_file_changed(step)

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
        self.stage3_reopt_sampling_edit = QtWidgets.QLineEdit("1001")
        self.stage3_reopt_sampling_edit.setToolTip(
            "Sampling points: AUTO or a count like 1001"
        )
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
        cfg_grid.addWidget(QtWidgets.QLabel("Points"), 2, 4)
        cfg_grid.addWidget(self.stage3_reopt_sampling_edit, 2, 5)
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
            700.0, 900.0, 778.04, " nm", 3
        )
        self.fast_channel_width_spin = self._spin(1, 256, 30)
        self.fast_channel_gap_spin = self._spin(0, 256, 15)
        self.fast_channel_count_spin = self._spin(1, 200, 20)
        self.fast_channel_skip_spin = self._spin(0, 20, 3)
        self.fast_channel_fine_check = QtWidgets.QCheckBox("OSA fine tune center")
        self.fast_channel_fine_check.setChecked(True)
        self.fast_channel_peak_nm_spin = self._double_spin(0.001, 50.0, 0.1, " nm", 3)
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
        self.fast_channel_guard_wl_edit = QtWidgets.QLineEdit("780.14, 775.94")
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
        self.fast_channel_center_edit = QtWidgets.QLineEdit("778.04nm")
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
        self.fast_channel_level_start_spin = self._spin(0, 1023, 350)
        self.fast_channel_level_stop_spin = self._spin(0, 1023, 950)
        self.fast_channel_level_step_spin = self._spin(1, 1023, 15)
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
        self.fast_channel_json_edit = QtWidgets.QLineEdit()
        self.fast_channel_json_edit.setPlaceholderText(
            "blank = src/calib_data/calib_step3b_MMDD_HHMM.json"
        )
        self.fast_channel_json_button = QtWidgets.QPushButton("Browse")
        self.fast_channel_json_button.clicked.connect(
            lambda: self._browse_save_into(
                self.fast_channel_json_edit,
                str(self._default_calib_dir() / self._default_calib_name("3b")),
                "JSON Files (*.json)",
            )
        )
        self.fast_channel_csv_edit = QtWidgets.QLineEdit()
        self.fast_channel_csv_edit.setPlaceholderText(
            "blank = src/calib_data/calib_step3b_MMDD_HHMM.csv"
        )
        self.fast_channel_csv_button = QtWidgets.QPushButton("Browse")
        self.fast_channel_csv_button.clicked.connect(
            lambda: self._browse_save_into(
                self.fast_channel_csv_edit,
                str(self._default_calib_dir() / self._default_calib_name("3b", ".csv")),
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

    def _build_step3_fit_backing(self) -> QtWidgets.QWidget:
        """Off-screen container for the legacy Fit-from-CSV controls + plots.

        These widgets are hidden on the Step-3 page but still populated by the
        calibration run/Run-All flow, so they are created here to keep those code
        paths (and their attribute references) working.
        """
        backing = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(backing)
        outer.setContentsMargins(0, 0, 0, 0)

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
        outer.addWidget(controls)

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
        outer.addWidget(split, 1)
        return backing

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
        widgets["sampling_points"] = QtWidgets.QLineEdit(
            defaults.get("sampling_points", "AUTO")
        )
        widgets["sampling_points"].setToolTip(
            "Sampling points: AUTO or a count like 1001"
        )
        grid.addWidget(QtWidgets.QLabel("Center λ"), 0, 0)
        grid.addWidget(widgets["center_wl"], 0, 1)
        grid.addWidget(QtWidgets.QLabel("Span"), 0, 2)
        grid.addWidget(widgets["span"], 0, 3)
        grid.addWidget(QtWidgets.QLabel("Sensitivity"), 1, 0)
        grid.addWidget(widgets["sensitivity"], 1, 1)
        grid.addWidget(QtWidgets.QLabel("Ref level"), 1, 2)
        grid.addWidget(widgets["ref_level"], 1, 3)
        grid.addWidget(QtWidgets.QLabel("Points"), 2, 0)
        grid.addWidget(widgets["sampling_points"], 2, 1)
        return box

    def _level_sweep_row(self, step: int | str, *, start: int = 0,
                         stop: int = 1023, stepv: int = 64) -> QtWidgets.QWidget:
        """A 'Levels start / stop / step' row stored on self.step_widgets[step].

        All three ends are per-step: step 1 wants the full 0-1023 to find the
        min and max at all, while 3c sweeps the usable middle of that range and
        starts well above 0.
        """
        row = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        widgets = self.step_widgets[step]
        widgets["level_start"] = self._spin(0, 1023, start)
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

    def _default_calib_name(self, step: int | str, suffix: str = ".json") -> str:
        """Default calibration output filename, e.g. calib_step1_0704_1530.json.

        The MMDD_HHMM timestamp keeps successive runs from overwriting each
        other. The `calib_step` prefix is preserved so the encoder's
        auto-discovery (_CALIB_RE) still matches the file.
        """
        return f"calib_step{step}_{time.strftime('%m%d_%H%M')}{suffix}"

    def _default_calib_dir(self) -> Path:
        """src/calib_data — where blank output fields put their results.

        ``parents[1]`` is ``src``: this file is ``src/gui/app.py``.  It was
        ``parents[2]`` from the package reorganisation until 2026-09-30, i.e. the
        repo root, so the default pointed at a directory that does not exist and
        auto-discovery silently found nothing.
        """
        return Path(__file__).resolve().parents[1] / "calib_data"

    def _min_max_row(self, step: int | str, label: str) -> QtWidgets.QWidget:
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

    def _region_row(self, step: int | str) -> QtWidgets.QWidget:
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
        start = self._spin(0, 8191, 600)
        end = self._spin(0, 8191, 1700)
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

    def _run_row(self, step: int | str, run_text: str, slot: Callable[[], None]) -> QtWidgets.QWidget:
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
        # Step 1 names a file that need not exist: this is where a calibration
        # starts.  Nothing is written until the sweep finishes.
        layout.addWidget(self._calib_file_row(1, create=True))
        layout.addWidget(self._build_measurement_group(1, {}))
        layout.addWidget(self._level_sweep_row(1, stop=1023, stepv=64))
        layout.addWidget(self._run_row(1, "Run Step 1", self._run_step1))
        layout.addStretch(1)
        return page

    def _build_step2_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.addWidget(self._calib_file_row(2))
        layout.addWidget(self._build_measurement_group(2, {}))

        cfg = QtWidgets.QHBoxLayout()
        widgets = self.step_widgets[2]
        widgets["window"] = self._spin(1, 8191, 20)
        widgets["peak_nm"] = self._double_spin(0.0, 50.0, 0.2, " nm", 3)
        widgets["peak_nm"].setToolTip("Centroid half-window around the peak, in nm")
        widgets["stride"] = self._spin(1, 8191, 30)
        widgets["stride"].setToolTip(
            "Measure every Nth column; the near-linear wavelength fit fills "
            "in the skipped columns (1 = measure every column)."
        )
        widgets["sweep_nm"] = self._double_spin(0.0, 50.0, 1.0, " nm", 2)
        widgets["sweep_nm"].setToolTip(
            "0 = off: wide OSA span at every position. >0: measure the two "
            "region-edge positions with the wide span first (anchors), then "
            "re-center this narrow span on the predicted wavelength at every "
            "other position — much faster."
        )
        # The usable BAND, applied to the finished mapping -- not a peak-search
        # mask.  The sweep measures and fits every peak it finds, however far
        # outside the band it lands; only the saved/handed-on mapping is clipped.
        usable_tip = (
            "Usable wavelength band for the steps downstream (0 = off).\n"
            "The sweep itself is NOT restricted: every window peak is measured "
            "and goes into the one linear coordinate -> wavelength fit, so a "
            "region whose edges fall outside the band is fine and in fact pins "
            "the slope down better.\nThe band is applied when the mapping is "
            "handed on: the saved JSON and the in-memory result keep only the "
            "coordinates that map into it, with the full-region fit "
            "coefficients carried over."
        )
        widgets["min_wl"] = self._double_spin(0.0, 2000.0, 775.0, " nm", 2)
        widgets["min_wl"].setToolTip(usable_tip)
        widgets["max_wl"] = self._double_spin(0.0, 2000.0, 781.0, " nm", 2)
        widgets["max_wl"].setToolTip(usable_tip)
        cfg.addWidget(QtWidgets.QLabel("Window px"))
        cfg.addWidget(widgets["window"])
        cfg.addWidget(QtWidgets.QLabel("Peak ± window"))
        cfg.addWidget(widgets["peak_nm"])
        cfg.addWidget(QtWidgets.QLabel("Stride"))
        cfg.addWidget(widgets["stride"])
        cfg.addWidget(QtWidgets.QLabel("Sweep span"))
        cfg.addWidget(widgets["sweep_nm"])
        cfg.addWidget(QtWidgets.QLabel("Usable λ ≥"))
        cfg.addWidget(widgets["min_wl"])
        cfg.addWidget(QtWidgets.QLabel("Usable λ ≤"))
        cfg.addWidget(widgets["max_wl"])
        cfg.addStretch(1)
        layout.addLayout(cfg)
        layout.addWidget(self._region_row(2))

        # Where the off/on levels come from.  Two choices, not three: step 1
        # writes into the very file named above, so reading them back out of it
        # is both simpler and what the CLI does -- and typing them is a
        # first-class path, not a fallback, because a calibration seeded by hand
        # is complete and never has a step 1 (see calib_step2.py --min-max).
        src_row = QtWidgets.QHBoxLayout()
        widgets["source"] = QtWidgets.QComboBox()
        widgets["source"].addItems(
            ["The file's step 1", "Manual min/max"]
        )
        widgets["source"].currentIndexChanged.connect(self._toggle_step2_source)
        src_row.addWidget(QtWidgets.QLabel("Min/max source"))
        src_row.addWidget(widgets["source"])
        src_row.addStretch(1)
        layout.addLayout(src_row)

        widgets["manual_row"] = self._min_max_row(2, "Manual levels")
        layout.addWidget(widgets["manual_row"])

        layout.addWidget(self._run_row(2, "Run Step 2", self._run_step2))
        layout.addStretch(1)
        self._toggle_step2_source()
        return page

    def _build_step3_daq_group(self, step: int | str = 3) -> QtWidgets.QGroupBox:
        """DAQ acquisition settings for a bucket-detector sweep (Step 3 / 3c).

        Defaults mirror tests/slm_sin2_level_sweep_test.py (ai1, 100 kS/s, 1 s
        average, 150 ms settle, ±0.1 V). Hidden unless Detector = DAQ.
        """
        box = QtWidgets.QGroupBox("DAQ acquisition (NI-DAQmx)")
        grid = QtWidgets.QGridLayout(box)
        widgets = self.step_widgets[step]
        widgets["daq_channel"] = QtWidgets.QLineEdit("ai1")
        widgets["daq_channel"].setMaximumWidth(90)
        widgets["daq_sample_rate"] = QtWidgets.QDoubleSpinBox()
        widgets["daq_sample_rate"].setRange(1.0, 2_000_000.0)
        widgets["daq_sample_rate"].setDecimals(0)
        widgets["daq_sample_rate"].setValue(1_000.0)
        widgets["daq_sample_rate"].setSuffix(" S/s")
        widgets["daq_hold"] = QtWidgets.QDoubleSpinBox()
        widgets["daq_hold"].setRange(0.0, 5_000.0)
        widgets["daq_hold"].setValue(250.0)
        widgets["daq_hold"].setSuffix(" ms")
        widgets["daq_hold"].setToolTip("Settle after each SLM frame before the DAQ reads.")
        widgets["daq_duration"] = QtWidgets.QDoubleSpinBox()
        widgets["daq_duration"].setRange(0.001, 10.0)
        widgets["daq_duration"].setDecimals(3)
        widgets["daq_duration"].setValue(1.0)
        widgets["daq_duration"].setSuffix(" s")
        widgets["daq_duration"].setToolTip("Averaging window per reading.")
        widgets["daq_range"] = QtWidgets.QComboBox()
        for lo, hi in self._DAQ_RANGES:
            widgets["daq_range"].addItem(f"\N{PLUS-MINUS SIGN}{hi:g} V", (lo, hi))
        widgets["daq_range"].setCurrentIndex(0)
        pairs = [
            ("Channel", widgets["daq_channel"]),
            ("Sample rate", widgets["daq_sample_rate"]),
            ("Hold", widgets["daq_hold"]),
            ("Average for", widgets["daq_duration"]),
            ("Range", widgets["daq_range"]),
        ]
        for col, (label, widget) in enumerate(pairs):
            grid.addWidget(QtWidgets.QLabel(label), 0, 2 * col)
            grid.addWidget(widget, 0, 2 * col + 1)
        return box

    def _build_step3c_page(self) -> QtWidgets.QWidget:
        """Step 3c: the Step-3 DAQ intensity sweep at encoding-channel centres.

        Mirrors Step 3 (one lit window at a time, dark-frame-subtracted DAQ
        readings) but structures the scan like Step 3b: the channel grid is
        tiled around a target centre wavelength with mirror-symmetric pairs,
        skipping any channel that overlaps a guard band.
        """
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(18, 14, 18, 14)
        widgets = self.step_widgets["3c"]

        layout.addWidget(self._calib_file_row("3c"))
        widgets["daq_group"] = self._build_step3_daq_group("3c")
        layout.addWidget(widgets["daq_group"])

        grid_panel = self._panel("Channel grid")
        grid = QtWidgets.QGridLayout(grid_panel)
        widgets["target"] = self._double_spin(700.0, 900.0, 778.04, " nm", 3)
        widgets["window"] = self._spin(1, 256, 15)
        widgets["pad"] = self._spin(0, 256, 35)
        widgets["count"] = self._spin(1, 200, 10)
        widgets["guard_check"] = QtWidgets.QCheckBox("Min-level guard bands")
        widgets["guard_check"].setChecked(True)
        widgets["guard_wl"] = QtWidgets.QLineEdit("780.14, 775.94")
        widgets["guard_wl"].setToolTip(
            "Comma/space separated guard center wavelengths in nm."
        )
        widgets["guard_nm"] = self._double_spin(0.001, 5.0, 0.06, " nm", 3)
        widgets["guard_nm"].setToolTip(
            "Half-width around each guard wavelength; channels overlapping a "
            "guard band are skipped."
        )
        grid.addWidget(QtWidgets.QLabel("Target center"), 0, 0)
        grid.addWidget(widgets["target"], 0, 1)
        grid.addWidget(QtWidgets.QLabel("Channel width"), 0, 2)
        grid.addWidget(widgets["window"], 0, 3)
        grid.addWidget(QtWidgets.QLabel("Gap px"), 0, 4)
        grid.addWidget(widgets["pad"], 0, 5)
        grid.addWidget(QtWidgets.QLabel("Channels/side"), 1, 0)
        grid.addWidget(widgets["count"], 1, 1)
        grid.addWidget(widgets["guard_check"], 2, 0, 1, 2)
        grid.addWidget(QtWidgets.QLabel("Guard centers"), 2, 2)
        grid.addWidget(widgets["guard_wl"], 2, 3)
        grid.addWidget(QtWidgets.QLabel("Guard half-width"), 2, 4)
        grid.addWidget(widgets["guard_nm"], 2, 5)
        layout.addWidget(grid_panel)

        layout.addWidget(
            self._level_sweep_row("3c", start=350, stop=950, stepv=25))

        # No wavelength-source choice: the map is the named file's step-2
        # section, exactly as calib_step3.py reads it.  The off/on levels ride
        # in that section too, so there is nothing left to pick.
        run_row = self._run_row("3c", "Run Step 3c", self._run_step3c)
        widgets["stop"] = QtWidgets.QPushButton("Stop")
        widgets["stop"].setProperty("variant", "danger")
        widgets["stop"].setEnabled(False)
        widgets["stop"].clicked.connect(self._stop_full_calibration)
        run_row.layout().addWidget(widgets["stop"])
        layout.addWidget(run_row)
        layout.addStretch(1)
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
            "Divide the panel into bands along x (vertical) or y (horizontal) "
            "and assign a phase level to each."
        )
        subtitle.setObjectName("PageSubtitle")
        page.layout().addWidget(subtitle)

        controls = self._panel("Segments")
        controls_layout = QtWidgets.QGridLayout(controls)
        self.segment_axis_combo = QtWidgets.QComboBox()
        self.segment_axis_combo.addItems(["Vertical (along x)", "Horizontal (along y)"])
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

        controls_layout.addWidget(QtWidgets.QLabel("Axis"), 0, 0)
        controls_layout.addWidget(self.segment_axis_combo, 0, 1)
        controls_layout.addWidget(QtWidgets.QLabel("Mode"), 0, 2)
        controls_layout.addWidget(self.segment_mode_combo, 0, 3)
        controls_layout.addWidget(QtWidgets.QLabel("Parts"), 0, 4)
        controls_layout.addWidget(self.segment_count_spin, 0, 5)
        controls_layout.addWidget(self.segment_fill_spin, 0, 6)
        controls_layout.addWidget(fill_button, 0, 7)
        controls_layout.addWidget(add_row_button, 0, 8)
        controls_layout.addWidget(remove_row_button, 0, 9)

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

        self.segment_axis_combo.currentIndexChanged.connect(self._on_segment_axis_changed)
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

        self.enc_calib_label = QtWidgets.QLabel("Calibration: (none loaded)")
        self.enc_calib_label.setObjectName("PageSubtitle")
        enc_reload = QtWidgets.QPushButton("Load other…")
        enc_reload.setProperty("variant", "ghost")
        enc_reload.setToolTip("Override the local calibration with another result file")
        enc_reload.clicked.connect(self._enc_browse_calib)

        self.enc_build_button = QtWidgets.QPushButton("Reload Layout")
        self.enc_build_button.setToolTip(
            "Re-read the channel structure from the calibration file; centres, "
            "pitch and guard skips come verbatim from the Step 3b/3c grid"
        )
        self.enc_build_button.clicked.connect(self._enc_build_layout)

        self.enc_layout_status = QtWidgets.QLabel(
            "The channel structure is loaded from the Step 3b/3c calibration."
        )
        self.enc_layout_status.setWordWrap(True)

        cfg_grid.addWidget(self.enc_calib_label,   0, 0)
        cfg_grid.addWidget(self.enc_layout_status, 0, 1)
        cfg_grid.addWidget(enc_reload,             0, 2)
        cfg_grid.addWidget(self.enc_build_button,  0, 3)
        cfg_grid.setColumnStretch(1, 1)

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

        # Encoding info only: the page used to carry a scope/DAQ readout, now
        # deprecated -- acquisitions run on the Step 6 / Step 7 tabs.
        main = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(main)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(left_split, 1)
        main_layout.addLayout(ctrl_row)

        # --- Feedback log, spanning full width ---
        self.enc_log = QtWidgets.QPlainTextEdit()
        self.enc_log.setReadOnly(True)
        self.enc_log.setObjectName("LogBox")
        self.enc_log.setMaximumHeight(120)
        log_panel = self._panel_with_widget("Feedback", self.enc_log)

        page.layout().addWidget(cfg_panel)
        page.layout().addWidget(main, 1)
        page.layout().addWidget(log_panel)

        # auto-load the local calibration and build a default layout so the
        # value table is populated and ready for manual input immediately
        QtCore.QTimer.singleShot(0, self._enc_autostart)
        return page

    def _enc_autostart(self) -> None:
        """Load local calibration and build the default layout on first show."""
        calib = self._enc_get_calib()
        if calib is None or not calib.encodable:
            self._enc_log(
                "No calibration found. Run Step 3 on the Calibration page, or use "
                "'Load other…' to pick a calib.json."
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

    # A run directory's calib.json, or a pre-v3 standalone calib_step*.json.
    # Both are things the encoder can take a step-3 channel grid out of, and
    # leaving calib.json out of this pattern is why auto-discovery stopped
    # finding anything once the runs were migrated.
    _CALIB_RE = re.compile(r"^(calib\.json|calib_step.*\.json)$", re.IGNORECASE)

    def _enc_local_calib_path(self) -> Path | None:
        """Locate the newest usable project-local calibration result.

        Scans the working dir, the project root, src/calib_data and each run
        directory under it, and returns the most recently modified calibration
        that can actually drive the encoder.  One whose step-3 section is missing
        -- a run that has only reached step 1 or 2 -- is skipped rather than
        auto-loaded, which is the whole reason this walks the files itself.
        """
        search_dirs = [
            Path.cwd(),
            Path(__file__).resolve().parents[3],
            self._default_calib_dir(),
        ]
        # Each run directory holds its own calib.json, so the run folders are
        # where the calibrations actually live now.
        try:
            search_dirs += [d for d in self._default_calib_dir().iterdir()
                            if d.is_dir() and d.name != "archive"]
        except OSError:
            pass
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
            if calib.encodable:
                return path
        return None

    def _enc_get_calib(self) -> CalibrationResult | None:
        """Calibration source: explicit override → in-memory result → local file."""
        if self._enc_calib_override is not None:
            return self._enc_calib_override
        if self.calibration_result is not None and self.calibration_result.encodable:
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

    def _enc_browse_calib(self) -> None:
        # A run's calib.json by name; load_calibration_result also takes the
        # directory, and the pre-v3 standalone results still open.
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open a run's calib.json", "", "JSON Files (*.json)"
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
        """Load the channel structure verbatim from the Step 3b/3c calibration.

        The scanned grid already encodes every layout decision (target centre,
        pitch, guard skips), so the encoder takes the calibration rows as the
        channels directly instead of re-tiling its own geometry. Only the
        window/gap split is not recorded in the file; the width is assumed
        ``pitch - 5`` (the Step-3c default gap).
        """
        calib = self._enc_get_calib()
        if calib is None or not calib.encodable:
            self.enc_layout_status.setText(
                "No calibration available. Run Step 3 or load a run's calib.json."
            )
            return

        try:
            layout = channel_layout_from_calibration(calib)
        except Exception as exc:
            self.enc_layout_status.setText(f"Layout error: {exc}")
            self.enc_generate_button.setEnabled(False)
            return

        self.encoding_layout = layout
        self._edge_optimization_result = None
        self.enc_use_optimized_lut.setChecked(False)
        self.enc_use_optimized_lut.setEnabled(False)
        self._enc_populate_val_table(layout)
        self._edge_sync_layout(layout)
        gap = layout.pitch_px - layout.channel_width_px
        self.enc_layout_status.setText(
            f"{layout.n_channels} ch/side  |  centre {layout.center_wl:.4f} nm "
            f"(px {layout.center_x:.1f})  |  pitch {layout.pitch_px} px  |  "
            f"width {layout.channel_width_px} px (assumed gap {gap} px)"
        )
        self.enc_generate_button.setEnabled(True)
        self._enc_log(
            f"Layout loaded from calibration: {layout.n_channels} channels/side "
            f"around {layout.center_wl:.4f} nm, pitch {layout.pitch_px} px, "
            f"width {layout.channel_width_px} px. Edit values in the table, "
            "then Generate & Preview."
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
        self.shape_enabled_check.setChecked(False)
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
        self.edge_table.setEnabled(self.shape_enabled_check.isChecked())
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
            "built. Sweeps use the default Modulation Error settings "
            "(0.8 nm span per channel, HIGH3, background-subtracted)."
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

        # --- OSA optimisation settings: span / sampling points / batch mode ---
        self.edge_osa_span_edit = QtWidgets.QLineEdit("0.8nm")
        self.edge_osa_span_edit.setMaximumWidth(70)
        self.edge_osa_points_edit = QtWidgets.QLineEdit("1001")
        self.edge_osa_points_edit.setMaximumWidth(70)
        self.edge_osa_points_edit.setToolTip(
            "Sampling points: AUTO or a count like 1001"
        )
        self.edge_batch_check = QtWidgets.QCheckBox("Batch over sampling points")
        self.edge_batch_check.setToolTip(
            "Run the full optimisation once per sampling-point count and "
            "compare the outcomes"
        )
        self.edge_batch_points_edit = QtWidgets.QLineEdit("501, 1001, 2001")
        self.edge_batch_points_edit.setEnabled(False)
        self.edge_batch_check.toggled.connect(
            self.edge_batch_points_edit.setEnabled
        )
        osa_cfg_row = QtWidgets.QHBoxLayout()
        osa_cfg_row.addWidget(QtWidgets.QLabel("Opt. span"))
        osa_cfg_row.addWidget(self.edge_osa_span_edit)
        osa_cfg_row.addWidget(QtWidgets.QLabel("points"))
        osa_cfg_row.addWidget(self.edge_osa_points_edit)
        osa_cfg_row.addSpacing(16)
        osa_cfg_row.addWidget(self.edge_batch_check)
        osa_cfg_row.addWidget(self.edge_batch_points_edit, 1)

        self.edge_live_canvas = LiveLossCanvas(self)
        self.edge_batch_table = BatchResultsTable(self)
        self.edge_batch_table.hide()

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
        gain_layout.addLayout(osa_cfg_row)
        gain_layout.addWidget(self.edge_gain_bar)
        gain_layout.addWidget(self.edge_gain_status)
        gain_layout.addWidget(self.edge_live_canvas)
        gain_layout.addWidget(self.edge_batch_table)
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
            span=self.edge_osa_span_edit.text().strip() or "0.8nm",
            sensitivity=ana.sensitivity,
            sampling_points=self.edge_osa_points_edit.text().strip() or "1001",
            y_unit=ana.y_unit,
            reference_level=ana.reference_level,
            trace_id=ana.trace_id,
            trace_mode=ana.trace_mode,
        )
        config = OSAOptimizationConfig(settings=settings)
        initial_l = ratio[:8].copy()

        from dataclasses import replace

        batch_variants: list[OSABatchVariant] = []
        if self.edge_batch_check.isChecked():
            counts = [
                part.strip()
                for part in self.edge_batch_points_edit.text().replace(";", ",").split(",")
                if part.strip()
            ]
            if not counts:
                self._edge_log("Batch mode needs at least one sampling-point count.")
                return
            batch_variants = [
                OSABatchVariant(
                    f"sp{count}", replace(settings, sampling_points=count)
                )
                for count in counts
            ]

        stop_event = threading.Event()
        self.edge_gain_stop_event = stop_event
        self._edge_gain_running(True)
        self.edge_gain_bar.setRange(0, 0)
        self.edge_gain_status.setText("Preparing fixed OSA bins and references…")
        self.edge_live_canvas.reset()
        self.edge_batch_table.setVisible(bool(batch_variants))
        self.edge_batch_table.setRowCount(0)
        self._edge_log(
            "OSA optimisation started with 8 intensity ratios: "
            + np.array2string(initial_l, precision=4)
            + (f"  ·  batch: {[v.label for v in batch_variants]}"
               if batch_variants else "")
        )

        def report(progress: OptimizationProgress) -> None:
            self.edge_optimization_progress.emit(progress)

        if batch_variants:
            def work() -> dict[str, Any]:
                outcomes = run_osa_optimization_batch(
                    osa, controller, layout, initial_l,
                    base_config=config, variants=batch_variants,
                    stop_event=stop_event, progress_callback=report,
                )
                return {"status": "ok", "outcomes": outcomes}

            self._run_slm_task(
                "OSA batch optimisation",
                work,
                self._edge_batch_finished,
                self._edge_optimization_error,
            )
            return

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

    def _edge_batch_finished(self, payload: dict[str, Any]) -> None:
        """Show the per-variant comparison; profiles are NOT auto-applied."""
        self.edge_gain_stop_event = None
        self._edge_gain_running(False)
        self.edge_gain_bar.setRange(0, 100)
        self.edge_gain_bar.setValue(0)
        outcomes = payload.get("outcomes", [])
        self.edge_batch_table.show()
        self.edge_batch_table.show_outcomes(outcomes)
        finished = sum(1 for o in outcomes if o.result is not None)
        stopped = any(o.stopped for o in outcomes)
        state = "stopped early" if stopped else "done"
        self.edge_gain_status.setText(
            f"Batch {state}: {finished}/{len(outcomes)} variants completed. "
            "Use 'Load optimized result…' on a run dir to apply a profile."
        )
        for outcome in outcomes:
            self._edge_log(
                f"batch {outcome.variant.label}: "
                + (outcome.error or ("stopped" if outcome.stopped
                   else f"accepted={outcome.result.accepted} -> {outcome.run_dir}"))
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
        # stream per-evaluation metrics + the flat reference into the live plot
        self.edge_live_canvas.on_progress(progress)

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
            "Encoding page first. OSA sweeps use the default Modulation Error "
            "settings (0.8 nm span, HIGH3, LOG)."
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

        # layout geometry, used only for the coarse Step 1+2 tiling (a Step 3
        # file already carries the measured grid and is loaded verbatim)
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
        self.qt_calib_params.setVisible(mode == 2)
        self.qt_calib_build_btn.setVisible(mode in (1, 2))
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
        """Tile a coarse layout from a synthetic Step 1+2 calibration.

        Only for the quick-test mode with no Step 3 data: a Step-2 map has no
        measured channel grid to load, so the geometry is tiled here. Any real
        Step-3b/3c result is loaded verbatim via
        :func:`channel_layout_from_calibration` instead.
        """
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
                # the measured grid IS the layout: load it verbatim, no re-tiling
                layout = channel_layout_from_calibration(calib)
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
                layout = self._layout_from_calib(
                    calib,
                    center_wl=self.qt_calib_center.value(),
                    channel_width_px=self.qt_calib_width.value(),
                    gap_px=self.qt_calib_pad.value(),
                )
            else:
                return
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
        self.osv_dock_button = QtWidgets.QPushButton("Monitor dock")
        self.osv_dock_button.setProperty("variant", "ghost")
        self.osv_dock_button.setCheckable(True)
        self.osv_dock_button.setToolTip(
            "Show the dockable live monitor that follows every OSA sweep, "
            "from any page"
        )
        self.osv_dock_button.toggled.connect(self.osa_monitor_dock.setVisible)
        self.osa_monitor_dock.visibilityChanged.connect(
            self.osv_dock_button.setChecked
        )
        ctrl.addWidget(self.osv_single_button)
        ctrl.addWidget(self.osv_cont_button)
        ctrl.addWidget(self.osv_stop_button)
        ctrl.addWidget(self.osv_save_button)
        ctrl.addWidget(self.osv_logy_check)
        ctrl.addWidget(self.osv_dock_button)
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
        busy = self._calibration_busy()
        self.osv_single_button.setEnabled(not running and not busy)
        self.osv_cont_button.setChecked(running and continuous)
        self.osv_cont_button.setEnabled((not running and not busy) or (running and continuous))
        self.osv_stop_button.setEnabled(running)
        self.osv_save_button.setEnabled(not running and self.osa_view_trace is not None)

    def _osa_view_single(self) -> None:
        if self._calibration_busy():
            self.osv_cont_button.setChecked(False)
            self.osv_status.setText("A calibration is running; OSA reads are locked.")
            return
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
        if self._calibration_busy():
            self.osv_cont_button.setChecked(False)
            self.osv_status.setText("A calibration is running; OSA reads are locked.")
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
            # display happens via the controller's trace listener; this loop
            # only drives the sweeps
            try:
                while not stop_event.is_set():
                    osa.measure(settings, averages=averages, stop_event=stop_event)
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
    # ===================== TPA efficiency (step 6) tab ==================
    #: Result-table columns.  Every one is a scalar per pair, which is what
    #: makes ten pairs readable at once: the six diagnostic panels are all
    #: per-pair and only ever show the selected row.
    _TPA_COLUMNS = ("pair", "η", "±η", "a_x (mV)", "a_w (mV)",
                    "d (mV)", "R²", "max|pull|", "checks")

    def _build_tpa_tab(self) -> QtWidgets.QWidget:
        """Step 6 v2: controls on the left, results on the right.

        A single horizontal split rather than the old vertical stack.  The page
        has to carry a nine-control DAQ group, a sweep group and a per-pair
        results table, and stacking those above the plots left the plots with
        nothing.  Everything you touch is now in a narrow left column in the
        order you touch it, ending at Run; everything you read has the rest of
        the width and the full height.
        """
        page = self._page_shell("Channel TPA Efficiency (η) Calibration")
        self.tpa_status = self._page_status(page, "\N{EN DASH}")
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.addWidget(self._build_tpa_controls())
        split.addWidget(self._build_tpa_results())
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([360, 1040])
        page.layout().addWidget(split, 1)

        self._tpa_redraw()
        return page

    def _build_tpa_controls(self) -> QtWidgets.QWidget:
        """The left column: the calib file, DAQ, sweep, buttons -- in that order.

        The file first, because it is the one thing that has to be right before
        anything else on the page means anything: every level below is encoded
        through its step-3 section, so a sweep against the wrong one does not
        fail, it calibrates a different aperture and reports numbers that look
        fine.  Naming it here is also what makes a GUI run and a terminal run
        the same run -- ``--calib`` on calib_step6_v2.py is this field.

        It does not scroll, and that is a constraint on what may go in it: the
        acquisition settings that are not bench constants, the sweep, and the
        buttons.  A column taller than the window either loses the height out of
        its panels or grows a scrollbar, and a setting you have to scroll to
        find is a setting you will not check.
        """
        col = QtWidgets.QWidget()
        box = QtWidgets.QVBoxLayout(col)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(12)

        box.addWidget(self._calib_file_row(6))
        self.tpa_step3_edit = self._calib_edits[6]
        self.tpa_step3_label = self._calib_labels[6]

        # --- DAQ acquisition --------------------------------------------
        daq = self._panel("DAQ · acquisition")
        grid = QtWidgets.QGridLayout(daq)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        self.tpa_daq_channel = QtWidgets.QLineEdit("ai0")
        self.tpa_daq_rate = QtWidgets.QDoubleSpinBox()
        self.tpa_daq_rate.setRange(1.0, 2_000_000.0)
        self.tpa_daq_rate.setDecimals(0)
        self.tpa_daq_rate.setValue(1000.0)
        self.tpa_daq_rate.setSuffix(" S/s")
        self.tpa_daq_range = QtWidgets.QComboBox()
        for lo, hi in self._DAQ_RANGES:
            self.tpa_daq_range.addItem(f"\N{PLUS-MINUS SIGN}{hi:g} V", (lo, hi))
        self._select_daq_range(self.tpa_daq_range, BENCH_ACQ.range_v)
        self.tpa_daq_range.setToolTip(BENCH_RANGE_TIP)
        self.tpa_daq_fcut = QtWidgets.QDoubleSpinBox()
        self.tpa_daq_fcut.setRange(0.1, 100_000.0)
        self.tpa_daq_fcut.setDecimals(1)
        self.tpa_daq_fcut.setValue(20.0)
        self.tpa_daq_fcut.setSuffix(" Hz")
        self.tpa_daq_fcut.setToolTip(
            "Detector 3 dB bandwidth: the low-pass behind the reported mean and "
            "std. That std is the sigma the fit weights by."
        )
        self.tpa_settle = self._double_spin(0.0, 10.0, 0.25, " s", 3)
        self.tpa_settle.setToolTip(
            "Wait after each SLM pattern change, before reading."
        )
        self.tpa_tmeas = self._double_spin(0.001, 60.0, BENCH_ACQ.t_meas_s, " s", 3)
        self.tpa_tmeas.setToolTip(BENCH_WINDOW_TIP)
        self.tpa_invert = QtWidgets.QCheckBox("Invert sign (TIA)")
        self.tpa_invert.setChecked(True)
        self.tpa_invert.setToolTip(
            "The transimpedance amplifier outputs NEGATIVE volts for positive "
            "light, so recording a positive light signal means inverting.\n"
            "Leave this on. The estimator fits b = η² and takes its square "
            "root, so a sign-flipped run gives b < 0 and η = NaN rather than a "
            "plausible wrong answer."
        )
        self.tpa_pmt = QtWidgets.QCheckBox("Log PMT temperature")
        self.tpa_pmt.setChecked(True)
        self.tpa_pmt.setToolTip(PMT_CHECKBOX_TIP)

        rows = [
            ("Channel", self.tpa_daq_channel),
            ("Range", self.tpa_daq_range),
            ("T_meas", self.tpa_tmeas),
        ]
        for r, (label, widget) in enumerate(rows):
            grid.addWidget(QtWidgets.QLabel(label), r, 0)
            grid.addWidget(widget, r, 1)
        grid.addWidget(self.tpa_invert, len(rows), 0, 1, 2)
        # Bench constants, and off the panel for it.  The sample rate, the
        # low-pass and the settle have been the same on every run this bench
        # has taken, and the PMT log is always on; what is left on the panel is
        # what actually gets set.  They are still live widgets that _tpa_acq
        # and the sweep read, so this hides a control, not a setting.
        for widget in (self.tpa_daq_rate, self.tpa_daq_fcut,
                       self.tpa_settle, self.tpa_pmt):
            widget.setParent(daq)
            widget.hide()
        box.addWidget(daq)

        # --- sweep -------------------------------------------------------
        sweep = self._panel("Sweep")
        sgrid = QtWidgets.QGridLayout(sweep)
        sgrid.setHorizontalSpacing(8)
        sgrid.setVerticalSpacing(6)

        self.tpa_pairs_edit = QtWidgets.QLineEdit("2-6")
        self.tpa_pairs_edit.setToolTip(
            "Pair labels to calibrate, 1-based: \"2-6\", \"1,3,5\" or a mix.\n"
            "Pair i is the i-th channel of the Step-3 calibration."
        )
        sgrid.addWidget(QtWidgets.QLabel("Pairs"), 0, 0)
        sgrid.addWidget(self.tpa_pairs_edit, 0, 1, 1, 2)

        # The cross line -- the levels eta is the slope of.  The single-beam
        # block is not on here: it is the five levels the background fit needs
        # and no ramp changes them.  The fit window follows this ramp, so the
        # estimator fits exactly what was measured.
        self.tpa_sweep_min = self._double_spin(0.01, 1.0, 0.20, "", 2)
        self.tpa_sweep_min.setSingleStep(0.05)
        self.tpa_sweep_min.setToolTip(
            "Lowest per-side level on the cross line (x = 1, w = this).\n"
            "Also the bottom of the fit window."
        )
        self.tpa_sweep_max = self._double_spin(0.02, 1.0, 0.90, "", 2)
        self.tpa_sweep_max.setSingleStep(0.05)
        self.tpa_sweep_max.setToolTip(
            "Highest level on the cross line, and the top of the fit window.\n"
            "Below 1.0, the (1, 1) level is still measured but kept OUT of the "
            "fit as the top-drive compression diagnostic. Take this to 1.0 and "
            "that diagnostic is gone — compression then sits inside the fit."
        )
        self.tpa_points = self._spin(2, 15, 4)
        self.tpa_points.setToolTip(
            "Points on the cross-line ramp, evenly spaced from min to max.\n"
            "The last one gets 6 repeats instead of 4: the top of the window "
            "carries the most leverage on the slope."
        )
        sgrid.addWidget(QtWidgets.QLabel("Sweep min"), 1, 0)
        sgrid.addWidget(self.tpa_sweep_min, 1, 1, 1, 2)
        sgrid.addWidget(QtWidgets.QLabel("Sweep max"), 2, 0)
        sgrid.addWidget(self.tpa_sweep_max, 2, 1, 1, 2)
        sgrid.addWidget(QtWidgets.QLabel("Ramp points"), 3, 0)
        sgrid.addWidget(self.tpa_points, 3, 1, 1, 2)

        # The rail ceiling, a page setting rather than outliers.BURST_MAX_V:
        # it is a property of this bench at this gain, it has to clear the
        # brightest honest read of the sweep, and nobody but the operator
        # knows what that will be.
        self.tpa_rail = self._double_spin(
            1.0, 1000.0, 1e3 * BURST_MAX_V, " mV", 0)
        self.tpa_rail.setSingleStep(50.0)
        self.tpa_rail.setToolTip(
            "Any read at or above this is a railed detector, not a "
            "measurement, and is dropped.\n"
            "A bench-fault ceiling, not a saturation limit: it only has to "
            "sit above the brightest read this sweep honestly takes.\n"
            "Applies to the next sweep AND to a Load's re-fit."
        )
        sgrid.addWidget(QtWidgets.QLabel("Rail"), 4, 0)
        sgrid.addWidget(self.tpa_rail, 4, 1, 1, 2)

        box.addWidget(sweep)

        # --- buttons -----------------------------------------------------
        # No progress bar here: a run takes tens of minutes, so it gets the
        # shared CalibrationProgressDialog (bar, elapsed/ETA, live plot,
        # Stop) that steps 1-3 pop, rather than a strip in a column the
        # user is not watching.
        #
        # No status line either: `tpa_status` is beside the page title.  The
        # read-rejection rules used to be spelled out down here as well -- the
        # rail is a control now and says what it does in its own tooltip, and
        # the burst guard and the sigma floor announce themselves in the log
        # when they fire, which is when anybody wants them.
        self.tpa_run_button = QtWidgets.QPushButton("Run Sweep")
        self.tpa_run_button.clicked.connect(self._tpa_run)
        self.tpa_stop_button = QtWidgets.QPushButton("Stop")
        self.tpa_stop_button.setProperty("variant", "danger")
        self.tpa_stop_button.setEnabled(False)
        self.tpa_stop_button.clicked.connect(self._tpa_stop)
        self.tpa_load_button = QtWidgets.QPushButton("Load\N{HORIZONTAL ELLIPSIS}")
        self.tpa_load_button.setProperty("variant", "ghost")
        self.tpa_load_button.setToolTip(
            "Re-fit the step-6 sweep this calibration file records and show it: "
            "every pair, from the CSV its step6 section names.\n"
            "Reads only — nothing is written back."
        )
        self.tpa_load_button.clicked.connect(self._tpa_load)
        btns = QtWidgets.QGridLayout()
        btns.addWidget(self.tpa_load_button, 0, 0)
        btns.addWidget(self.tpa_run_button, 1, 0)
        btns.addWidget(self.tpa_stop_button, 1, 1)
        box.addLayout(btns)

        box.addStretch(1)
        return col

    def _build_tpa_results(self) -> QtWidgets.QWidget:
        """The right column: the all-pairs table over the selected pair's panels."""
        self.tpa_table = QtWidgets.QTableWidget(0, len(self._TPA_COLUMNS))
        self.tpa_table.setHorizontalHeaderLabels(list(self._TPA_COLUMNS))
        self.tpa_table.verticalHeader().setVisible(False)
        self.tpa_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tpa_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tpa_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.tpa_table.setAlternatingRowColors(True)
        self.tpa_table.horizontalHeader().setStretchLastSection(True)
        # The table IS the pair selector -- no separate combo box.
        self.tpa_table.itemSelectionChanged.connect(self._tpa_redraw)

        # Estimator over its own pulls, sharing the w axis: both are "vs w", and
        # side by side they would each get half the width for no reason.
        self.tpa_est_fig = Figure(figsize=(5.4, 4.4), tight_layout=True)
        self.tpa_est_canvas = FigureCanvas(self.tpa_est_fig)
        self.tpa_check_fig = Figure(figsize=(4.2, 3.0), tight_layout=True)
        self.tpa_check_canvas = FigureCanvas(self.tpa_check_fig)

        # No verification text panel.  Both checks still run, and they are still
        # reported three ways -- the table's `checks` column flags a pull past
        # 3σ, the product check is the panel beside this one, and the saved JSON
        # and PNG carry the full records.  Repeating them as prose here spent a
        # quarter of the results pane restating what the table already says.
        panels = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        panels.addWidget(
            self._panel_with_widget("Difference estimator · η is this slope",
                                    self.tpa_est_canvas))
        panels.addWidget(
            self._panel_with_widget("Product check · same x·w, different split",
                                    self.tpa_check_canvas))
        panels.setSizes([600, 400])

        stack = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        stack.addWidget(self._panel_with_widget("Pairs", self.tpa_table))
        stack.addWidget(panels)
        stack.setStretchFactor(0, 0)
        stack.setStretchFactor(1, 1)
        stack.setSizes([220, 520])
        return stack

    # ---- inputs ------------------------------------------------------------
    @staticmethod
    def _tpa_parse_pairs(text: str) -> list[int]:
        """\"2-6\", \"1,3,5\" or a mix -> sorted unique 1-based pair labels."""
        out: set[int] = set()
        for part in text.replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part[1:]:                      # not a leading minus sign
                lo, hi = part.split("-", 1)
                lo_i, hi_i = int(lo), int(hi)
                if hi_i < lo_i:
                    raise ValueError(f"empty range {part!r}")
                out.update(range(lo_i, hi_i + 1))
            else:
                out.add(int(part))
        if not out:
            raise ValueError("no pairs given")
        return sorted(out)

    def _tpa_step3_path(self) -> Path | None:
        """This page's calib.json -- a typed directory means the file inside it."""
        return self._calib_file_path(6)

    def _tpa_load_step3(self, calib_file: "Path | None" = None):
        """``(calibration, layout)`` from a file's step-3 section, or raise.

        ``calib_file`` names the file; omitted, it is the one on the page.  A
        Load passes the file it was pointed at, so the wavelengths on screen
        belong to the sweep on screen.

        The layout is rebuilt here rather than taken from the Encoding page so
        that what this sweep drives is decided by one named calib file, exactly
        as the offline script's ``--calib`` decides it.  Same loader, so a GUI
        run and a terminal run against the same file drive the same pixels.
        """
        run = calib_file if calib_file is not None else self._tpa_step3_path()
        if run is None:
            raise ValueError("no calibration file chosen")
        calib = load_calib(run)
        result = calib.step3_result()
        return result, channel_layout_from_calibration(result)

    def _tpa_describe_step3(self) -> None:
        """Say what the chosen run holds, before an hour is spent on it."""
        run = self._tpa_step3_path()
        if run is None:
            self._set_status(self.tpa_step3_label,
                             "No calibration file chosen — required to run.",
                             "off")
            return
        try:
            _calib, layout = self._tpa_load_step3()
        except Exception as exc:
            self._set_status(self.tpa_step3_label,
                             f"{self._calib_name(run)}: {exc}", "error")
            return
        self._set_status(
            self.tpa_step3_label,
            f"{self._calib_name(run)} · {self._run_chain(run)} "
            f"· {layout.n_channels} pairs · "
            f"{layout.channel_width_px} px window + "
            f"{layout.pitch_px - layout.channel_width_px} px pad",
            "ok",
        )

    def _tpa_config(self) -> PairV2Config:
        """The estimator's setup, with the cross line built from the ramp controls.

        Falls back to the validated :data:`DEFAULT_GRID` if the three ramp
        values do not make a grid -- a half-typed spinbox must not stop the page
        redrawing, and Run validates them again before it drives anything.
        """
        try:
            return PairV2Config.from_ramp(
                self.tpa_sweep_min.value(),
                self.tpa_sweep_max.value(),
                self.tpa_points.value(),
            )
        except ValueError:
            return PairV2Config()

    @staticmethod
    def _select_daq_range(combo: QtWidgets.QComboBox, hi: float) -> None:
        """Select the range whose upper bound is ``hi``; leave it alone if absent."""
        for i in range(combo.count()):
            if float(combo.itemData(i)[1]) == float(hi):
                combo.setCurrentIndex(i)
                return

    def _tpa_acq(self) -> PairV2Acq:
        """Acquisition timing and input range, straight off the DAQ group."""
        _lo, hi = self.tpa_daq_range.currentData()
        return PairV2Acq(
            t_meas_s=float(self.tpa_tmeas.value()),
            settle_s=float(self.tpa_settle.value()),
            range_v=float(hi),
            invert=self.tpa_invert.isChecked(),
        )

    def _tpa_set_running(self, running: bool) -> None:
        """Gate this page while a sweep is in flight (and re-gate when it ends)."""
        self._tpa_set_running_impl(running)
        if not running:
            self._calib_gate_step(6)

    def _tpa_set_running_impl(self, running: bool) -> None:
        self._mark_tpa_calibration("step6", running)
        self.tpa_run_button.setEnabled(not running)
        self.tpa_stop_button.setEnabled(running)
        self.tpa_load_button.setEnabled(not running)
        self.tpa_pairs_edit.setEnabled(not running)

    @staticmethod
    def _tpa_wavelengths(layout, cfg: PairV2Config, index: int):
        """A pair's two channel wavelengths, or NaN when the layout lacks the slot."""
        nan = float("nan")
        slot = cfg.slot(index)
        if not (0 <= slot < min(len(layout.x_channels), len(layout.w_channels))):
            return nan, nan, nan
        x_ch = layout.x_channels[slot]
        w_ch = layout.w_channels[slot]
        return (float(x_ch.wavelength_nm), float(w_ch.wavelength_nm),
                0.5 * (float(x_ch.wavelength_nm) + float(w_ch.wavelength_nm)))

    # ---- run ---------------------------------------------------------------
    def _tpa_run(self) -> None:
        try:
            _calib, layout = self._tpa_load_step3()
        except Exception as exc:
            self.tpa_status.setText(f"Step 3: {exc}")
            return
        daq = self.daq_controller
        if daq is None or not daq.is_connected:
            self.tpa_status.setText("Connect the DAQ first (DAQ page).")
            return
        controller = self._controller()
        if not getattr(controller, "is_open", False):
            self.tpa_status.setText("Open the SLM on the SLM Control page first.")
            return
        cfg = self._tpa_config()
        try:
            pairs = self._tpa_parse_pairs(self.tpa_pairs_edit.text())
        except ValueError as exc:
            self.tpa_status.setText(f"Bad pair list: {exc}")
            return
        bad = [p for p in pairs if not (0 <= cfg.slot(p) < layout.n_channels)]
        if bad:
            self.tpa_status.setText(
                f"Pair(s) {bad} out of range — the layout has "
                f"{layout.n_channels} pairs, numbered from {cfg.pair_index_base}."
            )
            return

        acq = self._tpa_acq()
        schedule = build_schedule(cfg)
        total = len(schedule) * len(pairs)
        lo, hi = self.tpa_daq_range.currentData()
        settings = DAQMonitorSettings(
            channel=self.tpa_daq_channel.text().strip() or "ai0",
            sample_rate=self.tpa_daq_rate.value(),
            duration=acq.t_meas_s,
            single_duration=acq.t_meas_s,
            hold=0.0,                 # measure_pair owns the settle
            min_val=lo, max_val=hi,
            f_cut=self.tpa_daq_fcut.value(),
        )
        read_timeout = max(30.0, acq.t_meas_s * 3.0 + 10.0)

        self.tpa_status.setText(
            f"Running\N{HORIZONTAL ELLIPSIS} {len(pairs)} pair(s), "
            f"{len(schedule)} acquisitions each"
        )
        self._tpa_set_running(True)
        self._open_calibration_dialog(on_stop=self._tpa_stop)

        stop_event = threading.Event()
        self.tpa_stop_event = stop_event
        col_ratio = self._active_col_ratio()

        def report(progress: PairV2Progress) -> None:
            self.tpa_progress.emit(progress)

        pmt = open_pmt(enabled=self.tpa_pmt.isChecked(),
                       log=lambda m: self._log(f"[step 6] {m}"))

        def work() -> dict[str, Any]:
            daq.configure_monitor(settings)
            rows_by_pair: dict[int, list] = {}
            meta = _pmt_meta_start(pmt)
            try:
                for n, index in enumerate(pairs):
                    rows_by_pair[index] = measure_pair(
                        daq, controller, layout, index, schedule,
                        cfg=cfg, acq=acq, col_ratio=col_ratio,
                        read_timeout=read_timeout,
                        step0=n * len(schedule), total=total,
                        progress_callback=report, stop_event=stop_event,
                        row_hook=None if pmt is None else pmt.sample,
                    )
            except PairV2Aborted:
                _pmt_meta_end(pmt, meta)
                return {"status": "aborted", "rows": rows_by_pair, "cfg": cfg,
                        "layout": layout, "meta": meta}
            _pmt_meta_end(pmt, meta)
            return {"status": "ok", "rows": rows_by_pair, "cfg": cfg,
                    "layout": layout, "meta": meta}

        self._run_slm_task(
            "TPA η pair sweep (v2)", work, self._tpa_finished, self._tpa_error
        )

    def _tpa_stop(self) -> None:
        if self.tpa_stop_event is not None:
            self.tpa_stop_event.set()
            self.tpa_status.setText("Stopping\N{HORIZONTAL ELLIPSIS}")

    def _on_tpa_progress(self, progress: PairV2Progress) -> None:
        """Feed one acquisition to the shared progress dialog.

        Adapted rather than reported natively: the dialog already owns the
        elapsed/ETA arithmetic and the live plot, and it already carries a
        ``pair_eta`` phase, so step 6 gets the same window steps 1-3 pop instead
        of a second progress widget with its own idea of how to estimate time.
        ``y`` is the reading, so the plot fills in as the sweep runs and a dead
        or blocked beam is visible immediately rather than at the fit.
        """
        wide = "  [wide range]" if progress.range_v != self._tpa_acq().range_v else ""
        message = (
            f"pair {progress.pair_index} rep {progress.repeat} "
            f"x={progress.x:.2f} w={progress.w:.2f} → "
            f"{progress.mean_v*1e3:.4f} mV  "
            f"std {progress.std_ratio*100:.2f}%{wide}"
        )
        self.tpa_status.setText(message)
        if self.calibration_dialog is not None:
            self.calibration_dialog.update_progress(CalibrationProgress(
                phase="pair_eta",
                step=progress.step - 1,        # the dialog adds 1 back
                total=progress.total,
                message=message,
                x=float(progress.step),
                y=progress.mean_v,
            ))

    def _tpa_rail_v(self) -> float:
        """The page's rail ceiling in VOLTS, for the guards that reject by it."""
        return float(self.tpa_rail.value()) / 1e3

    def _tpa_fit_rows(self, rows_by_pair: dict[int, list],
                      layout=None) -> list[str]:
        """Fit every pair into ``self.tpa_fits``; return the pairs that failed.

        The fit reads the LEVELS off the rows, so a loaded file is fitted as it
        was swept and the ramp controls play no part.  The rail does: it is a
        rejection rule, not a drive setting, so a re-fit applies whatever is in
        the box now -- which is the point of it being a box.
        """
        self.tpa_rows = rows_by_pair
        self.tpa_fits = []
        failed: list[str] = []
        # Over the whole sweep, not per pair: the floor is a property of the
        # detector, and it is a voltage, so it has to be re-measured whenever
        # the gain moves.  A file with no dark level has no floor and no fit.
        try:
            floor = sweep_floor(rows_by_pair)
        except ValueError as exc:
            self.tpa_floor = None
            return [f"sweep: {exc}"]
        self.tpa_floor = floor
        self._log(f"[step 6] sigma floor {floor[0]*1e6:.1f} µV [{floor[1]}]")
        rail_v = self._tpa_rail_v()
        for index in sorted(rows_by_pair):
            # average_levels drops these itself; logged here because a burst is a
            # BENCH fault, and silent cleaning would let a glitching bench read
            # as a clean run (the offline step-6 script prints the same thing).
            rows = list(rows_by_pair[index])
            for row in (r for r, bad in zip(rows, burst_rows(rows, rail_v=rail_v))
                        if bad):
                self._log(
                    f"[step 6] pair {index}: BURST acquisition dropped — "
                    f"x={row[1]:g} w={row[2]:g} trial {int(row[0])} read "
                    f"{row[3]*1e3:+.4f} mV with a {row[4]*1e3:.4f} mV trace std"
                )
            try:
                fit = fit_pair(index, average_levels(rows, floor=floor,
                                                     rail_v=rail_v))
            except (ValueError, np.linalg.LinAlgError) as exc:
                failed.append(f"pair {index}: {exc}")
                continue
            if layout is not None:
                fit.wl_x_nm, fit.wl_w_nm, fit.nominal_wl_nm = self._tpa_wavelengths(
                    layout, fit.cfg, index
                )
            self.tpa_fits.append(fit)
        return failed

    def _tpa_finished(self, payload: dict[str, Any]) -> None:
        self.tpa_stop_event = None
        aborted = payload.get("status") == "aborted"
        rows = payload.get("rows") or {}
        if not rows:
            self._tpa_set_running(False)
            self.tpa_status.setText("Sweep stopped before any pair finished.")
            self._tpa_close_dialog(False, "Stopped before any pair finished.")
            return
        self.tpa_meta = payload.get("meta") or {}
        failed = self._tpa_fit_rows(rows, payload.get("layout"))
        self._tpa_set_running(False)
        self._tpa_fill_table()
        self._tpa_redraw()

        etas = [f.eta for f in self.tpa_fits if np.isfinite(f.eta)]
        if not etas:
            summ = "no pair fitted"
        elif len(etas) == 1:
            summ = f"η = {etas[0]:.4g}"
        else:
            summ = (f"{len(etas)} pairs · η "
                    f"{min(etas):.3g}–{max(etas):.3g}")
        head = "Stopped" if aborted else "Done"
        note = f"  · {len(failed)} fit(s) failed" if failed else ""
        self.tpa_status.setText(f"{head} · {summ}{note}")
        for line in failed:
            self._log(f"[step 6] fit failed — {line}")
        if any(not np.isfinite(f.eta) for f in self.tpa_fits):
            # b < 0 is almost always the sign convention, not the physics.
            self.tpa_status.setText(
                self.tpa_status.text()
                + "  — η undefined (b < 0); check the Invert checkbox"
            )
        self._tpa_autosave()
        self._tpa_close_dialog(not aborted, self.tpa_status.text())

    def _tpa_close_dialog(self, success: bool, message: str) -> None:
        """Freeze the progress window and let it be closed.

        The dialog is left open rather than dismissed: on a run this long the
        last thing a user wants is the log and the trace disappearing the
        moment it ends.
        """
        if self.calibration_dialog is not None:
            self.calibration_dialog.finish(success, message)

    def _tpa_error(self, _error: str) -> None:
        self.tpa_stop_event = None
        self._tpa_set_running(False)
        self.tpa_status.setText("TPA sweep failed (see Status log)")
        self._tpa_close_dialog(False, "TPA sweep failed (see Status log)")

    # ---- results table -----------------------------------------------------
    def _tpa_fill_table(self) -> None:
        """One row per pair: every column a scalar, so ten pairs stay readable."""
        self.tpa_table.blockSignals(True)
        self.tpa_table.setRowCount(len(self.tpa_fits))
        for row, fit in enumerate(self.tpa_fits):
            pulls = fit.pulls
            checks = fit.checks or {}
            marks = []
            for rec in checks.get("product") or []:
                for pr in rec.get("pairs") or []:
                    if abs(pr.get("pull", 0.0)) > 3.0:
                        marks.append("x·w")
            flags = ",".join(m for m in marks if m) or "OK"
            cells = [
                str(fit.index),
                f"{fit.eta:.5f}",
                f"{fit.eta_err:.5f}",
                f"{fit.bg['a_x'][0]*1e3:.4f}",
                f"{fit.bg['a_w'][0]*1e3:.4f}",
                f"{fit.bg['d'][0]*1e3:.4f}",
                f"{fit.r2:.5f}",
                f"{np.max(np.abs(pulls)):.2f}" if pulls.size else "–",
                flags,
            ]
            for col, text in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(text)
                if col:
                    item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                self.tpa_table.setItem(row, col, item)
        self.tpa_table.resizeColumnsToContents()
        self.tpa_table.blockSignals(False)
        if self.tpa_fits:
            self.tpa_table.selectRow(0)

    def _tpa_selected_fit(self) -> "PairV2Fit | None":
        if not self.tpa_fits:
            return None
        row = self.tpa_table.currentRow()
        if row < 0 or row >= len(self.tpa_fits):
            row = 0
        return self.tpa_fits[row]

    def _tpa_redraw(self) -> None:
        fit = self._tpa_selected_fit()
        self._tpa_draw_estimator(fit)
        self._tpa_draw_check(fit)

    # ---- panels ------------------------------------------------------------
    def _tpa_draw_estimator(self, fit) -> None:
        """D(w) with the GLS line, over its own pulls on a shared w axis.

        Drawn natively rather than through ``fit.pair_v2.make_plot`` because
        that renders six panels on a light ground for the PNG; the Save button
        still writes exactly that file, so the artifact on disk and this view
        come from the same fit even though they are laid out differently.
        """
        self.tpa_est_fig.clear()
        self.tpa_est_fig.patch.set_facecolor("#101820")
        ax, axp = self.tpa_est_fig.subplots(
            2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
        self._style_dark_axes(ax)
        self._style_dark_axes(axp)
        axp.set_xlabel("w   (x = 1 on this line)")
        ax.set_ylabel("D(w) = Y(1,w) − B̂(w)   (mV)")
        axp.set_ylabel("pull")
        if fit is None:
            ax.text(0.5, 0.5, "Run or load a sweep", ha="center", va="center",
                    transform=ax.transAxes, color="#d8dee9")
            self.tpa_est_canvas.draw_idle()
            return

        lo, hi = fit.fit_w_range
        ax.axvspan(lo, hi, color="#8fd14f", alpha=0.10,
                   label=f"fitted range [{lo:g}, {hi:g}]")
        w = fit.fit_w
        ax.errorbar(w, fit.fit_d * 1e3, yerr=fit.fit_sigma * 1e3, fmt="o",
                    color="#8fd14f", ecolor="#41515c", ms=5, lw=0, elinewidth=0.9,
                    label="D(w) fitted")
        grid_w = np.linspace(0.0, 1.05, 50)
        ax.plot(grid_w, (fit.b * grid_w + fit.beta0) * 1e3, "-", color="#8fd14f",
                lw=1.2, label="GLS  D = η²w + β0")
        ax.set_title(f"pair {fit.index}   η = {fit.eta:.4g} ± {fit.eta_err:.2g}",
                     color="#d8dee9", fontsize=9)
        ax.legend(loc="upper left", fontsize=7)

        axp.axhspan(-1, 1, color="#8fd14f", alpha=0.12)
        axp.axhline(0.0, color="#e0a447", ls="--", lw=1.0)
        if fit.pulls.size:
            axp.scatter(w, fit.pulls, c="#8fd14f", s=30,
                        edgecolor="#101820", lw=0.4)
            axp.set_ylim(-max(1.5, float(np.max(np.abs(fit.pulls))) * 1.3),
                         max(1.5, float(np.max(np.abs(fit.pulls))) * 1.3))
        self.tpa_est_canvas.draw_idle()

    def _tpa_draw_check(self, fit) -> None:
        """Product check: levels sharing one x·w must give one TPA residue.

        The model says Y depends on the two drives only through their product,
        so these points sit at different drive splits and must agree.  A split
        here does not make eta wrong -- it means the pair has no single eta at
        all, which is what steps 7 and 8 assume it has.
        """
        self.tpa_check_fig.clear()
        self.tpa_check_fig.patch.set_facecolor("#101820")
        ax = self.tpa_check_fig.add_subplot(111)
        self._style_dark_axes(ax)
        ax.set_ylabel("TPA residue  Y − B̂(x,w)   (mV)")
        records = (fit.checks or {}).get("product") if fit is not None else None
        if not records:
            ax.text(0.5, 0.5, "no product check in this run", ha="center",
                    va="center", transform=ax.transAxes, color="#d8dee9",
                    fontsize=8)
            self.tpa_check_canvas.draw_idle()
            return

        labels: list[str] = []
        pos = 0
        for rec in records:
            pts = rec.get("points") or []
            xs = list(range(pos, pos + len(pts)))
            ax.errorbar(xs, [p["tpa"] * 1e3 for p in pts],
                        yerr=[p["tpa_err"] * 1e3 for p in pts], fmt="o",
                        color="#a678de", ecolor="#a678de", ms=6, lw=0,
                        elinewidth=0.9)
            expected = rec.get("expected")
            if expected is not None and np.isfinite(expected):
                ax.hlines(expected * 1e3, xs[0] - 0.4, xs[-1] + 0.4,
                          color="#8fd14f", ls="--", lw=1.1,
                          label="η²(x·w) from the slope fit")
            labels += [f"({p['x']:g}, {p['w']:g})\nx·w={rec['product']:g}"
                       for p in pts]
            pos += len(pts)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=7, color="#d8dee9")
        ax.set_xlim(-0.6, len(labels) - 0.4)
        handles, lbls = ax.get_legend_handles_labels()
        if handles:
            ax.legend([handles[0]], [lbls[0]], loc="best", fontsize=7)
        self.tpa_check_canvas.draw_idle()

    # ---- files -------------------------------------------------------------
    def _tpa_load(self) -> None:
        """Re-fit the sweep this calib file records, and show it.  Never writes.

        ONE input, asked for: a calib.json.  Its step6 section says which CSV it
        was fitted from and its step3 section labels the wavelengths, so looking
        at last night's sweep is opening last night's calib file and there is no
        second path to keep in step.  The fit reads that file's own levels; the
        ramp controls describe the NEXT sweep and play no part here.

        Reads only, and the page's own calib file is not touched -- an offline
        re-fit is a copy of the JSON and its CSV and a run of
        ``calib_step6_v2.py`` against the copy, which is where a fit that is
        meant to be kept comes from.
        """
        picked = self._open_calib("Open a calibration to re-fit its step 6", 6)
        if picked is None:
            return
        try:
            calib = load_calib(picked)
            path = calib.csv_path(6)
        except Exception as exc:                        # noqa: BLE001
            self.tpa_status.setText(f"Load failed: {exc}")
            return
        if not path.is_file():
            self.tpa_status.setText(
                f"Load failed: {calib.path.name} names {path.name}, which is "
                f"not in {path.parent} — copy the CSV in beside it."
            )
            return
        try:
            rows = load_meas_csv(path)
        except Exception as exc:
            self.tpa_status.setText(f"Load failed: {exc}")
            return
        # A re-fit needs only the CSV.  The step-3 section labels the
        # wavelengths when it is there, and is simply left out when it is not --
        # the same tolerance `fit_csv` has, so a sweep is always re-fittable.
        try:
            _calib, layout = self._tpa_load_step3(picked)
            note3 = ""
        except Exception as exc:
            layout = None
            note3 = f"  · wavelengths NaN ({exc})"
        # A Load re-fits someone else's sweep, so its thermistor bracket comes
        # from the file rather than from this session -- otherwise a Save would
        # stamp the loaded rows with the meta of whatever ran here last.
        self.tpa_meta = meas_csv_meta(path)
        failed = self._tpa_fit_rows(rows, layout)
        self._tpa_fill_table()
        self._tpa_redraw()
        note = f"  · {len(failed)} fit(s) failed" if failed else ""
        if self.tpa_fits:
            lo, hi = self.tpa_fits[0].fit_w_range
            note += f"  · fitted over w {lo:g}–{hi:g}, as swept"
        self.tpa_status.setText(
            f"Loaded {self._calib_name(calib.path)} · {Path(path).name} · "
            f"{len(self.tpa_fits)} pair(s) re-fit{note}{note3}"
        )
        for line in failed:
            self._log(f"[step 6] fit failed — {line}")

    def _tpa_autosave(self) -> None:
        """A finished sweep writes itself: the raw CSV, then the step6 section.

        Both into the calib file named on this page -- the CSV beside it, the
        section in it -- so there is nothing to keep in step and no dialog to
        get wrong.  Writing DROPS that file's step 7 and step 8: they were
        fitted against the etas this sweep has just replaced.

        No PNGs.  Load draws the same fit from the CSV the section points at,
        which is better than a picture of it.

        Only a sweep calls this.  Loading a recorded sweep re-fits and draws but
        never writes -- Load is for inspection, and opening last week's file to
        look at it must not overwrite its calibration.

        The rows are written first and unconditionally.  A measurement is an
        hour of bench time and must never be lost to a missing or unreadable
        calibration, which is only needed for the derived section.
        """
        if not self.tpa_fits:
            return
        stamp = time.strftime("%m%d_%H%M")
        calib_file = self._calib_file_path(6)
        if calib_file is None:
            folder = self._unfiled_run(stamp)
            self._log(
                f"[step 6] no calibration file on the page — rows parked in "
                f"{folder}"
            )
        else:
            folder = calib_file.parent
            folder.mkdir(parents=True, exist_ok=True)
        csv_path = folder / f"calib_step6v2_meas_{stamp}.csv"
        written = [Path(write_meas_csv(self.tpa_rows, csv_path,
                                       meta=self.tpa_meta)).name]

        try:
            _calib, layout = self._tpa_load_step3()
        except Exception as exc:                        # noqa: BLE001
            self.tpa_status.setText(
                f"Saved {written[0]} — no usable calibration file ({exc}), so "
                "the step6 section was not written (step 7 could not read it)."
            )
            return
        try:
            calib = save_step6_section(
                self.tpa_fits, calib_file, csv=csv_path,
                center_wl=float(getattr(layout, "center_wl", 0.0)), log=None)
            written.append(f"step6 -> {calib.path.name}")
        except Exception as exc:
            self.tpa_status.setText(f"Saved {written[0]}; section failed: {exc}")
            return
        self._calib_describe_all()
        self.tpa_status.setText(
            self.tpa_status.text() + "  ·  saved " + "  +  ".join(written))

    # ===================== TPA comb phase (step 7) tab ==================
    _TPA_PHASE_COLUMNS = (
        "pair", "ΔΦ (°)", "±tot", "±fringe", "±η",
        "a (mV½)", "b (mV½)", "R²", "max|pull|", "resid (mV)",
    )

    def _build_tpa_phase_tab(self) -> QtWidgets.QWidget:
        """Step 7 v2: controls on the left, results on the right.

        The same split as step 6, for the same reason -- a DAQ group, a sweep
        group and a per-target results table do not fit above the plots.  Read
        top to bottom on the left, then across.
        """
        page = self._page_shell("Comb Phase (ΔΦ_comb) Calibration")

        self.tpa_phase_status = self._page_status(page, "\N{EN DASH}")
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.addWidget(self._build_tpa_phase_controls())
        split.addWidget(self._build_tpa_phase_results())
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([360, 1040])
        page.layout().addWidget(split, 1)

        self._tpa_phase_redraw()
        return page

    def _build_tpa_phase_controls(self) -> QtWidgets.QWidget:
        """The left column: the calib file, DAQ, sweep, buttons -- as on step 6.

        The file first, and it is the one required input: it decides both what
        is driven (its step-3 section IS the channel layout) and what the fringe
        amplitudes are pinned to (its step-6 etas), so a run against the wrong
        one does not fail -- it reports a phase measured under a different
        aperture against someone else's etas.

        It does not scroll, for the reason step 6's does not: everything that
        is not a bench constant fits, and a setting you have to scroll to find
        is a setting you will not check.
        """
        col = QtWidgets.QWidget()
        box = QtWidgets.QVBoxLayout(col)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(12)

        box.addWidget(self._calib_file_row(7))
        self.tpa_phase_step6_edit = self._calib_edits[7]
        self.tpa_phase_step6_label = self._calib_labels[7]

        # --- DAQ acquisition --------------------------------------------
        daq = self._panel("DAQ · acquisition")
        grid = QtWidgets.QGridLayout(daq)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        self.tpa_phase_daq_channel = QtWidgets.QLineEdit("ai0")
        self.tpa_phase_daq_rate = QtWidgets.QDoubleSpinBox()
        self.tpa_phase_daq_rate.setRange(1.0, 2_000_000.0)
        self.tpa_phase_daq_rate.setDecimals(0)
        self.tpa_phase_daq_rate.setValue(1000.0)
        self.tpa_phase_daq_rate.setSuffix(" S/s")
        self.tpa_phase_daq_range = QtWidgets.QComboBox()
        for lo, hi in self._DAQ_RANGES:
            self.tpa_phase_daq_range.addItem(f"\N{PLUS-MINUS SIGN}{hi:g} V", (lo, hi))
        self._select_daq_range(self.tpa_phase_daq_range, BENCH_ACQ.range_v)
        self.tpa_phase_daq_range.setToolTip(BENCH_RANGE_TIP)
        self.tpa_phase_daq_fcut = QtWidgets.QDoubleSpinBox()
        self.tpa_phase_daq_fcut.setRange(0.1, 100_000.0)
        self.tpa_phase_daq_fcut.setDecimals(1)
        self.tpa_phase_daq_fcut.setValue(20.0)
        self.tpa_phase_daq_fcut.setSuffix(" Hz")
        self.tpa_phase_daq_fcut.setToolTip(
            "Detector 3 dB bandwidth: the low-pass behind the reported mean and "
            "std. That std is the sigma the fit weights by."
        )
        self.tpa_phase_settle = self._double_spin(0.0, 10.0, 0.25, " s", 3)
        self.tpa_phase_settle.setToolTip(
            "Wait after each SLM pattern change, before reading."
        )
        # Step 6's windows, not its own: step 7 pins its amplitudes to step 6's
        # etas, so the two have to have averaged the same noise.
        self.tpa_phase_tmeas = self._double_spin(
            0.001, 60.0, BENCH_ACQ.t_meas_s, " s", 3)
        self.tpa_phase_tmeas.setToolTip(BENCH_WINDOW_TIP)
        self.tpa_phase_invert = QtWidgets.QCheckBox("Invert sign (TIA)")
        self.tpa_phase_invert.setChecked(True)
        self.tpa_phase_invert.setToolTip(
            "The transimpedance amplifier outputs NEGATIVE volts for positive "
            "light, so recording a positive light signal means inverting.\n"
            "Leave this on. A sign-flipped run does not fail here — it fits a "
            "fringe upside down against pinned step-6 amplitudes and reports a "
            "phase, so check the residual, not just that a number came out."
        )
        self.tpa_phase_pmt = QtWidgets.QCheckBox("Log PMT temperature")
        self.tpa_phase_pmt.setChecked(True)
        self.tpa_phase_pmt.setToolTip(PMT_CHECKBOX_TIP)

        rows = [
            ("Channel", self.tpa_phase_daq_channel),
            ("Range", self.tpa_phase_daq_range),
            ("T_meas", self.tpa_phase_tmeas),
        ]
        for r, (label, widget) in enumerate(rows):
            grid.addWidget(QtWidgets.QLabel(label), r, 0)
            grid.addWidget(widget, r, 1)
        grid.addWidget(self.tpa_phase_invert, len(rows), 0, 1, 2)
        # Bench constants, hidden for the reason step 6's are: they are the
        # same on every run, and the four that are not are easier to check on a
        # panel that holds only them.  Still live -- _tpa_phase_acq and the
        # sweep read them.
        for widget in (self.tpa_phase_daq_rate, self.tpa_phase_daq_fcut,
                       self.tpa_phase_settle, self.tpa_phase_pmt):
            widget.setParent(daq)
            widget.hide()
        box.addWidget(daq)

        # --- sweep -------------------------------------------------------
        sweep = self._panel("Sweep")
        sgrid = QtWidgets.QGridLayout(sweep)
        sgrid.setHorizontalSpacing(8)
        sgrid.setVerticalSpacing(6)

        self.tpa_phase_ref = self._spin(0, 63, 1)
        self.tpa_phase_ref.setToolTip(
            "Reference pair label, 1-based. It defines ΔΦ_comb = 0, so every "
            "phase in the result is relative to this one."
        )
        self.tpa_phase_targets = QtWidgets.QLineEdit("2-6")
        self.tpa_phase_targets.setToolTip(
            "Pair labels to measure, 1-based: \"2-6\", \"1,3,5\" or a mix.\n"
            "The reference is skipped if it appears here.\n\n"
            "A pair CANNOT be swept against its neighbour — the two sit side by "
            "side on the panel and the leakage is not in the model — so any pair "
            "within ±1 of the reference is measured against a hub further out "
            "instead, and its phase is chained back to the reference. The plan "
            "is worked out for you and printed in the progress window; the run "
            "is refused up front if some pair cannot be reached at all."
        )
        sgrid.addWidget(QtWidgets.QLabel("Reference"), 2, 0)
        sgrid.addWidget(self.tpa_phase_ref, 2, 1, 1, 2)
        sgrid.addWidget(QtWidgets.QLabel("Targets"), 3, 0)
        sgrid.addWidget(self.tpa_phase_targets, 3, 1, 1, 2)

        # The ramp.  Both arms deliberately stop below 1.0: step 6 fits its etas
        # over [0.2, 0.9] and EXCLUDES the measured (1, 1) point, so driving
        # either arm fully on pins this fringe to an extrapolation.
        self.tpa_phase_sweep_min = self._double_spin(0.01, 1.0, 0.10, "", 2)
        self.tpa_phase_sweep_min.setSingleStep(0.05)
        self.tpa_phase_sweep_min.setToolTip(
            "Lowest per-side target intensity in the ramp (x_t = w_t = this)."
        )
        self.tpa_phase_sweep_max = self._double_spin(0.02, 1.0, 0.90, "", 2)
        self.tpa_phase_sweep_max.setSingleStep(0.05)
        self.tpa_phase_sweep_max.setToolTip(
            "Highest per-side target intensity in the ramp.\n"
            "Keep it at 0.9. At 1.0 the amplitudes are pinned to a step-6 "
            "extrapolation, and d(ΔΦ_SLM)/dv diverges there while the trace std "
            "is smallest — so 1/std² weighting hands that one point most of the "
            "fit (~70% of the Fisher information on the 0903 pair-3 fringe)."
        )
        self.tpa_phase_points = self._spin(2, 60, 10)
        self.tpa_phase_points.setToolTip(
            "Points on the ramp, evenly spaced from min to max.\n"
            "0.1 → 0.9 sweeps the shared panel phase over ~37..143°, most of "
            "the half fringe."
        )
        self.tpa_phase_ref_level = self._double_spin(0.01, 1.0, 0.90, "", 2)
        self.tpa_phase_ref_level.setSingleStep(0.05)
        self.tpa_phase_ref_level.setToolTip(
            "The reference pair is held at x_r = w_r = this for every point.\n"
            "0.9 for the same reason the ramp stops there — the fit takes "
            "a = η_ref·√(x_r·w_r), so a reference below 1.0 needs no other "
            "change."
        )
        sgrid.addWidget(QtWidgets.QLabel("Sweep min"), 4, 0)
        sgrid.addWidget(self.tpa_phase_sweep_min, 4, 1, 1, 2)
        sgrid.addWidget(QtWidgets.QLabel("Sweep max"), 5, 0)
        sgrid.addWidget(self.tpa_phase_sweep_max, 5, 1, 1, 2)
        sgrid.addWidget(QtWidgets.QLabel("Ramp points"), 6, 0)
        sgrid.addWidget(self.tpa_phase_points, 6, 1, 1, 2)
        sgrid.addWidget(QtWidgets.QLabel("Ref level"), 7, 0)
        sgrid.addWidget(self.tpa_phase_ref_level, 7, 1, 1, 2)

        self.tpa_phase_single_beam = QtWidgets.QCheckBox(
            "Step-6 single-beam background"
        )
        self.tpa_phase_single_beam.setChecked(True)
        self.tpa_phase_single_beam.setToolTip(
            "Subtract both pairs' step-6 single-beam response as a FIXED "
            "background. The reference contributes a constant, the swept target "
            "a ramp; without this the fringe has to absorb that ramp.\n"
            "Leave it on — turn it off only to diagnose step 6."
        )
        sgrid.addWidget(self.tpa_phase_single_beam, 8, 0, 1, 3)

        # The same ceiling step 6 carries, and it matters more here: step 7's
        # all-off dark is ONE unrepeated read, subtracted from all ten of its
        # target's drive rows, so a single railed dark takes the whole fringe
        # with it (run_0930_noYsat pair 4, R^2 = -11438).  It is the only guard
        # that catches that, because it needs no reference beyond itself.
        self.tpa_phase_rail = self._double_spin(
            10.0, 1000.0, 1e3 * BURST_MAX_V, " mV", 0)
        self.tpa_phase_rail.setSingleStep(50.0)
        self.tpa_phase_rail.setToolTip(
            "Any read at or above this is a railed detector, not a measurement, "
            "and is dropped \N{EM DASH} drive rows and the all-off dark alike.\n"
            "A bench-fault ceiling, not a saturation limit. A railed DARK is the "
            "expensive one: it is subtracted from every drive row of its target, "
            "so it is replaced with the other targets' median and the swap is "
            "recorded in the step7 section.\n"
            "Applies to the next sweep AND to a Load's re-fit.")
        sgrid.addWidget(QtWidgets.QLabel("Rail"), 9, 0)
        sgrid.addWidget(self.tpa_phase_rail, 9, 1, 1, 2)
        box.addWidget(sweep)

        # --- buttons -----------------------------------------------------
        # No progress bar here, same as step 6: a run takes tens of minutes, so
        # it gets the shared CalibrationProgressDialog (bar, elapsed/ETA, live
        # plot, Stop) rather than a strip in a column nobody is watching.
        #
        # No status line either: `tpa_phase_status` is beside the page title.
        # The read-rejection rules used to be spelled out down here as well --
        # the rail is a control now and says what it does in its own tooltip,
        # and the burst guard and the inherited sigma floor announce themselves
        # in the log when they fire, which is when anybody wants them.
        self.tpa_phase_run_button = QtWidgets.QPushButton("Run Sweep")
        self.tpa_phase_run_button.clicked.connect(self._tpa_phase_run)
        self.tpa_phase_stop_button = QtWidgets.QPushButton("Stop")
        self.tpa_phase_stop_button.setProperty("variant", "danger")
        self.tpa_phase_stop_button.setEnabled(False)
        self.tpa_phase_stop_button.clicked.connect(self._tpa_phase_stop)
        self.tpa_phase_load_button = QtWidgets.QPushButton(
            "Load\N{HORIZONTAL ELLIPSIS}"
        )
        self.tpa_phase_load_button.setProperty("variant", "ghost")
        self.tpa_phase_load_button.setToolTip(
            "Load a recorded step-7 measurement CSV; every target in it is "
            "re-fit against the step-6 JSON named above."
        )
        self.tpa_phase_load_button.clicked.connect(self._tpa_phase_load)
        btns = QtWidgets.QGridLayout()
        btns.addWidget(self.tpa_phase_load_button, 0, 0)
        btns.addWidget(self.tpa_phase_run_button, 1, 0)
        btns.addWidget(self.tpa_phase_stop_button, 1, 1)
        box.addLayout(btns)

        box.addStretch(1)
        return col

    def _build_tpa_phase_results(self) -> QtWidgets.QWidget:
        """The right column: the all-targets table over the selected fringe."""
        self.tpa_phase_table = QtWidgets.QTableWidget(
            0, len(self._TPA_PHASE_COLUMNS)
        )
        self.tpa_phase_table.setHorizontalHeaderLabels(
            list(self._TPA_PHASE_COLUMNS)
        )
        self.tpa_phase_table.verticalHeader().setVisible(False)
        self.tpa_phase_table.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers
        )
        self.tpa_phase_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows
        )
        self.tpa_phase_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SingleSelection
        )
        self.tpa_phase_table.setAlternatingRowColors(True)
        self.tpa_phase_table.horizontalHeader().setStretchLastSection(True)
        # The table IS the target selector -- no separate combo box.
        self.tpa_phase_table.itemSelectionChanged.connect(self._tpa_phase_redraw)

        # One figure, not two panels: `plot_fringe` lays the fringe and its
        # pulls out side by side itself, and it is the same renderer the saved
        # PNG uses -- so what is reviewed here and what is archived beside the
        # JSON are the same picture rather than two drawings of one fit.
        self.tpa_phase_fig = Figure(figsize=(10, 4.2), tight_layout=True)
        self.tpa_phase_canvas = FigureCanvas(self.tpa_phase_fig)
        self.tpa_phase_canvas.setMinimumHeight(300)

        stack = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        stack.addWidget(self._panel_with_widget("Targets", self.tpa_phase_table))
        stack.addWidget(self._panel_with_widget(
            "Fringe fit · ΔΦ_comb is the only free parameter",
            self.tpa_phase_canvas,
        ))
        stack.setStretchFactor(0, 0)
        stack.setStretchFactor(1, 1)
        stack.setSizes([220, 520])
        return stack

    # ---- inputs ------------------------------------------------------------
    def _tpa_phase_step6_path(self) -> Path | None:
        """This page's calib.json -- a typed directory means the file inside it."""
        return self._calib_file_path(7)

    @staticmethod
    def _tpa_phase_pool(paths: "list[Path]") -> tuple:
        """``(targets, ref_of_each, csv_of_each)`` over every CSV a section names.

        One sweep is the normal case.  A section naming several is a run that was
        driven stage by stage and filed as one calibration, so the targets are
        pooled and each remembers which file to be read out of; a target in two
        files (the shared reference is in both) belongs to the first.
        """
        targets: list[int] = []
        recorded: dict[int, int] = {}
        csv_of: dict[int, Path] = {}
        for path in paths:
            refs = references_in_csv(path)
            for k in targets_in_csv(path):
                if k in csv_of:
                    continue
                targets.append(k)
                csv_of[k] = path
                if k in refs:
                    recorded[k] = refs[k]
        return targets, recorded, csv_of

    def _tpa_phase_load_step6(self, calib_file: "Path | None" = None):
        """``(layout, models)`` from a calib file, or raise.

        ``calib_file`` names the file; omitted, it is the one on the page.  A
        Load passes the file it was pointed at, so the etas the fringes are
        pinned to belong to the sweep being looked at.

        Both halves come out of the one calib.json, exactly as the offline
        runner's ``--calib`` does it: the step-3 section builds the layout, the
        step-6 pairs give the etas the fringe amplitudes are pinned to.  That is
        what guarantees the phase is measured through the aperture those etas
        were calibrated under.
        """
        path = (calib_file if calib_file is not None
                else self._tpa_phase_step6_path())
        if path is None:
            raise ValueError("no calibration file chosen")
        calib = load_calib(path)
        layout = channel_layout_from_calibration(calib.step3_result())
        models = load_pair_models([calib.path])
        if not models:
            raise ValueError("no fitted pairs in its step6 section")
        return layout, models

    def _tpa_phase_describe_step6(self) -> None:
        """Say what the chosen file actually is, before an hour is spent on it."""
        path = self._tpa_phase_step6_path()
        if path is None:
            self._set_status(
                self.tpa_phase_step6_label,
                "No calibration file chosen — required to run.", "off",
            )
            return
        try:
            layout, models = self._tpa_phase_load_step6()
        except Exception as exc:
            self._set_status(
                self.tpa_phase_step6_label,
                f"{self._calib_name(path)}: {exc}", "error"
            )
            return
        pairs = sorted(models)
        self._set_status(
            self.tpa_phase_step6_label,
            f"{self._calib_name(path)} · {self._run_chain(path)} "
            f"· {layout.n_channels} pairs "
            f"· η for "
            f"{self._compact_indices(pairs)}",
            "ok",
        )

    @staticmethod
    def _compact_indices(values: Sequence[int]) -> str:
        """[2,3,4,6] -> \"2-4,6\" — a pair list that stays one line at ten pairs."""
        out: list[str] = []
        for v in sorted(values):
            if out and v == int(out[-1].split("-")[-1]) + 1:
                lo = out[-1].split("-")[0]
                out[-1] = f"{lo}-{v}"
            else:
                out.append(str(v))
        return ",".join(out) or "none"

    def _tpa_phase_config(self) -> PhaseV2Config:
        """The drive, straight off the sweep controls.

        Falls back to the validated defaults if the spinboxes do not make a
        ramp -- a half-typed value must not stop the page redrawing, and Run
        validates them again before it drives anything.
        """
        try:
            return PhaseV2Config(
                ref_index=self.tpa_phase_ref.value(),
                ref_level=self.tpa_phase_ref_level.value(),
                sweep_min=self.tpa_phase_sweep_min.value(),
                sweep_max=self.tpa_phase_sweep_max.value(),
                n_points=self.tpa_phase_points.value(),
            )
        except ValueError:
            return PhaseV2Config()

    def _tpa_phase_acq(self) -> PairV2Acq:
        """Acquisition timing and input range, straight off the DAQ group."""
        _lo, hi = self.tpa_phase_daq_range.currentData()
        return PairV2Acq(
            t_meas_s=float(self.tpa_phase_tmeas.value()),
            settle_s=float(self.tpa_phase_settle.value()),
            range_v=float(hi),
            invert=self.tpa_phase_invert.isChecked(),
        )

    def _tpa_phase_set_running(self, running: bool) -> None:
        """See :meth:`_tpa_set_running`."""
        self._tpa_phase_set_running_impl(running)
        if not running:
            self._calib_gate_step(7)

    def _tpa_phase_set_running_impl(self, running: bool) -> None:
        self._mark_tpa_calibration("step7", running)
        self.tpa_phase_run_button.setEnabled(not running)
        self.tpa_phase_stop_button.setEnabled(running)
        self.tpa_phase_load_button.setEnabled(not running)
        self.tpa_phase_targets.setEnabled(not running)

    # ---- run ---------------------------------------------------------------
    def _tpa_phase_plan(self):
        """``(cfg, layout, models, stages)`` for the run, or raise ``ValueError``.

        Every check that does NOT need an instrument, in one place: it is what
        the page can say about a run before anything is connected, and it is
        what the tests can drive without a bench.

        The target list is what you want phases FOR, not what one sweep can
        reach: a pair next to the reference is unmeasurable against it, so
        :func:`~calibration_module.fit.cascade.plan_cascade` splits the run into
        stages and the neighbours are measured against a hub and chained back.
        The cheap, local complaints are raised FIRST — a label outside the
        layout, or one step 6 has no η for, is a typo worth naming as a typo
        rather than reporting as an unreachable pair.
        """
        layout, models = self._tpa_phase_load_step6()
        cfg = self._tpa_phase_config()
        try:
            targets = [k for k in self._tpa_parse_pairs(
                self.tpa_phase_targets.text()) if k != cfg.ref_index]
        except ValueError as exc:
            raise ValueError(f"Bad target list: {exc}") from exc
        if not targets:
            raise ValueError(
                "Need at least one target pair that is not the reference."
            )
        bad = [k for k in [cfg.ref_index, *targets]
               if not (0 <= cfg.slot(k) < layout.n_channels)]
        if bad:
            raise ValueError(
                f"Pair(s) {bad} out of range — the layout has "
                f"{layout.n_channels} pairs, numbered from {cfg.pair_index_base}."
            )
        missing = [k for k in [cfg.ref_index, *targets] if k not in models]
        if missing:
            raise ValueError(
                f"No step-6 η for pair(s) {missing}; the file covers "
                f"{self._compact_indices(sorted(models))}."
            )
        # A step 6 that cannot give this fit its sigma floor: say so before
        # the hour is spent, not after.
        self._tpa_phase_floor()
        # CascadeError is a ValueError, so an unreachable pair reaches the
        # status label by the same path as every other refusal here.
        stages = plan_cascade(targets, cfg.ref_index, n_pairs=layout.n_channels)
        return cfg, layout, models, stages

    def _tpa_phase_plan_lines(self, stages: "Sequence[Stage]") -> list[str]:
        """The stage breakdown, costed with THIS page's drive and acquisition."""
        secs = phase_run_seconds(build_xw_sweep(self._tpa_phase_config()),
                                 self._tpa_phase_acq())
        return plan_lines(stages, self._tpa_phase_config().ref_index,
                          seconds_per_target=secs)

    def _tpa_phase_run(self) -> None:
        # The plan first: a typo in the target list is worth saying before
        # "connect the DAQ", because fixing that one does not need the bench.
        try:
            cfg, layout, models, stages = self._tpa_phase_plan()
        except (ValueError, FileNotFoundError) as exc:
            self.tpa_phase_status.setText(str(exc))
            return
        except Exception as exc:
            self.tpa_phase_status.setText(f"Step 6: {exc}")
            return
        daq = self.daq_controller
        if daq is None or not daq.is_connected:
            self.tpa_phase_status.setText("Connect the DAQ first (DAQ page).")
            return
        controller = self._controller()
        if not getattr(controller, "is_open", False):
            self.tpa_phase_status.setText(
                "Open the SLM on the SLM Control page first."
            )
            return

        acq = self._tpa_phase_acq()
        drive = build_xw_sweep(cfg)
        per_target = len(drive) + 1               # + the all-off dark
        n_targets = sum(len(s) for s in stages)
        total = per_target * n_targets
        lo, hi = self.tpa_phase_daq_range.currentData()
        settings = DAQMonitorSettings(
            channel=self.tpa_phase_daq_channel.text().strip() or "ai0",
            sample_rate=self.tpa_phase_daq_rate.value(),
            duration=acq.t_meas_s,
            single_duration=acq.t_meas_s,
            hold=0.0,                 # measure_target owns the settle
            min_val=lo, max_val=hi,
            f_cut=self.tpa_phase_daq_fcut.value(),
        )
        read_timeout = max(30.0, acq.t_meas_s * 3.0 + 10.0)

        plan_text = self._tpa_phase_plan_lines(stages)
        hops = [s for s in stages if s.depth]
        self.tpa_phase_status.setText(
            f"Running\N{HORIZONTAL ELLIPSIS} {n_targets} target(s) vs ref "
            f"{cfg.ref_index}, {len(drive)} points each"
            + (f" \N{MIDDLE DOT} {len(stages)} stages, "
               f"{sum(len(s) for s in hops)} chained" if hops else "")
        )
        self._tpa_phase_set_running(True)
        self._open_calibration_dialog(on_stop=self._tpa_phase_stop)
        # The plan goes in the dialog, not on the page: it is the one window
        # that is open for the whole run, and a cascade is worth seeing BEFORE
        # the hour is spent rather than inferred afterwards from which pair was
        # measured against what.
        if self.calibration_dialog is not None:
            self.calibration_dialog.note(
                [f"Reference {cfg.ref_index} \N{MIDDLE DOT} "
                 f"{self._compact_indices(sorted(models))} have step-6 \N{GREEK SMALL LETTER ETA}"]
                + plan_text
            )
        for line in plan_text:
            self._log(f"[step 7] {line}")

        stop_event = threading.Event()
        self.tpa_phase_stop_event = stop_event
        col_ratio = self._active_col_ratio()

        def report(progress: PhaseV2Progress) -> None:
            self.tpa_phase_progress.emit(progress)

        pmt = open_pmt(enabled=self.tpa_phase_pmt.isChecked(),
                       log=lambda m: self._log(f"[step 7] {m}"))

        def work() -> dict[str, Any]:
            daq.configure_monitor(settings)
            results: dict[int, PhaseResult] = {}
            done = 0
            meta = _pmt_meta_start(pmt)
            try:
                for stage in stages:
                    # Each stage drives its OWN reference; everything else about
                    # the sweep is shared, so only ref_index moves.
                    stage_cfg = replace(cfg, ref_index=stage.ref)
                    for k in stage.targets:
                        results[k] = measure_target(
                            daq, controller, layout, k, drive,
                            cfg=stage_cfg, acq=acq, col_ratio=col_ratio,
                            read_timeout=read_timeout,
                            step0=done * per_target, total=total,
                            progress_callback=report, stop_event=stop_event,
                            row_hook=None if pmt is None else pmt.sample,
                        )
                        done += 1
            except PhaseV2Aborted:
                _pmt_meta_end(pmt, meta)
                return {"status": "aborted", "results": results, "cfg": cfg,
                        "models": models, "stages": stages, "meta": meta}
            _pmt_meta_end(pmt, meta)
            return {"status": "ok", "results": results, "cfg": cfg,
                    "models": models, "stages": stages, "meta": meta}

        self._run_slm_task(
            "TPA comb-phase sweep (v2)", work,
            self._tpa_phase_finished, self._tpa_phase_error,
        )

    def _tpa_phase_stop(self) -> None:
        if self.tpa_phase_stop_event is not None:
            self.tpa_phase_stop_event.set()
            self.tpa_phase_status.setText("Stopping\N{HORIZONTAL ELLIPSIS}")

    def _on_tpa_phase_progress(self, progress: PhaseV2Progress) -> None:
        """Feed one acquisition to the shared progress dialog.

        Adapted rather than reported natively, exactly as step 6 does it: the
        dialog already owns the elapsed/ETA arithmetic and the live plot and
        already carries a ``comb_phase`` phase.  ``y`` is the reading, so the
        fringe draws itself as the sweep runs -- a dead reference or a blocked
        target shows immediately rather than at the fit.
        """
        wide = ("  [wide range]"
                if progress.range_v != self._tpa_phase_acq().range_v else "")
        if progress.single:
            message = (f"pair {progress.tgt_index} dark (all off) → "
                       f"{progress.mean_v*1e3:.4f} mV{wide}")
        else:
            message = (
                f"pair {progress.tgt_index} vs {progress.ref_index} "
                f"x=w={progress.x_t:.2f} → {progress.mean_v*1e3:.4f} mV  "
                f"std {progress.std_ratio*100:.2f}%{wide}"
            )
        self.tpa_phase_status.setText(message)
        if self.calibration_dialog is not None:
            self.calibration_dialog.update_progress(CalibrationProgress(
                phase="comb_phase",
                step=progress.step - 1,        # the dialog adds 1 back
                total=progress.total,
                message=message,
                x=float(progress.step),
                y=progress.mean_v,
            ))

    def _tpa_phase_fit_results(self, results: dict[int, PhaseResult],
                               models: dict[int, PairModel],
                               ref_index: int,
                               stages: "Sequence[Stage] | None" = None, *,
                               floor: "tuple[float, str] | None" = None) -> list[str]:
        """Fit every target into ``self.tpa_phase_results``; return the failures.

        Each target is fit against the reference it was actually SWEPT against
        -- its own stage's, which for a one-stage run is ``ref_index`` for
        everything -- and then :func:`~calibration_module.fit.cascade.chain_fits`
        refers the lot back to ``ref_index``.  The fits keep the hop they
        measured; only ``self.tpa_phase_chains`` speaks in root coordinates.
        """
        single_beam_bg = self.tpa_phase_single_beam.isChecked()
        # Step 7 has no dark LEVEL of its own to measure a floor from -- one
        # all-off read per target and ten drive rows -- so it inherits the one
        # step 6 measured on this run, through the same detector on the same
        # night.  See calibration_module.fit.sigma.
        # ``floor`` is given only by a caller that already holds it (a test
        # driving this path without a run on the page); a sweep inherits it.
        floor = self._tpa_phase_floor() if floor is None else floor
        self.tpa_phase_floor = floor
        self._log(f"[step 7] sigma floor {floor[0]*1e6:.1f} µV [{floor[1]}]")
        for r in results.values():
            r.sigma_floor_v, r.sigma_floor_source = float(floor[0]), str(floor[1])
        # The dark is the one quantity no other guard sees: no repeat, no sweep
        # median, and it is subtracted from every row of its target.
        for key, entry in guard_darks(results).items():
            self._log(
                f"[step 7] pair {key}: DARK OVER CEILING — recorded "
                f"{entry['recorded_v']*1e3:+.4f} mV, using "
                f"{entry['used_v']*1e3:+.4f} mV (drive rows untouched)"
            )
        self.tpa_phase_results = {}
        self.tpa_phase_chains = {}
        self.tpa_phase_root = int(ref_index)
        if stages is None:                      # a loaded CSV: one stage per ref
            stages = self._tpa_phase_stages_for(results, ref_index)
        self.tpa_phase_stages = list(stages)
        ref_of = {k: s.ref for s in stages for k in s.targets}
        failed: list[str] = []
        for k in sorted(results):
            result = results[k]
            # A live sweep never goes through load_phase_csv, so the guard has to
            # be applied here too -- otherwise the same sweep would fit one way
            # now and another way when it is refit from the CSV it just wrote.
            for b in drop_bursts(result):
                self._log(
                    f"[step 7] pair {k}: BURST acquisition dropped — "
                    f"x_t=w_t={b['x_t']:g} read {b['voltage_mean_v']*1e3:+.4f} mV "
                    f"with a {b['voltage_std_v']*1e3:.4f} mV trace std"
                )
            ref = ref_of.get(k, ref_index)
            if k not in models or ref not in models:
                failed.append(f"pair {k}: no step-6 η")
                self.tpa_phase_results[k] = result
                continue
            try:
                fit_result(result, models[k], models[ref],
                           comb_only=True, single_beam_bg=single_beam_bg)
            except (ValueError, np.linalg.LinAlgError) as exc:
                failed.append(f"pair {k}: {exc}")
            self.tpa_phase_results[k] = result
        self.tpa_phase_chains = chain_fits(
            {k: r.fit for k, r in self.tpa_phase_results.items()
             if r.fit is not None},
            self.tpa_phase_stages, ref_index,
        )
        # A pair whose HUB failed to fit has no route back to the root, so it
        # has no phase to report even though its own fringe was fine.  Say so
        # here rather than letting it show up as a silently missing row.
        for k, r in sorted(self.tpa_phase_results.items()):
            if r.fit is not None and k not in self.tpa_phase_chains:
                failed.append(
                    f"pair {k}: fit ok but its reference {ref_of.get(k, ref_index)} "
                    f"did not, so it cannot be chained to {ref_index}"
                )
        return failed

    @staticmethod
    def _tpa_phase_stages_for(results: "dict[int, PhaseResult]",
                              root: int) -> "list[Stage]":
        """Rebuild the stage list from results that already know their reference.

        Loading a CSV skips the planner entirely -- the file records what was
        driven, and that is the truth about it, plan or no plan.  Stages come
        out ordered by how far their reference is from the root so
        :func:`chain_fits` always resolves a hub before what hangs off it.
        """
        by_ref: dict[int, list[int]] = {}
        for k, r in sorted(results.items()):
            by_ref.setdefault(int(r.ref_index), []).append(int(k))
        depth = {int(root): 0}
        stages: list[Stage] = []
        pending = dict(by_ref)
        while pending:
            ready = [ref for ref in sorted(pending) if ref in depth]
            if not ready:                       # orphaned refs: keep them last
                ready = sorted(pending)
            for ref in ready:
                targets = pending.pop(ref)
                stages.append(Stage(ref=ref, targets=tuple(targets),
                                    depth=depth.get(ref, 0)))
                for k in targets:
                    depth.setdefault(k, depth.get(ref, 0) + 1)
        return stages

    def _tpa_phase_finished(self, payload: dict[str, Any]) -> None:
        self.tpa_phase_stop_event = None
        aborted = payload.get("status") == "aborted"
        results = payload.get("results") or {}
        if not results:
            self._tpa_phase_set_running(False)
            self.tpa_phase_status.setText(
                "Sweep stopped before any target finished."
            )
            self._tpa_phase_close_dialog(
                False, "Stopped before any target finished."
            )
            return
        self.tpa_phase_meta = payload.get("meta") or {}
        cfg = payload["cfg"]
        # A run stopped part-way may have finished stage 1 but not the stage
        # hanging off it, so plan only over what actually came back -- otherwise
        # chain_fits waits on a hub target that was never measured.
        stages = [replace(s, targets=tuple(k for k in s.targets if k in results))
                  for s in payload.get("stages") or []]
        failed = self._tpa_phase_fit_results(
            results, payload["models"], cfg.ref_index,
            [s for s in stages if s.targets] or None,
        )
        self._tpa_phase_set_running(False)
        self._tpa_phase_fill_table()
        self._tpa_phase_redraw()

        parts = [
            f"ΔΦ[{k}]={np.degrees(c.dphi):+.2f}°"
            for k, c in sorted(self.tpa_phase_chains.items())
        ]
        head = "Stopped" if aborted else "Done"
        n_chained = sum(1 for c in self.tpa_phase_chains.values() if c.chained)
        note = f"  · {len(failed)} fit(s) failed" if failed else ""
        if n_chained:
            note += f"  · {n_chained} chained to ref {cfg.ref_index}"
        self.tpa_phase_status.setText(
            f"{head} · " + ("; ".join(parts) or "no fit") + note
        )
        for line in failed:
            self._log(f"[step 7] fit failed — {line}")
        self._tpa_phase_autosave()
        self._tpa_phase_close_dialog(not aborted, self.tpa_phase_status.text())

    def _tpa_phase_close_dialog(self, success: bool, message: str) -> None:
        """Freeze the progress window and let it be closed (see step 6)."""
        if self.calibration_dialog is not None:
            self.calibration_dialog.finish(success, message)

    def _tpa_phase_error(self, _error: str) -> None:
        self.tpa_phase_stop_event = None
        self._tpa_phase_set_running(False)
        self.tpa_phase_status.setText(
            "Comb-phase sweep failed (see Status log)"
        )
        self._tpa_phase_close_dialog(
            False, "Comb-phase sweep failed (see Status log)"
        )

    # ======================================================================
    # Step 8 -- the saturation constant
    # ======================================================================
    #
    # Deliberately the thinnest calibration page here.  Steps 6 and 7 fit a
    # model per pair and need a table, a per-target plot and a cascade to be
    # read at all; step 8 fits ONE number from ONE sweep, so anything more than
    # "which step 7, how many rungs, Run" would be furniture.  The run itself is
    # driven through the same measure/patterns loop the CLI uses, against the
    # DAQ and SLM this window already has open -- calib_step8.collect() opens
    # its own and must not be called from here.

    def _build_step8_tab(self) -> QtWidgets.QWidget:
        # No prose on this page.  The only text it shows is measured or
        # predicted numbers; what the step is for lives in this module's
        # comments and in calib_step8's docstring, where it does not take up
        # half the panel every time the tab is opened.
        page = self._page_shell("Two-Photon Saturation (Y_sat)")
        self.step8_status = self._page_status(page, "Idle")
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.addWidget(self._build_step8_controls())
        split.addWidget(self._build_step8_results())
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([360, 1040])
        page.layout().addWidget(split, 1)
        return page

    def _build_step8_controls(self) -> QtWidgets.QWidget:
        col = QtWidgets.QWidget()
        box = QtWidgets.QVBoxLayout(col)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(12)

        # The forward model this sweep drives: step 3's layout, step 6's etas
        # and step 7's phases, all out of the one file.
        box.addWidget(self._calib_file_row(8))
        self.step8_step7_edit = self._calib_edits[8]

        # The description of this file -- its chain, its pairs, its Y_max --
        # goes to the page's status line beside the title (_STEP_STATUS), so
        # the column is controls and nothing else.

        probe = self._panel("Probe one level")
        grid = QtWidgets.QGridLayout(probe)
        self.step8_probe_alpha = self._double_spin(0.0, 1.0, 0.3, "", 4)
        self.step8_probe_alpha.setSingleStep(0.01)
        self.step8_probe_alpha.setToolTip(
            f"Fraction of Y_max to display. Defaults to Alpha max, the "
            f"brightest rung the sweep will take.\n"
            f"This is how the bench finds out where the READOUT stops being "
            f"linear, which is a question about the electronics and not about "
            f"the atoms: the amplifier rails at "
            f"{1e3 * STEP8.AMP_LIMIT_V:.0f} mV and begins to compress "
            f"somewhere below that, and either bends the top of a band down "
            f"exactly the way saturation does \N{EM DASH} the fit has one free "
            f"parameter and cannot tell them apart.\n"
            f"Walk alpha up and watch the read fall behind the linear "
            f"prediction, and the raw peak approach the rail. That "
            f"measurement is what LADDER_CEILING_V "
            f"({1e3 * STEP8.LADDER_CEILING_V:.0f} mV) encodes.")
        self.step8_probe_alpha.valueChanged.connect(self._step8_probe_predict)
        self.step8_probe_pred = self._caption("")
        self.step8_probe_button = QtWidgets.QPushButton("Display && Read")
        self.step8_probe_button.setEnabled(False)
        self.step8_probe_button.setToolTip(
            "One all-off dark, the frame, one reading. Nothing is written. It "
            "goes through the same run_patterns loop and the same instruments "
            "the sweep uses, so what it reports is literally a sweep read.")
        self.step8_probe_button.clicked.connect(self._step8_probe)
        grid.addWidget(QtWidgets.QLabel("alpha"), 0, 0)
        grid.addWidget(self.step8_probe_alpha, 0, 1)
        grid.addWidget(self.step8_probe_pred, 0, 2)
        # Display alone needs the SLM and not the DAQ, which is the point of
        # it: a known level on the panel while something that is not this
        # detector is looked at or adjusted.  It leaves the frame up.
        self.step8_display_button = QtWidgets.QPushButton("Display")
        self.step8_display_button.setProperty("variant", "ghost")
        self.step8_display_button.setEnabled(False)
        self.step8_display_button.setToolTip(
            "Solve the drive for this alpha and put it on the panel. No "
            "read, no DAQ, nothing written \N{EM DASH} the frame stays up until "
            "something else displays.\n"
            "The solver is the sweep's own, so what goes up is a rung the "
            "sweep could have chosen.")
        self.step8_display_button.clicked.connect(self._step8_display)
        grid.addWidget(self.step8_display_button, 1, 0)
        grid.addWidget(self.step8_probe_button, 1, 1, 1, 2)
        grid.setColumnStretch(2, 1)
        box.addWidget(probe)

        # --- DAQ acquisition --------------------------------------------
        # Its own group, and its own settings -- not STEP8.ACQ.  Steps 6 and 7
        # put the window, the range and the sign on the page; this step read
        # them off a module constant, which made the one page whose fit is a
        # saturation curve the one page that could not say what it read at.
        # Same four controls and the same BENCH_ACQ defaults, so an untouched
        # page still runs exactly what the CLI runs.
        daq = self._panel("DAQ \N{MIDDLE DOT} acquisition")
        dgrid = QtWidgets.QGridLayout(daq)
        dgrid.setHorizontalSpacing(8)
        dgrid.setVerticalSpacing(6)

        self.step8_daq_channel = QtWidgets.QLineEdit("ai0")
        self.step8_daq_range = QtWidgets.QComboBox()
        for lo, hi in self._DAQ_RANGES:
            self.step8_daq_range.addItem(
                f"\N{PLUS-MINUS SIGN}{hi:g} V", (lo, hi))
        self._select_daq_range(self.step8_daq_range, BENCH_ACQ.range_v)
        self.step8_daq_range.setToolTip(BENCH_RANGE_TIP)
        self.step8_tmeas = self._double_spin(
            0.001, 60.0, BENCH_ACQ.t_meas_s, " s", 3)
        self.step8_tmeas.setToolTip(BENCH_WINDOW_TIP)
        self.step8_invert = QtWidgets.QCheckBox("Invert sign (TIA)")
        self.step8_invert.setChecked(BENCH_ACQ.invert)
        self.step8_invert.setToolTip(
            "The transimpedance amplifier outputs NEGATIVE volts for positive "
            "light, so recording a positive light signal means inverting.\n"
            "Leave this on. A sign-flipped run does not fail here \N{EM DASH} "
            "it fits a saturation curve to reads that bend the wrong way and "
            "reports a Y_sat for it."
        )
        for r, (label, widget) in enumerate((
                ("Channel", self.step8_daq_channel),
                ("Range", self.step8_daq_range),
                ("T_meas", self.step8_tmeas))):
            dgrid.addWidget(QtWidgets.QLabel(label), r, 0)
            dgrid.addWidget(widget, r, 1)
        dgrid.addWidget(self.step8_invert, 3, 0, 1, 2)
        dgrid.setColumnStretch(1, 1)

        # Bench constants, off the panel the way steps 6 and 7 keep theirs:
        # live widgets the run still reads, just not ones anybody sets.
        self.step8_daq_rate = QtWidgets.QDoubleSpinBox()
        self.step8_daq_rate.setRange(1.0, 2_000_000.0)
        self.step8_daq_rate.setDecimals(0)
        self.step8_daq_rate.setValue(1000.0)
        self.step8_daq_rate.setSuffix(" S/s")
        self.step8_daq_fcut = self._double_spin(0.1, 100_000.0, 20.0, " Hz", 1)
        self.step8_pmt = QtWidgets.QCheckBox(
            "Log PMT thermistor (Keithley 2400)")
        for widget in (self.step8_daq_rate, self.step8_daq_fcut,
                       self.step8_pmt):
            widget.setParent(daq)
            widget.hide()
        box.addWidget(daq)

        # The estimate is quoted under these, so it has to move with them.
        self.step8_tmeas.valueChanged.connect(self._step8_update_estimate)
        self.step8_daq_range.currentIndexChanged.connect(
            self._step8_update_estimate)

        sweep = self._panel("Sweep")
        grid = QtWidgets.QGridLayout(sweep)
        self.step8_points = self._spin(4, 400, STEP8.LADDER)
        self.step8_points.setToolTip(
            "Rungs in the band, one per stratum, each a fresh random drive.\n"
            "A rung's information about 1/Y_sat goes as "
            "(Y²/σ)², so the top of the band is worth "
            "orders more than the bottom — 30 over the bright end "
            "return Y_sat to ~4 %, against the ~11 % it would have to be "
            "wrong by before it cost as much as the model error beside it."
        )
        self.step8_points.valueChanged.connect(self._step8_update_estimate)
        self.step8_points.valueChanged.connect(self._step8_refresh_levels)
        grid.addWidget(QtWidgets.QLabel("Rungs"), 0, 0)
        grid.addWidget(self.step8_points, 0, 1)

        # The band in ALPHA, which is what the sweep is specified in and what
        # every output records.  The MILLIVOLTS beside each box are the
        # DERIVED column: alpha x Y_max, so they move with the PMT gain --
        # and they are what the amplifier argument is actually about, which
        # is why they are on screen rather than left to be worked out.  Blank
        # until a calib file names a Y_max, because before that there are no
        # millivolts to show.
        self.step8_alpha_hi = self._double_spin(0.0001, 1.0, 0.2945, "", 4)
        self.step8_alpha_hi.setSingleStep(0.01)
        self.step8_alpha_hi.setToolTip(
            f"The brightest rung, as a fraction of Y_max — and the only "
            f"thing that decides where the sweep stops.\n"
            f"The default is {1e3 * STEP8.LADDER_CEILING_V:.0f} mV converted "
            f"against this file's Y_max. That figure is UNVERIFIED: nobody "
            f"has measured where the amplifier starts to compress, only its "
            f"{STEP8.AMP_LIMIT_V:g} V hard rail. Compression bends the top "
            f"of the band down exactly the way saturation does, and the fit "
            f"has one free parameter and cannot tell them apart — so too "
            f"high is SILENT, while too low only costs precision. Use the "
            f"probe below to replace the guess."
        )
        self.step8_alpha_lo = self._double_spin(0.0001, 1.0, STEP8.ALPHA_LO,
                                                "", 4)
        self.step8_alpha_lo.setSingleStep(0.01)
        self.step8_alpha_lo.setToolTip(
            "The dimmest rung, as a fraction of Y_max.\n"
            "The bottom of the band weighs almost nothing in the fit: a "
            "rung's leverage on Y_sat falls off as the square of its level, "
            "so set this by what you want COVERED rather than by what you "
            "want measured."
        )
        self.step8_alpha_hi_mv = self._caption("")
        self.step8_alpha_lo_mv = self._caption("")
        rows = (("Alpha max", self.step8_alpha_hi, self.step8_alpha_hi_mv),
                ("Alpha min", self.step8_alpha_lo, self.step8_alpha_lo_mv))
        for r, (label, spin, mv) in enumerate(rows, start=1):
            spin.valueChanged.connect(self._step8_refresh_levels)
            grid.addWidget(QtWidgets.QLabel(label), r, 0)
            grid.addWidget(spin, r, 1)
            grid.addWidget(mv, r, 2, 1, 2)
        self.step8_alpha_hi.valueChanged.connect(self._step8_probe_predict)
        grid.setColumnStretch(2, 1)

        box.addWidget(sweep)

        run = self._panel("Run")
        grid = QtWidgets.QGridLayout(run)
        self.step8_run_button = QtWidgets.QPushButton("Run Sweep")
        self.step8_run_button.clicked.connect(self._step8_run)
        self.step8_stop_button = QtWidgets.QPushButton("Stop")
        self.step8_stop_button.setEnabled(False)
        self.step8_stop_button.clicked.connect(self._step8_stop)
        # Load, the same one step 6 and 7 carry: re-fit the sweep this calib
        # file already records and show it.  There is no Save -- a finished
        # sweep writes itself beside the file it was driven from, because that
        # file is what was named before the bench was touched.
        self.step8_load_button = QtWidgets.QPushButton("Load")
        self.step8_load_button.setProperty("variant", "ghost")
        self.step8_load_button.setToolTip(
            "Re-fit the step-8 sweep this calibration file records and show it "
            "\N{EM DASH} no hardware, no sweep.\n"
            "Reads only: the fit on screen is not written back, and the plot it "
            "draws goes to a temp directory.")
        self.step8_load_button.clicked.connect(self._step8_load)
        grid.addWidget(self.step8_run_button, 0, 0)
        grid.addWidget(self.step8_stop_button, 0, 1)
        grid.addWidget(self.step8_load_button, 1, 0, 1, 2)
        # The estimate stays: it is a prediction about the sweep you are about
        # to start, not feedback about one that ran.  step8_status is beside
        # the page title.
        self.step8_estimate = self._caption("")
        grid.addWidget(self.step8_estimate, 2, 0, 1, 2)
        box.addWidget(run)

        box.addStretch(1)
        self._step8_update_estimate()
        self._step8_refresh_levels()
        self._step8_refresh_buttons()
        return col

    def _build_step8_results(self) -> QtWidgets.QWidget:
        col = QtWidgets.QWidget()
        box = QtWidgets.QVBoxLayout(col)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(12)

        self.step8_result_text = QtWidgets.QPlainTextEdit()
        self.step8_result_text.setReadOnly(True)
        self.step8_result_text.setMaximumHeight(190)
        box.addWidget(self._panel_with_widget("Fit", self.step8_result_text))

        self.step8_plot_label = QtWidgets.QLabel("")
        self.step8_plot_label.setAlignment(QtCore.Qt.AlignCenter)
        self.step8_plot_label.setMinimumHeight(360)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.step8_plot_label)
        box.addWidget(self._panel_with_widget("Report", scroll), 1)
        return col

    # ---- plan and estimate -------------------------------------------------
    def _step8_step7_path(self) -> Path:
        path = self._calib_file_path(8)
        if path is None:
            raise ValueError("Pick a calibration file first.")
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist")
        return path

    def _step8_acq(self) -> PairV2Acq:
        """Window, range and sign, straight off this page's DAQ group.

        The settle is not on the page: ``run_patterns`` owns it for step 8, so
        a control for it would set nothing.
        """
        _lo, hi = self.step8_daq_range.currentData()
        return PairV2Acq(
            t_meas_s=float(self.step8_tmeas.value()),
            settle_s=BENCH_ACQ.settle_s,
            range_v=float(hi),
            invert=self.step8_invert.isChecked(),
        )

    def _step8_update_estimate(self) -> None:
        """Reads and minutes, before the bench is asked for anything.

        Only the count and the windows, not the solved plan: solving the ladder
        takes seconds and this runs on every spinbox tick.
        """
        n = self.step8_points.value()
        acq = self._step8_acq()
        self.step8_estimate.setText(
            f"{n} reads, roughly {STEP8.run_seconds(n, acq) / 60:.0f} min "
            f"({acq.t_meas_s:g} s every read at "
            f"\N{PLUS-MINUS SIGN}{acq.range_v:g} V, dark included)")

    def _step8_step7_changed(self) -> None:
        """Describe this page's file, show where its Y_max lands, gate the probe.

        This page's entry in :attr:`_STEP_DESCRIBE`, so it owns the one line the
        status line carries about the file -- the chain, the pairs and the Y_max
        the band is placed on, which is the only number this page needs from the
        file before it runs.

        Runs on every keystroke, so it does nothing until the text names a
        calib.json that is there (a directory means the file inside it) -- which
        for a browsed path is once.  ``full_scale`` predicts and touches no
        hardware.
        """
        path = self._calib_file_path(8)
        if path is None or not path.is_file():
            self._set_status(
                self.step8_status,
                "No calibration file chosen \N{EM DASH} required to run."
                if path is None else
                f"{path.name} does not exist yet ({path.parent})",
                "off")
            self.step8_y_max = None
            self._step8_plan_for = None
            # Dropping Y_max drops the alphas with it, and a stale alpha
            # beside a live box is the one thing this column must never show.
            self._step8_refresh_levels()
            self._step8_probe_predict()
            self._step8_refresh_buttons()
            return
        # The cache is only a guard against re-parsing the JSON on every
        # keystroke of a path being typed; it must not survive Y_max being
        # dropped, or clearing the box and retyping the same file would leave the
        # panel blank and the button dead.
        if (getattr(self, "_step8_plan_for", None) == str(path.resolve())
                and getattr(self, "step8_y_max", None) is not None):
            return
        try:
            models, phases, _ref = verify_load_forward_model(path)
            fs = float(STEP8.full_scale(models, phases, sorted(models)).fs)
        except Exception as exc:                      # noqa: BLE001
            self.step8_y_max = None
            self._step8_plan_for = None
            # Dropping Y_max drops the alphas with it, and a stale alpha beside a
            # live box is the one thing this column must never show.
            self._step8_refresh_levels()
            self._set_status(self.step8_status,
                             f"Cannot read the forward model: {exc}", "error")
            self._step8_probe_predict()
            self._step8_refresh_buttons()
            return
        self.step8_y_max = fs
        self._step8_plan_for = str(path.resolve())
        self._set_status(
            self.step8_status,
            f"{self._calib_name(path)} \N{MIDDLE DOT} {self._run_chain(path)} "
            f"\N{MIDDLE DOT} {len(models)} pairs "
            f"\N{MIDDLE DOT} Y_max {fs * 1e3:.1f} mV "
            f"\N{MIDDLE DOT} 4-bit step {fs / 15 * 1e3:.3f} mV",
            "ok")
        self._step8_refresh_levels()
        # A fresh forward model means a fresh cap, and the cap is the alpha worth
        # probing by default -- it is the brightest read the sweep will take.
        self.step8_probe_alpha.setValue(
            round(float(self.step8_alpha_hi.value()), 4))
        self._step8_probe_predict()
        self._step8_refresh_buttons()

    def _step8_alphas(self) -> dict[str, float]:
        """The band, as :func:`calibration_module.steps.calib_step8.build` keywords.

        One place reads the two boxes and spells the keyword names, so a box
        cannot end up wired to the wrong end of the band.
        """
        return {"alpha_lo": float(self.step8_alpha_lo.value()),
                "alpha_hi": float(self.step8_alpha_hi.value())}

    def _step8_refresh_levels(self) -> None:
        """The millivolts beside each alpha box, and the Y_max note above them.

        Pure arithmetic on a cached ``Y_max`` -- no solve, no file read -- so it
        runs on every spinbox tick.  BLANK until a calib file is loaded: the
        millivolts are alpha x Y_max, and there is no Y_max before that.

        The top row also says what the raw trace PEAK will be, because that is
        what the amplifier limit is about: the 20 Hz low-pass averages a flat
        top away, so a railed read still reports a civil mean.  It warns and
        does not refuse -- what guards this is the recorded ``peak_v``.
        """
        if not hasattr(self, "step8_alpha_lo_mv"):
            return
        fs = getattr(self, "step8_y_max", None)
        a = self._step8_alphas()
        if fs is None:
            for cap in (self.step8_alpha_hi_mv, self.step8_alpha_lo_mv):
                cap.clear()
            return
        peak = a["alpha_hi"] * fs * STEP8.PEAK_OVER_MEAN
        note = ""
        if peak > 0.85 * STEP8.AMP_LIMIT_V:
            note = (f"   \N{MIDDLE DOT}  raw peak ~{peak * 1e3:.0f} mV, "
                    f"{100 * peak / STEP8.AMP_LIMIT_V:.0f} % of the "
                    f"{STEP8.AMP_LIMIT_V:g} V limit")
        self.step8_alpha_hi_mv.setText(
            f"= {a['alpha_hi'] * fs * 1e3:.1f} mV" + note)
        band = (f"   \N{MIDDLE DOT}  {self.step8_points.value()} rungs up to alpha max"
                if a["alpha_lo"] < a["alpha_hi"] else
                "   \N{MIDDLE DOT}  above alpha max: the band is empty")
        self.step8_alpha_lo_mv.setText(
            f"= {a['alpha_lo'] * fs * 1e3:.1f} mV" + band)

    def _step8_probe_predict(self) -> None:
        """The predicted millivolts for the alpha in the box, before committing.

        Pure arithmetic -- ``alpha`` IS the fraction of ``Y_max`` -- so this can
        run on every spinbox tick without solving anything.  It exists so the
        number is visible BEFORE the frame goes up: at the present gain
        ``alpha = 1`` is ~1534 mV into a 1 V amplifier, and that should not be
        one keystroke away without warning.
        """
        # Both, because the probe panel is built ABOVE the sweep panel: between
        # the two, step8_probe_pred exists and step8_alpha_hi does not.
        if not hasattr(self, "step8_alpha_hi"):
            return
        fs = getattr(self, "step8_y_max", None)
        if fs is None:
            self.step8_probe_pred.setText("")
            return
        a = float(self.step8_probe_alpha.value())
        v = a * float(fs)
        top = self._step8_alphas()["alpha_hi"]
        peak = v * STEP8.PEAK_OVER_MEAN
        note = ""
        # Said, never refused.  The peak is what reaches the rail, and the
        # mean cannot show it: the low-pass averages a flat top away.
        if peak >= STEP8.AMP_LIMIT_V:
            note = (f"   \N{EM DASH}  raw peak ~{peak * 1e3:.0f} mV is AT the "
                    f"{STEP8.AMP_LIMIT_V:g} V limit: this read will be a "
                    f"flat-top, and only peak_v will show it")
        elif peak > 0.85 * STEP8.AMP_LIMIT_V:
            note = (f"   \N{EM DASH}  raw peak ~{peak * 1e3:.0f} mV, close to "
                    f"the {STEP8.AMP_LIMIT_V:g} V limit")
        elif a > top:
            note = (f"   \N{EM DASH}  above alpha max {top:.4f}, brighter than "
                    f"the sweep goes")
        self.step8_probe_pred.setText(f"predicts {v * 1e3:.1f} mV{note}")

    def _step8_refresh_buttons(self) -> None:
        """Everything whose live-ness depends on the run, or on another box."""
        if not hasattr(self, "step8_probe_button"):
            return
        running = self.step8_stop_button.isEnabled()
        ready = getattr(self, "step8_y_max", None) is not None
        self.step8_probe_button.setEnabled(not running and ready)
        self.step8_display_button.setEnabled(not running and ready)
        self.step8_probe_alpha.setEnabled(not running)
        for w in (self.step8_points, self.step8_alpha_lo,
                  self.step8_alpha_hi):
            w.setEnabled(not running)

    def _step8_set_running(self, running: bool) -> None:
        """See :meth:`_tpa_set_running`."""
        self._step8_set_running_impl(running)
        if not running:
            self._calib_gate_step(8)

    def _step8_set_running_impl(self, running: bool) -> None:
        self.step8_run_button.setEnabled(not running)
        self.step8_load_button.setEnabled(not running)
        self.step8_stop_button.setEnabled(running)
        self._step8_refresh_buttons()

    def _step8_stop(self) -> None:
        if getattr(self, "step8_stop_event", None) is not None:
            self.step8_stop_event.set()
            self.step8_status.setText("Stopping after the current read\N{HORIZONTAL ELLIPSIS}")

    # ---- the run -----------------------------------------------------------
    def _step8_run(self) -> None:
        alphas = self._step8_alphas()
        # No dialog.  The caption under Alpha max already says what the raw
        # peak will be against the amplifier limit, and it says it while the
        # box is being set rather than after Run is pressed.  A sweep that
        # tops out near the rail is a judgement call about the electronics,
        # and the recorded peak_v is what settles it afterwards.
        try:
            step7 = self._step8_step7_path()
            models, phases, _ref = verify_load_forward_model(step7)
            pairs = sorted(models)
            seed = int(np.random.SeedSequence().entropy % 2**32)
            scale, patterns, preds = STEP8.build(
                models, phases, pairs, seed=seed,
                n=self.step8_points.value(), **alphas)
            layout = STEP8.load_layout(step7, pairs)
        except (ValueError, FileNotFoundError) as exc:
            self.step8_status.setText(str(exc))
            return
        except Exception as exc:                      # noqa: BLE001
            self.step8_status.setText(f"Step 7: {exc}")
            return

        daq = self.daq_controller
        if daq is None or not daq.is_connected:
            self.step8_status.setText("Connect the DAQ first (DAQ page).")
            return
        controller = self._controller()
        if not getattr(controller, "is_open", False):
            self.step8_status.setText("Open the SLM on the SLM Control page first.")
            return

        acq = self._step8_acq()
        secs = STEP8.run_seconds(len(patterns), acq)
        settings = DAQMonitorSettings(
            channel=self.step8_daq_channel.text().strip() or "ai0",
            sample_rate=self.step8_daq_rate.value(),
            duration=acq.t_meas_s, single_duration=acq.t_meas_s,
            hold=0.0,                       # run_patterns owns the settle
            min_val=-acq.range_v, max_val=acq.range_v,
            f_cut=self.step8_daq_fcut.value(),
        )

        # Beside the calib file this sweep was driven from: its step 3/6/7
        # sections are the forward model, and the Y_sat measures THAT model.
        # There is no staging and no Save -- the destination was decided when
        # the file was named, before the bench was touched.
        out = Path(step7).parent
        out.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%m%d_%H%M")
        coh = [float(verify_predict(models, phases, pairs, p.x, p.w, coherent=True))
               for p in patterns]
        x = np.stack([p.x for p in patterns])[None]
        w = np.stack([p.w for p in patterns])[None]
        extra = [[{"kind": p.kind, "label": p.label, "seq": i,
                   "target_alpha": f"{p.target_alpha:.6f}",
                   "alpha": f"{p.alpha:.6f}",
                   "coherent_v": f"{c:.9g}", "sb_pred_v": f"{q - c:.9g}",
                   "t_meas_s": f"{acq.t_meas_s:.3f}"}
                  for i, (p, c, q) in enumerate(zip(patterns, coh, preds))]]

        self.step8_status.setText(
            f"Running\N{HORIZONTAL ELLIPSIS} {len(patterns)} reads, "
            f"~{secs / 60:.0f} min, Y_max {scale.fs * 1e3:.1f} mV")
        self._step8_set_running(True)
        self._open_calibration_dialog(on_stop=self._step8_stop)
        if self.calibration_dialog is not None:
            self.calibration_dialog.note([
                f"{len(patterns)} reads \N{MIDDLE DOT} {acq.t_meas_s:g} s at "
                f"\N{PLUS-MINUS SIGN}{acq.range_v:g} V",
                # The band is tunable, so the dialog has to say which one ran,
                # in both spellings: alpha is what was set, mV is what the
                # amplifier sees.
                f"band alpha {alphas['alpha_lo']:.4f}\N{EN DASH}"
                f"{alphas['alpha_hi']:.4f} = {min(preds) * 1e3:.1f}\N{EN DASH}"
                f"{max(preds) * 1e3:.1f} mV \N{MIDDLE DOT} raw peak at the top "
                f"~{max(preds) * STEP8.PEAK_OVER_MEAN * 1e3:.0f} mV of the "
                f"{STEP8.AMP_LIMIT_V:g} V limit",
                f"Y_max {scale.fs * 1e3:.3f} mV \N{MIDDLE DOT} 4-bit step "
                f"{scale.fs / 15 * 1e3:.3f} mV",
                "Predictions are LINEAR: this run fits Y_sat, so it assumes none.",
            ])

        stop_event = threading.Event()
        self.step8_stop_event = stop_event

        def report(progress: PatternsProgress) -> None:
            self.step8_progress.emit(progress)

        pmt = open_pmt(enabled=self.step8_pmt.isChecked(),
                       log=lambda m: self._log(f"[step 8] {m}"))

        def work() -> dict[str, Any]:
            daq.configure_monitor(settings)
            rows: list[dict] = []
            meta = {"n_pairs": len(pairs), "y_max_v": scale.fs,
                    "n": self.step8_points.value(),
                    # Alpha is the band; the mV are what it meant at THIS
                    # Y_max.  Both, because neither alone survives a gain
                    # change -- same reason collect() writes both.
                    "alpha_lo": f"{alphas['alpha_lo']:.6g}",
                    "alpha_hi": f"{alphas['alpha_hi']:.6g}",
                    "alpha_lo_mv": f"{alphas['alpha_lo'] * scale.fs * 1e3:.6g}",
                    "alpha_hi_mv": f"{alphas['alpha_hi'] * scale.fs * 1e3:.6g}"}
            columns: list[str] = []
            status = "complete"
            if pmt is not None:
                columns = list(pmt.COLUMNS)
                meta.update({f"{k}_start": v for k, v in pmt.read().items()})
            try:
                run_patterns(daq, controller, layout, [tuple(pairs)], x, w,
                             models, phases, rows=rows, acq=acq, extra=extra,
                             y_sat=None, row_hook=pmt,
                             stop_event=stop_event, progress_callback=report,
                             log=lambda m: self._log(f"[step 8] {m}"))
            except PatternsAborted as exc:
                status = f"partial ({exc})"
            finally:
                if pmt is not None:
                    meta.update({f"{k}_end": v for k, v in pmt.read().items()})
            if not rows:
                return {"status": status, "csv": None}
            csv_path = verify_write_csv(
                out / f"calib_step8_{stamp}.csv", pairs, rows,
                meta={"step7_json": Path(step7).resolve(), "seed": seed,
                      "status": status, **meta},
                extra=["kind", "label", "seq", "target_alpha", "alpha",
                       "coherent_v", "sb_pred_v", "t_meas_s", "peak_v",
                       *columns])
            result = STEP8.fit_and_write(
                csv_path, step7,
                log=lambda m: self._log(f"[step 8] {m}"))
            # The calib file travels with the fit: editing the box between a
            # run and its report must not change what the report says it was
            # measured against.
            return {"status": status, "csv": csv_path, "written": True,
                    "step7": Path(step7), "result": result}

        self._run_slm_task("Step 8 saturation sweep", work,
                           self._step8_finished, self._step8_error)

    def _on_step8_progress(self, progress: PatternsProgress) -> None:
        """Feed one ladder read to the shared progress dialog.

        Adapted rather than reported natively, exactly as steps 6 and 7 do it.
        Unlike them this fires TWICE per read -- once when the read starts, once
        when it lands -- because a step-8 read is a five-to-ten second average
        behind a ten second opening dark, and a window that only moved on
        finished reads spent that first quarter minute indistinguishable from a
        hang.  The darks move the message but not the bar: they are not part of
        ``total``, and ``step`` already counts only finished pattern reads.

        The plot is measured against PREDICTED, not against read number.  The
        ladder is driven shuffled on purpose (see ``calib_step8.shuffle``),
        so read order carries no physics, while this is the saturation curve the
        run exists to fit -- it leaves the diagonal exactly where Y_sat bites.
        """
        n, total = progress.step, progress.total
        if progress.single:
            message = (f"dark (all off, {progress.t_read_s:g} s)…"
                       if progress.mean_v is None else
                       f"dark (all off) → {progress.mean_v * 1e3:.4f} mV")
        elif progress.mean_v is None:
            message = (f"read {n + 1}/{total} · "
                       f"{progress.t_read_s:.1f} s window · "
                       f"pred {progress.pred_v * 1e3:.2f} mV"
                       f"…")
        else:
            net = progress.net_v
            # Every read picks its own range from its prediction, so "wide"
            # can only mean the near-rail escalation fired on top of that.
            wide = ("  [wide range]"
                    if progress.range_v != range_for(progress.pred_v)[0] else "")
            message = (f"[{n}/{total}] {net * 1e3:.4f} mV  "
                       f"pred {progress.pred_v * 1e3:.4f}  "
                       f"diff {(net - progress.pred_v) * 1e3:+.4f}  "
                       f"std {progress.std_ratio * 100:.2f}%{wide}")
        self.step8_status.setText(message)
        if self.calibration_dialog is None:
            return
        plot = progress.mean_v is not None and not progress.single
        self.calibration_dialog.update_progress(CalibrationProgress(
            phase="ladder",
            step=n - 1,                   # the dialog adds 1 back
            total=total,
            message=message,
            x=progress.pred_v * 1e3 if plot else None,
            y=progress.net_v * 1e3 if plot else None,
        ))

    def _step8_display(self) -> None:
        """Put one level on the panel and leave it there.  No read, no DAQ.

        The other half of the probe: same solver, same drive, the reading left
        off.  It needs the SLM only, so it works before the DAQ is up -- which
        is most of what it is for, along with holding a known level while
        something that is not this detector is looked at.

        Nothing is written, and nothing clears the frame afterwards: the next
        display wins, as it would on the SLM page.
        """
        alpha = float(self.step8_probe_alpha.value())
        if alpha <= 0.0:
            self.step8_status.setText(
                "alpha has to be above 0 to display anything.")
            return
        try:
            calib = self._step8_step7_path()
            models, phases, _ref = verify_load_forward_model(calib)
            pairs = sorted(models)
            layout = STEP8.load_layout(calib, pairs)
        except (ValueError, FileNotFoundError) as exc:
            self.step8_status.setText(str(exc))
            return
        except Exception as exc:                      # noqa: BLE001
            self.step8_status.setText(f"Forward model: {exc}")
            return
        controller = self._controller()
        if not getattr(controller, "is_open", False):
            self.step8_status.setText("Open the SLM on the SLM Control page first.")
            return

        self._step8_set_running(True)
        self.step8_status.setText(f"Displaying alpha = {alpha:.4f}\N{HORIZONTAL ELLIPSIS}")

        def work() -> dict[str, Any]:
            pattern = STEP8.display_alpha(
                controller, layout, models, phases, pairs, alpha,
                log=lambda m: self._log(f"[step 8] {m}"))
            pred = float(verify_predict(models, phases, pairs,
                                        pattern.x, pattern.w))
            return {"alpha": pattern.alpha, "pred_v": pred,
                    "exact": pattern.exact}

        self._run_slm_task("Step 8 display", work,
                           self._step8_display_finished,
                           self._step8_display_error)

    def _step8_display_finished(self, info: dict[str, Any]) -> None:
        self._step8_set_running(False)
        self.step8_status.setText(
            f"Displaying alpha = {info['alpha']:.4f}, predicts "
            f"{info['pred_v'] * 1e3:.1f} mV (LINEAR). Nothing was read or "
            f"written."
            + ("" if info["exact"] else               "  The projection fell back to the deterministic drive."))

    def _step8_display_error(self, _error: str) -> None:
        self._step8_set_running(False)
        self.step8_status.setText("Display failed (see Status log)")

    def _step8_probe(self) -> None:
        """Display one chosen level, read it once, report it.  Nothing written.

        It goes through the same ``run_patterns`` loop and the same instruments
        the sweep uses -- the same all-off dark, the same fixed range, the same
        inversion -- so what comes back is literally a sweep read.

        The alpha is the caller's, and nothing here refuses one.  The predicted
        millivolts and the raw peak they imply are on screen beside the box
        before the press; past the amplifier limit the read is a flat-top and
        ``peak_v`` is what says so.  This replaced a button that could only
        drive ``alpha = 1``, which at the present gain is ~1534 mV into a 1 V
        amplifier: it clipped by construction and then inverted the clipped
        read to report a confident ceiling.
        """
        alpha = float(self.step8_probe_alpha.value())
        if alpha <= 0.0:
            self.step8_status.setText(
                "alpha has to be above 0 \N{EM DASH} a zero drive is the dark, "
                "which every read already takes first.")
            return
        try:
            step7 = self._step8_step7_path()
            models, phases, _ref = verify_load_forward_model(step7)
            pairs = sorted(models)
            scale = STEP8.full_scale(models, phases, pairs)
            # Same solver the ladder uses, so a probe at the cap IS a ceiling
            # rung: closed form at alpha = 1, a least-move projection below it.
            # Same solver the sweep uses, so what goes on the panel is a rung
            # the sweep could have chosen: closed form at alpha = 1, a
            # least-move projection below it.
            pattern = STEP8.pattern_for_alpha(
                models, phases, pairs, alpha,
                seed=int(np.random.SeedSequence().entropy % 2**32),
                label="probe")
            layout = STEP8.load_layout(step7, pairs)
        except (ValueError, FileNotFoundError) as exc:
            self.step8_status.setText(str(exc))
            return
        except Exception as exc:                      # noqa: BLE001
            self.step8_status.setText(f"Step 7: {exc}")
            return

        daq = self.daq_controller
        if daq is None or not daq.is_connected:
            self.step8_status.setText("Connect the DAQ first (DAQ page).")
            return
        controller = self._controller()
        if not getattr(controller, "is_open", False):
            self.step8_status.setText("Open the SLM on the SLM Control page first.")
            return

        acq = self._step8_acq()
        coh = float(verify_predict(models, phases, pairs, pattern.x, pattern.w,
                                   coherent=True))
        tot = float(verify_predict(models, phases, pairs, pattern.x, pattern.w))
        settings = DAQMonitorSettings(
            channel=self.step8_daq_channel.text().strip() or "ai0",
            sample_rate=self.step8_daq_rate.value(),
            duration=acq.t_meas_s, single_duration=acq.t_meas_s,
            hold=0.0,                       # run_patterns owns the settle
            min_val=-acq.range_v, max_val=acq.range_v,
            f_cut=self.step8_daq_fcut.value(),
        )
        x = pattern.x[None, None]
        w = pattern.w[None, None]

        self._step8_set_running(True)
        self.step8_status.setText(
            f"Probe\N{HORIZONTAL ELLIPSIS} one {acq.t_meas_s:g} s read at alpha "
            f"= {alpha:.4f} after a {acq.t_meas_s:g} s dark "
            f"(predicted {tot * 1e3:.1f} mV of Y_max {scale.fs * 1e3:.1f} mV)")

        def work() -> dict[str, Any]:
            daq.configure_monitor(settings)
            rows: list[dict] = []
            run_patterns(daq, controller, layout, [tuple(pairs)], x, w,
                         models, phases, rows=rows, acq=acq, y_sat=None,
                         log=lambda m: self._log(f"[probe] {m}"))
            if not rows:
                raise RuntimeError("the probe read produced no row")
            r = rows[0]
            return STEP8.probe_verdict(
                alpha=alpha, y_max_v=scale.fs,
                coherent_v=coh, sb_pred_v=tot - coh,
                mean_v=r["mean_v"], dark_v=r["dark_v"], std_v=r["std_v"],
                peak_v=r.get("peak_v", float("nan")), range_v=r["range_v"],)

        self._run_slm_task("Step 8 probe", work,
                           self._step8_probe_finished, self._step8_probe_error)

    def _step8_probe_finished(self, verdict: dict[str, Any]) -> None:
        self._step8_set_running(False)
        lines: list[str] = []
        STEP8.print_probe(verdict, log=lines.append)
        self.step8_result_text.setPlainText("\n".join(lines).strip())
        for line in lines:
            self._log(f"[probe] {line}")
        y = verdict["y_sat_implied_v"]
        self.step8_status.setText(
            f"{verdict['verdict']}: alpha {verdict['alpha']:.4f} read "
            f"{verdict['measured_v'] * 1e3:.1f} mV "
            f"({100 * verdict['shortfall']:+.2f} % vs linear) "
            f"\N{MIDDLE DOT} peak {verdict['peak_v'] * 1e3:.0f} mV = "
            f"{100 * verdict['peak_fraction']:.0f} % of "
            f"{verdict['limit_v']:g} V "
            + (f"\N{MIDDLE DOT} implied Y_sat {y:.2f} V"
               if np.isfinite(y) else
               "\N{MIDDLE DOT} implied Y_sat none"))

    def _step8_probe_error(self, _error: str) -> None:
        self._step8_set_running(False)
        self.step8_status.setText("Probe read failed (see Status log)")

    def _step8_render_dir(self) -> Path:
        """Where a Load draws its report PNG: a temp directory, outside the repo.

        The page displays an image from a path, so a fit it is only LOOKING at
        still has to put one somewhere -- and that somewhere must not be beside
        the calibration, because a Load writes nothing there.  Previously this
        held whole staged runs waiting for a Save; a sweep now goes straight to
        the file it was driven from, so all that is left here is a picture.
        """
        out = Path(tempfile.gettempdir()) / "rbonn_step8"
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _step8_load(self) -> None:
        """Re-fit the step-8 sweep this calib file records, and show it.

        ONE input, asked for, as on steps 6 and 7: a calib.json.  Its step8
        section says which CSV the Y_sat came from, and its step 3/6/7 sections
        are the forward model the fit needs -- all of it in the file, so there
        is nothing else to pick, and the page's own field does not move.

        Reads only.  ``report_csv`` rather than ``fit_and_write``: the section
        on disk stays whatever produced it, and the PNG goes to a temp
        directory.  A fit meant to be KEPT comes from a sweep, or from
        ``calib_step8.py`` against a copy of the file.
        """
        picked = self._open_calib("Open a calibration to re-fit its step 8", 8)
        if picked is None:
            return
        try:
            calib_file = picked
            calib = load_calib(calib_file)
            path = calib.csv_path(8)
        except Exception as exc:                        # noqa: BLE001
            self.step8_status.setText(f"Load failed: {exc}")
            return
        if not path.is_file():
            self.step8_status.setText(
                f"Load failed: {calib.path.name} names {path.name}, which is "
                f"not in {path.parent} — copy the CSV in beside it."
            )
            return
        out = self._step8_render_dir()
        self._step8_set_running(True)
        self.step8_status.setText(f"Fitting {path.name}\N{HORIZONTAL ELLIPSIS}")

        def work() -> dict[str, Any]:
            from calibration_module.fit.step8_report import report_csv

            return {"status": "complete", "csv": path, "written": False,
                    "step7": Path(calib_file),
                    "result": report_csv(
                        path, calib_file, out_dir=out,
                        log=lambda m: self._log(f"[step 8] {m}"))}

        self._run_task("Step 8 fit", work,
                       self._step8_finished, self._step8_error)

    def _step8_finished(self, payload: dict[str, Any]) -> None:
        self.step8_stop_event = None
        csv_path = payload.get("csv")
        # The last fit this page produced, sweep or Load.  Nothing acts on it
        # any more (Save is gone); it is kept because a run that read nothing
        # has to be distinguishable from one that was never started.
        self.step8_last = payload if csv_path is not None else None
        self._step8_set_running(False)
        if csv_path is None:
            self.step8_status.setText("Stopped before the first read; nothing written.")
            if self.calibration_dialog is not None:
                self.calibration_dialog.finish(False, "Nothing was read")
            return

        result = payload["result"]
        partial = not str(payload.get("status", "")).startswith("complete")
        lines: list[str] = []
        step8_print_result(result, log=lines.append)
        if partial:
            lines.insert(0, f"RUN WAS {payload['status'].upper()} -- the fit below "
                            f"is on what was read before it stopped.")
        self.step8_result_text.setPlainText("\n".join(lines).strip())

        self._step8_show_png(Path(result["png"]))
        # A sweep wrote the section; a Load only looked.  Saying which is the
        # whole of what Save used to be for.
        where = (f"wrote step8 + {Path(csv_path).name}"
                 if payload.get("written") else
                 f"re-fit {Path(csv_path).name} \N{EM DASH} nothing written")
        # Re-describe BEFORE the result: a written section changes the chain
        # every page quotes, and _calib_describe_all writes into this very
        # line, so doing it after would replace the run with the file.
        if payload.get("written"):
            self._calib_describe_all()
        self.step8_status.setText(
            f"Y_sat = {result['y_sat_v']:.4f} \N{PLUS-MINUS SIGN} "
            f"{result['sigma_v']:.4f} V"
            + ("  (LIMIT: the sweep could not bound it)" if result["unbounded"] else "")
            + f"  \N{MIDDLE DOT} {where}")
        if self.calibration_dialog is not None:
            self.calibration_dialog.finish(
                not partial and not result["unbounded"],
                f"Y_sat = {result['y_sat_v']:.4f} V")

    def _step8_show_png(self, png: Path) -> None:
        if not png.exists():
            self.step8_plot_label.setText(f"{png.name} was not written")
            return
        pixmap = QtGui.QPixmap(str(png))
        if pixmap.isNull():
            self.step8_plot_label.setText(f"{png.name} could not be rendered")
            return
        self.step8_plot_label.setPixmap(pixmap.scaledToWidth(
            max(720, self.step8_plot_label.width()),
            QtCore.Qt.SmoothTransformation))

    def _step8_error(self, _error: str) -> None:
        self.step8_stop_event = None
        self._step8_set_running(False)
        self.step8_status.setText("Saturation sweep failed (see Status log)")
        if self.calibration_dialog is not None:
            self.calibration_dialog.finish(False, "Saturation sweep failed")

    # ---- results table -----------------------------------------------------
    def _tpa_phase_fill_table(self) -> None:
        """One row per target: every column a scalar, so ten targets stay readable.

        The PHASE columns are the chained ones -- referred to the root reference,
        whichever hub actually measured the pair -- because a table that mixed
        two reference frames would need reading pair by pair to be believed.
        Everything after them (a, b, R², pull, residual) describes the hop that
        was measured, unchained: that is what says whether the fringe was any
        good, and it is not the root's business.  A hopped row says so in its
        tooltip.
        """
        self.tpa_phase_table.blockSignals(True)
        items = sorted(self.tpa_phase_results.items())
        self.tpa_phase_table.setRowCount(len(items))
        for row, (k, result) in enumerate(items):
            fit = result.fit
            chain = self.tpa_phase_chains.get(k)
            if fit is None or chain is None:
                cells = [str(k)] + ["\N{EN DASH}"] * (
                    len(self._TPA_PHASE_COLUMNS) - 1
                )
            else:
                cells = [
                    str(k),
                    f"{chain.dphi_deg:+.2f}",
                    # The quoted error is the TOTAL: fringe noise and the pinned
                    # step-6 eta in quadrature.  a and b do not float, so the
                    # fitter's own error cannot see the eta term -- it is
                    # invisible there, not absent, hence all three columns.
                    # Chained, these gain the hub's hop: its fringe noise adds
                    # in quadrature, but its eta enters BOTH hops and so is
                    # summed coherently first (see cascade.ChainedPhase).
                    f"{np.degrees(chain.err_total):.2f}",
                    f"{np.degrees(chain.err_fringe):.2f}",
                    f"{np.degrees(chain.err_eta):.2f}",
                    f"{fit.a*1e3:.3f}",
                    f"{fit.b*1e3:.3f}",
                    f"{fit.r2:.5f}",
                    f"{float(np.max(np.abs(fit.pulls))):.2f}",
                    # NOT fitted in v2, so it is a pure check on step 6: a big
                    # mean residual means the pinned amplitudes are off.
                    f"{float(np.mean(fit.residuals))*1e3:+.4f}",
                ]
            tip = self._tpa_phase_row_tip(k, result, chain)
            for col, text in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(text)
                if col:
                    item.setTextAlignment(
                        QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter
                    )
                item.setToolTip(tip)
                self.tpa_phase_table.setItem(row, col, item)
        self.tpa_phase_table.resizeColumnsToContents()
        self.tpa_phase_table.blockSignals(False)
        if items:
            self.tpa_phase_table.selectRow(0)

    def _tpa_phase_row_tip(self, k: int, result: "PhaseResult", chain) -> str:
        """Where this row's phase came from — the only place the hub is named.

        A chained pair looks exactly like a measured one in the table, which is
        the point (one reference frame, one column), so the provenance has to
        live somewhere: here, at no cost in layout.
        """
        if chain is None or result.fit is None:
            return f"Pair {k}: no fit."
        if not chain.chained:
            return f"Pair {k} measured directly against reference {chain.via}."
        return (
            f"Pair {k} could not be swept against reference {chain.root} "
            f"(neighbours leak), so it was measured against hub {chain.via} "
            f"and chained:\n"
            f"  ΔΦ[{k}|{chain.via}] = {np.degrees(result.fit.dphi_comb):+.2f}°\n"
            f"  ΔΦ[{chain.via}|{chain.root}] = "
            f"{np.degrees(chain.dphi_raw - result.fit.dphi_comb):+.2f}°\n"
            f"  ΔΦ[{k}|{chain.root}] = {chain.dphi_deg:+.2f}°\n"
            f"The a/b/R²/pull columns describe the measured hop, not the chain."
        )

    def _tpa_phase_selected(self) -> "tuple[int, PhaseResult] | None":
        items = sorted(self.tpa_phase_results.items())
        if not items:
            return None
        row = self.tpa_phase_table.currentRow()
        if row < 0 or row >= len(items):
            row = 0
        return items[row]

    def _tpa_phase_redraw(self) -> None:
        """The selected target's fringe, drawn by the renderer the PNG uses."""
        picked = self._tpa_phase_selected()
        if picked is None or picked[1].fit is None:
            self.tpa_phase_fig.clear()
            ax = self.tpa_phase_fig.add_subplot(111)
            ax.set_axis_off()
            ax.text(0.5, 0.5, "Run or load a sweep", ha="center", va="center",
                    transform=ax.transAxes)
            self.tpa_phase_canvas.draw_idle()
            return
        k, result = picked
        plot_fringe(self.tpa_phase_fig, result.fit, k)
        self.tpa_phase_canvas.draw_idle()

    # ---- files -------------------------------------------------------------
    @staticmethod
    def _tpa_phase_root_of(refs: "dict[int, int]") -> "int | None":
        """Which reference in a recorded run is the ROOT: the one nothing measured.

        A cascade's hub is itself a target of an earlier stage, so it appears in
        both columns; the root only ever appears as a reference.  Exactly one
        such pair means one run rooted there.  None means the file is not one
        run -- no root at all is a cycle, two roots is two runs concatenated --
        and the caller says so rather than picking one.

        A row whose target IS its reference does not count as measuring it: a
        recorded run carries the reference's own self-rows (0907 on), and
        counting those would leave every file looking like a cycle.
        """
        measured = {k for k, r in refs.items() if k != r}
        roots = sorted(set(refs.values()) - measured)
        return roots[0] if len(roots) == 1 else None

    def _tpa_phase_load(self) -> None:
        """Re-fit the sweep this calib file records: every target, in one go.

        ONE input, asked for: a calib.json.  Its step7 section says which
        sweep(s) it was fitted from and its step6 section gives the etas the
        amplitudes are pinned to -- which is the whole of what a step-7 fit
        needs, and why this takes one file and not a file plus a CSV.  The
        page's own field is not touched: it is where a SWEEP would write.

        Cascades included.  A CSV records a reference PER ROW, so a run that
        needed a hub carries two of them, and a section may even name one CSV
        per stage (the hand-merged 0916 run does).  Each target is re-fit
        against the reference it was actually swept against, out of the file
        that carries it, before the lot is chained back to the root.  Nothing
        here consults the planner: what was driven is what the files say.

        Reads only -- an offline re-fit is a copy of the JSON and its CSVs and a
        run of ``calib_step7_v2.py`` against the copy.
        """
        picked = self._open_calib("Open a calibration to re-fit its step 7", 7)
        if picked is None:
            return
        try:
            calib = load_calib(picked)
            paths = calib.csv_paths(7)
        except Exception as exc:                        # noqa: BLE001
            self.tpa_phase_status.setText(f"Load failed: {exc}")
            return
        if not paths:
            self.tpa_phase_status.setText(
                f"Load failed: {calib.path.name}'s step7 section names no CSV, "
                f"so there is no sweep to re-fit."
            )
            return
        gone = [q for q in paths if not q.is_file()]
        if gone:
            self.tpa_phase_status.setText(
                f"Load failed: {calib.path.name} names "
                f"{', '.join(q.name for q in gone)}, which is not in "
                f"{paths[0].parent} — copy the CSV in beside it."
            )
            return
        try:
            _layout, models = self._tpa_phase_load_step6(picked)
            floor = self._tpa_phase_floor(picked)
        except Exception as exc:
            # Unlike step 6, this is fatal: the amplitudes are PINNED to step 6
            # and the points are weighted by its floor, so without them there
            # is no fit to do at all.
            self.tpa_phase_status.setText(f"Step 6 required to re-fit: {exc}")
            return
        name = (paths[0].name if len(paths) == 1
                else f"{len(paths)} sweeps ({paths[0].name} + …)")
        try:
            targets, recorded, csv_of = self._tpa_phase_pool(paths)
        except Exception as exc:
            self.tpa_phase_status.setText(f"Load failed: {exc}")
            return

        # The CSV knows its own reference(s) -- every row records one -- so it
        # wins over the spinbox, which is a setting for the NEXT sweep and has
        # no business deciding how an existing one is read.  The spinbox is
        # moved to match, so the page never shows a reference the results are
        # not against.
        root = self._tpa_phase_root_of(recorded) if recorded else None
        if recorded and root is None:
            self.tpa_phase_status.setText(
                f"{name} is not one run: its references "
                f"{self._compact_indices(sorted(set(recorded.values())))} are all "
                f"themselves measured in it, so no pair defines ΔΦ = 0. It looks "
                f"like two runs concatenated — load them separately."
            )
            return
        if root is None:                       # old CSV with no ref_index column
            root = self._tpa_phase_config().ref_index
        elif root != self.tpa_phase_ref.value():
            self.tpa_phase_ref.setValue(root)
            self._log(f"[step 7] {name} was swept against reference "
                      f"{root}; using that, not the {root} on the page")
        ref_of = {k: recorded.get(k, root) for k in targets}
        if root not in models:
            # One run-level failure, not one per target: with no eta for the
            # reference there is no `a` to pin, so nothing in this file is
            # fittable and saying it five times explains it no better.
            self.tpa_phase_status.setText(
                f"{name} was swept against reference {root}, but the Step-6 file "
                f"has no η for it — it covers "
                f"{self._compact_indices(sorted(models))}. Point Step 6 at the "
                f"result this sweep was measured against."
            )
            return
        fittable = [k for k in targets
                    if k in models and k != root and ref_of[k] in models]
        if not fittable:
            missing = [k for k in targets if k not in models and k != root]
            why = (f"step 6 has no η for {self._compact_indices(missing)}"
                   if missing else "it carries only the reference")
            self.tpa_phase_status.setText(
                f"No fittable target in {name}: it carries "
                f"{self._compact_indices(targets)} against reference {root}, and "
                f"{why}."
            )
            return

        single_beam_bg = self.tpa_phase_single_beam.isChecked()
        loaded: dict[int, PhaseResult] = {}
        failed: list[str] = []
        for k in fittable:
            try:
                loaded[k] = load_phase_csv(
                    csv_of[k], models[k], models[ref_of[k]],
                    comb_only=True, single_beam_bg=single_beam_bg, only_tgt=k,
                    floor=floor, rail_v=self._tpa_phase_rail_v(),
                )
            except (ValueError, KeyError, np.linalg.LinAlgError) as exc:
                failed.append(f"pair {k}: {exc}")
                continue
            # See the step-6 note in _tpa_fit_rows: load_phase_csv drops these,
            # and this is the only place a GUI user would ever hear about it.
            for b in loaded[k].dropped_bursts:
                self._log(
                    f"[step 7] pair {k}: BURST acquisition dropped — "
                    f"x_t=w_t={b['x_t']:g} read {b['voltage_mean_v']*1e3:+.4f} mV "
                    f"with a {b['voltage_std_v']*1e3:.4f} mV trace std"
                )
        self.tpa_phase_floor = floor
        self._log(f"[step 7] sigma floor {floor[0]*1e6:.1f} µV [{floor[1]}]")
        # Already fitted by load_phase_csv; this only re-refers them to the root,
        # over stages rebuilt from what the file recorded rather than planned.
        self.tpa_phase_results = loaded
        # From the first sweep the section names: a re-save writes ONE CSV, so
        # a merged run's second bracket has nowhere to go anyway.
        self.tpa_phase_meta = meas_csv_meta(paths[0])  # see the step-6 Load
        self.tpa_phase_root = root
        self.tpa_phase_stages = self._tpa_phase_stages_for(loaded, root)
        self.tpa_phase_chains = chain_fits(
            {k: r.fit for k, r in loaded.items() if r.fit is not None},
            self.tpa_phase_stages, root,
        )
        self._tpa_phase_fill_table()
        self._tpa_phase_redraw()
        skipped = [k for k in targets if k not in fittable and k != root]
        note = ""
        if failed:
            note += "  · " + "; ".join(failed)
        if skipped:
            note += (f"  · skipped {self._compact_indices(skipped)} "
                     f"(no step-6 η)")
        chained = sorted(k for k, c in self.tpa_phase_chains.items() if c.chained)
        if chained:
            note += (f"  · {self._compact_indices(chained)} chained to {root} "
                     f"via {self._compact_indices(sorted({self.tpa_phase_chains[k].via for k in chained}))}")
        self.tpa_phase_status.setText(
            f"Loaded {self._calib_name(calib.path)} · {name} · "
            f"{len(self.tpa_phase_results)} target(s) re-fit vs ref {root}{note}"
        )
        for line in failed:
            self._log(f"[step 7] fit failed — {line}")

    def _tpa_phase_plan_payload(self) -> dict:
        """The stage breakdown, stored in the JSON's step-7 section.

        One CSV holds a whole cascaded run, and the chained pairs' own blocks
        say which hub each took, but nothing there records the SHAPE of the run.
        This does, in one place, so a file can be read back as the run it was
        rather than reconstructed from nine per-pair footnotes.  Omitted
        entirely for a plain single-reference run, which has no shape to record.
        """
        if not any(s.depth for s in self.tpa_phase_stages):
            return {}
        return {"cascade": {
            "min_sep": MIN_PAIR_SEP,
            "stages": [
                {"ref": s.ref, "targets": list(s.targets), "depth": s.depth}
                for s in self.tpa_phase_stages
            ],
        }}

    def _tpa_phase_extra(self) -> dict:
        """What rides alongside the phases: the plan, the floor, repaired darks."""
        extra = dict(self._tpa_phase_plan_payload())
        floor = getattr(self, "tpa_phase_floor", None)
        if floor is not None:
            extra["sigma_floor_v"] = float(floor[0])
            extra["sigma_floor_source"] = str(floor[1])
        repaired: dict = {}
        for r in self.tpa_phase_results.values():
            repaired.update(getattr(r, "dark_overrides", {}) or {})
        if repaired:
            extra["dark_overrides"] = repaired
        return extra

    def _tpa_phase_rail_v(self) -> float:
        """The page's rail ceiling in VOLTS, for the guards that reject by it."""
        return float(self.tpa_phase_rail.value()) / 1e3

    def _tpa_phase_floor(self,
                         calib_file: "Path | None" = None) -> tuple[float, str]:
        """The sigma floor to weight this fit by: the run's step-6 measurement.

        Step 7 has no dark LEVEL of its own to measure a floor from -- one
        all-off read per target and ten drive rows -- so it inherits the one
        step 6 measured on this run, through the same detector on the same
        night.  A step 6 that predates the recorded floor is measured from its
        own CSV; one with neither refuses.  No constant stands in.
        """
        run = (calib_file if calib_file is not None
               else self._tpa_phase_step6_path())
        if run is None:
            raise ValueError("no calibration file chosen")
        return step6_floor(run)

    def _tpa_phase_autosave(self) -> None:
        """A finished sweep writes itself: the raw CSV, then the step7 section.

        Both into the calib file named on this page -- the CSV beside it, the
        section in it -- whose step3 and step6 sections are what this spectrum
        is referred to.  Writing DROPS that file's step 8, which was fitted
        against the phases this sweep has just replaced.

        No PNGs, for the same reason step 6 writes none: Load redraws the fringe
        from the CSV the section points at.

        Only a sweep calls this.  Loading a recorded sweep re-fits and draws but
        never writes.

        The rows are written first and unconditionally.  A measurement is an
        hour of bench time and must never be lost to an unreadable calibration,
        which is only needed for the derived section.
        """
        if not self.tpa_phase_results:
            return
        stamp = time.strftime("%m%d_%H%M")
        calib_file = self._calib_file_path(7)
        if calib_file is None:
            folder = self._unfiled_run(stamp)
            self._log(
                f"[step 7] no calibration file on the page — rows parked in "
                f"{folder}"
            )
        else:
            folder = calib_file.parent
            folder.mkdir(parents=True, exist_ok=True)
        csv_path = folder / f"calib_step7_meas_{stamp}.csv"
        results = [r for _, r in sorted(self.tpa_phase_results.items())]
        written = [Path(write_phase_meas_csv(results, csv_path,
                                             meta=self.tpa_phase_meta)).name]

        # The file's step3 and step6 sections are what this spectrum is referred
        # to, and save_step7_section checks them.
        try:
            load_calib(calib_file or folder).require(3, 6)
        except Exception as exc:                        # noqa: BLE001
            self.tpa_phase_status.setText(
                f"Saved {written[0]} — {exc}, so the step7 section was not "
                "written (step 8 could not read it)."
            )
            return
        fits = {
            (k, "fixed_comb_only"): r.fit
            for k, r in sorted(self.tpa_phase_results.items())
            if r.fit is not None
        }
        if not fits:
            self.tpa_phase_status.setText(
                f"Saved {written[0]} — no fitted target, so no JSON was written."
            )
            return
        # The ROOT, not the spinbox: a loaded file may have moved it, and after
        # a cascade the spinbox is the root anyway.  Every phase in the JSON is
        # referred here, which is the contract step 8 reads it under.
        root = (self.tpa_phase_root if self.tpa_phase_root is not None
                else self._tpa_phase_config().ref_index)
        try:
            calib = save_step7_section(
                fits, calib_file,
                ref_index=root,
                csv=csv_path,
                single_beam_bg=self.tpa_phase_single_beam.isChecked(),
                chains=self.tpa_phase_chains,
                extra=self._tpa_phase_extra(),
                log=None,
            )
            written.append(f"step7 -> {calib.path.name}")
        except Exception as exc:
            self.tpa_phase_status.setText(
                f"Saved {written[0]}; section failed: {exc}"
            )
            return
        self._calib_describe_all()
        self.tpa_phase_status.setText(
            self.tpa_phase_status.text() + "  ·  saved " + "  +  ".join(written))

    def _build_tpa_center_tab(self) -> QtWidgets.QWidget:
        page = self._page_shell("TPA Centre-Wavelength Calibration")
        subtitle = QtWidgets.QLabel(
            "Scan the symmetric layout centre wavelength, read the TPA fluorescence "
            "brightness from the DAQ (Step 6 acquisition settings), and fit the peak with a "
            "weighted quadratic. Build the channel grid on the TPA Encoding page "
            "first; this scan reuses its width, spacing and channel count."
        )
        subtitle.setObjectName("PageSubtitle")
        subtitle.setWordWrap(True)
        page.layout().addWidget(subtitle)

        cfg = self._panel("Scan Settings")
        grid = QtWidgets.QGridLayout(cfg)
        self.tpa_center_pair_index = self._spin(0, 63, 0)
        self.tpa_center_pair_index.setToolTip("Which symmetric x/w pair to keep on")
        self.tpa_center_drive = QtWidgets.QDoubleSpinBox()
        self.tpa_center_drive.setRange(0.0, 1.0)
        self.tpa_center_drive.setSingleStep(0.05)
        self.tpa_center_drive.setDecimals(2)
        self.tpa_center_drive.setValue(1.0)
        self.tpa_center_drive.setToolTip("Per-side drive level for the selected pair")
        self.tpa_center_start = self._double_spin(700.0, 900.0, 777.8, " nm", 4)
        self.tpa_center_stop = self._double_spin(700.0, 900.0, 778.2, " nm", 4)
        self.tpa_center_points = self._spin(3, 51, 9)
        self.tpa_center_trials = self._spin(1, 500, 5)
        self.tpa_center_repeats = self._spin(1, 20, 1)
        self.tpa_center_bg_check = QtWidgets.QCheckBox("Subtract all-off background")
        self.tpa_center_bg_check.setChecked(True)
        self.tpa_center_bg_check.setToolTip(
            "Measure an all-off pattern at each centre first and subtract it."
        )
        widgets = [
            ("Pair index", self.tpa_center_pair_index),
            ("Drive", self.tpa_center_drive),
            ("Start λ", self.tpa_center_start),
            ("Stop λ", self.tpa_center_stop),
            ("Points", self.tpa_center_points),
            ("Trials", self.tpa_center_trials),
            ("Repeats", self.tpa_center_repeats),
        ]
        for i, (label, widget) in enumerate(widgets):
            r, c = i // 2, (i % 2) * 2
            grid.addWidget(QtWidgets.QLabel(label), r, c)
            grid.addWidget(widget, r, c + 1)
        grid.addWidget(self.tpa_center_bg_check, 4, 0, 1, 4)
        page.layout().addWidget(cfg)

        self.tpa_center_fig = Figure(figsize=(8, 3.4), tight_layout=True)
        self.tpa_center_canvas = FigureCanvas(self.tpa_center_fig)
        page.layout().addWidget(
            self._panel_with_widget("Brightness vs centre wavelength", self.tpa_center_canvas),
            1,
        )

        self.tpa_center_report = QtWidgets.QLabel("Centre: (run a scan)")
        self.tpa_center_report.setObjectName("PageSubtitle")
        self.tpa_center_report.setWordWrap(True)
        self.tpa_center_apply_button = QtWidgets.QPushButton("Apply to TPA Encoding")
        self.tpa_center_apply_button.setProperty("variant", "ghost")
        self.tpa_center_apply_button.setEnabled(False)
        self.tpa_center_apply_button.clicked.connect(self._tpa_center_apply)
        report_row = QtWidgets.QHBoxLayout()
        report_row.addWidget(self.tpa_center_report, 1)
        report_row.addWidget(self.tpa_center_apply_button)
        page.layout().addLayout(report_row)

        self.tpa_center_bar = QtWidgets.QProgressBar()
        self.tpa_center_bar.setValue(0)
        self.tpa_center_status = QtWidgets.QLabel("\N{EN DASH}")
        self.tpa_center_run_button = QtWidgets.QPushButton("Run Centre Scan")
        self.tpa_center_run_button.clicked.connect(self._tpa_center_run)
        self.tpa_center_stop_button = QtWidgets.QPushButton("Stop")
        self.tpa_center_stop_button.setProperty("variant", "danger")
        self.tpa_center_stop_button.setEnabled(False)
        self.tpa_center_stop_button.clicked.connect(self._tpa_center_stop)
        ctrl = QtWidgets.QHBoxLayout()
        ctrl.addWidget(self.tpa_center_status, 1)
        ctrl.addWidget(self.tpa_center_run_button)
        ctrl.addWidget(self.tpa_center_stop_button)
        page.layout().addWidget(self.tpa_center_bar)
        page.layout().addLayout(ctrl)

        self._tpa_center_draw(None)
        return page

    def _tpa_center_set_running(self, running: bool) -> None:
        self._mark_tpa_calibration("step6b", running)
        self.tpa_center_run_button.setEnabled(not running)
        self.tpa_center_stop_button.setEnabled(running)
        can_apply = (
            not running
            and self.tpa_center_result is not None
            and self.tpa_center_result.fit is not None
            and self.tpa_center_result.fit.valid
        )
        self.tpa_center_apply_button.setEnabled(can_apply)

    def _tpa_center_run(self) -> None:
        from dataclasses import replace

        layout = self.encoding_layout
        if layout is None:
            self.tpa_center_status.setText(
                "No channel grid — build a layout on the TPA Encoding page first."
            )
            return
        calib = self._enc_get_calib()
        if calib is None or not calib.encodable:
            self.tpa_center_status.setText(
                "No Step 3 calibration available for rebuilding the layout."
            )
            return
        pair_index = self.tpa_center_pair_index.value()
        if pair_index >= layout.n_channels:
            self.tpa_center_status.setText(
                f"Pair index out of range (layout has {layout.n_channels} pairs)."
            )
            return
        start = self.tpa_center_start.value()
        stop = self.tpa_center_stop.value()
        if stop <= start:
            self.tpa_center_status.setText("Stop λ must be greater than start λ.")
            return

        monitor = self.daq_controller
        if monitor is None or not monitor.is_connected:
            self.tpa_center_status.setText("Connect the DAQ first (Connections page).")
            return
        controller = self._controller()
        if not getattr(controller, "is_open", False):
            self.tpa_center_status.setText("Open the SLM on the SLM Control page first.")
            return

        centers = np.linspace(start, stop, self.tpa_center_points.value())
        n_trials = self.tpa_center_trials.value()
        repeats = self.tpa_center_repeats.value()
        subtract_bg = self.tpa_center_bg_check.isChecked()
        total = max(n_trials * centers.size * (2 if subtract_bg else 1), 1)
        gap_px = layout.pitch_px - layout.channel_width_px

        # the Encoding page's readout is gone; the scan reads the DAQ with the
        # Step 6 tab's acquisition settings
        lo, hi = self.tpa_daq_range.currentData()
        settings = DAQMonitorSettings(
            channel=self.tpa_daq_channel.text().strip() or "ai0",
            sample_rate=self.tpa_daq_rate.value(),
            duration=self.tpa_tmeas.value(),
            hold=self.tpa_settle.value(),
            min_val=lo, max_val=hi,
            f_cut=self.tpa_daq_fcut.value(),
        )
        settle = float(settings.hold)
        read_timeout = max(30.0, settings.duration * 3.0 + 10.0)
        settings0 = replace(settings, hold=0.0)

        self.tpa_center_bar.setMaximum(total)
        self.tpa_center_bar.setValue(0)
        self.tpa_center_status.setText(
            f"Starting… pair[{pair_index}] · {start:.4f}–{stop:.4f} nm · "
            f"{centers.size} points via DAQ"
        )
        self._tpa_center_set_running(True)

        stop_event = threading.Event()
        self.tpa_center_stop_event = stop_event

        def report(progress: TPACenterProgress) -> None:
            self.tpa_center_progress.emit(progress)

        def work() -> dict[str, Any]:
            monitor.configure_monitor(settings0)
            try:
                result = measure_center_scan(
                    monitor,
                    controller,
                    calib,
                    center_wavelengths_nm=centers,
                    n_channels=layout.n_channels,
                    channel_width_px=layout.channel_width_px,
                    gap_px=gap_px,
                    center_gap_px=layout.center_gap_px,
                    pair_index=pair_index,
                    drive_level=self.tpa_center_drive.value(),
                    n_trials=n_trials,
                    repeats=repeats,
                    settle=settle,
                    read_timeout=read_timeout,
                    col_ratio=self._active_col_ratio(),
                    subtract_background=subtract_bg,
                    stop_event=stop_event,
                    progress_callback=report,
                )
            except TPACenterAborted:
                return {"status": "aborted"}
            return {"status": "ok", "result": result}

        self._run_slm_task(
            "TPA centre scan", work, self._tpa_center_finished, self._tpa_center_error
        )

    def _tpa_center_stop(self) -> None:
        if self.tpa_center_stop_event is not None:
            self.tpa_center_stop_event.set()
            self.tpa_center_status.setText("Stopping…")

    def _on_tpa_center_progress(self, progress: TPACenterProgress) -> None:
        self.tpa_center_bar.setMaximum(max(progress.total, 1))
        self.tpa_center_bar.setValue(min(progress.step, progress.total))
        self.tpa_center_status.setText(progress.message)

    def _tpa_center_finished(self, payload: dict[str, Any]) -> None:
        self.tpa_center_stop_event = None
        self._tpa_center_set_running(False)
        if payload.get("status") == "aborted":
            self.tpa_center_status.setText("Centre scan stopped.")
            return
        result = payload["result"]
        self.tpa_center_result = result
        self._tpa_center_update_report(result)
        self._tpa_center_draw(result)
        fit = result.fit
        if fit is not None and fit.valid:
            center_x = self._tpa_center_wavelength_to_x(fit.center_wl_nm, result)
            if center_x is None:
                self.tpa_center_status.setText(
                    f"Done · centre {fit.center_wl_nm:.4f} nm"
                )
            else:
                self.tpa_center_status.setText(
                    f"Done · centre {fit.center_wl_nm:.4f} nm (x={center_x:.3f} px)"
                )
        elif fit is not None:
            self.tpa_center_status.setText(
                f"Scan finished · best sampled {fit.best_sample_center_wl_nm:.4f} nm "
                f"({fit.message})"
            )
        else:
            self.tpa_center_status.setText("Scan finished.")
        self._tpa_center_set_running(False)

    def _tpa_center_error(self, _error: str) -> None:
        self.tpa_center_stop_event = None
        self._tpa_center_set_running(False)
        self.tpa_center_status.setText("TPA centre scan failed (see Status log)")

    def _tpa_center_wavelength_to_x(
        self,
        wavelength_nm: float,
        result: TPACenterResult | None = None,
    ) -> float | None:
        if not np.isfinite(wavelength_nm):
            return None
        calib = self._enc_get_calib()
        if calib is not None:
            try:
                return float(interpolate_coordinate_for_wavelength(calib, wavelength_nm))
            except Exception:
                pass
        if result is None or result.center_wl_nm.size == 0:
            return None
        grouped: dict[float, list[float]] = {}
        for wl, x in zip(result.center_wl_nm, result.center_x_px):
            grouped.setdefault(float(wl), []).append(float(x))
        wl_sorted = np.array(sorted(grouped), dtype=float)
        x_sorted = np.array(
            [float(np.mean(grouped[float(wl)])) for wl in wl_sorted],
            dtype=float,
        )
        if wavelength_nm < wl_sorted.min() or wavelength_nm > wl_sorted.max():
            return None
        return float(np.interp(wavelength_nm, wl_sorted, x_sorted))

    def _tpa_center_update_report(self, result: TPACenterResult | None) -> None:
        if result is None or result.fit is None:
            self.tpa_center_report.setText("Centre: (run a scan)")
            self.tpa_center_apply_button.setEnabled(False)
            return
        fit = result.fit
        if fit.valid:
            center_x = self._tpa_center_wavelength_to_x(fit.center_wl_nm, result)
            x_text = f"x = {center_x:.3f} px   " if center_x is not None else ""
            self.tpa_center_report.setText(
                f"centre λ = {fit.center_wl_nm:.4f} ± {fit.center_wl_err_nm:.4f} nm   "
                f"{x_text}"
                f"peak = {fit.peak_signal_v * 1000:.4f} ± "
                f"{fit.peak_signal_err_v * 1000:.4f} mV   "
                f"χ²/dof = {fit.chi2_red:.2f} (Birge ×{fit.birge:.2f})"
            )
        else:
            best_x = self._tpa_center_wavelength_to_x(fit.best_sample_center_wl_nm, result)
            x_text = f"x = {best_x:.3f} px   " if best_x is not None else ""
            self.tpa_center_report.setText(
                f"best sampled λ = {fit.best_sample_center_wl_nm:.4f} nm   "
                f"{x_text}"
                f"signal = {fit.best_sample_signal_v * 1000:.4f} mV   "
                f"fit invalid: {fit.message}"
            )
        self.tpa_center_apply_button.setEnabled(fit.valid and self.tpa_center_stop_event is None)

    def _tpa_center_draw(self, result: TPACenterResult | None) -> None:
        self.tpa_center_fig.clear()
        self.tpa_center_fig.patch.set_facecolor("#101820")
        ax = self.tpa_center_fig.add_subplot(111)
        self._style_dark_axes(ax)
        ax.set_xlabel("Centre wavelength (nm)")
        ax.set_ylabel("Net signal (mV)")
        if result is None:
            ax.text(0.5, 0.5, "Run a centre scan", ha="center", va="center",
                    transform=ax.transAxes, color="#d8dee9")
            self.tpa_center_canvas.draw_idle()
            return

        fit = result.fit
        if fit is not None:
            wl = fit.center_wl
            signal = fit.signal_v
            sem = fit.sem_v
        else:
            wl, signal, sem = average_trace_points(result.center_wl_nm, result.net_signal_v)
        signal_mv = signal * 1e3
        sem_mv = sem * 1e3
        ax.errorbar(
            wl,
            signal_mv,
            yerr=sem_mv,
            fmt="o",
            color="#88c0d0",
            ecolor="#41515c",
            capsize=3,
            label="measured",
        )
        if fit is not None:
            dense = np.linspace(float(np.min(wl)), float(np.max(wl)), 200)
            a, b, c = fit.coeffs
            curve_mv = (a * dense**2 + b * dense + c) * 1e3
            ax.plot(dense, curve_mv, color="#ebcb8b", linewidth=1.4, label="quadratic fit")
            if fit.valid:
                ax.axvline(fit.center_wl_nm, color="#e05a5a", linestyle="--", linewidth=1.0)
                ax.plot(
                    [fit.center_wl_nm],
                    [fit.peak_signal_v * 1e3],
                    "o",
                    color="#e05a5a",
                    markersize=6,
                    label="fit centre",
                )
            else:
                ax.plot(
                    [fit.best_sample_center_wl_nm],
                    [fit.best_sample_signal_v * 1e3],
                    "s",
                    color="#e0a447",
                    markersize=5,
                    label="best sample",
                )
        ax.legend(loc="best", fontsize=8)
        self.tpa_center_canvas.draw_idle()

    def _tpa_center_apply(self) -> None:
        result = self.tpa_center_result
        fit = None if result is None else result.fit
        if fit is None or not fit.valid:
            return
        self.tpa_center_status.setText(
            f"Fitted centre {fit.center_wl_nm:.4f} nm — the encoding layout is "
            "read verbatim from the Step 3c calibration, so re-run Step 3c "
            "with this target centre to shift the channels."
        )
        self._enc_log(
            f"TPA centre scan fitted {fit.center_wl_nm:.4f} nm. The encoding "
            "layout comes from the Step 3c calibration; re-run Step 3c with "
            "this target centre to re-centre the channels."
        )

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
        self.daq_controller = daq
        self._set_status(self.daq_status_label, "DAQ: open", "ok")
        self.daq_connect_button.setEnabled(False)
        self.daq_disconnect_button.setEnabled(True)
        self._log(f"DAQ connected: {identity.strip()}")
        self._refresh_step3_run_button()

    def _on_daq_error(self, _error: str) -> None:
        self._set_status(self.daq_status_label, "DAQ: error", "error")
        self.daq_connect_button.setEnabled(True)

    def _disconnect_daq(self) -> None:
        daq = self.daq_controller
        if self.daq_stream_stop_event is not None:
            self.daq_stream_stop_event.set()   # the loop is holding this card
        self.daq_controller = None
        self._set_status(self.daq_status_label, "DAQ: closed", "off")
        self.daq_connect_button.setEnabled(True)
        self.daq_disconnect_button.setEnabled(False)
        if daq is not None:
            self._run_task("Disconnect DAQ", daq.disconnect)

    # ===================== Heater (Thorlabs TC300B) =====================
    def _connect_heater(self) -> None:
        port = self.heater_port_edit.text().strip()
        if not port:
            self._log("Enter the heater serial port first")
            return
        self.heater_connect_button.setEnabled(False)

        def connect() -> tuple[TC300Controller, str]:
            heater = TC300Controller(port=port)
            heater.connect()
            return heater, heater.identify()

        self._run_task("Connect heater", connect,
                       self._on_heater_connected, self._on_heater_error)

    def _on_heater_connected(self, payload: tuple[TC300Controller, str]) -> None:
        heater, identity = payload
        self.heater_controller = heater
        self._set_status(self.heater_status_label, "Heater: open", "ok")
        self.heater_connect_button.setEnabled(False)
        self.heater_disconnect_button.setEnabled(True)
        self._log(f"Heater connected: {identity.strip()}")
        self._heater_refresh_buttons()
        if hasattr(self, "heater_page_status"):
            self.heater_page_status.setText(
                "Heater connected. Choose channels + target, then Ramp & Hold "
                "(cold start) or Monitor (watch a running hold)."
            )

    def _on_heater_error(self, _error: str) -> None:
        self._set_status(self.heater_status_label, "Heater: error", "error")
        self.heater_connect_button.setEnabled(True)

    def _disconnect_heater(self) -> None:
        # A running loop owns the serial link; stop it first and defer the actual
        # close to when the loop unwinds (its finally may still disable channels).
        if self.heater_stop_event is not None:
            self._heater_disconnect_pending = True
            self.heater_stop_event.set()
            self._set_status(self.heater_status_label, "Heater: stopping…", "off")
            if hasattr(self, "heater_page_status"):
                self.heater_page_status.setText("Stopping the heater loop, then disconnecting…")
            return
        self._heater_do_disconnect()

    def _heater_do_disconnect(self) -> None:
        heater = self.heater_controller
        self.heater_controller = None
        self._heater_disconnect_pending = False
        self._set_status(self.heater_status_label, "Heater: closed", "off")
        self.heater_connect_button.setEnabled(True)
        self.heater_disconnect_button.setEnabled(False)
        self._heater_refresh_buttons()
        if heater is not None:
            self._run_task("Disconnect heater", heater.disconnect)

    # ------------------------------------------------------------ heater page
    def _build_heater_page(self) -> QtWidgets.QWidget:
        """Dedicated page: TC300B staircase ramp/hold + live temperature monitor.

        Mirrors ``src/drafts/heat_controller.py`` -- the same watchdog-safe
        staircase, adaptive step and tuned hold PID -- but driven interactively
        with a live numeric readout. Connect the heater on the Connections
        page first. The 79.5 C hold needs a constant-voltage DC base heater on
        the block (see the tooltip); without it the trim heater rails and
        limit-cycles.
        """
        page = self._page_shell("Heater")
        subtitle = QtWidgets.QLabel(
            "Thorlabs TC300B temperature control. <b>Ramp &amp; Hold</b> climbs to the "
            "target in watchdog-safe steps then holds, showing the live temperature; "
            "<b>Monitor</b> watches read-only without touching the drive. Connect the "
            "heater on the Connections page first."
        )
        subtitle.setObjectName("PageSubtitle")
        subtitle.setWordWrap(True)
        page.layout().addWidget(subtitle)

        # --- setpoint / staircase / PID parameters ---
        self.heater_cfg = self._panel("Setpoint, staircase & PID")
        grid = QtWidgets.QGridLayout(self.heater_cfg)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        self.heater_ch1_check = QtWidgets.QCheckBox("CH1")
        self.heater_ch1_check.setChecked(True)
        self.heater_ch2_check = QtWidgets.QCheckBox("CH2")
        self.heater_ch2_check.setChecked(True)
        ch_row = QtWidgets.QHBoxLayout()
        ch_row.addWidget(self.heater_ch1_check)
        ch_row.addWidget(self.heater_ch2_check)
        ch_row.addStretch(1)
        ch_wrap = QtWidgets.QWidget()
        ch_wrap.setLayout(ch_row)

        def dspin(lo, hi, val, dec, step, suffix, tip=""):
            s = QtWidgets.QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setDecimals(dec)
            s.setSingleStep(step)
            s.setValue(val)
            if suffix:
                s.setSuffix(suffix)
            if tip:
                s.setToolTip(tip)
            return s

        pid = HEATER_PID_DEFAULTS.get(1, {"kp": 0.5, "ti": 20.0, "td": 2.0})
        self.heater_target = dspin(0.0, 200.0, StaircaseSettings.target, 2, 0.5, " \N{DEGREE SIGN}C",
                                   "Final hold target")
        self.heater_step = dspin(0.2, 20.0, StaircaseSettings.step, 1, 0.5, " \N{DEGREE SIGN}C",
                                 "Max staircase step above current temperature")
        self.heater_rest = dspin(0.0, 30.0, StaircaseSettings.rest, 1, 0.5, " s",
                                 "Calm dwell at a landing before the next step")
        self.heater_railmax = dspin(2.0, 22.0, StaircaseSettings.railmax, 1, 1.0, " s",
                                    "Continuous railed seconds before an emergency EN=0 rest "
                                    "(keep < ~23 s no-load watchdog)")
        self.heater_period = dspin(0.1, 5.0, StaircaseSettings.period, 1, 0.1, " s",
                                   "Loop / sample period")
        self.heater_kp = dspin(0.0, 100.0, pid["kp"], 2, 0.05, "", "Proportional gain")
        self.heater_ti = dspin(0.0, 200.0, pid["ti"], 2, 1.0, " s", "Integral time (larger = gentler)")
        self.heater_td = dspin(0.0, 100.0, pid["td"], 2, 0.5, " s", "Derivative time")

        self.heater_vmax_check = QtWidgets.QCheckBox("Cap VMAX")
        self.heater_vmax_check.setToolTip("Cap the max drive voltage (unchecked = device max)")
        self.heater_vmax = dspin(1.0, 24.0, 24.0, 1, 0.5, " V", "Voltage cap when enabled")
        self.heater_vmax.setEnabled(False)
        self.heater_vmax_check.toggled.connect(self.heater_vmax.setEnabled)

        self.heater_keep_enabled = QtWidgets.QCheckBox("Leave channels enabled on stop")
        self.heater_keep_enabled.setToolTip(
            "Keep the hold running after Stop/Disconnect. The TC300 holds "
            "autonomously once the link closes, so the block stays hot."
        )
        self.heater_history = QtWidgets.QSpinBox()
        self.heater_history.setRange(10, 100_000)
        self.heater_history.setValue(1200)
        self.heater_history.setToolTip("How many recent points to keep on the chart")

        rows = [
            ("Channels", ch_wrap, "Target", self.heater_target, "Max step", self.heater_step),
            ("Rest", self.heater_rest, "Rail max", self.heater_railmax, "Period", self.heater_period),
            ("KP", self.heater_kp, "TI", self.heater_ti, "TD", self.heater_td),
            (None, self.heater_vmax_check, None, self.heater_vmax, "History", self.heater_history),
        ]
        for r, cells in enumerate(rows):
            for c in range(0, 6, 2):
                label, widget = cells[c], cells[c + 1]
                if label is not None:
                    grid.addWidget(QtWidgets.QLabel(label), r, c)
                grid.addWidget(widget, r, c + 1)
        grid.addWidget(self.heater_keep_enabled, 4, 0, 1, 4)
        grid.setColumnStretch(6, 1)
        page.layout().addWidget(self.heater_cfg)

        # --- live per-channel numeric readout ---
        self.heater_readout = QtWidgets.QLabel("\N{EM DASH}")
        self.heater_readout.setAlignment(QtCore.Qt.AlignCenter)
        self.heater_readout.setStyleSheet(
            "font-size: 20px; font-weight: 600; color: #88c0d0; padding: 6px;"
        )
        page.layout().addWidget(self._panel_with_widget("Live readout", self.heater_readout))

        # --- controls ---
        ctrl = QtWidgets.QHBoxLayout()
        self.heater_run_button = QtWidgets.QPushButton("Ramp & Hold")
        self.heater_run_button.clicked.connect(self._heater_ramp)
        self.heater_monitor_button = QtWidgets.QPushButton("Monitor")
        self.heater_monitor_button.setToolTip("Read-only live monitor (does not change the drive)")
        self.heater_monitor_button.clicked.connect(self._heater_monitor)
        self.heater_stop_button = QtWidgets.QPushButton("Stop")
        self.heater_stop_button.setProperty("variant", "danger")
        self.heater_stop_button.setEnabled(False)
        self.heater_stop_button.clicked.connect(self._heater_stop)
        self.heater_disable_button = QtWidgets.QPushButton("Disable now")
        self.heater_disable_button.setProperty("variant", "ghost")
        self.heater_disable_button.setToolTip("Force EN=0 on the selected channels (drive off)")
        self.heater_disable_button.clicked.connect(self._heater_disable_now)
        self.heater_save_button = QtWidgets.QPushButton("Save CSV…")
        self.heater_save_button.setProperty("variant", "ghost")
        self.heater_save_button.setEnabled(False)
        self.heater_save_button.clicked.connect(self._heater_save)
        for b in (self.heater_run_button, self.heater_monitor_button, self.heater_stop_button,
                  self.heater_disable_button, self.heater_save_button):
            ctrl.addWidget(b)
        ctrl.addStretch(1)
        page.layout().addLayout(ctrl)

        self.heater_page_status = QtWidgets.QLabel("Idle. Connect the heater on the Connections page.")
        self.heater_page_status.setWordWrap(True)
        page.layout().addWidget(self.heater_page_status)

        page.layout().addStretch(1)
        self._heater_refresh_buttons()
        return page

    def _heater_selected_channels(self) -> list[int]:
        chs = []
        if self.heater_ch1_check.isChecked():
            chs.append(1)
        if self.heater_ch2_check.isChecked():
            chs.append(2)
        return chs

    def _heater_settings(self) -> StaircaseSettings:
        return StaircaseSettings(
            target=self.heater_target.value(),
            step=self.heater_step.value(),
            rest=self.heater_rest.value(),
            railmax=self.heater_railmax.value(),
            vmax=self.heater_vmax.value() if self.heater_vmax_check.isChecked() else None,
            period=self.heater_period.value(),
        )

    def _heater_pid_map(self, channels: list[int]) -> dict[int, dict[str, float]]:
        pid = {"kp": self.heater_kp.value(), "ti": self.heater_ti.value(),
               "td": self.heater_td.value()}
        return {ch: dict(pid) for ch in channels}

    def _heater_ready(self) -> TC300Controller | None:
        heater = self.heater_controller
        if heater is None or not heater.is_connected:
            self._log("Connect the heater first (Connections page)")
            self.heater_page_status.setText("Connect the heater on the Connections page first.")
            return None
        return heater

    def _heater_has_data(self) -> bool:
        return any(self._heater_times[ch] for ch in (1, 2))

    def _heater_refresh_buttons(self) -> None:
        connected = self.heater_controller is not None
        running = self.heater_stop_event is not None
        self.heater_run_button.setEnabled(connected and not running)
        self.heater_monitor_button.setEnabled(connected and not running)
        self.heater_disable_button.setEnabled(connected and not running)
        self.heater_stop_button.setEnabled(connected and running)
        self.heater_cfg.setEnabled(not running)   # freeze params while a loop runs
        self.heater_save_button.setEnabled(not running and self._heater_has_data())

    def _heater_reset_history(self) -> None:
        for ch in (1, 2):
            self._heater_times[ch] = []
            self._heater_temps[ch] = []
            self._heater_volts[ch] = []
        self.heater_readout.setText("\N{EM DASH}")

    def _heater_ramp(self) -> None:
        heater = self._heater_ready()
        if heater is None:
            return
        channels = self._heater_selected_channels()
        if not channels:
            self.heater_page_status.setText("Select at least one channel.")
            return
        settings = self._heater_settings()
        pid_map = self._heater_pid_map(channels)
        disable_on_exit = not self.heater_keep_enabled.isChecked()
        stop_event = threading.Event()
        self.heater_stop_event = stop_event
        self._heater_reset_history()
        self._heater_refresh_buttons()
        chlbl = "+".join(f"CH{c}" for c in channels)
        self.heater_page_status.setText(f"Ramping {chlbl} to {settings.target:.2f} \N{DEGREE SIGN}C…")

        def work() -> dict[str, Any]:
            heater.run_staircase(
                channels, settings, pid_map=pid_map,
                sample_cb=self.heater_sample.emit,
                stop_event=stop_event, disable_on_exit=disable_on_exit,
            )
            return {"mode": "ramp", "kept_enabled": not disable_on_exit}

        self._run_task("Heater ramp", work, self._heater_run_done, self._heater_run_error)

    def _heater_monitor(self) -> None:
        heater = self._heater_ready()
        if heater is None:
            return
        channels = self._heater_selected_channels() or [1, 2]
        period = self.heater_period.value()
        stop_event = threading.Event()
        self.heater_stop_event = stop_event
        self._heater_reset_history()
        self._heater_refresh_buttons()
        chlbl = "+".join(f"CH{c}" for c in channels)
        self.heater_page_status.setText(f"Monitoring {chlbl} (read-only)…")

        def work() -> dict[str, Any]:
            heater.monitor(
                channels, sample_cb=self.heater_sample.emit,
                stop_event=stop_event, period=period,
            )
            return {"mode": "monitor"}

        self._run_task("Heater monitor", work, self._heater_run_done, self._heater_run_error)

    def _heater_stop(self) -> None:
        if self.heater_stop_event is not None:
            self.heater_stop_event.set()
            self.heater_page_status.setText("Stopping…")

    def _heater_disable_now(self) -> None:
        heater = self._heater_ready()
        if heater is None:
            return
        channels = self._heater_selected_channels() or [1, 2]

        def work() -> dict[str, Any]:
            for ch in channels:
                heater.disable(ch)
            return {"channels": channels}

        self._run_task("Heater disable", work,
                       lambda _p: self.heater_page_status.setText(
                           "Selected channels disabled (drive off)."))

    def _heater_run_done(self, payload: dict[str, Any]) -> None:
        self.heater_stop_event = None
        self._heater_refresh_buttons()
        n = sum(len(self._heater_times[ch]) for ch in (1, 2))
        if (payload or {}).get("kept_enabled"):
            self.heater_page_status.setText(
                f"Stopped — channels left enabled (still holding). {n} points recorded.")
        else:
            self.heater_page_status.setText(f"Stopped. {n} points recorded.")
        if self._heater_disconnect_pending:
            self._heater_do_disconnect()

    def _heater_run_error(self, _error: str) -> None:
        self.heater_stop_event = None
        self._heater_refresh_buttons()
        self.heater_page_status.setText("Heater loop failed (see Status log).")
        if self._heater_disconnect_pending:
            self._heater_do_disconnect()

    def _on_heater_sample(self, cycle: HeaterCycle) -> None:
        """Append one live cycle and redraw (GUI thread, via heater_sample)."""
        keep = int(self.heater_history.value())
        parts = []
        for ch in sorted(cycle.channels):
            s = cycle.channels[ch]
            self._heater_times[ch].append(cycle.elapsed)
            self._heater_temps[ch].append(s.temp if s.temp is not None else float("nan"))
            self._heater_volts[ch].append(s.volt)
            if len(self._heater_times[ch]) > keep:
                self._heater_times[ch] = self._heater_times[ch][-keep:]
                self._heater_temps[ch] = self._heater_temps[ch][-keep:]
                self._heater_volts[ch] = self._heater_volts[ch][-keep:]
            tstr = f"{s.temp:.3f}" if s.temp is not None else "\N{EN DASH}"
            cstr = f"{s.curr:.0f}" if s.curr is not None else "\N{EN DASH}"
            parts.append(f"CH{ch}  {tstr} \N{DEGREE SIGN}C   {s.volt:.2f} V   {cstr} mA")
        err = cycle.err or "0"
        if err not in ("", "0"):
            parts.append(f"ERR={err}")
        self.heater_readout.setText("      ".join(parts))

    def _heater_save(self) -> None:
        if not self._heater_has_data():
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save heater monitor log", "heater_monitor.csv", "CSV (*.csv)")
        if not path:
            return
        try:
            import csv

            with open(path, "w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["channel", "elapsed_s", "temp_C", "volt_V"])
                for ch in (1, 2):
                    for t, temp, volt in zip(self._heater_times[ch], self._heater_temps[ch],
                                             self._heater_volts[ch]):
                        writer.writerow([ch, f"{t:.3f}", f"{temp:.6g}", f"{volt:.6g}"])
            self._log(f"Saved heater monitor log → {path}")
        except Exception as exc:
            self._log(f"Save failed: {exc}")

    # ===================== DAQ live monitor page ========================
    # ------------------------------------------------------------------- PSU
    def _connect_psu(self) -> None:
        port = self.psu_port_edit.text().strip()
        if not port:
            self._log("Enter the PSU serial port first")
            return
        self.psu_connect_button.setEnabled(False)

        def connect() -> tuple[PSUController, str, PSUState]:
            # Caps are the PMT module's spec, not the supply's 60 V rating: the
            # controller refuses any setpoint above them, and refuses to enable
            # the output if the front panel left one there.
            psu = PSUController(port=port, max_voltage=PMT_CONTROL_V_MAX,
                                max_current=PMT_CONTROL_I_MAX)
            psu.connect()
            return psu, psu.identify(), psu.read_state()

        self._run_task("Connect PSU", connect, self._on_psu_connected,
                       self._on_psu_connect_error)

    def _on_psu_connected(self, payload: tuple[PSUController, str, PSUState]) -> None:
        psu, identity, state = payload
        self.psu_controller = psu
        self._set_status(self.psu_status_label, "PSU: open", "ok")
        self.psu_connect_button.setEnabled(False)
        self.psu_disconnect_button.setEnabled(True)
        self._log(f"PSU connected: {identity.strip()}")
        # Adopt the supply's live setpoints rather than imposing the spinbox
        # defaults: the PMT may already be running at a gain someone chose.
        self._psu_show_state(state, adopt=True)
        self._psu_refresh_buttons()
        self.psu_page_status.setText(
            f"Connected. Output is {'ON' if state.output_on else 'off'}; "
            f"setpoint {state.vset:.3f} V, limit {state.iset:.3f} A."
        )

    def _on_psu_connect_error(self, _error: str) -> None:
        self._set_status(self.psu_status_label, "PSU: error", "error")
        self.psu_connect_button.setEnabled(True)

    def _disconnect_psu(self) -> None:
        # A running monitor would raise when the port closes under it, so stop
        # it first and finish the disconnect once the loop unwinds.
        if self.psu_stop_event is not None:
            self._psu_disconnect_pending = True
            self.psu_stop_event.set()
            self._set_status(self.psu_status_label, "PSU: stopping\N{HORIZONTAL ELLIPSIS}", "off")
            self.psu_page_status.setText(
                "Stopping the monitor, then disconnecting\N{HORIZONTAL ELLIPSIS}")
            return
        self._psu_do_disconnect()

    def _psu_do_disconnect(self) -> None:
        psu = self.psu_controller
        self.psu_controller = None
        self.psu_state = None
        self._psu_disconnect_pending = False
        self._set_status(self.psu_status_label, "PSU: closed", "off")
        self.psu_connect_button.setEnabled(True)
        self.psu_disconnect_button.setEnabled(False)
        self._psu_refresh_buttons()
        if psu is not None:
            # Leaves the output exactly as it is: closing a port must not change
            # the PMT gain in the middle of a measurement.
            self._run_task("Disconnect PSU", psu.disconnect)
            self.psu_page_status.setText(
                "Disconnected. The output was left exactly as it was.")

    def _psu_ready(self) -> PSUController | None:
        psu = self.psu_controller
        if psu is None or not psu.is_connected:
            self.psu_page_status.setText(
                "Connect the power supply on the Connections page first.")
            return None
        return psu

    # ---------------------------------------------------------------- PSU page
    def _build_psu_page(self) -> QtWidgets.QWidget:
        """KWR103 as a constant-voltage source for the PMT gain-control input.

        Constant voltage is the only mode this page offers. The control input is
        high impedance, so the current limit is a fault detector rather than an
        operating parameter: if the mode pill ever reads CC, something is
        shorted or mis-wired, not merely drawing more. Gain rises steeply with
        control voltage, hence mV steps and a spinbox that stops at the module
        spec (1.1 V) -- with the controller refusing anything above it even if
        the front panel was left higher.
        """
        page = self._page_shell("PMT Gain \N{EM DASH} Control Voltage")

        # --- setpoints ---
        self.psu_cfg = self._panel("Setpoints (constant voltage)")
        grid = QtWidgets.QGridLayout(self.psu_cfg)
        self.psu_vset = QtWidgets.QDoubleSpinBox()
        self.psu_vset.setRange(0.0, PMT_CONTROL_V_MAX)
        self.psu_vset.setDecimals(3)
        self.psu_vset.setSingleStep(0.005)     # gain is steep in this voltage
        self.psu_vset.setSuffix(" V")
        self.psu_vset.setValue(0.5)
        self.psu_iset = QtWidgets.QDoubleSpinBox()
        self.psu_iset.setRange(0.0, PMT_CONTROL_I_MAX)
        self.psu_iset.setDecimals(3)
        self.psu_iset.setSingleStep(0.005)
        self.psu_iset.setSuffix(" A")
        self.psu_iset.setValue(PMT_CONTROL_I_MAX)
        self.psu_apply_button = QtWidgets.QPushButton("Apply setpoints")
        self.psu_apply_button.setToolTip(
            "Write both setpoints and read them back; a value the supply does "
            "not take is reported as an error instead of passing silently")
        self.psu_apply_button.clicked.connect(self._psu_apply)
        self.psu_read_button = QtWidgets.QPushButton("Read from supply")
        self.psu_read_button.setProperty("variant", "ghost")
        self.psu_read_button.clicked.connect(self._psu_read)
        grid.addWidget(QtWidgets.QLabel("Control voltage"), 0, 0)
        grid.addWidget(self.psu_vset, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Current limit"), 0, 2)
        grid.addWidget(self.psu_iset, 0, 3)
        grid.addWidget(self.psu_apply_button, 0, 4)
        grid.addWidget(self.psu_read_button, 0, 5)
        note = QtWidgets.QLabel(
            f"PMT gain rises steeply with control voltage; the spinbox stops at "
            f"the module spec of {PMT_CONTROL_V_MAX:g} V. The current limit is a "
            f"fault detector \N{EM DASH} the control input should draw microamps."
        )
        note.setWordWrap(True)
        grid.addWidget(note, 1, 0, 1, 6)
        grid.setColumnStretch(6, 1)
        page.layout().addWidget(self.psu_cfg)

        # --- output ---
        out = self._panel("Output")
        outl = QtWidgets.QHBoxLayout(out)
        self.psu_on_button = QtWidgets.QPushButton("Output On")
        self.psu_on_button.clicked.connect(self._psu_output_on)
        self.psu_off_button = QtWidgets.QPushButton("Output Off")
        self.psu_off_button.setProperty("variant", "ghost")
        self.psu_off_button.clicked.connect(self._psu_output_off)
        self.psu_output_label = QtWidgets.QLabel("Output: unknown")
        self._set_status(self.psu_output_label, "Output: unknown", "off")
        self.psu_mode_label = QtWidgets.QLabel("Mode: \N{EN DASH}")
        self._set_status(self.psu_mode_label, "Mode: \N{EN DASH}", "off")
        self.psu_prot_label = QtWidgets.QLabel(
            "OVP \N{EN DASH}  \N{BULLET}  OCP \N{EN DASH}")
        outl.addWidget(self.psu_on_button)
        outl.addWidget(self.psu_off_button)
        outl.addWidget(self.psu_output_label)
        outl.addWidget(self.psu_mode_label)
        outl.addWidget(self.psu_prot_label)
        outl.addStretch(1)
        page.layout().addWidget(out)

        # --- measured output (numbers only) ---
        self.psu_readout = QtWidgets.QLabel("\N{EN DASH}")
        self.psu_readout.setAlignment(QtCore.Qt.AlignCenter)
        self.psu_readout.setStyleSheet(
            "font-size: 30px; font-weight: 600; color: #88c0d0; padding: 6px;"
        )
        page.layout().addWidget(
            self._panel_with_widget("Measured output", self.psu_readout))

        ctrl = QtWidgets.QHBoxLayout()
        self.psu_monitor_button = QtWidgets.QPushButton("Monitor")
        self.psu_monitor_button.setCheckable(True)
        self.psu_monitor_button.setToolTip(
            "Poll the supply read-only until stopped. Setpoints and the output "
            "stay usable while it runs: the driver locks the port per exchange")
        self.psu_monitor_button.clicked.connect(self._psu_monitor_toggled)
        self.psu_monitor_stop_button = QtWidgets.QPushButton("Stop")
        self.psu_monitor_stop_button.setProperty("variant", "ghost")
        self.psu_monitor_stop_button.clicked.connect(self._psu_monitor_stop)
        self.psu_period = QtWidgets.QDoubleSpinBox()
        self.psu_period.setRange(0.1, 10.0)
        self.psu_period.setDecimals(1)
        self.psu_period.setSingleStep(0.1)
        self.psu_period.setValue(0.5)
        self.psu_period.setSuffix(" s")
        ctrl.addWidget(self.psu_monitor_button)
        ctrl.addWidget(self.psu_monitor_stop_button)
        ctrl.addWidget(QtWidgets.QLabel("Interval"))
        ctrl.addWidget(self.psu_period)
        ctrl.addStretch(1)
        page.layout().addLayout(ctrl)

        self.psu_page_status = QtWidgets.QLabel(
            "Idle. Connect the power supply on the Connections page.")
        self.psu_page_status.setWordWrap(True)
        page.layout().addWidget(self.psu_page_status)
        page.layout().addStretch(1)
        self._psu_refresh_buttons()
        return page

    def _psu_refresh_buttons(self) -> None:
        connected = self.psu_controller is not None
        running = self.psu_stop_event is not None
        # Apply and the output stay live during a monitor run: the port is
        # locked per exchange, so a setpoint cannot land inside a poll.
        free = connected and not self._psu_busy
        self.psu_apply_button.setEnabled(free)
        self.psu_read_button.setEnabled(free)
        self.psu_on_button.setEnabled(free)
        self.psu_off_button.setEnabled(free)
        self.psu_monitor_button.setEnabled(free and not running)
        self.psu_monitor_stop_button.setEnabled(running)
        self.psu_period.setEnabled(not running)   # the loop takes it at start

    # ------------------------------------------------------------ PSU commands
    def _psu_apply(self) -> None:
        psu = self._psu_ready()
        if psu is None:
            return
        volts, amps = self.psu_vset.value(), self.psu_iset.value()
        self._psu_busy = True
        self._psu_refresh_buttons()
        self.psu_page_status.setText(
            f"Applying {volts:.3f} V / {amps:.3f} A\N{HORIZONTAL ELLIPSIS}")

        def work() -> PSUState:
            psu.set_current(amps)    # limit first, so the new voltage is always
            psu.set_voltage(volts)   # governed by the new limit
            return psu.read_state()

        self._run_task("PSU apply setpoints", work, self._psu_command_done,
                       self._psu_command_error)

    def _psu_read(self) -> None:
        psu = self._psu_ready()
        if psu is None:
            return
        self._psu_busy = True
        self._psu_refresh_buttons()
        self._run_task("PSU read state", psu.read_state,
                       self._psu_command_done, self._psu_command_error)

    def _psu_output_on(self) -> None:
        psu = self._psu_ready()
        if psu is None:
            return
        state = self.psu_state
        vset = state.vset if state is not None else self.psu_vset.value()
        iset = state.iset if state is not None else self.psu_iset.value()
        lines = [
            f"Apply {vset:.3f} V (limit {iset:.3f} A) to the PMT gain-control "
            f"input?",
            "",
            "PMT gain rises steeply with control voltage.",
        ]
        # The dialog quotes the supply, not the spinbox -- what it will actually
        # put out is the setpoint the hardware holds.
        if abs(self.psu_vset.value() - vset) > SET_TOL_V:
            lines.append(
                f"The spinbox shows {self.psu_vset.value():.3f} V, which is not "
                f"applied \N{EM DASH} press Apply first if that is what you want."
            )
        if state is not None and state.ovp <= vset:
            lines.append(
                f"OVP is {state.ovp:.3f} V, at or below the setpoint: the supply "
                f"may trip the moment the output goes live."
            )
        answer = QtWidgets.QMessageBox.question(
            self, "Enable PSU output", "\n".join(lines),
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            self.psu_page_status.setText("Output left off.")
            return
        self._psu_busy = True
        self._psu_refresh_buttons()

        def work() -> PSUState:
            psu.enable()             # re-checks the live setpoints against the caps
            return psu.read_state()

        self._run_task("PSU output on", work, self._psu_command_done,
                       self._psu_command_error)

    def _psu_output_off(self) -> None:
        psu = self._psu_ready()
        if psu is None:
            return
        self._psu_busy = True
        self._psu_refresh_buttons()

        def work() -> PSUState:
            psu.disable()            # no confirmation: off is the safe direction
            return psu.read_state()

        self._run_task("PSU output off", work, self._psu_command_done,
                       self._psu_command_error)

    def _psu_command_done(self, state: PSUState | None) -> None:
        self._psu_busy = False
        if state is not None:
            self._psu_show_state(state, adopt=True)
            self.psu_page_status.setText(self._psu_state_summary(state))
        self._psu_refresh_buttons()

    def _psu_command_error(self, _error: str) -> None:
        self._psu_busy = False
        self._psu_refresh_buttons()
        self.psu_page_status.setText("PSU command failed (see the Status log).")

    # -------------------------------------------------------------- PSU monitor
    def _psu_monitor_toggled(self) -> None:
        if not self.psu_monitor_button.isChecked():
            self._psu_monitor_stop()          # toggled off -> treat as stop
            return
        psu = self._psu_ready()
        if psu is None:
            self.psu_monitor_button.setChecked(False)
            return
        period = self.psu_period.value()
        stop_event = threading.Event()
        self.psu_stop_event = stop_event
        self._psu_refresh_buttons()
        self.psu_page_status.setText(
            f"Monitoring every {period:g} s (read-only)\N{HORIZONTAL ELLIPSIS}")

        def work() -> dict[str, Any]:
            psu.monitor(sample_cb=self.psu_sample.emit, stop_event=stop_event,
                        period=period, full=True)
            return {"mode": "monitor"}

        self._run_task("PSU monitor", work, self._psu_monitor_done,
                       self._psu_monitor_error)

    def _psu_monitor_stop(self) -> None:
        if self.psu_stop_event is not None:
            self.psu_stop_event.set()
            self.psu_page_status.setText(
                "Stopping the monitor\N{HORIZONTAL ELLIPSIS}")

    def _psu_monitor_done(self, _payload: Any = None) -> None:
        self.psu_stop_event = None
        self.psu_monitor_button.setChecked(False)
        self._psu_refresh_buttons()
        if self._psu_disconnect_pending:
            self._psu_do_disconnect()
            return
        self.psu_page_status.setText("Monitor stopped.")

    def _psu_monitor_error(self, _error: str) -> None:
        self.psu_stop_event = None
        self.psu_monitor_button.setChecked(False)
        self._psu_refresh_buttons()
        if self._psu_disconnect_pending:
            self._psu_do_disconnect()
            return
        self.psu_page_status.setText("PSU monitor failed (see the Status log).")

    def _on_psu_sample(self, sample: Any) -> None:
        """One live snapshot from the monitor worker (GUI thread, queued)."""
        if isinstance(sample, PSUState):
            self._psu_show_state(sample)      # never adopt: the user may be typing
        else:
            self._psu_paint_reading(sample)

    # --------------------------------------------------------------- PSU paint
    def _psu_show_state(self, state: PSUState, *, adopt: bool = False) -> None:
        """Paint a full snapshot. ``adopt`` also pulls the setpoints into the
        spinboxes -- for connect / apply / an explicit read, never for a poll,
        which must not move a box the user is editing."""
        self.psu_state = state
        if adopt:
            for spin, value in ((self.psu_vset, state.vset),
                                (self.psu_iset, state.iset)):
                spin.blockSignals(True)
                spin.setValue(min(value, spin.maximum()))
                spin.blockSignals(False)
        self._psu_paint_reading(state)
        prot = f"OVP {state.ovp:.3f} V  \N{BULLET}  OCP {state.ocp:.3f} A"
        if state.ovp <= state.vset:
            # Would trip the instant the output went live; say so rather than
            # leaving it as a mystery at the front panel.
            prot += f"  \N{BULLET}  OVP \N{LESS-THAN OR EQUAL TO} setpoint, would trip"
        self.psu_prot_label.setText(prot)

    def _psu_paint_reading(self, sample: Any) -> None:
        """Readout + pills from either a PSUReading or a full PSUState."""
        self.psu_readout.setText(
            f"{sample.volts:.3f} V      {sample.amps * 1e3:.1f} mA"
            f"      {sample.watts * 1e3:.1f} mW"
        )
        self._set_status(self.psu_output_label,
                         "Output: ON" if sample.output_on else "Output: off",
                         "ok" if sample.output_on else "off")
        limited = getattr(sample, "current_limited", None)
        if not sample.output_on:
            self._set_status(self.psu_mode_label,
                             "Mode: \N{EN DASH} (output off)", "off")
        elif limited is None:
            pass                     # a fast reading carries no CV/CC verdict
        elif limited:
            self._set_status(self.psu_mode_label, "Mode: CC \N{EM DASH} check wiring",
                             "error")
        else:
            self._set_status(self.psu_mode_label, "Mode: CV", "ok")

    @staticmethod
    def _psu_state_summary(state: PSUState) -> str:
        mode = "output off" if not state.output_on else (
            "CC \N{EM DASH} the limit is binding" if state.current_limited else "CV")
        return (
            f"Setpoint {state.vset:.3f} V, limit {state.iset:.3f} A \N{BULLET} "
            f"measured {state.volts:.3f} V / {state.amps * 1e3:.1f} mA \N{BULLET} "
            f"{mode}."
        )

    def _build_daq_monitor_page(self) -> QtWidgets.QWidget:
        """Dedicated page: DAQ waveform diagnostic (time + spectrum) + live readout.

        Mirrors ``src/drafts/daq_read_waveform.py``: one untriggered finite
        acquisition, sign-inverted (the TIA outputs negative volts for
        positive light), digitally low-passed, with the leading
        settle-cycles/f_cut turn-on transient discarded before any
        statistics. Replaces the deprecated live strip-chart monitor -- the
        bring-up view is the raw trace and its amplitude spectrum, not a
        scrolling voltmeter.

        Two modes over the same reduction (``waveform_stats``): *Read waveform*
        takes one ``Duration`` acquisition and plots it, while *Continuous*
        loops back-to-back ``History`` acquisitions and reports only the
        reading, leaving the plot on the last one-shot trace. Connect the DAQ
        on the Connections page first.
        """
        page = self._page_shell("DAQ Monitor")
        subtitle = QtWidgets.QLabel(
            "Waveform diagnostic of the NI-DAQ analog input (the "
            "daq_read_waveform draft view): one finite acquisition, "
            "sign-inverted and low-passed, warmup dropped before the stats. "
            "Continuous repeats independent History-long acquisitions and "
            "reports mean ± std for each. "
            "Connect the DAQ on the Connections page first."
        )
        subtitle.setObjectName("PageSubtitle")
        subtitle.setWordWrap(True)
        page.layout().addWidget(subtitle)

        # --- acquisition parameters (defaults mirror the draft) ---
        self.daqmon_cfg = self._panel("Acquisition")
        grid = QtWidgets.QGridLayout(self.daqmon_cfg)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        self.daqmon_channel = QtWidgets.QLineEdit("ai0")
        self.daqmon_channel.setMaximumWidth(90)
        self.daqmon_sample_rate = QtWidgets.QDoubleSpinBox()
        self.daqmon_sample_rate.setRange(1.0, 2_000_000.0)
        self.daqmon_sample_rate.setDecimals(0)
        self.daqmon_sample_rate.setValue(1000.0)
        self.daqmon_sample_rate.setSuffix(" S/s")
        self.daqmon_range = QtWidgets.QComboBox()
        for lo, hi in self._DAQ_RANGES:
            self.daqmon_range.addItem(f"\N{PLUS-MINUS SIGN}{hi:g} V", (lo, hi))
        self.daqmon_range.setCurrentIndex(0)   # smallest / most sensitive by default
        self.daqmon_duration = QtWidgets.QDoubleSpinBox()
        self.daqmon_duration.setRange(0.001, 60.0)
        self.daqmon_duration.setDecimals(3)
        self.daqmon_duration.setValue(10.0)
        self.daqmon_duration.setSuffix(" s")
        self.daqmon_duration.setToolTip("One finite acquisition of this length")
        self.daqmon_history = QtWidgets.QDoubleSpinBox()
        self.daqmon_history.setRange(0.05, 60.0)
        self.daqmon_history.setDecimals(3)
        self.daqmon_history.setSingleStep(0.5)
        self.daqmon_history.setValue(1.0)
        self.daqmon_history.setSuffix(" s")
        self.daqmon_history.setToolTip(
            "Continuous mode: averaging window per reported reading. Each "
            "reading is an independent acquisition of this length (the warmup "
            "guard is acquired on top, so the full window is averaged). Read "
            "live every loop, so it can be retuned while running."
        )
        self.daqmon_fcut_hw = QtWidgets.QDoubleSpinBox()
        self.daqmon_fcut_hw.setRange(0.1, 100_000.0)
        self.daqmon_fcut_hw.setDecimals(1)
        self.daqmon_fcut_hw.setValue(150.0)
        self.daqmon_fcut_hw.setSuffix(" Hz")
        self.daqmon_fcut_hw.setToolTip(
            "Analog 3 dB bandwidth already in the signal chain (detector/TIA) "
            "-- physics, not a filter applied here; drawn on the spectrum and "
            "used for the settle guard when it is the narrower cutoff"
        )
        self.daqmon_fcut_dig = QtWidgets.QDoubleSpinBox()
        self.daqmon_fcut_dig.setRange(0.1, 100_000.0)
        self.daqmon_fcut_dig.setDecimals(1)
        self.daqmon_fcut_dig.setValue(20.0)
        self.daqmon_fcut_dig.setSuffix(" Hz")
        self.daqmon_fcut_dig.setToolTip(
            "Digital Butterworth low-pass applied after acquisition "
            "(typically \N{LESS-THAN OR EQUAL TO} hardware bandwidth to "
            "reject more noise)"
        )
        self.daqmon_invert = QtWidgets.QCheckBox(
            "Invert (TIA: \N{MINUS SIGN}V \N{RIGHTWARDS ARROW} +light)")
        self.daqmon_invert.setChecked(True)
        self.daqmon_invert.setToolTip(
            "The transimpedance amplifier outputs negative volts for positive "
            "light; inverting recovers a positive light signal"
        )
        pairs = [("Channel", self.daqmon_channel), ("Sample rate", self.daqmon_sample_rate),
                 ("Range", self.daqmon_range), ("Duration", self.daqmon_duration),
                 ("HW bandwidth", self.daqmon_fcut_hw), ("Digital low-pass", self.daqmon_fcut_dig),
                 ("History (continuous)", self.daqmon_history)]
        for i, (label, widget) in enumerate(pairs):
            r, c = i // 3, (i % 3) * 2
            grid.addWidget(QtWidgets.QLabel(label), r, c)
            grid.addWidget(widget, r, c + 1)
        grid.addWidget(self.daqmon_invert, 3, 0, 1, 4)   # row 2 now holds History
        grid.setColumnStretch(6, 1)   # absorb leftover width instead of stretching fields
        page.layout().addWidget(self.daqmon_cfg)

        # Continuous mode takes its parameters from this snapshot, never from
        # the widgets themselves: its loop runs on a worker thread, which must
        # not touch Qt objects. Every control rebinds the snapshot on the GUI
        # thread and the worker reads it once per acquisition, so a rebind is
        # atomic and the loop always sees one consistent set of values. That is
        # what lets the controls stay live while streaming -- retuning History
        # or the range takes effect on the next reading.
        self._daqmon_stream_cfg: dict[str, Any] = {}
        for widget, signal in (
            (self.daqmon_channel, "textChanged"),
            (self.daqmon_sample_rate, "valueChanged"),
            (self.daqmon_range, "currentIndexChanged"),
            (self.daqmon_duration, "valueChanged"),
            (self.daqmon_history, "valueChanged"),
            (self.daqmon_fcut_hw, "valueChanged"),
            (self.daqmon_fcut_dig, "valueChanged"),
            (self.daqmon_invert, "toggled"),
        ):
            getattr(widget, signal).connect(self._daqmon_sync_cfg)
        self._daqmon_sync_cfg()

        # --- big numeric readout of the last read ---
        self.daqmon_readout = QtWidgets.QLabel("\N{EN DASH}")
        self.daqmon_readout.setAlignment(QtCore.Qt.AlignCenter)
        self.daqmon_readout.setStyleSheet(
            "font-size: 30px; font-weight: 600; color: #88c0d0; padding: 6px;"
        )
        page.layout().addWidget(
            self._panel_with_widget("Filtered mean \N{PLUS-MINUS SIGN} std",
                                    self.daqmon_readout))

        # --- controls ---
        ctrl = QtWidgets.QHBoxLayout()
        self.daqmon_read_button = QtWidgets.QPushButton("Read waveform")
        self.daqmon_read_button.clicked.connect(self._daq_waveform_read)
        self.daqmon_cont_button = QtWidgets.QPushButton("Continuous")
        self.daqmon_cont_button.setCheckable(True)
        self.daqmon_cont_button.setToolTip(
            "Report mean ± std every History seconds until "
            "stopped (readout only; the plot keeps the last one-shot trace)"
        )
        self.daqmon_cont_button.clicked.connect(self._daq_stream_toggled)
        self.daqmon_stop_button = QtWidgets.QPushButton("Stop")
        self.daqmon_stop_button.setProperty("variant", "ghost")
        self.daqmon_stop_button.setEnabled(False)
        self.daqmon_stop_button.clicked.connect(self._daq_stream_stop)
        ctrl.addWidget(self.daqmon_read_button)
        ctrl.addWidget(self.daqmon_cont_button)
        ctrl.addWidget(self.daqmon_stop_button)
        ctrl.addStretch(1)
        page.layout().addLayout(ctrl)

        self.daqmon_status = QtWidgets.QLabel(
            "Idle. Connect the DAQ, then press Read waveform.")
        self.daqmon_status.setWordWrap(True)
        page.layout().addWidget(self.daqmon_status)

        # --- time domain + amplitude spectrum ---
        self.daqmon_fig = Figure(figsize=(10, 6.0), tight_layout=True)
        self.daqmon_canvas = FigureCanvas(self.daqmon_fig)
        self.daqmon_canvas.setMinimumHeight(360)
        plot_panel = self._panel(
            "Last one-shot acquisition — waveform + amplitude spectrum")
        plot_layout = QtWidgets.QVBoxLayout(plot_panel)
        plot_layout.addWidget(self.daqmon_canvas, 1)
        page.layout().addWidget(plot_panel, 1)

        self._daq_waveform_draw(None)
        return page

    # The warmup guard (SETTLE_CYCLES / min(f_hw, f_dig)) and the
    # invert -> low-pass -> drop-warmup -> mean/std reduction both live in
    # daq_module.controller, shared by the one-shot read, the continuous loop
    # and the unit tests.

    @staticmethod
    def _daqmon_spectrum(v: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
        """Single-sided Hann-windowed amplitude spectrum, DC bin dropped."""
        n = int(v.size)
        if n < 2 or fs <= 0.0:
            return np.array([]), np.array([])
        win = np.hanning(n)
        scale = 2.0 / np.sum(win)   # coherent gain -> single-sided amplitude in volts
        spec = np.abs(np.fft.rfft(v * win)) * scale
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        return freqs[1:], spec[1:]  # drop DC bin

    def _daqmon_cfg_snapshot(self) -> dict[str, Any]:
        """Plain-value copy of the acquisition controls (GUI thread only)."""
        min_val, max_val = self.daqmon_range.currentData()
        return {
            "channel": self.daqmon_channel.text().strip() or "ai0",
            "fs": float(self.daqmon_sample_rate.value()),
            "min_val": min_val,
            "max_val": max_val,
            "duration": float(self.daqmon_duration.value()),
            "history": float(self.daqmon_history.value()),
            "f_hw": float(self.daqmon_fcut_hw.value()),
            "f_dig": float(self.daqmon_fcut_dig.value()),
            "invert": self.daqmon_invert.isChecked(),
            "order": DAQMonitorSettings().filter_order,
        }

    def _daqmon_sync_cfg(self, *_args: Any) -> None:
        """Republish the snapshot the continuous loop reads (see the builder)."""
        self._daqmon_stream_cfg = self._daqmon_cfg_snapshot()

    def _daqmon_streaming(self) -> bool:
        return self.daq_stream_stop_event is not None

    def _daqmon_set_running(self, *, continuous: bool = False) -> None:
        """Gate the buttons on the card's one owner (read, loop or calibration)."""
        busy = self._calibration_busy()
        streaming = self._daqmon_streaming()
        idle = not busy and not self._daqmon_reading and not streaming
        self.daqmon_read_button.setEnabled(idle)
        self.daqmon_cont_button.setChecked(streaming and continuous)
        self.daqmon_cont_button.setEnabled(idle or (streaming and continuous))
        self.daqmon_stop_button.setEnabled(streaming)
        # A one-shot read freezes its parameters; the continuous loop re-reads
        # them per acquisition, so they stay editable while it runs.
        self.daqmon_cfg.setEnabled(not self._daqmon_reading)

    def _daq_waveform_read(self) -> None:
        if self._calibration_busy():
            self.daqmon_status.setText("A calibration is running; DAQ reads are locked.")
            return
        if self._daqmon_streaming():
            self.daqmon_status.setText("Continuous mode is running; press Stop first.")
            return
        daq = self._daq_ready()
        if daq is None:
            self.daqmon_status.setText("Connect the DAQ on the Connections page first.")
            return
        cfg = self._daqmon_cfg_snapshot()
        duration = cfg["duration"]
        self._daqmon_reading = True
        self._daqmon_set_running()
        self.daqmon_status.setText(
            f"Reading {duration:g} s @ {cfg['fs']:g} S/s\N{HORIZONTAL ELLIPSIS}")

        def work() -> dict[str, Any]:
            return self._daqmon_acquire(daq, cfg, duration)

        self._run_task("DAQ waveform", work,
                       self._daq_waveform_done, self._daq_waveform_error)

    @staticmethod
    def _daqmon_acquire(daq: DAQController, cfg: dict[str, Any],
                        duration: float) -> dict[str, Any]:
        """One acquisition reduced to a reading (worker thread; no Qt access).

        ``duration`` is the span that must survive the warmup guard, so the
        request is lengthened by the guard: a 1 s History really averages 1 s.
        """
        fs = cfg["fs"]
        n_guard = settle_samples(fs, cfg["f_hw"], cfg["f_dig"])
        request = duration + n_guard / fs if fs > 0 else duration
        voltages = daq.driver.read_waveform(
            channel=cfg["channel"], sample_rate=fs, duration=request,
            min_val=cfg["min_val"], max_val=cfg["max_val"],
            timeout=request + 10.0,
        )
        stats = waveform_stats(
            voltages, sample_rate=fs, f_hw=cfg["f_hw"], f_dig=cfg["f_dig"],
            invert=cfg["invert"], order=cfg["order"],
        )
        n = stats.raw.size
        return {
            "times": np.arange(n) / fs if fs > 0 else np.zeros(n),
            "raw": stats.raw, "filtered": stats.filtered, "n_samples": n,
            "n_settle": stats.n_settle, "fs": fs,
            "mean": stats.mean, "std": stats.std,
            "raw_mean": stats.raw_mean, "raw_std": stats.raw_std,
            "rel": stats.rel_percent,
            "f_hw": cfg["f_hw"], "f_dig": cfg["f_dig"],
            "duration": request, "kept": duration,
            "channel": cfg["channel"],
        }

    def _daqmon_show_reading(self, payload: dict[str, Any]) -> None:
        """Paint the big mean +/- std readout (shared by both modes)."""
        self.daqmon_readout.setText(
            f"{payload['mean']*1000:+.4f} \N{PLUS-MINUS SIGN} "
            f"{payload['std']*1000:.4f} mV   ({payload['rel']:.2f}%)"
        )

    @staticmethod
    def _daqmon_reading_summary(payload: dict[str, Any]) -> str:
        """The raw-vs-filtered one-liner both modes put in the status label."""
        n0 = int(payload["n_settle"])
        fs = payload["fs"]
        f_eff = min(payload["f_hw"], payload["f_dig"])
        return (
            f"{payload['n_samples']} samples ({payload['duration']:g} s "
            f"@ {fs:g} S/s); dropped {n0} warmup samples "
            f"({n0 / fs * 1000.0:.0f} ms); "
            f"raw {payload['raw_mean']*1000:+.4f} \N{PLUS-MINUS SIGN} "
            f"{payload['raw_std']*1000:.4f} mV, "
            f"filtered ({f_eff:g} Hz) {payload['mean']*1000:+.4f} "
            f"\N{PLUS-MINUS SIGN} {payload['std']*1000:.4f} mV"
        )

    def _daq_waveform_done(self, payload: dict[str, Any]) -> None:
        self._daqmon_reading = False
        self._refresh_manual_reads()
        self._daqmon_show_reading(payload)
        self.daqmon_status.setText(f"Read {self._daqmon_reading_summary(payload)}.")
        self._daq_waveform_draw(payload)

    def _daq_waveform_error(self, _error: str) -> None:
        self._daqmon_reading = False
        self._refresh_manual_reads()
        self.daqmon_status.setText("DAQ read failed (see Status log).")

    # --- continuous mode: back-to-back independent readings ----------------

    def _daq_stream_toggled(self) -> None:
        if not self.daqmon_cont_button.isChecked():
            # toggled off by the user's click -> treat as stop
            self._daq_stream_stop()
            return
        if self._calibration_busy():
            self.daqmon_cont_button.setChecked(False)
            self.daqmon_status.setText("A calibration is running; DAQ reads are locked.")
            return
        if self._daqmon_reading:
            self.daqmon_cont_button.setChecked(False)
            self.daqmon_status.setText("A one-shot read is in flight; try again.")
            return
        daq = self._daq_ready()
        if daq is None:
            self.daqmon_cont_button.setChecked(False)
            self.daqmon_status.setText("Connect the DAQ on the Connections page first.")
            return
        self._daqmon_sync_cfg()
        stop_event = threading.Event()
        self.daq_stream_stop_event = stop_event
        self._daqmon_set_running(continuous=True)
        self.daqmon_status.setText(
            "Continuous\N{HORIZONTAL ELLIPSIS} press Stop to end.")

        def work() -> dict[str, Any]:
            n = 0
            while not stop_event.is_set():
                # Re-read the snapshot every pass: the controls stay live, so a
                # retuned History (or range, or cutoff) applies from here on.
                cfg = self._daqmon_stream_cfg
                payload = self._daqmon_acquire(daq, cfg, cfg["history"])
                if stop_event.is_set():
                    break            # drop a reading the user already stopped
                n += 1
                payload["count"] = n
                for trace in ("times", "raw", "filtered"):
                    payload.pop(trace)     # readout only; see _daqmon_acquire
                self.daq_reading.emit(payload)
            return {"readings": n}

        self._run_task("DAQ continuous", work,
                       self._daq_stream_done, self._daq_stream_error)

    def _on_daq_reading(self, payload: dict[str, Any]) -> None:
        """One continuous reading arriving from the worker (GUI thread)."""
        self._daqmon_show_reading(payload)
        self.daqmon_status.setText(
            f"Continuous \N{BULLET} reading {int(payload.get('count', 0))} over "
            f"{payload['kept']:g} s \N{BULLET} "
            f"{self._daqmon_reading_summary(payload)}."
        )

    def _daq_stream_stop(self) -> None:
        if self.daq_stream_stop_event is not None:
            self.daq_stream_stop_event.set()
            self.daqmon_status.setText(
                "Stopping\N{HORIZONTAL ELLIPSIS} "
                "(after the acquisition in flight)")

    def _daq_stream_done(self, payload: dict[str, Any]) -> None:
        self.daq_stream_stop_event = None
        self._daqmon_set_running()
        n = int((payload or {}).get("readings", 0))
        self.daqmon_status.setText(f"Continuous stopped after {n} readings.")

    def _daq_stream_error(self, _error: str) -> None:
        self.daq_stream_stop_event = None
        self._daqmon_set_running()
        self.daqmon_status.setText("DAQ continuous read failed (see Status log).")

    def _daq_waveform_draw(self, payload: dict[str, Any] | None) -> None:
        """Time-domain trace (top) + amplitude spectrum (bottom), draft-style."""
        self.daqmon_fig.clear()
        self.daqmon_fig.patch.set_facecolor("#101820")
        ax_t = self.daqmon_fig.add_subplot(211)
        ax_f = self.daqmon_fig.add_subplot(212)
        for ax in (ax_t, ax_f):
            self._style_dark_axes(ax)
        ax_t.set_xlabel("Time (s)")
        ax_t.set_ylabel("Voltage (mV)")
        ax_f.set_xlabel("Frequency (Hz)")
        ax_f.set_ylabel("Amplitude (\N{MICRO SIGN}V)")
        if payload is None:
            ax_t.text(0.5, 0.5, "Press Read waveform to acquire one trace",
                      ha="center", va="center", color="#d8dee9",
                      transform=ax_t.transAxes, fontsize=9)
            self.daqmon_canvas.draw_idle()
            return
        t = payload["times"]
        raw = payload["raw"]
        filt = payload["filtered"]
        fs = payload["fs"]
        n0 = int(payload["n_settle"])
        f_hw, f_dig = payload["f_hw"], payload["f_dig"]
        # --- time domain (full trace shown; shaded span = discarded warmup) ---
        ax_t.plot(t, raw * 1000.0, color="#4c6a78", linewidth=0.8, alpha=0.8,
                  label=f"raw (hw {f_hw:g} Hz)")
        ax_t.plot(t, filt * 1000.0, color="#88c0d0", linewidth=1.4,
                  label=f"low-pass {f_dig:g} Hz")
        if n0 > 0:
            ax_t.axvspan(0.0, n0 / fs, color="#d8dee9", alpha=0.12,
                         label=f"warmup ({n0 / fs * 1000.0:.0f} ms)")
        ax_t.set_title(f"{payload['channel']}  ({fs:g} S/s, {payload['duration']:g} s)",
                       color="#d8dee9", fontsize=9)
        # --- amplitude spectrum of the kept (steady-state) trace ---
        fr, sr = self._daqmon_spectrum(raw[n0:], fs)
        ff, sf = self._daqmon_spectrum(filt[n0:], fs)
        if fr.size:
            ax_f.loglog(fr, sr * 1e6, color="#4c6a78", linewidth=0.8, alpha=0.8,
                        label="raw")
        if ff.size:
            ax_f.loglog(ff, sf * 1e6, color="#88c0d0", linewidth=1.4,
                        label=f"low-pass {f_dig:g} Hz")
        ax_f.axvline(f_hw, color="#d8dee9", linestyle="--", linewidth=1.0, alpha=0.6,
                     label=f"hardware {f_hw:g} Hz")
        if f_dig != f_hw:
            ax_f.axvline(f_dig, color="#bf616a", linestyle=":", linewidth=1.2,
                         alpha=0.8, label=f"digital {f_dig:g} Hz")
        for ax in (ax_t, ax_f):
            ax.legend(fontsize=7, facecolor="#101820", edgecolor="#41515c",
                      labelcolor="#d8dee9")
        self.daqmon_canvas.draw_idle()

    # NI-DAQmx programmable-gain input ranges (device-independent common set;
    # the driver silently snaps to the closest one the connected card supports).
    _DAQ_RANGES = [(-0.1, 0.1), (-0.2, 0.2), (-0.5, 0.5), (-1.0, 1.0),
                   (-2.0, 2.0), (-5.0, 5.0), (-10.0, 10.0)]

    def _page_shell(self, title: str) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(26, 24, 26, 24)
        layout.setSpacing(18)
        heading = QtWidgets.QLabel(title)
        heading.setObjectName("PageTitle")
        # A row, not a bare label: _page_status hangs this page's one-line
        # feedback off the end of it.  Item 0 of the page layout IS this row.
        header = QtWidgets.QHBoxLayout()
        header.setSpacing(16)
        header.addWidget(heading)
        layout.addLayout(header)
        return page

    def _page_status(self, page: QtWidgets.QWidget,
                     text: str = "") -> StatusLine:
        """This page's one-line status, beside its title.

        Beside the title, not at the foot of the control column: it is the one
        thing on the page that changes while you are reading it, and down there
        it was a second paragraph of prose under a first one, which is what a
        log looks like.  One line, newest only, nothing kept.
        """
        status = StatusLine(text)
        page.layout().itemAt(0).layout().addWidget(status, 1)
        return status

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

    def _unfiled_run(self, stamp: str) -> Path:
        """Where a sweep goes when no calib file was named: never nowhere.

        Auto-save removed the dialog, and with it the chance to type a path for
        a sweep whose page named no file.  The rows are still an hour of bench
        time, so they are parked in a dated folder under the calibration
        directory rather than dropped -- no section is written there (there is
        no step 3 to interpret them against), and the page says so.

        Gating should make this unreachable: a page with no file has no live Run
        button.  It stays because the cost of being wrong about that is an hour
        of bench time, and the cost of keeping it is a folder.
        """
        out = self._default_calib_dir() / f"unfiled_{stamp}"
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _run_chain(self, run: Path) -> str:
        """``"step2 09-28 · step3 09-29 · step6 —"`` for a calib file's status line.

        The same chain the offline runners print before touching the bench, short
        enough for one line.  A section that predates one above it is marked, not
        refused -- whether that matters is the operator's call, but it is not
        going to be a surprise.
        """
        try:
            calib = load_calib(run)
        except Exception as exc:                        # noqa: BLE001
            return f"no calib.json ({exc.__class__.__name__})"
        stale = calib.stale()
        parts = []
        for step in CALIB_STEP_ORDER:
            ran = calib.ran(step)
            if ran is None:
                parts.append(f"step{step} —")
                continue
            mark = " (stale)" if step in stale else ""
            parts.append(f"step{step} {ran[5:10].replace('-', '-')}{mark}")
        return " · ".join(parts)

    def _set_status(self, label: QtWidgets.QLabel, text: str, status: str) -> None:
        """Update a status pill label (status in: ok, error, off).

        A :class:`StatusLine` takes the text and none of the pill: it already
        sits beside the title as the page's one line of feedback, and a green
        capsule up there would be the loudest thing on the page saying the least.
        """
        label.setText(text)
        if isinstance(label, StatusLine):
            return
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
        # The SLM Control page scans the WHOLE panel by default, so its end
        # follows the real width.
        self.start_x_spin.setMaximum(width - 1)
        self.end_x_spin.setMaximum(width - 1)
        self.end_x_spin.setValue(width - 1)
        # The calibration region is a different thing: a deliberate band of
        # columns Step 2 sweeps, defaulted on the page.  Bound it to the real
        # width -- setMaximum pulls an over-wide value down on its own -- but do
        # NOT widen it.  This used to reset the end to the full width whenever
        # the limit was off, which silently replaced that default with the panel
        # width and left the start where it was, so the two ends disagreed about
        # what the region was.
        for step in (2, 3):
            widgets = getattr(self, "step_widgets", {}).get(step, {})
            if "region_end" in widgets:
                widgets["region_start"].setMaximum(width - 1)
                widgets["region_end"].setMaximum(width - 1)
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

    def _browse_dir_into(self, edit: QtWidgets.QLineEdit, caption: str) -> None:
        """Pick a run FOLDER.  Steps 6, 7 and 8 all take one, not a named file."""
        path = QtWidgets.QFileDialog.getExistingDirectory(self, caption, "")
        if path:
            edit.setText(path)

    def _toggle_step2_source(self) -> None:
        index = self.step_widgets[2]["source"].currentIndex()
        self.step_widgets[2]["manual_row"].setVisible(index == 1)

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
        # every measure() from any module now streams to the live monitor
        osa.add_trace_listener(self._osa_bridge.on_trace)
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
            osa.remove_trace_listener(self._osa_bridge.on_trace)
            self._run_task("Disconnect OSA", osa.disconnect)

    def _calibration_busy(self) -> bool:
        """True while any GUI calibration (steps 1-3c, 4b, 6, 6b, 7) is running."""
        return self._calibration_is_running or bool(self._running_tpa_calibrations)

    def _mark_tpa_calibration(self, name: str, running: bool) -> None:
        if running:
            self._running_tpa_calibrations.add(name)
        else:
            self._running_tpa_calibrations.discard(name)
        self._refresh_manual_reads()

    def _refresh_manual_reads(self) -> None:
        """Lock the OSA Viewer sweeps and the DAQ Monitor reads during a calibration.

        A manual read would share the instrument with the calibration's own
        acquisitions, so a continuous OSA sweep or DAQ reading loop already
        running is stopped too.
        """
        busy = self._calibration_busy()
        if busy and self.osa_view_stop_event is not None:
            self.osa_view_stop_event.set()
            self._log("OSA Viewer sweep stopped: a calibration started")
        if hasattr(self, "osv_single_button"):
            self._osa_view_set_running(
                self.osa_view_stop_event is not None,
                continuous=self.osv_cont_button.isChecked(),
            )
        if busy and self.daq_stream_stop_event is not None:
            self.daq_stream_stop_event.set()
            self._log("DAQ continuous reading stopped: a calibration started")
        if hasattr(self, "daqmon_read_button"):
            self._daqmon_set_running(
                continuous=self.daqmon_cont_button.isChecked())

    def _set_calibration_running(self, running: bool) -> None:
        self._calibration_is_running = running
        self._refresh_manual_reads()
        connected = self.osa_controller is not None
        for button in getattr(self, "calibration_run_buttons", []):
            button.setEnabled(connected and not running)
        if hasattr(self, "stage3_reopt_stop_button"):
            self.stage3_reopt_stop_button.setEnabled(running)
        if hasattr(self, "fast_channel_stop_button"):
            self.fast_channel_stop_button.setEnabled(running)
        step3c = getattr(self, "step_widgets", {}).get("3c", {})
        if "stop" in step3c:
            step3c["stop"].setEnabled(running)
        self.osa_connect_button.setEnabled(not running and not connected)
        self.osa_disconnect_button.setEnabled(not running and connected)
        # Step 3 may run on the DAQ instead of the OSA, so its Run button is
        # gated on whichever detector it currently targets.
        self._refresh_step3_run_button()

    def _refresh_step3_run_button(self) -> None:
        """Step 3's Run: gated on the DAQ (not the OSA) AND on its calib file.

        Two owners would fight over one button, so the file requirement is
        folded in here rather than given its own entry in ``_STEP_BUTTONS``.
        """
        widgets = getattr(self, "step_widgets", {}).get("3c")
        if not widgets or "run" not in widgets:
            return
        if getattr(self, "_calibration_is_running", False):
            widgets["run"].setEnabled(False)
            return
        missing = self._calib_missing("3c")
        widgets["run"].setEnabled(
            not missing
            and self.daq_controller is not None
            and self.daq_controller.is_connected
        )
        widgets["run"].setToolTip(
            "" if not missing else
            "that file has no " + ", ".join(f"step{n}" for n in missing))

    # ----- per-step config readers (GUI thread) -----
    def _step_settings(self, step: int) -> MeasurementSettings:
        widgets = self.step_widgets[step]
        return MeasurementSettings(
            center_wl=widgets["center_wl"].text().strip() or "778nm",
            span=widgets["span"].text().strip() or "8nm",
            sensitivity=widgets["sensitivity"].currentText(),
            sampling_points=(
                widgets["sampling_points"].text().strip() or "AUTO"
                if "sampling_points" in widgets else "AUTO"
            ),
            reference_level=widgets["ref_level"].text().strip() or "10uW",
            y_unit="LINear",
        )

    def _step_levels(self, step: int | str) -> list[int]:
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
        return self._parse_guard_bands(
            self.fast_channel_guard_wl_edit.text(),
            self.fast_channel_guard_nm_spin.value(),
        )

    def _step3c_guard_bands(self) -> list[tuple[float, float]]:
        widgets = self.step_widgets["3c"]
        if not widgets["guard_check"].isChecked():
            return []
        return self._parse_guard_bands(
            widgets["guard_wl"].text(), widgets["guard_nm"].value()
        )

    def _parse_guard_bands(
        self, value_text: str, half_width: float
    ) -> list[tuple[float, float]]:
        value_text = value_text.strip()
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
        if half_width <= 0.0:
            raise ValueError("guard half-width must be positive")
        return [(center, half_width) for center in centers]

    def _step_region(self, step: int | str) -> tuple[int, int] | None:
        widgets = self.step_widgets[step]
        if not widgets["region_check"].isChecked():
            return None
        start = widgets["region_start"].value()
        end = widgets["region_end"].value()
        if end < start:
            raise ValueError("region end must be >= region start")
        return (start, end)

    def _resolve_output_path(self, text: str, step: int | str, suffix: str = ".json") -> Path:
        """An explicit path is used as-is; blank saves to the default calib dir."""
        text = text.strip()
        if text:
            return Path(text)
        out_dir = self._default_calib_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / self._default_calib_name(step, suffix)

    def _resolve_step_input(self, step: int | str) -> CalibrationResult:
        """What a sweep needs out of its calib file: levels for 2, the map for 3.

        Both come from the one file the page names, which is what the runners
        do -- ``calib_step2.py`` reads its ``step1`` section and
        ``calib_step3.py`` its ``step2`` section.  Step 2's levels can be typed
        instead, and that is a first-class path (see its ``--min-max``); step 3
        has nothing to type, so it has no choice to make.
        """
        widgets = self.step_widgets[step]
        calib_file = self._require_calib_file(step)
        if step == 2:
            if widgets["source"].currentIndex() == 1:        # manual min/max
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
            section = load_calib(calib_file, missing_ok=True).get(1)
            if section is None:
                raise ValueError(
                    f"{calib_file.name} has no step1 section — run step 1 "
                    f"against it, or switch the source to Manual min/max"
                )
            return CalibrationResult(
                wavelength=np.asarray([]),
                coordinates=np.asarray([]),
                max_level=int(section["max_level"]),
                min_level=int(section["min_level"]),
                level_range=np.asarray([], dtype=int),
            )

        # step 3: the map IS the file's step-2 section plus the CSV it points at
        result = load_step2_map(calib_file)
        if (
            np.asarray(result.coordinates).size == 0
            or np.asarray(result.wavelength).size == 0
        ):
            raise ValueError("the step2 section has no coordinate -> wavelength map")
        self._require_levels(result)
        return result

    def _resolve_fast_channel_step2_input(self) -> CalibrationResult:
        if self.fast_channel_source_combo.currentIndex() == 1:
            path = self.fast_channel_step2_edit.text().strip()
            if not path:
                raise ValueError("choose a run folder, its calib.json, or a "
                                 "wavelength-map CSV")
            min_level = self.fast_channel_min_spin.value()
            max_level = self.fast_channel_max_spin.value()
            if max_level < min_level:
                raise ValueError("max level must be >= min level")
            result = self._load_step2_source(path, min_level, max_level)
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

    def _load_step2_source(self, path: str, min_level: int,
                           max_level: int) -> CalibrationResult:
        """A step-2 wavelength map from whatever the operator pointed at.

        A run folder or its ``calib.json`` goes through
        :func:`~calibration_module.fit.slm_response.load_step2_map`, which pairs
        the section's levels with the CSV's line.  A bare coordinate ->
        wavelength CSV has no levels of its own, so the spinboxes supply them.
        A pre-v3 standalone step-2 JSON still loads as itself.
        """
        target = Path(path).expanduser()
        if target.is_dir() or target.name == CALIB_FILE_NAME:
            return load_step2_map(target, fallback_min_max=(min_level, max_level))
        if target.suffix.lower() == ".csv":
            return load_wavelength_map_csv(
                target, min_level=min_level, max_level=max_level)
        return load_calibration_result(target)

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

    def _daq_ready(self) -> DAQController | None:
        daq = self.daq_controller
        if daq is None or not daq.is_connected:
            self._log("Connect the DAQ first (Connections page)")
            return None
        return daq

    def _step3_daq_settings(self, step: int | str = 3) -> DAQMonitorSettings:
        widgets = self.step_widgets[step]
        min_val, max_val = widgets["daq_range"].currentData()
        return DAQMonitorSettings(
            channel=widgets["daq_channel"].text().strip() or "ai1",
            sample_rate=widgets["daq_sample_rate"].value(),
            duration=widgets["daq_duration"].value(),
            hold=widgets["daq_hold"].value() / 1000.0,  # ms -> s
            min_val=min_val,
            max_val=max_val,
        )

    def _run_step1(self) -> None:
        osa = self._osa_ready()
        if osa is None:
            return
        try:
            settings = self._step_settings(1)
            levels = self._step_levels(1)
            calib_file = self._require_calib_file(1)
        except ValueError as exc:
            return self._reject_calibration(exc)
        controller = self._controller()
        self._log(f"Step 1 started: {len(levels)} levels")

        def work(report: ProgressEmit, stop_event: threading.Event) -> dict[str, Any]:
            result = run_step1(
                osa, controller, levels, settings,
                stop_event=stop_event, progress_callback=report,
            )
            # Written only now.  A step 1 drops every section under it, so an
            # abandoned sweep must not have cost the file anything -- naming it
            # in the dialog is not yet a decision to replace it.
            calib = write_section(calib_file, 1, step1_section(result),
                                  log=lambda m: self._log(m))
            return {
                "status": "ok", "step": 1, "result": result, "saved": calib.path,
                "summary": f"min level {result.min_level}, max level {result.max_level}",
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
            stride = self.step_widgets[2]["stride"].value()
            sweep_nm = self.step_widgets[2]["sweep_nm"].value() or None
            # The usable band is NOT a peak-search mask -- see below.
            min_wl = self.step_widgets[2]["min_wl"].value() or None
            max_wl = self.step_widgets[2]["max_wl"].value() or None
            if min_wl is not None and max_wl is not None and min_wl >= max_wl:
                raise ValueError("usable λ minimum must be below the maximum")
            region = self._step_region(2)
            calib_file = self._require_calib_file(2)
        except ValueError as exc:
            return self._reject_calibration(exc)
        controller = self._controller()
        lo_txt = "open" if min_wl is None else f"{min_wl:.2f}"
        hi_txt = "open" if max_wl is None else f"{max_wl:.2f}"
        band = "off" if min_wl is None and max_wl is None else f"{lo_txt}-{hi_txt} nm"
        self._log(
            f"Step 2 started: window {window} px, usable band {band} "
            "(applied to the result, not to the peak search)"
        )

        def work(report: ProgressEmit, stop_event: threading.Event) -> dict[str, Any]:
            # The band clips the finished map, never the peak search -- see
            # run_step2.
            clipped, result = run_step2(
                osa, controller, settings, seed,
                window_size=window, peak_half_window_nm=peak_nm, region=region,
                coordinate_stride=stride,
                sweep_span_nm=sweep_nm,
                min_wavelength_nm=min_wl, max_wavelength_nm=max_wl,
                stop_event=stop_event, progress_callback=report,
            )
            swept = result.coordinates.size
            span = (
                f"{result.wavelength.min():.3f}-{result.wavelength.max():.3f} nm"
                if swept else "no coordinates"
            )
            kept = clipped.coordinates.size
            note = "" if kept == swept else f" of {swept} swept ({span})"
            # CSV first and unconditionally: the map is on disk before the
            # section that points at it can fail.
            stamp = time.strftime("%m%d_%H%M")
            calib_file.parent.mkdir(parents=True, exist_ok=True)
            csv_path = write_wavelength_map_csv(
                clipped, calib_file.parent / f"step2_{stamp}.csv")
            calib = write_section(calib_file, 2, step2_section(clipped),
                                  csv=csv_path, log=lambda m: self._log(m))
            return {
                "status": "ok", "step": 2, "result": clipped,
                "saved": calib.path, "csv": csv_path,
                "summary": f"{kept} coordinates{note}",
            }

        self._launch_calibration("Run step 2", work)

    def _run_step3c(self) -> None:
        """Step 3c: the DAQ intensity sweep over Step-3b-style channel centres."""
        daq = self._daq_ready()
        if daq is None:
            return
        widgets = self.step_widgets["3c"]
        try:
            mapping = self._resolve_step_input("3c")
            levels = self._step_levels("3c")
            target = widgets["target"].value()
            window = widgets["window"].value()
            pad = widgets["pad"].value()
            n_channels = widgets["count"].value()
            guard_bands = self._step3c_guard_bands()
            daq_settings = self._step3_daq_settings("3c")
            calib_file = self._require_calib_file("3c")
        except ValueError as exc:
            return self._reject_calibration(exc)
        controller = self._controller()
        daq.configure_monitor(daq_settings)
        read_timeout = max(30.0, daq_settings.duration * 3.0 + 10.0)
        self._log(
            f"Step 3c (DAQ) started: {len(levels)} levels, "
            f"{2 * n_channels} channels around {target:g} nm, "
            f"width {window} px, gap {pad} px, "
            f"{daq_settings.channel} @ {daq_settings.sample_rate:g} S/s, "
            f"avg {daq_settings.duration:g}s"
        )
        if guard_bands:
            guard_text = ", ".join(
                f"{center:g}±{half:g} nm" for center, half in guard_bands
            )
            self._log(f"Step 3c guard bands: {guard_text} -> skipped")

        def work(report: ProgressEmit, stop_event: threading.Event) -> dict[str, Any]:
            sweep = run_step3c(
                daq, controller, levels, mapping,
                target_wavelength_nm=target,
                n_channels_per_side=n_channels,
                channel_width_px=window,
                gap_px=pad,
                guard_bands_nm=guard_bands,
                read_timeout=read_timeout,
                stop_event=stop_event, progress_callback=report,
            )
            result = sweep.result
            # CSV first: the sweep is on disk before a fit can fail on it.
            stamp = time.strftime("%m%d_%H%M")
            csv_path = write_intensity_calibration_csv(
                result, calib_file.parent / f"step3_{stamp}.csv")
            calib = write_section(calib_file, 3, step3_section(result),
                                  csv=csv_path, log=lambda m: self._log(m))
            return {
                "status": "ok", "step": "3c", "result": result,
                "saved": calib.path, "csv": csv_path,
                "center_coordinate": sweep.center_coordinate,
                "summary": (
                    f"{result.coordinates.size} channels, "
                    f"pitch {window + pad} px"
                ),
            }

        self._launch_calibration("Run step 3c (DAQ)", work)

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

        out_json = self._resolve_output_path(self.fast_channel_json_edit.text(), "3b")
        out_csv = self._resolve_output_path(
            self.fast_channel_csv_edit.text(), "3b", ".csv"
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
            sweep = run_step3b(
                osa,
                controller,
                levels,
                settings,
                step2_mapping,
                target_wavelength_nm=target_wavelength,
                n_channels_per_side=n_channels,
                channel_width_px=channel_width,
                gap_px=gap_px,
                group_skip_channels=group_skip,
                fine_tune_center=fine_tune_center,
                peak_half_window_nm=peak_half_window_nm,
                wavelength_window_nm=avg_nm,
                refine_wavelength=refine,
                refine_half_window_nm=refine_half_window_nm,
                guard_bands_nm=guard_bands,
                stop_event=stop_event,
                progress_callback=report,
            )
            final = sweep.result
            save_calibration_result(final, out_json)
            csv_path = write_intensity_calibration_csv(final, out_csv)
            group_count = min(group_skip + 1, int(final.coordinates.size))
            return {
                "status": "ok",
                "step": "fast_channels",
                "result": final,
                "saved": out_json,
                "csv": csv_path,
                "center_coordinate": sweep.center_coordinate,
                "coarse_center": sweep.coarse_center,
                "measured_peak": sweep.measured_peak_nm,
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
                sampling_points=(
                    self.stage3_reopt_sampling_edit.text().strip() or "1001"
                ),
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

    def _open_calibration_dialog(self, on_stop: Callable[[], None] | None = None) -> None:
        """Pop the shared live-progress window.

        ``on_stop`` lets a page that owns its own stop event use this dialog --
        step 6 sets its sweep's event rather than the full-calibration one.
        Defaults to the full-calibration stop, which is what steps 1-3 want.
        """
        if self.calibration_dialog is not None:
            self.calibration_dialog.close()
        dialog = CalibrationProgressDialog(
            self, on_stop=on_stop or self._stop_full_calibration
        )
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

        if step in (1, 2, "3c"):
            written = [f"step{3 if step == '3c' else step}"]
            if payload.get("csv") is not None:
                written.append(Path(payload["csv"]).name)
            self.step_widgets[step]["status"].setText(
                f"Done \N{MIDDLE DOT} {summary} \N{MIDDLE DOT} wrote "
                + " + ".join(written))
            # The file it went into may now carry a section it did not before,
            # and every page that names it has to re-gate on that.
            self._calib_describe_all()
            center_coordinate = payload.get("center_coordinate")
            if step == "3c" and center_coordinate is not None:
                self._log(
                    "Step 3c center: "
                    f"x={float(center_coordinate):.3f} px "
                    f"(physical pixel {int(round(float(center_coordinate)))})"
                )
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

        if step == "stage3_reopt":
            label = "Stage 3 re-optimization"
        elif step == "fast_channels":
            label = "Fast channel calibration"
        else:
            label = f"Step {step}"
        self._log(f"{label} done: {summary}")

        csv_path = payload.get("csv")
        if csv_path is not None:
            self._log(f"Calibration CSV saved: {csv_path}")

        if self.calibration_dialog is not None:
            self.calibration_dialog.finish(True, f"{label} done \N{MIDDLE DOT} {summary}")

    def _on_step_error(self, _error: str) -> None:
        # _fail_task already logged the traceback and showed a dialog
        active_label = self._active_calibration_label
        self._active_calibration_label = None
        self.calibration_stop_event = None
        self._set_calibration_running(False)
        if active_label == "Stage 3 re-optimization":
            self.stage3_reopt_status_label.setText("Failed")
            self.edge_gain_stop_event = None
            self._edge_gain_running(False)
            self.edge_gain_bar.setRange(0, 100)
            self.edge_gain_bar.setValue(0)
            self.edge_gain_status.setText("Stage 3 re-optimization failed.")
        if active_label == "Fast channel calibration":
            self.fast_channel_status_label.setText("Failed")
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

    def _segment_axis(self) -> str:
        return "x" if self.segment_axis_combo.currentIndex() == 0 else "y"

    def _segment_axis_size(self) -> int:
        width, height = self.slm_size
        return width if self._segment_axis() == "x" else height

    def _on_segment_axis_changed(self) -> None:
        axis = self._segment_axis()
        self.segments_table.setHorizontalHeaderLabels(
            [f"{axis} start", f"{axis} end", "Level"]
        )
        if self._segment_mode_is_equal():
            self._rebuild_equal_segment_rows()
        else:
            self._update_segment_preview()

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
        size = self._segment_axis_size()
        count = min(self.segment_count_spin.value(), size)
        edges = equal_segment_edges(size, count)

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
        size = self._segment_axis_size()
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
                row, 0, QtWidgets.QTableWidgetItem(str(min(previous_end, size - 1)))
            )
            self.segments_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(size)))
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
        axis = self._segment_axis()
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
            return make_equal_segments(width, height, levels, axis=axis)
        segments = [
            (
                cell(row, 0, f"{axis} start"),
                cell(row, 1, f"{axis} end"),
                cell(row, 2, "level"),
            )
            for row in range(rows)
        ]
        return make_segments(width, height, segments, axis=axis)

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
        if self.daq_stream_stop_event is not None:
            self.daq_stream_stop_event.set()
        if self.psu_stop_event is not None:
            self.psu_stop_event.set()
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
        if self.heater_stop_event is not None:
            self.heater_stop_event.set()
        if self.daq_controller is not None:
            try:
                self.daq_controller.disconnect()
            except Exception:
                pass
            self.daq_controller = None
        if self.heater_controller is not None:
            try:
                self.heater_controller.disconnect()
            except Exception:
                pass
            self.heater_controller = None
        if self.psu_controller is not None:
            # Closes the port only: the PMT keeps whatever gain voltage it has,
            # so quitting the GUI never changes a running measurement.
            try:
                self.psu_controller.disconnect()
            except Exception:
                pass
            self.psu_controller = None
        super().closeEvent(event)

    def _apply_style(self) -> None:
        self.setStyleSheet(DARK_STYLESHEET)


def main(argv: list[str] | None = None) -> int:
    app = QtWidgets.QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("Santec SLM Control")
    window = MainWindow()
    window.show()
    return app.exec_()
