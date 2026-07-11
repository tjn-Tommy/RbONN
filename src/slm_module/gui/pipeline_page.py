"""Unified calibration pipeline page (replaces the old file-only Pipeline tab).

Drives :mod:`slm_module.pipeline` — the Qt-free orchestrator for
wl_map (1+2) -> intensity (3, batch fast-channels) -> tpa_center -> pair_eta
(6) -> comb_phase (7) — from one form: per-stage enable checkbox, per-input
"From memory / From file" source, always-saved output paths, and per-stage
settings (every OSA stage exposes sampling points + sensitivity; steps 2/3
expose the outlier auto-remeasure policy; the layout group exposes the
crosstalk-reducing centre gap).

The page only talks to the main window through a small duck-typed ``host``
contract: ``_osa_ready() / _enc_active_monitor() / _controller() /
_monitor_settings() / _daq_monitor_settings() / _active_col_ratio() /
_run_slm_task() / _set_calibration_running() / _log()``.
"""
from __future__ import annotations

import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtWidgets

from ..calibration.calibration_new import CalibrationProgress
from ..calibration.outliers import OutlierRemeasurePolicy
from ..pipeline import (
    STAGES,
    CombPhaseConfig,
    InputSpec,
    IntensityConfig,
    LayoutConfig,
    OSAStageSettings,
    PairEtaConfig,
    PipelineAborted,
    PipelineInstruments,
    PipelineRequest,
    PipelineStageError,
    StagePlan,
    TPACenterConfig,
    WlMapConfig,
    plot_point,
    required_instruments,
    run_pipeline,
    validate_request,
)
from ..tpa_phase_report import plot_fringe
from .common import CalibrationProgressDialog
from .style import DARK_STYLESHEET

_INPUT_LABELS = {
    "wl_map": "Step 1+2 wavelength map",
    "intensity_calib": "Step 3 intensity calibration",
    "center_fit": "TPA centre fit (optional)",
    "pair_etas": "Step 6 pair etas",
}
_SENSITIVITIES = ("NORM", "MID", "HIGH1", "HIGH2", "HIGH3")


def _default_out_dir() -> Path:
    candidate = Path(__file__).resolve().parents[3] / "src" / "calib_data"
    return candidate if candidate.is_dir() else Path.cwd()


class _StageRow:
    """Widgets of one pipeline stage: enable box, inputs, output, settings."""

    def __init__(self, spec, page: "PipelinePage"):
        self.spec = spec
        self.page = page
        self.group = QtWidgets.QGroupBox(spec.label)
        self.group.setCheckable(True)
        self.group.setChecked(False)
        self.group.toggled.connect(page._refresh_ui)
        self.layout = QtWidgets.QGridLayout(self.group)
        self._row = 0

        # ---- inputs -------------------------------------------------------
        self.input_combos: dict[str, QtWidgets.QComboBox] = {}
        self.input_edits: dict[str, QtWidgets.QLineEdit] = {}
        for key in tuple(spec.requires) + tuple(spec.optional):
            optional = key in spec.optional
            combo = QtWidgets.QComboBox()
            if optional:
                combo.addItem("Skip")
            combo.addItem("From memory")
            combo.addItem("From file…")
            combo.currentIndexChanged.connect(page._refresh_ui)
            edit = QtWidgets.QLineEdit()
            edit.setPlaceholderText("input file path")
            browse = QtWidgets.QPushButton("Browse…")
            browse.setProperty("variant", "ghost")
            browse.clicked.connect(
                lambda _=False, e=edit: page._browse_open(e)
            )
            self.layout.addWidget(
                QtWidgets.QLabel(_INPUT_LABELS.get(key, key)), self._row, 0
            )
            self.layout.addWidget(combo, self._row, 1)
            self.layout.addWidget(edit, self._row, 2)
            self.layout.addWidget(browse, self._row, 3)
            self.input_combos[key] = combo
            self.input_edits[key] = edit
            self._row += 1

        # ---- output -------------------------------------------------------
        self.output_edit = QtWidgets.QLineEdit()
        self.output_edit.setPlaceholderText("output JSON path (always written)")
        out_browse = QtWidgets.QPushButton("Browse…")
        out_browse.setProperty("variant", "ghost")
        out_browse.clicked.connect(
            lambda _=False: page._browse_save(self.output_edit)
        )
        self.layout.addWidget(QtWidgets.QLabel("Output"), self._row, 0)
        self.layout.addWidget(self.output_edit, self._row, 2)
        self.layout.addWidget(out_browse, self._row, 3)
        self._row += 1

        self.status = QtWidgets.QLabel("\N{EN DASH}")
        self.status.setWordWrap(True)
        self.layout.addWidget(self.status, self._row, 0, 1, 4)
        self._row += 1

    def add_settings(self, widget: QtWidgets.QWidget) -> None:
        self.layout.addWidget(widget, self._row, 0, 1, 4)
        self._row += 1

    def input_spec(self, key: str) -> InputSpec | None:
        combo = self.input_combos[key]
        choice = combo.currentText()
        if choice == "Skip":
            return None
        if choice == "From memory":
            return InputSpec("memory")
        text = self.input_edits[key].text().strip()
        return InputSpec("file", Path(text) if text else None)


def _levels_row(
    grid: QtWidgets.QGridLayout, row: int, label: str,
    start: int, stop: int, step: int,
) -> tuple[QtWidgets.QSpinBox, QtWidgets.QSpinBox, QtWidgets.QSpinBox]:
    def spin(value: int, lo: int = 0, hi: int = 1023) -> QtWidgets.QSpinBox:
        widget = QtWidgets.QSpinBox()
        widget.setRange(lo, hi)
        widget.setValue(value)
        return widget

    start_spin, stop_spin, step_spin = spin(start), spin(stop), spin(step, 1, 512)
    grid.addWidget(QtWidgets.QLabel(label), row, 0)
    grid.addWidget(start_spin, row, 1)
    grid.addWidget(QtWidgets.QLabel("to"), row, 2)
    grid.addWidget(stop_spin, row, 3)
    grid.addWidget(QtWidgets.QLabel("step"), row, 4)
    grid.addWidget(step_spin, row, 5)
    return start_spin, stop_spin, step_spin


def _levels_from(start: QtWidgets.QSpinBox, stop: QtWidgets.QSpinBox,
                 step: QtWidgets.QSpinBox) -> list[int]:
    lo, hi, inc = start.value(), stop.value(), step.value()
    if hi < lo:
        raise ValueError("level stop must be >= level start")
    levels = list(range(lo, hi + 1, inc)) or [lo]
    if levels[-1] != hi:
        levels.append(hi)
    return levels


class _OsaSettingsGroup(QtWidgets.QGroupBox):
    """center/span/sensitivity/sampling points/ref level for one OSA stage."""

    def __init__(self, *, span: str = "8nm"):
        super().__init__("OSA sweep")
        grid = QtWidgets.QGridLayout(self)
        self.center = QtWidgets.QLineEdit("778nm")
        self.span = QtWidgets.QLineEdit(span)
        self.sensitivity = QtWidgets.QComboBox()
        self.sensitivity.addItems(_SENSITIVITIES)
        self.sensitivity.setCurrentText("HIGH2")
        self.points = QtWidgets.QLineEdit("AUTO")
        self.points.setToolTip("Sampling points: AUTO or a count like 1001")
        self.ref_level = QtWidgets.QLineEdit("10uW")
        grid.addWidget(QtWidgets.QLabel("Center"), 0, 0)
        grid.addWidget(self.center, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Span"), 0, 2)
        grid.addWidget(self.span, 0, 3)
        grid.addWidget(QtWidgets.QLabel("Sensitivity"), 0, 4)
        grid.addWidget(self.sensitivity, 0, 5)
        grid.addWidget(QtWidgets.QLabel("Points"), 1, 0)
        grid.addWidget(self.points, 1, 1)
        grid.addWidget(QtWidgets.QLabel("Ref level"), 1, 2)
        grid.addWidget(self.ref_level, 1, 3)

    def to_settings(self) -> OSAStageSettings:
        return OSAStageSettings(
            center_wl=self.center.text().strip() or "778nm",
            span=self.span.text().strip() or "8nm",
            sensitivity=self.sensitivity.currentText(),
            sampling_points=self.points.text().strip() or "AUTO",
            reference_level=self.ref_level.text().strip() or "10uW",
        )


class _OutlierGroup(QtWidgets.QGroupBox):
    """Auto-remeasure policy controls (checkable; ON by default)."""

    def __init__(self):
        super().__init__("Outlier auto-remeasure")
        self.setCheckable(True)
        self.setChecked(True)
        grid = QtWidgets.QGridLayout(self)
        self.k_sigma = QtWidgets.QDoubleSpinBox()
        self.k_sigma.setRange(1.0, 20.0)
        self.k_sigma.setValue(4.0)
        self.k_sigma.setSingleStep(0.5)
        self.retries = QtWidgets.QSpinBox()
        self.retries.setRange(1, 10)
        self.retries.setValue(3)
        grid.addWidget(QtWidgets.QLabel("k·sigma"), 0, 0)
        grid.addWidget(self.k_sigma, 0, 1)
        grid.addWidget(QtWidgets.QLabel("max retries"), 0, 2)
        grid.addWidget(self.retries, 0, 3)

    def policy(self) -> OutlierRemeasurePolicy | None:
        if not self.isChecked():
            return None
        return OutlierRemeasurePolicy(
            k_sigma=self.k_sigma.value(), max_retries=self.retries.value()
        )


class PipelinePage(QtWidgets.QWidget):
    """The unified wl_map -> ... -> comb_phase pipeline runner."""

    pipeline_progress = QtCore.pyqtSignal(object)   # PipelineProgress (worker -> GUI)

    def __init__(self, host, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.host = host
        self._stop_event: threading.Event | None = None
        self._dialog: CalibrationProgressDialog | None = None
        self._phase_results: dict[int, Any] = {}     # step-7 PhaseResult per target
        self._settings = QtCore.QSettings("santec", "slm-suite")

        outer = QtWidgets.QVBoxLayout(self)
        caption = QtWidgets.QLabel(
            "Unified calibration pipeline: enable any consecutive stages; "
            "inputs come from an earlier enabled stage (memory) or from a "
            "file, and every stage's output is saved to disk as soon as it "
            "finishes. Steps 1-3 need the OSA; the TPA stages need the "
            "scope/DAQ monitor. The SLM must be open."
        )
        caption.setWordWrap(True)
        caption.setObjectName("PageSubtitle")
        outer.addWidget(caption)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        body = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(body)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        layout.addWidget(self._build_layout_group())

        self.rows: dict[str, _StageRow] = {}
        for spec in STAGES:
            row = _StageRow(spec, self)
            self.rows[spec.stage_id] = row
            layout.addWidget(row.group)
        self._build_stage_settings()

        layout.addWidget(self._build_phase_view())
        layout.addStretch(1)

        controls = QtWidgets.QHBoxLayout()
        self.run_button = QtWidgets.QPushButton("Run pipeline")
        self.run_button.clicked.connect(self._run)
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.setProperty("variant", "danger")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop)
        self.status_label = QtWidgets.QLabel("Ready")
        controls.addWidget(self.run_button)
        controls.addWidget(self.stop_button)
        controls.addWidget(self.status_label, 1)
        outer.addLayout(controls)

        self.pipeline_progress.connect(self._on_progress)
        self._load_settings()
        self._refresh_ui()

    # ------------------------------------------------------------------ UI
    def _build_layout_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Channel layout (shared by all stages)")
        grid = QtWidgets.QGridLayout(group)

        def spin(value: int, lo: int, hi: int) -> QtWidgets.QSpinBox:
            widget = QtWidgets.QSpinBox()
            widget.setRange(lo, hi)
            widget.setValue(value)
            return widget

        self.lay_channels = spin(20, 1, 100)
        self.lay_width = spin(15, 1, 101)
        self.lay_gap = spin(5, 0, 100)
        self.lay_center_gap_check = QtWidgets.QCheckBox("Centre gap (px)")
        self.lay_center_gap_check.setToolTip(
            "Widen the dark pad between the innermost x/w pair to reduce "
            "crosstalk between the two centre spectral bands. Applied to BOTH "
            "the step-3 measurement grid and the encoding layout."
        )
        self.lay_center_gap = spin(10, 0, 200)
        self.lay_center_gap.setEnabled(False)
        self.lay_center_gap_check.toggled.connect(self.lay_center_gap.setEnabled)
        self.lay_center_wl = QtWidgets.QDoubleSpinBox()
        self.lay_center_wl.setRange(700.0, 900.0)
        self.lay_center_wl.setDecimals(4)
        self.lay_center_wl.setValue(778.0)
        self.lay_use_center_fit = QtWidgets.QCheckBox(
            "Use TPA-centre fit downstream"
        )
        self.lay_use_center_fit.setChecked(True)
        self.lay_use_center_fit.setToolTip(
            "When a valid TPA centre fit is available (from this run or a "
            "file), pair_eta and comb_phase rebuild their layout at the "
            "fitted centre wavelength."
        )
        self.lay_guard_check = QtWidgets.QCheckBox("Rb guard bands (nm)")
        self.lay_guard_check.setChecked(True)
        self.lay_guard_centers = QtWidgets.QLineEdit("780.0, 776.0")
        self.lay_guard_half = QtWidgets.QDoubleSpinBox()
        self.lay_guard_half.setRange(0.001, 5.0)
        self.lay_guard_half.setDecimals(3)
        self.lay_guard_half.setValue(0.1)

        grid.addWidget(QtWidgets.QLabel("Channels/side"), 0, 0)
        grid.addWidget(self.lay_channels, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Width (px)"), 0, 2)
        grid.addWidget(self.lay_width, 0, 3)
        grid.addWidget(QtWidgets.QLabel("Gap (px)"), 0, 4)
        grid.addWidget(self.lay_gap, 0, 5)
        grid.addWidget(self.lay_center_gap_check, 1, 0)
        grid.addWidget(self.lay_center_gap, 1, 1)
        grid.addWidget(QtWidgets.QLabel("Centre λ (nm)"), 1, 2)
        grid.addWidget(self.lay_center_wl, 1, 3)
        grid.addWidget(self.lay_use_center_fit, 1, 4, 1, 2)
        grid.addWidget(self.lay_guard_check, 2, 0)
        grid.addWidget(self.lay_guard_centers, 2, 1, 1, 3)
        grid.addWidget(QtWidgets.QLabel("± half (nm)"), 2, 4)
        grid.addWidget(self.lay_guard_half, 2, 5)
        return group

    def _build_stage_settings(self) -> None:
        # ---- wl_map ---------------------------------------------------------
        widget = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(widget)
        grid.setContentsMargins(0, 0, 0, 0)
        self.wl_levels = _levels_row(grid, 0, "Levels", 0, 1023, 64)
        self.wl_window = QtWidgets.QSpinBox()
        self.wl_window.setRange(1, 200)
        self.wl_window.setValue(8)
        self.wl_stride = QtWidgets.QSpinBox()
        self.wl_stride.setRange(1, 256)
        self.wl_stride.setValue(1)
        self.wl_stride.setToolTip(
            "Measure every Nth column; the near-linear wavelength fit fills "
            "in the skipped columns (1 = measure every column)"
        )
        self.wl_peak_nm = QtWidgets.QDoubleSpinBox()
        self.wl_peak_nm.setRange(0.01, 20.0)
        self.wl_peak_nm.setValue(1.0)
        self.wl_peak_check = QtWidgets.QCheckBox("Peak ± (nm)")
        self.wl_peak_check.setChecked(True)
        self.wl_region_check = QtWidgets.QCheckBox("Region (px)")
        self.wl_region_start = QtWidgets.QSpinBox()
        self.wl_region_start.setRange(0, 4096)
        self.wl_region_end = QtWidgets.QSpinBox()
        self.wl_region_end.setRange(0, 4096)
        self.wl_region_end.setValue(1200)
        grid.addWidget(QtWidgets.QLabel("Window (px)"), 1, 0)
        grid.addWidget(self.wl_window, 1, 1)
        grid.addWidget(self.wl_peak_check, 1, 2)
        grid.addWidget(self.wl_peak_nm, 1, 3)
        grid.addWidget(QtWidgets.QLabel("Stride (px)"), 1, 4)
        grid.addWidget(self.wl_stride, 1, 5)
        grid.addWidget(self.wl_region_check, 2, 0)
        grid.addWidget(self.wl_region_start, 2, 1)
        grid.addWidget(self.wl_region_end, 2, 3)
        self.wl_osa = _OsaSettingsGroup()
        self.wl_outliers = _OutlierGroup()
        grid.addWidget(self.wl_osa, 3, 0, 1, 6)
        grid.addWidget(self.wl_outliers, 4, 0, 1, 6)
        self.rows["wl_map"].add_settings(widget)

        # ---- intensity ------------------------------------------------------
        widget = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(widget)
        grid.setContentsMargins(0, 0, 0, 0)
        self.int_levels = _levels_row(grid, 0, "Levels", 400, 900, 10)
        self.int_skip = QtWidgets.QSpinBox()
        self.int_skip.setRange(0, 10)
        self.int_skip.setValue(2)
        self.int_refine_center = QtWidgets.QCheckBox("Refine centre on OSA")
        self.int_refine_center.setChecked(True)
        self.int_refine_wl = QtWidgets.QCheckBox("Refine wavelengths")
        grid.addWidget(QtWidgets.QLabel("Group skip"), 1, 0)
        grid.addWidget(self.int_skip, 1, 1)
        grid.addWidget(self.int_refine_center, 1, 2, 1, 2)
        grid.addWidget(self.int_refine_wl, 1, 4, 1, 2)
        self.int_csv_edit = QtWidgets.QLineEdit()
        self.int_csv_edit.setPlaceholderText("optional CSV output path")
        csv_browse = QtWidgets.QPushButton("Browse…")
        csv_browse.setProperty("variant", "ghost")
        csv_browse.clicked.connect(
            lambda _=False: self._browse_save(self.int_csv_edit, "CSV (*.csv)")
        )
        grid.addWidget(QtWidgets.QLabel("CSV output"), 2, 0)
        grid.addWidget(self.int_csv_edit, 2, 1, 1, 4)
        grid.addWidget(csv_browse, 2, 5)
        self.int_osa = _OsaSettingsGroup()
        self.int_outliers = _OutlierGroup()
        grid.addWidget(self.int_osa, 3, 0, 1, 6)
        grid.addWidget(self.int_outliers, 4, 0, 1, 6)
        self.rows["intensity"].add_settings(widget)

        # ---- tpa_center -----------------------------------------------------
        widget = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(widget)
        grid.setContentsMargins(0, 0, 0, 0)
        self.ctr_halfspan = QtWidgets.QDoubleSpinBox()
        self.ctr_halfspan.setRange(0.001, 2.0)
        self.ctr_halfspan.setDecimals(4)
        self.ctr_halfspan.setValue(0.05)
        self.ctr_points = QtWidgets.QSpinBox()
        self.ctr_points.setRange(3, 201)
        self.ctr_points.setValue(11)
        self.ctr_pair = QtWidgets.QSpinBox()
        self.ctr_pair.setRange(0, 99)
        self.ctr_drive = QtWidgets.QDoubleSpinBox()
        self.ctr_drive.setRange(0.0, 1.0)
        self.ctr_drive.setSingleStep(0.05)
        self.ctr_drive.setValue(1.0)
        self.ctr_trials = QtWidgets.QSpinBox()
        self.ctr_trials.setRange(1, 100)
        self.ctr_repeats = QtWidgets.QSpinBox()
        self.ctr_repeats.setRange(1, 100)
        self.ctr_bg = QtWidgets.QCheckBox("Subtract background")
        self.ctr_bg.setChecked(True)
        grid.addWidget(QtWidgets.QLabel("Half-span (nm)"), 0, 0)
        grid.addWidget(self.ctr_halfspan, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Points"), 0, 2)
        grid.addWidget(self.ctr_points, 0, 3)
        grid.addWidget(QtWidgets.QLabel("Pair"), 0, 4)
        grid.addWidget(self.ctr_pair, 0, 5)
        grid.addWidget(QtWidgets.QLabel("Drive"), 1, 0)
        grid.addWidget(self.ctr_drive, 1, 1)
        grid.addWidget(QtWidgets.QLabel("Trials"), 1, 2)
        grid.addWidget(self.ctr_trials, 1, 3)
        grid.addWidget(QtWidgets.QLabel("Repeats"), 1, 4)
        grid.addWidget(self.ctr_repeats, 1, 5)
        grid.addWidget(self.ctr_bg, 2, 0, 1, 3)
        self.rows["tpa_center"].add_settings(widget)

        # ---- pair_eta -------------------------------------------------------
        widget = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(widget)
        grid.setContentsMargins(0, 0, 0, 0)
        self.eta_pairs = QtWidgets.QLineEdit("0, 3")
        self.eta_pairs.setToolTip("Comma-separated pair indices, or 'all'")
        self.eta_min = QtWidgets.QDoubleSpinBox()
        self.eta_min.setRange(0.0, 1.0)
        self.eta_min.setSingleStep(0.05)
        self.eta_min.setValue(0.3)
        self.eta_max = QtWidgets.QDoubleSpinBox()
        self.eta_max.setRange(0.0, 1.0)
        self.eta_max.setSingleStep(0.05)
        self.eta_max.setValue(1.0)
        self.eta_points = QtWidgets.QSpinBox()
        self.eta_points.setRange(2, 50)
        self.eta_points.setValue(5)
        self.eta_reduced = QtWidgets.QCheckBox("Reduced 1-D curves")
        self.eta_reduced.setChecked(True)
        self.eta_reduced.setToolTip(
            "x-only / w-only / cross lines instead of the full 2-D grid"
        )
        self.eta_trials = QtWidgets.QSpinBox()
        self.eta_trials.setRange(1, 100)
        self.eta_trials.setValue(5)
        self.eta_repeats = QtWidgets.QSpinBox()
        self.eta_repeats.setRange(1, 100)
        grid.addWidget(QtWidgets.QLabel("Pairs"), 0, 0)
        grid.addWidget(self.eta_pairs, 0, 1, 1, 3)
        grid.addWidget(self.eta_reduced, 0, 4, 1, 2)
        grid.addWidget(QtWidgets.QLabel("Sweep"), 1, 0)
        grid.addWidget(self.eta_min, 1, 1)
        grid.addWidget(QtWidgets.QLabel("to"), 1, 2)
        grid.addWidget(self.eta_max, 1, 3)
        grid.addWidget(QtWidgets.QLabel("points"), 1, 4)
        grid.addWidget(self.eta_points, 1, 5)
        grid.addWidget(QtWidgets.QLabel("Trials"), 2, 0)
        grid.addWidget(self.eta_trials, 2, 1)
        grid.addWidget(QtWidgets.QLabel("Repeats"), 2, 2)
        grid.addWidget(self.eta_repeats, 2, 3)
        csv_edit = QtWidgets.QLineEdit()
        csv_edit.setPlaceholderText("optional CSV output path")
        self.eta_csv_edit = csv_edit
        csv_browse = QtWidgets.QPushButton("Browse…")
        csv_browse.setProperty("variant", "ghost")
        csv_browse.clicked.connect(
            lambda _=False: self._browse_save(csv_edit, "CSV (*.csv)")
        )
        grid.addWidget(QtWidgets.QLabel("CSV output"), 3, 0)
        grid.addWidget(csv_edit, 3, 1, 1, 4)
        grid.addWidget(csv_browse, 3, 5)
        self.rows["pair_eta"].add_settings(widget)

        # ---- comb_phase -----------------------------------------------------
        widget = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(widget)
        grid.setContentsMargins(0, 0, 0, 0)
        self.ph_ref = QtWidgets.QSpinBox()
        self.ph_ref.setRange(0, 99)
        self.ph_targets = QtWidgets.QLineEdit("3")
        self.ph_targets.setToolTip("Comma-separated target pair indices")
        self.ph_points = QtWidgets.QSpinBox()
        self.ph_points.setRange(3, 200)
        self.ph_points.setValue(15)
        self.ph_start = QtWidgets.QDoubleSpinBox()
        self.ph_start.setRange(0.0, 180.0)
        self.ph_start.setValue(0.0)
        self.ph_stop = QtWidgets.QDoubleSpinBox()
        self.ph_stop.setRange(0.0, 180.0)
        self.ph_stop.setValue(180.0)
        self.ph_ref_phase = QtWidgets.QDoubleSpinBox()
        self.ph_ref_phase.setRange(0.0, 180.0)
        self.ph_ref_phase.setValue(180.0)
        self.ph_trials = QtWidgets.QSpinBox()
        self.ph_trials.setRange(1, 100)
        self.ph_trials.setValue(10)
        self.ph_repeats = QtWidgets.QSpinBox()
        self.ph_repeats.setRange(1, 100)
        self.ph_bound = QtWidgets.QDoubleSpinBox()
        self.ph_bound.setRange(0.01, 10.0)
        self.ph_bound.setSingleStep(0.1)
        self.ph_bound.setValue(1.0)
        self.ph_unconstrained = QtWidgets.QCheckBox("Unconstrained fit")
        self.ph_unconstrained.setToolTip(
            "Ignore the step-6 eta ratio lock (closed-form fit)"
        )
        self.ph_unconstrained.toggled.connect(
            lambda checked: self.ph_bound.setEnabled(not checked)
        )
        self.ph_single_beam = QtWidgets.QCheckBox("Step-6 single-beam background")
        self.ph_single_beam.setChecked(True)
        self.ph_dark = QtWidgets.QCheckBox("Measure dark")
        self.ph_dark.setChecked(True)
        self.ph_dark_per_trial = QtWidgets.QCheckBox("Dark per trial")
        self.ph_dark_per_trial.setChecked(True)
        grid.addWidget(QtWidgets.QLabel("Reference"), 0, 0)
        grid.addWidget(self.ph_ref, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Targets"), 0, 2)
        grid.addWidget(self.ph_targets, 0, 3)
        grid.addWidget(QtWidgets.QLabel("Points"), 0, 4)
        grid.addWidget(self.ph_points, 0, 5)
        grid.addWidget(QtWidgets.QLabel("φ start"), 1, 0)
        grid.addWidget(self.ph_start, 1, 1)
        grid.addWidget(QtWidgets.QLabel("φ stop"), 1, 2)
        grid.addWidget(self.ph_stop, 1, 3)
        grid.addWidget(QtWidgets.QLabel("Ref phase"), 1, 4)
        grid.addWidget(self.ph_ref_phase, 1, 5)
        grid.addWidget(QtWidgets.QLabel("Trials"), 2, 0)
        grid.addWidget(self.ph_trials, 2, 1)
        grid.addWidget(QtWidgets.QLabel("Repeats"), 2, 2)
        grid.addWidget(self.ph_repeats, 2, 3)
        grid.addWidget(QtWidgets.QLabel("Bound ±frac"), 3, 0)
        grid.addWidget(self.ph_bound, 3, 1)
        grid.addWidget(self.ph_unconstrained, 3, 2, 1, 2)
        grid.addWidget(self.ph_single_beam, 3, 4, 1, 2)
        grid.addWidget(self.ph_dark, 4, 0, 1, 2)
        grid.addWidget(self.ph_dark_per_trial, 4, 2, 1, 2)
        self.rows["comb_phase"].add_settings(widget)

    def _build_phase_view(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Step 7 results")
        layout = QtWidgets.QVBoxLayout(group)
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("Target pair"))
        self.phase_combo = QtWidgets.QComboBox()
        self.phase_combo.currentIndexChanged.connect(self._plot_phase_result)
        bar.addWidget(self.phase_combo)
        bar.addStretch(1)
        layout.addLayout(bar)
        self.phase_table = QtWidgets.QTableWidget(0, 6)
        self.phase_table.setHorizontalHeaderLabels(
            ["pair", "ΔΦ_comb (deg)", "± (deg)", "a", "b", "flags"]
        )
        self.phase_table.horizontalHeader().setStretchLastSection(True)
        self.phase_table.setMaximumHeight(140)
        layout.addWidget(self.phase_table)
        self.phase_figure = Figure(figsize=(9, 3.6), tight_layout=True)
        self.phase_canvas = FigureCanvas(self.phase_figure)
        self.phase_canvas.setMinimumHeight(240)
        layout.addWidget(self.phase_canvas)
        return group

    # -------------------------------------------------------------- helpers
    def _browse_open(self, edit: QtWidgets.QLineEdit) -> None:
        start = edit.text().strip() or str(_default_out_dir())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select input file", start,
            "Data files (*.json *.csv);;All files (*)",
        )
        if path:
            edit.setText(path)

    def _browse_save(
        self, edit: QtWidgets.QLineEdit, filter_: str = "JSON (*.json)"
    ) -> None:
        start = edit.text().strip() or str(_default_out_dir())
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Select output file", start, f"{filter_};;All files (*)"
        )
        if path:
            edit.setText(path)

    def _refresh_ui(self) -> None:
        """Dependency-aware input toggling: memory only from an enabled producer."""
        produced: set[str] = set()
        for spec in STAGES:
            row = self.rows[spec.stage_id]
            enabled = row.group.isChecked()
            for key, combo in row.input_combos.items():
                memory_index = combo.findText("From memory")
                model = combo.model()
                item = model.item(memory_index)
                memory_ok = key in produced
                item.setEnabled(memory_ok)
                if not memory_ok and combo.currentIndex() == memory_index:
                    combo.setCurrentText("From file…")
                is_file = combo.currentText() == "From file…"
                row.input_edits[key].setEnabled(is_file)
            if enabled:
                produced.add(spec.produces)

    @staticmethod
    def _parse_indices(text: str) -> list[int]:
        text = text.strip()
        if not text or text.lower() == "all":
            return []
        try:
            return [int(part) for part in text.replace(";", ",").split(",") if part.strip()]
        except ValueError as exc:
            raise ValueError(f"bad pair index list: {text!r}") from exc

    def _default_output(self, stage_id: str) -> Path:
        stamp = time.strftime("%m%d_%H%M")
        return _default_out_dir() / f"calib_pipe_{stage_id}_{stamp}.json"

    # ---------------------------------------------------------- request build
    def _layout_config(self) -> LayoutConfig:
        guard: tuple[tuple[float, float], ...] = ()
        if self.lay_guard_check.isChecked():
            centers = [
                float(part) for part in
                self.lay_guard_centers.text().replace(";", ",").split(",")
                if part.strip()
            ]
            half = self.lay_guard_half.value()
            guard = tuple((center, half) for center in centers)
        return LayoutConfig(
            n_channels=self.lay_channels.value(),
            channel_width_px=self.lay_width.value(),
            gap_px=self.lay_gap.value(),
            center_gap_px=(
                self.lay_center_gap.value()
                if self.lay_center_gap_check.isChecked() else None
            ),
            center_wl=self.lay_center_wl.value(),
            guard_bands=guard,
        )

    def _stage_config(self, stage_id: str, monitor_settle: float) -> Any:
        if stage_id == "wl_map":
            return WlMapConfig(
                levels=_levels_from(*self.wl_levels),
                window_size=self.wl_window.value(),
                coordinate_stride=self.wl_stride.value(),
                peak_half_window_nm=(
                    self.wl_peak_nm.value()
                    if self.wl_peak_check.isChecked() else None
                ),
                region=(
                    (self.wl_region_start.value(), self.wl_region_end.value())
                    if self.wl_region_check.isChecked() else None
                ),
                osa=self.wl_osa.to_settings(),
                outlier_policy=self.wl_outliers.policy(),
            )
        if stage_id == "intensity":
            return IntensityConfig(
                levels=_levels_from(*self.int_levels),
                window_size=self.lay_width.value(),
                group_skip_channels=self.int_skip.value(),
                refine_center=self.int_refine_center.isChecked(),
                refine_wavelength=self.int_refine_wl.isChecked(),
                osa=self.int_osa.to_settings(),
                outlier_policy=self.int_outliers.policy(),
            )
        if stage_id == "tpa_center":
            return TPACenterConfig(
                scan_center_nm=None,                  # scan around the layout centre
                scan_halfspan_nm=self.ctr_halfspan.value(),
                n_points=self.ctr_points.value(),
                pair_index=self.ctr_pair.value(),
                drive_level=self.ctr_drive.value(),
                n_trials=self.ctr_trials.value(),
                repeats=self.ctr_repeats.value(),
                settle=monitor_settle,
                subtract_background=self.ctr_bg.isChecked(),
            )
        if stage_id == "pair_eta":
            return PairEtaConfig(
                pair_indices=self._parse_indices(self.eta_pairs.text()),
                sweep_min=self.eta_min.value(),
                sweep_max=self.eta_max.value(),
                n_points=self.eta_points.value(),
                reduced_points=self.eta_reduced.isChecked(),
                n_trials=self.eta_trials.value(),
                repeats=self.eta_repeats.value(),
                settle=monitor_settle,
            )
        if stage_id == "comb_phase":
            targets = self._parse_indices(self.ph_targets.text())
            if not targets:
                raise ValueError("comb_phase needs at least one target pair index")
            return CombPhaseConfig(
                ref_index=self.ph_ref.value(),
                tgt_indices=targets,
                sweep_points=self.ph_points.value(),
                phi_start_deg=self.ph_start.value(),
                phi_stop_deg=self.ph_stop.value(),
                ref_phase_deg=self.ph_ref_phase.value(),
                n_trials=self.ph_trials.value(),
                repeats=self.ph_repeats.value(),
                settle=monitor_settle,
                bound_frac=(
                    None if self.ph_unconstrained.isChecked()
                    else self.ph_bound.value()
                ),
                single_beam_bg=self.ph_single_beam.isChecked(),
                measure_dark=self.ph_dark.isChecked(),
                dark_per_trial=self.ph_dark_per_trial.isChecked(),
            )
        raise ValueError(f"unknown stage {stage_id!r}")

    def _build_request(self, monitor_settle: float) -> PipelineRequest:
        stages: list[StagePlan] = []
        for spec in STAGES:
            row = self.rows[spec.stage_id]
            if not row.group.isChecked():
                continue
            inputs: dict[str, InputSpec] = {}
            for key in tuple(spec.requires) + tuple(spec.optional):
                source = row.input_spec(key)
                if source is not None:
                    inputs[key] = source
            output_text = row.output_edit.text().strip()
            output_path = (
                Path(output_text) if output_text
                else self._default_output(spec.stage_id)
            )
            if not output_text:
                row.output_edit.setText(str(output_path))
            extra: dict[str, Path] = {}
            if spec.stage_id == "intensity":
                csv_text = self.int_csv_edit.text().strip()
                if csv_text:
                    extra["csv"] = Path(csv_text)
            if spec.stage_id == "pair_eta":
                csv_text = self.eta_csv_edit.text().strip()
                if csv_text:
                    extra["csv"] = Path(csv_text)
            stages.append(
                StagePlan(
                    stage_id=spec.stage_id,
                    config=self._stage_config(spec.stage_id, monitor_settle),
                    inputs=inputs,
                    output_path=output_path,
                    extra_outputs=extra,
                )
            )
        return PipelineRequest(
            stages=stages,
            layout=self._layout_config(),
            col_ratio=self.host._active_col_ratio(),
            use_center_fit=self.lay_use_center_fit.isChecked(),
        )

    # ------------------------------------------------------------------- run
    def _run(self) -> None:
        host = self.host
        try:
            # a provisional settle; replaced below once the monitor is known
            request = self._build_request(0.15)
            needed = required_instruments(request)

            slm = host._controller()
            if not getattr(slm, "is_open", False):
                raise ValueError("Open the SLM on the SLM Control page first.")

            osa = None
            if "osa" in needed:
                osa = host._osa_ready()
                if osa is None:
                    raise ValueError(
                        "Connect the OSA on the Connections page first."
                    )

            monitor = None
            read_timeout = 30.0
            if "monitor" in needed:
                active = host._enc_active_monitor()
                if active is None:
                    raise ValueError(
                        "Connect the scope or DAQ first (Scope / DAQ page)."
                    )
                kind, monitor = active
                if kind == "scope":
                    settings = host._monitor_settings(trigger_mode="AUTO")
                else:
                    settings = host._daq_monitor_settings()
                settle = float(settings.hold)
                read_timeout = max(30.0, settings.duration * 3.0 + 10.0)
                monitor.configure_monitor(replace(settings, hold=0.0))
                request = self._build_request(settle)   # bake the real settle in

            validate_request(request)
        except ValueError as exc:
            self.status_label.setText(str(exc))
            return

        self._save_settings()
        self._phase_results.clear()
        self.phase_combo.clear()
        self.phase_table.setRowCount(0)
        for row in self.rows.values():
            row.status.setText(
                "queued" if row.group.isChecked() else "\N{EN DASH}"
            )

        stop_event = threading.Event()
        self._stop_event = stop_event
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("Running…")
        host._set_calibration_running(True)

        dialog = CalibrationProgressDialog(self, on_stop=self._stop)
        dialog.setStyleSheet(DARK_STYLESHEET)
        self._dialog = dialog
        dialog.show()

        instruments = PipelineInstruments(
            slm=slm, osa=osa, monitor=monitor,
            monitor_read_timeout=read_timeout,
        )

        def report(progress) -> None:
            self.pipeline_progress.emit(progress)

        def work() -> dict[str, Any]:
            try:
                outcome = run_pipeline(
                    request, instruments,
                    stop_event=stop_event, progress_callback=report,
                )
            except PipelineAborted as exc:
                return {
                    "status": "aborted",
                    "stage_id": exc.stage_id,
                    "saved_files": exc.saved_files,
                }
            except PipelineStageError as exc:
                return {
                    "status": "error",
                    "stage_id": exc.stage_id,
                    "message": str(exc.error),
                    "saved_files": exc.saved_files,
                }
            return {"status": "ok", "outcome": outcome}

        host._run_slm_task("Unified pipeline", work, self._finished, self._failed)

    def _stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
            self.status_label.setText("Stopping…")

    def _on_progress(self, progress) -> None:
        phase, step, total, message, x, y = plot_point(progress.inner)
        row = self.rows.get(progress.stage_id)
        if row is not None:
            row.status.setText(
                f"running [{progress.stage_index + 1}/{progress.n_stages}] "
                f"{message}"
            )
        if self._dialog is not None:
            self._dialog.update_progress(
                CalibrationProgress(
                    phase=phase, step=step, total=total,
                    message=f"[{progress.stage_id}] {message}", x=x, y=y,
                )
            )

    def _finish_common(self) -> None:
        self._stop_event = None
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.host._set_calibration_running(False)

    def _finished(self, payload: dict[str, Any]) -> None:
        self._finish_common()
        status = payload.get("status")
        for path in payload.get("saved_files", []):
            self.host._log(f"Pipeline saved: {path}")
        if status == "aborted":
            stage = payload.get("stage_id", "?")
            self.status_label.setText(f"Stopped during {stage}")
            self.rows[stage].status.setText("stopped")
            if self._dialog is not None:
                self._dialog.finish(False, f"Pipeline stopped during {stage}")
            return
        if status == "error":
            stage = payload.get("stage_id", "?")
            message = payload.get("message", "")
            self.status_label.setText(f"Failed in {stage}: {message}")
            self.rows[stage].status.setText(f"failed: {message}")
            if self._dialog is not None:
                self._dialog.finish(False, f"Stage {stage} failed: {message}")
            return

        outcome = payload["outcome"]
        for stage_id, summary in outcome.summaries.items():
            row = self.rows.get(stage_id)
            if row is not None:
                row.status.setText(f"done \N{MIDDLE DOT} {summary}")
        for path in outcome.saved_files:
            self.host._log(f"Pipeline saved: {path}")
        self.status_label.setText(
            "Done \N{MIDDLE DOT} " + "; ".join(
                f"{k}: {v}" for k, v in outcome.summaries.items()
            )
        )
        comb = outcome.artifacts.get("comb_phase")
        if comb is not None:
            self._show_phase_results(comb)
        if self._dialog is not None:
            self._dialog.finish(True, "Pipeline done")

    def _failed(self, _error: str) -> None:
        self._finish_common()
        self.status_label.setText("Pipeline failed (see Status log)")
        if self._dialog is not None:
            self._dialog.finish(False, "Pipeline failed")

    # ------------------------------------------------------------ step-7 view
    def _show_phase_results(self, comb: dict[str, Any]) -> None:
        self._phase_results = dict(comb.get("results", {}))
        phases: dict[int, dict[str, Any]] = comb.get("phases", {})
        ref_index = comb.get("ref_index", 0)

        self.phase_table.setRowCount(len(phases) + 1)
        self.phase_table.setItem(0, 0, QtWidgets.QTableWidgetItem(str(ref_index)))
        self.phase_table.setItem(0, 1, QtWidgets.QTableWidgetItem("0.00"))
        self.phase_table.setItem(0, 2, QtWidgets.QTableWidgetItem("—"))
        self.phase_table.setItem(0, 5, QtWidgets.QTableWidgetItem("reference"))
        for row_index, (k, entry) in enumerate(sorted(phases.items()), start=1):
            flags = []
            if entry.get("a_at_bound"):
                flags.append("a@bound")
            if entry.get("b_at_bound"):
                flags.append("b@bound")
            values = [
                str(k),
                f"{entry['dphi_comb_deg']:+.2f}",
                f"{entry['dphi_comb_err_deg']:.2f}",
                f"{entry['a']:.4g}",
                f"{entry['b']:.4g}",
                ", ".join(flags) or "ok",
            ]
            for col, value in enumerate(values):
                self.phase_table.setItem(
                    row_index, col, QtWidgets.QTableWidgetItem(value)
                )

        self.phase_combo.blockSignals(True)
        self.phase_combo.clear()
        for k in sorted(self._phase_results):
            self.phase_combo.addItem(f"pair {k}", k)
        self.phase_combo.blockSignals(False)
        if self.phase_combo.count():
            self.phase_combo.setCurrentIndex(0)
            self._plot_phase_result()

    def _plot_phase_result(self) -> None:
        k = self.phase_combo.currentData()
        result = self._phase_results.get(k)
        if result is None or result.fit is None:
            return
        plot_fringe(self.phase_figure, result.fit, k)
        self.phase_canvas.draw_idle()

    # ------------------------------------------------------------- persistence
    _PERSIST_SPINS = (
        "lay_channels", "lay_width", "lay_gap", "lay_center_gap",
        "wl_window", "wl_stride", "int_skip", "ctr_points", "ctr_pair", "ctr_trials",
        "ctr_repeats", "eta_points", "eta_trials", "eta_repeats",
        "ph_ref", "ph_points", "ph_trials", "ph_repeats",
    )
    _PERSIST_DSPINS = (
        "lay_center_wl", "lay_guard_half", "ctr_halfspan", "ctr_drive",
        "eta_min", "eta_max", "ph_start", "ph_stop", "ph_ref_phase", "ph_bound",
        "wl_peak_nm",
    )
    _PERSIST_CHECKS = (
        "lay_center_gap_check", "lay_use_center_fit", "lay_guard_check",
        "wl_peak_check", "wl_region_check", "int_refine_center",
        "int_refine_wl", "ctr_bg", "eta_reduced", "ph_unconstrained",
        "ph_single_beam", "ph_dark", "ph_dark_per_trial",
    )
    _PERSIST_EDITS = (
        "lay_guard_centers", "eta_pairs", "ph_targets", "int_csv_edit",
        "eta_csv_edit",
    )

    def _load_settings(self) -> None:
        settings = self._settings
        settings.beginGroup("pipeline_page")
        try:
            for name in self._PERSIST_SPINS:
                value = settings.value(name)
                if value is not None:
                    getattr(self, name).setValue(int(value))
            for name in self._PERSIST_DSPINS:
                value = settings.value(name)
                if value is not None:
                    getattr(self, name).setValue(float(value))
            for name in self._PERSIST_CHECKS:
                value = settings.value(name)
                if value is not None:
                    getattr(self, name).setChecked(value in (True, "true", "1"))
            for name in self._PERSIST_EDITS:
                value = settings.value(name)
                if value is not None:
                    getattr(self, name).setText(str(value))
            for stage_id, row in self.rows.items():
                enabled = settings.value(f"stage_{stage_id}_enabled")
                if enabled is not None:
                    row.group.setChecked(enabled in (True, "true", "1"))
                for key, combo in row.input_combos.items():
                    value = settings.value(f"stage_{stage_id}_{key}_source")
                    if value is not None and combo.findText(str(value)) >= 0:
                        combo.setCurrentText(str(value))
                    path = settings.value(f"stage_{stage_id}_{key}_path")
                    if path is not None:
                        row.input_edits[key].setText(str(path))
        finally:
            settings.endGroup()

    def _save_settings(self) -> None:
        settings = self._settings
        settings.beginGroup("pipeline_page")
        try:
            for name in self._PERSIST_SPINS + self._PERSIST_DSPINS:
                settings.setValue(name, getattr(self, name).value())
            for name in self._PERSIST_CHECKS:
                settings.setValue(name, getattr(self, name).isChecked())
            for name in self._PERSIST_EDITS:
                settings.setValue(name, getattr(self, name).text())
            for stage_id, row in self.rows.items():
                settings.setValue(
                    f"stage_{stage_id}_enabled", row.group.isChecked()
                )
                for key, combo in row.input_combos.items():
                    settings.setValue(
                        f"stage_{stage_id}_{key}_source", combo.currentText()
                    )
                    settings.setValue(
                        f"stage_{stage_id}_{key}_path",
                        row.input_edits[key].text(),
                    )
        finally:
            settings.endGroup()


__all__ = ["PipelinePage"]
