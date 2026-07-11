"""Qt-free orchestrator for the unified calibration pipeline.

Chains the full workflow

    wl_map (steps 1+2)  ->  intensity (step 3, OSA batch fast-channels)
        ->  tpa_center  ->  pair_eta (step 6)  ->  comb_phase (step 7)

with a uniform file interface: every enabled stage declares its input sources
(``memory`` = the artifact produced by an earlier enabled stage in the same
run, or ``file`` = load from disk) and an output path that is ALWAYS written
as soon as the stage finishes -- an abort or failure later in the chain never
loses completed work.

This module owns no hardware and no Qt.  The GUI builds a
:class:`PipelineRequest` + :class:`PipelineInstruments` and calls
:func:`run_pipeline` on a worker thread with the usual
``stop_event`` / ``progress_callback`` pair; every stage's native progress
object is forwarded wrapped in a :class:`PipelineProgress`.
"""
from __future__ import annotations

import json
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from osa_module.controller import MeasurementSettings

from .calibration.calibration_new import (
    CalibrationAborted,
    CalibrationResult,
    batch_intensity_calibration,
    build_channel_calibration_grid,
    find_min_max_intensity_levels,
    load_calibration_result,
    refine_center_coordinate_with_osa,
    save_calibration_result,
    wavelength_calibration,
    write_intensity_calibration_csv,
)
from .calibration.outliers import OutlierRemeasurePolicy
from .encoding import ChannelLayout, build_channel_layout
from .tpa_center import (
    TPACenterAborted,
    TPACenterResult,
    load_tpa_center_json,
    measure_center_scan,
    save_tpa_center_json,
)
from .tpa_pair import (
    TPAPairAborted,
    TPAPairResult,
    build_pair_points,
    build_sweep,
    measure_pair_grids,
    save_tpa_pair_json,
    write_tpa_pair_csv,
)
from .tpa_phase import (
    PairModel,
    PhaseResult,
    TPAPhaseAborted,
    build_phase_sweep,
    load_pair_models,
    measure_phase_sweep,
    save_phase_json,
    write_phase_csv,
)

_STAGE_ABORTS = (
    CalibrationAborted,
    TPACenterAborted,
    TPAPairAborted,
    TPAPhaseAborted,
)


class PipelineAborted(Exception):
    """A stage was stopped by request; earlier outputs are already on disk."""

    def __init__(self, stage_id: str, saved_files: list[Path]):
        super().__init__(f"pipeline stopped during stage {stage_id!r}")
        self.stage_id = stage_id
        self.saved_files = list(saved_files)


class PipelineStageError(Exception):
    """A stage failed; earlier outputs are already on disk."""

    def __init__(self, stage_id: str, error: BaseException, saved_files: list[Path]):
        super().__init__(f"stage {stage_id!r} failed: {error}")
        self.stage_id = stage_id
        self.error = error
        self.saved_files = list(saved_files)


# ======================================================================
# stage registry
# ======================================================================

@dataclass(frozen=True)
class StageSpec:
    stage_id: str
    label: str
    requires: tuple[str, ...]            # artifact keys that MUST be supplied
    optional: tuple[str, ...]            # artifact keys that MAY be supplied
    produces: str                        # artifact key this stage yields
    instruments: frozenset[str]          # {"osa", "slm", "monitor"}


STAGES: tuple[StageSpec, ...] = (
    StageSpec(
        "wl_map", "Steps 1+2 · Min/Max + Wavelength map",
        requires=(), optional=(), produces="wl_map",
        instruments=frozenset({"osa", "slm"}),
    ),
    StageSpec(
        "intensity", "Step 3 · Batch fast-channel intensity",
        requires=("wl_map",), optional=(), produces="intensity_calib",
        instruments=frozenset({"osa", "slm"}),
    ),
    StageSpec(
        "tpa_center", "TPA centre scan",
        requires=("intensity_calib",), optional=(), produces="center_fit",
        instruments=frozenset({"monitor", "slm"}),
    ),
    StageSpec(
        "pair_eta", "Step 6 · TPA pair efficiency (eta)",
        requires=("intensity_calib",), optional=("center_fit",),
        produces="pair_etas", instruments=frozenset({"monitor", "slm"}),
    ),
    StageSpec(
        "comb_phase", "Step 7 · Comb phase (dPhi_comb)",
        requires=("intensity_calib", "pair_etas"), optional=("center_fit",),
        produces="comb_phase", instruments=frozenset({"monitor", "slm"}),
    ),
)

STAGE_BY_ID: dict[str, StageSpec] = {spec.stage_id: spec for spec in STAGES}
_STAGE_ORDER: dict[str, int] = {spec.stage_id: i for i, spec in enumerate(STAGES)}


# ======================================================================
# configuration dataclasses
# ======================================================================

@dataclass
class LayoutConfig:
    """Channel-layout geometry shared by every layout-consuming stage.

    ``guard_bands`` are ``(center_nm, half_width_nm)`` pairs (the Rb lines);
    they feed :func:`build_channel_calibration_grid` directly and are converted
    to ``(lo, hi)`` ranges for :func:`build_channel_layout`, so measurement and
    encoding always agree.  ``center_gap_px`` (None = legacy) likewise reaches
    both sides.
    """

    n_channels: int = 20
    channel_width_px: int = 15
    gap_px: int = 5
    center_gap_px: int | None = None
    center_wl: float = 778.0
    guard_bands: tuple[tuple[float, float], ...] = ((780.0, 0.1), (776.0, 0.1))

    def dark_wl_bands(self) -> tuple[tuple[float, float], ...]:
        return tuple(
            (center - half, center + half) for center, half in self.guard_bands
        )

    def build_layout(
        self, calib: CalibrationResult, *, center_wl: float | None = None
    ) -> ChannelLayout:
        return build_channel_layout(
            calib,
            n_channels=self.n_channels,
            channel_width_px=self.channel_width_px,
            gap_px=self.gap_px,
            center_gap_px=self.center_gap_px,
            center_wl=self.center_wl if center_wl is None else float(center_wl),
            dark_wl_bands=self.dark_wl_bands(),
        )


@dataclass
class OSAStageSettings:
    """Per-stage OSA sweep parameters, incl. sampling points + sensitivity."""

    center_wl: str = "778nm"
    span: str = "8nm"
    sensitivity: str = "HIGH2"
    sampling_points: str = "AUTO"
    reference_level: str = "10uW"

    def to_measurement_settings(self) -> MeasurementSettings:
        return MeasurementSettings(
            center_wl=self.center_wl,
            span=self.span,
            sensitivity=self.sensitivity,
            sampling_points=self.sampling_points,
            reference_level=self.reference_level,
            y_unit="LINear",
        )


@dataclass
class WlMapConfig:
    levels: list[int] = field(default_factory=lambda: list(range(0, 1024, 64)))
    window_size: int = 8
    peak_half_window_nm: float | None = None
    region: tuple[int, int] | None = None
    osa: OSAStageSettings = field(default_factory=OSAStageSettings)
    outlier_policy: OutlierRemeasurePolicy | None = None


@dataclass
class IntensityConfig:
    levels: list[int] = field(default_factory=lambda: list(range(400, 901, 10)))
    window_size: int = 15
    wavelength_window_nm: float | None = None
    group_skip_channels: int = 2
    refine_center: bool = True
    refine_wavelength: bool = False
    osa: OSAStageSettings = field(default_factory=OSAStageSettings)
    outlier_policy: OutlierRemeasurePolicy | None = None


@dataclass
class TPACenterConfig:
    scan_center_nm: float | None = None      # None -> LayoutConfig.center_wl
    scan_halfspan_nm: float = 0.05
    n_points: int = 11
    pair_index: int = 0
    drive_level: float = 1.0
    n_trials: int = 1
    repeats: int = 1
    settle: float = 0.15
    subtract_background: bool = True


@dataclass
class PairEtaConfig:
    pair_indices: list[int] = field(default_factory=list)   # [] -> all pairs
    sweep_min: float = 0.3
    sweep_max: float = 1.0
    n_points: int = 5
    reduced_points: bool = True       # 1-D curves (x-only/w-only/cross) vs full grid
    n_trials: int = 5
    repeats: int = 1
    settle: float = 0.15


@dataclass
class CombPhaseConfig:
    ref_index: int = 0
    tgt_indices: list[int] = field(default_factory=lambda: [3])
    sweep_points: int = 15
    phi_start_deg: float = 0.0
    phi_stop_deg: float = 180.0
    ref_phase_deg: float = 180.0
    n_trials: int = 10
    repeats: int = 1
    settle: float = 0.15
    bound_frac: float | None = 1.0    # None -> unconstrained closed-form fit
    single_beam_bg: bool = True
    measure_dark: bool = True
    dark_per_trial: bool = True


_CONFIG_TYPES: dict[str, type] = {
    "wl_map": WlMapConfig,
    "intensity": IntensityConfig,
    "tpa_center": TPACenterConfig,
    "pair_eta": PairEtaConfig,
    "comb_phase": CombPhaseConfig,
}


# ======================================================================
# request / instruments / progress
# ======================================================================

@dataclass(frozen=True)
class InputSpec:
    source: Literal["memory", "file"]
    path: Path | None = None              # required when source == "file"


@dataclass
class StagePlan:
    stage_id: str
    config: Any
    inputs: dict[str, InputSpec]
    output_path: Path
    extra_outputs: dict[str, Path] = field(default_factory=dict)


@dataclass
class PipelineRequest:
    stages: list[StagePlan]
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    col_ratio: np.ndarray | None = None   # per-column encoding shape (TPA stages)
    use_center_fit: bool = True           # feed a valid centre fit downstream


@dataclass
class PipelineInstruments:
    """Hardware handles the enabled stages need (any may be None if unused).

    ``monitor`` must already be configured (``configure_monitor`` with hold=0);
    the TPA stages own their settle waits.
    """

    slm: Any
    osa: Any = None
    monitor: Any = None
    monitor_read_timeout: float = 30.0


@dataclass(frozen=True)
class PipelineProgress:
    stage_id: str
    stage_label: str
    stage_index: int                      # 0-based among the ENABLED stages
    n_stages: int
    inner: Any                            # the stage's native progress object


PipelineProgressCallback = Callable[[PipelineProgress], None]


@dataclass
class PipelineOutcome:
    artifacts: dict[str, Any]
    saved_files: list[Path]
    summaries: dict[str, str]             # stage_id -> one-line result summary


def plot_point(
    inner: Any,
) -> tuple[str, int, int, str, float | None, float | None]:
    """Map any stage's native progress onto (phase, step, total, message, x, y).

    Matches the CalibrationProgressDialog convention so one live plot can
    follow the whole pipeline.
    """
    phase = getattr(inner, "phase", None)
    step = int(getattr(inner, "step", 0))
    total = int(getattr(inner, "total", 0))
    message = str(getattr(inner, "message", ""))
    if phase is not None:                                  # CalibrationProgress
        return str(phase), step, total, message, inner.x, inner.y
    if hasattr(inner, "center_wl_nm"):                     # TPACenterProgress
        return ("tpa_center", step, total, message,
                inner.center_wl_nm, inner.signal_v)
    if hasattr(inner, "pair_index"):                       # TPAPairProgress
        eta = getattr(inner, "eta", None)
        return "pair_eta", step, total, message, float(step), eta
    if hasattr(inner, "dphi_comb"):                        # TPAPhaseProgress
        return "comb_phase", step, total, message, float(step), None
    return "pipeline", step, total, message, None, None


# ======================================================================
# validation
# ======================================================================

def required_instruments(request: PipelineRequest) -> frozenset[str]:
    """Union of instrument names the enabled stages need (for GUI pre-checks)."""
    needed: set[str] = set()
    for plan in request.stages:
        needed |= STAGE_BY_ID[plan.stage_id].instruments
    return frozenset(needed)


def validate_request(request: PipelineRequest) -> None:
    """Fail fast on anything that would die mid-run: order, inputs, outputs."""
    if not request.stages:
        raise ValueError("no pipeline stages selected")

    seen_ids: list[str] = []
    produced_earlier: set[str] = set()
    output_paths: set[Path] = set()
    for plan in request.stages:
        spec = STAGE_BY_ID.get(plan.stage_id)
        if spec is None:
            raise ValueError(f"unknown pipeline stage {plan.stage_id!r}")
        if plan.stage_id in seen_ids:
            raise ValueError(f"stage {plan.stage_id!r} appears twice")
        if seen_ids and _STAGE_ORDER[plan.stage_id] <= _STAGE_ORDER[seen_ids[-1]]:
            raise ValueError(
                f"stage {plan.stage_id!r} is out of order (canonical order: "
                + " -> ".join(s.stage_id for s in STAGES) + ")"
            )
        expected = _CONFIG_TYPES[plan.stage_id]
        if not isinstance(plan.config, expected):
            raise ValueError(
                f"stage {plan.stage_id!r} needs a {expected.__name__}, got "
                f"{type(plan.config).__name__}"
            )

        allowed = set(spec.requires) | set(spec.optional)
        for key in plan.inputs:
            if key not in allowed:
                raise ValueError(
                    f"stage {plan.stage_id!r} does not accept input {key!r}"
                )
        for key in spec.requires:
            if key not in plan.inputs:
                raise ValueError(
                    f"stage {plan.stage_id!r} is missing required input {key!r}"
                )
        for key, source in plan.inputs.items():
            if source.source == "memory":
                if key not in produced_earlier:
                    raise ValueError(
                        f"stage {plan.stage_id!r} wants {key!r} from memory, but "
                        f"no earlier enabled stage produces it"
                    )
            elif source.source == "file":
                if source.path is None:
                    raise ValueError(
                        f"stage {plan.stage_id!r} input {key!r}: file source "
                        f"needs a path"
                    )
                if not Path(source.path).is_file():
                    raise ValueError(
                        f"stage {plan.stage_id!r} input {key!r}: file not found: "
                        f"{source.path}"
                    )
            else:
                raise ValueError(
                    f"stage {plan.stage_id!r} input {key!r}: unknown source "
                    f"{source.source!r}"
                )

        out = Path(plan.output_path).resolve()
        for candidate in [out] + [
            Path(p).resolve() for p in plan.extra_outputs.values()
        ]:
            if candidate in output_paths:
                raise ValueError(f"output path used twice: {candidate}")
            output_paths.add(candidate)
            if not candidate.parent.exists():
                raise ValueError(
                    f"output directory does not exist: {candidate.parent}"
                )

        seen_ids.append(plan.stage_id)
        produced_earlier.add(spec.produces)


# ======================================================================
# execution
# ======================================================================

_FILE_LOADERS: dict[str, Callable[[Path], Any]] = {
    "wl_map": load_calibration_result,
    "intensity_calib": load_calibration_result,
    "center_fit": load_tpa_center_json,
    # pair_etas from file stays a Path: CSV re-fits need the layout, which
    # only exists inside the consuming stage (see _as_pair_models)
    "pair_etas": lambda path: Path(path),
}


@dataclass
class _Context:
    request: PipelineRequest
    instruments: PipelineInstruments
    stop_event: threading.Event | None
    progress_callback: PipelineProgressCallback | None
    artifacts: dict[str, Any] = field(default_factory=dict)
    saved_files: list[Path] = field(default_factory=list)
    summaries: dict[str, str] = field(default_factory=dict)
    stage_index: int = 0
    n_stages: int = 0

    def resolve(self, plan: StagePlan, key: str) -> Any:
        spec = plan.inputs.get(key)
        if spec is None:
            return None
        if spec.source == "memory":
            return self.artifacts[key]
        return _FILE_LOADERS[key](Path(spec.path))

    def forward(self, spec: StageSpec) -> Callable[[Any], None]:
        def _cb(inner: Any) -> None:
            if self.progress_callback is not None:
                self.progress_callback(
                    PipelineProgress(
                        stage_id=spec.stage_id,
                        stage_label=spec.label,
                        stage_index=self.stage_index,
                        n_stages=self.n_stages,
                        inner=inner,
                    )
                )
        return _cb

    def record(self, path: str | Path) -> Path:
        resolved = Path(path).resolve()
        self.saved_files.append(resolved)
        return resolved

    def center_wl(self, plan: StagePlan) -> float:
        """The centre wavelength downstream stages should build layouts at."""
        layout = self.request.layout
        if not self.request.use_center_fit:
            return layout.center_wl
        center_fit = self.resolve(plan, "center_fit")
        if center_fit is None:
            return layout.center_wl
        fit = getattr(center_fit, "fit", None)
        if fit is not None and fit.valid:
            return float(fit.center_wl_nm)
        return layout.center_wl


def _run_wl_map(ctx: _Context, plan: StagePlan) -> Any:
    cfg: WlMapConfig = plan.config
    osa = ctx.instruments.osa
    slm = ctx.instruments.slm
    settings = cfg.osa.to_measurement_settings()
    report = ctx.forward(STAGE_BY_ID["wl_map"])

    _min_i, _max_i, min_level, max_level, _records = find_min_max_intensity_levels(
        osa, slm, cfg.levels, settings,
        stop_event=ctx.stop_event, progress_callback=report,
    )
    seed = CalibrationResult(
        wavelength=np.asarray([]),
        coordinates=np.asarray([]),
        max_level=max_level,
        min_level=min_level,
        level_range=np.asarray(cfg.levels, dtype=int),
    )
    result = wavelength_calibration(
        osa, slm, [], settings, seed,
        window_size=cfg.window_size,
        peak_half_window_nm=cfg.peak_half_window_nm,
        region=cfg.region,
        outlier_policy=cfg.outlier_policy,
        stop_event=ctx.stop_event, progress_callback=report,
    )
    save_calibration_result(result, plan.output_path)
    ctx.record(plan.output_path)
    ctx.summaries["wl_map"] = (
        f"{result.coordinates.size} points, levels {min_level}..{max_level}"
    )
    return result


def _run_intensity(ctx: _Context, plan: StagePlan) -> Any:
    cfg: IntensityConfig = plan.config
    layout = ctx.request.layout
    osa = ctx.instruments.osa
    slm = ctx.instruments.slm
    settings = cfg.osa.to_measurement_settings()
    report = ctx.forward(STAGE_BY_ID["intensity"])
    wl_map: CalibrationResult = ctx.resolve(plan, "wl_map")

    center_coordinate = None
    if cfg.refine_center:
        center_coordinate, _measured_nm, _coarse = refine_center_coordinate_with_osa(
            osa, slm, settings, wl_map,
            target_wavelength_nm=layout.center_wl,
            window_size=cfg.window_size,
            stop_event=ctx.stop_event, progress_callback=report,
        )

    slm_width, _slm_height = slm.get_slm_info()
    grid_seed, _center = build_channel_calibration_grid(
        wl_map,
        target_wavelength_nm=layout.center_wl,
        center_coordinate=center_coordinate,
        n_channels_per_side=layout.n_channels,
        channel_width_px=layout.channel_width_px,
        gap_px=layout.gap_px,
        center_gap_px=layout.center_gap_px,
        slm_width=int(slm_width),
        guard_bands_nm=layout.guard_bands,
    )
    result = batch_intensity_calibration(
        osa, slm, cfg.levels, settings, grid_seed,
        cfg.window_size,
        wavelength_window_nm=cfg.wavelength_window_nm,
        group_skip_channels=cfg.group_skip_channels,
        guard_bands_nm=layout.guard_bands,
        refine_wavelength=cfg.refine_wavelength,
        outlier_policy=cfg.outlier_policy,
        stop_event=ctx.stop_event, progress_callback=report,
    )
    save_calibration_result(result, plan.output_path)
    ctx.record(plan.output_path)
    csv_path = plan.extra_outputs.get("csv")
    if csv_path is not None:
        write_intensity_calibration_csv(result, csv_path)
        ctx.record(csv_path)
    ctx.summaries["intensity"] = (
        f"{result.coordinates.size} channels x {len(cfg.levels)} levels"
    )
    return result


def _run_tpa_center(ctx: _Context, plan: StagePlan) -> Any:
    cfg: TPACenterConfig = plan.config
    layout = ctx.request.layout
    report = ctx.forward(STAGE_BY_ID["tpa_center"])
    calib: CalibrationResult = ctx.resolve(plan, "intensity_calib")

    scan_center = (
        layout.center_wl if cfg.scan_center_nm is None else float(cfg.scan_center_nm)
    )
    centers = np.linspace(
        scan_center - cfg.scan_halfspan_nm,
        scan_center + cfg.scan_halfspan_nm,
        int(cfg.n_points),
    )
    result = measure_center_scan(
        ctx.instruments.monitor, ctx.instruments.slm, calib,
        center_wavelengths_nm=centers,
        n_channels=layout.n_channels,
        channel_width_px=layout.channel_width_px,
        gap_px=layout.gap_px,
        center_gap_px=layout.center_gap_px,
        pair_index=cfg.pair_index,
        drive_level=cfg.drive_level,
        n_trials=cfg.n_trials,
        repeats=cfg.repeats,
        settle=cfg.settle,
        read_timeout=ctx.instruments.monitor_read_timeout,
        col_ratio=ctx.request.col_ratio,
        subtract_background=cfg.subtract_background,
        stop_event=ctx.stop_event, progress_callback=report,
    )
    save_tpa_center_json(result, plan.output_path)
    ctx.record(plan.output_path)
    fit = result.fit
    if fit is not None and fit.valid:
        ctx.summaries["tpa_center"] = (
            f"centre {fit.center_wl_nm:.4f} +/- {fit.center_wl_err_nm:.4f} nm"
        )
    else:
        message = fit.message if fit is not None else "no fit"
        ctx.summaries["tpa_center"] = f"fit invalid: {message}"
    return result


def _run_pair_eta(ctx: _Context, plan: StagePlan) -> Any:
    cfg: PairEtaConfig = plan.config
    report = ctx.forward(STAGE_BY_ID["pair_eta"])
    calib: CalibrationResult = ctx.resolve(plan, "intensity_calib")
    layout_obj = ctx.request.layout.build_layout(
        calib, center_wl=ctx.center_wl(plan)
    )

    indices = list(cfg.pair_indices) or list(range(layout_obj.n_channels))
    for index in indices:
        if not (0 <= index < layout_obj.n_channels):
            raise ValueError(
                f"pair index {index} out of range "
                f"(layout has {layout_obj.n_channels} pairs)"
            )
    sweep = build_sweep(cfg.sweep_min, cfg.sweep_max, cfg.n_points)
    points = (
        build_pair_points(cfg.sweep_min, cfg.sweep_max, cfg.n_points)
        if cfg.reduced_points else None
    )
    result = measure_pair_grids(
        ctx.instruments.monitor, ctx.instruments.slm, layout_obj,
        pair_indices=indices, sweep=sweep, points=points,
        n_trials=cfg.n_trials, repeats=cfg.repeats, settle=cfg.settle,
        read_timeout=ctx.instruments.monitor_read_timeout,
        col_ratio=ctx.request.col_ratio,
        stop_event=ctx.stop_event, progress_callback=report,
    )
    save_tpa_pair_json(result, plan.output_path)
    ctx.record(plan.output_path)
    csv_path = plan.extra_outputs.get("csv")
    if csv_path is not None:
        write_tpa_pair_csv(result, csv_path)
        ctx.record(csv_path)
    etas = ", ".join(
        f"eta[{grid.index}]={grid.fit.eta:.4g}"
        for grid in result.channels if grid.fit is not None
    )
    ctx.summaries["pair_eta"] = etas or "no fits"
    return result


def _as_pair_models(
    artifact: Any, layout_obj: ChannelLayout
) -> dict[int, PairModel]:
    """Normalise a pair_etas artifact (memory result or file path) to models."""
    if isinstance(artifact, TPAPairResult):
        return {
            grid.index: PairModel.from_fit(grid.index, grid.fit)
            for grid in artifact.channels if grid.fit is not None
        }
    if isinstance(artifact, (str, Path)):
        return load_pair_models(artifact, layout=layout_obj)
    raise TypeError(
        f"cannot build pair models from {type(artifact).__name__}"
    )


def _run_comb_phase(ctx: _Context, plan: StagePlan) -> Any:
    cfg: CombPhaseConfig = plan.config
    report = ctx.forward(STAGE_BY_ID["comb_phase"])
    calib: CalibrationResult = ctx.resolve(plan, "intensity_calib")
    layout_obj = ctx.request.layout.build_layout(
        calib, center_wl=ctx.center_wl(plan)
    )
    models = _as_pair_models(ctx.resolve(plan, "pair_etas"), layout_obj)

    needed = [("reference", cfg.ref_index)] + [
        ("target", k) for k in cfg.tgt_indices
    ]
    for role, index in needed:
        if index not in models:
            raise ValueError(
                f"no step-6 model for {role} pair {index}; found {sorted(models)}"
            )

    drive = build_phase_sweep(
        n_points=cfg.sweep_points,
        phi_start_deg=cfg.phi_start_deg,
        phi_stop_deg=cfg.phi_stop_deg,
        ref_phase_deg=cfg.ref_phase_deg,
    )
    out = Path(plan.output_path)
    spectrum: dict[int, dict[str, Any]] = {}
    results: dict[int, PhaseResult] = {}
    for k in cfg.tgt_indices:
        result = measure_phase_sweep(
            ctx.instruments.monitor, ctx.instruments.slm, layout_obj,
            tgt_index=k, ref_index=cfg.ref_index,
            drive=drive, tgt_model=models[k], ref_model=models[cfg.ref_index],
            n_trials=cfg.n_trials, repeats=cfg.repeats, settle=cfg.settle,
            read_timeout=ctx.instruments.monitor_read_timeout,
            measure_dark=cfg.measure_dark, dark_per_trial=cfg.dark_per_trial,
            col_ratio=ctx.request.col_ratio,
            frac=cfg.bound_frac, single_beam_bg=cfg.single_beam_bg,
            stop_event=ctx.stop_event, progress_callback=report,
        )
        results[k] = result
        csv_path = out.with_name(f"{out.stem}_pair{k}.csv")
        json_path = out.with_name(f"{out.stem}_pair{k}.json")
        write_phase_csv(result, csv_path)
        ctx.record(csv_path)
        save_phase_json(result, json_path)
        ctx.record(json_path)
        fit = result.fit
        spectrum[k] = {
            "dphi_comb_deg": fit.dphi_comb_deg,
            "dphi_comb_err_deg": float(np.degrees(fit.dphi_comb_err)),
            "a": fit.a,
            "b": fit.b,
            "a_at_bound": fit.a_at_bound,
            "b_at_bound": fit.b_at_bound,
            "chi2_red": fit.chi2_red,
            "csv": str(csv_path),
        }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"ref_index": cfg.ref_index, "phases": spectrum}, indent=2
        ),
        encoding="utf-8",
    )
    ctx.record(out)
    ctx.summaries["comb_phase"] = "; ".join(
        f"Phi[{k}]={entry['dphi_comb_deg']:+.2f}"
        f"+/-{entry['dphi_comb_err_deg']:.2f} deg"
        for k, entry in spectrum.items()
    )
    return {"ref_index": cfg.ref_index, "phases": spectrum, "results": results}


_STAGE_RUNNERS: dict[str, Callable[[_Context, StagePlan], Any]] = {
    "wl_map": _run_wl_map,
    "intensity": _run_intensity,
    "tpa_center": _run_tpa_center,
    "pair_eta": _run_pair_eta,
    "comb_phase": _run_comb_phase,
}


def run_pipeline(
    request: PipelineRequest,
    instruments: PipelineInstruments,
    *,
    stop_event: threading.Event | None = None,
    progress_callback: PipelineProgressCallback | None = None,
) -> PipelineOutcome:
    """Run the enabled stages in order, saving each result before the next.

    Raises :class:`PipelineAborted` when a stage is stopped by request and
    :class:`PipelineStageError` on any other stage failure; both carry the
    files already saved, which stay on disk.
    """
    validate_request(request)
    for name in sorted(required_instruments(request)):
        if getattr(instruments, name, None) is None:
            raise ValueError(f"the selected stages need the {name!r} instrument")

    ctx = _Context(
        request=request,
        instruments=instruments,
        stop_event=stop_event,
        progress_callback=progress_callback,
        n_stages=len(request.stages),
    )
    for index, plan in enumerate(request.stages):
        ctx.stage_index = index
        spec = STAGE_BY_ID[plan.stage_id]
        try:
            artifact = _STAGE_RUNNERS[plan.stage_id](ctx, plan)
        except _STAGE_ABORTS as exc:
            raise PipelineAborted(plan.stage_id, ctx.saved_files) from exc
        except (PipelineAborted, PipelineStageError):
            raise
        except Exception as exc:
            raise PipelineStageError(plan.stage_id, exc, ctx.saved_files) from exc
        ctx.artifacts[spec.produces] = artifact

    return PipelineOutcome(
        artifacts=ctx.artifacts,
        saved_files=ctx.saved_files,
        summaries=ctx.summaries,
    )


__all__ = [
    "CombPhaseConfig",
    "InputSpec",
    "IntensityConfig",
    "LayoutConfig",
    "OSAStageSettings",
    "PairEtaConfig",
    "PipelineAborted",
    "PipelineInstruments",
    "PipelineOutcome",
    "PipelineProgress",
    "PipelineRequest",
    "PipelineStageError",
    "STAGES",
    "STAGE_BY_ID",
    "StagePlan",
    "StageSpec",
    "TPACenterConfig",
    "WlMapConfig",
    "plot_point",
    "required_instruments",
    "run_pipeline",
    "validate_request",
]
