"""Hardware-in-the-loop optimisation of a symmetric SLM intensity profile.

The optimisation variables are *intensity* ratios.  For a requested channel
amplitude ``a``, a channel-level LUT produces a scalar intensity command
``u = T(a)`` and column ``j`` requests intensity ``u * l[j]`` from the existing
per-column calibration.  Initial-profile generation and file/model loading are
outside this module.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from osa_module.controller import MeasurementSettings, OSAController, TraceData
from osa_module.driver import OSAError

from .controller import SLMController
from .encoding import ChannelLayout, EncodingChannel, encode_to_pattern

_trapz = getattr(np, "trapezoid", None) or np.trapz
_OFFSETS = (-2, -1, 0, 1, 2)


class OptimizationAborted(Exception):
    """Raised when a hardware optimisation is stopped by the caller."""


@dataclass(frozen=True)
class EvaluationSample:
    """One hardware evaluation's headline metrics, for live plotting.

    ``eta``/``c_total`` are the main-anchor (offset 0) values of the candidate
    just measured; ``evaluation`` is its 1-based index within ``stage``.
    """

    stage: str
    evaluation: int
    loss: float
    eta: float | None
    c_total: float | None
    feasible: bool


@dataclass(frozen=True)
class OptimizationProgress:
    stage: str
    step: int
    total: int
    message: str
    best_loss: float | None = None
    # metrics of the evaluation that triggered this report (None for
    # book-keeping reports such as baseline / setup)
    sample: EvaluationSample | None = None
    # {"loss", "eta", "c_total"} of the flat (all-ones) profile, sent once
    # early in the run so a live plot can draw the comparison reference lines
    flat_reference: dict[str, float] | None = None


ProgressCallback = Callable[[OptimizationProgress], None]


def validate_independent_profile(values: Sequence[float], width: int = 15) -> np.ndarray:
    """Validate the independent half-profile of normalised intensity ratios."""
    profile = np.asarray(values, dtype=float)
    expected = (int(width) + 1) // 2
    if profile.shape != (expected,):
        raise ValueError(
            f"initial intensity profile must have shape ({expected},), got {profile.shape}"
        )
    if not np.all(np.isfinite(profile)):
        raise ValueError("initial intensity profile contains NaN or infinity")
    if np.any((profile < 0.0) | (profile > 1.0)):
        raise ValueError("all intensity ratios must be in [0, 1]")
    return profile.copy()


def mirror_intensity_profile(values: Sequence[float], width: int = 15) -> np.ndarray:
    """Mirror independent intensity ratios into a full symmetric profile."""
    half = validate_independent_profile(values, width)
    if width % 2:
        return np.concatenate((half, half[-2::-1]))
    return np.concatenate((half, half[::-1]))


def independent_intensity_profile(values: Sequence[float]) -> np.ndarray:
    """Extract and validate the independent half of a symmetric full profile."""
    full = np.asarray(values, dtype=float)
    if full.ndim != 1 or full.size < 1:
        raise ValueError("full intensity profile must be a non-empty 1-D array")
    half_n = (full.size + 1) // 2
    half = validate_independent_profile(full[:half_n], full.size)
    if not np.allclose(mirror_intensity_profile(half, full.size), full, atol=1e-9):
        raise ValueError("full intensity profile is not symmetric")
    return half


def round_encoding_profile(values: Sequence[float], threshold: float = 0.99) -> np.ndarray:
    """Snap intensity ratios above ``threshold`` up to exactly 1.0.

    The live OSA optimiser leaves near-flat columns at values such as 0.998; the
    difference quantises to the same SLM level as 1.0 anyway, so rounding gives a
    clean flat top and a profile that is easy to read and to reason about.
    """
    profile = np.asarray(values, dtype=float).copy()
    profile[profile > float(threshold)] = 1.0
    return profile


# Learned encoding shape from encoding/2026-07-03_run162902/best_so_far.json
# (stage3_rerank: the crosstalk + modulation-fidelity optimum), with every value
# > 0.99 rounded up to 1.0. These are the eight independent intensity ratios for
# the 15-pixel channel; ``mirror_intensity_profile`` expands them to the full,
# symmetric per-column profile used by ``encode_to_pattern(col_ratio=...)``.
OPTIMIZED_ENCODING_SHAPE = round_encoding_profile(
    (
        0.3847935183091822,
        0.6138519649340355,
        0.7910183260349569,
        1.0,
        0.9981799670282598,
        1.0,
        0.9968928821893347,
        1.0,
    )
)

# The trivial rectangular ("flat band") encoding: the A/B baseline the optimised
# shape is compared against.
FLAT_ENCODING_SHAPE = np.ones(OPTIMIZED_ENCODING_SHAPE.shape[0], dtype=float)


def amplitudes_to_intensity_commands(
    x_amplitudes: Sequence[float],
    w_amplitudes: Sequence[float],
    layout: ChannelLayout,
    references: dict[int, "StageAmplitudeReference"],
) -> tuple[np.ndarray, np.ndarray]:
    """Map per-channel target amplitudes through the nearest anchor LUT.

    The learned intensity profile is universal, while the measured scalar LUTs
    may vary with position. Channels between the three measured anchors use the
    LUT from the nearest anchor in channel-index space.
    """
    x_values = np.asarray(x_amplitudes, dtype=float)
    w_values = np.asarray(w_amplitudes, dtype=float)
    expected = (layout.n_channels,)
    if x_values.shape != expected or w_values.shape != expected:
        raise ValueError(f"amplitude arrays must each have shape {expected}")
    if not references:
        raise ValueError("at least one final amplitude LUT is required")
    if (
        not np.all(np.isfinite(x_values)) or not np.all(np.isfinite(w_values))
        or np.any((x_values < 0.0) | (x_values > 1.0))
        or np.any((w_values < 0.0) | (w_values > 1.0))
    ):
        raise ValueError("target amplitudes must be finite and in [0, 1]")

    ordered = sorted(layout.all_channels, key=lambda channel: channel.wavelength_nm)
    center_position = len(ordered) // 2
    anchor_offsets = np.asarray(sorted(references), dtype=int)
    x_commands = np.empty_like(x_values)
    w_commands = np.empty_like(w_values)
    for position, channel in enumerate(ordered):
        relative = position - center_position
        nearest = int(anchor_offsets[np.argmin(np.abs(anchor_offsets - relative))])
        amplitude = x_values[channel.index] if channel.side == "x" else w_values[channel.index]
        command = float(references[nearest].lut.command_for(amplitude))
        if channel.side == "x":
            x_commands[channel.index] = command
        else:
            w_commands[channel.index] = command
    return x_commands, w_commands


def efficiency_penalty(eta: float, threshold: float = 0.87, scale: float = 0.02) -> float:
    return (max(0.0, float(threshold) - float(eta)) / float(scale)) ** 2


def stage1_loss(
    crosstalk: Sequence[float],
    eta: Sequence[float],
    c_init: Sequence[float],
    *,
    c_floor: float,
    beta_eta: float = 1.0,
    eta_threshold: float = 0.87,
    eta_scale: float = 0.02,
) -> float:
    """The exact Stage-1 objective from the optimisation plan."""
    c = np.asarray(crosstalk, dtype=float)
    e = np.asarray(eta, dtype=float)
    base = np.asarray(c_init, dtype=float)
    if c.shape != e.shape or c.shape != base.shape or c.size == 0:
        raise ValueError("Stage-1 metric arrays must be non-empty and have equal shape")
    c_norm = c / np.maximum(base, float(c_floor))
    penalties = [efficiency_penalty(v, eta_threshold, eta_scale) for v in e]
    return float(np.mean(c_norm) + beta_eta * np.mean(penalties))


def stage3_loss(
    rmse: Sequence[float],
    crosstalk: Sequence[float],
    eta: Sequence[float],
    rmse_stage1: Sequence[float],
    c_stage1: Sequence[float],
    *,
    sigma_a: float,
    c_floor: float,
    beta_c: float = 1.0,
    beta_eta: float = 1.0,
    eta_threshold: float = 0.87,
    eta_scale: float = 0.02,
) -> float:
    """The exact fixed-LUT Stage-3 objective from the optimisation plan."""
    r = np.asarray(rmse, dtype=float)
    c = np.asarray(crosstalk, dtype=float)
    e = np.asarray(eta, dtype=float)
    r0 = np.asarray(rmse_stage1, dtype=float)
    c0 = np.asarray(c_stage1, dtype=float)
    if not (r.shape == c.shape == e.shape == r0.shape == c0.shape) or r.size == 0:
        raise ValueError("Stage-3 metric arrays must be non-empty and have equal shape")
    e_norm = r / np.maximum(r0, float(sigma_a))
    ratio = c / np.maximum(c0, float(c_floor))
    c_guard = (np.maximum(0.0, ratio - 1.05) / 0.05) ** 2
    eta_penalty = np.array(
        [efficiency_penalty(v, eta_threshold, eta_scale) for v in e], dtype=float
    )
    return float(
        np.mean(e_norm) + beta_c * np.mean(c_guard) + beta_eta * np.mean(eta_penalty)
    )


@dataclass(frozen=True)
class ChannelRef:
    side: str
    index: int
    wavelength_nm: float

    @property
    def key(self) -> str:
        return f"{self.side}{self.index}"


@dataclass
class FixedChannelBins:
    """Five fixed, non-overlapping OSA bins centred on an anchor channel."""

    anchor_key: str
    centers_nm: np.ndarray
    edges_nm: np.ndarray

    def __post_init__(self) -> None:
        self.centers_nm = np.asarray(self.centers_nm, dtype=float)
        self.edges_nm = np.asarray(self.edges_nm, dtype=float)
        if self.centers_nm.shape != (5,) or self.edges_nm.shape != (6,):
            raise ValueError("fixed channel bins require five centers and six edges")
        if not np.all(np.diff(self.centers_nm) > 0.0):
            raise ValueError("channel centers must be strictly increasing")
        if not np.all(np.diff(self.edges_nm) > 0.0):
            raise ValueError("channel edges must be strictly increasing")

    @classmethod
    def from_centers(cls, anchor_key: str, centers_nm: Sequence[float]) -> "FixedChannelBins":
        centers = np.asarray(centers_nm, dtype=float)
        if centers.shape != (5,) or not np.all(np.diff(centers) > 0.0):
            raise ValueError("five strictly increasing centers are required")
        edges = np.empty(6, dtype=float)
        edges[1:5] = (centers[:-1] + centers[1:]) / 2.0
        edges[0] = centers[0] - (centers[1] - centers[0]) / 2.0
        edges[5] = centers[4] + (centers[4] - centers[3]) / 2.0
        return cls(anchor_key=anchor_key, centers_nm=centers, edges_nm=edges)

    def integrate(self, trace: TraceData, dark: TraceData | None = None) -> np.ndarray:
        """Integrate signed dark-subtracted power, then clamp each bin at zero."""
        wl = np.asarray(trace.wavelengths_nm, dtype=float)
        signal = trace_power_w(trace)
        if dark is not None:
            dark_wl = np.asarray(dark.wavelengths_nm, dtype=float)
            dark_power = trace_power_w(dark)
            signal = signal - np.interp(wl, dark_wl, dark_power, left=dark_power[0], right=dark_power[-1])
        result = np.zeros(5, dtype=float)
        for i, (lo, hi) in enumerate(zip(self.edges_nm[:-1], self.edges_nm[1:])):
            mask = (wl >= lo) & (wl <= hi if i == 4 else wl < hi)
            if np.count_nonzero(mask) >= 2:
                result[i] = max(float(_trapz(signal[mask], wl[mask])), 0.0)
        return result


def trace_power_w(trace: TraceData) -> np.ndarray:
    powers = np.asarray(trace.powers, dtype=float)
    if trace.power_label == "power_dBm":
        powers = 1e-3 * 10.0 ** (powers / 10.0)
    return np.nan_to_num(powers, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass
class AmplitudeLUT:
    """Monotone inverse LUT: desired channel amplitude -> intensity command."""

    commands: np.ndarray
    measured_amplitudes: np.ndarray

    def __post_init__(self) -> None:
        commands = np.asarray(self.commands, dtype=float)
        amplitudes = np.asarray(self.measured_amplitudes, dtype=float)
        if commands.ndim != 1 or amplitudes.shape != commands.shape or commands.size < 2:
            raise ValueError("LUT commands and amplitudes must be equal 1-D arrays")
        order = np.argsort(commands)
        commands = np.clip(commands[order], 0.0, 1.0)
        amplitudes = np.maximum.accumulate(np.maximum(amplitudes[order], 0.0))
        if np.any(np.diff(commands) <= 0.0):
            raise ValueError("LUT commands must be unique")
        # Endpoint definitions come directly from E_blank and E_full.
        amplitudes[0] = 0.0
        amplitudes[-1] = max(amplitudes[-1], 1.0)
        self.commands = commands
        self.measured_amplitudes = amplitudes

    def command_for(self, amplitude: float | np.ndarray) -> float | np.ndarray:
        target = np.clip(np.asarray(amplitude, dtype=float), 0.0, 1.0)
        # Remove flat sections before inversion; use the lowest command that
        # reaches each measured amplitude.
        keep = np.r_[True, np.diff(self.measured_amplitudes) > 1e-12]
        amp = self.measured_amplitudes[keep]
        cmd = self.commands[keep]
        out = np.interp(target, amp, cmd, left=0.0, right=1.0)
        out = np.where(target <= 0.0, 0.0, np.where(target >= 1.0, 1.0, out))
        if np.ndim(amplitude) == 0:
            return float(out)
        return out

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "commands": self.commands.tolist(),
            "measured_amplitudes": self.measured_amplitudes.tolist(),
        }


@dataclass
class AnchorCalibration:
    offset: int
    position: int
    anchor: ChannelRef
    settings: MeasurementSettings
    bins: FixedChannelBins
    dark_trace: TraceData
    flat_main_energy: float
    reference_version: int = 1


@dataclass
class EnergyMeasurement:
    energies: np.ndarray
    pattern_hash: str
    trace_path: str | None = None

    @property
    def main(self) -> float:
        return float(self.energies[2])


@dataclass
class AnchorMetrics:
    anchor_offset: int
    c_total: float
    eta: float
    reference_version: int = 0
    rmse: float = math.nan
    bias: float = math.nan
    mae: float = math.nan
    p95: float = math.nan
    max_error: float = math.nan
    amplitude_errors: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "anchor_offset": self.anchor_offset,
            "c_total": self.c_total,
            "eta": self.eta,
            "reference_version": self.reference_version,
            "rmse": self.rmse,
            "bias": self.bias,
            "mae": self.mae,
            "p95": self.p95,
            "max_error": self.max_error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "AnchorMetrics":
        return cls(
            anchor_offset=int(data["anchor_offset"]),
            c_total=float(data["c_total"]),
            eta=float(data["eta"]),
            reference_version=int(data.get("reference_version", 0)),
            rmse=float(data.get("rmse", math.nan)),
            bias=float(data.get("bias", math.nan)),
            mae=float(data.get("mae", math.nan)),
            p95=float(data.get("p95", math.nan)),
            max_error=float(data.get("max_error", math.nan)),
        )


@dataclass
class StageAmplitudeReference:
    lut: AmplitudeLUT
    e_blank: float
    e_full: float

    def to_dict(self) -> dict[str, object]:
        return {"lut": self.lut.to_dict(), "e_blank": self.e_blank, "e_full": self.e_full}


@dataclass
class OSAOptimizationConfig:
    """Fixed configuration for one complete hardware optimisation run."""

    settings: MeasurementSettings = field(
        default_factory=lambda: MeasurementSettings(
            center_wl="778nm", span="0.8nm", sensitivity="HIGH2",
            sampling_points="1001", y_unit="LINear", reference_level="10uW"
        )
    )
    anchor_offsets: tuple[int, ...] = (0, -10, 10)
    averages: int = 1
    rerank_averages: int = 3
    baseline_repeats: int = 10
    stage2_repeats: int = 3
    stage1_maxfev: int = 200
    stage3_maxfev: int = 100
    stage1_top_k: int = 10
    stage3_top_k: int = 10
    stage1_initial_radius: float = 0.12
    stage3_initial_radius: float = 0.05
    final_radius: float = 0.01
    stage3_intensity_delta: float = 0.10
    alternation_threshold: float = 0.03
    max_alternations: int = 2
    beta_eta: float = 1.0
    beta_c: float = 1.0
    eta_threshold: float = 0.87
    eta_accept: float = 0.85
    eta_scale: float = 0.02
    c_floor_min: float = 1e-9
    lut_amplitudes: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    final_lut_points: int = 11
    lut_self_consistency: bool = True
    modulation_amplitudes: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    discrete_refine: bool = True
    discrete_step: float = 0.01
    skip_stage1: bool = False
    # measure the flat (all-ones) profile right after the initial baseline so
    # a live plot can show current-vs-flat during the run (the end-of-run
    # comparison_metrics re-measure stays: references drift over hours)
    flat_reference_up_front: bool = True
    flat_reference_repeats: int = 3
    output_root: str = "data/osa_optimization"
    run_name: str | None = None
    resume: bool = False
    settle_seconds: float = 0.0
    reference_interval_candidates: int = 25
    full_validation: bool = True
    full_validation_stride: int = 1
    reference_profile_validation: bool = True

    def validate(self, layout: ChannelLayout) -> None:
        if layout.channel_width_px != 15:
            raise ValueError("the current optimisation plan requires channel_width_px=15")
        if self.averages < 1 or self.rerank_averages < 1:
            raise ValueError("OSA averages must be >= 1")
        if self.baseline_repeats < 2 or self.stage2_repeats < 1:
            raise ValueError("baseline_repeats must be >=2 and stage2_repeats >=1")
        if self.stage1_maxfev < 1 or self.stage3_maxfev < 1:
            raise ValueError("optimizer evaluation budgets must be positive")
        if self.final_lut_points < 5:
            raise ValueError("final_lut_points must be >= 5")
        if self.reference_interval_candidates < 0:
            raise ValueError("reference_interval_candidates must be >= 0")
        if self.full_validation_stride < 1:
            raise ValueError("full_validation_stride must be >= 1")
        if 0 not in self.anchor_offsets:
            raise ValueError("anchor_offsets must contain the main offset 0")

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["settings"] = asdict(self.settings)
        return data


@dataclass
class OptimizationResult:
    initial_l: np.ndarray
    stage1_l: np.ndarray
    stage3_l: np.ndarray
    final_l: np.ndarray
    final_profile: np.ndarray
    final_luts: dict[int, StageAmplitudeReference]
    final_metrics: dict[int, AnchorMetrics]
    run_dir: str
    comparison_metrics: dict[str, dict[int, AnchorMetrics]] = field(default_factory=dict)
    full_validation: dict[str, list[dict[str, object]]] = field(default_factory=dict)
    accepted: bool = False
    acceptance_issues: list[str] = field(default_factory=list)
    stopped: bool = False


class RunStore:
    """Crash-safe append-only records for a long hardware run."""

    _CANDIDATE_FIELDS = (
        "timestamp", "stage", "evaluation", "loss", "feasible", "l",
        "metrics", "pattern_hashes",
    )

    def __init__(self, config: OSAOptimizationConfig):
        root = Path(config.output_root)
        if config.run_name:
            name = config.run_name
        else:
            name = time.strftime("%Y-%m-%d_run%H%M%S")
        self.run_dir = (root / name).resolve()
        if self.run_dir.exists() and not config.resume:
            suffix = 1
            base = self.run_dir
            while self.run_dir.exists():
                self.run_dir = Path(f"{base}_{suffix:02d}")
                suffix += 1
        self.traces_dir = self.run_dir / "traces"
        self.patterns_dir = self.run_dir / "patterns"
        self.references_dir = self.run_dir / "references"
        for directory in (self.run_dir, self.traces_dir, self.patterns_dir, self.references_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.candidate_path = self.run_dir / "candidates.csv"
        existing_counters: list[int] = []
        for path in self.traces_dir.glob("*.npz"):
            try:
                existing_counters.append(int(path.name.split("_", 1)[0]))
            except ValueError:
                continue
        self._trace_counter = max(existing_counters, default=0)
        if not self.candidate_path.exists():
            with self.candidate_path.open("w", encoding="utf-8", newline="") as handle:
                csv.DictWriter(handle, fieldnames=self._CANDIDATE_FIELDS).writeheader()

    @staticmethod
    def _json_default(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, MeasurementSettings):
            return asdict(value)
        raise TypeError(f"cannot serialise {type(value).__name__}")

    def write_json(self, name: str, payload: object) -> Path:
        path = self.run_dir / name
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=self._json_default),
            encoding="utf-8",
        )
        tmp.replace(path)
        return path

    def save_pattern(self, pattern_hash: str, pattern: np.ndarray) -> Path:
        path = self.patterns_dir / f"{pattern_hash}.npz"
        if not path.exists():
            np.savez_compressed(path, pattern=np.asarray(pattern, dtype=np.uint16))
        return path

    def save_trace(
        self, stage: str, anchor_key: str, trace: TraceData, pattern_hash: str
    ) -> Path:
        self._trace_counter += 1
        safe_stage = stage.replace("/", "_").replace(" ", "_")
        path = self.traces_dir / f"{self._trace_counter:07d}_{safe_stage}_{anchor_key}.npz"
        np.savez_compressed(
            path,
            wavelengths=np.asarray(trace.wavelengths),
            powers=np.asarray(trace.powers),
            y_unit=np.asarray(trace.y_unit),
            pattern_hash=np.asarray(pattern_hash),
        )
        return path

    def append_candidate(
        self,
        stage: str,
        evaluation: int,
        loss: float,
        profile: np.ndarray,
        metrics: Sequence[AnchorMetrics],
        pattern_hashes: Sequence[str],
        feasible: bool,
    ) -> None:
        row = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "stage": stage,
            "evaluation": evaluation,
            "loss": f"{loss:.16g}",
            "feasible": int(bool(feasible)),
            "l": json.dumps(np.asarray(profile, dtype=float).tolist()),
            "metrics": json.dumps([m.to_dict() for m in metrics], allow_nan=True),
            "pattern_hashes": json.dumps(list(pattern_hashes)),
        }
        with self.candidate_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._CANDIDATE_FIELDS)
            writer.writerow(row)
            handle.flush()

    def load_best_profile(self, stage: str) -> np.ndarray | None:
        if not self.candidate_path.exists():
            return None
        best: tuple[float, np.ndarray] | None = None
        with self.candidate_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("stage") != stage:
                    continue
                try:
                    loss = float(row["loss"])
                    profile = np.asarray(json.loads(row["l"]), dtype=float)
                except (ValueError, TypeError, json.JSONDecodeError):
                    continue
                if np.isfinite(loss) and (best is None or loss < best[0]):
                    best = (loss, profile)
        return None if best is None else best[1]


class OSAEvaluator:
    """Translate candidates into SLM patterns and fixed-bin OSA measurements."""

    def __init__(
        self,
        osa: OSAController,
        slm: SLMController,
        layout: ChannelLayout,
        config: OSAOptimizationConfig,
        store: RunStore,
        *,
        stop_event: threading.Event | None = None,
    ):
        self.osa = osa
        self.slm = slm
        self.layout = layout
        self.config = config
        self.store = store
        self.stop_event = stop_event
        self.slm_width, self.slm_height = slm.get_slm_info()
        self._ordered: list[tuple[ChannelRef, EncodingChannel]] = sorted(
            [
                (ChannelRef(ch.side, ch.index, float(ch.wavelength_nm)), ch)
                for ch in layout.all_channels
            ],
            key=lambda item: item[0].wavelength_nm,
        )
        self._center_position = len(self._ordered) // 2
        self.anchor_positions: dict[int, int] = {}
        for offset in config.anchor_offsets:
            position = self._center_position + int(offset)
            if position < 2 or position > len(self._ordered) - 3:
                raise ValueError(
                    f"anchor offset {offset} cannot provide fixed +/-2 bins on "
                    f"a {len(self._ordered)}-channel layout"
                )
            self.anchor_positions[int(offset)] = position
        self.calibrations: dict[int, AnchorCalibration] = {}

    def _check_stop(self) -> None:
        if self.stop_event is not None and self.stop_event.is_set():
            raise OptimizationAborted("OSA optimisation stopped by request")

    @staticmethod
    def _pattern_hash(pattern: np.ndarray) -> str:
        data = np.asarray(pattern, dtype=np.uint16)
        return hashlib.sha256(data.tobytes(order="C")).hexdigest()[:20]

    def _settings_for(self, position: int) -> MeasurementSettings:
        center = self._ordered[position][0].wavelength_nm
        return replace(self.config.settings, center_wl=f"{center:.6f}nm")

    def _pattern_for_positions(
        self, commands: dict[int, float], independent_l: Sequence[float]
    ) -> np.ndarray:
        x_values = np.zeros(self.layout.n_channels, dtype=float)
        w_values = np.zeros(self.layout.n_channels, dtype=float)
        for position, command in commands.items():
            ref, _channel = self._ordered[position]
            value = float(np.clip(command, 0.0, 1.0))
            if ref.side == "x":
                x_values[ref.index] = value
            else:
                w_values[ref.index] = value
        full_l = mirror_intensity_profile(independent_l, self.layout.channel_width_px)
        return encode_to_pattern(
            x_values,
            w_values,
            self.layout,
            self.slm_width,
            self.slm_height,
            col_ratio=full_l,
        )

    def _acquire(
        self,
        pattern: np.ndarray,
        settings: MeasurementSettings,
        *,
        averages: int,
        stage: str,
        anchor_key: str,
    ) -> tuple[TraceData, str, str]:
        self._check_stop()
        pattern_hash = self._pattern_hash(pattern)
        self.store.save_pattern(pattern_hash, pattern)
        self.slm.display_array(pattern)
        if self.config.settle_seconds > 0.0:
            deadline = time.monotonic() + self.config.settle_seconds
            while time.monotonic() < deadline:
                self._check_stop()
                time.sleep(min(0.05, deadline - time.monotonic()))
        try:
            trace = self.osa.measure(
                settings,
                averages=averages,
                stop_event=self.stop_event,
            )
        except OSAError as exc:
            if self.stop_event is not None and self.stop_event.is_set():
                raise OptimizationAborted("OSA optimisation stopped by request") from exc
            raise
        trace_path = self.store.save_trace(stage, anchor_key, trace, pattern_hash)
        return trace, pattern_hash, str(trace_path)

    def calibrate_anchor(self, offset: int) -> AnchorCalibration:
        """Measure dark and five flat one-hot peaks, then freeze bin boundaries."""
        self._check_stop()
        position = self.anchor_positions[offset]
        anchor_ref = self._ordered[position][0]
        settings = self._settings_for(position)
        flat_l = np.ones((self.layout.channel_width_px + 1) // 2, dtype=float)

        dark_pattern = self._pattern_for_positions({}, flat_l)
        dark, _dark_hash, _ = self._acquire(
            dark_pattern,
            settings,
            averages=self.config.rerank_averages,
            stage="reference_dark",
            anchor_key=anchor_ref.key,
        )

        nominal = np.array(
            [self._ordered[position + rel][0].wavelength_nm for rel in _OFFSETS],
            dtype=float,
        )
        pitch_nm = float(np.median(np.diff(nominal)))
        measured = nominal.copy()
        flat_traces: dict[int, TraceData] = {}
        for index, rel in enumerate(_OFFSETS):
            target_position = position + rel
            pattern = self._pattern_for_positions({target_position: 1.0}, flat_l)
            trace, _pattern_hash, _ = self._acquire(
                pattern,
                settings,
                averages=self.config.rerank_averages,
                stage="reference_flat_peak",
                anchor_key=anchor_ref.key,
            )
            flat_traces[rel] = trace
            wl = np.asarray(trace.wavelengths_nm, dtype=float)
            power = trace_power_w(trace)
            dark_power = np.interp(
                wl,
                np.asarray(dark.wavelengths_nm, dtype=float),
                trace_power_w(dark),
            )
            signal = power - dark_power
            local = np.abs(wl - nominal[index]) <= 0.45 * pitch_nm
            if np.any(local):
                local_idx = np.flatnonzero(local)
                peak_idx = int(local_idx[np.argmax(signal[local_idx])])
                if signal[peak_idx] > 0.0:
                    measured[index] = float(wl[peak_idx])

        # Reject a noisy/non-physical peak assignment as a whole.  Fixed nominal
        # centers are safer than silently creating overlapping bins.
        if (
            not np.all(np.diff(measured) > 0.25 * pitch_nm)
            or np.any(np.abs(measured - nominal) > 0.45 * pitch_nm)
        ):
            measured = nominal
        bins = FixedChannelBins.from_centers(anchor_ref.key, measured)
        flat_energy = float(bins.integrate(flat_traces[0], dark)[2])
        if flat_energy <= 0.0:
            raise RuntimeError(f"flat reference at {anchor_ref.key} has zero in-bin power")
        calibration = AnchorCalibration(
            offset=offset,
            position=position,
            anchor=anchor_ref,
            settings=settings,
            bins=bins,
            dark_trace=dark,
            flat_main_energy=flat_energy,
        )
        self.calibrations[offset] = calibration
        self.store.write_json(
            f"references/anchor_{offset:+d}.json",
            {
                "offset": offset,
                "position": position,
                "anchor": asdict(anchor_ref),
                "settings": asdict(settings),
                "centers_nm": measured,
                "edges_nm": bins.edges_nm,
                "flat_main_energy": flat_energy,
            },
        )
        return calibration

    def calibrate_all(self) -> dict[int, AnchorCalibration]:
        for offset in self.config.anchor_offsets:
            self.calibrate_anchor(int(offset))
        return self.calibrations

    def refresh_anchor_reference(self, offset: int, *, stage: str) -> AnchorCalibration:
        """Refresh dark/flat powers without moving the already-frozen bins."""
        calibration = self.calibrations.get(offset)
        if calibration is None:
            raise RuntimeError(f"anchor {offset} has not been calibrated")
        flat_l = np.ones((self.layout.channel_width_px + 1) // 2, dtype=float)
        dark_pattern = self._pattern_for_positions({}, flat_l)
        dark, _hash, _path = self._acquire(
            dark_pattern,
            calibration.settings,
            averages=self.config.rerank_averages,
            stage=f"{stage}_dark",
            anchor_key=calibration.anchor.key,
        )
        flat_pattern = self._pattern_for_positions(
            {calibration.position: 1.0}, flat_l
        )
        flat, _hash, _path = self._acquire(
            flat_pattern,
            calibration.settings,
            averages=self.config.rerank_averages,
            stage=f"{stage}_flat",
            anchor_key=calibration.anchor.key,
        )
        flat_energy = float(calibration.bins.integrate(flat, dark)[2])
        if flat_energy <= 0.0:
            raise RuntimeError(
                f"refreshed flat reference at {calibration.anchor.key} has zero power"
            )
        calibration.dark_trace = dark
        calibration.flat_main_energy = flat_energy
        calibration.reference_version += 1
        return calibration

    def refresh_all_references(self, *, stage: str) -> None:
        for offset in self.config.anchor_offsets:
            self.refresh_anchor_reference(int(offset), stage=stage)

    def measure_commands(
        self,
        offset: int,
        commands: Sequence[float],
        independent_l: Sequence[float],
        *,
        averages: int | None = None,
        stage: str,
        pattern_cache: dict[str, EnergyMeasurement] | None = None,
    ) -> EnergyMeasurement:
        """Measure [left, center, right] scalar intensity commands."""
        if len(commands) != 3:
            raise ValueError("exactly three channel commands are required")
        calibration = self.calibrations.get(offset)
        if calibration is None:
            raise RuntimeError(f"anchor {offset} has not been calibrated")
        positions = {
            calibration.position - 1: float(commands[0]),
            calibration.position: float(commands[1]),
            calibration.position + 1: float(commands[2]),
        }
        pattern = self._pattern_for_positions(positions, independent_l)
        expected_hash = self._pattern_hash(pattern)
        if pattern_cache is not None and expected_hash in pattern_cache:
            return pattern_cache[expected_hash]
        trace, pattern_hash, trace_path = self._acquire(
            pattern,
            calibration.settings,
            averages=self.config.averages if averages is None else averages,
            stage=stage,
            anchor_key=calibration.anchor.key,
        )
        energies = calibration.bins.integrate(trace, calibration.dark_trace)
        measurement = EnergyMeasurement(energies, pattern_hash, trace_path)
        if pattern_cache is not None:
            pattern_cache[pattern_hash] = measurement
        return measurement

    def command_pattern_hash(
        self,
        offset: int,
        commands: Sequence[float],
        independent_l: Sequence[float],
    ) -> str:
        """Hash the exact quantised SLM pattern without displaying it."""
        if len(commands) != 3:
            raise ValueError("exactly three channel commands are required")
        calibration = self.calibrations.get(offset)
        if calibration is None:
            raise RuntimeError(f"anchor {offset} has not been calibrated")
        positions = {
            calibration.position - 1: float(commands[0]),
            calibration.position: float(commands[1]),
            calibration.position + 1: float(commands[2]),
        }
        pattern = self._pattern_for_positions(positions, independent_l)
        return self._pattern_hash(pattern)

    def modulation_pattern_signature(
        self,
        offset: int,
        independent_l: Sequence[float],
        reference: StageAmplitudeReference,
    ) -> tuple[str, ...]:
        hashes: list[str] = []
        for amplitudes in self.modulation_cases():
            commands = tuple(float(reference.lut.command_for(a)) for a in amplitudes)
            hashes.append(self.command_pattern_hash(offset, commands, independent_l))
        return tuple(hashes)

    def measure_one_hot(
        self,
        offset: int,
        independent_l: Sequence[float],
        *,
        averages: int | None = None,
        stage: str,
    ) -> tuple[AnchorMetrics, EnergyMeasurement]:
        measurement = self.measure_commands(
            offset, (0.0, 1.0, 0.0), independent_l, averages=averages, stage=stage
        )
        main = measurement.main
        if main <= 0.0:
            c_total = math.inf
            eta = 0.0
        else:
            c_total = float((measurement.energies.sum() - main) / main)
            eta = main / self.calibrations[offset].flat_main_energy
        return AnchorMetrics(
            offset, c_total, eta,
            reference_version=self.calibrations[offset].reference_version,
        ), measurement

    def build_amplitude_lut(
        self,
        offset: int,
        independent_l: Sequence[float],
        *,
        amplitudes: Sequence[float] | None = None,
        averages: int | None = None,
        self_consistent: bool | None = None,
        stage: str = "stage2_lut",
    ) -> StageAmplitudeReference:
        target_a = np.asarray(
            self.config.lut_amplitudes if amplitudes is None else amplitudes,
            dtype=float,
        )
        if target_a.ndim != 1 or target_a.size < 2:
            raise ValueError("at least two LUT amplitude samples are required")
        if np.any(np.diff(target_a) <= 0.0) or target_a[0] != 0.0 or target_a[-1] != 1.0:
            raise ValueError("LUT amplitudes must increase strictly from 0 to 1")
        commands = target_a ** 2
        use_consistency = (
            self.config.lut_self_consistency
            if self_consistent is None else bool(self_consistent)
        )
        u_half = 0.25

        def sweep(half_command: float, label: str) -> StageAmplitudeReference:
            energies: list[float] = []
            pattern_cache: dict[str, EnergyMeasurement] = {}
            for command in commands:
                measured = self.measure_commands(
                    offset,
                    (half_command, float(command), half_command),
                    independent_l,
                    averages=averages,
                    stage=f"{stage}_{label}",
                    pattern_cache=pattern_cache,
                )
                energies.append(measured.main)
            e_blank = float(energies[0])
            e_full = float(energies[-1])
            denominator = e_full - e_blank
            if denominator <= 0.0:
                raise RuntimeError(
                    f"LUT at anchor {offset} has non-positive E_full-E_blank"
                )
            measured_a = np.sqrt(
                np.maximum((np.asarray(energies) - e_blank) / denominator, 0.0)
            )
            return StageAmplitudeReference(
                lut=AmplitudeLUT(commands.copy(), measured_a),
                e_blank=e_blank,
                e_full=e_full,
            )

        reference = sweep(u_half, "initial")
        if use_consistency:
            u_half = float(reference.lut.command_for(0.5))
            reference = sweep(u_half, "consistent")
        return reference

    def modulation_cases(self) -> list[tuple[float, float, float]]:
        cases: list[tuple[float, float, float]] = []
        for center in self.config.modulation_amplitudes:
            cases.extend(
                [
                    (0.5, float(center), 0.5),
                    (0.0, float(center), 0.0),
                    (1.0, float(center), 1.0),
                ]
            )
        cases.extend([(1.0, 0.5, 0.0), (0.0, 0.5, 1.0)])
        return cases

    def measure_modulation(
        self,
        offset: int,
        independent_l: Sequence[float],
        reference: StageAmplitudeReference,
        *,
        averages: int | None = None,
        stage: str,
    ) -> tuple[AnchorMetrics, list[str]]:
        denominator = reference.e_full - reference.e_blank
        if denominator <= 0.0:
            raise ValueError("fixed amplitude reference has a non-positive denominator")
        cache: dict[tuple[float, float, float], EnergyMeasurement] = {}
        pattern_cache: dict[str, EnergyMeasurement] = {}
        errors: list[float] = []
        hashes: list[str] = []
        for amplitudes in self.modulation_cases():
            commands = tuple(float(reference.lut.command_for(a)) for a in amplitudes)
            key = tuple(round(value, 12) for value in commands)
            measurement = cache.get(key)
            if measurement is None:
                measurement = self.measure_commands(
                    offset,
                    commands,
                    independent_l,
                    averages=averages,
                    stage=stage,
                    pattern_cache=pattern_cache,
                )
                cache[key] = measurement
                hashes.append(measurement.pattern_hash)
            a_hat = math.sqrt(
                max((measurement.main - reference.e_blank) / denominator, 0.0)
            )
            errors.append(a_hat - amplitudes[1])

        one_hot_commands = (
            float(reference.lut.command_for(0.0)),
            float(reference.lut.command_for(1.0)),
            float(reference.lut.command_for(0.0)),
        )
        one_hot = cache[tuple(round(value, 12) for value in one_hot_commands)]
        main = one_hot.main
        if main <= 0.0:
            c_total = math.inf
            eta = 0.0
        else:
            c_total = float((one_hot.energies.sum() - main) / main)
            eta = main / self.calibrations[offset].flat_main_energy
        error_array = np.asarray(errors, dtype=float)
        abs_error = np.abs(error_array)
        metrics = AnchorMetrics(
            anchor_offset=offset,
            c_total=c_total,
            eta=eta,
            reference_version=self.calibrations[offset].reference_version,
            rmse=float(np.sqrt(np.mean(error_array ** 2))),
            bias=float(np.mean(error_array)),
            mae=float(np.mean(abs_error)),
            p95=float(np.percentile(abs_error, 95.0)),
            max_error=float(np.max(abs_error)),
            amplitude_errors=error_array,
        )
        return metrics, hashes

    def validate_all_channels(
        self,
        profiles: dict[str, Sequence[float]],
        *,
        averages: int,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> dict[str, list[dict[str, object]]]:
        """One-hot validation for every channel that has measured +/-2 neighbours.

        A dark and flat one-hot trace establish a common peak shift and fixed
        five-bin grid at each position.  All supplied profiles are then measured
        against that same flat reference.  Edge channels without two physical
        neighbours on each side are intentionally excluded from crosstalk sums.
        """
        validated = {
            name: validate_independent_profile(values, self.layout.channel_width_px)
            for name, values in profiles.items()
        }
        if "flat" not in validated:
            raise ValueError("full validation requires a 'flat' reference profile")
        positions = list(
            range(2, len(self._ordered) - 2, self.config.full_validation_stride)
        )
        output: dict[str, list[dict[str, object]]] = {
            name: [] for name in validated
        }
        flat_l = validated["flat"]
        total = len(positions)
        for step, position in enumerate(positions, start=1):
            self._check_stop()
            anchor = self._ordered[position][0]
            settings = self._settings_for(position)
            dark_pattern = self._pattern_for_positions({}, flat_l)
            dark, _hash, _path = self._acquire(
                dark_pattern, settings, averages=averages,
                stage="full_validation_dark", anchor_key=anchor.key,
            )
            flat_pattern = self._pattern_for_positions({position: 1.0}, flat_l)
            flat_trace, flat_hash, _path = self._acquire(
                flat_pattern, settings, averages=averages,
                stage="full_validation_flat", anchor_key=anchor.key,
            )

            nominal = np.array(
                [self._ordered[position + rel][0].wavelength_nm for rel in _OFFSETS],
                dtype=float,
            )
            pitch_nm = float(np.median(np.diff(nominal)))
            wl = np.asarray(flat_trace.wavelengths_nm, dtype=float)
            signal = trace_power_w(flat_trace) - np.interp(
                wl,
                np.asarray(dark.wavelengths_nm, dtype=float),
                trace_power_w(dark),
            )
            local = np.abs(wl - nominal[2]) <= 0.45 * pitch_nm
            shift = 0.0
            if np.any(local):
                local_idx = np.flatnonzero(local)
                peak_idx = int(local_idx[np.argmax(signal[local_idx])])
                if signal[peak_idx] > 0.0:
                    shift = float(wl[peak_idx] - nominal[2])
            bins = FixedChannelBins.from_centers(anchor.key, nominal + shift)
            flat_energies = bins.integrate(flat_trace, dark)
            flat_main = float(flat_energies[2])
            if flat_main <= 0.0:
                raise RuntimeError(
                    f"full-validation flat reference at {anchor.key} has zero power"
                )

            traces: dict[str, tuple[TraceData, str]] = {
                "flat": (flat_trace, flat_hash)
            }
            other_names = [name for name in validated if name != "flat"]
            if position % 2:
                other_names.reverse()
            for name in other_names:
                pattern = self._pattern_for_positions({position: 1.0}, validated[name])
                trace, pattern_hash, _path = self._acquire(
                    pattern, settings, averages=averages,
                    stage=f"full_validation_{name}", anchor_key=anchor.key,
                )
                traces[name] = (trace, pattern_hash)

            for name, (trace, pattern_hash) in traces.items():
                energies = bins.integrate(trace, dark)
                main = float(energies[2])
                c_total = (
                    float((energies.sum() - main) / main) if main > 0.0 else math.inf
                )
                output[name].append(
                    {
                        "position": position,
                        "channel": anchor.key,
                        "wavelength_nm": anchor.wavelength_nm,
                        "c_total": c_total,
                        "eta": main / flat_main if main > 0.0 else 0.0,
                        "energies": energies.tolist(),
                        "pattern_hash": pattern_hash,
                    }
                )
            if progress_callback is not None:
                progress_callback(step, total, anchor.key)
        return output


@dataclass
class _CandidateRecord:
    loss: float
    profile: np.ndarray
    metrics: list[AnchorMetrics]
    pattern_hashes: list[str]


def _aggregate_metrics(repeats: Sequence[AnchorMetrics]) -> AnchorMetrics:
    if not repeats:
        raise ValueError("cannot aggregate an empty metric sequence")
    errors = [m.amplitude_errors for m in repeats if m.amplitude_errors.size]
    joined = np.concatenate(errors) if errors else np.empty(0)
    abs_error = np.abs(joined)
    return AnchorMetrics(
        anchor_offset=repeats[0].anchor_offset,
        c_total=float(np.mean([m.c_total for m in repeats])),
        eta=float(np.mean([m.eta for m in repeats])),
        reference_version=max(m.reference_version for m in repeats),
        rmse=float(np.sqrt(np.mean(joined ** 2))) if joined.size else math.nan,
        bias=float(np.mean(joined)) if joined.size else math.nan,
        mae=float(np.mean(abs_error)) if joined.size else math.nan,
        p95=float(np.percentile(abs_error, 95.0)) if joined.size else math.nan,
        max_error=float(np.max(abs_error)) if joined.size else math.nan,
        amplitude_errors=joined,
    )


class OptimizationRunner:
    """Execute Stage 1, fixed-LUT Stage 3, final LUT and validation."""

    def __init__(
        self,
        evaluator: OSAEvaluator,
        initial_l: Sequence[float],
        config: OSAOptimizationConfig,
        store: RunStore,
        *,
        progress_callback: ProgressCallback | None = None,
    ):
        self.evaluator = evaluator
        self.config = config
        self.store = store
        self.initial_l = validate_independent_profile(
            initial_l, evaluator.layout.channel_width_px
        )
        self.progress_callback = progress_callback
        self.histories: dict[str, list[_CandidateRecord]] = {}
        self._best_loss: dict[str, float] = {}
        self.c_floor = config.c_floor_min
        self.initial_metrics: dict[int, AnchorMetrics] = {}
        self._last_sample: EvaluationSample | None = None
        self._pending_flat_reference: dict[str, float] | None = None

    def _report(self, stage: str, step: int, total: int, message: str) -> None:
        if self.progress_callback is None:
            self._last_sample = None
            self._pending_flat_reference = None
            return
        sample, self._last_sample = self._last_sample, None
        flat_ref, self._pending_flat_reference = self._pending_flat_reference, None
        self.progress_callback(
            OptimizationProgress(
                stage, step, total, message, self._best_loss.get(stage),
                sample=sample, flat_reference=flat_ref,
            )
        )

    def _maybe_refresh_main_reference(self, stage: str) -> bool:
        interval = self.config.reference_interval_candidates
        evaluations = len(self.histories.get(stage, []))
        if interval > 0 and evaluations > 0 and evaluations % interval == 0:
            self.evaluator.refresh_anchor_reference(
                0, stage=f"{stage}_periodic_reference"
            )
            return True
        return False

    @staticmethod
    def _is_feasible(metrics: Sequence[AnchorMetrics], eta_accept: float) -> bool:
        return bool(metrics) and all(
            np.isfinite(m.c_total) and np.isfinite(m.eta) and m.eta >= eta_accept
            for m in metrics
        )

    def _record(
        self,
        stage: str,
        profile: np.ndarray,
        metrics: list[AnchorMetrics],
        hashes: list[str],
        loss: float,
    ) -> float:
        history = self.histories.setdefault(stage, [])
        record = _CandidateRecord(float(loss), np.asarray(profile).copy(), metrics, hashes)
        history.append(record)
        self._best_loss[stage] = min(self._best_loss.get(stage, math.inf), float(loss))
        # headline metrics for the live plot; the next _report attaches them
        main = metrics[0] if metrics else None
        self._last_sample = EvaluationSample(
            stage=stage,
            evaluation=len(history),
            loss=float(loss),
            eta=float(main.eta) if main is not None else None,
            c_total=float(main.c_total) if main is not None else None,
            feasible=self._is_feasible(metrics, self.config.eta_accept),
        )
        self.store.append_candidate(
            stage,
            len(history),
            float(loss),
            record.profile,
            metrics,
            hashes,
            self._is_feasible(metrics, self.config.eta_accept),
        )
        self.store.write_json(
            "best_so_far.json",
            {
                name: {
                    "loss": min(item.loss for item in records),
                    "l": min(records, key=lambda item: item.loss).profile,
                }
                for name, records in self.histories.items() if records
            },
        )
        self.store.write_json(
            "optimizer_state.json",
            {
                name: {
                    "evaluations": len(records),
                    "best_loss": min(item.loss for item in records),
                    "best_l": min(records, key=lambda item: item.loss).profile,
                }
                for name, records in self.histories.items() if records
            },
        )
        return float(loss)

    @staticmethod
    def _unique_top(records: Iterable[_CandidateRecord], count: int) -> list[np.ndarray]:
        result: list[np.ndarray] = []
        seen: set[tuple[float, ...]] = set()
        for record in sorted(records, key=lambda item: item.loss):
            key = tuple(np.round(record.profile, 8))
            if key in seen:
                continue
            seen.add(key)
            result.append(record.profile.copy())
            if len(result) >= count:
                break
        return result

    def _measure_initial_baseline(self) -> None:
        c_noise: list[float] = []
        for index, offset in enumerate(self.config.anchor_offsets, start=1):
            repeats: list[AnchorMetrics] = []
            for _ in range(self.config.baseline_repeats):
                metrics, _ = self.evaluator.measure_one_hot(
                    int(offset), self.initial_l, stage="baseline_initial"
                )
                repeats.append(metrics)
            aggregate = _aggregate_metrics(repeats)
            self.initial_metrics[int(offset)] = aggregate
            if len(repeats) > 1:
                c_noise.append(float(np.std([m.c_total for m in repeats], ddof=1)))
            self._report(
                "baseline", index, len(self.config.anchor_offsets),
                f"initial one-hot baseline at anchor {offset:+d}",
            )
        self.c_floor = max(
            self.config.c_floor_min,
            2.0 * max(c_noise, default=0.0),
        )
        self.store.write_json(
            "baseline.json",
            {
                "l_init": self.initial_l,
                "profile15": mirror_intensity_profile(self.initial_l),
                "c_floor": self.c_floor,
                "anchors": {key: value.to_dict() for key, value in self.initial_metrics.items()},
            },
        )

    def _measure_flat_reference(self) -> None:
        """Measure the flat (all-ones) profile once, early in the run.

        Gives live plots a fixed current-vs-flat reference line; sent to the
        progress callback exactly once (attached to a dedicated report) and
        persisted as flat_reference.json. When the initial profile IS flat the
        just-measured baseline is reused instead of extra sweeps.
        """
        if not self.config.flat_reference_up_front:
            return
        flat = np.ones_like(self.initial_l)
        if np.allclose(self.initial_l, flat):
            aggregate = self.initial_metrics[0]
        else:
            repeats: list[AnchorMetrics] = []
            for _ in range(max(1, self.config.flat_reference_repeats)):
                metrics, _ = self.evaluator.measure_one_hot(
                    0, flat, stage="flat_reference"
                )
                repeats.append(metrics)
            aggregate = _aggregate_metrics(repeats)
        loss = stage1_loss(
            [aggregate.c_total], [aggregate.eta], [self.initial_metrics[0].c_total],
            c_floor=self.c_floor,
            beta_eta=self.config.beta_eta,
            eta_threshold=self.config.eta_threshold,
            eta_scale=self.config.eta_scale,
        )
        reference = {
            "loss": float(loss),
            "eta": float(aggregate.eta),
            "c_total": float(aggregate.c_total),
        }
        self.store.write_json("flat_reference.json", reference)
        self._pending_flat_reference = reference
        self._report("flat_reference", 1, 1, "flat (all-ones) reference measured")

    def _minimize(
        self,
        objective: Callable[[np.ndarray], float],
        x0: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        *,
        maxfev: int,
        initial_radius: float,
    ):
        from scipy.optimize import Bounds, minimize

        return minimize(
            objective,
            np.asarray(x0, dtype=float),
            method="COBYQA",
            bounds=Bounds(lower, upper),
            options={
                "maxfev": int(maxfev),
                "initial_tr_radius": float(initial_radius),
                "final_tr_radius": float(self.config.final_radius),
                "scale": True,
                "disp": False,
            },
        )

    def _run_stage1(self) -> np.ndarray:
        stage = "stage1"
        start = self.initial_l
        measurement_cache: dict[str, tuple[AnchorMetrics, EnergyMeasurement]] = {}
        if self.config.resume:
            resumed = self.store.load_best_profile(stage)
            if resumed is not None:
                start = validate_independent_profile(resumed)

        def objective(profile: np.ndarray) -> float:
            profile = validate_independent_profile(profile)
            if self._maybe_refresh_main_reference(stage):
                measurement_cache.clear()
            signature = self.evaluator.command_pattern_hash(
                0, (0.0, 1.0, 0.0), profile
            )
            cached = measurement_cache.get(signature)
            if cached is None:
                metrics, measurement = self.evaluator.measure_one_hot(
                    0, profile, stage=stage
                )
                measurement_cache[signature] = (metrics, measurement)
            else:
                metrics, measurement = cached
            loss = stage1_loss(
                [metrics.c_total], [metrics.eta], [self.initial_metrics[0].c_total],
                c_floor=self.c_floor,
                beta_eta=self.config.beta_eta,
                eta_threshold=self.config.eta_threshold,
                eta_scale=self.config.eta_scale,
            )
            value = self._record(stage, profile, [metrics], [measurement.pattern_hash], loss)
            self._report(stage, len(self.histories[stage]), self.config.stage1_maxfev,
                         "one-hot crosstalk search")
            return value

        result = self._minimize(
            objective,
            start,
            np.zeros_like(start),
            np.ones_like(start),
            maxfev=self.config.stage1_maxfev,
            initial_radius=self.config.stage1_initial_radius,
        )
        candidates = self._unique_top(self.histories[stage], self.config.stage1_top_k)
        if not any(np.allclose(candidate, result.x) for candidate in candidates):
            candidates.append(np.clip(result.x, 0.0, 1.0))
        return self._rerank_stage1(candidates)

    def _rerank_stage1(self, candidates: Sequence[np.ndarray]) -> np.ndarray:
        stage = "stage1_rerank"
        self.evaluator.refresh_all_references(stage="stage1_rerank_reference")
        records: list[_CandidateRecord] = []
        for index, profile in enumerate(candidates, start=1):
            per_anchor: list[AnchorMetrics] = []
            hashes: list[str] = []
            for offset in self.config.anchor_offsets:
                repeats: list[AnchorMetrics] = []
                for _ in range(self.config.rerank_averages):
                    metric, measurement = self.evaluator.measure_one_hot(
                        int(offset), profile, stage=stage
                    )
                    repeats.append(metric)
                    hashes.append(measurement.pattern_hash)
                per_anchor.append(_aggregate_metrics(repeats))
            loss = stage1_loss(
                [m.c_total for m in per_anchor],
                [m.eta for m in per_anchor],
                [self.initial_metrics[int(o)].c_total for o in self.config.anchor_offsets],
                c_floor=self.c_floor,
                beta_eta=self.config.beta_eta,
                eta_threshold=self.config.eta_threshold,
                eta_scale=self.config.eta_scale,
            )
            self._record(stage, profile, per_anchor, hashes, loss)
            records.append(_CandidateRecord(loss, profile.copy(), per_anchor, hashes))
            self._report(stage, index, len(candidates), "multi-anchor Stage-1 rerank")
        feasible = [
            record for record in records
            if self._is_feasible(record.metrics, self.config.eta_accept)
        ]
        return min(feasible or records, key=lambda record: record.loss).profile.copy()

    def _build_luts(
        self,
        profile: np.ndarray,
        *,
        amplitudes: Sequence[float] | None = None,
        self_consistent: bool | None = None,
        stage: str,
    ) -> dict[int, StageAmplitudeReference]:
        references: dict[int, StageAmplitudeReference] = {}
        for index, offset in enumerate(self.config.anchor_offsets, start=1):
            references[int(offset)] = self.evaluator.build_amplitude_lut(
                int(offset), profile,
                amplitudes=amplitudes,
                averages=self.config.averages,
                self_consistent=self_consistent,
                stage=stage,
            )
            self._report(stage, index, len(self.config.anchor_offsets),
                         f"amplitude LUT at anchor {offset:+d}")
        self.store.write_json(
            f"{stage}.json",
            {key: value.to_dict() for key, value in references.items()},
        )
        return references

    def _measure_stage3_baseline(
        self,
        profile: np.ndarray,
        references: dict[int, StageAmplitudeReference],
        *,
        stage: str,
    ) -> tuple[dict[int, AnchorMetrics], float]:
        baselines: dict[int, AnchorMetrics] = {}
        noise: list[float] = []
        for offset in self.config.anchor_offsets:
            repeats: list[AnchorMetrics] = []
            for _ in range(self.config.stage2_repeats):
                metric, _ = self.evaluator.measure_modulation(
                    int(offset), profile, references[int(offset)], stage=stage
                )
                repeats.append(metric)
            baselines[int(offset)] = _aggregate_metrics(repeats)
            if len(repeats) > 1:
                matrix = np.vstack([m.amplitude_errors for m in repeats])
                noise.append(float(np.sqrt(np.mean(np.var(matrix, axis=0, ddof=1)))))
        sigma_a = max(max(noise, default=0.0), 1e-6)
        self.store.write_json(
            f"{stage}.json",
            {
                "sigma_a": sigma_a,
                "anchors": {key: value.to_dict() for key, value in baselines.items()},
            },
        )
        return baselines, sigma_a

    def _run_stage3_pass(
        self,
        start: np.ndarray,
        references: dict[int, StageAmplitudeReference],
        baselines: dict[int, AnchorMetrics],
        sigma_a: float,
        *,
        tag: str,
        maxfev: int,
        top_k: int,
    ) -> np.ndarray:
        lower = np.maximum(0.0, start - self.config.stage3_intensity_delta)
        upper = np.minimum(1.0, start + self.config.stage3_intensity_delta)
        if self.config.resume:
            resumed = self.store.load_best_profile(tag)
            if resumed is not None:
                start = np.clip(validate_independent_profile(resumed), lower, upper)
        measurement_cache: dict[tuple[str, ...], tuple[AnchorMetrics, list[str]]] = {}

        def objective(profile: np.ndarray) -> float:
            profile = validate_independent_profile(profile)
            if self._maybe_refresh_main_reference(tag):
                measurement_cache.clear()
            signature = self.evaluator.modulation_pattern_signature(
                0, profile, references[0]
            )
            cached = measurement_cache.get(signature)
            if cached is None:
                metric, hashes = self.evaluator.measure_modulation(
                    0, profile, references[0], stage=tag
                )
                measurement_cache[signature] = (metric, hashes)
            else:
                metric, hashes = cached
            baseline = baselines[0]
            loss = stage3_loss(
                [metric.rmse], [metric.c_total], [metric.eta],
                [baseline.rmse], [baseline.c_total],
                sigma_a=sigma_a,
                c_floor=self.c_floor,
                beta_c=self.config.beta_c,
                beta_eta=self.config.beta_eta,
                eta_threshold=self.config.eta_threshold,
                eta_scale=self.config.eta_scale,
            )
            value = self._record(tag, profile, [metric], hashes, loss)
            self._report(tag, len(self.histories[tag]), maxfev, "fixed-LUT fidelity search")
            return value

        result = self._minimize(
            objective, start, lower, upper,
            maxfev=maxfev,
            initial_radius=min(self.config.stage3_initial_radius,
                               self.config.stage3_intensity_delta),
        )
        candidates = self._unique_top(self.histories[tag], top_k)
        if not any(np.allclose(candidate, result.x) for candidate in candidates):
            candidates.append(np.clip(result.x, lower, upper))
        return self._rerank_stage3(candidates, references, baselines, sigma_a, tag)

    def _rerank_stage3(
        self,
        candidates: Sequence[np.ndarray],
        references: dict[int, StageAmplitudeReference],
        baselines: dict[int, AnchorMetrics],
        sigma_a: float,
        tag: str,
    ) -> np.ndarray:
        stage = f"{tag}_rerank"
        self.evaluator.refresh_all_references(stage=f"{stage}_reference")
        records: list[_CandidateRecord] = []
        for index, profile in enumerate(candidates, start=1):
            metrics: list[AnchorMetrics] = []
            hashes: list[str] = []
            for offset in self.config.anchor_offsets:
                repeats: list[AnchorMetrics] = []
                for _ in range(self.config.rerank_averages):
                    metric, measured_hashes = self.evaluator.measure_modulation(
                        int(offset), profile, references[int(offset)], stage=stage
                    )
                    repeats.append(metric)
                    hashes.extend(measured_hashes)
                metrics.append(_aggregate_metrics(repeats))
            loss = stage3_loss(
                [m.rmse for m in metrics],
                [m.c_total for m in metrics],
                [m.eta for m in metrics],
                [baselines[int(o)].rmse for o in self.config.anchor_offsets],
                [baselines[int(o)].c_total for o in self.config.anchor_offsets],
                sigma_a=sigma_a,
                c_floor=self.c_floor,
                beta_c=self.config.beta_c,
                beta_eta=self.config.beta_eta,
                eta_threshold=self.config.eta_threshold,
                eta_scale=self.config.eta_scale,
            )
            self._record(stage, profile, metrics, hashes, loss)
            records.append(_CandidateRecord(loss, profile.copy(), metrics, hashes))
            self._report(stage, index, len(candidates), "multi-anchor Stage-3 rerank")
        feasible = [
            record for record in records
            if self._is_feasible(record.metrics, self.config.eta_accept)
        ]
        return min(feasible or records, key=lambda record: record.loss).profile.copy()

    def _discrete_refine(
        self,
        profile: np.ndarray,
        reference: StageAmplitudeReference,
        baseline: AnchorMetrics,
        sigma_a: float,
    ) -> np.ndarray:
        if not self.config.discrete_refine or self.config.discrete_step <= 0.0:
            return profile
        current = profile.copy()
        seen_signatures: set[tuple[str, ...]] = set()

        def score(candidate: np.ndarray) -> float:
            signature = self.evaluator.modulation_pattern_signature(
                0, candidate, reference
            )
            if signature in seen_signatures:
                return math.inf
            seen_signatures.add(signature)
            metric, hashes = self.evaluator.measure_modulation(
                0, candidate, reference, stage="discrete_refine"
            )
            loss = stage3_loss(
                [metric.rmse], [metric.c_total], [metric.eta],
                [baseline.rmse], [baseline.c_total],
                sigma_a=sigma_a, c_floor=self.c_floor,
                beta_c=self.config.beta_c, beta_eta=self.config.beta_eta,
                eta_threshold=self.config.eta_threshold,
                eta_scale=self.config.eta_scale,
            )
            return self._record("discrete_refine", candidate, [metric], hashes, loss)

        best_loss = score(current)
        for index in range(current.size):
            base = current.copy()
            local_best = current
            local_loss = best_loss
            for direction in (-1.0, 1.0):
                candidate = base.copy()
                candidate[index] = np.clip(
                    base[index] + direction * self.config.discrete_step, 0.0, 1.0
                )
                if candidate[index] == base[index]:
                    continue
                loss = score(candidate)
                if loss < local_loss:
                    local_best, local_loss = candidate, loss
            current, best_loss = local_best, local_loss
        return current

    def run(self) -> OptimizationResult:
        self.store.write_json("run_config.json", self.config.to_dict())
        self._report("setup", 0, len(self.config.anchor_offsets), "calibrating fixed bins")
        self.evaluator.calibrate_all()
        self._measure_initial_baseline()
        self._measure_flat_reference()

        if self.config.skip_stage1:
            stage1_l = self.initial_l.copy()
            self.store.write_json(
                "stage1_result.json",
                {"l": stage1_l, "skipped": True},
            )
        else:
            stage1_l = self._run_stage1()
            self.store.write_json("stage1_result.json", {"l": stage1_l})

        self.evaluator.refresh_all_references(stage="stage2_reference")
        references = self._build_luts(stage1_l, stage="stage2_lut")
        baselines, sigma_a = self._measure_stage3_baseline(
            stage1_l, references, stage="stage2_modulation_baseline"
        )
        stage3_l = self._run_stage3_pass(
            stage1_l,
            references,
            baselines,
            sigma_a,
            tag="stage3",
            maxfev=self.config.stage3_maxfev,
            top_k=self.config.stage3_top_k,
        )

        current_l = stage3_l
        alternation = 1
        while (
            alternation < self.config.max_alternations
            and float(np.max(np.abs(current_l - stage1_l)))
                > self.config.alternation_threshold
        ):
            references = self._build_luts(
                current_l, stage=f"alternation_{alternation}_lut"
            )
            baselines, sigma_a = self._measure_stage3_baseline(
                current_l, references, stage=f"alternation_{alternation}_baseline"
            )
            current_l = self._run_stage3_pass(
                current_l,
                references,
                baselines,
                sigma_a,
                tag=f"stage3_alt{alternation}",
                maxfev=max(20, self.config.stage3_maxfev // 2),
                top_k=max(3, self.config.stage3_top_k // 2),
            )
            alternation += 1

        current_l = self._discrete_refine(
            current_l, references[0], baselines[0], sigma_a
        )
        final_amplitudes = np.linspace(0.0, 1.0, self.config.final_lut_points)
        final_luts = self._build_luts(
            current_l,
            amplitudes=final_amplitudes,
            self_consistent=True,
            stage="final_lut",
        )
        final_metrics: dict[int, AnchorMetrics] = {}
        for offset in self.config.anchor_offsets:
            metric, _ = self.evaluator.measure_modulation(
                int(offset),
                current_l,
                final_luts[int(offset)],
                averages=self.config.rerank_averages,
                stage="final_validation",
            )
            final_metrics[int(offset)] = metric

        comparison_metrics: dict[str, dict[int, AnchorMetrics]] = {
            "optimized": final_metrics
        }
        if self.config.reference_profile_validation:
            measured_profiles: list[tuple[np.ndarray, dict[int, AnchorMetrics]]] = []
            for label, profile in (
                ("flat", np.ones_like(self.initial_l)),
                ("initial", self.initial_l),
            ):
                reused = next(
                    (metrics for known, metrics in measured_profiles
                     if np.allclose(known, profile, atol=1e-12)),
                    None,
                )
                if reused is not None:
                    comparison_metrics[label] = reused
                    continue
                profile_luts = self._build_luts(
                    profile,
                    amplitudes=final_amplitudes,
                    self_consistent=True,
                    stage=f"comparison_{label}_lut",
                )
                metrics_by_anchor: dict[int, AnchorMetrics] = {}
                for offset in self.config.anchor_offsets:
                    metric, _ = self.evaluator.measure_modulation(
                        int(offset), profile, profile_luts[int(offset)],
                        averages=self.config.rerank_averages,
                        stage=f"comparison_{label}_validation",
                    )
                    metrics_by_anchor[int(offset)] = metric
                comparison_metrics[label] = metrics_by_anchor
                measured_profiles.append((profile.copy(), metrics_by_anchor))

        acceptance_issues: list[str] = []
        for offset, metric in final_metrics.items():
            if metric.eta < self.config.eta_accept:
                acceptance_issues.append(
                    f"anchor {offset:+d}: eta {metric.eta:.4f} < {self.config.eta_accept:.4f}"
                )
        initial_comparison = comparison_metrics.get("initial")
        if initial_comparison is not None:
            for offset in self.config.anchor_offsets:
                initial_metric = initial_comparison[int(offset)]
                final_metric = final_metrics[int(offset)]
                if final_metric.rmse > initial_metric.rmse + 2.0 * sigma_a:
                    acceptance_issues.append(
                        f"anchor {offset:+d}: RMSE did not match the initial profile within 2 sigma"
                    )
                if (
                    not self.config.skip_stage1
                    and initial_metric.c_total - final_metric.c_total <= self.c_floor
                ):
                    acceptance_issues.append(
                        f"anchor {offset:+d}: crosstalk improvement did not exceed 2-sigma floor"
                    )

        full_validation: dict[str, list[dict[str, object]]] = {}
        if self.config.full_validation:
            full_validation = self.evaluator.validate_all_channels(
                {
                    "flat": np.ones_like(self.initial_l),
                    "initial": self.initial_l,
                    "optimized": current_l,
                },
                averages=self.config.rerank_averages,
                progress_callback=lambda step, total, key: self._report(
                    "full_validation", step, total,
                    f"one-hot comparison at {key}",
                ),
            )
            self.store.write_json("full_validation.json", full_validation)
            held_out_failures = [
                row for row in full_validation.get("optimized", [])
                if float(row["eta"]) < self.config.eta_accept
            ]
            if held_out_failures:
                acceptance_issues.append(
                    f"{len(held_out_failures)} full-validation channels have eta below "
                    f"{self.config.eta_accept:.4f}"
                )

        result = OptimizationResult(
            initial_l=self.initial_l.copy(),
            stage1_l=stage1_l.copy(),
            stage3_l=stage3_l.copy(),
            final_l=current_l.copy(),
            final_profile=mirror_intensity_profile(current_l),
            final_luts=final_luts,
            final_metrics=final_metrics,
            run_dir=str(self.store.run_dir),
            comparison_metrics=comparison_metrics,
            full_validation=full_validation,
            accepted=not acceptance_issues,
            acceptance_issues=acceptance_issues,
        )
        self.store.write_json(
            "final_result.json",
            {
                "initial_l": result.initial_l,
                "stage1_l": result.stage1_l,
                "stage3_l": result.stage3_l,
                "final_l": result.final_l,
                "final_profile": result.final_profile,
                "luts": {key: value.to_dict() for key, value in final_luts.items()},
                "metrics": {key: value.to_dict() for key, value in final_metrics.items()},
                "comparison_metrics": {
                    label: {key: value.to_dict() for key, value in metrics.items()}
                    for label, metrics in comparison_metrics.items()
                },
                "accepted": result.accepted,
                "acceptance_issues": result.acceptance_issues,
                "full_validation_file": (
                    "full_validation.json" if full_validation else None
                ),
            },
        )
        return result


def run_osa_optimization(
    osa: OSAController,
    slm: SLMController,
    layout: ChannelLayout,
    initial_l: Sequence[float],
    *,
    config: OSAOptimizationConfig | None = None,
    stop_event: threading.Event | None = None,
    progress_callback: ProgressCallback | None = None,
) -> OptimizationResult:
    """Run the complete live OSA optimisation from an eight-value intensity input."""
    config = OSAOptimizationConfig() if config is None else config
    config.validate(layout)
    initial = validate_independent_profile(initial_l, layout.channel_width_px)
    store = RunStore(config)
    evaluator = OSAEvaluator(
        osa, slm, layout, config, store, stop_event=stop_event
    )
    runner = OptimizationRunner(
        evaluator, initial, config, store, progress_callback=progress_callback
    )
    return runner.run()


@dataclass(frozen=True)
class OSABatchVariant:
    """One OSA-settings variant of a batch run (e.g. a sampling-point count)."""

    label: str                       # short tag, becomes part of the run name
    settings: MeasurementSettings


@dataclass
class OSABatchOutcome:
    """Result envelope for one variant of a batch optimisation."""

    variant: OSABatchVariant
    result: OptimizationResult | None
    run_dir: str | None
    error: str | None = None
    stopped: bool = False


def run_osa_optimization_batch(
    osa: OSAController,
    slm: SLMController,
    layout: ChannelLayout,
    initial_l: Sequence[float],
    *,
    base_config: OSAOptimizationConfig,
    variants: Sequence[OSABatchVariant],
    stop_event: threading.Event | None = None,
    progress_callback: ProgressCallback | None = None,
) -> list[OSABatchOutcome]:
    """Run the full optimisation once per OSA-settings variant.

    Each variant gets its own run directory (``<run_name>_<label>``) and its
    progress stages are prefixed with ``[i/n label]`` so a single live view can
    follow the whole batch. A stop request ends the batch at the next variant
    boundary (and aborts the in-flight run as usual); any other per-variant
    failure is recorded and the batch continues. A summary JSON comparing the
    variants is written next to the run directories.
    """
    if not variants:
        raise ValueError("batch optimisation needs at least one variant")
    labels = [variant.label for variant in variants]
    if len(set(labels)) != len(labels):
        raise ValueError("batch variant labels must be unique")

    base_name = base_config.run_name or time.strftime("%Y-%m-%d_batch%H%M%S")
    outcomes: list[OSABatchOutcome] = []
    for index, variant in enumerate(variants, start=1):
        if stop_event is not None and stop_event.is_set():
            outcomes.append(
                OSABatchOutcome(variant, None, None, stopped=True)
            )
            continue

        config = replace(
            base_config,
            settings=variant.settings,
            run_name=f"{base_name}_{variant.label}",
        )

        wrapped: ProgressCallback | None = None
        if progress_callback is not None:
            prefix = f"[{index}/{len(variants)} {variant.label}] "

            def wrapped(
                progress: OptimizationProgress, _prefix: str = prefix
            ) -> None:
                progress_callback(
                    replace(progress, stage=f"{_prefix}{progress.stage}")
                )

        try:
            result = run_osa_optimization(
                osa, slm, layout, initial_l,
                config=config, stop_event=stop_event,
                progress_callback=wrapped,
            )
            outcomes.append(
                OSABatchOutcome(variant, result, result.run_dir)
            )
        except OptimizationAborted:
            outcomes.append(OSABatchOutcome(variant, None, None, stopped=True))
            break
        except Exception as exc:  # noqa: BLE001 - isolate per-variant failures
            outcomes.append(
                OSABatchOutcome(variant, None, None, error=str(exc))
            )

    summary = {
        "base_name": base_name,
        "variants": [
            {
                "label": outcome.variant.label,
                "settings": asdict(outcome.variant.settings),
                "run_dir": outcome.run_dir,
                "stopped": outcome.stopped,
                "error": outcome.error,
                "accepted": (
                    outcome.result.accepted if outcome.result is not None else None
                ),
                "final_metrics": (
                    {
                        key: value.to_dict()
                        for key, value in outcome.result.final_metrics.items()
                    }
                    if outcome.result is not None
                    else None
                ),
            }
            for outcome in outcomes
        ],
    }
    summary_path = Path(base_config.output_root) / f"{base_name}_batch_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = summary_path.with_suffix(summary_path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=RunStore._json_default),
        encoding="utf-8",
    )
    tmp.replace(summary_path)
    return outcomes


def load_optimization_result(path: str | Path) -> OptimizationResult:
    """Load a persisted final result for later amplitude-mode encoding."""
    source = Path(path).resolve()
    if source.is_dir():
        source = source / "final_result.json"
    data = json.loads(source.read_text(encoding="utf-8"))

    luts: dict[int, StageAmplitudeReference] = {}
    for key, value in data["luts"].items():
        lut_data = value["lut"]
        luts[int(key)] = StageAmplitudeReference(
            lut=AmplitudeLUT(
                np.asarray(lut_data["commands"], dtype=float),
                np.asarray(lut_data["measured_amplitudes"], dtype=float),
            ),
            e_blank=float(value["e_blank"]),
            e_full=float(value["e_full"]),
        )
    metrics = {
        int(key): AnchorMetrics.from_dict(value)
        for key, value in data.get("metrics", {}).items()
    }
    comparison = {
        label: {
            int(key): AnchorMetrics.from_dict(value)
            for key, value in values.items()
        }
        for label, values in data.get("comparison_metrics", {}).items()
    }
    full_validation: dict[str, list[dict[str, object]]] = {}
    validation_name = data.get("full_validation_file")
    if validation_name:
        validation_path = source.parent / str(validation_name)
        if validation_path.is_file():
            full_validation = json.loads(validation_path.read_text(encoding="utf-8"))
    return OptimizationResult(
        initial_l=np.asarray(data["initial_l"], dtype=float),
        stage1_l=np.asarray(data["stage1_l"], dtype=float),
        stage3_l=np.asarray(data["stage3_l"], dtype=float),
        final_l=np.asarray(data["final_l"], dtype=float),
        final_profile=np.asarray(data["final_profile"], dtype=float),
        final_luts=luts,
        final_metrics=metrics,
        run_dir=str(source.parent),
        comparison_metrics=comparison,
        full_validation=full_validation,
        accepted=bool(data.get("accepted", False)),
        acceptance_issues=[str(item) for item in data.get("acceptance_issues", [])],
    )
