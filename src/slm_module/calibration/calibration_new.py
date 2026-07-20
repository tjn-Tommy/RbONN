from __future__ import annotations

import csv
import json
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from osa_module.controller import MeasurementSettings, OSAController, TraceData
from slm_module.controller import SLMController

from .outliers import (
    OutlierRemeasurePolicy,
    flag_by_residual,
    linear_fit_residuals,
    transfer_fit_residuals,
)

if TYPE_CHECKING:  # avoid importing daq_module at runtime
    from daq_module.controller import DAQController


"""
Calibration module for Santec SLM with AQ637X OSA.

Step 1: find rough minimum and maximum intensity levels by sweeping full-screen
grayscale levels.
Step 2: use a bright window sweep to map SLM x coordinates to wavelengths.
Step 3: for each calibrated coordinate, sweep grayscale levels and measure both
the absolute (background-subtracted, in watts) and the normalized intensity
averaged around that coordinate's calibrated wavelength.
"""


# Normalized traces are only trusted where the bright reference rises above
# this fraction of its own peak; below it the divide inflates noise/drift into
# spurious peaks (see _reduce_arrays).
_MIN_REFERENCE_FRACTION = 0.05

# Minimum normalized peak strength for a step-2 anchor measurement to count as
# a real peak (a genuine window peak normalizes to ~1).
_MIN_ANCHOR_PEAK_STRENGTH = 0.1


class CalibrationAborted(Exception):
    """Raised when a stop_event interrupts a calibration sweep."""


@dataclass
class CalibrationProgress:
    """A single live update emitted during calibration acquisition.

    phase is one of "min_max", "wavelength", or "intensity". step is the 0-based
    index within that phase and total is the number of steps in it, so a UI can
    drive a per-phase progress bar. message describes the step's result; x/y are
    an optional data point for a live plot (units depend on the phase).
    """

    phase: str
    step: int
    total: int
    message: str
    x: float | None = None
    y: float | None = None


ProgressCallback = Callable[["CalibrationProgress"], None]


def _report(
    progress_callback: ProgressCallback | None,
    phase: str,
    step: int,
    total: int,
    message: str,
    *,
    x: float | None = None,
    y: float | None = None,
) -> None:
    if progress_callback is not None:
        progress_callback(
            CalibrationProgress(
                phase=phase, step=step, total=total, message=message, x=x, y=y
            )
        )


@dataclass
class CalibrationResult:
    wavelength: np.ndarray
    coordinates: np.ndarray
    max_level: int | np.ndarray
    min_level: int | np.ndarray
    level_range: np.ndarray
    intensity_levels: np.ndarray | None = None
    raw_intensity_levels: np.ndarray | None = None
    wavelength_fit_coefficients: np.ndarray | None = None


def find_min_max_intensity_levels(
    osa: OSAController,
    slm: SLMController,
    levels: Iterable[int],
    measure_settings: MeasurementSettings,
    *,
    stop_event: threading.Event | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[float, float, int, int, dict[int, float]]:
    """Sweep full-screen grayscale levels and find rough min/max output power."""

    level_values = _validate_levels(levels)
    total = int(level_values.size)
    min_intensity = float("inf")
    max_intensity = float("-inf")
    min_level = int(level_values[0])
    max_level = int(level_values[0])
    intensity_records: dict[int, float] = {}

    for index, level in enumerate(level_values):
        _check_stop(stop_event)
        level_int = int(level)
        slm.display_grayscale(level_int)
        trace = osa.measure(measure_settings)
        intensity = float(np.mean(_trace_power_w(trace)))

        if intensity < min_intensity:
            min_intensity = intensity
            min_level = level_int
        if intensity > max_intensity:
            max_intensity = intensity
            max_level = level_int
        intensity_records[level_int] = intensity
        _report(
            progress_callback,
            "min_max",
            index,
            total,
            f"Level {level_int} -> {intensity:.3e} W",
            x=float(level_int),
            y=intensity,
        )

    return min_intensity, max_intensity, min_level, max_level, intensity_records


def local_peak_centroid(
    wavelengths_m: np.ndarray,
    intensity_W: np.ndarray,
    half_window: int = 100,
    *,
    half_window_nm: float | None = None,
) -> tuple[float, int, float]:
    """
    Estimate peak center by local weighted centroid.

    The returned center uses the same unit as wavelengths_m. The name is kept
    for compatibility with earlier code, but callers may pass nm or m.

    The local window is either a fixed number of samples on each side of the
    peak (``half_window``) or, when ``half_window_nm`` is given, every sample
    within +/- half_window_nm of the peak wavelength. The nm form keeps the
    averaging width physical and independent of the OSA sampling density.

    Returns:
        center_wavelength
        argmax_index
        peak_strength
    """

    wavelengths = np.asarray(wavelengths_m, dtype=float)
    intensity = np.asarray(intensity_W, dtype=float)
    half_window = _validate_non_negative_int(half_window, "half_window")

    if wavelengths.ndim != 1 or intensity.ndim != 1:
        raise ValueError("wavelengths_m and intensity_W must be 1D arrays.")
    if wavelengths.size != intensity.size:
        raise ValueError(
            f"wavelengths and intensity size mismatch: "
            f"{wavelengths.size} vs {intensity.size}"
        )
    if wavelengths.size == 0:
        raise ValueError("Empty trace.")

    y = np.nan_to_num(intensity, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.clip(y, 0.0, None)

    idx = int(np.argmax(y))
    peak_strength = float(y[idx])

    if half_window_nm is not None:
        window = float(half_window_nm)
        if not np.isfinite(window) or window <= 0:
            raise ValueError("half_window_nm must be a positive, finite number")
        mask = np.abs(wavelengths - wavelengths[idx]) <= window
        x_local = wavelengths[mask]
        y_local = y[mask].copy()
    else:
        lo = max(0, idx - half_window)
        hi = min(y.size, idx + half_window + 1)
        x_local = wavelengths[lo:hi]
        y_local = y[lo:hi].copy()

    y_local -= np.min(y_local)
    y_local = np.clip(y_local, 0.0, None)

    weight_sum = float(np.sum(y_local))
    if weight_sum <= 0:
        return float(wavelengths[idx]), idx, peak_strength

    center = float(np.sum(x_local * y_local) / weight_sum)
    return center, idx, peak_strength


def local_peak_centroid_near(
    wavelengths_nm: np.ndarray,
    intensity_W: np.ndarray,
    target_wavelength_nm: float,
    *,
    half_window_nm: float,
) -> tuple[float, int, float]:
    """Centroid the strongest local peak inside a target-centered nm window."""

    wavelengths = np.asarray(wavelengths_nm, dtype=float)
    intensity = np.asarray(intensity_W, dtype=float)
    target = float(target_wavelength_nm)
    window = float(half_window_nm)
    if wavelengths.ndim != 1 or intensity.ndim != 1:
        raise ValueError("wavelengths_nm and intensity_W must be 1D arrays")
    if wavelengths.size != intensity.size:
        raise ValueError(
            f"wavelengths and intensity size mismatch: "
            f"{wavelengths.size} vs {intensity.size}"
        )
    if wavelengths.size == 0:
        raise ValueError("Empty trace.")
    if not np.isfinite(target):
        raise ValueError("target_wavelength_nm must be finite")
    if not np.isfinite(window) or window <= 0.0:
        raise ValueError("half_window_nm must be positive")

    tolerance = np.finfo(float).eps * max(1.0, abs(target), window) * 8.0
    local_indices = np.flatnonzero(np.abs(wavelengths - target) <= window + tolerance)
    if local_indices.size == 0:
        local_indices = np.asarray([int(np.argmin(np.abs(wavelengths - target)))])

    x_local = wavelengths[local_indices]
    y_local = np.nan_to_num(
        intensity[local_indices], nan=0.0, posinf=0.0, neginf=0.0
    )
    y_local = np.clip(y_local, 0.0, None)
    local_peak_offset = int(np.argmax(y_local))
    peak_index = int(local_indices[local_peak_offset])
    peak_strength = float(y_local[local_peak_offset])

    y_weights = y_local - float(np.min(y_local))
    y_weights = np.clip(y_weights, 0.0, None)
    weight_sum = float(np.sum(y_weights))
    if weight_sum <= 0.0:
        return float(wavelengths[peak_index]), peak_index, peak_strength

    center = float(np.sum(x_local * y_weights) / weight_sum)
    return center, peak_index, peak_strength


def mean_near_wavelength(
    wavelengths_nm: np.ndarray,
    intensity: np.ndarray,
    target_wavelength_nm: float,
    *,
    half_window_points: int = 2,
    window_nm: float | None = None,
) -> float:
    """Average intensity around target_wavelength_nm.

    If window_nm is provided, all samples within +/- window_nm / 2 are used.
    Otherwise the nearest sample and half_window_points neighbors on each side
    are averaged.
    """

    wavelengths = np.asarray(wavelengths_nm, dtype=float)
    values = np.asarray(intensity, dtype=float)
    half_window_points = _validate_non_negative_int(
        half_window_points, "half_window_points"
    )

    if wavelengths.ndim != 1 or values.ndim != 1:
        raise ValueError("wavelengths_nm and intensity must be 1D arrays")
    if wavelengths.size != values.size:
        raise ValueError(
            f"wavelengths and intensity size mismatch: "
            f"{wavelengths.size} vs {values.size}"
        )
    if wavelengths.size == 0:
        raise ValueError("Empty trace.")

    target = float(target_wavelength_nm)
    if not np.isfinite(target):
        raise ValueError("target_wavelength_nm must be finite")

    if window_nm is not None:
        window = float(window_nm)
        if not np.isfinite(window) or window <= 0:
            raise ValueError("window_nm must be positive")
        mask = np.abs(wavelengths - target) <= window / 2.0
        if np.any(mask):
            return _finite_mean(values[mask])

    idx = int(np.argmin(np.abs(wavelengths - target)))
    lo = max(0, idx - half_window_points)
    hi = min(values.size, idx + half_window_points + 1)
    return _finite_mean(values[lo:hi])


def wavelength_calibration(
    osa: OSAController,
    slm: SLMController,
    levels: Iterable[int],
    measure_settings: MeasurementSettings,
    calibration_results: CalibrationResult,
    window_size: int = 8,
    peak_half_window: int = 100,
    *,
    peak_half_window_nm: float | None = None,
    region: tuple[int, int] | None = None,
    coordinate_stride: int = 1,
    sweep_span_nm: float | None = None,
    min_peak_wavelength_nm: float | None = None,
    max_peak_wavelength_nm: float | None = None,
    outlier_policy: OutlierRemeasurePolicy | None = None,
    stop_event: threading.Event | None = None,
    progress_callback: ProgressCallback | None = None,
) -> CalibrationResult:
    """Map SLM x coordinates to wavelengths using a bright-window sweep.

    The peak of each window measurement is located by weighted centroid over a
    +/- peak_half_window_nm wavelength window when that is given, otherwise over
    peak_half_window samples on each side.

    ``region`` (x_start, x_end) limits the sweep to that inclusive band of SLM
    columns, which is useful when the source only illuminates part of the SLM
    width (e.g. a ~6 nm pulse on a ~20 nm aperture); None sweeps the full width.

    ``coordinate_stride`` speeds up acquisition: only every Nth window position
    is measured (the last position is always kept so the fit spans the whole
    region). The coordinate->wavelength mapping is nearly linear, so the
    polynomial fit over the strided points fills in the skipped coordinates:
    the returned result still carries the dense per-column grid a stride-1
    sweep would produce. 1 (default) measures every position.

    ``sweep_span_nm`` speeds up acquisition further: when set, the two
    region-edge positions are measured first with the wide ``measure_settings``
    span (anchors -- their wavelengths are unknown until measured), a straight
    line through the two anchor peaks predicts every other position's
    wavelength, and each remaining position is measured with a narrow OSA span
    (``sweep_span_nm`` wide) re-centered on its prediction -- far fewer samples
    per sweep with AUTO sampling. The dark/bright references keep the wide span
    and are interpolated onto each narrow grid. None (default) uses the wide
    span everywhere.

    ``min_peak_wavelength_nm`` / ``max_peak_wavelength_nm`` exclude trace
    samples below / above that wavelength from the peak search, e.g. to mask a
    fixed leakage artifact the SLM never modulates (light falling outside the
    active area). None (default) leaves that end of the trace unclipped.

    ``outlier_policy`` enables post-sweep auto-remeasurement: after the sweep, a
    linear coordinate->wavelength fit flags points whose residual exceeds
    ``k_sigma`` robust sigmas; each flagged window is re-displayed and
    re-measured (up to ``max_retries`` rounds) and the point becomes the median
    of its re-measurements. ``None`` (default) disables it.
    """

    del levels
    slm_width, slm_height = slm.get_slm_info()
    window_size = _validate_window_size(window_size, slm_width)
    coordinate_stride = int(coordinate_stride)
    if coordinate_stride < 1:
        raise ValueError("coordinate_stride must be >= 1")
    use_narrow = sweep_span_nm is not None
    sweep_value = 0.0
    if use_narrow:
        sweep_value = float(sweep_span_nm)
        if not sweep_value > 0:
            raise ValueError("sweep_span_nm must be positive when provided")
    min_peak = None
    if min_peak_wavelength_nm is not None:
        min_peak = float(min_peak_wavelength_nm)
        if not np.isfinite(min_peak):
            raise ValueError("min_peak_wavelength_nm must be finite")
    max_peak = None
    if max_peak_wavelength_nm is not None:
        max_peak = float(max_peak_wavelength_nm)
        if not np.isfinite(max_peak):
            raise ValueError("max_peak_wavelength_nm must be finite")
    if min_peak is not None and max_peak is not None and min_peak >= max_peak:
        raise ValueError(
            "min_peak_wavelength_nm must be below max_peak_wavelength_nm"
        )
    min_level = _level_value(calibration_results.min_level, "min_level")
    max_level = _level_value(calibration_results.max_level, "max_level")
    region_lo, region_hi = _resolve_scan_region(region, slm_width, window_size)

    dark_pattern = np.full(slm_width, min_level, dtype=int)
    _display_1d_pattern(slm, dark_pattern, slm_height)
    background_trace = osa.measure(measure_settings)
    background_power = _trace_power_w(background_trace)

    bright_pattern = np.full(slm_width, max_level, dtype=int)
    _display_1d_pattern(slm, bright_pattern, slm_height)
    reference_trace = osa.measure(measure_settings)
    reference_power = _trace_power_w(reference_trace)

    reference_axis = np.asarray(reference_trace.wavelengths_nm, dtype=float)
    n_ref = min(reference_axis.size, background_power.size, reference_power.size)
    denominator_scale = (
        float(np.max(reference_power[:n_ref] - background_power[:n_ref]))
        if n_ref else 0.0
    )

    def _acquire_peak(x_start: int, center_nm: float | None) -> tuple[float, float]:
        """Display the bright window at x_start, measure, return (wl, strength).

        center_nm re-centers a sweep_span_nm-wide sweep on that wavelength;
        None measures with the wide measure_settings span."""
        pattern = dark_pattern.copy()
        pattern[x_start : x_start + window_size] = max_level
        _display_1d_pattern(slm, pattern, slm_height)
        if center_nm is None:
            trace = osa.measure(measure_settings)
            trace_wavelengths, _signal, normalized = _reduce_trace(
                trace, _trace_power_w(trace), background_power, reference_power
            )
        else:
            narrow_settings = replace(
                measure_settings,
                center_wl=f"{center_nm:.4f}nm",
                span=f"{sweep_value}nm",
            )
            trace = osa.measure(narrow_settings)
            trace_wavelengths, _signal, normalized = _reduce_trace_resampled(
                trace,
                _trace_power_w(trace),
                reference_axis,
                background_power,
                reference_power,
                denominator_scale=denominator_scale,
            )
        if min_peak is not None:
            normalized[trace_wavelengths < min_peak] = 0.0
        if max_peak is not None:
            normalized[trace_wavelengths > max_peak] = 0.0
        wavelength, _, strength = local_peak_centroid(
            trace_wavelengths,
            normalized,
            half_window=peak_half_window,
            half_window_nm=peak_half_window_nm,
        )
        return float(wavelength), float(strength)

    coordinates: list[int] = []
    wavelengths: list[float] = []

    x_starts = list(range(region_lo, region_hi, coordinate_stride))
    if x_starts[-1] != region_hi - 1:
        x_starts.append(region_hi - 1)   # anchor the fit at the far edge
    total = len(x_starts)

    predict = None
    anchor_wavelengths: dict[int, float] = {}
    if use_narrow:
        if len(x_starts) < 2:
            raise ValueError(
                "sweep_span_nm needs a region with at least two window positions"
            )
        # Measure the two region-edge positions with the wide span first: their
        # wavelengths bootstrap the linear prediction the narrow sweeps center on.
        for anchor_index, a_start in ((0, x_starts[0]), (total - 1, x_starts[-1])):
            _check_stop(stop_event)
            anchor_wl, anchor_strength = _acquire_peak(a_start, None)
            if anchor_strength < _MIN_ANCHOR_PEAK_STRENGTH:
                raise ValueError(
                    f"anchor window at x={a_start} shows no clear peak "
                    f"(normalized strength {anchor_strength:.3g}); check the "
                    "scan region, OSA settings, and min/max_peak_wavelength_nm"
                )
            anchor_wavelengths[a_start] = anchor_wl
            _report(
                progress_callback,
                "wavelength",
                anchor_index,
                total,
                f"anchor x={a_start + window_size // 2} -> {anchor_wl:.3f} nm",
                x=float(a_start + window_size // 2),
                y=float(anchor_wl),
            )
        c_lo = float(x_starts[0] + window_size // 2)
        c_hi = float(x_starts[-1] + window_size // 2)
        w_lo = anchor_wavelengths[x_starts[0]]
        w_hi = anchor_wavelengths[x_starts[-1]]
        if abs(w_hi - w_lo) < 1e-9:
            raise ValueError(
                f"anchor peaks coincide (both read {w_lo:.4f} nm); cannot "
                "predict sweep centers"
            )
        slope = (w_hi - w_lo) / (c_hi - c_lo)

        def predict(coordinate: float) -> float:
            return w_lo + slope * (coordinate - c_lo)

    for index, x_start in enumerate(x_starts):
        _check_stop(stop_event)
        coordinate = x_start + window_size // 2
        if x_start in anchor_wavelengths:
            wavelength = anchor_wavelengths[x_start]
        else:
            center = predict(float(coordinate)) if predict is not None else None
            wavelength, _strength = _acquire_peak(x_start, center)
        coordinates.append(coordinate)
        wavelengths.append(wavelength)
        _report(
            progress_callback,
            "wavelength",
            index,
            total,
            f"x={coordinate} -> {wavelength:.3f} nm",
            x=float(coordinate),
            y=float(wavelength),
        )

    coordinate_array = np.asarray(coordinates, dtype=float)
    wavelength_array = np.asarray(wavelengths, dtype=float)

    if (
        outlier_policy is not None
        and coordinate_array.size >= outlier_policy.min_points
    ):
        # Post-sweep auto-remeasure: flag points off the linear map, re-measure
        # them, and replace each with the median of its re-measurements (the
        # flagged original is discarded). Repeat until clean or retries exhaust.
        readings: dict[int, list[float]] = {}
        wl_scale = float(np.nanmax(np.abs(wavelength_array)))
        for retry in range(1, outlier_policy.max_retries + 1):
            residuals = linear_fit_residuals(coordinate_array, wavelength_array)
            flagged = np.flatnonzero(
                flag_by_residual(
                    residuals,
                    k_sigma=outlier_policy.k_sigma,
                    rel_floor_scale=wl_scale,
                )
            )
            if flagged.size == 0:
                break
            for i in flagged:
                _check_stop(stop_event)
                x_start = int(coordinates[i]) - window_size // 2
                center = (
                    predict(float(coordinates[i])) if predict is not None else None
                )
                wavelength, _strength = _acquire_peak(x_start, center)
                readings.setdefault(int(i), []).append(float(wavelength))
                wavelength_array[i] = float(np.median(readings[int(i)]))
                _report(
                    progress_callback,
                    "wavelength",
                    total,
                    total,
                    (
                        f"recheck x={coordinates[i]} retry "
                        f"{retry}/{outlier_policy.max_retries} -> "
                        f"{wavelength_array[i]:.3f} nm"
                    ),
                    x=float(coordinates[i]),
                    y=float(wavelength_array[i]),
                )

    fitted_wavelengths, coeffs = _fit_wavelength_mapping(
        coordinate_array, wavelength_array
    )

    if coordinate_stride > 1:
        # Fill the skipped columns from the fitted curve so downstream
        # consumers see the same dense grid a stride-1 sweep would produce.
        coordinate_array = (
            np.arange(region_lo, region_hi, dtype=float) + window_size // 2
        )
        fitted_wavelengths = np.polyval(coeffs, coordinate_array)

    return CalibrationResult(
        wavelength=fitted_wavelengths,
        coordinates=coordinate_array,
        max_level=max_level,
        min_level=min_level,
        level_range=np.asarray(calibration_results.level_range, dtype=int),
        intensity_levels=calibration_results.intensity_levels,
        wavelength_fit_coefficients=coeffs,
    )


def intensity_calibration(
    osa: OSAController,
    slm: SLMController,
    levels: Iterable[int],
    measure_settings: MeasurementSettings,
    calibration_results: CalibrationResult,
    window_size: int,
    *,
    average_half_window: int = 2,
    wavelength_window_nm: float | None = None,
    sweep_span_nm: float | None = None,
    coordinate_stride: int = 1,
    refine_wavelength: bool = False,
    region: tuple[int, int] | None = None,
    stop_event: threading.Event | None = None,
    progress_callback: ProgressCallback | None = None,
) -> CalibrationResult:
    """Sweep levels and measure intensity near each calibrated wavelength.

    intensity_levels (normalized to the bright reference) and
    raw_intensity_levels (background-subtracted power, in watts) both have shape
    (n_coordinates, n_levels). Each row belongs to
    calibration_results.coordinates / calibration_results.wavelength; each
    column belongs to the corresponding entry in levels.

    ``region`` (x_start, x_end) restricts the sweep to the calibrated
    coordinates that fall within that inclusive band of SLM columns. This also
    applies to a mapping loaded from a file, so only the selected slice of the
    loaded range is calibrated; None calibrates every coordinate.

    ``sweep_span_nm`` speeds up acquisition: when set, each coordinate's signal
    sweep uses a narrow OSA span (``sweep_span_nm`` wide) re-centered on that
    coordinate's calibrated wavelength, instead of the wide ``measure_settings``
    span. Far fewer points per sweep (with AUTO sampling) means much faster
    measurements. The dark/bright reference traces are still measured once with
    the wide ``measure_settings`` span, so the narrow signal trace no longer
    shares their wavelength grid; intensities are therefore reduced by sampling
    each trace at the calibrated wavelength (see ``mean_near_wavelength``) rather
    than the element-wise ``_reduce_trace`` used when ``sweep_span_nm`` is None.

    ``coordinate_stride`` measures only every Nth calibrated coordinate (after the
    region selection), trading spatial density for speed; 1 measures every
    coordinate.

    ``refine_wavelength`` (narrow path only) re-fits the coordinate->wavelength
    mapping from Step 3's data: the narrow sweep resolves the peak more finely
    than Step 2's wide sweep, so for each coordinate the strongest level's trace
    is centroided to a refined peak wavelength. The returned ``wavelength`` array
    and ``wavelength_fit_coefficients`` then reflect those refined values.
    """

    level_values = _validate_levels(levels)
    coordinates, wavelengths = _calibrated_mapping(calibration_results)
    coordinates, wavelengths = _select_region_mapping(coordinates, wavelengths, region)

    coordinate_stride = int(coordinate_stride)
    if coordinate_stride < 1:
        raise ValueError("coordinate_stride must be >= 1")
    if coordinate_stride > 1:
        coordinates = coordinates[::coordinate_stride]
        wavelengths = wavelengths[::coordinate_stride]

    slm_width, slm_height = slm.get_slm_info()
    window_size = _validate_window_size(window_size, slm_width)
    average_half_window = _validate_non_negative_int(
        average_half_window, "average_half_window"
    )
    sweep_value = 0.0
    if sweep_span_nm is not None:
        sweep_value = float(sweep_span_nm)
        if not sweep_value > 0:
            raise ValueError("sweep_span_nm must be positive when provided")
    use_narrow = sweep_span_nm is not None
    min_level = _level_value(calibration_results.min_level, "min_level")
    max_level = _level_value(calibration_results.max_level, "max_level")

    dark_pattern = np.full(slm_width, min_level, dtype=int)
    _display_1d_pattern(slm, dark_pattern, slm_height)
    background_trace = osa.measure(measure_settings)
    background_power = _trace_power_w(background_trace)

    bright_pattern = np.full(slm_width, max_level, dtype=int)
    _display_1d_pattern(slm, bright_pattern, slm_height)
    reference_trace = osa.measure(measure_settings)
    reference_power = _trace_power_w(reference_trace)

    intensity_levels = np.zeros((coordinates.size, level_values.size), dtype=float)
    raw_intensity_levels = np.zeros((coordinates.size, level_values.size), dtype=float)
    refine = refine_wavelength and use_narrow
    refined_wavelengths = np.array(wavelengths, dtype=float)
    total = int(coordinates.size * level_values.size)
    step = 0

    for coordinate_index, (coordinate, wavelength_nm) in enumerate(
        zip(coordinates, wavelengths)
    ):
        x_start = _window_start_from_coordinate(coordinate, window_size, slm_width)
        best_strength = 0.0
        best_wavelengths_nm: np.ndarray | None = None
        best_power: np.ndarray | None = None

        if use_narrow:
            # Re-center a narrow sweep on this coordinate's wavelength so the OSA
            # only scans a tiny band (few AUTO points -> fast). Configure once and
            # reuse it for every level below. The dark/bright references keep the
            # wide span, so they no longer share this trace's wavelength grid:
            # sample all three at the calibrated wavelength instead of subtracting
            # element-wise (which _reduce_trace requires).
            narrow_settings = replace(
                measure_settings,
                center_wl=f"{float(wavelength_nm):.4f}nm",
                span=f"{sweep_value}nm",
            )
            osa.configure(narrow_settings)
            background_at = mean_near_wavelength(
                background_trace.wavelengths_nm,
                background_power,
                float(wavelength_nm),
                half_window_points=average_half_window,
                window_nm=wavelength_window_nm,
            )
            reference_at = mean_near_wavelength(
                reference_trace.wavelengths_nm,
                reference_power,
                float(wavelength_nm),
                half_window_points=average_half_window,
                window_nm=wavelength_window_nm,
            )
            denominator = reference_at - background_at

        for level_index, level in enumerate(level_values):
            _check_stop(stop_event)
            pattern = dark_pattern.copy()
            pattern[x_start : x_start + window_size] = int(level)
            _display_1d_pattern(slm, pattern, slm_height)

            if use_narrow:
                # measure() with no settings reuses the per-coordinate config
                # above, so no 7-command reconfigure per level.
                trace = osa.measure()
                power = _trace_power_w(trace)
                signal_at = mean_near_wavelength(
                    trace.wavelengths_nm,
                    power,
                    float(wavelength_nm),
                    half_window_points=average_half_window,
                    window_nm=wavelength_window_nm,
                )
                raw_value = max(0.0, signal_at - background_at)
                if abs(denominator) > np.finfo(float).eps:
                    normalized_value = max(0.0, raw_value / denominator)
                else:
                    normalized_value = 0.0
                if refine and signal_at > best_strength:
                    # keep the brightest trace; its peak localizes λ best
                    best_strength = signal_at
                    best_wavelengths_nm = trace.wavelengths_nm
                    best_power = power
            else:
                trace = osa.measure(measure_settings)
                trace_wavelengths, signal, normalized = _reduce_trace(
                    trace, _trace_power_w(trace), background_power, reference_power
                )
                raw_value = mean_near_wavelength(
                    trace_wavelengths,
                    signal,
                    float(wavelength_nm),
                    half_window_points=average_half_window,
                    window_nm=wavelength_window_nm,
                )
                normalized_value = mean_near_wavelength(
                    trace_wavelengths,
                    normalized,
                    float(wavelength_nm),
                    half_window_points=average_half_window,
                    window_nm=wavelength_window_nm,
                )

            raw_intensity_levels[coordinate_index, level_index] = raw_value
            intensity_levels[coordinate_index, level_index] = normalized_value
            _report(
                progress_callback,
                "intensity",
                step,
                total,
                f"λ {wavelength_nm:.2f} nm "
                f"({coordinate_index + 1}/{coordinates.size}), "
                f"level {int(level)} -> {normalized_value:.3f}",
                x=float(level),
                y=float(normalized_value),
            )
            step += 1

        if refine and best_wavelengths_nm is not None and best_power is not None:
            refined_center, _, _ = local_peak_centroid(
                best_wavelengths_nm,
                best_power,
                half_window_nm=sweep_value / 2.0,
            )
            refined_wavelengths[coordinate_index] = refined_center

    if refine:
        result_wavelengths = refined_wavelengths
        _, fit_coefficients = _fit_wavelength_mapping(coordinates, refined_wavelengths)
    else:
        result_wavelengths = wavelengths
        fit_coefficients = calibration_results.wavelength_fit_coefficients

    return CalibrationResult(
        wavelength=result_wavelengths,
        coordinates=coordinates,
        max_level=max_level,
        min_level=min_level,
        level_range=level_values,
        intensity_levels=intensity_levels,
        raw_intensity_levels=raw_intensity_levels,
        wavelength_fit_coefficients=fit_coefficients,
    )


def build_channel_calibration_grid(
    calibration_results: CalibrationResult,
    *,
    target_wavelength_nm: float = 778.0,
    center_coordinate: float | None = None,
    n_channels_per_side: int = 20,
    channel_width_px: int = 15,
    gap_px: int = 5,
    center_gap_px: int | None = None,
    slm_width: int | None = None,
    guard_bands_nm: Iterable[tuple[float, float]] | None = None,
    require_symmetric_guard_bands: bool = True,
) -> tuple[CalibrationResult, float]:
    """Generate a Step-3 seed only at channel centers around a target wavelength.

    The center coordinate is the middle of the gap between the nearest two
    channels.  For the default 15 px channel plus 5 px gap, the first channels
    sit at center +/- 10 px, leaving the target wavelength 2.5 px from each
    inner channel edge.

    ``center_gap_px`` widens the central dark pad to at least that many pixels
    (first offset ``ceil((width + center_gap_px)/2)`` instead of half a pitch).
    The measured coordinates ARE the encoding layout's channel centers: the
    encoder loads the result verbatim via
    :func:`slm_module.encoding.channel_layout_from_calibration`, so the grid
    designed here is the single source of the channel geometry.
    ``None`` keeps the legacy half-pitch placement.

    When ``center_coordinate`` is supplied, the linear wavelength map is shifted
    so that this refined coordinate is exactly ``target_wavelength_nm``.  That
    keeps OSA center fine tuning compatible with the very linear Step-2 fit.

    If guard bands are supplied, any candidate channel whose active window
    overlaps one of those wavelength bands is skipped and the next pitch outward
    is tried. The requested number of channels per side is preserved when the
    Step-2 range is wide enough.

    Channels are placed at mirror-image offsets ``center +/- (k + 0.5) * pitch``,
    so the k-th channel on each side pairs up at equal wavelength offset from the
    target only when the skipped (guard) bands are themselves symmetric about
    ``target_wavelength_nm``. With ``require_symmetric_guard_bands`` (the
    default), an asymmetric guard set is rejected up front so a lopsided pairing
    cannot slip through; pass False to allow it (e.g. to probe the skip
    mechanics in isolation).
    """

    target = float(target_wavelength_nm)
    if not np.isfinite(target):
        raise ValueError("target_wavelength_nm must be finite")
    n_channels = _validate_non_negative_int(
        n_channels_per_side, "n_channels_per_side"
    )
    if n_channels <= 0:
        raise ValueError("n_channels_per_side must be positive")
    width = _validate_non_negative_int(channel_width_px, "channel_width_px")
    if width <= 0:
        raise ValueError("channel_width_px must be positive")
    gap = _validate_non_negative_int(gap_px, "gap_px")
    pitch = width + gap
    if pitch <= 0:
        raise ValueError("channel pitch must be positive")
    first_offset: float | None = None
    if center_gap_px is not None:
        center_gap = _validate_non_negative_int(center_gap_px, "center_gap_px")
        # same formula as encoding.compute_channel_geometry: centre pad >= gap
        first_offset = float((width + center_gap + 1) // 2)

    source_coordinates, _source_wavelengths = _calibrated_mapping(
        calibration_results
    )
    slope, intercept = _linear_wavelength_fit(calibration_results)
    if center_coordinate is None:
        center = (target - intercept) / slope
    else:
        center = float(center_coordinate)
        if not np.isfinite(center):
            raise ValueError("center_coordinate must be finite")
        intercept = target - slope * center

    if slm_width is not None:
        slm_width = _validate_non_negative_int(slm_width, "slm_width")
        if slm_width <= 0:
            raise ValueError("slm_width must be positive")

    guard_bands = (
        []
        if guard_bands_nm is None
        else _validate_guard_bands_nm(guard_bands_nm)
    )
    if require_symmetric_guard_bands:
        _validate_guard_bands_symmetric(guard_bands, target)
    coord_min = float(source_coordinates[0])
    coord_max = float(source_coordinates[-1])

    def valid_channel(coordinate: float) -> bool:
        if coordinate < coord_min or coordinate > coord_max:
            return False
        if slm_width is not None:
            start, end = _channel_window_bounds(coordinate, width)
            if start < 0 or end > slm_width:
                return False
        return not _channel_window_overlaps_guard(
            coordinate, width, slope, intercept, guard_bands
        )

    left = _collect_guarded_channel_side(
        center,
        -1,
        pitch,
        n_channels,
        valid_channel,
        coord_min,
        coord_max,
        first_offset_px=first_offset,
    )
    right = _collect_guarded_channel_side(
        center,
        1,
        pitch,
        n_channels,
        valid_channel,
        coord_min,
        coord_max,
        first_offset_px=first_offset,
    )
    if len(left) < n_channels or len(right) < n_channels:
        raise ValueError(
            "not enough non-guard channel centers fit inside the Step 2 range "
            f"(left {len(left)}/{n_channels}, right {len(right)}/{n_channels})"
        )
    channel_coordinates = np.asarray(sorted(left + right), dtype=float)

    channel_wavelengths = slope * channel_coordinates + intercept
    return (
        CalibrationResult(
            wavelength=channel_wavelengths,
            coordinates=channel_coordinates,
            max_level=calibration_results.max_level,
            min_level=calibration_results.min_level,
            level_range=np.asarray(calibration_results.level_range, dtype=int),
            wavelength_fit_coefficients=np.asarray([slope, intercept], dtype=float),
        ),
        float(center),
    )


def refine_center_coordinate_with_osa(
    osa: OSAController,
    slm: SLMController,
    measure_settings: MeasurementSettings,
    calibration_results: CalibrationResult,
    *,
    target_wavelength_nm: float = 778.0,
    window_size: int = 15,
    peak_half_window_nm: float = 0.2,
    stop_event: threading.Event | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[float, float, float]:
    """Refine the target coordinate by measuring the target window once on OSA.

    Returns ``(refined_coordinate, measured_peak_nm, coarse_coordinate)``.
    The coordinate correction is ``(target - measured_peak) / linear_slope``.
    """

    target = float(target_wavelength_nm)
    if not np.isfinite(target):
        raise ValueError("target_wavelength_nm must be finite")
    peak_window = float(peak_half_window_nm)
    if not np.isfinite(peak_window) or peak_window <= 0.0:
        raise ValueError("peak_half_window_nm must be positive")

    slope, intercept = _linear_wavelength_fit(calibration_results)
    coarse_center = (target - intercept) / slope
    if not np.isfinite(coarse_center):
        raise ValueError("could not locate target wavelength from Step 2")

    slm_width, slm_height = slm.get_slm_info()
    window_size = _validate_window_size(window_size, slm_width)
    min_level = _level_value(calibration_results.min_level, "min_level")
    max_level = _level_value(calibration_results.max_level, "max_level")

    measure = _configured_measurement(osa, measure_settings)
    dark_pattern = np.full(slm_width, min_level, dtype=int)
    _display_1d_pattern(slm, dark_pattern, slm_height)
    _check_stop(stop_event)
    background_trace = measure()
    background_power = _trace_power_w(background_trace)

    _report(
        progress_callback,
        "fast_center",
        0,
        2,
        f"Step 2 predicts {target:.4f} nm at x={coarse_center:.3f} px",
        x=float(coarse_center),
        y=target,
    )

    x_start = _window_start_from_coordinate(coarse_center, window_size, slm_width)
    pattern = dark_pattern.copy()
    pattern[x_start : x_start + window_size] = max_level
    _display_1d_pattern(slm, pattern, slm_height)
    _check_stop(stop_event)
    trace = measure()
    count = min(trace.wavelengths_nm.size, trace.powers.size, background_power.size)
    if count <= 0:
        raise ValueError("OSA center refinement trace is empty")
    signal = np.clip(
        _trace_power_w(trace)[:count] - background_power[:count],
        0.0,
        None,
    )
    measured_peak, _idx, _strength = local_peak_centroid_near(
        trace.wavelengths_nm[:count],
        signal,
        target,
        half_window_nm=peak_window,
    )
    refined_center = coarse_center + (target - measured_peak) / slope
    if not np.isfinite(refined_center):
        raise ValueError("OSA center refinement produced an invalid coordinate")

    _report(
        progress_callback,
        "fast_center",
        1,
        2,
        (
            f"OSA peak {measured_peak:.4f} nm -> refined center "
            f"x={refined_center:.3f} px"
        ),
        x=float(refined_center),
        y=target,
    )
    return float(refined_center), float(measured_peak), float(coarse_center)


def batch_intensity_calibration(
    osa: OSAController,
    slm: SLMController,
    levels: Iterable[int],
    measure_settings: MeasurementSettings,
    calibration_results: CalibrationResult,
    window_size: int,
    *,
    average_half_window: int = 2,
    wavelength_window_nm: float | None = None,
    group_skip_channels: int = 2,
    guard_bands_nm: Iterable[tuple[float, float]] | None = None,
    refine_wavelength: bool = False,
    refine_half_window_nm: float | None = None,
    outlier_policy: OutlierRemeasurePolicy | None = None,
    stop_event: threading.Event | None = None,
    progress_callback: ProgressCallback | None = None,
) -> CalibrationResult:
    """Calibrate several non-neighboring channels from each full-span OSA trace.

    Channels are split into groups by sorted coordinate index.  With the default
    ``group_skip_channels=2``, one trace measures channels 0, 3, 6, ... while
    leaving two channel pitches between simultaneously active windows to reduce
    crosstalk.  Each trace is reduced at every active channel wavelength, so the
    output shape and normalization match :func:`intensity_calibration`.

    ``outlier_policy`` enables post-sweep auto-remeasurement: after each group's
    level sweep, every channel's curve is fitted to the sin^2 transfer model and
    cells whose residual exceeds ``k_sigma`` robust sigmas are re-measured (the
    group pattern for that level is re-displayed); each flagged cell becomes the
    median of its re-measurements. Up to ``max_retries`` rounds per group.
    ``None`` (default) disables it.
    """

    level_values = _validate_levels(levels)
    coordinates, wavelengths = _calibrated_mapping(calibration_results)
    if coordinates.size == 0:
        raise ValueError("batch calibration needs at least one channel coordinate")
    slm_width, slm_height = slm.get_slm_info()
    window_size = _validate_window_size(window_size, slm_width)
    average_half_window = _validate_non_negative_int(
        average_half_window, "average_half_window"
    )
    group_skip = _validate_non_negative_int(
        group_skip_channels, "group_skip_channels"
    )
    group_step = group_skip + 1
    min_level = _level_value(calibration_results.min_level, "min_level")
    max_level = _level_value(calibration_results.max_level, "max_level")
    guard_mask = _wavelength_guard_mask(slm_width, calibration_results, guard_bands_nm)

    measure = _configured_measurement(osa, measure_settings)
    dark_pattern = np.full(slm_width, min_level, dtype=int)
    _display_1d_pattern(slm, dark_pattern, slm_height)
    _check_stop(stop_event)
    background_trace = measure()
    background_power = _trace_power_w(background_trace)

    bright_pattern = np.full(slm_width, max_level, dtype=int)
    bright_pattern[guard_mask] = min_level
    _display_1d_pattern(slm, bright_pattern, slm_height)
    _check_stop(stop_event)
    reference_trace = measure()
    reference_power = _trace_power_w(reference_trace)

    intensity_levels = np.zeros((coordinates.size, level_values.size), dtype=float)
    raw_intensity_levels = np.zeros((coordinates.size, level_values.size), dtype=float)
    refined_wavelengths = np.array(wavelengths, dtype=float)
    best_strength = np.full(coordinates.size, -np.inf, dtype=float)
    best_wavelengths: list[np.ndarray | None] = [None] * coordinates.size
    best_signal: list[np.ndarray | None] = [None] * coordinates.size

    groups = [
        np.arange(offset, coordinates.size, group_step, dtype=int)
        for offset in range(group_step)
    ]
    groups = [group for group in groups if group.size]
    total = int(len(groups) * level_values.size)
    step = 0

    def _acquire_group(
        group: np.ndarray, level: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Display the group's windows at ``level`` and reduce one trace."""
        pattern = dark_pattern.copy()
        for channel_index in group:
            x_start = _window_start_from_coordinate(
                coordinates[channel_index], window_size, slm_width
            )
            pattern[x_start : x_start + window_size] = int(level)
        pattern[guard_mask] = min_level
        _display_1d_pattern(slm, pattern, slm_height)
        trace = measure()
        return _reduce_trace(
            trace,
            _trace_power_w(trace),
            background_power,
            reference_power,
        )

    def _channel_values(
        trace_wavelengths: np.ndarray,
        signal: np.ndarray,
        normalized: np.ndarray,
        channel_index: int,
    ) -> tuple[float, float]:
        """(raw, normalized) intensity at one channel's calibrated wavelength."""
        wavelength_nm = float(wavelengths[channel_index])
        raw_value = mean_near_wavelength(
            trace_wavelengths,
            signal,
            wavelength_nm,
            half_window_points=average_half_window,
            window_nm=wavelength_window_nm,
        )
        normalized_value = mean_near_wavelength(
            trace_wavelengths,
            normalized,
            wavelength_nm,
            half_window_points=average_half_window,
            window_nm=wavelength_window_nm,
        )
        return raw_value, normalized_value

    for group_number, group in enumerate(groups, start=1):
        for level_index, level in enumerate(level_values):
            _check_stop(stop_event)
            trace_wavelengths, signal, normalized = _acquire_group(group, int(level))
            group_values: list[float] = []
            for channel_index in group:
                raw_value, normalized_value = _channel_values(
                    trace_wavelengths, signal, normalized, channel_index
                )
                raw_intensity_levels[channel_index, level_index] = raw_value
                intensity_levels[channel_index, level_index] = normalized_value
                group_values.append(normalized_value)
                if refine_wavelength and raw_value > best_strength[channel_index]:
                    best_strength[channel_index] = raw_value
                    best_wavelengths[channel_index] = trace_wavelengths.copy()
                    best_signal[channel_index] = signal.copy()

            mean_value = float(np.mean(group_values)) if group_values else 0.0
            _report(
                progress_callback,
                "batch_intensity",
                step,
                total,
                (
                    f"group {group_number}/{len(groups)}, level {int(level)} -> "
                    f"{len(group)} channels"
                ),
                x=float(level),
                y=mean_value,
            )
            step += 1

        if (
            outlier_policy is not None
            and level_values.size >= outlier_policy.min_points
        ):
            # Post-sweep auto-remeasure for this group: fit each channel's
            # curve to the sin^2 transfer model, re-measure flagged cells (one
            # re-displayed group pattern per flagged level), replace each cell
            # with the median of its re-measurements. The refine_wavelength
            # best-trace tracking is intentionally left to the forward pass.
            cell_readings: dict[tuple[int, int], list[tuple[float, float]]] = {}
            for retry in range(1, outlier_policy.max_retries + 1):
                flagged_levels: dict[int, list[int]] = {}
                for channel_index in group:
                    curve = intensity_levels[channel_index]
                    residuals = transfer_fit_residuals(
                        level_values.astype(float), curve
                    )
                    finite = curve[np.isfinite(curve)]
                    mask = flag_by_residual(
                        residuals,
                        k_sigma=outlier_policy.k_sigma,
                        rel_floor_scale=(
                            float(np.max(np.abs(finite))) if finite.size else None
                        ),
                    )
                    for level_index in np.flatnonzero(mask):
                        flagged_levels.setdefault(int(level_index), []).append(
                            int(channel_index)
                        )
                if not flagged_levels:
                    break
                for level_index in sorted(flagged_levels):
                    _check_stop(stop_event)
                    level = int(level_values[level_index])
                    trace_wavelengths, signal, normalized = _acquire_group(
                        group, level
                    )
                    for channel_index in flagged_levels[level_index]:
                        raw_value, normalized_value = _channel_values(
                            trace_wavelengths, signal, normalized, channel_index
                        )
                        key = (channel_index, level_index)
                        cell_readings.setdefault(key, []).append(
                            (raw_value, normalized_value)
                        )
                        cell = np.asarray(cell_readings[key], dtype=float)
                        raw_intensity_levels[channel_index, level_index] = float(
                            np.median(cell[:, 0])
                        )
                        intensity_levels[channel_index, level_index] = float(
                            np.median(cell[:, 1])
                        )
                    _report(
                        progress_callback,
                        "batch_intensity",
                        step,
                        total,
                        (
                            f"recheck group {group_number}/{len(groups)}, "
                            f"level {level} retry "
                            f"{retry}/{outlier_policy.max_retries} -> "
                            f"{len(flagged_levels[level_index])} channels"
                        ),
                        x=float(level),
                        y=float(
                            intensity_levels[
                                flagged_levels[level_index][0], level_index
                            ]
                        ),
                    )

    if refine_wavelength:
        half_window = _resolve_refine_half_window_nm(
            wavelengths,
            refine_half_window_nm,
        )
        for channel_index, (wl_axis, sig) in enumerate(
            zip(best_wavelengths, best_signal)
        ):
            if wl_axis is None or sig is None:
                continue
            refined_center, _idx, _strength = local_peak_centroid_near(
                wl_axis,
                sig,
                float(wavelengths[channel_index]),
                half_window_nm=half_window,
            )
            refined_wavelengths[channel_index] = refined_center
        _, fit_coefficients = _fit_wavelength_mapping(coordinates, refined_wavelengths)
    else:
        fit_coefficients = calibration_results.wavelength_fit_coefficients

    return CalibrationResult(
        wavelength=refined_wavelengths,
        coordinates=coordinates,
        max_level=max_level,
        min_level=min_level,
        level_range=level_values,
        intensity_levels=intensity_levels,
        raw_intensity_levels=raw_intensity_levels,
        wavelength_fit_coefficients=fit_coefficients,
    )


def _read_daq_value(
    daq: "DAQController",
    index: int,
    read_timeout: float,
    stop_event: threading.Event | None,
) -> float:
    """One averaged DAQ reading (volts); aborts cleanly on a stop request.

    ``monitor_cycle`` sleeps the DAQ's own configured settle (hold) before it
    reads, so the SLM frame has time to latch, and returns None when
    ``stop_event`` is already set.
    """
    sample = daq.monitor_cycle(
        index=index, timeout=read_timeout, stop_event=stop_event
    )
    if sample is None:
        raise CalibrationAborted("calibration stopped by request")
    return float(sample.value)


def intensity_calibration_daq(
    daq: "DAQController",
    slm: SLMController,
    levels: Iterable[int],
    calibration_results: CalibrationResult,
    window_size: int,
    *,
    coordinate_stride: int = 1,
    region: tuple[int, int] | None = None,
    read_timeout: float = 30.0,
    stop_event: threading.Event | None = None,
    progress_callback: ProgressCallback | None = None,
) -> CalibrationResult:
    """Step-3 intensity calibration read from a DAQ bucket detector.

    This is the OSA ``intensity_calibration`` with the spectrometer swapped for a
    single-photodiode DAQ: it walks the same calibrated coordinates, lights the
    same one window at a time (rest of the panel at ``min_level``), and produces
    the same ``CalibrationResult`` (``intensity_levels`` /
    ``raw_intensity_levels`` shaped ``(n_coordinates, n_levels)``) so Step 3's
    fit/plots and the downstream encoder are unchanged.

    Because a bucket detector has no spectral resolution, intensity is a plain
    dark-frame subtraction rather than a per-wavelength reduction: an all-``min``
    frame is read once as the DC background, then every window reading is
    ``reading - dark`` (in volts, clamped at 0). No all-``max`` bright reference
    is taken: the downstream sin^2 model ``I0 * sin(theta/2)^2`` fits ``I0`` as a
    free amplitude, so absolute scale is irrelevant to the fit, and lighting the
    whole panel at ``max_level`` could saturate or damage the photodiode. The
    OSA-only knobs (wavelength averaging, narrow sweep span, wavelength refine)
    have no analogue here and are intentionally absent.

    ``daq`` must already be connected and ``configure_monitor``-ed with the
    channel / rate / averaging / range / hold for this sweep; each reading uses
    that config (the hold provides the per-frame settle).
    """

    level_values = _validate_levels(levels)
    coordinates, wavelengths = _calibrated_mapping(calibration_results)
    coordinates, wavelengths = _select_region_mapping(coordinates, wavelengths, region)

    coordinate_stride = int(coordinate_stride)
    if coordinate_stride < 1:
        raise ValueError("coordinate_stride must be >= 1")
    if coordinate_stride > 1:
        coordinates = coordinates[::coordinate_stride]
        wavelengths = wavelengths[::coordinate_stride]

    slm_width, slm_height = slm.get_slm_info()
    window_size = _validate_window_size(window_size, slm_width)
    min_level = _level_value(calibration_results.min_level, "min_level")
    max_level = _level_value(calibration_results.max_level, "max_level")

    read_index = 0

    # Dark reference: whole panel at min_level. Its reading is the DC background
    # (detector dark current + stray light) that the offset-free downstream sin^2
    # model cannot absorb, so it is subtracted from every window reading. No
    # all-bright reference is taken (see the docstring): the fit's amplitude I0 is
    # free, so scale does not matter, and a full-bright panel could harm the APD.
    dark_pattern = np.full(slm_width, min_level, dtype=int)
    _display_1d_pattern(slm, dark_pattern, slm_height)
    dark_value = _read_daq_value(daq, read_index, read_timeout, stop_event)
    read_index += 1

    intensity_levels = np.zeros((coordinates.size, level_values.size), dtype=float)
    raw_intensity_levels = np.zeros((coordinates.size, level_values.size), dtype=float)
    total = int(coordinates.size * level_values.size)
    step = 0

    for coordinate_index, (coordinate, wavelength_nm) in enumerate(
        zip(coordinates, wavelengths)
    ):
        x_start = _window_start_from_coordinate(coordinate, window_size, slm_width)
        for level_index, level in enumerate(level_values):
            _check_stop(stop_event)
            pattern = dark_pattern.copy()
            pattern[x_start : x_start + window_size] = int(level)
            _display_1d_pattern(slm, pattern, slm_height)

            reading = _read_daq_value(daq, read_index, read_timeout, stop_event)
            read_index += 1
            # Dark-frame subtraction only; no bright normalization. The sin^2 fit
            # absorbs the absolute scale in its free amplitude, so raw
            # background-subtracted volts feed both arrays directly.
            value = max(0.0, reading - dark_value)

            raw_intensity_levels[coordinate_index, level_index] = value
            intensity_levels[coordinate_index, level_index] = value
            _report(
                progress_callback,
                "intensity",
                step,
                total,
                f"λ {wavelength_nm:.2f} nm "
                f"({coordinate_index + 1}/{coordinates.size}), "
                f"level {int(level)} -> {value * 1e3:.3f} mV",
                x=float(level),
                y=float(value),
            )
            step += 1

    return CalibrationResult(
        wavelength=np.asarray(wavelengths, dtype=float),
        coordinates=coordinates,
        max_level=max_level,
        min_level=min_level,
        level_range=level_values,
        intensity_levels=intensity_levels,
        raw_intensity_levels=raw_intensity_levels,
        wavelength_fit_coefficients=calibration_results.wavelength_fit_coefficients,
    )


def restrict_to_measured_intensity_range(
    calibration_results: CalibrationResult,
) -> CalibrationResult:
    """Keep and locally normalise the measured off-to-on range at one pixel.

    Quick single-channel calibration may probe an arbitrary user-supplied list
    of SLM levels.  The level with the least measured power becomes the local
    off level and the level with the greatest power becomes the local on level.
    Samples outside that inclusive rising segment are discarded so downstream
    encoding cannot accidentally use a falling or unmeasured branch.
    """
    levels = _validate_levels(calibration_results.level_range)
    if levels.size < 2 or np.any(np.diff(levels) <= 0):
        raise ValueError(
            "quick calibration levels must contain at least two unique, "
            "increasing values"
        )
    intensities = calibration_results.intensity_levels
    if intensities is None:
        raise ValueError("quick calibration has no measured intensity values")
    intensities = np.asarray(intensities, dtype=float)
    expected = (1, levels.size)
    if intensities.shape != expected:
        raise ValueError(
            f"quick calibration intensity shape must be {expected}, "
            f"got {intensities.shape}"
        )

    raw = calibration_results.raw_intensity_levels
    measured = intensities
    raw_array: np.ndarray | None = None
    if raw is not None:
        raw_array = np.asarray(raw, dtype=float)
        if raw_array.shape != expected:
            raise ValueError(
                f"quick calibration raw intensity shape must be {expected}, "
                f"got {raw_array.shape}"
            )
        measured = raw_array
    if not np.all(np.isfinite(measured)):
        raise ValueError("quick calibration contains NaN or infinity")

    row = measured[0]
    off_index = int(np.argmin(row))
    on_index = int(np.argmax(row))
    if off_index >= on_index:
        raise ValueError(
            "measured maximum must occur at a higher SLM level than the minimum"
        )
    denominator = float(row[on_index] - row[off_index])
    if denominator <= np.finfo(float).eps:
        raise ValueError("measured quick-calibration intensity range is zero")

    segment = slice(off_index, on_index + 1)
    local_intensity = np.clip(
        (measured[:, segment] - row[off_index]) / denominator,
        0.0,
        1.0,
    )
    return replace(
        calibration_results,
        min_level=int(levels[off_index]),
        max_level=int(levels[on_index]),
        level_range=levels[segment].copy(),
        intensity_levels=local_intensity,
        raw_intensity_levels=(
            None if raw_array is None else raw_array[:, segment].copy()
        ),
    )


def write_intensity_calibration_csv(
    calibration_results: CalibrationResult,
    csv_path: str | Path,
) -> Path:
    """Write intensity calibration data in long format.

    The first three measurement columns match calibration.load_calibration_csv:
    wavelength_nm, level, intensity (normalized). coordinate_px and
    raw_intensity_w (background-subtracted power, in watts) are included as
    useful extra metadata and are ignored by the existing loader.
    """

    if calibration_results.intensity_levels is None:
        raise ValueError("calibration_results.intensity_levels is empty")

    coordinates, wavelengths = _calibrated_mapping(calibration_results)
    levels = _validate_levels(calibration_results.level_range)
    intensities = np.asarray(calibration_results.intensity_levels, dtype=float)
    expected_shape = (coordinates.size, levels.size)
    if intensities.shape != expected_shape:
        raise ValueError(
            f"intensity_levels shape {intensities.shape} does not match "
            f"(n_coordinates, n_levels) {expected_shape}"
        )

    raw = calibration_results.raw_intensity_levels
    if raw is not None:
        raw = np.asarray(raw, dtype=float)
        if raw.shape != expected_shape:
            raise ValueError(
                f"raw_intensity_levels shape {raw.shape} does not match "
                f"(n_coordinates, n_levels) {expected_shape}"
            )

    path = Path(csv_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["coordinate_px", "wavelength_nm", "level", "intensity", "raw_intensity_w"]
        )
        for index, (coordinate, wavelength_nm, row) in enumerate(
            zip(coordinates, wavelengths, intensities)
        ):
            raw_row = raw[index] if raw is not None else None
            for level_index, (level, intensity) in enumerate(zip(levels, row)):
                raw_value = (
                    "" if raw_row is None else float(raw_row[level_index])
                )
                writer.writerow(
                    [
                        float(coordinate),
                        float(wavelength_nm),
                        int(level),
                        float(intensity),
                        raw_value,
                    ]
                )
    return path


def save_calibration_result(
    calibration_results: CalibrationResult,
    path: str | Path,
) -> Path:
    """Write a full CalibrationResult to JSON so a later step can resume from it.

    Every field is stored (arrays as nested lists), so loading the file rebuilds
    an equivalent CalibrationResult.
    """

    payload = {
        "schema": "calibration_result_v1",
        "wavelength": _to_jsonable(calibration_results.wavelength),
        "coordinates": _to_jsonable(calibration_results.coordinates),
        "max_level": _to_jsonable(calibration_results.max_level),
        "min_level": _to_jsonable(calibration_results.min_level),
        "level_range": _to_jsonable(calibration_results.level_range),
        "intensity_levels": _to_jsonable(calibration_results.intensity_levels),
        "raw_intensity_levels": _to_jsonable(calibration_results.raw_intensity_levels),
        "wavelength_fit_coefficients": _to_jsonable(
            calibration_results.wavelength_fit_coefficients
        ),
    }
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    return out


def calibration_result_from_dict(payload: dict) -> CalibrationResult:
    """Rebuild a CalibrationResult from a save_calibration_result payload dict.

    Accepts the parsed JSON either straight from a calibration file or embedded
    in another output (e.g. the step-6 combined result stores the raw step-3
    payload under its ``"step3"`` key).
    """

    return CalibrationResult(
        wavelength=_array_or_empty(payload.get("wavelength")),
        coordinates=_array_or_empty(payload.get("coordinates")),
        max_level=_scalar_level(payload.get("max_level")),
        min_level=_scalar_level(payload.get("min_level")),
        level_range=_array_or_empty(payload.get("level_range")),
        intensity_levels=_array_or_none(payload.get("intensity_levels")),
        raw_intensity_levels=_array_or_none(payload.get("raw_intensity_levels")),
        wavelength_fit_coefficients=_array_or_none(
            payload.get("wavelength_fit_coefficients")
        ),
    )


def load_calibration_result(path: str | Path) -> CalibrationResult:
    """Rebuild a CalibrationResult written by save_calibration_result."""

    src = Path(path).resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Calibration result not found: {src}")
    with open(src, "r", encoding="utf-8") as file:
        payload = json.load(file)

    return calibration_result_from_dict(payload)


def load_wavelength_map_csv(
    path: str | Path,
    *,
    min_level: int | None = None,
    max_level: int | None = None,
    level_range: Iterable[int] | None = None,
) -> CalibrationResult:
    """Build a CalibrationResult from a coordinate->wavelength CSV.

    Reads the ``coordinate_px`` and ``wavelength_nm`` columns (extra columns,
    e.g. the long-format calibration CSV's level/intensity, are ignored). One
    row per coordinate is kept. min/max levels and the level range come from the
    caller, since a bare wavelength map does not carry them; they default to the
    full 0..1023 range so the result is still usable for the dark/bright
    reference patterns when not supplied.
    """

    src = Path(path).resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Wavelength map CSV not found: {src}")

    with open(src, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError("Wavelength map CSV is empty")
        normalized = {name.strip(): name for name in reader.fieldnames}
        for column in ("coordinate_px", "wavelength_nm"):
            if column not in normalized:
                raise ValueError(
                    f"Wavelength map CSV missing required column: {column}"
                )

        mapping: dict[float, float] = {}
        for row in reader:
            if not any((value or "").strip() for value in row.values()):
                continue
            coordinate = float(row[normalized["coordinate_px"]])
            wavelength = float(row[normalized["wavelength_nm"]])
            mapping.setdefault(coordinate, wavelength)

    if not mapping:
        raise ValueError("Wavelength map CSV does not contain any rows")

    coordinates = np.asarray(list(mapping.keys()), dtype=float)
    wavelengths = np.asarray(list(mapping.values()), dtype=float)
    levels = (
        np.asarray(list(level_range), dtype=int)
        if level_range is not None
        else np.asarray([], dtype=int)
    )
    return CalibrationResult(
        wavelength=wavelengths,
        coordinates=coordinates,
        max_level=int(max_level) if max_level is not None else 1023,
        min_level=int(min_level) if min_level is not None else 0,
        level_range=levels,
    )


def _to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _array_or_empty(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=float)
    return np.asarray(value, dtype=float)


def _array_or_none(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=float)


def _scalar_level(value: Any) -> int | np.ndarray:
    if isinstance(value, list):
        return np.asarray(value, dtype=int)
    return int(value)


def _trace_power_w(trace: TraceData) -> np.ndarray:
    powers = np.asarray(trace.powers, dtype=float)
    if trace.power_label == "power_dBm":
        powers = 1e-3 * (10.0 ** (powers / 10.0))
    return np.nan_to_num(powers, nan=0.0, posinf=0.0, neginf=0.0)


def _check_stop(stop_event: threading.Event | None) -> None:
    if stop_event is not None and stop_event.is_set():
        raise CalibrationAborted("calibration stopped by request")


def _reduce_arrays(
    wavelengths: np.ndarray,
    power_w: np.ndarray,
    background_power_w: np.ndarray,
    reference_power_w: np.ndarray,
    *,
    denominator_scale: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    signal = np.asarray(power_w, dtype=float) - background_power_w
    signal = np.clip(np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)

    denominator = reference_power_w - background_power_w
    # Only normalize where the bright reference actually carries light. Where
    # it does not (outside the source spectrum, or light the SLM never
    # modulates), the ratio is drift-residue over nothing and inflates into
    # spurious peaks that can beat the real one.
    scale = denominator_scale
    if scale is None:
        scale = float(np.max(denominator)) if denominator.size else 0.0
    floor = max(_MIN_REFERENCE_FRACTION * scale, float(np.finfo(float).eps))
    normalized = np.zeros(signal.size, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(signal, denominator, out=normalized, where=denominator > floor)

    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    return wavelengths, signal, np.clip(normalized, 0.0, None)


def _reduce_trace(
    trace: TraceData,
    power_w: np.ndarray,
    background_power_w: np.ndarray,
    reference_power_w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reduce a measured trace to (wavelengths, signal_W, normalized).

    signal_W is the background-subtracted power in watts (clipped at 0);
    normalized divides that by the bright reference (reference - background),
    also clipped at 0, and only where the reference rises above
    _MIN_REFERENCE_FRACTION of its own peak. Both share the returned
    wavelength axis.
    """
    count = min(
        trace.wavelengths_nm.size,
        power_w.size,
        background_power_w.size,
        reference_power_w.size,
    )
    if count <= 0:
        raise ValueError("Trace, background, and reference must not be empty")

    return _reduce_arrays(
        trace.wavelengths_nm[:count],
        power_w[:count],
        background_power_w[:count],
        reference_power_w[:count],
    )


def _reduce_trace_resampled(
    trace: TraceData,
    power_w: np.ndarray,
    reference_axis_nm: np.ndarray,
    background_power_w: np.ndarray,
    reference_power_w: np.ndarray,
    *,
    denominator_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reduce a trace whose wavelength grid differs from the reference grid.

    The wide-span background/reference are interpolated onto the (narrow)
    trace axis. ``denominator_scale`` carries the wide-grid peak reference so
    the validity floor stays relative to the source maximum rather than to
    whatever slice of it the narrow span happens to cover.
    """
    count = min(trace.wavelengths_nm.size, power_w.size)
    if count <= 0:
        raise ValueError("Trace must not be empty")
    n_ref = min(
        np.asarray(reference_axis_nm).size,
        background_power_w.size,
        reference_power_w.size,
    )
    if n_ref <= 0:
        raise ValueError("Background and reference must not be empty")

    wavelengths = np.asarray(trace.wavelengths_nm[:count], dtype=float)
    axis = np.asarray(reference_axis_nm, dtype=float)[:n_ref]
    background = np.interp(wavelengths, axis, background_power_w[:n_ref])
    reference = np.interp(wavelengths, axis, reference_power_w[:n_ref])
    return _reduce_arrays(
        wavelengths,
        power_w[:count],
        background,
        reference,
        denominator_scale=denominator_scale,
    )


def _display_1d_pattern(
    slm: SLMController,
    pattern: np.ndarray,
    slm_height: int,
) -> None:
    pattern = np.asarray(pattern, dtype=int)
    if pattern.ndim != 1:
        raise ValueError("pattern must be a 1D array")
    slm.display_array(np.broadcast_to(pattern[None, :], (slm_height, pattern.size)).copy())


def _fit_wavelength_mapping(
    coordinates: np.ndarray,
    wavelengths_nm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if coordinates.size == 0:
        raise ValueError("No wavelength calibration points were collected")
    if coordinates.size != wavelengths_nm.size:
        raise ValueError("coordinates and wavelengths_nm must have the same length")

    if coordinates.size == 1:
        return wavelengths_nm.astype(float, copy=True), np.asarray([wavelengths_nm[0]])

    degree = min(3, coordinates.size - 1)
    coeffs = np.polyfit(coordinates, wavelengths_nm, deg=degree)
    return np.polyval(coeffs, coordinates), coeffs


def _linear_wavelength_fit(
    calibration_results: CalibrationResult,
) -> tuple[float, float]:
    coordinates, wavelengths = _calibrated_mapping(calibration_results)
    if coordinates.size < 2:
        raise ValueError("at least two Step 2 points are required for a linear fit")
    slope, intercept = np.polyfit(coordinates, wavelengths, 1)
    slope = float(slope)
    intercept = float(intercept)
    if not np.isfinite(slope) or not np.isfinite(intercept):
        raise ValueError("Step 2 linear fit is invalid")
    if abs(slope) <= np.finfo(float).eps:
        raise ValueError("Step 2 wavelength slope is zero")
    return slope, intercept


def _configured_measurement(
    osa: OSAController,
    settings: MeasurementSettings,
) -> Callable[[], TraceData]:
    """Configure once when the OSA object supports it, then reuse the settings."""

    configure = getattr(osa, "configure", None)
    if callable(configure):
        configure(settings)
        return lambda: osa.measure()
    return lambda: osa.measure(settings)


def _resolve_refine_half_window_nm(
    wavelengths: np.ndarray,
    requested: float | None,
) -> float:
    if requested is not None:
        window = float(requested)
        if not np.isfinite(window) or window <= 0.0:
            raise ValueError("refine_half_window_nm must be positive")
        return window

    ordered = np.sort(np.asarray(wavelengths, dtype=float))
    diffs = np.abs(np.diff(ordered))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if diffs.size:
        return max(float(np.min(diffs)) * 0.45, 1e-6)
    return 0.2


def _wavelength_guard_mask(
    slm_width: int,
    calibration_results: CalibrationResult,
    guard_bands_nm: Iterable[tuple[float, float]] | None,
) -> np.ndarray:
    mask = np.zeros(int(slm_width), dtype=bool)
    if guard_bands_nm is None:
        return mask

    bands = _validate_guard_bands_nm(guard_bands_nm)
    if not bands:
        return mask

    slope, intercept = _linear_wavelength_fit(calibration_results)
    columns = np.arange(int(slm_width), dtype=float)
    wavelengths = slope * columns + intercept
    tolerance = np.finfo(float).eps * max(1.0, float(np.max(np.abs(wavelengths)))) * 8.0
    for center_nm, half_width_nm in bands:
        mask |= np.abs(wavelengths - center_nm) <= half_width_nm + tolerance
    return mask


def _channel_window_bounds(coordinate: float, channel_width_px: int) -> tuple[int, int]:
    start = int(round(float(coordinate))) - int(channel_width_px) // 2
    return start, start + int(channel_width_px)


def _channel_window_overlaps_guard(
    coordinate: float,
    channel_width_px: int,
    slope: float,
    intercept: float,
    guard_bands: list[tuple[float, float]],
) -> bool:
    if not guard_bands:
        return False
    start, end = _channel_window_bounds(coordinate, channel_width_px)
    if end <= start:
        return False
    wavelengths = slope * np.arange(start, end, dtype=float) + intercept
    if wavelengths.size == 0:
        return False
    tolerance = np.finfo(float).eps * max(1.0, float(np.max(np.abs(wavelengths)))) * 8.0
    for center_nm, half_width_nm in guard_bands:
        if np.any(np.abs(wavelengths - center_nm) <= half_width_nm + tolerance):
            return True
    return False


def _collect_guarded_channel_side(
    center: float,
    direction: int,
    pitch_px: int,
    desired_count: int,
    is_valid: Callable[[float], bool],
    coord_min: float,
    coord_max: float,
    first_offset_px: float | None = None,
) -> list[float]:
    channels: list[float] = []
    index = 0
    while len(channels) < desired_count:
        if first_offset_px is None:
            offset = (index + 0.5) * pitch_px          # legacy half-pitch start
        else:
            offset = float(first_offset_px) + index * pitch_px
        coordinate = center + int(direction) * offset
        if direction < 0 and coordinate < coord_min:
            break
        if direction > 0 and coordinate > coord_max:
            break
        if is_valid(coordinate):
            channels.append(float(coordinate))
        index += 1
    return channels


def _validate_guard_bands_nm(
    guard_bands_nm: Iterable[tuple[float, float]],
) -> list[tuple[float, float]]:
    bands: list[tuple[float, float]] = []
    for index, band in enumerate(guard_bands_nm):
        try:
            center_nm, half_width_nm = band
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "guard_bands_nm entries must be (center_nm, half_width_nm) pairs"
            ) from exc
        center = float(center_nm)
        half_width = float(half_width_nm)
        if not np.isfinite(center):
            raise ValueError(f"guard band {index} center must be finite")
        if not np.isfinite(half_width) or half_width <= 0.0:
            raise ValueError(f"guard band {index} half-width must be positive")
        bands.append((center, half_width))
    return bands


def _validate_guard_bands_symmetric(
    guard_bands: list[tuple[float, float]],
    target_wavelength_nm: float,
    *,
    tolerance_nm: float = 1e-6,
) -> None:
    """Reject guard bands that are not mirror-symmetric about the target.

    Channel candidates sit at ``center +/- (k + 0.5) * pitch``, so a guard band
    at ``(c, w)`` skips the same k-th candidate on its side that a partner band
    at ``(2 * target - c, w)`` skips on the other side. Without that partner the
    two sides drop different candidates and the k-th left/right pair no longer
    shares a wavelength offset. Each band must therefore have a mirror partner
    (a band centred on the target is its own mirror); multiplicities must match
    too, so two identical bands need two mirrors.
    """

    target = float(target_wavelength_nm)
    remaining = list(guard_bands)
    for center, half_width in guard_bands:
        mirror = 2.0 * target - center
        match_index: int | None = None
        for index, (other_center, other_half) in enumerate(remaining):
            if (
                abs(other_center - mirror) <= tolerance_nm
                and abs(other_half - half_width) <= tolerance_nm
            ):
                match_index = index
                break
        if match_index is None:
            raise ValueError(
                f"guard band ({center:g} +/- {half_width:g} nm) has no mirror "
                f"about the target {target:g} nm (expected a band near "
                f"{mirror:g} +/- {half_width:g} nm); guard bands must be "
                f"symmetric so left/right channel pairs stay wavelength-symmetric"
            )
        remaining.pop(match_index)


def _calibrated_mapping(
    calibration_results: CalibrationResult,
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.asarray(calibration_results.coordinates, dtype=float)
    wavelengths = np.asarray(calibration_results.wavelength, dtype=float)

    if coordinates.ndim != 1 or wavelengths.ndim != 1:
        raise ValueError("coordinates and wavelength must be 1D arrays")
    if coordinates.size == 0:
        raise ValueError("wavelength calibration must run before intensity calibration")
    if coordinates.size != wavelengths.size:
        raise ValueError(
            f"coordinates and wavelength size mismatch: "
            f"{coordinates.size} vs {wavelengths.size}"
        )
    if not np.all(np.isfinite(coordinates)) or not np.all(np.isfinite(wavelengths)):
        raise ValueError("coordinates and wavelength must be finite")

    order = np.argsort(coordinates)
    return coordinates[order], wavelengths[order]


def _window_start_from_coordinate(
    coordinate: float,
    window_size: int,
    slm_width: int,
) -> int:
    start = int(round(float(coordinate))) - window_size // 2
    return max(0, min(start, slm_width - window_size))


def _region_bounds(region: tuple[int, int]) -> tuple[int, int]:
    start, end = int(region[0]), int(region[1])
    if end < start:
        raise ValueError("region end must be >= region start")
    return start, end


def _resolve_scan_region(
    region: tuple[int, int] | None,
    slm_width: int,
    window_size: int,
) -> tuple[int, int]:
    """Window-start range [lo, hi) that keeps the window inside the region.

    Returns the full-width range when region is None; raises when the region is
    narrower than the window.
    """
    max_start = slm_width - window_size + 1
    if region is None:
        return 0, max_start
    start, end = _region_bounds(region)
    lo = max(0, start)
    hi = min(max_start, end - window_size + 2)
    if hi <= lo:
        raise ValueError("region is too small for the window size")
    return lo, hi


def _select_region_mapping(
    coordinates: np.ndarray,
    wavelengths: np.ndarray,
    region: tuple[int, int] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep only the calibrated coordinates within the region (inclusive)."""
    if region is None:
        return coordinates, wavelengths
    start, end = _region_bounds(region)
    mask = (coordinates >= start) & (coordinates <= end)
    if not np.any(mask):
        raise ValueError("no calibrated coordinates fall within the region")
    return coordinates[mask], wavelengths[mask]


def _validate_levels(levels: Iterable[int]) -> np.ndarray:
    try:
        values = np.asarray(list(levels), dtype=float)
    except TypeError as exc:
        raise ValueError("levels must be an iterable of integers") from exc

    if values.ndim != 1 or values.size == 0:
        raise ValueError("levels must be a non-empty 1D sequence")
    if not np.all(np.isfinite(values)):
        raise ValueError("levels must be finite")

    rounded = np.rint(values)
    if not np.array_equal(values, rounded):
        raise ValueError("levels must contain integer grayscale levels")
    if np.any(rounded < 0) or np.any(rounded > 1023):
        raise ValueError("levels must be in 0..1023")
    return rounded.astype(int)


def _level_value(value: int | np.ndarray, name: str) -> int:
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    if array.size > 1 and not np.all(array == array.flat[0]):
        raise ValueError(f"{name} must be a scalar level")

    level = float(array.flat[0])
    rounded = round(level)
    if level != rounded:
        raise ValueError(f"{name} must be an integer level")
    if rounded < 0 or rounded > 1023:
        raise ValueError(f"{name} must be in 0..1023")
    return int(rounded)


def _validate_window_size(window_size: int, slm_width: int) -> int:
    result = _validate_non_negative_int(window_size, "window_size")
    if result <= 0:
        raise ValueError("window_size must be positive")
    if result > slm_width:
        raise ValueError("window_size cannot exceed SLM width")
    return result


def _validate_non_negative_int(value: Any, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _finite_mean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    return float(np.mean(finite))
