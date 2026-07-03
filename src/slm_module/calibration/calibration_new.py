from __future__ import annotations

import csv
import json
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from osa_module.controller import MeasurementSettings, OSAController, TraceData
from slm_module.controller import SLMController


"""
Calibration module for Santec SLM with AQ637X OSA.

Step 1: find rough minimum and maximum intensity levels by sweeping full-screen
grayscale levels.
Step 2: use a bright window sweep to map SLM x coordinates to wavelengths.
Step 3: for each calibrated coordinate, sweep grayscale levels and measure both
the absolute (background-subtracted, in watts) and the normalized intensity
averaged around that coordinate's calibrated wavelength.
"""


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
    """

    del levels
    slm_width, slm_height = slm.get_slm_info()
    window_size = _validate_window_size(window_size, slm_width)
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

    coordinates: list[int] = []
    wavelengths: list[float] = []

    total = max(0, region_hi - region_lo)
    for index, x_start in enumerate(range(region_lo, region_hi)):
        _check_stop(stop_event)
        pattern = dark_pattern.copy()
        pattern[x_start : x_start + window_size] = max_level
        _display_1d_pattern(slm, pattern, slm_height)

        trace = osa.measure(measure_settings)
        trace_wavelengths, _signal, normalized = _reduce_trace(
            trace, _trace_power_w(trace), background_power, reference_power
        )
        wavelength, _, _ = local_peak_centroid(
            trace_wavelengths,
            normalized,
            half_window=peak_half_window,
            half_window_nm=peak_half_window_nm,
        )
        coordinate = x_start + window_size // 2
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
    fitted_wavelengths, coeffs = _fit_wavelength_mapping(
        coordinate_array, wavelength_array
    )

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


def load_calibration_result(path: str | Path) -> CalibrationResult:
    """Rebuild a CalibrationResult written by save_calibration_result."""

    src = Path(path).resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Calibration result not found: {src}")
    with open(src, "r", encoding="utf-8") as file:
        payload = json.load(file)

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


def _reduce_trace(
    trace: TraceData,
    power_w: np.ndarray,
    background_power_w: np.ndarray,
    reference_power_w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reduce a measured trace to (wavelengths, signal_W, normalized).

    signal_W is the background-subtracted power in watts (clipped at 0);
    normalized divides that by the bright reference (reference - background),
    also clipped at 0. Both share the returned wavelength axis.
    """
    count = min(
        trace.wavelengths_nm.size,
        power_w.size,
        background_power_w.size,
        reference_power_w.size,
    )
    if count <= 0:
        raise ValueError("Trace, background, and reference must not be empty")

    wavelengths = trace.wavelengths_nm[:count]
    signal = np.asarray(power_w[:count], dtype=float) - background_power_w[:count]
    signal = np.clip(np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)

    denominator = reference_power_w[:count] - background_power_w[:count]
    normalized = np.zeros(count, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(
            signal,
            denominator,
            out=normalized,
            where=np.abs(denominator) > np.finfo(float).eps,
        )

    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    return wavelengths, signal, np.clip(normalized, 0.0, None)


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
