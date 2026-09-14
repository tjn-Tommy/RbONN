"""SLM response calibration sweeps (steps 1-3), read on the OSA or the DAQ.

Each step is one ``run_step*`` call -- the GUI's Run buttons and the
``steps/calib_step{1,2,3b,3c}.py`` scripts both make it, so the sequence a step
runs is written once, here:

    run_step1    find_min_max_intensity_levels        full-screen level sweep (OSA)
    run_step2    wavelength_calibration               bright-window sweep -> the
                 + clip_to_wavelength_range           coordinate-to-wavelength map (OSA)
    run_step3b   refine_center_coordinate_with_osa    target window -> grid centre,
                 + build_channel_calibration_grid     channel grid,
                 + batch_intensity_calibration        grouped channel sweeps (OSA)
    run_step3c   build_channel_calibration_grid       channel grid,
                 + intensity_calibration_daq          one window at a time (DAQ)

``intensity_calibration`` is the retired step 3a (per-coordinate OSA sweep); no
runner calls it.  A runner returns the ``CalibrationResult`` and saves nothing:
writing it is ``fit.slm_response.save_calibration_result``, which also fits the
transfer curves.  The result format, and all the array math these sweeps call,
live in ``calibration_module.fit.slm_response``.
"""
from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from osa_module.controller import MeasurementSettings, OSAController, TraceData
from slm_module.controller import SLMController

from ..fit.outliers import (
    OutlierRemeasurePolicy,
    flag_by_residual,
    linear_fit_residuals,
    transfer_fit_residuals,
)
from ..fit.slm_response import (
    CalibrationResult,
    _calibrated_mapping,
    _driven_geometry,
    _fit_wavelength_mapping,
    _level_value,
    _linear_wavelength_fit,
    _resolve_refine_half_window_nm,
    _resolve_scan_region,
    _select_region_mapping,
    _validate_levels,
    _validate_non_negative_int,
    _validate_window_size,
    _wavelength_guard_mask,
    _window_start_from_coordinate,
    build_channel_calibration_grid,
    clip_to_wavelength_range,
    local_peak_centroid,
    local_peak_centroid_near,
    mean_near_wavelength,
)

if TYPE_CHECKING:  # avoid importing daq_module at runtime
    from daq_module.controller import DAQController


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
    active area). None (default) leaves that end of the trace unclipped.  This
    is a mask on the SEARCH, so a window whose real peak falls outside it reads
    as dark and aborts the run; it is not the usable output band.  To keep only
    part of the finished mapping -- the band downstream steps can actually use
    -- sweep unmasked and pass the result through
    :func:`clip_to_wavelength_range`, which is what the GUI's step 2 does.

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
                    f"(normalized strength {anchor_strength:.3g}); check that "
                    "the scan region is actually illuminated and that the OSA "
                    "span covers it. If a peak-search mask "
                    "(min/max_peak_wavelength_nm) is set, a real peak outside "
                    "it reads as dark here -- sweep unmasked and narrow the "
                    "finished mapping with clip_to_wavelength_range instead"
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

    driven_width, driven_gap = _driven_geometry(
        calibration_results, window_size, coordinates
    )
    return CalibrationResult(
        wavelength=result_wavelengths,
        coordinates=coordinates,
        max_level=max_level,
        min_level=min_level,
        level_range=level_values,
        intensity_levels=intensity_levels,
        raw_intensity_levels=raw_intensity_levels,
        wavelength_fit_coefficients=fit_coefficients,
        channel_width_px=driven_width,
        gap_px=driven_gap,
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

    driven_width, driven_gap = _driven_geometry(
        calibration_results, window_size, coordinates
    )
    return CalibrationResult(
        wavelength=refined_wavelengths,
        coordinates=coordinates,
        max_level=max_level,
        min_level=min_level,
        level_range=level_values,
        intensity_levels=intensity_levels,
        raw_intensity_levels=raw_intensity_levels,
        wavelength_fit_coefficients=fit_coefficients,
        channel_width_px=driven_width,
        gap_px=driven_gap,
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

    driven_width, driven_gap = _driven_geometry(
        calibration_results, window_size, coordinates
    )
    return CalibrationResult(
        wavelength=np.asarray(wavelengths, dtype=float),
        coordinates=coordinates,
        max_level=max_level,
        min_level=min_level,
        level_range=level_values,
        intensity_levels=intensity_levels,
        raw_intensity_levels=raw_intensity_levels,
        wavelength_fit_coefficients=calibration_results.wavelength_fit_coefficients,
        channel_width_px=driven_width,
        gap_px=driven_gap,
    )


# ======================================================================
# step runners -- one call per step, shared by the GUI and steps/calib_step*.py
# ======================================================================


@dataclass
class ChannelSweep:
    """A step-3b/3c run: the calibration plus where its channel grid sits.

    ``center_coordinate`` is the SLM column the grid was tiled around.
    ``coarse_center`` and ``measured_peak_nm`` are set only when step 3b
    fine-tuned that centre on the OSA: the step-2 prediction it started from,
    and the peak the OSA actually saw there.
    """

    result: CalibrationResult
    center_coordinate: float
    coarse_center: float | None = None
    measured_peak_nm: float | None = None


def _report_grid_center(
    progress_callback: ProgressCallback | None,
    target_wavelength_nm: float,
    center_coordinate: float,
) -> None:
    _report(
        progress_callback,
        "fast_center",
        0,
        1,
        f"Step 2 predicts {target_wavelength_nm:.4f} nm at "
        f"x={center_coordinate:.3f} px",
        x=center_coordinate,
        y=target_wavelength_nm,
    )


def run_step1(
    osa: OSAController,
    slm: SLMController,
    levels: Iterable[int],
    measure_settings: MeasurementSettings,
    *,
    stop_event: threading.Event | None = None,
    progress_callback: ProgressCallback | None = None,
) -> CalibrationResult:
    """Step 1: the off/on levels, as the seed step 2 reads.

    The result carries no wavelength map -- only ``min_level``, ``max_level``
    and the swept ``level_range``.
    """
    level_values = [int(level) for level in levels]
    _mn, _mx, min_level, max_level, _records = find_min_max_intensity_levels(
        osa, slm, level_values, measure_settings,
        stop_event=stop_event, progress_callback=progress_callback,
    )
    return CalibrationResult(
        wavelength=np.asarray([]),
        coordinates=np.asarray([]),
        max_level=max_level,
        min_level=min_level,
        level_range=np.asarray(level_values, dtype=int),
    )


def run_step2(
    osa: OSAController,
    slm: SLMController,
    measure_settings: MeasurementSettings,
    seed: CalibrationResult,
    *,
    window_size: int,
    peak_half_window_nm: float | None = None,
    region: tuple[int, int] | None = None,
    coordinate_stride: int = 1,
    sweep_span_nm: float | None = None,
    min_wavelength_nm: float | None = None,
    max_wavelength_nm: float | None = None,
    stop_event: threading.Event | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[CalibrationResult, CalibrationResult]:
    """Step 2: the coordinate -> wavelength map, clipped to the usable band.

    Returns ``(clipped, swept)``.  The region is swept unmasked: this step only
    has to land one straight coordinate -> wavelength line, and a window whose
    peak sits outside the usable band still measures a real peak that belongs
    in the fit.  Masking the peak search instead used to fail the run outright
    ("shows no clear peak") whenever a region edge dispersed past the band.
    The band is applied only on the way out, so ``clipped`` -- what is saved
    and handed to step 3 -- keeps just the usable coordinates, while its fit
    coefficients still describe the whole swept region.
    """
    swept = wavelength_calibration(
        osa, slm, [], measure_settings, seed,
        window_size=window_size,
        peak_half_window_nm=peak_half_window_nm,
        region=region,
        coordinate_stride=coordinate_stride,
        sweep_span_nm=sweep_span_nm,
        stop_event=stop_event,
        progress_callback=progress_callback,
    )
    clipped = clip_to_wavelength_range(swept, min_wavelength_nm, max_wavelength_nm)
    return clipped, swept


def run_step3b(
    osa: OSAController,
    slm: SLMController,
    levels: Iterable[int],
    measure_settings: MeasurementSettings,
    step2: CalibrationResult,
    *,
    target_wavelength_nm: float,
    n_channels_per_side: int,
    channel_width_px: int,
    gap_px: int,
    group_skip_channels: int,
    fine_tune_center: bool = True,
    peak_half_window_nm: float = 0.2,
    wavelength_window_nm: float | None = None,
    refine_wavelength: bool = False,
    refine_half_window_nm: float | None = None,
    guard_bands_nm: Iterable[tuple[float, float]] | None = None,
    stop_event: threading.Event | None = None,
    progress_callback: ProgressCallback | None = None,
) -> ChannelSweep:
    """Step 3b: tile the channel grid, then sweep it in groups on the OSA.

    With ``fine_tune_center`` the target window is measured once first and the
    grid is tiled around the column the OSA puts the target at, rather than
    where the step-2 line predicts it.
    """
    slm_width, _slm_height = slm.get_slm_info()
    coarse_center = measured_peak = refined_center = None
    if fine_tune_center:
        refined_center, measured_peak, coarse_center = refine_center_coordinate_with_osa(
            osa, slm, measure_settings, step2,
            target_wavelength_nm=target_wavelength_nm,
            window_size=channel_width_px,
            peak_half_window_nm=peak_half_window_nm,
            stop_event=stop_event,
            progress_callback=progress_callback,
        )
    grid_seed, center_coordinate = build_channel_calibration_grid(
        step2,
        target_wavelength_nm=target_wavelength_nm,
        center_coordinate=refined_center,
        n_channels_per_side=n_channels_per_side,
        channel_width_px=channel_width_px,
        gap_px=gap_px,
        slm_width=slm_width,
        guard_bands_nm=guard_bands_nm,
    )
    if not fine_tune_center:
        _report_grid_center(progress_callback, target_wavelength_nm, center_coordinate)
    result = batch_intensity_calibration(
        osa, slm, levels, measure_settings, grid_seed,
        window_size=channel_width_px,
        wavelength_window_nm=wavelength_window_nm,
        group_skip_channels=group_skip_channels,
        guard_bands_nm=guard_bands_nm,
        refine_wavelength=refine_wavelength,
        refine_half_window_nm=refine_half_window_nm,
        stop_event=stop_event,
        progress_callback=progress_callback,
    )
    return ChannelSweep(result, center_coordinate, coarse_center, measured_peak)


def run_step3c(
    daq: "DAQController",
    slm: SLMController,
    levels: Iterable[int],
    step2: CalibrationResult,
    *,
    target_wavelength_nm: float,
    n_channels_per_side: int,
    channel_width_px: int,
    gap_px: int,
    guard_bands_nm: Iterable[tuple[float, float]] | None = None,
    read_timeout: float = 30.0,
    stop_event: threading.Event | None = None,
    progress_callback: ProgressCallback | None = None,
) -> ChannelSweep:
    """Step 3c: tile the channel grid, then sweep it one window at a time on the DAQ.

    ``daq`` must already be connected and ``configure_monitor``-ed, as for
    :func:`intensity_calibration_daq`.
    """
    slm_width, _slm_height = slm.get_slm_info()
    grid_seed, center_coordinate = build_channel_calibration_grid(
        step2,
        target_wavelength_nm=target_wavelength_nm,
        n_channels_per_side=n_channels_per_side,
        channel_width_px=channel_width_px,
        gap_px=gap_px,
        slm_width=slm_width,
        guard_bands_nm=guard_bands_nm,
    )
    _report_grid_center(progress_callback, target_wavelength_nm, center_coordinate)
    result = intensity_calibration_daq(
        daq, slm, levels, grid_seed,
        window_size=channel_width_px,
        read_timeout=read_timeout,
        stop_event=stop_event,
        progress_callback=progress_callback,
    )
    return ChannelSweep(result, center_coordinate)


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
