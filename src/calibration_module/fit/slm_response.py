"""SLM response calibration (steps 1-3): the result format and its array math.

``CalibrationResult`` is what steps 1, 2 and 3 hand on -- to each other, to the
encoder (``slm_module.encoding``) and to steps 6-8 -- so it lives here, beside
the JSON/CSV readers and writers that define its file layout.  Also here: the
step-3 channel grid designed from a step-2 map, the peak centroids, the band
clip, and the validators the sweeps share.

Nothing in here imports a driver.  The sweeps that fill these arrays are in
``calibration_module.measure.slm_response``.
"""
from __future__ import annotations

import csv
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .transfer import TransferFit, TransferFitError, fit_transfer_curves


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
    # One fitted sin^2 transfer curve per channel, in `coordinates` order --
    # what the encoder inverts to turn a requested value into a grayscale level
    # (calibration_module.fit.transfer, slm_module.encoding.EncodingChannel).
    # Filled in by save_calibration_result so the encoding is frozen with the
    # calibration rather than re-derived on every load; None on files written
    # before that, which the encoder then fits at load time.
    transfer_fits: list[TransferFit] | None = None
    # The SLM window geometry this grid was designed and swept with, in pixels:
    # `channel_width_px` lit columns per channel and `gap_px` dark columns
    # between neighbours, so width + gap is the coordinate pitch.  Step 3 is the
    # only place that knows the split -- `coordinates` records the channel
    # CENTRES and nothing else -- so it is stored here and read back by
    # slm_module.encoding.channel_layout_from_calibration instead of being
    # guessed from the pitch.  None on a dense scan, which has no channels, and
    # on files written before this was recorded; the encoder then falls back to
    # its `pitch - assumed_gap_px` guess and says so.
    channel_width_px: int | None = None
    gap_px: int | None = None


def _driven_geometry(
    seed: "CalibrationResult",
    window_size: int,
    coordinates: Any,
) -> tuple[int | None, int | None]:
    """The (window, pad) geometry a Step-3 sweep actually drove, in pixels.

    ``window_size`` is what was lit, so it is the width the measured transfer
    curves belong to -- not whatever the seed was designed at, if the two ever
    disagree.  The pad is what the channel pitch leaves over, except that a seed
    already designed at this width keeps its own ``gap_px`` verbatim.

    Returns ``(None, None)`` when the coordinates are not a channel grid (a
    dense scan steps a pixel at a time, so neighbouring windows overlap and
    "pad" means nothing).
    """
    width = int(window_size)
    if width < 1:
        return None, None
    coords = np.asarray(coordinates, dtype=float).ravel()
    if coords.size < 2:
        return None, None
    pitch = int(round(float(np.min(np.diff(np.sort(coords))))))
    if pitch <= width:
        return None, None
    if seed.channel_width_px == width and seed.gap_px is not None:
        return width, int(seed.gap_px)
    return width, pitch - width


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


def clip_to_wavelength_range(
    calibration_results: CalibrationResult,
    min_wavelength_nm: float | None = None,
    max_wavelength_nm: float | None = None,
) -> CalibrationResult:
    """Keep only the coordinates whose mapped wavelength lies in [min, max] nm.

    Step 2 sweeps whatever SLM region is illuminated and fits ONE line through
    every peak it measured.  A peak outside the usable band is good fit data,
    not an error -- the wider lever arm pins the slope down better -- so the
    band is not a peak-search mask.  It is a property of the DOWNSTREAM steps:
    only channels inside it carry usable light.  Hence it is applied here,
    where the mapping is handed on (saved to JSON, or passed to step 3), and
    never during acquisition.

    ``wavelength_fit_coefficients`` is carried over untouched: it is the fit
    over the whole swept region, and it is what predicts a wavelength at a
    coordinate this clip dropped.  Per-coordinate arrays (the intensity sweeps,
    the transfer fits, per-coordinate min/max levels) are sliced along with the
    mapping; ``level_range`` and scalar levels are not per coordinate and are
    left alone.  ``None`` on either bound leaves that end unclipped.
    """
    coordinates = np.asarray(calibration_results.coordinates).ravel()
    wavelengths = np.asarray(calibration_results.wavelength, dtype=float).ravel()
    if coordinates.size != wavelengths.size:
        raise ValueError(
            f"coordinate/wavelength size mismatch: {coordinates.size} vs "
            f"{wavelengths.size}"
        )
    if coordinates.size == 0:
        raise ValueError("calibration has no coordinate -> wavelength mapping")

    lo = -np.inf
    if min_wavelength_nm is not None:
        lo = float(min_wavelength_nm)
        if not np.isfinite(lo):
            raise ValueError("min_wavelength_nm must be finite")
    hi = np.inf
    if max_wavelength_nm is not None:
        hi = float(max_wavelength_nm)
        if not np.isfinite(hi):
            raise ValueError("max_wavelength_nm must be finite")
    if lo >= hi:
        raise ValueError("min_wavelength_nm must be below max_wavelength_nm")

    keep = np.flatnonzero((wavelengths >= lo) & (wavelengths <= hi))
    if keep.size == 0:
        raise ValueError(
            f"no calibrated coordinate maps into {lo:g}-{hi:g} nm; the swept "
            f"region covers {wavelengths.min():.3f}-{wavelengths.max():.3f} nm"
        )
    if keep.size == wavelengths.size:
        return calibration_results

    def _rows(value: Any) -> Any:
        """Slice an intensity sweep, which is one ROW per coordinate."""
        if value is None:
            return None
        array = np.asarray(value)
        # A step-1 seed's sweep rides along in the same field but is 1-D (one
        # entry per LEVEL), so only a 2-D per-coordinate block is sliced.
        if array.ndim != 2 or array.shape[0] != wavelengths.size:
            return value
        return array[keep].copy()

    def _levels(value: Any) -> Any:
        """Slice a per-coordinate min/max level; a scalar level is shared."""
        array = np.asarray(value)
        if array.ndim != 1 or array.size != wavelengths.size:
            return value
        return array[keep].copy()

    fits = calibration_results.transfer_fits
    if fits is not None and len(fits) == wavelengths.size:
        fits = [fits[int(i)] for i in keep]

    return replace(
        calibration_results,
        wavelength=wavelengths[keep],
        coordinates=coordinates[keep],
        max_level=_levels(calibration_results.max_level),
        min_level=_levels(calibration_results.min_level),
        intensity_levels=_rows(calibration_results.intensity_levels),
        raw_intensity_levels=_rows(calibration_results.raw_intensity_levels),
        transfer_fits=fits,
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
            channel_width_px=width,
            gap_px=gap,
        ),
        float(center),
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
    *,
    fit_transfer: bool = True,
) -> Path:
    """Write a full CalibrationResult to JSON so a later step can resume from it.

    Every field is stored (arrays as nested lists), so loading the file rebuilds
    an equivalent CalibrationResult.

    ``fit_transfer`` (default on) fits each channel's measured curve to the
    4-parameter sin^2 transfer model and stores the coefficients under
    ``transfer_fits``.  That is what
    :func:`slm_module.encoding.channel_layout_from_calibration` inverts to turn
    a requested value into a grayscale level, so fitting HERE -- once, at write
    time -- freezes the encoding with the calibration.  Steps 6, 7 and 8 run at
    different moments against the same JSON and must agree on what "x = 0.5"
    means; re-deriving the fit on every load would let an optimizer path or a
    scipy version quietly move it between them.  The raw ``intensity_levels``
    stay in the file alongside the fits, so a curve can always be refitted.

    A channel whose curve will not fit is reported and ``transfer_fits`` is
    stored as null -- the measurement is never lost to a fit failure, and the
    raw sweep is still in the file.  The encoder then fails loudly on load,
    which is the right moment for it: before a run rather than after one.  Pass
    ``fit_transfer=False`` to skip the fit entirely.
    """

    fits = calibration_results.transfer_fits
    if fit_transfer and fits is None and calibration_results.intensity_levels is not None:
        try:
            fits = fit_transfer_curves(
                calibration_results.level_range, calibration_results.intensity_levels
            )
        except TransferFitError as exc:
            print(f"  [calibration] transfer fit failed, storing curves only: {exc}")
            fits = None

    payload = {
        "schema": "calibration_result_v2",
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
        "transfer_fits": None if fits is None else [f.to_dict() for f in fits],
        # The window/pad split, so the encoder drives the aperture that was
        # actually calibrated instead of inferring one from the pitch.
        "channel_width_px": calibration_results.channel_width_px,
        "gap_px": calibration_results.gap_px,
    }
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    return out


# ======================================================================
# calib.json sections  (steps 1-3)
# ======================================================================
#
# One directory, one JSON: each of these returns what its step contributes, and
# the raw sweep stays in the sibling CSV.  What a section keeps is what cannot
# be re-derived from that CSV without re-running an optimizer -- the fitted
# transfer curves above all, because re-deriving them per load would let a scipy
# version move the encoding between steps 6, 7 and 8.

def step1_section(result: CalibrationResult) -> dict:
    """Step 1: the off/on grayscale levels, and the sweep that found them."""
    return {
        "min_level": _to_jsonable(result.min_level),
        "max_level": _to_jsonable(result.max_level),
        "level_range": _to_jsonable(result.level_range),
    }


def step2_section(result: CalibrationResult) -> dict:
    """Step 2: the px -> nm fit, with the map itself in the sibling CSV.

    The 1000-odd measured coordinates were the bulk of the old step-2 JSON and
    are a two-column table, so they live in ``step2_*.csv`` now -- written by
    :func:`write_wavelength_map_csv` and read back by
    :func:`load_wavelength_map_csv`, which steps 3 already accepted as a hand
    CSV.  What stays here is what that CSV cannot carry: the polynomial and the
    off/on levels.
    """
    return {
        "min_level": _to_jsonable(result.min_level),
        "max_level": _to_jsonable(result.max_level),
        "wavelength_fit_coefficients": _to_jsonable(
            result.wavelength_fit_coefficients),
    }


def step3_section(result: CalibrationResult, *, fit_transfer: bool = True) -> dict:
    """Step 3: the channel grid and its FITTED transfer curves, no raw sweep.

    ``intensity_levels`` / ``raw_intensity_levels`` are exactly the columns of
    ``step3_*.csv``, so they are not duplicated here.  What is left is the
    channel layout -- a wavelength and coordinate per channel, the window/pad
    split it was swept with, the levels swept -- plus one
    :class:`~calibration_module.fit.transfer.TransferFit` per channel, which IS
    the encoding from here on: ``channel_layout_from_calibration`` inverts these
    and nothing re-fits them.

    A channel whose curve will not fit is fatal here, unlike in
    :func:`save_calibration_result`, which could fall back on storing the curves.
    There is no fallback left: without a fit that channel cannot be encoded at
    all, and the right moment to say so is now -- the sweep is already safe in
    the CSV -- rather than at the start of a step-6 run.
    """
    fits = result.transfer_fits
    if fit_transfer and fits is None and result.intensity_levels is not None:
        fits = fit_transfer_curves(result.level_range, result.intensity_levels)
    if fits is None:
        raise TransferFitError(
            "step 3 has no transfer fits and none could be made -- the section "
            "would carry no encoding at all.  The sweep is in the CSV; re-fit it "
            "or re-measure the channels that failed"
        )
    return {
        "wavelength": _to_jsonable(result.wavelength),
        "coordinates": _to_jsonable(result.coordinates),
        "min_level": _to_jsonable(result.min_level),
        "max_level": _to_jsonable(result.max_level),
        "level_range": _to_jsonable(result.level_range),
        "wavelength_fit_coefficients": _to_jsonable(
            result.wavelength_fit_coefficients),
        "transfer_fits": [f.to_dict() for f in fits],
        # The window/pad split, so the encoder drives the aperture that was
        # actually calibrated instead of inferring one from the pitch.
        "channel_width_px": result.channel_width_px,
        "gap_px": result.gap_px,
    }


def write_wavelength_map_csv(result: CalibrationResult, path: str | Path) -> Path:
    """Step 2's px -> nm map as the two-column CSV its section points at."""
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    coordinates = np.asarray(result.coordinates, dtype=float).reshape(-1)
    wavelengths = np.asarray(result.wavelength, dtype=float).reshape(-1)
    with open(out, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["coordinate_px", "wavelength_nm"])
        for coordinate, wavelength in zip(coordinates, wavelengths):
            writer.writerow([repr(float(coordinate)), repr(float(wavelength))])
    return out


def calibration_result_from_dict(payload: dict) -> CalibrationResult:
    """Rebuild a CalibrationResult from a save_calibration_result payload dict.

    Accepts the parsed JSON either straight from a calibration file or embedded
    in another output (e.g. the step-6 combined result stores the raw step-3
    payload under its ``"step3"`` key).

    ``transfer_fits`` is absent from files written before schema v2; it stays
    None then, and the encoder fits the stored curves at load time instead.
    """

    raw_fits = payload.get("transfer_fits")
    fits = (
        [TransferFit.from_dict(entry) for entry in raw_fits]
        if isinstance(raw_fits, list) and raw_fits
        else None
    )
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
        transfer_fits=fits,
        channel_width_px=_int_or_none(payload.get("channel_width_px")),
        gap_px=_int_or_none(payload.get("gap_px")),
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


def _int_or_none(value: Any) -> int | None:
    """A stored pixel count, or None when the file predates it / has no grid."""
    if value is None:
        return None
    return int(value)


def _scalar_level(value: Any) -> int | np.ndarray:
    if isinstance(value, list):
        return np.asarray(value, dtype=int)
    return int(value)


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
