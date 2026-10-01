"""Post-sweep outlier detection for calibration measurements.

A sweep is measured once, then fitted against the step's theoretical model
(linear coordinate->wavelength map for Step 2, the sin^2 transfer curve for
Step 3).  Points whose residual exceeds ``k_sigma`` robust standard deviations
(MAD-based, so the outliers themselves do not inflate the threshold) are
flagged and re-measured; the flagged original is discarded and the cell becomes
the median of its re-measurements.  The flag -> re-measure round repeats up to
``max_retries`` times per sweep.

Only the pure numerics live here (Qt-free, hardware-free); the calibration
sweeps in :mod:`calibration_module.measure.slm_response` own the actual
re-display + re-measure loops.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .legacy_intensity import intensity_model

_SIGMA_FLOOR = 1e-15


@dataclass(frozen=True)
class OutlierRemeasurePolicy:
    """How aggressively to flag and re-measure sweep outliers.

    k_sigma:     residual threshold in robust-sigma units (MAD * 1.4826).
    max_retries: maximum flag -> re-measure rounds per sweep.
    min_points:  below this many samples the model fit is too unconstrained
                 to trust, so flagging is skipped entirely.
    """

    k_sigma: float = 4.0
    max_retries: int = 3
    min_points: int = 8

    def __post_init__(self) -> None:
        if not np.isfinite(self.k_sigma) or self.k_sigma <= 0.0:
            raise ValueError("k_sigma must be positive")
        if self.max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        if self.min_points < 3:
            raise ValueError("min_points must be at least 3")


def mad_sigma(residuals: np.ndarray) -> float:
    """Robust sigma estimate: 1.4826 * median(|r - median(r)|), floored.

    Non-finite residuals are ignored; an empty / all-NaN input returns the
    floor so a caller dividing by the result never explodes.
    """
    r = np.asarray(residuals, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return _SIGMA_FLOOR
    sigma = 1.4826 * float(np.median(np.abs(r - np.median(r))))
    return max(sigma, _SIGMA_FLOOR)


def flag_by_residual(
    residuals: np.ndarray,
    *,
    k_sigma: float,
    sigma: float | None = None,
    rel_floor_scale: float | None = None,
) -> np.ndarray:
    """Boolean mask of residuals beyond ``k_sigma`` robust sigmas.

    Non-finite residuals (e.g. a NaN centroid from a dead trace) are always
    flagged -- they are outliers by definition.

    Residuals are centered on their median before thresholding: a non-robust
    fit dragged by an outlier gives the clean points a common residual offset,
    and comparing raw |r| against the (tiny) MAD would flag all of them.

    ``rel_floor_scale`` (typically the data's max magnitude) floors the sigma
    at ``1e-6 * |rel_floor_scale|``: on numerically clean data the MAD collapses
    toward zero and would otherwise flag harmless fit noise forever. A one-ppm
    floor is far below any physically meaningful deviation.
    """
    r = np.asarray(residuals, dtype=float)
    finite = r[np.isfinite(r)]
    centered = r - (np.median(finite) if finite.size else 0.0)
    floor = _SIGMA_FLOOR
    if rel_floor_scale is not None and np.isfinite(rel_floor_scale):
        floor = max(floor, 1e-6 * abs(float(rel_floor_scale)))
    if sigma is None:
        scale = max(mad_sigma(centered), floor)
    else:
        scale = max(float(sigma), floor)
    return ~np.isfinite(r) | (np.abs(centered) > float(k_sigma) * scale)


def linear_fit_residuals(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Residuals of a robust degree-1 fit y ~ a*x + b (Step-2 map shape).

    Uses the Theil-Sen estimator (median of pairwise slopes) so an outlier
    cannot drag the reference line -- with plain least squares the outlier
    shifts every clean point's residual and the flagging degenerates. Falls
    back to ``np.polyfit`` if scipy is unavailable.

    The fit runs on the finite subset only; entries with non-finite ``y`` keep
    a NaN residual so :func:`flag_by_residual` picks them up.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be matching 1-D arrays")
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        return np.full(y.shape, np.nan)
    try:
        from scipy.stats import theilslopes

        slope, intercept, _, _ = theilslopes(y[finite], x[finite])
        predicted = slope * x + intercept
    except Exception:
        coeffs = np.polyfit(x[finite], y[finite], 1)
        predicted = np.polyval(coeffs, x)
    residuals = y - predicted
    residuals[~np.isfinite(y)] = np.nan
    return residuals


def transfer_fit_residuals(levels: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Residuals of the sin^2 SLM transfer model (Step-3 curve shape).

    Fits ``I = i0 * sin((a*level + b)/2)^2`` (the
    :func:`calibration_module.fit.legacy_intensity.intensity_model`) by nonlinear
    least squares.  If the fit cannot converge it falls back to a cubic
    polynomial -- the point is a smooth reference to flag spikes against, not a
    physics-grade fit -- so this never raises for finite input arrays.
    """
    levels = np.asarray(levels, dtype=float)
    values = np.asarray(values, dtype=float)
    if levels.shape != values.shape or levels.ndim != 1:
        raise ValueError("levels and values must be matching 1-D arrays")
    finite = np.isfinite(levels) & np.isfinite(values)
    if finite.sum() < 4:
        return np.full(values.shape, np.nan)
    lv = levels[finite]
    val = values[finite]

    predicted: np.ndarray | None = None
    try:
        from scipy.optimize import curve_fit

        i0_guess = float(np.max(val))
        if i0_guess <= 0.0:
            raise ValueError("all values non-positive; sin^2 fit is degenerate")
        lo = float(lv[int(np.argmin(val))])
        hi = float(lv[int(np.argmax(val))])
        span = hi - lo
        if span == 0.0:
            raise ValueError("min and max at the same level")
        slope_guess = np.pi / span            # theta sweeps ~pi from off to on
        offset_guess = -slope_guess * lo      # theta(off level) ~ 0
        # soft_l1 loss keeps a single spiked cell from dragging the reference
        # curve (and thereby smearing residuals over the clean cells)
        popt, _ = curve_fit(
            intensity_model,
            lv,
            val,
            p0=(i0_guess, slope_guess, offset_guess),
            method="trf",
            loss="soft_l1",
            f_scale=max(1e-3, 0.05 * i0_guess),
            maxfev=2000,
        )
        predicted = intensity_model(levels, *popt)
    except Exception:
        predicted = None
    if predicted is None:
        coeffs = np.polyfit(lv, val, 3)
        predicted = np.polyval(coeffs, levels)

    residuals = values - predicted
    residuals[~np.isfinite(values)] = np.nan
    return residuals


# ----------------------------------------------------------------------
# burst traces (steps 6/7)
# ----------------------------------------------------------------------

#: Trace-std multiple above the SWEEP median that counts as a burst, for a sweep
#: measured once per point (step 7).  Measured over every step-7 sweep on disk:
#: clean sweeps peak at 1.1-3.7x their own median, real bursts at 80-228x.  10
#: sits in that gap with 2.7x margin over the noisiest honest sweep.
BURST_K = 10.0

#: Trace-std multiple above the MINIMUM of a cell's own repeats (step 6).
#: Repeats are the same commanded (x, w) measured again, so this comparison is
#: free of the sweep's dark-to-full-scale dynamic range and is far tighter: over
#: the 1482 cells on disk the clean population sits at p99 = 1.6x and tops out
#: at 1.81x, while every real anomaly is >= 5.1x.  3 splits that gap.
BURST_K_REPEAT = 3.0


def flag_burst_traces(
    trace_stds,
    *,
    k: float = BURST_K,
    min_points: int = 4,
) -> np.ndarray:
    """Boolean mask of acquisitions whose trace std is a burst, not noise.

    A burst is a transient that lands part-way through one acquisition window:
    the mean comes out multiples too high AND the within-trace spread explodes
    with it.  On run_0929 step 7 one point read 0.910 V against a step-6 full
    scale of 0.122 V, with a 0.354 V trace std.

    Thresholds the **absolute** trace std against ``k * median(trace_stds)``.
    Deliberately NOT the std/|mean| ratio: at a near-zero background level that
    ratio is meaningless and reaches 44 on perfectly good data, whereas the
    absolute spread of a clean acquisition tracks the signal smoothly.

    The median (not the MAD) sets the scale here because trace std is bounded
    below by detector noise and only ever spikes upward -- the distribution has
    no left tail for a robust sigma to measure, and a couple of bursts cannot
    drag a median.  Non-finite entries are always flagged.

    Below ``min_points`` acquisitions the median is too soft to threshold
    against, so nothing is flagged.
    """
    s = np.asarray(trace_stds, dtype=float)
    if s.ndim != 1:
        raise ValueError("trace_stds must be 1-D")
    if not np.isfinite(k) or k <= 1.0:
        raise ValueError("k must be > 1")
    bad = ~np.isfinite(s)
    finite = s[~bad]
    if finite.size < min_points:
        return bad
    scale = float(np.median(finite))
    if not (scale > 0.0):
        return bad
    return bad | (s > float(k) * scale)


def flag_burst_repeats(
    trace_stds,
    group_ids,
    *,
    k: float = BURST_K_REPEAT,
) -> np.ndarray:
    """Boolean mask of burst acquisitions, judged against their own repeats.

    ``group_ids`` is any sequence of hashable labels marking which acquisitions
    are repeats of one another (for step 6, the commanded ``(x, w)`` cell).
    Within a group every entry measures the identical setting, so the group is a
    reference with no dynamic range in it at all -- unlike
    :func:`flag_burst_traces`, whose sweep median has to straddle everything from
    dark to full scale and therefore reads the honest full-scale point as 10x
    the typical one.

    Flags any entry above ``k`` times its group's MINIMUM trace std.  The minimum
    is the right reference because trace std only ever spikes upward: it is
    bounded below by detector noise, so the smallest repeat is the one least
    disturbed.  Measured over the 1482 cells on disk the clean population tops
    out at 1.81x while every real anomaly is >= 5.1x, so k = 3 sits in the gap
    with 1.7x margin.

    This catches the mild kind of burst the sweep test cannot: on run_0906 one
    repeat of ``x=1, w=0.7`` carried a 3.976 mV trace std against siblings at
    0.72-0.81 mV while its MEAN, 25.46 mV, sat right on the cell's 25.33 mV.
    Only the trace was disturbed, so nothing reading the mean could have seen it.

    Degrades gracefully with no special case: a single-acquisition group is its
    own minimum, giving a ratio of 1, and is never flagged -- correctly, since
    nothing in the data can say whether it was disturbed.  Note the minimum is a
    downward-biased scale estimate, so the ratio creeps up with group size; that
    is measured out to the 6 repeats the step-6 grids actually use.
    """
    s_arr = np.asarray(trace_stds, dtype=float)
    # kept as a plain sequence, not an array: the ids are typically (x, w)
    # tuples, which numpy would helpfully turn into a 2-D array of floats
    ids = list(group_ids)
    if s_arr.ndim != 1 or len(ids) != s_arr.size:
        raise ValueError("trace_stds and group_ids must be matching 1-D sequences")
    if not np.isfinite(k) or k <= 1.0:
        raise ValueError("k must be > 1")
    out = ~np.isfinite(s_arr)
    groups: dict[object, list[int]] = {}
    for i, g in enumerate(ids):
        groups.setdefault(g, []).append(i)
    for idx in groups.values():
        vals = s_arr[idx]
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            continue
        scale = float(np.min(finite))
        if not (scale > 0.0):
            continue
        for i in idx:
            if np.isfinite(s_arr[i]) and s_arr[i] > float(k) * scale:
                out[i] = True
    return out


__all__ = [
    "BURST_K",
    "BURST_K_REPEAT",
    "OutlierRemeasurePolicy",
    "flag_burst_repeats",
    "flag_burst_traces",
    "flag_by_residual",
    "linear_fit_residuals",
    "mad_sigma",
    "transfer_fit_residuals",
]
