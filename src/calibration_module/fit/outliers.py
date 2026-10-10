"""Post-sweep outlier detection for calibration measurements.

In:  measured values, and for the burst guards their trace stds.
Out: boolean masks.  Pure numerics, Qt-free and hardware-free; the re-display
     and re-measure loops belong to
     :mod:`calibration_module.measure.slm_response`.

Two unrelated jobs live here.  The residual guards fit a sweep against the
step's theoretical model -- the linear px -> nm map for step 2, the sin^2
transfer curve for step 3 -- and flag points beyond ``k_sigma`` robust sigmas
(MAD-based, so the outliers cannot inflate the threshold they are judged
against) for re-measurement, up to ``max_retries`` rounds per sweep.  The burst
guards judge a single acquisition's TRACE instead, for steps 6 to 8, where a
transient landing inside the read window is the recurring bench fault.
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

    Non-finite residuals -- a NaN centroid from a dead trace -- are always
    flagged, being outliers by definition.

    Residuals are CENTERED on their median first: a non-robust fit dragged by an
    outlier gives the clean points a common offset, and raw |r| against the tiny
    MAD would flag all of them.  ``rel_floor_scale`` (typically the data's max
    magnitude) floors the sigma at one ppm of it, since on numerically clean
    data the MAD collapses toward zero and would flag harmless fit noise
    forever.
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
    """Residuals of a robust degree-1 fit y ~ a*x + b (step-2 map shape).

    Theil-Sen (median of pairwise slopes), so an outlier cannot drag the
    reference line the way it would under least squares, where it shifts every
    clean point's residual and the flagging degenerates.  Falls back to
    ``np.polyfit`` without scipy.

    Fitted on the finite subset only; a non-finite ``y`` keeps a NaN residual so
    :func:`flag_by_residual` picks it up.
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
    """Residuals of the sin^2 SLM transfer model (step-3 curve shape).

    Fits ``I = i0 * sin((a*level + b)/2)^2`` by nonlinear least squares, falling
    back to a cubic polynomial if that will not converge -- the point is a smooth
    reference to flag spikes against, not a physics-grade fit -- so this never
    raises for finite input.
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

#: Hard ceiling on a step-6/step-7 READ, volts.  Anything above it is a bench
#: fault, dropped without appeal.
#:
#: Both guards above judge an acquisition by its TRACE SPREAD, which is the
#: right test for a transient that lands inside one acquisition window but blind
#: to two things.  A read can be far too high with a perfectly calm trace -- the
#: rail latched before the window opened, so there is nothing within it to
#: spread.  And a quantity recorded ONCE, with no repeats and no sweep median to
#: sit against, is judged by neither: run_0930_noYsat's step-7 all-off dark for
#: pair 4 read +345.7 mV where its eight neighbours read -60 uV, was subtracted
#: from all ten of that pair's drive rows, and put the fit at R^2 = -11438 with
#: ten good drive rows underneath it.  An absolute ceiling catches both, because
#: it needs no reference beyond itself.
#:
#: 300 mV against what steps 6 and 7 actually read: the 0930 13:25 sweep peaked
#: at 163 mV and the 20:41 sweep at 58 mV, so this is ~1.8x headroom over the
#: HIGHER of the two gains on disk.  It is a bench-fault ceiling, not a
#: saturation limit -- raise the detection gain much past 13:25's and it will
#: start rejecting honest reads, which is why it says so when it fires.
BURST_MAX_V = 0.300


def flag_over_range(values, *, limit: float = BURST_MAX_V) -> np.ndarray:
    """Boolean mask of reads at or above ``limit`` volts -- a bench fault, flatly.

    Orthogonal to the trace-spread guards and deliberately dumb: no median, no
    neighbours, no repeats.  That is the point -- the acquisitions this is for
    have no reference to be judged against (see :data:`BURST_MAX_V`).  A
    non-finite entry is flagged too; a NaN read is not a read.
    """
    v = np.asarray(values, dtype=float)
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("limit must be a positive voltage")
    return ~np.isfinite(v) | (np.abs(v) >= float(limit))


def flag_burst_traces(
    trace_stds,
    *,
    k: float = BURST_K,
    min_points: int = 4,
) -> np.ndarray:
    """Boolean mask of acquisitions whose trace std is a burst, not noise.

    A burst is a transient landing part-way through one acquisition window: the
    mean comes out multiples too high AND the within-trace spread explodes with
    it.

    Thresholds the ABSOLUTE trace std against ``k * median(trace_stds)``.
    Deliberately not the std/|mean| ratio, which is meaningless at a near-zero
    background level and reaches 44 on perfectly good data, while the absolute
    spread of a clean acquisition tracks the signal smoothly.

    The median rather than the MAD sets the scale: trace std is bounded below by
    detector noise and only spikes upward, so the distribution has no left tail
    for a robust sigma to measure, and a couple of bursts cannot drag a median.
    Non-finite entries are always flagged; below ``min_points`` acquisitions
    nothing is, the median being too soft to threshold against.
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

    In:  ``group_ids``, any hashable labels marking which acquisitions are
         repeats of one another -- for step 6, the commanded ``(x, w)`` cell.
    Out: a mask flagging any entry above ``k`` times its group's MINIMUM trace
         std.

    Within a group every entry measures the identical setting, so the group has
    no dynamic range in it at all -- unlike :func:`flag_burst_traces`, whose
    sweep median straddles dark to full scale and so reads the honest full-scale
    point as 10x the typical one.

    The minimum is the right reference because trace std only spikes upward,
    bounded below by detector noise, so the smallest repeat is the least
    disturbed.  Over the 1482 cells on disk the clean population tops out at
    1.81x while every real anomaly is >= 5.1x, so k = 3 sits in the gap.

    This catches the mild burst the sweep test cannot -- a repeat whose TRACE was
    disturbed while its mean sat right on the cell's, which nothing reading the
    mean could see.

    Degrades with no special case: a single-acquisition group is its own minimum,
    ratio 1, never flagged -- correctly, since nothing in the data can say
    whether it was disturbed.  The minimum is a downward-biased scale estimate,
    so the ratio creeps up with group size; that is measured out to the 6 repeats
    the step-6 grids use.
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
    "BURST_MAX_V",
    "flag_over_range",
    "OutlierRemeasurePolicy",
    "flag_burst_repeats",
    "flag_burst_traces",
    "flag_by_residual",
    "linear_fit_residuals",
    "mad_sigma",
    "transfer_fit_residuals",
]
