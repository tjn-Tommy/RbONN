"""Post-sweep outlier detection for calibration measurements.

A sweep is measured once, then fitted against the step's theoretical model
(linear coordinate->wavelength map for Step 2, the sin^2 transfer curve for
Step 3).  Points whose residual exceeds ``k_sigma`` robust standard deviations
(MAD-based, so the outliers themselves do not inflate the threshold) are
flagged and re-measured; the flagged original is discarded and the cell becomes
the median of its re-measurements.  The flag -> re-measure round repeats up to
``max_retries`` times per sweep.

Only the pure numerics live here (Qt-free, hardware-free); the calibration
functions in :mod:`slm_module.calibration.calibration_new` own the actual
re-display + re-measure loops.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calibration import intensity_model

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
    :func:`slm_module.calibration.calibration.intensity_model`) by nonlinear
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


__all__ = [
    "OutlierRemeasurePolicy",
    "flag_by_residual",
    "linear_fit_residuals",
    "mad_sigma",
    "transfer_fit_residuals",
]
