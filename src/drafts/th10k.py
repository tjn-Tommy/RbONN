"""Thorlabs TH10K (10 kOhm NTC) resistance -> temperature.

Split out of ``plot_osa_monitor.py`` so anything that logs a thermistor in ohms
-- the monitors, the system-accuracy run -- can report degrees without pulling
in matplotlib and the OSA analysis with it.  numpy only.

The curve is the spec sheet's (4813-S01)::

    T[K] = 1 / (a + b*x + c*x^2 + d*x^3),   x = ln(R / 10 kOhm)

with a different coefficient set per resistance decade.

Good to a few tens of mK *relative*; the absolute value still carries the
sensor's +-1 C accuracy at 25 C plus whatever offset sits between the bead and
the thing you care about.  Read the changes, not the absolute number.

This is the TH10K curve and is **wrong for any other thermistor**.
"""
from __future__ import annotations

import numpy as np

R25_OHM = 10_000.0
TH10K_COEFFS = (   # (R_max, R_min, a, b, c, d)
    (692_600.0, 32_770.0, 0.003357042, 0.000252143, 3.37742e-06, -6.54336e-08),
    (32_770.0,   3_599.0, 0.003354016, 0.000256173, 2.13941e-06, -7.25325e-08),
    (3_599.0,      681.6, 0.003353045, 0.0002542,   1.14261e-06, -6.93803e-08),
    (681.6,        187.0, 0.003353609, 0.000253768, 8.53411e-07, -8.79629e-08),
)


def th10k_celsius(ohm) -> np.ndarray:
    """Thorlabs TH10K resistance (Ohm) -> temperature (C).

    Uses the spec-sheet cubic in ``ln(R/R25)``, picking the coefficient set for
    each reading's resistance decade.  Readings outside the tabulated span
    (692.6 kOhm ... 187 Ohm) come back as NaN rather than extrapolated.
    """
    r = np.asarray(ohm, dtype=float)
    out = np.full(r.shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.log(r / R25_OHM)
    last = len(TH10K_COEFFS) - 1
    for k, (r_hi, r_lo, a, b, c, d) in enumerate(TH10K_COEFFS):
        # The spec sheet's range bounds are rounded (its own R-T table lists
        # 692650 Ohm and 187 Ohm at the extremes), so widen the outer edges.
        hi = r_hi * 1.001 if k == 0 else r_hi
        sel = np.isfinite(r) & (r <= hi) & (r >= r_lo * 0.999 if k == last else r > r_lo)
        if sel.any():
            xs = x[sel]
            out[sel] = 1.0 / (a + b * xs + c * xs**2 + d * xs**3) - 273.15
    return out


def th10k_celsius_one(ohm: float) -> float:
    """Scalar convenience wrapper: one resistance -> one temperature (NaN off-curve)."""
    return float(th10k_celsius(np.asarray([float(ohm)]))[0])


__all__ = ["R25_OHM", "TH10K_COEFFS", "th10k_celsius", "th10k_celsius_one"]
