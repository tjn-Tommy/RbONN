"""The per-point sigma floor -- one number, one owner.

Fits weight by ``1/sigma^2`` and the DAQ's trace spread is shot-like, so
near-zero points would own the fit (355:1 bright-to-dark across one step-6
sweep).  The floor is added in quadrature::

    sigma^2 = trace_std^2 + floor^2

In:  a sweep's values and trace stds, volts.
Out: ``(floor_volts, FloorSource)`` from :func:`measured_floor`; the weighted
     sigma from :func:`floor_std`.  Steps 6/7 record the source they used.

A floor cannot rescue a clipped point -- that is the input range's job.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

__all__ = ["STD_FLOOR_V", "FloorSource", "floor_std", "measured_floor"]


#: FALLBACK per-point sigma floor, volts, for a sweep that cannot measure its
#: own; 0.0 disables the floor.  Prefer :func:`measured_floor`: a floor frozen
#: in volts cannot survive a gain change, and this one is 8x too big at the
#: 0930 detection gain.
STD_FLOOR_V = 2.0e-4

#: A sweep needs at least this many near-zero acquisitions before their median
#: is a floor rather than an anecdote.  Step 6's grid gives 2 dark repeats per
#: pair, so a ten-pair sweep clears it.
MIN_FLOOR_POINTS = 6


class FloorSource(str, Enum):
    """Where a fit's sigma floor came from.

    Stored as ``sigma_floor_source`` in the step-6/step-7 sections, so an
    ``eta_err`` from two nights can be compared knowing which.
    """

    #: Median trace std of this sweep's own all-off acquisitions.
    MEASURED = "measured"
    #: From the step-6 section of the same run -- step 7 has no dark of its own.
    INHERITED = "inherited_step6"
    #: :data:`STD_FLOOR_V` -- no near-zero acquisition to measure from.
    FALLBACK = "fallback_const"


def measured_floor(
    values,
    stds,
    *,
    quiet_frac: float = 0.06,
    min_points: int = MIN_FLOOR_POINTS,
) -> tuple[float, FloorSource]:
    """This sweep's own sigma floor: the median trace std where there is no light.

    In:  ``values``/``stds`` from the sweep, volts.  ``quiet_frac`` is the
         fraction taken as quiet, lowest by value; pass 1.0 when the caller
         already selected them by commanded drive (step 6 does, on
         ``x == w == 0``, which is the physical all-off however much stray
         light sits on it).
    Out: ``(floor_volts, source)``.  Fewer than ``min_points`` quiet readings
         returns ``(STD_FLOOR_V, FALLBACK)`` -- check ``source``, not just the
         number.

    Median of the quietest points, not a fitted intercept: a sweep that never
    reaches zero sends all its variance into the slope and reports
    ``sigma0 = 0``.  Not the minimum either -- trace std only spikes upward, so
    the minimum of a few acquisitions under-floors the fit it protects.
    """
    y = np.asarray(values, dtype=float).ravel()
    s = np.asarray(stds, dtype=float).ravel()
    if y.size != s.size:
        raise ValueError("values and stds must be the same length")
    ok = np.isfinite(y) & np.isfinite(s) & (s > 0.0)
    y, s = y[ok], s[ok]
    if y.size < min_points:
        return STD_FLOOR_V, FloorSource.FALLBACK

    quiet = y <= np.quantile(y, float(quiet_frac))
    if int(quiet.sum()) < min_points:
        return STD_FLOOR_V, FloorSource.FALLBACK

    floor = float(np.median(s[quiet]))
    if not (floor > 0.0) or not np.isfinite(floor):
        return STD_FLOOR_V, FloorSource.FALLBACK
    return floor, FloorSource.MEASURED


def floor_std(std, floor: float | None = None) -> np.ndarray:
    """``hypot(std, floor)`` -- the sigma a fit should actually weight by.

    In:  ``std`` of any shape; ``floor`` this sweep's own, from
         :func:`measured_floor`.  Omitting it falls back to
         :data:`STD_FLOOR_V`.
    Out: the same shape, so it drops in wherever the raw std was used (a scalar
         gives a 0-d array, which floats fine).
    """
    f = STD_FLOOR_V if floor is None else float(floor)
    return np.hypot(np.asarray(std, dtype=float), f)
