"""The fitted sin^2 transfer curve -- the encoder's model of one channel.

A grayscale level ``L`` sets an LC retardance ``Delta(L)``, and an amplitude
modulator between polarizers passes::

    I(L) = floor + contrast * sin(Delta(L) / 2)^2
    Delta(L) = phase_curv * L^2 + phase_slope * L + phase_offset

``floor`` is the extinction leak the panel cannot close, ``contrast`` the swing
above it.  ``Delta`` is quadratic because the panel's response is sublinear
(~7.8 mrad/level near extinction, ~5.6 near full scale); ``phase_curv = 0`` is
the old 4-parameter linear model, and what pre-curvature files load back as.

In:  one channel's swept levels and normalized powers -- a row of
     ``CalibrationResult.intensity_levels``.
Out: a :class:`TransferFit`: the five parameters, residual diagnostics, and
     :meth:`TransferFit.level_for`, which is how the encoder picks a level.

Conventions.  ``Delta`` is normalized to increase across the sweep with
``Delta = 0`` at the darkest swept level, so the encoding window is always
``Delta in [0, pi]`` and the stored parameters are self-contained.
``v -> Delta = 2 asin(sqrt(v))`` is the DEFINITION of a commanded ``v`` -- what
:func:`calibration_module.fit.phase.phi_half` assumes -- and the curvature does
not touch it, living only in ``Delta -> L``.

The fit is robust twice over: a soft-L1 loss, then a clip-and-refit of what the
loss only down-weighted.  A latched DAQ cell lands tens of sigma out and will
otherwise take the whole channel; ``n_clipped`` records what was dropped.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

#: Panel grayscale is 10-bit over DVI; see :mod:`slm_module.controller`.
MAX_LEVEL = 1023

#: The curvature term is kept only if it moves some ``level_for(v)`` by at
#: least this many grayscale levels.  rms alone cannot judge it -- a fifth
#: parameter always shaves the optimizer's convergence floor -- but "does it
#: change a level the panel can be written to" is the physical question, and it
#: separates the cases by four orders of magnitude.
_CURV_MIN_LEVEL_SHIFT = 0.5

#: Default outlier threshold, in robust sigmas of the fit residual.  Honest
#: points stay inside ~3.7 sigma while a latched cell sits at 28, so 6 is in the
#: gap -- close enough to the data that one sweep endpoint is a marginal drop,
#: which is preferred to leaving room for a spike to hide behind.  ``clip=0``
#: fits every point; a larger value removes only the unambiguous ones.
_CLIP_SIGMA = 6.0

#: The residual scale is floored at this fraction of the fitted contrast before
#: anything is called an outlier: on a curve the model fits almost exactly, the
#: MAD collapses to the optimizer's convergence floor and a pure ratio test
#: starts eating good data.  Real sweeps sit far above it, so it never binds on
#: measured data.
_CLIP_MIN_SIGMA_FRAC = 1e-3

#: Give up on clipping if it wants more than this fraction of the sweep: a few
#: bad cells are an artifact, a fifth of the curve is a channel that is not
#: sin^2, and deleting the disagreement would hide it.  A bound, not an observed
#: path -- points wild enough to capture the fit have small residuals, and what
#: catches those is :attr:`TransferFit.extrapolated`/``clipped``.
_CLIP_MAX_FRACTION = 0.2

#: Clip/refit passes.  Spikes inflate the first fit's residual scale, so a
#: second pass can expose an outlier the first one hid.  One extra, in practice.
_CLIP_MAX_PASSES = 3

__all__ = [
    "MAX_LEVEL",
    "TransferFit",
    "TransferFitError",
    "fit_transfer_curve",
    "fit_transfer_curves",
]


class TransferFitError(ValueError):
    """The sin^2 transfer model could not be fitted to a channel's curve."""


@dataclass(frozen=True)
class TransferFit:
    """``I(L) = floor + contrast * sin(Delta(L)/2)^2``, ``Delta`` quadratic in L.

    The five fitted parameters, plus the diagnostics needed to judge a stored
    fit without the original curve.  ``phase_curv = 0.0`` -- the default, and
    what pre-curvature files load as -- reduces every method to the linear model.
    """

    floor: float          # normalized power at Delta = 0 (extinction leak)
    contrast: float       # peak-to-floor swing
    phase_slope: float    # LINEAR coefficient of Delta(L); see phase_rate()
    phase_offset: float   # retardance at level 0, folded onto the [0, pi] branch
    phase_curv: float = 0.0   # quadratic coefficient, rad/level^2 (0 = linear)

    # --- diagnostics (not model) ---
    level_min: int = 0    # lowest level actually swept
    level_max: int = 0    # highest level actually swept
    rms: float = 0.0      # rms residual, normalized power units
    max_abs: float = 0.0  # largest |residual|
    n_used: int = 0       # finite points the fit saw
    n_clipped: int = 0    # finite points dropped as outliers (see _CLIP_SIGMA)

    # ---------------------------------------------------------------- model

    def retardance(self, level) -> np.ndarray:
        """Delta(L) in radians."""
        lv = np.asarray(level, dtype=float)
        return self.phase_curv * lv * lv + self.phase_slope * lv + self.phase_offset

    def phase_rate(self, level) -> np.ndarray:
        """dDelta/dL at a level, rad per grayscale level.

        What "the slope" informally means; with curvature it varies across the
        sweep, which is the whole point of the term.
        """
        return 2.0 * self.phase_curv * np.asarray(level, dtype=float) + self.phase_slope

    def intensity(self, level) -> np.ndarray:
        """Predicted normalized output power at a grayscale level."""
        return self.floor + self.contrast * np.sin(self.retardance(level) / 2.0) ** 2

    @property
    def turning_level(self) -> float:
        """Level where Delta stops increasing -- ``+inf`` when purely linear.

        The parabola's vertex, and the edge of where the model means anything.
        The fitter guarantees it falls outside ``level_min..level_max``.
        """
        if self.phase_curv == 0.0:
            return math.inf
        return -self.phase_slope / (2.0 * self.phase_curv)

    def level_at(self, target: float) -> float:
        """Level where ``Delta = target``, on the increasing branch.

        That is the branch the sweep was measured on and the only one the panel
        is used over.  NaN when the target is past the vertex, i.e. unreachable.
        """
        target = float(target)
        c = self.phase_offset - target
        if self.phase_curv == 0.0:
            return math.nan if self.phase_slope == 0.0 else -c / self.phase_slope

        disc = self.phase_slope * self.phase_slope - 4.0 * self.phase_curv * c
        if disc < 0.0:
            return math.nan                      # target beyond the turning point
        root = math.sqrt(disc)
        candidates = (
            (-self.phase_slope + root) / (2.0 * self.phase_curv),
            (-self.phase_slope - root) / (2.0 * self.phase_curv),
        )
        rising = [L for L in candidates if float(self.phase_rate(L)) > 0.0]
        if not rising:
            return math.nan
        if len(rising) == 1:
            return rising[0]
        mid = 0.5 * (self.level_min + self.level_max)   # degenerate; pick the near one
        return min(rising, key=lambda L: abs(L - mid))

    @property
    def off_level_exact(self) -> float:
        """Level where Delta = 0 -- the model's extinction point, unrounded."""
        return self.level_at(0.0)

    @property
    def on_level_exact(self) -> float:
        """Level where Delta = pi -- the model's full scale, unrounded."""
        return self.level_at(math.pi)

    def _clamped(self, exact: float) -> int:
        """An exact level as a usable panel grayscale.

        An unreachable target (NaN) clamps to the turning point, the furthest
        the panel can be driven; :attr:`reaches_full_scale` is how a caller
        tells that apart from an ordinary in-range answer.
        """
        if not math.isfinite(exact):
            exact = self.turning_level
            if not math.isfinite(exact):
                return MAX_LEVEL
        return int(round(min(max(exact, 0.0), float(MAX_LEVEL))))

    @property
    def off_level(self) -> int:
        return self._clamped(self.off_level_exact)

    @property
    def on_level(self) -> int:
        return self._clamped(self.on_level_exact)

    @property
    def reaches_full_scale(self) -> bool:
        """True if ``Delta = pi`` is attainable at all.

        Only ever False with curvature: a parabola peaking below pi cannot be
        driven to ``v = 1`` at any level.
        """
        return bool(math.isfinite(self.on_level_exact))

    @property
    def turns_in_panel(self) -> bool:
        """True if ``Delta`` turns over inside the panel's 0..1023.

        Not a bad fit -- the vertex is outside the SWEPT range -- but past it
        the model is not invertible.  Encoding never goes there, staying in
        ``off_level..on_level``.
        """
        return bool(0.0 <= self.turning_level <= float(MAX_LEVEL))

    @property
    def extrapolated(self) -> bool:
        """True if the encoding window runs outside the levels actually swept.

        Not an error, but ``v = 1`` is then an extrapolation, worth knowing
        before a run rather than after; widening step 3b's ``level_range`` is
        the fix.  A full scale the model cannot reach counts as extrapolated too.
        """
        if not self.reaches_full_scale:
            return True
        lo = min(self.off_level_exact, self.on_level_exact)
        hi = max(self.off_level_exact, self.on_level_exact)
        return bool(lo < self.level_min - 0.5 or hi > self.level_max + 0.5)

    @property
    def clipped(self) -> bool:
        """True if the encoding window runs outside the panel's 0..1023."""
        if not self.reaches_full_scale:
            return True
        return bool(
            self.off_level_exact < 0.0
            or self.on_level_exact > float(MAX_LEVEL)
        )

    # -------------------------------------------------------------- encoding

    def level_for(self, val: float) -> int:
        """Grayscale level delivering normalized output ``val`` in [0, 1].

        ``val`` is the fraction of fitted contrast above the fitted floor, so
        ``sin^2(Delta/2) = val`` exactly and ``Delta = 2 asin(sqrt(val))``
        (= ``2 * phase.phi_half(val)``) -- the retardance the step-6/7/8 model
        assumes for a channel commanded at ``val``.  Inverting ``Delta(L)``
        gives the level, rounded and clamped to the panel, so it is not
        restricted to the swept levels.  This is the ONLY place the curvature
        reaches the panel.
        """
        v = float(np.clip(val, 0.0, 1.0))
        theta = 2.0 * math.asin(math.sqrt(v))          # in [0, pi]
        return self._clamped(self.level_at(theta))

    # ----------------------------------------------------------------- io

    def to_dict(self) -> dict:
        return {
            "floor": float(self.floor),
            "contrast": float(self.contrast),
            "phase_slope": float(self.phase_slope),
            "phase_offset": float(self.phase_offset),
            "phase_curv": float(self.phase_curv),
            "level_min": int(self.level_min),
            "level_max": int(self.level_max),
            "rms": float(self.rms),
            "max_abs": float(self.max_abs),
            "n_used": int(self.n_used),
            "n_clipped": int(self.n_clipped),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "TransferFit":
        """Rebuild a stored fit.

        ``phase_curv`` and ``n_clipped`` are absent from older files; both
        default to 0, which reproduces those fits exactly.
        """
        try:
            return cls(
                floor=float(payload["floor"]),
                contrast=float(payload["contrast"]),
                phase_slope=float(payload["phase_slope"]),
                phase_offset=float(payload["phase_offset"]),
                phase_curv=float(payload.get("phase_curv", 0.0)),
                level_min=int(payload.get("level_min", 0)),
                level_max=int(payload.get("level_max", 0)),
                rms=float(payload.get("rms", 0.0)),
                max_abs=float(payload.get("max_abs", 0.0)),
                n_used=int(payload.get("n_used", 0)),
                n_clipped=int(payload.get("n_clipped", 0)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise TransferFitError(f"malformed transfer fit payload: {exc}") from exc


def _model(level, floor, contrast, slope, offset):
    return floor + contrast * np.sin((slope * level + offset) / 2.0) ** 2


def _quad_model(level, floor, contrast, p0, p1, p2, mid, half):
    """The same law with a quadratic ``Delta``, on ``u = (L-mid)/half``.

    Fitting in ``u`` is not cosmetic: on raw ``L`` the three ``Delta``
    parameters differ by twelve orders of magnitude, while on ``u`` in [-1, 1]
    they are all order 1 and ``p2`` reads as the radians ``Delta`` departs from
    linear at the sweep edge.  Converted back to raw-L for storage.
    """
    u = (level - mid) / half
    return floor + contrast * np.sin((p0 + p1 * u + p2 * u * u) / 2.0) ** 2


def fit_transfer_curve(
    levels: np.ndarray,
    values: np.ndarray,
    *,
    robust: bool = True,
    curvature: bool = True,
    clip: float = _CLIP_SIGMA,
) -> TransferFit:
    """Fit one channel's measured curve to the sin^2 transfer model.

    In:  ``levels`` the swept grayscale levels, ``values`` the normalized output
         power at each (a row of ``CalibrationResult.intensity_levels``); NaNs
         are dropped.  ``robust`` picks a soft-L1 loss, ``curvature`` the fifth
         parameter, ``clip`` the outlier threshold in robust sigmas.
    Out: a :class:`TransferFit`.  Raises :class:`TransferFitError` if there is
         nothing to fit or the linear stage will not converge -- there is
         deliberately no fallback SHAPE, since a channel that is not sin^2 is
         one to investigate, not to encode against a curve with no phase meaning.

    Two stages: the 4-parameter linear-``Delta`` fit, then a refit with the
    quadratic term seeded from it at zero (a cold start on five parameters lands
    on the wrong branch of a periodic model often enough to matter).  The
    refinement is kept only if it converges, improves the rms and leaves
    ``Delta`` monotonic across the sweep, so ``phase_curv == 0.0`` always means
    "no curvature in effect" and ``curvature=False`` reproduces the original.

    ``floor >= 0`` and ``contrast >= 0`` are bounded, not free: a normalized
    power cannot be negative, and the downstream phase model has no floor term
    to absorb one.  Outliers are judged against the FINAL model, so a curved
    sweep's smooth mismatch is not read as bad points at the ends.
    """
    fit, _dropped = fit_transfer_curve_clipped(
        levels, values, robust=robust, curvature=curvature, clip=clip
    )
    return fit


def fit_transfer_curve_clipped(
    levels: np.ndarray,
    values: np.ndarray,
    *,
    robust: bool = True,
    curvature: bool = True,
    clip: float = _CLIP_SIGMA,
) -> tuple[TransferFit, np.ndarray]:
    """:func:`fit_transfer_curve`, plus WHICH points the clip dropped.

    Out: the same fit, and a boolean mask over the input arrays -- True where a
    measured point was taken out of the fit, so a plot can mark the cells the
    stored calibration was built without.

    The mask comes out of the clip itself rather than being re-derived from the
    returned fit, which cannot be done reliably: once a point is out, both its
    residual and the MAD move, so judging it against the fit that resulted
    disagrees with ``n_clipped`` on the marginal cases.  ``n_clipped`` and this
    mask are therefore always the same points.
    """
    levels = np.asarray(levels, dtype=float)
    values = np.asarray(values, dtype=float)
    if levels.shape != values.shape or levels.ndim != 1:
        raise TransferFitError("levels and values must be matching 1-D arrays")

    finite = np.isfinite(levels) & np.isfinite(values)
    if int(finite.sum()) < 5:
        raise TransferFitError(
            f"need at least 5 finite points to fit 4 parameters, got {int(finite.sum())}"
        )
    lv = levels[finite]
    val = values[finite]

    dropped = np.zeros(levels.shape, dtype=bool)
    fit = _fit_points(lv, val, robust=robust, curvature=curvature)
    if clip <= 0.0:
        return fit, dropped
    fit, keep = _clip_and_refit(
        lv, val, fit, robust=robust, curvature=curvature, clip=float(clip)
    )
    dropped[np.flatnonzero(finite)[~keep]] = True
    return fit, dropped


def _fit_points(
    lv: np.ndarray,
    val: np.ndarray,
    *,
    robust: bool,
    curvature: bool,
) -> TransferFit:
    """Both fit stages on an already-cleaned point set; no outlier handling.

    Split out so the clip loop can refit the same model on a subset, and so
    ``n_used`` and the swept range always describe the points the fit saw.
    """
    span = float(np.ptp(lv))
    swing = float(np.ptp(val))
    if span <= 0.0 or swing <= 0.0:
        raise TransferFitError("curve is flat in level or in value; nothing to fit")

    # seed: the darkest swept level is Delta ~ 0, the brightest Delta ~ pi
    lo = float(lv[int(np.argmin(val))])
    hi = float(lv[int(np.argmax(val))])
    slope0 = math.pi / (hi - lo) if hi != lo else math.pi / span
    seed = (max(float(np.min(val)), 0.0), swing, slope0, -slope0 * lo)
    bounds = (
        (0.0, 0.0, -np.inf, -np.inf),
        (max(float(np.max(val)), 1e-9), 5.0 * swing, np.inf, np.inf),
    )

    try:
        from scipy.optimize import curve_fit

        kwargs = {}
        if robust:
            # f_scale ~ the noise we expect; residuals past it are down-weighted
            kwargs = {"loss": "soft_l1", "f_scale": max(1e-4, 0.05 * swing)}
        popt, _ = curve_fit(
            _model, lv, val, p0=seed, bounds=bounds, method="trf",
            maxfev=40000, **kwargs,
        )
    except TransferFitError:
        raise
    except Exception as exc:                       # convergence, import, ...
        raise TransferFitError(f"sin^2 transfer fit did not converge: {exc}") from exc

    floor, contrast, slope, offset = (float(v) for v in popt)
    if not all(math.isfinite(p) for p in (floor, contrast, slope, offset)):
        raise TransferFitError("sin^2 transfer fit returned non-finite parameters")
    if slope == 0.0 or contrast <= 0.0:
        raise TransferFitError(
            f"degenerate transfer fit (slope {slope:.3g}, contrast {contrast:.3g})"
        )

    # Branch: the model is invariant under (slope, offset) -> (-slope, -offset),
    # so take slope > 0, then fold offset by whole periods so Delta = 0 sits at
    # the darkest swept level.
    if slope < 0.0:
        slope, offset = -slope, -offset
    k = round((slope * lo + offset) / (2.0 * math.pi))
    offset -= 2.0 * math.pi * k

    resid = val - _model(lv, floor, contrast, slope, offset)
    linear = TransferFit(
        floor=floor,
        contrast=contrast,
        phase_slope=slope,
        phase_offset=offset,
        level_min=int(round(float(np.min(lv)))),
        level_max=int(round(float(np.max(lv)))),
        rms=float(np.sqrt(np.mean(resid ** 2))),
        max_abs=float(np.max(np.abs(resid))),
        n_used=int(lv.size),
    )

    if not curvature or lv.size < 6:
        return linear
    return _refine_curvature(lv, val, linear, robust=robust) or linear


def _clip_and_refit(
    lv: np.ndarray,
    val: np.ndarray,
    fit: TransferFit,
    *,
    robust: bool,
    curvature: bool,
    clip: float,
) -> tuple[TransferFit, np.ndarray]:
    """Drop points past ``clip`` robust sigmas and refit, until nothing is found.

    Out: the fit and the mask of points it kept, so a caller that wants to show
    the dropped cells does not have to guess at them (see
    :func:`fit_transfer_curve_clipped`).

    Each pass measures the residual scale with a MAD, so the points being judged
    cannot inflate their own threshold the way an rms would, floored at
    :data:`_CLIP_MIN_SIGMA_FRAC` of contrast.  Every accepted pass refits from
    scratch on the survivors, so the returned fit describes exactly the points
    it was built from.

    Bails out keeping the last accepted fit when the clip wants more than
    :data:`_CLIP_MAX_FRACTION` of the sweep, when too few points would survive,
    or when the refit fails: those all mean "not a curve with a few bad cells",
    and a loud rms is the honest answer.

    Limit: this only removes outliers the fit disagrees with.  Points numerous
    or coherent enough to capture the fit have small residuals and no threshold
    finds them; the absurd ``contrast`` and window they leave behind is what
    catches that.
    """
    keep = np.ones(lv.size, dtype=bool)
    floor = max(1, int(_CLIP_MAX_FRACTION * lv.size))

    for _ in range(_CLIP_MAX_PASSES):
        resid = val[keep] - fit.intensity(lv[keep])
        sigma = 1.4826 * float(np.median(np.abs(resid - np.median(resid))))
        sigma = max(sigma, _CLIP_MIN_SIGMA_FRAC * fit.contrast)
        if not (sigma > 0.0):
            break                                  # flat and exact; nothing to judge

        bad = np.abs(resid) > clip * sigma
        if not bad.any():
            break                                  # converged

        candidate = keep.copy()
        candidate[np.flatnonzero(keep)[bad]] = False
        if int(lv.size - candidate.sum()) > floor or int(candidate.sum()) < 5:
            break                                  # too much of the curve; keep it loud
        try:
            refit = _fit_points(
                lv[candidate], val[candidate], robust=robust, curvature=curvature
            )
        except TransferFitError:
            break                                  # the subset will not fit; keep the fit we have
        keep, fit = candidate, refit

    n_clipped = int(lv.size - keep.sum())
    return (fit if n_clipped == 0 else replace(fit, n_clipped=n_clipped)), keep


def _refine_curvature(
    lv: np.ndarray,
    val: np.ndarray,
    linear: TransferFit,
    *,
    robust: bool,
) -> TransferFit | None:
    """Refit with a quadratic ``Delta``, or None if that does not earn its keep.

    None rather than raising: a curvature term that will not converge, does not
    improve the rms, or turns ``Delta`` over inside the sweep is a refinement
    that failed, not a calibration that failed.
    """
    mid = 0.5 * (float(np.max(lv)) + float(np.min(lv)))
    half = 0.5 * (float(np.max(lv)) - float(np.min(lv)))
    if half <= 0.0:
        return None

    swing = float(np.ptp(val))
    seed = (
        linear.floor,
        linear.contrast,
        linear.phase_slope * mid + linear.phase_offset,   # Delta at u = 0
        linear.phase_slope * half,                        # dDelta per half-range
        0.0,                                              # start from linear
    )
    bounds = (
        (0.0, 0.0, -np.inf, -np.inf, -np.inf),
        (max(float(np.max(val)), 1e-9), 5.0 * swing, np.inf, np.inf, np.inf),
    )

    try:
        from scipy.optimize import curve_fit

        kwargs = {}
        if robust:
            kwargs = {"loss": "soft_l1", "f_scale": max(1e-4, 0.05 * swing)}
        popt, _ = curve_fit(
            lambda L, f, c, q0, q1, q2: _quad_model(L, f, c, q0, q1, q2, mid, half),
            lv, val, p0=seed, bounds=bounds, method="trf", maxfev=60000, **kwargs,
        )
    except Exception:                              # convergence, import, ...
        return None

    floor, contrast, p0, p1, p2 = (float(v) for v in popt)
    if not all(math.isfinite(p) for p in (floor, contrast, p0, p1, p2)):
        return None
    if contrast <= 0.0 or p1 == 0.0:
        return None

    # Branch, in u-space: negate to make Delta increase, then fold whole
    # periods so Delta = 0 sits at the darkest swept level, as above.
    if p1 < 0.0:
        p0, p1, p2 = -p0, -p1, -p2
    if p2 != 0.0 and abs(-p1 / (2.0 * p2)) <= 1.0:
        return None                                # turns over inside the sweep
    u_lo = (float(lv[int(np.argmin(val))]) - mid) / half
    p0 -= 2.0 * math.pi * round((p0 + p1 * u_lo + p2 * u_lo * u_lo) / (2.0 * math.pi))

    resid = val - _quad_model(lv, floor, contrast, p0, p1, p2, mid, half)
    rms = float(np.sqrt(np.mean(resid ** 2)))
    if not (rms < linear.rms):
        return None                                # no better than linear

    # u-space -> raw-L coefficients, expanding p2*((L-mid)/half)^2 + ...
    quad = TransferFit(
        floor=floor,
        contrast=contrast,
        phase_slope=p1 / half - 2.0 * p2 * mid / (half * half),
        phase_offset=p0 - p1 * mid / half + p2 * mid * mid / (half * half),
        phase_curv=p2 / (half * half),
        level_min=linear.level_min,
        level_max=linear.level_max,
        rms=rms,
        max_abs=float(np.max(np.abs(resid))),
        n_used=int(lv.size),
    )
    if not quad.reaches_full_scale:
        return None
    if _max_level_shift(linear, quad) < _CURV_MIN_LEVEL_SHIFT:
        return None                                # too small to write down
    return quad


def _max_level_shift(a: TransferFit, b: TransferFit) -> float:
    """Largest unrounded ``level_for`` disagreement between two fits, v in [0, 1]."""
    theta = 2.0 * np.arcsin(np.sqrt(np.linspace(0.0, 1.0, 101)))
    diff = [abs(b.level_at(t) - a.level_at(t)) for t in theta]
    finite = [d for d in diff if math.isfinite(d)]
    return max(finite) if finite else math.inf


def fit_transfer_curves(
    levels: np.ndarray,
    curves: np.ndarray,
    *,
    robust: bool = True,
    curvature: bool = True,
    clip: float = _CLIP_SIGMA,
) -> list[TransferFit]:
    """Fit every row of an ``intensity_levels`` array; one TransferFit per channel.

    In:  ``levels``, and a 2-D (channel, level) array of normalized powers.
    Out: one fit per row, each clipped on its own residuals so one channel's
         bad cells never affect another's threshold (see ``n_clipped``).
         Raises :class:`TransferFitError` naming the row that failed.
    """
    curves = np.asarray(curves, dtype=float)
    if curves.ndim != 2:
        raise TransferFitError("curves must be a 2-D (channel, level) array")
    out: list[TransferFit] = []
    for row in range(curves.shape[0]):
        try:
            out.append(
                fit_transfer_curve(
                    levels, curves[row], robust=robust, curvature=curvature,
                    clip=clip,
                )
            )
        except TransferFitError as exc:
            raise TransferFitError(f"channel row {row}: {exc}") from exc
    return out
