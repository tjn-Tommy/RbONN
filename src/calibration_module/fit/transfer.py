"""The fitted sin^2 transfer curve -- the encoder's model of one channel.

A grayscale level ``L`` sets an LC retardance ``Delta(L)``, and an amplitude
modulator between polarizers passes::

    I(L) = floor + contrast * sin(Delta(L) / 2)^2
    Delta(L) = phase_curv * L^2 + phase_slope * L + phase_offset

Five parameters per channel.  ``floor`` is the extinction leak the panel cannot
close, ``contrast`` the swing above it, and ``Delta`` is quadratic in level
because the panel's phase response measurably is (see "Why a curvature term").

``phase_curv = 0`` recovers the original 4-parameter linear model exactly, and
that is what every fit stored before this term existed loads back as -- so the
extra parameter costs nothing in compatibility.  Note that with a curvature
term ``phase_slope`` is only the *linear coefficient*, no longer "the slope":
the local rate is ``phase_rate(L) = 2*phase_curv*L + phase_slope``.

Why fit at all
--------------
:meth:`slm_module.encoding.EncodingChannel.level_for` used to invert the
*measured* Step-3b curve point by point.  Three problems, all visible in the
0903 calibration:

1. **``v = 1`` was defined by ``argmax``.**  One noisy sample redefines full
   scale.  Between the 0828 and 0903 calibrations of the same ten channels the
   argmax peak moved by 45 levels rms -- and by 120 levels on one channel, where
   a single +6% spike at level 770 sat 9 swept levels below the real peak.  The
   fitted ``Delta = pi`` point moved 17.7 levels rms, and 10 on that channel.
2. **Every point carried the sweep's single-shot noise.**  The 4-parameter fit
   pools 36 points, so a level comes from the whole curve rather than from the
   two samples that happen to bracket the target.
3. **The delivered phase disagreed with the phase the fits assume.**
   :func:`calibration_module.fit.phase.phi_half` takes ``phi/2 = asin(sqrt(v))``,
   i.e. that the commanded ``v`` *is* ``sin^2(Delta/2)``.  Interpolating raw
   samples between an argmin and an argmax does not make that true: on the 0903
   channels the retardance actually delivered ran up to 15 deg from the assumed
   value, against the +/-1 deg step 7 quotes for ``dPhi_comb``.

Inverting this model instead makes point 3 an identity.  ``level_for(v)`` solves
``sin^2(Delta/2) = v`` for ``Delta = 2 asin(sqrt(v))`` -- which is exactly
``2 * phi_half(v)`` -- and then inverts ``Delta(L)``.  The amplitude the encoder
delivers and the phase steps 6/7/8 assume now come from one fit.

Note where the nonlinearity is and is not.  ``v -> Delta = 2 asin(sqrt(v))`` is
the *definition* of what a commanded ``v`` means, and it is untouched by any of
this; ``phi_half`` stays exactly right.  The curvature is in ``Delta(L)``, the
panel's own level-to-phase response, so it changes only which grayscale is
written to reach a given ``v``.  Steps 6/7/8 need no change -- they consume
``level_for`` and a phase model that never depended on ``Delta`` being linear.

Why a curvature term
--------------------
The 0903 Step-3b (OSA) sweeps did not support one: residuals were white --
median lag-1 autocorrelation 0.095, sign runs on the white-noise expectation
(17 vs 19) -- and their 2-3% rms was single-shot sweep noise, not model error.
This docstring said that if a finer sweep ever resolved a real nonlinearity,
this was the place to add it.  The 0907 Step-3c (DAQ) sweep did:

* the linear model's residual is a clean sinusoid, in phase on all 12 channels,
  far above the scatter about it -- structure, not noise;
* one quadratic term takes rms from 1.25% of contrast to 0.23%, a 5.5x gain,
  and every channel improves by 4.8-6.1x;
* the fitted curvature agrees across channels that were fitted independently
  and seeded at zero -- -0.21 rad at the sweep edge, spread only -0.186..-0.241,
  never changing sign.  Twelve independent fits do not agree that closely on a
  free parameter unless it is measuring something.

Concretely the panel's response is sublinear: on the 0907 channels the local
rate falls from ~7.8 mrad/level near extinction to ~5.6 near full scale.
Encoding that curve with a linear ``Delta`` over-delivers by up to +2.4% of full
scale, peaking near ``v = 0.65``, with the same sign on every channel.

A cubic is deliberately not offered.  The quadratic is what the data resolves;
past that the residual is at the sweep's noise and more parameters would fit it.

Bad cells
---------
The fit is robust twice over, because once was not enough.  A soft-L1 loss
down-weights points that disagree with the curve, and then :func:`fit_transfer_curve`
drops what is still past ``clip`` robust sigmas and refits without it.

The second stage exists because of 0907 channel 11.  Four adjacent DAQ cells
(levels 845..890) read ~25x the rest of the curve, and the soft-L1 loss alone
did not save the fit: contrast collapsed to 60% of its neighbours', the residual
came out at 824% of contrast, and the encoding window landed at 417..573 against
399..887 on the eleven channels either side.  Because ``save_calibration_result``
fits once at write time and freezes the result, that fit is what every later
step 6, 7 and 8 run against that file would have encoded channel 11 with -- a
channel driven badly wrong, from a JSON that looks entirely normal.  Dropping
the four cells recovers rms 0.18% of contrast and a window in family.

The threshold is judged on the *final* model, curvature included, against a
MAD-based sigma with a floor.  Both details matter and both are in the
constants: clipping on the linear stage would read a curved sweep's smooth
model mismatch as outliers at the ends, and an unfloored MAD collapses on a
curve the model fits almost exactly, at which point every point is thousands of
"sigma" out.  Clipping stops if it wants more than a fifth of the sweep -- that
is a channel that is not sin^2, and deleting the disagreement would hide it.
``n_clipped`` records what was dropped, so a stored fit says whether it needed
this.

Branch convention
-----------------
``sin^2`` is periodic and even, so the raw fit leaves ``Delta`` on an arbitrary
branch and sign.  :func:`fit_transfer_curve` normalizes both: ``Delta`` is made
increasing across the sweep (the model is invariant under negating every
coefficient), and ``phase_offset`` is folded by whole periods so ``Delta = 0``
lands at the darkest swept level.  The encoding window is then always ``Delta in
[0, pi]``, i.e. levels ``off_level..on_level``, and the stored parameters are
self-contained.

A quadratic ``Delta`` turns over at ``turning_level``, beyond which it is no
longer invertible.  The fitter rejects a curvature solution that turns inside
the swept range (falling back to linear), and :attr:`TransferFit.turns_in_panel`
reports the case where the vertex lands inside the panel's 0..1023 but outside
the sweep -- fine for encoding, which never leaves ``off_level..on_level``, but
worth seeing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

#: Panel grayscale is 10-bit over DVI; see :mod:`slm_module.controller`.
MAX_LEVEL = 1023

#: The curvature term is kept only if it moves some ``level_for(v)`` by at least
#: this many grayscale levels.  Judging it by rms alone does not work: on an
#: exactly linear curve the linear fit's residual is already the optimizer's
#: convergence floor (~5e-7), and a fifth parameter shaves even that, so any
#: relative-rms threshold keeps a meaningless curvature.  Asking whether the
#: term changes a level the panel can actually be written to is the physical
#: question, and it separates the two cases by four orders of magnitude: the
#: synthetic linear curves move 0.001 levels, the 0907 channels move 21.
_CURV_MIN_LEVEL_SHIFT = 0.5

#: Default outlier threshold, in robust sigmas of the fit residual.  A DAQ cell
#: that latches or gets hit by a stray reflection lands tens of sigma out, while
#: an honest point stays a few: across the twelve 0907 channels the worst clean
#: residual is 2.4 to 3.7 sigma, and the four bad cells on channel 11 sit at 28.
#: Six is inside that gap with room on both sides.
#:
#: It is not far enough inside to be a bright line.  One sweep endpoint (channel
#: 0 at level 350) sits at 6.06 and is dropped, which is a marginal call rather
#: than a clear artifact -- endpoints carry the most model mismatch, since the
#: quadratic Delta is least constrained there.  It costs one grayscale level in
#: that channel's encoding window, so the threshold is left where a spike cannot
#: hide behind it.  Pass ``clip=0`` to fit every point, or a larger value to
#: remove only the unambiguous ones.
_CLIP_SIGMA = 6.0

#: The residual scale is floored at this fraction of the fitted contrast before
#: anything is called an outlier.  On a curve the model happens to fit almost
#: exactly, the residual collapses to the optimizer's convergence floor and the
#: MAD collapses with it -- every point is then thousands of "sigma" out and a
#: pure ratio test starts eating good data.  A residual below 0.1% of contrast
#: is not an outlier at any ratio.  Real sweeps sit far above this (0907: 1.5%
#: of contrast), so on measured data the MAD always wins and this never binds.
_CLIP_MIN_SIGMA_FRAC = 1e-3

#: Give up on clipping if it wants more than this fraction of the sweep, on the
#: principle that a handful of bad cells is an artifact worth removing while a
#: fifth of the curve is a channel that is not sin^2, and deleting the
#: disagreement would turn a visible problem into a confident wrong answer.
#:
#: It is a bound, not an observed path: no curve tried so far reaches it, because
#: the failure it guards against arrives by a different route.  Once enough
#: points are wild they capture the fit itself, and a captured fit has a small
#: residual by construction -- corrupting 40% of a synthetic sweep still flags
#: zero points at 6 sigma.  Such a channel is loud in its parameters instead
#: (contrast and the encoding window come out physically absurd), which is what
#: :attr:`TransferFit.extrapolated` and :attr:`TransferFit.clipped` report.
_CLIP_MAX_FRACTION = 0.2

#: Clip/refit passes.  The first fit on a badly spiked channel is dragged by the
#: spikes, so its residual scale is inflated and a second pass can expose an
#: outlier the first one hid.  Converges in one extra pass in practice -- 0907
#: channel 11 finds all four cells on pass 0 and nothing on pass 1.
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

    The five fitted parameters, plus the diagnostics needed to judge them.  Only
    the five are model; everything else records what the fit was built from, so
    a stored fit can be sanity-checked without the original curve.

    ``phase_curv = 0.0`` -- the default, and what pre-curvature files load as --
    makes every method below reduce exactly to the original linear model.
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
        """dDelta/dL at a level, in radians per grayscale level.

        With curvature this is what "the slope" informally means; it varies
        across the sweep, which is the whole point of the term.
        """
        return 2.0 * self.phase_curv * np.asarray(level, dtype=float) + self.phase_slope

    def intensity(self, level) -> np.ndarray:
        """Predicted normalized output power at a grayscale level."""
        return self.floor + self.contrast * np.sin(self.retardance(level) / 2.0) ** 2

    @property
    def turning_level(self) -> float:
        """Level where Delta stops increasing -- ``+inf`` when purely linear.

        The vertex of the parabola.  ``Delta`` is invertible only on the branch
        the sweep sits on, so this is the edge of where the model means
        anything; the fitter guarantees it is outside ``level_min..level_max``.
        """
        if self.phase_curv == 0.0:
            return math.inf
        return -self.phase_slope / (2.0 * self.phase_curv)

    def level_at(self, target: float) -> float:
        """Level where ``Delta = target``, on the increasing branch.

        Solves ``phase_curv L^2 + phase_slope L + (phase_offset - target) = 0``
        and keeps the root where ``Delta`` is still rising, which is the branch
        the sweep was measured on and the only one the panel is used over.
        Returns NaN when the target is unreachable (past the vertex).
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

        An unreachable target (NaN -- ``Delta`` turns over before it gets there)
        clamps to the turning point, the furthest the panel can actually be
        driven.  :attr:`reaches_full_scale` is how a caller tells that apart
        from an ordinary in-range answer.
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

        Only ever False with curvature: if the parabola peaks below pi the
        channel cannot be driven to ``v = 1``, whatever level is written.
        """
        return bool(math.isfinite(self.on_level_exact))

    @property
    def turns_in_panel(self) -> bool:
        """True if ``Delta`` turns over inside the panel's 0..1023.

        The fitter keeps the vertex outside the swept range, so this is not a
        bad fit -- but past it the model is no longer invertible, so it is worth
        seeing.  Encoding never goes there: it stays in ``off_level..on_level``.
        """
        return bool(0.0 <= self.turning_level <= float(MAX_LEVEL))

    @property
    def extrapolated(self) -> bool:
        """True if the encoding window runs outside the levels actually swept.

        Not an error -- a channel whose peak sits past the top of the sweep is
        still encodable -- but its ``v = 1`` is an extrapolation, so it is worth
        knowing before a run rather than after.  Widening ``level_range`` in
        Step 3b is the fix.  A full scale the model cannot reach at all counts
        as extrapolated too.
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

        ``val`` is the fraction of this channel's fitted contrast above its
        fitted floor, so ``sin^2(Delta/2) = val`` exactly and

            Delta = 2 asin(sqrt(val)) = 2 * phase.phi_half(val)

        which is the retardance the downstream step-6/7/8 model assumes for a
        channel commanded at ``val``.  Inverting ``Delta(L)`` gives the level;
        the result is rounded to an integer grayscale and clamped to the panel's
        range, so it is not restricted to the swept levels.

        This is the ONLY place the curvature reaches the panel.  ``v -> Delta``
        above is a definition and does not change; ``Delta -> L`` is the panel's
        measured response, and that is what the quadratic term corrects.
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

        ``phase_curv`` and ``n_clipped`` are absent from files written before
        those existed; both default to 0, which reproduces those fits exactly.
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
    """The same law with a quadratic ``Delta``, parameterized on u = (L-mid)/half.

    Fitting in ``u`` rather than raw ``L`` is not cosmetic: over a 350..1000
    sweep ``L^2`` reaches 10^6, so the raw coefficient is ~10^-6 and the three
    ``Delta`` parameters differ by twelve orders of magnitude.  On ``u`` in
    [-1, 1] they are all order 1, and ``p2`` reads directly as the radians by
    which ``Delta`` departs from linear at the edge of the sweep.  The result is
    converted back to raw-L coefficients for storage.
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

    ``levels`` are the swept grayscale levels and ``values`` the normalized
    output power at each (``CalibrationResult.intensity_levels`` rows).  NaNs
    are dropped.

    Two stages.  The 4-parameter linear-``Delta`` model is fitted first, and
    then -- with ``curvature`` on, the default -- refitted with the quadratic
    ``Delta`` term seeded from it at zero.  Seeding this way matters: the
    curvature is a small correction on a periodic model, and a cold start on
    five parameters lands on the wrong branch often enough to be a problem.

    The refinement is kept only if it earns its parameter: it must converge,
    improve the rms, and leave ``Delta`` monotonic across the whole swept range
    (a parabola that turns over inside the sweep is not a transfer curve).
    Otherwise the linear fit is returned with ``phase_curv = 0.0`` -- which is
    also what happens when there are too few points for five parameters.  So
    ``phase_curv == 0.0`` exactly means "no curvature in effect", and
    ``curvature=False`` reproduces the original 4-parameter behaviour.

    Constrained, not free-running: ``floor >= 0`` because a normalized power
    cannot be negative (unconstrained, all ten 0903 channels want a slightly
    negative floor -- the reference normalization over-subtracts by ~1%), and
    ``contrast >= 0``.  Pinning the floor at zero where the fit wants to go
    below it is also what keeps ``v -> Delta`` honest for the downstream phase
    model, which has no floor term to absorb it.

    ``robust`` uses a soft-L1 loss so one spiked cell cannot drag the curve --
    the failure mode this whole module exists to fix.  Turn it off for a plain
    least-squares fit.

    ``clip`` then removes what the robust loss only down-weights.  Points whose
    residual exceeds ``clip`` robust sigmas (MAD-based, so a spike cannot
    inflate the threshold it is judged against) are dropped and the curve is
    refitted without them, repeating until nothing more is found.  Down-weighting
    alone is not enough at this scale: on the 0907 run four latched DAQ cells on
    channel 11 survived the soft-L1 loss and left a fit with an rms of 824% of
    contrast and an encoding window of 417..573 instead of 399..887 -- a channel
    that would have been driven badly wrong by every step-6/7/8 run reading that
    file.  Dropping them recovers rms 0.18% and a window in line with the other
    eleven channels.  The count is kept in ``n_clipped`` so a stored fit says so.

    The sigma is judged against the *final* model, curvature included: a
    quadratic-vs-linear mismatch is smooth structure, not noise, and clipping on
    the linear stage would trim the ends of a perfectly good curved sweep.  Set
    ``clip=0`` to fit every finite point.

    Raises :class:`TransferFitError` if there is nothing to fit or the linear
    stage will not converge.  There is deliberately no polynomial fallback for
    the *shape*: a channel whose transfer curve is not sin^2 is a channel to
    investigate, not to silently encode against a curve with no phase meaning.
    The curvature term is not that -- it stays inside ``Delta``, so ``v`` still
    means ``sin^2(Delta/2)`` exactly and the phase identity is preserved.
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

    fit = _fit_points(lv, val, robust=robust, curvature=curvature)
    if clip <= 0.0:
        return fit
    return _clip_and_refit(
        lv, val, fit, robust=robust, curvature=curvature, clip=float(clip)
    )


def _fit_points(
    lv: np.ndarray,
    val: np.ndarray,
    *,
    robust: bool,
    curvature: bool,
) -> TransferFit:
    """Both fit stages on an already-cleaned point set; no outlier handling.

    Split out so the clip loop can refit an identical model on a subset without
    re-running the input validation, and so ``n_used`` and the swept range
    always describe the points the returned fit actually saw.
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

    # --- normalize the branch (see the module docstring) ---
    # The model is invariant under (slope, offset) -> (-slope, -offset), so take
    # slope > 0; then fold offset by whole periods so Delta = 0 sits at the
    # darkest swept level, putting the encoding window on Delta in [0, pi].
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
) -> TransferFit:
    """Drop points past ``clip`` robust sigmas and refit, until nothing is found.

    Each pass measures the residual scale from the current fit with a MAD --
    the median absolute deviation, scaled to a Gaussian sigma -- so the points
    being judged cannot inflate their own threshold the way an rms would.  The
    scale is floored at :data:`_CLIP_MIN_SIGMA_FRAC` of the fitted contrast,
    which is what keeps the test from eating good points on a curve the model
    fits almost exactly (see the constant).

    Every accepted pass refits from scratch on the survivors rather than
    updating in place, so the returned fit -- including ``n_used``, ``rms`` and
    the swept range -- describes exactly the points it was built from.

    Bails out, keeping whatever the last accepted pass produced, when the clip
    wants more than :data:`_CLIP_MAX_FRACTION` of the sweep, when too few points
    would survive to fit, or when the refit fails.  Those are all "this is not a
    curve with a few bad cells" and the honest answer is the unclipped fit with
    its large rms visible, not a tidy fit of whatever agreed with itself.

    Know the limit: this can only remove outliers the fit disagrees with.  If
    the bad points are numerous or coherent enough to capture the fit, their
    residuals are small and no threshold finds them -- four adjacent cells at
    twice full scale drag the synthetic sweep's ``on_level`` past 1600 and are
    clipped at no sigma at all, while lowering the threshold only starts
    deleting the good points instead.  What catches that case is the absurd
    ``contrast`` and encoding window it leaves behind, not this.
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
    return fit if n_clipped == 0 else replace(fit, n_clipped=n_clipped)


def _refine_curvature(
    lv: np.ndarray,
    val: np.ndarray,
    linear: TransferFit,
    *,
    robust: bool,
) -> TransferFit | None:
    """Refit with a quadratic ``Delta``, or None if that does not earn its keep.

    Returns None rather than raising: a curvature term that will not converge,
    does not improve the rms, or turns ``Delta`` over inside the sweep is a
    refinement that failed, not a calibration that failed.  The caller already
    holds a valid linear fit.
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

    # Branch, in u-space: the model is invariant under negating every
    # coefficient, so make Delta increase; then fold whole periods so Delta = 0
    # sits at the darkest swept level, as in the linear case.
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
    """Largest ``level_for`` disagreement between two fits, over v in [0, 1].

    Unrounded, so a shift below one grayscale still reads as what it is.
    """
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

    Raises :class:`TransferFitError` naming the row that failed, so a bad
    channel is identifiable without re-running the fits one at a time.

    Each row is clipped on its own residuals, so one channel's bad cells never
    affect another's threshold.  Check ``n_clipped`` on the returned fits to see
    which channels needed it.
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
