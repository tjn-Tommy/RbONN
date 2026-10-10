"""Drives that land on a chosen fraction of full scale.

In:  the step-6 pair models and step-7 phases, plus a target ``alpha``.
Out: :class:`Pattern` objects -- commanded ``(x, w)`` per pair, the alpha
     achieved, and whether the solve converged.  Imports no driver, so all of it
     runs offline.

Driving ``x_k, w_k`` uniform in a box is useless for anything but the bottom of
the scale: ``E = sum_k z_k`` is a random phasor sum, so ``|E|^2`` concentrates
near the INCOHERENT value ``sum_k |z_k|^2`` ~ Y_max/10.  Measured, uniform
drives give ``Y/Y_max`` a mean of 0.096 and ``P(Y > 0.5 Y_max) = 5e-6``.  A
4-bit computation divides the WHOLE scale into sixteen levels, so it has to be
verified over the whole scale.

The pattern families::

    A  level_patterns      one pattern per stratum of [0, Y_max], each a fresh
                           random drive -- model error against level
    F  full_scale_pattern  the closed-form drive reaching Y_max exactly -- the
                           ceiling, measured rather than predicted
    B  repeat_patterns     ONE drive per level, repeated -- readout
                           reproducibility, the 4-bit noise budget
    C  psi_scan_patterns   the support-optimal drive for a fan of projection
                           axes -- where full scale is, and the phase error

``calib_step8.py`` drives A and F: one free parameter, and the ceiling is where
its leverage lives.  B and C are kept because runs on disk contain them and
``levels_report.py`` still reads those.

Only A needs solving.  Roll a uniform random drive and move it as little as
possible until it predicts ``alpha * Y_max``::

    minimize ||d - d0||^2    subject to    |E(d)|^2 = alpha * Y_max

The least-move constraint is what keeps the randomness: millions of drives reach
any given level and an unconstrained solve would return the same one every time,
making all of A one drive repeated.

The (u, v) parametrisation
--------------------------
Solving in ``x, w`` is badly conditioned -- ``dphi/dx`` diverges at ``x = 1``.
With ``u = phi_half(x) = asin sqrt(x)`` and ``u, v`` in ``[0, pi/2]``::

    z_k(u, v) = eta_k sin u sin v exp(i [u + v + Phi_k])

which is exactly the field :func:`verify.predict` sums, with ``|E|^2`` and its
gradient both closed form.  That is the difference between 10 ms and 3 s per
pattern, so every pattern is solved before the hardware is touched.

Note what the formula says: a pair's amplitude and phase move TOGETHER, so
turning a pair up lengthens its phasor and rotates it.  Two knobs per pair is
what makes a level reachable at all -- at fixed ``u + v`` the phase is fixed,
and trading ``u`` against ``v`` changes the amplitude.

Alpha is COHERENT
-----------------
``alpha`` is a fraction of ``FS = Y_max = max |E|^2``, which EXCLUDES the
single-beam background: that term is detector response to each beam separately,
not TPA field.  So every target here is on ``|E|^2`` alone
(:func:`verify.predict` with ``coherent=True``).  The background is put back for
two purposes only -- choosing a read's input range, since the ADC sees the
total, and predicting the raw voltage in the CSV -- and the report subtracts it
again.  Mixing the two conventions tilts the level axis: the background ramps
with level, which is a third of the real bias at level 1.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Mapping, Sequence

import numpy as np
from scipy.optimize import brentq, minimize

from .phase import PairModel
from .verify import FullScale, full_scale, support_h

__all__ = [
    "U_MAX",
    "Pattern",
    "uv_from_xw",
    "xw_from_uv",
    "field_uv",
    "y_coherent_uv",
    "grad_y_coherent_uv",
    "drive_at_psi",
    "attenuated_drive",
    "project_to_level",
    "level_patterns",
    "repeat_patterns",
    "psi_scan_patterns",
    "full_scale_pattern",
]

#: ``u = asin sqrt(x)`` for ``x`` in [0, 1], so the box is [0, pi/2] per knob.
U_MAX = np.pi / 2

#: Drives are stored to this many decimals, which is what the CSV keeps, so a
#: prediction recomputed from the file is the one made here.
DRIVE_DECIMALS = 6


@dataclass(frozen=True)
class Pattern:
    """One pattern to drive: what to display, and what it is for.

    ``alpha`` is the coherent level ACHIEVED after rounding the drive to
    :data:`DRIVE_DECIMALS`, not the target that was asked for -- so a report
    reading the JSON never has to re-solve anything.  ``label`` is carried into
    the CSV so a row can be traced back to the family that produced it.
    """

    kind: str                   # "A" ladder, "B" repeats, "C" psi scan, "F" ceiling
    x: np.ndarray
    w: np.ndarray
    alpha: float
    label: str
    target_alpha: float = float("nan")   # A, B: what was asked for
    psi_deg: float = float("nan")        # C: the projection axis
    repeat_id: int = -1                  # B: which repeat of its level
    exact: bool = True                   # False -> the solver fell back


# ======================================================================
# the (u, v) parametrisation
# ======================================================================

def uv_from_xw(x, w) -> np.ndarray:
    """``[u, v]`` for drives ``x, w`` -- the inverse of :func:`xw_from_uv`."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    w = np.clip(np.asarray(w, dtype=float), 0.0, 1.0)
    return np.concatenate([np.arcsin(np.sqrt(x)), np.arcsin(np.sqrt(w))])


def xw_from_uv(uv) -> tuple[np.ndarray, np.ndarray]:
    """``(x, w)`` for a packed ``[u, v]`` vector."""
    uv = np.asarray(uv, dtype=float)
    n = uv.size // 2
    return np.sin(uv[:n]) ** 2, np.sin(uv[n:]) ** 2


def field_uv(uv, eta, phase) -> np.ndarray:
    """Each pair's phasor ``z_k = eta_k sin u sin v exp(i[u + v + Phi_k])``."""
    uv = np.asarray(uv, dtype=float)
    n = uv.size // 2
    u, v = uv[:n], uv[n:]
    return (np.asarray(eta, dtype=float) * np.sin(u) * np.sin(v)
            * np.exp(1j * (u + v + np.asarray(phase, dtype=float))))


def y_coherent_uv(uv, eta, phase) -> float:
    """``|sum_k z_k|^2`` -- the same quantity ``verify.predict(coherent=True)`` gives."""
    return float(np.abs(field_uv(uv, eta, phase).sum()) ** 2)


def grad_y_coherent_uv(uv, eta, phase) -> np.ndarray:
    """``d|E|^2 / d(u, v)``, closed form.

    ``dz_k/du = eta_k sin v (cos u + i sin u) e^{i(u+v+Phi)}`` (the amplitude and
    the phase both move, which is the whole point), and
    ``d|E|^2/du_k = 2 Re(conj(E) dz_k/du)``.
    """
    uv = np.asarray(uv, dtype=float)
    n = uv.size // 2
    u, v = uv[:n], uv[n:]
    eta = np.asarray(eta, dtype=float)
    ph = np.exp(1j * (u + v + np.asarray(phase, dtype=float)))
    field = eta * np.sin(u) * np.sin(v) * ph
    total = field.sum()
    dz_du = eta * np.sin(v) * (np.cos(u) + 1j * np.sin(u)) * ph
    dz_dv = eta * np.sin(u) * (np.cos(v) + 1j * np.sin(v)) * ph
    return np.concatenate([2.0 * np.real(np.conj(total) * dz_du),
                           2.0 * np.real(np.conj(total) * dz_dv)])


# ======================================================================
# the deterministic drives -- full scale, and the fallback ladder
# ======================================================================

def drive_at_psi(scale: FullScale, psi: float) -> np.ndarray:
    """``x = w`` per pair maximising the projection of ``E`` onto axis ``psi``.

    At ``psi = scale.psi`` this is exactly ``scale.drive`` (the Y_max drive);
    away from it the phasors line up on the wrong axis, so ``|E|^2`` must come
    out below ``Y_max``.  That fall-off on both sides is what confirms the
    measured peak is the peak.
    """
    t = np.array([support_h(np.array([float(psi)]), eta, phi)[1][0]
                  for eta, phi in zip(scale.amplitudes, scale.phases)])
    return (1.0 - np.cos(t)) / 2.0


def attenuated_drive(scale: FullScale, alpha: float) -> np.ndarray:
    """The ``x = w`` drive on the Y_max ray scaled down to ``alpha * Y_max``.

    Scaling every ``t_k = u_k + v_k`` by a common ``gamma`` in [0, 1] walks
    ``|E|^2`` monotonically from 0 to ``Y_max``, so a 1-D root-find always lands
    on the target.

    The fallback for :func:`project_to_level`, and DETERMINISTIC -- it carries no
    randomness.  It should fire only above ``alpha ~ 0.95``, where the level set
    has collapsed to one drive and there is no diversity left to lose.
    """
    eta = np.asarray(scale.amplitudes, dtype=float)
    phi = np.asarray(scale.phases, dtype=float)
    t_full = np.array([support_h(np.array([scale.psi]), e, p)[1][0]
                       for e, p in zip(eta, phi)])

    def shortfall(gamma: float) -> float:
        t = gamma * t_full
        return y_coherent_uv(np.concatenate([t / 2.0, t / 2.0]), eta, phi) - alpha * scale.fs

    if alpha <= 0.0:
        return np.zeros(eta.size)
    if shortfall(1.0) <= 0.0:                # alpha >= 1: the best we can do
        return (1.0 - np.cos(t_full)) / 2.0
    gamma = brentq(shortfall, 0.0, 1.0, xtol=1e-12)
    t = gamma * t_full
    return np.sin(t / 2.0) ** 2              # x = w = sin^2(t/2), i.e. u = v = t/2


# ======================================================================
# the solve
# ======================================================================

def project_to_level(alpha: float, uv0, eta, phase, *, ymax: float,
                     u_max: float = U_MAX) -> tuple[np.ndarray, float]:
    """``(uv, rel_err)`` -- the point nearest ``uv0`` with ``|E|^2 = alpha * ymax``.

    ``rel_err`` is ``|achieved - target| / ymax``, so a caller can reject a
    solve that did not converge rather than drive a pattern at the wrong level.

    The bounds are the FULL ``[0, u_max]`` box even when ``uv0`` was rolled from
    a narrower one: a bright target genuinely needs some pairs at ``x ~ 0`` --
    those whose ``Phi_k`` cannot be brought into line contribute least when off
    -- and clamping them away makes the high levels infeasible.
    """
    uv0 = np.asarray(uv0, dtype=float)
    target = float(alpha) * float(ymax)
    result = minimize(
        lambda d: float(np.sum((d - uv0) ** 2)),
        uv0,
        jac=lambda d: 2.0 * (d - uv0),
        method="SLSQP",
        bounds=[(0.0, float(u_max))] * uv0.size,
        constraints=[{
            "type": "eq",
            "fun": lambda d: (y_coherent_uv(d, eta, phase) - target) / ymax,
            "jac": lambda d: grad_y_coherent_uv(d, eta, phase) / ymax,
        }],
        options={"maxiter": 400, "ftol": 1e-12},
    )
    err = abs(y_coherent_uv(result.x, eta, phase) - target) / ymax
    return result.x, float(err)


def _scale_of(models: Mapping[int, PairModel], phases: Mapping[int, float],
              driven: Sequence[int]) -> FullScale:
    return full_scale(models, phases, driven)


def _round_drive(x, w) -> tuple[np.ndarray, np.ndarray]:
    return (np.round(np.clip(x, 0.0, 1.0), DRIVE_DECIMALS),
            np.round(np.clip(w, 0.0, 1.0), DRIVE_DECIMALS))


def _achieved(x, w, eta, phase, ymax: float) -> float:
    """The coherent level a ROUNDED drive actually reaches, as a fraction of ymax."""
    return y_coherent_uv(uv_from_xw(x, w), eta, phase) / ymax


# ======================================================================
# A -- the ladder
# ======================================================================

def level_patterns(models: Mapping[int, PairModel], phases: Mapping[int, float],
                   driven: Sequence[int], count: int, *, seed: int,
                   alpha_lo: float = 0.0, alpha_hi: float = 0.995,
                   x_lo: float = 0.02, x_hi: float = 1.0,
                   tries: int = 12, tol: float = 1e-4) -> list[Pattern]:
    """``count`` patterns, one per stratum of ``[alpha_lo, alpha_hi]``.

    Stratified rather than uniform: ``alpha_i = alpha_lo + (i + u_i) * width /
    count`` with ``u_i`` uniform in [0, 1).  The strata guarantee even coverage,
    which uniform sampling only gets on average, and the jitter inside each keeps
    every prediction distinct, so the measured-vs-predicted plot is a scatter
    rather than vertical stripes.  Binning by 4-bit level afterwards still leaves
    ``count/16`` different drives per level, so "do different drives at one level
    read the same" comes out of the same data at no extra cost.

    ``alpha_hi`` stops short of 1, where the level set is a single point and
    patterns carry no diversity -- that is :func:`full_scale_pattern`'s job.
    ``x_lo``/``x_hi`` bound the random START only, not the solution.
    """
    scale = _scale_of(models, phases, driven)
    eta = np.asarray(scale.amplitudes, dtype=float)
    phi = np.asarray(scale.phases, dtype=float)
    n = eta.size
    # Two streams, not one: the stratum jitter must not depend on how many
    # retries the solver needed, or a single extra try shifts every later
    # target -- and then the SAME seed on two different pair subsets asks them
    # for different levels, which is exactly what a leave-one-out sweep must
    # not do.
    rng = np.random.default_rng([seed, 0])
    starts = np.random.default_rng([seed, 1])
    width = (float(alpha_hi) - float(alpha_lo)) / int(count)

    out: list[Pattern] = []
    for i in range(int(count)):
        target = float(alpha_lo) + (i + rng.random()) * width
        best, exact = None, False
        for _ in range(int(tries)):
            uv0 = uv_from_xw(*np.split(starts.uniform(x_lo, x_hi, 2 * n), 2))
            uv, err = project_to_level(target, uv0, eta, phi, ymax=scale.fs)
            if err < tol:
                best, exact = uv, True
                break
        if best is None:
            # Deterministic, and it costs this pattern its randomness -- so the
            # flag rides along into the CSV rather than being swallowed.
            drive = attenuated_drive(scale, target)
            best = uv_from_xw(drive, drive)
        x, w = _round_drive(*xw_from_uv(best))
        out.append(Pattern(kind="A", x=x, w=w,
                           alpha=_achieved(x, w, eta, phi, scale.fs),
                           label=f"A{i:03d}", target_alpha=target, exact=exact))
    return out


# ======================================================================
# F -- the ceiling
# ======================================================================

def full_scale_pattern(models: Mapping[int, PairModel], phases: Mapping[int, float],
                       driven: Sequence[int], *, label: str = "FS") -> Pattern:
    """The closed-form drive that reaches ``Y_max`` exactly -- the top of the scale.

    The single point :func:`level_patterns` stops short of, and it needs no
    solver: :func:`attenuated_drive` at ``alpha = 1`` is the ``Y_max`` ray.

    Driving it turns a PREDICTED full scale into a measured one, and it is where
    a saturation fit has nearly all its leverage -- the curvature grows with
    ``|E|^2`` and vanishes at the bottom of the scale.
    """
    scale = _scale_of(models, phases, driven)
    drive = attenuated_drive(scale, 1.0)
    x, w = _round_drive(drive, drive)
    eta = np.asarray(scale.amplitudes, dtype=float)
    phi = np.asarray(scale.phases, dtype=float)
    return Pattern(kind="F", x=x, w=w,
                   alpha=_achieved(x, w, eta, phi, scale.fs),
                   label=label, target_alpha=1.0, exact=True)


# ======================================================================
# B -- the repeats
# ======================================================================

def repeat_patterns(models: Mapping[int, PairModel], phases: Mapping[int, float],
                    driven: Sequence[int], alphas: Sequence[float], repeats: int,
                    *, seed: int, x_lo: float = 0.02, x_hi: float = 1.0,
                    tries: int = 40, tol: float = 1e-4) -> list[Pattern]:
    """One drive per level in ``alphas``, each repeated ``repeats`` times.

    Solved ONCE per level and copied, so every repeat displays a byte-identical
    frame: the spread of its reads is readout reproducibility with the model held
    fixed, which is what decides whether a single read separates two 4-bit
    levels.  A changes the drive every time and tests the MODEL; B never changes
    it and tests the MACHINE, and A alone cannot separate the two.

    ``alpha = 1`` is allowed here and uses the exact Y_max drive, so B doubles as
    the drift anchor at the top of the scale.
    """
    scale = _scale_of(models, phases, driven)
    eta = np.asarray(scale.amplitudes, dtype=float)
    phi = np.asarray(scale.phases, dtype=float)
    n = eta.size
    rng = np.random.default_rng(seed)

    out: list[Pattern] = []
    for target in alphas:
        target = float(target)
        exact = True
        if target >= 1.0:
            drive = np.asarray(scale.drive, dtype=float)
            uv = uv_from_xw(drive, drive)
        else:
            uv = None
            for _ in range(int(tries)):
                uv0 = uv_from_xw(*np.split(rng.uniform(x_lo, x_hi, 2 * n), 2))
                candidate, err = project_to_level(target, uv0, eta, phi, ymax=scale.fs)
                if err < tol:
                    uv = candidate
                    break
            if uv is None:
                drive = attenuated_drive(scale, target)
                uv = uv_from_xw(drive, drive)
                exact = False
        x, w = _round_drive(*xw_from_uv(uv))
        alpha = _achieved(x, w, eta, phi, scale.fs)
        for r in range(int(repeats)):
            out.append(Pattern(kind="B", x=x.copy(), w=w.copy(), alpha=alpha,
                               label=f"B{target:.2f}r{r + 1}", target_alpha=target,
                               repeat_id=r + 1, exact=exact))
    return out


# ======================================================================
# C -- the psi scan
# ======================================================================

#: Default fan for :func:`psi_scan_patterns`.  A maximum is quadratic, so the
#: scan is flat near the peak and the offsets have to be chosen against the read
#: noise.  On the 0911 runs (sigma ~ 3..5 mV at the peak) the drop from Y_max is
#: 1..3 mV at +/-10 deg -- under 1 sigma, useless on its own -- 4..10 mV at
#: +/-20, 9..28 mV at +/-30 and 14..47 mV at +/-45.  So the fan reaches +/-45 for
#: a lever arm that beats the noise several times over, and keeps +/-10 only
#: because a parabola fit still wants points near the vertex.
PSI_OFFSETS_DEG = (-45, -30, -20, -10, 0, 10, 20, 30, 45)


def psi_scan_patterns(models: Mapping[int, PairModel], phases: Mapping[int, float],
                      driven: Sequence[int],
                      offsets_deg: Sequence[float] = PSI_OFFSETS_DEG,
                      ) -> list[Pattern]:
    """The support-optimal ``x = w`` drive for ``psi* + offset``, one per offset.

    Offset 0 is the drive :func:`verify.reachable_max` says attains ``Y_max``
    and the rest fall off either side.  Reading the FAN rather than the single
    peak turns full scale into a measured quantity: the peak of the scan is
    ``FS``, and its displacement from offset 0 is the model's global phase error,
    which one read at ``psi*`` cannot show.

    The measured peak is a LOWER bound on the true ceiling -- this scans one
    direction of a 2n-dimensional space, so a wrong ``eta`` or ``Phi`` can put
    the real optimum off this curve entirely.  An NRMSE quoted against it is
    conservative, which is the safe direction.

    What it cannot measure: a pair's field phase is ``Phi_k + t`` with ``t`` in
    ``[0, pi]``, so it can only be ADVANCED, never retarded.  With the ``Phi_k``
    tightly clustered, ``psi*`` lands at the edge of that reachable half-plane,
    every pair pins at the corner ``x = w = 1``, and that whole side of the fan
    returns the same drive -- one-sided, so no peak to bracket.  Check
    :func:`verify.full_scale`'s ``headroom = Y_max / S^2`` first: near 1 means
    clustered phases.
    """
    scale = _scale_of(models, phases, driven)
    eta = np.asarray(scale.amplitudes, dtype=float)
    phi = np.asarray(scale.phases, dtype=float)

    out: list[Pattern] = []
    for off in offsets_deg:
        off = float(off)
        drive = drive_at_psi(scale, scale.psi + np.radians(off))
        x, w = _round_drive(drive, drive)
        out.append(Pattern(kind="C", x=x, w=w,
                           alpha=_achieved(x, w, eta, phi, scale.fs),
                           label=f"C{off:+.0f}deg", psi_deg=np.degrees(scale.psi) + off))
    return out
