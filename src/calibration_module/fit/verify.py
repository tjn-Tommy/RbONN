"""n-pair verification -- predict every pattern from the calibration, compare.

In:  one ``calib.json`` through :func:`load_forward_model` (steps 6 and 7 out of
     the SAME file, which is what makes the etas and phases a matched set), plus
     the drives to predict.
Out: predictions from :func:`predict`, the block's ceiling from
     :class:`FullScale`, and the fitted ``Y_sat`` from :func:`fit_y_sat`.
     Imports no driver, so all of it runs against a saved CSV.

Steps 3-7 calibrate every parameter the model needs.  For a block of ``n`` pairs
driven at ``(x_k, w_k)`` with every other pair off, the dark-subtracted output
is predicted with nothing left free::

    u_k = asin sqrt(x_k)                    v_k = asin sqrt(w_k)
    E   = sum_k  eta_k sqrt(x_k w_k) exp(i [u_k + v_k + Phi_k])
    Y   = |E|^2 / (1 + |E|^2 / Y_sat)  +  sum_k single_beam_k(x_k, w_k)

THE model -- there is one, it is this, and every fit in the project is taken
under it.  ``u_k = phi_half(x_k)``: a channel's field phase is half its panel
retardance, the same number that sets its amplitude, so amplitude and phase are
welded together (see the note above :func:`support_h`).  ``Phi_k`` is step 7's
comb phase in exactly the stored sign -- no consumer negates it.  Saturation
acts on the TPA field only; ``single_beam_k`` is per-beam detector response and
adds outside it, linearly.

Error is quoted against the block's FULL SCALE, the range an n-pair computation
has to divide into levels::

    FS = Y_max - Y_min = Y_max

``Y_min = 0`` because a pair driven at ``x_k = 0`` contributes no field, so
RbONN never needs the phasor polygon to close.  ``Y_max`` is the reachable
ceiling from :func:`reachable_max`, which is NOT ``(sum_k eta_k)^2`` -- again,
the note above :func:`support_h`.
"""
from __future__ import annotations

import csv
import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from .phase import PairModel, load_comb_phase_json, load_pair_models, phi_half

__all__ = [
    "load_forward_model",
    "predict",
    "saturate",
    "FullScale",
    "full_scale",
    "support_h",
    "reachable_max",
    "blocks_for",
    "random_patterns",
    "VerifyBlock",
    "BlockError",
    "evaluate",
    "pairs_tag",
    "write_plan_csv",
    "write_verify_csv",
    "load_verify_csv",
    "drive_bounds",
    "check_sign",
    "note_step7_moved",
]


# ======================================================================
# the forward model
# ======================================================================

def note_step7_moved(run, meta, log=print) -> bool:
    """Warn when a recorded CSV was driven under a different step 7 than this one.

    Every file is called ``calib.json``, so a filename cannot identify a step 7;
    its ``ran`` stamp can, and the driving run writes that into the CSV meta as
    ``step7_ran``.

    Out: True when it said something.  A CSV from before the stamp carries none
    and passes in silence -- there is nothing to compare, and warning on every
    historical file would train the operator to ignore the one that matters.
    """
    from calibration_module.calib_file import load_calib

    recorded = str(meta.get("step7_ran") or "").strip()
    if not recorded:
        return False
    try:
        current = load_calib(run).ran(7)
    except Exception:                       # not a v3 file; nothing to compare
        return False
    if current and recorded != current:
        log(f"NOTE: this CSV was driven under the step 7 of {recorded}, and is "
            f"being fitted against the one of {current} -- the phases have been "
            f"re-fitted since the sweep")
        return True
    return False



def load_forward_model(
    run: str | Path, *, pairs: Sequence[int] | None = None,
    method: str | None = None,
) -> tuple[dict[int, PairModel], dict[int, float], int]:
    """``(models, phases, ref)`` for ``pairs`` out of one calib.json.

    In:  ``run``, a ``calib.json`` or the directory holding one.  The step-6
         etas and step-7 phases both come from that ONE file, which is what
         makes them a matched set.  ``pairs=None`` takes every pair the file
         calibrates end to end -- the reference plus each target with a fit.
    Out: ``phases[k]`` is ``Phi_k`` in radians exactly as :func:`predict` uses
         it, with ``phases[ref] = 0``.

    A stored ``dphi_comb_rad`` IS ``Phi_k``, step 7 fitting
    ``cos(dPhi_SLM + dPhi_comb)``, so it reads through untouched.  A file
    tagged ``"convention": "comb-slm"`` holds the NEGATED phase and is read
    as-is, not migrated: re-run step 7 rather than trust one.
    """
    from calibration_module.calib_file import calib_path

    step7 = calib_path(run)
    models = load_pair_models([step7])
    ref, entries = load_comb_phase_json(step7, method=method)
    phases = {int(ref): 0.0}
    for k, ch in entries.items():
        if int(k) == int(ref):
            continue                       # the reference defines Phi = 0
        phases[int(k)] = float(ch["fit"]["dphi_comb_rad"])
    calibrated = sorted(k for k in phases if k in models)
    pairs = calibrated if pairs is None else sorted(int(k) for k in pairs)
    missing = [k for k in pairs if k not in calibrated]
    if missing:
        raise ValueError(
            f"pair(s) {missing} are not calibrated end to end in {step7.name} "
            f"(a step-6 eta and, unless it is reference {ref}, a step-7 phase); "
            f"it covers {calibrated}"
        )
    return {k: models[k] for k in pairs}, {k: phases[k] for k in pairs}, int(ref)


def saturate(y, y_sat: float | None):
    """Two-photon saturation of the readout, ``Y / (1 + Y / Y_sat)``.

    ``Y_sat`` is the ``|E|^2``, in detector volts, at which the response has
    fallen to half the linear one.  ``None`` or ``0`` is the linear model, which
    is what steps 6 and 7 use: they read at most ~66 mV, where the correction is
    ~1%.
    """
    if not y_sat:
        return y
    return y / (1.0 + y / float(y_sat))


def fit_y_sat(y, pred, sigma=None, *, y_lo: float = 2.0, y_hi: float = 40.0,
              points: int = 20001) -> dict:
    """Fit ``Y_sat`` to reads ``y`` against LINEAR predictions ``pred``.

    In:  ``pred`` the unsaturated model output (:func:`predict` with
         ``y_sat=None``), ``y`` the dark- and background-subtracted read, and
         ``sigma`` the residual weights (``None`` weights every read equally).
    Out: the fit and its interval.  ONE free parameter -- the gain is pinned
         to 1.

    The grid is uniform in ``1/Y_sat``, the parameter the model is linear in
    (``1/y = 1/Y + 1/Y_sat``), so a uniform ``Y_sat`` grid would spend most of
    its points where the data says nothing.  The quoted uncertainty is the
    ``Delta chi^2 = 1`` interval mapped back through the inverse, hence
    asymmetric in ``Y_sat``.

    THE GAIN CAVEAT, which decides what this number means.  With ``g`` pinned at
    1, any gain error is absorbed into ``Y_sat`` and levered by roughly
    ``Y_sat / max(pred)``, about 12x on a sweep reaching 400 mV.  Measured
    overnight on an unchanged bench, a ``g = 1`` ``Y_sat`` swung 4.39..5.44 V
    while tracking the separately measured gain at r = 0.98; with the gain free
    it sat at 6.84 +- 0.20 V with no trend.

    That is the point, not a defect: this fit DEFINES ``g = 1``, and a later run
    measures its gain as the ratio of its reads to this run's on the same
    drives.  Re-fitting another day gives a different ``Y_sat`` without either
    being wrong.  What must not happen is mixing one day's ``Y_sat`` with
    another day's reads.
    """
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    w = (np.ones_like(y) if sigma is None
         else 1.0 / np.asarray(sigma, dtype=float))
    if y.size < 3:
        raise ValueError(f"need at least 3 reads to fit Y_sat, got {y.size}")

    inv = np.linspace(1.0 / float(y_hi), 1.0 / float(y_lo), int(points))
    # (points, n): the saturated prediction at every trial 1/Y_sat at once
    resid = (y[None, :] - pred[None, :] / (1.0 + pred[None, :] * inv[:, None])) * w
    chi2 = np.sum(resid ** 2, axis=1)
    i = int(np.argmin(chi2))

    if sigma is None:
        # Delta chi^2 = 1 only means something once chi^2 is in units of sigma.
        # With no sigma supplied the residuals are raw volts, chi^2 is ~1e-6 and
        # the interval would swallow the whole grid, so take the scatter at the
        # best fit AS the sigma.  This is NOT Birge-scaling a fit that already
        # has real sigmas -- load_reads always supplies those, and that branch
        # is left exactly as the weights say.
        var = float(chi2[i]) / max(y.size - 1, 1)
        if var > 0:
            chi2 = chi2 / var
        else:
            # Exact data: the fit is pinned to one grid point, and the honest
            # interval is the grid resolution, not zero.
            chi2 = np.where(np.arange(chi2.size) == i, 0.0, np.inf)

    within = chi2 <= chi2[i] + 1.0
    lo_inv, hi_inv = float(inv[within].min()), float(inv[within].max())
    y_sat = float(1.0 / inv[i])
    # hi in 1/Y_sat is lo in Y_sat, hence the swap
    lo_v, hi_v = 1.0 / hi_inv, 1.0 / lo_inv
    edge = i in (0, len(inv) - 1) or within[0] or within[-1]

    def rms(model):
        return float(np.sqrt(np.mean((y - model) ** 2)))

    return {
        "y_sat_v": y_sat,
        "sigma_v": float((hi_v - lo_v) / 2.0),
        "lo_v": float(lo_v), "hi_v": float(hi_v),
        "rmse_v": rms(saturate(pred, y_sat)),
        "rmse_linear_v": rms(pred),
        "n": int(y.size),
        "max_pred_v": float(pred.max()),
        #: Y_sat / max(pred) -- how hard a gain error is levered into Y_sat.
        "gain_lever": float(y_sat / pred.max()) if pred.max() > 0 else float("nan"),
        #: True -> the Delta chi^2 = 1 interval ran into the grid; the sweep
        #: did not go bright enough to bound Y_sat and the value is a limit.
        "unbounded": bool(edge),
    }


def predict(models: Mapping[int, PairModel], phases: Mapping[int, float],
            driven: Sequence[int], x, w, *, coherent: bool = False,
            y_sat: float | None = None) -> np.ndarray:
    """Dark-free ``Y`` for drives ``x``, ``w`` of shape ``(..., len(driven))``.

    Column ``i`` drives pair ``driven[i]``.  The single-beam response is detector
    background rather than TPA field, so it adds OUTSIDE the coherent sum, and
    ``y_sat`` applies :func:`saturate` to ``|E|^2`` alone.

    ``coherent=True`` returns ``|E|^2`` by itself.  That is what ``FS = Y_max``
    is defined on (:class:`FullScale`) and so what a level target means, while
    the full ``Y`` is what the ADC sees and so what an input range is chosen
    from.  Keeping them apart matters: the background ramps with level, a tilt
    that would otherwise read as model bias.
    """
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    if x.shape != w.shape or x.shape[-1] != len(driven):
        raise ValueError(f"x {x.shape} and w {w.shape} must both end in "
                         f"len(driven) = {len(driven)}")
    field = np.zeros(x.shape[:-1], dtype=complex)
    bg = np.zeros(x.shape[:-1])
    for i, k in enumerate(driven):
        m = models[k]
        xi, wi = x[..., i], w[..., i]
        field = field + m.amplitude(xi, wi) * np.exp(
            1j * (phi_half(xi) + phi_half(wi) + phases[k]))
        bg = bg + m.single_beam(xi, wi)
    coh = saturate(np.abs(field) ** 2, y_sat)
    return coh if coherent else coh + bg


# ======================================================================
# full scale
# ======================================================================
#
# Why FS is not (sum_k eta_k)^2
# -----------------------------
# That is the FREE-PHASE bound: n phasors of fixed magnitudes eta_k whose phases
# may be chosen at will, all lined up.  It is out of reach here, because a
# pair's phase and amplitude move together.  With u = phi_half(x) = asin sqrt(x)
# (so x = sin^2 u) and u, v in [0, pi/2],
#
#     z_k(u, v) = eta_k sin u sin v exp(i [u + v + Phi_k])
#
# -- the same field :func:`predict` sums.  A pair reaches |z_k| = eta_k only at
# u = v = pi/2, and there its phase is pinned to Phi_k.  Buying phase costs
# amplitude, so S^2 is attained only when every Phi_k is equal.  On the runs in
# src/calib_data the phases span ~290 deg and S^2 over-predicts the real ceiling
# by 54..67 %, which would flatter every NRMSE by roughly 2x.
#
# The exact Y_max is cheap.  The pairs are driven independently, so the
# reachable set of E = sum_k z_k is the Minkowski sum of the per-pair sets, and
# a Minkowski sum's support function is the SUM of the per-pair support
# functions -- so for a fixed projection axis psi each pair optimises on its own:
#
#     h_k(psi) = max_{u,v} Re(z_k e^{-i psi})
#              = max_{t in [0, pi]} eta_k (1 - cos t)/2 cos(a_k - t)
#
# with a_k = psi - Phi_k and t = u + v (at fixed t the phase is fixed and
# sin u sin v peaks at u = v = t/2, so u = v = t/2 is exactly optimal).  Since
# |E| = max_psi h(psi),
#
#     Y_max = max_psi ( sum_k h_k(psi) )^2 ,
#
# which stays a 1-D scan for every n -- O(n) per psi, against O(m^(2n)) for a
# grid sweep over the 2n drives.

#: psi samples for the Y_max scan, before the local refinement that follows it.
#: The grid only has to land in the right basin: 2001 and 200001 agree to 12
#: digits on every run in src/calib_data, so this is not a tolerance to tune.
PSI_GRID = 4001


def support_h(psi, eta: float, phase: float) -> tuple[np.ndarray, np.ndarray]:
    """``(h_k(psi), t)`` for one pair: its support, and the ``t = u + v`` attaining it.

    ``h_k(psi) = max_{t in [0, pi]} eta (1 - cos t)/2 cos(psi - phase - t)``,
    maximised in closed form: ``d/dt = 0`` gives ``t* = (pi + 2a)/3`` up to
    ``2 pi n/3``, so every branch landing in ``[0, pi]`` is tried along with both
    endpoints.  ``a = psi - phase`` spans ``(-2 pi, 2 pi]``, hence ``n = -3..3``
    -- too narrow a window undershoots the maximum silently rather than failing.
    """
    psi = np.atleast_1d(np.asarray(psi, dtype=float))
    a = psi - float(phase)
    n = np.arange(-3.0, 4.0)[:, None]
    t = np.concatenate([(np.pi + 2.0 * a) / 3.0 + 2.0 * np.pi * n / 3.0,
                        np.zeros_like(a)[None, :],
                        np.full_like(a, np.pi)[None, :]])
    ok = (t >= -1e-12) & (t <= np.pi + 1e-12)
    t = np.clip(t, 0.0, np.pi)
    f = np.where(ok, float(eta) * 0.5 * (1.0 - np.cos(t)) * np.cos(a - t), -np.inf)
    i = f.argmax(axis=0)
    j = np.arange(f.shape[1])
    return f[i, j], t[i, j]


def reachable_max(etas, phases) -> tuple[float, np.ndarray, float]:
    """``(Y_max, drive, psi)`` -- the largest ``|E|^2`` these pairs can be driven to.

    ``drive[k]`` is the ``x_k = w_k`` attaining it, the optimum being symmetric
    in x and w since ``sin u sin v`` peaks at ``u = v`` for fixed ``u + v``.  So
    ``predict(models, phases, driven, drive, drive)`` reproduces ``Y_max``
    exactly, which is how this is tested: :func:`predict` knows nothing about
    the support scan.
    """
    etas = np.asarray(etas, dtype=float)
    phases = np.asarray(phases, dtype=float)

    def total(psi):
        return sum(support_h(psi, etas[k], phases[k])[0] for k in range(etas.size))

    psi = np.linspace(-np.pi, np.pi, PSI_GRID)
    grid = total(psi)
    i = int(grid.argmax())
    step = float(psi[1] - psi[0])
    best = float(psi[i])
    try:
        # Brent off the grid maximum.  The bracket is only valid in the
        # interior, and a refinement that came out worse than the grid point
        # itself is discarded rather than trusted.
        if 0 < i < psi.size - 1:
            r = minimize_scalar(lambda q: -total(np.array([q]))[0],
                                bracket=(psi[i] - step, psi[i], psi[i] + step))
            if -float(r.fun) >= float(grid[i]):
                best = float(r.x)
    except (ValueError, RuntimeError):
        pass                            # the grid point stands
    h = float(total(np.array([best]))[0])
    t = np.array([support_h(np.array([best]), etas[k], phases[k])[1][0]
                  for k in range(etas.size)])
    return h ** 2, (1.0 - np.cos(t)) / 2.0, best


@dataclass(frozen=True)
class FullScale:
    """A block's full scale ``FS = Y_max``, the span a computation divides into levels.

    ``Y_min = 0`` is always reachable, so ``FS = Y_max``: the largest ``|E|^2``
    the pairs can ACTUALLY be driven to, phase and amplitude coupled as they
    are.  The single-beam background is not part of it, being detector offset
    rather than what a computation drives.

    :attr:`bound` is the free-phase ``S^2`` for comparison, :attr:`headroom` how
    much of it the measured phases allow.
    """

    driven: tuple[int, ...]
    amplitudes: tuple[float, ...]
    phases: tuple[float, ...]
    fs: float
    drive: tuple[float, ...]
    psi: float

    @property
    def s(self) -> float:
        return float(sum(self.amplitudes))

    @property
    def bound(self) -> float:
        """``S^2`` -- the free-phase ceiling, reached only if every ``Phi_k`` is equal."""
        return self.s ** 2

    @property
    def headroom(self) -> float:
        """``Y_max / S^2`` -- exactly 1 for a single pair, less once phases disagree."""
        return self.fs / self.bound if self.bound > 0 else float("nan")


def full_scale(models: Mapping[int, PairModel], phases: Mapping[int, float],
               driven: Sequence[int]) -> FullScale:
    """``FS = Y_max`` over the pairs in ``driven`` -- see :func:`reachable_max`.

    Takes the phases, not just the amplitudes: the ceiling depends on how far
    out of line the ``Phi_k`` are.
    """
    driven = tuple(int(k) for k in driven)
    etas = [float(models[k].amplitude(1.0, 1.0)) for k in driven]
    phi = [float(phases[k]) for k in driven]
    fs, drive, psi = reachable_max(etas, phi)
    return FullScale(driven, tuple(etas), tuple(phi), fs,
                     tuple(float(d) for d in drive), psi)


# ======================================================================
# the drive
# ======================================================================

def blocks_for(pairs: Sequence[int], n: int) -> list[tuple[int, ...]]:
    """Every ``n``-subset of ``pairs``, ``C(N, n)`` of them, in lexical order."""
    pairs = sorted(int(k) for k in pairs)
    if not 1 <= int(n) <= len(pairs):
        raise ValueError(f"n must be in 1..{len(pairs)} (the calibrated pairs "
                         f"{pairs}), got {n}")
    return list(itertools.combinations(pairs, int(n)))


def random_patterns(n_blocks: int, n: int, patterns: int, lo: float, hi: float,
                    *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """``(x, w)``, each ``(n_blocks, patterns, n)``, independent uniform in ``[lo, hi]``.

    Rounded to 6 decimals, which is what the CSV stores, so a prediction
    recomputed from the file is the one made at collect time.
    """
    if not 0.0 <= lo < hi <= 1.0:
        raise ValueError(f"drive box must satisfy 0 <= lo < hi <= 1, got {lo} .. {hi}")
    rng = np.random.default_rng(seed)
    shape = (int(n_blocks), int(patterns), int(n))
    return (np.round(rng.uniform(lo, hi, shape), 6),
            np.round(rng.uniform(lo, hi, shape), 6))


# ======================================================================
# measured blocks and their error
# ======================================================================

@dataclass
class VerifyBlock:
    """One driven block's recorded patterns: drives ``(P, n)``, reads ``(P,)``."""

    driven: tuple[int, ...]
    x: np.ndarray
    w: np.ndarray
    dark_v: np.ndarray
    mean_v: np.ndarray
    std_v: np.ndarray
    range_v: np.ndarray
    pred_v: np.ndarray
    #: Any column this module has no meaning for, carried through verbatim --
    #: the system-accuracy run writes ``kind``, ``seq``, ``target_alpha`` and
    #: friends, and its report reads them back without verify.py having to know
    #: what they are.  Numeric columns arrive as float arrays, the rest as str.
    extra: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.driven)

    @property
    def y(self) -> np.ndarray:
        """The dark-subtracted signal the model predicts."""
        return self.mean_v - self.dark_v


@dataclass
class BlockError:
    """One block's prediction against its measurement, on its full scale."""

    driven: tuple[int, ...]
    scale: FullScale
    y: np.ndarray
    pred: np.ndarray

    @property
    def resid(self) -> np.ndarray:
        return self.y - self.pred

    @property
    def rms_v(self) -> float:
        return float(np.sqrt(np.mean(self.resid ** 2)))

    @property
    def nrmse_pct(self) -> float:
        """``rms(meas - pred)`` as a percentage of the block's reachable ``Y_max``."""
        fs = self.scale.fs
        return 100.0 * self.rms_v / fs if fs > 0 else float("nan")

    @property
    def max_pct(self) -> float:
        fs = self.scale.fs
        return 100.0 * float(np.max(np.abs(self.resid))) / fs if fs > 0 else float("nan")

    def position(self, values) -> np.ndarray:
        """``values`` as a fraction of the full scale: 0 is dark, 1 is ``S^2``."""
        return np.asarray(values, dtype=float) / self.scale.fs


def evaluate(block: VerifyBlock, models, phases) -> BlockError:
    """Predict a recorded block and quote its error against its full scale."""
    return BlockError(
        driven=block.driven,
        scale=full_scale(models, phases, block.driven),
        y=block.y,
        pred=predict(models, phases, block.driven, block.x, block.w),
    )


# ======================================================================
# persistence
# ======================================================================

_TAIL = ["dark_v", "voltage_mean_v", "voltage_std_v", "range_v", "pred_v"]


def pairs_tag(driven: Sequence[int]) -> str:
    """``(2, 4)`` -> ``"2+4"`` -- the self-describing driven-set column."""
    return "+".join(str(int(k)) for k in driven)


def write_verify_csv(path: str | Path, pairs: Sequence[int], rows: Sequence[Mapping],
                     *, meta: Mapping[str, object],
                     extra: Sequence[str] = ()) -> str:
    """One row per pattern; ``x_k, w_k`` for every pair, 0 for the undriven ones.

    Each row mapping carries ``block, driven, x, w`` (in ``driven`` order) and
    ``dark_v, mean_v, std_v, range_v, pred_v``; ``meta`` goes in trailing
    ``# key,value`` lines.  The layout is step 8's with ``v`` dropped and
    ``range_v`` added, so :func:`load_verify_csv` reads both.

    ``extra`` names further per-row columns appended after ``pred_v``, blank
    where a row omits one.  That is how the system-accuracy run records which
    family a read belongs to without a second file format -- the leading columns
    being unchanged, such a CSV still loads as an ordinary n-pair block.
    """
    pairs = sorted(int(k) for k in pairs)
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["block", "pairs"]
                        + [c for k in pairs for c in (f"x_{k}", f"w_{k}")]
                        + _TAIL + list(extra))
        for r in rows:
            drive = dict(zip(r["driven"], zip(r["x"], r["w"])))
            line = [r["block"], pairs_tag(r["driven"])]
            for k in pairs:
                xk, wk = drive.get(k, (0.0, 0.0))
                line += [f"{float(xk):.6f}", f"{float(wk):.6f}"]
            line += [f"{r['dark_v']:.9g}", f"{r['mean_v']:.9g}", f"{r['std_v']:.9g}",
                     f"{r['range_v']:g}", f"{r['pred_v']:.9g}"]
            line += [r.get(c, "") for c in extra]
            writer.writerow(line)
        for key, value in meta.items():
            f.write(f"# {key},{value}\n")
    return str(out)


def write_plan_csv(path: str | Path, patterns: Sequence, preds: Sequence[float],
                   coh: Sequence[float], *, pairs: Sequence[int], scale,
                   meta: Mapping[str, object]) -> str:
    """Step 8's pattern plan, written BEFORE the hardware is touched.

    One row per pattern, laid out exactly as :func:`write_verify_csv` does, so
    the plan and the reads it produced line up column for column.  The scalars
    it was solved under -- the step-7 stamp, the seed, ``Y_max``, the ladder
    geometry -- go in the usual trailing ``# key,value`` meta lines.

    A CSV and not a JSON: a calibration is one JSON, and a list of patterns is a
    table.  ``exact`` is False where a pattern fell back to the deterministic
    drive because the solver could not hit its target alpha.
    """
    pairs = sorted(int(k) for k in pairs)
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["seq", "kind", "label", "target_alpha", "alpha",
                         "coherent_v", "pred_v", "exact"]
                        + [c for k in pairs for c in (f"x_{k}", f"w_{k}")])
        amps = dict(zip(pairs, getattr(scale, "amplitudes", [])))
        phis = dict(zip(pairs, getattr(scale, "phases", [])))
        for i, (p, c, q) in enumerate(zip(patterns, coh, preds)):
            drive = dict(zip(pairs, zip(p.x, p.w)))
            line = [i, p.kind, p.label,
                    "" if p.target_alpha is None else f"{p.target_alpha:.9g}",
                    f"{p.alpha:.9g}", f"{c:.9g}", f"{q:.9g}", p.exact]
            for k in pairs:
                xk, wk = drive.get(k, (0.0, 0.0))
                line += [f"{float(xk):.6f}", f"{float(wk):.6f}"]
            writer.writerow(line)
        for key, value in meta.items():
            f.write(f"# {key},{value}\n")
        for k in pairs:                 # the model, one line per pair
            f.write(f"# eta_{k},{amps.get(k, float('nan')):.9g}\n")
            f.write(f"# phi_rad_{k},{phis.get(k, float('nan')):.9g}\n")
    return str(out)

def load_verify_csv(path: str | Path) -> tuple[list[int], list[VerifyBlock], dict[str, str]]:
    """``(pairs, blocks, meta)`` from a verify CSV or an older step-8 CSV.

    Blocks come back in file order, grouped by the ``block`` and ``pairs``
    columns; each block's drives are read from its own ``x_k, w_k`` columns, so
    an old ``x = w = v`` file loads the same way.  Columns an old file lacks
    (``range_v``) come back NaN.
    """
    meta: dict[str, str] = {}
    lines: list[str] = []
    with open(Path(path), newline="", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                key, _, value = line[1:].strip().partition(",")
                meta[key.strip()] = value.strip()
            else:
                lines.append(line)
    rows = list(csv.DictReader(lines))
    if not rows:
        raise ValueError(f"{Path(path).name} has no data rows")
    if "pairs" not in rows[0]:
        raise ValueError(f"{Path(path).name} has no 'pairs' column, so it is not "
                         f"a verify or step-8 CSV")
    pairs = sorted(int(c[2:]) for c in rows[0] if c.startswith("x_"))

    grouped: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        grouped.setdefault((r["block"], r["pairs"]), []).append(r)

    def col(group, name) -> np.ndarray:
        return np.array([float(r[name]) if r.get(name) not in (None, "") else np.nan
                         for r in group])

    # Anything this module has no meaning for rides along untouched.  "v" is an
    # old step-8 column that x_k/w_k already carry, so it is not passed on.
    named = ({"block", "pairs", "v", None}
             | {c for k in pairs for c in (f"x_{k}", f"w_{k}")} | set(_TAIL))
    spare = [c for c in rows[0] if c not in named]

    def spare_cols(group) -> dict[str, np.ndarray]:
        out = {}
        for name in spare:
            raw = [r.get(name, "") for r in group]
            try:
                out[name] = np.array([float(v) if v not in (None, "") else np.nan
                                      for v in raw])
            except (TypeError, ValueError):
                out[name] = np.array([("" if v is None else str(v)) for v in raw])
        return out

    blocks = []
    for (_, tag), g in grouped.items():
        driven = tuple(int(t) for t in tag.split("+"))
        blocks.append(VerifyBlock(
            driven=driven,
            x=np.array([[float(r[f"x_{k}"]) for k in driven] for r in g]),
            w=np.array([[float(r[f"w_{k}"]) for k in driven] for r in g]),
            dark_v=col(g, "dark_v"), mean_v=col(g, "voltage_mean_v"),
            std_v=col(g, "voltage_std_v"), range_v=col(g, "range_v"),
            pred_v=col(g, "pred_v"), extra=spare_cols(g),
        ))
    check_sign(path, blocks)
    return pairs, blocks, meta


def drive_bounds(meta: Mapping[str, str], default: tuple[float, float]) -> tuple[float, float]:
    """The drive box a CSV was collected over.

    Read from its ``# drive`` line, else an old step-8 file's ``# sweep`` line,
    else ``default``.
    """
    for key in ("drive", "sweep"):
        fields = dict(p.split("=", 1) for p in meta.get(key, "").split(",") if "=" in p)
        if "min" in fields and "max" in fields:
            return float(fields["min"]), float(fields["max"])
    return default


def check_sign(path, blocks: Sequence[VerifyBlock]) -> None:
    """Refuse a CSV whose dark-subtracted signal is negative (inverted twice).

    The read inverts the TIA's negative-for-light output once, so a well
    collected file has ``mean - dark > 0``.  Taken as a MEDIAN, so one noisy
    near-dark pattern cannot trip it.  What this catches is that against a
    negated measurement the model yields a plausible table of large errors
    rather than an error.
    """
    y = np.concatenate([b.y for b in blocks])
    med = float(np.median(y))
    if med < 0.0:
        raise ValueError(
            f"{Path(path).name}: dark-subtracted signal is NEGATIVE (median "
            f"{med * 1e3:.4f} mV) -- the sign was inverted a second time on top of "
            f"the read's own inversion. Re-collect, or negate voltage_mean_v and "
            f"dark_v to undo it."
        )
