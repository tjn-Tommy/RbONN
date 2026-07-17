"""Comb-phase (dPhi_comb) calibration of each pair relative to a common reference.

Step 6 (:mod:`slm_module.tpa_pair`) calibrates each pair's two-photon efficiency
``eta`` *in isolation* -- one pair on at a time, so absolute optical phase never
enters.  This step drives **two pairs at once** and uses their coherent TPA
interference to recover the fixed comb phase offset ``dPhi_comb`` that a target
pair carries relative to a reference pair (the reference defines ``Phi = 0``).

Geometry.  A channel commanded at normalised INTENSITY ``x`` in [0, 1] (the
diffraction efficiency) sits at panel phase ``phi = 2 asin(sqrt(x))`` and has field
``sqrt(x) exp(i phi/2)``.  The measured Step-3 transfer curve is monotonic over the
calibrated level range, so ``x`` in [0, 1] reaches ``phi`` in [0, pi] only
(``phi = pi`` is exactly ``x = 1``, fully on) -- a *half* phase turn, enough to
sweep a fringe against a fully-on reference.

For a target pair driven at ``(x_t, w_t)`` against a reference at ``(x_r, w_r)``,
define the target field amplitude and the SLM phase difference::

    g        = sqrt(x_t w_t)
    dPhi_SLM = phi_half(x_t) + phi_half(w_t) - phi_half(x_r) - phi_half(w_r)

with ``phi_half(x) = asin(sqrt(x)) = phi/2`` (:func:`phi_half`,
:func:`slm_phase_diff`).  The measured signal is::

    Y = a^2                                   (reference self term)
      + b^2 g^2                               (target self term)
      + 2 a b g cos(dPhi_SLM + dPhi_comb)     (interference -> dPhi_comb)
      + step-6 single-beam background + d     (fixed background + dark)

where ``a`` := reference amplitude and ``b`` := target amplitude.  ``(g, dPhi_SLM)``
are computed per row straight from the commanded intensities, so the fit is
GEOMETRY-GENERAL: it does not care how the sweep was built.  The usual drive holds
the reference fully on and sweeps both target channels together (``x_t = w_t``, so
``g = sin(theta/2)^2``, ``dPhi_SLM = theta - pi``).  Because the target is calibrated
*against* the reference and the reference defines ``Phi = 0``, the fitted
``dPhi_comb`` IS the target pair's phase in the spectrum ``{Phi_k}``.

The pair amplitudes come from step 6 (``a = eta_ref``, ``b = eta_tgt``).  Physically
they should not diverge, so rather than boxing ``a`` and ``b`` independently (which
lets them trade off -- one collapsing while the other rails at its box, dragging
``a/b`` far from the step-6 ratio), the fit LOCKS the ratio ``a:b`` to
``eta_ref:eta_tgt`` and floats only a single shared scale ``s`` (``a = s eta_ref``,
``b = s eta_tgt``), boxed to ``[max(0, 1-frac), 1+frac]`` (``frac=1`` -> a 0..2x
common gain drift between step 6 and step 7 with the calibrated relative
efficiencies preserved).  It also folds in both pairs' step-6 single-beam response
(``a_x x + q_x x^2 + a_w w + q_w w^2`` per pair) as a FIXED additive background: the
fully-on reference contributes a constant, the swept target the ramp, so the fringe
never absorbs the single-beam ramp (which would bias dPhi_comb).  The three free
parameters ``s, dPhi_comb, d`` are solved by bounded nonlinear least squares
(:func:`fit_phase_ratio`); ``d`` should sit near 0 after per-row dark removal.
``frac = 0`` collapses the box: ``s`` is PINNED at exactly 1 (``a``/``b`` are the
step-6 etas verbatim) and only ``dPhi_comb, d`` float.
``a_at_bound``/``b_at_bound`` (both track the shared ``s``) warn when the scale box
actually bound.  An unconstrained closed-form variant is kept as :func:`fit_phase`
for diagnostics.

This module is fitting + IO only and geometry-general.  The instrument-facing
half (drive builders + the SLM/monitor sweep) lives in
:mod:`slm_module.tpa_phase_measure` for the pipeline, and in
``src/drafts/calib_step7_test.py`` for the offline driver.  Raw rows (one per
trial x point) are persisted as a CSV (:func:`write_phase_csv`) so a run can be
reloaded and re-fit offline (:func:`load_phase_csv`); the fitted spectrum is
persisted as a combined ``{step3, step6, step7}`` JSON
(:func:`save_comb_phase_json` / :func:`load_comb_phase_json`) that downstream
consumers read as their single input.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ======================================================================
# per-pair step-6 model  (background + eta, used to isolate the fringe)
# ======================================================================

@dataclass(frozen=True)
class PairModel:
    """One pair's step-6 fit: eta plus the single-beam / dark background terms.

    ``single_beam`` is the linear + quadratic single-channel response WITHOUT the
    dark offset (dark is shared between pairs and handled once, per run).
    """

    index: int
    eta: float
    a_x: float
    q_x: float
    a_w: float
    q_w: float
    d: float
    eta_err: float = 0.0

    def amplitude(self, x, w):
        """Field amplitude R = eta * sqrt(x*w) (= eta * sin(phi^x/2) sin(phi^w/2))."""
        x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
        w = np.clip(np.asarray(w, dtype=float), 0.0, 1.0)
        return self.eta * np.sqrt(x * w)

    def self_tpa(self, x, w):
        """Own two-photon pedestal R^2 = eta^2 * x * w."""
        return self.amplitude(x, w) ** 2

    def single_beam(self, x, w):
        """Linear + quadratic single-beam response a_x*x + q_x*x^2 + a_w*w + q_w*w^2."""
        x = np.asarray(x, dtype=float)
        w = np.asarray(w, dtype=float)
        return self.a_x * x + self.q_x * x**2 + self.a_w * w + self.q_w * w**2

    @classmethod
    def from_fit(cls, index: int, fit) -> "PairModel":
        """Build from a :class:`slm_module.tpa_pair.PairFit`."""
        p = fit.params
        return cls(
            index=index, eta=fit.eta, eta_err=fit.eta_err,
            a_x=p["a_x"][0], q_x=p["q_x"][0],
            a_w=p["a_w"][0], q_w=p["q_w"][0], d=p["d"][0],
        )

    @classmethod
    def from_json_channel(cls, ch: dict) -> "PairModel":
        """Build from one ``channels[]`` entry of a step-6 ``save_tpa_pair_json``."""
        fit = ch["fit"]
        p = fit["params"]
        return cls(
            index=int(ch["index"]), eta=float(fit["eta"]),
            eta_err=float(fit.get("eta_err", 0.0)),
            a_x=float(p["a_x"]["value"]), q_x=float(p["q_x"]["value"]),
            a_w=float(p["a_w"]["value"]), q_w=float(p["q_w"]["value"]),
            d=float(p["d"]["value"]),
        )


def load_pair_models(paths, *, layout=None) -> dict[int, PairModel]:
    """Load per-pair step-6 models from JSON summaries and/or raw CSVs.

    ``paths`` is one path or a sequence of paths.  ``.json`` files are read as
    step-6 ``save_tpa_pair_json`` output -- either a bare summary (``channels``
    at the top level) or a combined ``save_combined_json`` result
    (``{"step3": ..., "step6": {"channels": [...]}}``, so the pairs live under
    ``step6``).  Any other extension is treated as a raw step-6 CSV and re-fit
    through :mod:`slm_module.tpa_pair` (so the fit is byte-identical to step 6).
    ``layout`` is only needed for CSVs.  Later paths win on index collisions.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    models: dict[int, PairModel] = {}
    for path in paths:
        path = Path(path)
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            # combined result nests the step-6 payload under "step6"; a bare
            # save_tpa_pair_json summary has "channels" at the top level.
            if "channels" not in payload and isinstance(payload.get("step6"), dict):
                payload = payload["step6"]
            for ch in payload.get("channels", []):
                if ch.get("fit"):
                    m = PairModel.from_json_channel(ch)
                    models[m.index] = m
        else:
            from .tpa_pair import load_tpa_pair_csv
            result = load_tpa_pair_csv(path, layout=layout)
            for grid in result.channels:
                if grid.fit is not None:
                    models[grid.index] = PairModel.from_fit(grid.index, grid.fit)
    return models


# ======================================================================
# phase geometry  (intensity command <-> panel phase)
# ======================================================================

def phi_half(intensity) -> np.ndarray:
    """Half the panel phase depth of a channel, phi/2 = asin(sqrt(x)).

    ``x`` is the commanded normalised intensity (diffraction efficiency) in
    [0, 1]; the channel's field phase is exactly this value.
    """
    x = np.clip(np.asarray(intensity, dtype=float), 0.0, 1.0)
    return np.arcsin(np.sqrt(x))


def intensity_for_phase(phi_rad) -> np.ndarray:
    """Commanded intensity x = sin(phi/2)^2 for a target panel phase in [0, pi].

    Inverse of :func:`phi_half` on the reachable branch: ``phi = pi`` -> ``x = 1``.
    """
    phi = np.asarray(phi_rad, dtype=float)
    return np.sin(phi / 2.0) ** 2


def slm_phase_diff(x_t, w_t, x_r, w_r) -> np.ndarray:
    """dPhi_SLM = 1/2[(phi^x_t+phi^w_t) - (phi^x_r+phi^w_r)] from commanded intensities.

    Target (subscript t) minus reference (subscript r).  E.g. sweeping only ``w_t``
    against a fully-on reference (``x_t = x_r = w_r = 1``) gives
    ``phi_half(w_t) - pi/2``.
    """
    return phi_half(x_t) + phi_half(w_t) - phi_half(x_r) - phi_half(w_r)


# ======================================================================
# fit  (bounded nonlinear LS in a, b, dPhi_comb, d; a,b boxed to +/-frac*eta)
# ======================================================================

@dataclass
class PhaseFit:
    """Recovery of dPhi_comb (+ boxed pair amplitudes a, b) from the measured Y.

    Model, per fitted point (``g = sqrt(x_t w_t)`` the target field amplitude,
    ``dPhi_SLM`` from :func:`slm_phase_diff`)::

        Y = a^2 + b^2 g^2 + 2 a b g cos(dPhi_SLM + dPhi_comb)
          + fixed step-6 single-beam background + offset

    ``a``/``b`` are the reference/target pair amplitudes (step-6 eta_ref/eta_tgt);
    they float but are BOXED to ``+/- bound_frac`` of those etas.
    ``a_at_bound``/``b_at_bound`` flag a box constraint that bound.  ``bg0/bg1/bg2``
    hold the single-beam background written as a polynomial in ``g`` for the special
    case ``x_t = w_t`` (constant / linear / quadratic); kept only for reference.
    """

    dphi_comb: float           # radians, wrapped to (-pi, pi]
    dphi_comb_err: float
    a: float                   # reference amplitude R_1 = eta_ref (x_1 = w_1 = 1)
    a_err: float
    b: float                   # target amplitude scale eta_tgt
    b_err: float
    amp: float                 # interference amplitude 2 a b
    amp_err: float
    offset: float              # residual dark d (should be ~0)
    offset_err: float
    chi2_red: float
    dof: int
    birge: float
    r2: float
    eta_ref: float             # step-6 bound centre for a
    eta_tgt: float             # step-6 bound centre for b
    bound_frac: float          # box half-width as a fraction of eta (inf == free, 0 == s pinned at 1)
    a_at_bound: bool
    b_at_bound: bool
    bg0: float                 # step-6 single-beam background, constant
    bg1: float                 #   ... * g   (target single-beam, linear)
    bg2: float                 #   ... * g^2 (target single-beam, quadratic)
    # point arrays the fit ran on (kept for plotting)
    dphi_slm: np.ndarray = field(repr=False)     # dPhi_SLM per point (slm_phase_diff)
    g: np.ndarray = field(repr=False)            # target field amplitude = sqrt(x_t w_t)
    y: np.ndarray = field(repr=False)            # dark-subtracted measured Y
    sem: np.ndarray = field(repr=False)
    known: np.ndarray = field(repr=False)        # a^2 + b^2 g^2 + step-6 single-beam (no fringe/offset)
    y_pred: np.ndarray = field(repr=False)       # full model prediction
    residuals: np.ndarray = field(repr=False)
    # commanded intensities per fitted point (same order as g/dphi_slm), so the
    # plot can rebuild the swept geometry.  Optional -> None: old fits load.
    x_t: np.ndarray | None = field(default=None, repr=False)
    w_t: np.ndarray | None = field(default=None, repr=False)
    x_r: np.ndarray | None = field(default=None, repr=False)
    w_r: np.ndarray | None = field(default=None, repr=False)

    @property
    def dphi_comb_deg(self) -> float:
        return float(np.degrees(self.dphi_comb))


def fit_phase(
    dphi_slm: np.ndarray,
    g: np.ndarray,
    y: np.ndarray,
    sem: np.ndarray,
) -> PhaseFit:
    """Weighted LS fit of ``Y = a^2 + b^2 g^2 + 2 a b g cos(dPhi_SLM + dPhi_comb) + d``.

    ``g = sqrt(x_t w_t)`` is the target pair-field amplitude and ``dPhi_SLM`` the
    SLM phase difference (both per point).  The model is linear in the four
    coefficients of ``[1, g^2, g cos(dPhi_SLM), g sin(dPhi_SLM)]``::

        c0 = a^2 + d   c1 = b^2   c2 = 2ab cos(dPhi_comb)   c3 = -2ab sin(dPhi_comb)

    and the physical (a, b, dPhi_comb, d) follow in closed form (see module
    docstring).  Errors are covariance-propagated and Birge-scaled by
    ``sqrt(chi2/dof)`` when chi2/dof > 1.  The amplitude ``a`` is fixed by the
    interference + target self term, so it separates from the flat baseline (and
    hence from the residual dark ``d``).
    """
    dphi_slm = np.asarray(dphi_slm, dtype=float)
    g = np.asarray(g, dtype=float)
    y = np.asarray(y, dtype=float)
    sem = np.asarray(sem, dtype=float)

    cols = [np.ones_like(g), g**2, g * np.cos(dphi_slm), g * np.sin(dphi_slm)]
    A = np.column_stack(cols)

    Aw = A / sem[:, None]
    coeffs, *_ = np.linalg.lstsq(Aw, y / sem, rcond=None)
    cov = np.linalg.inv(Aw.T @ Aw)

    y_pred = A @ coeffs
    residuals = y - y_pred
    dof = max(len(y) - A.shape[1], 1)
    chi2_red = float(np.sum((residuals / sem) ** 2) / dof)
    birge = max(1.0, np.sqrt(chi2_red))
    cov = cov * birge**2

    c0, c1, c2, c3 = (float(coeffs[i]) for i in range(4))
    amp = float(np.hypot(c2, c3))                       # 2 a b
    b = float(np.sqrt(c1)) if c1 > 0 else 0.0
    a = amp / (2.0 * b) if b > 0 else float("nan")
    dphi = float(np.arctan2(-c3, c2))
    d = c0 - a**2 if np.isfinite(a) else float("nan")

    def _err(grad) -> float:
        gvec = np.asarray(grad, dtype=float)
        return float(np.sqrt(max(gvec @ cov @ gvec, 0.0)))

    if b > 0 and amp > 0:
        # gradients wrt (c0, c1, c2, c3)
        grad_b = [0.0, 1.0 / (2 * b), 0.0, 0.0]
        grad_a = [0.0, -a / (2 * c1), c2 / (2 * b * amp), c3 / (2 * b * amp)]
        grad_phi = [0.0, 0.0, c3 / amp**2, -c2 / amp**2]
        grad_d = [1.0, -2 * a * grad_a[1], -2 * a * grad_a[2], -2 * a * grad_a[3]]
        a_err, b_err = _err(grad_a), _err(grad_b)
        dphi_err, offset_err = _err(grad_phi), _err(grad_d)
    else:
        a_err = b_err = dphi_err = offset_err = float("nan")

    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    bg2_self = c1 if c1 > 0 else 0.0
    known = (a**2 if np.isfinite(a) else 0.0) + bg2_self * g**2
    return PhaseFit(
        dphi_comb=dphi, dphi_comb_err=dphi_err,
        a=a, a_err=a_err, b=b, b_err=b_err,
        amp=amp, amp_err=float("nan"),
        offset=d, offset_err=offset_err,
        chi2_red=chi2_red, dof=dof, birge=birge, r2=r2,
        eta_ref=a, eta_tgt=b, bound_frac=float("inf"),
        a_at_bound=False, b_at_bound=False,
        bg0=0.0, bg1=0.0, bg2=0.0,
        dphi_slm=dphi_slm, g=g, y=y, sem=sem, known=known,
        y_pred=y_pred, residuals=residuals,
    )


# ======================================================================
# result container + fit driver
# ======================================================================

@dataclass
class PhaseResult:
    """One target pair's phase sweep against the reference, plus its fit.

    Intensities are the canonical commanded values (``x = sin(phi/2)^2``); the
    ``_t`` columns are the swept target pair, the ``_r`` columns the fixed
    reference pair.
    """

    tgt_index: int
    ref_index: int
    # raw rows, one entry per (trial, point); kept for save + re-fit
    trial: np.ndarray = field(repr=False)
    x_t: np.ndarray = field(repr=False)
    w_t: np.ndarray = field(repr=False)
    x_r: np.ndarray = field(repr=False)
    w_r: np.ndarray = field(repr=False)
    voltage_mean_v: np.ndarray = field(repr=False)
    voltage_std_v: np.ndarray = field(repr=False)   # raw low-passed trace std (diagnostic)
    voltage_sem_v: np.ndarray = field(repr=False)    # SEM of the mean -> the fit weight
    # per-row dark measured at that row's trial start; subtracted per row before
    # averaging so per-trial dark drift is removed row-by-row (not as a constant)
    dark_v: np.ndarray = field(repr=False)
    tgt_model: PairModel | None = None
    ref_model: PairModel | None = None
    n_trials: int = 1
    fit: PhaseFit | None = None
    csv_path: str | None = None

    @property
    def dark(self) -> float:
        """Mean dark over all rows (for reporting / back-compat)."""
        return float(np.mean(self.dark_v)) if np.size(self.dark_v) else 0.0

    def per_trial_darks(self) -> np.ndarray:
        """The one dark value used for each trial (constant within a trial)."""
        out = []
        dark_v = np.asarray(self.dark_v)
        trial = np.asarray(self.trial)
        for t in range(self.n_trials):
            mask = trial == t
            if np.any(mask):
                out.append(float(dark_v[mask][0]))
        return np.asarray(out, dtype=float)


def _average_points(result: PhaseResult, dark_override: float | None = None):
    """Per-row dark-subtract, then average repeated trials per cell -> arrays + SEM.

    Each row's dark (measured at its trial's start) is removed BEFORE averaging,
    so per-trial dark drift is taken out row-by-row rather than as a single
    constant.  ``dark_override`` (a scalar) replaces the per-row dark uniformly.
    The returned ``y`` is therefore already dark-subtracted.

    ``sem`` is the across-trial standard error of the mean (std/sqrt(n)) for a
    cell measured more than once.  A cell measured only ONCE has no across-trial
    spread, so it falls back to that row's recorded ``voltage_sem_v`` (the DAQ's
    reported standard error of the mean) -- the real per-point uncertainty --
    which keeps the weighted fit meaningful with ``n_trials == 1`` (mirrors
    :func:`slm_module.tpa_pair.average_cells`; otherwise every cell would be
    floored to a bogus 1.0 V, flattening the fit).  Only cells with neither
    repeats nor a recorded SEM inherit the median positive SEM.
    """
    y_raw = np.asarray(result.voltage_mean_v, dtype=float)
    if dark_override is None:
        dark_row = np.asarray(result.dark_v, dtype=float)
    else:
        dark_row = np.full(y_raw.shape, float(dark_override))
    y_sub = y_raw - dark_row
    sem_row = np.asarray(result.voltage_sem_v, dtype=float)

    cells: dict[tuple, list[float]] = defaultdict(list)
    scells: dict[tuple, list[float]] = defaultdict(list)
    key = np.column_stack([result.x_t, result.w_t, result.x_r, result.w_r])
    for row, y, s in zip(key, y_sub, sem_row):
        rk = tuple(np.round(row, 9))
        cells[rk].append(float(y))
        scells[rk].append(float(s))

    keys, ys, sem = [], [], []
    for k, vals in sorted(cells.items()):
        arr = np.asarray(vals, dtype=float)
        keys.append(k)
        ys.append(arr.mean())
        if arr.size > 1:
            sem.append(arr.std(ddof=1) / np.sqrt(arr.size))    # across-trial spread
        else:
            rec = np.asarray(scells[k], dtype=float)            # recorded per-point SEM
            rec = rec[np.isfinite(rec) & (rec > 0)]
            sem.append(float(rec.mean()) if rec.size else np.nan)

    keys = np.asarray(keys, dtype=float)
    ys = np.asarray(ys, dtype=float)
    sem = np.asarray(sem, dtype=float)
    finite = sem[np.isfinite(sem) & (sem > 0)]
    floor = float(np.median(finite)) if finite.size else 1.0
    sem = np.where(np.isfinite(sem) & (sem > 0), sem, floor)
    return keys[:, 0], keys[:, 1], keys[:, 2], keys[:, 3], ys, sem


def fit_phase_ratio(
    dphi_slm: np.ndarray,
    g: np.ndarray,
    fixed_bg: np.ndarray,
    y: np.ndarray,
    sem: np.ndarray,
    *,
    eta_ref: float,
    eta_ref_err: float,
    eta_tgt: float,
    eta_tgt_err: float,
    bg0: float,
    bg1: float,
    bg2: float,
    frac: float = 1.0,
) -> PhaseFit:
    """Fit dPhi_comb with a,b LOCKED to the step-6 eta ratio via a shared scale.

    Model per row (``A := eta_ref``, ``B := eta_tgt`` are the fixed step-6
    amplitudes, ``s`` a single shared scale)::

        a = s A ,  b = s B
        Y = a^2 + b^2 g^2 + 2 a b g cos(dPhi_SLM + dPhi_comb) + fixed_bg + d
          = s^2 (A^2 + B^2 g^2 + 2 A B g cos(dPhi_SLM + dPhi_comb)) + fixed_bg + d

    Instead of boxing ``a`` and ``b`` independently (which let them trade off --
    ``a`` collapsing while ``b`` railed at its box, so ``a/b`` drifted far from the
    step-6 ratio), the *ratio* ``a:b`` is pinned to ``eta_ref:eta_tgt`` exactly and
    only the common scale ``s`` floats, boxed to ``[max(0,1-frac), 1+frac]`` (so
    ``frac=1`` allows a 0..2x overall gain drift between step 6 and step 7 while
    keeping the calibrated relative efficiencies).  Three free parameters
    (``s, dPhi_comb, d``); ``fixed_bg`` is the per-row step-6 single-beam
    background (dark already removed) that the amplitudes do NOT scale.  Solved as
    a bounded nonlinear least squares (:func:`scipy.optimize.least_squares`);
    errors are covariance-propagated from the Jacobian and Birge-scaled.

    ``frac = 0`` collapses the box: ``s`` is PINNED at exactly 1, so
    ``a = eta_ref`` and ``b = eta_tgt`` verbatim and only ``dPhi_comb`` and ``d``
    float (two free parameters) -- use when the step-6 amplitudes are trusted
    outright.  ``a_err``/``b_err`` are then 0 by construction, and the step-6 eta
    uncertainties are NOT propagated into ``dphi_comb_err``.

    ``bg0/bg1/bg2`` re-express the background as a polynomial in ``g`` for the
    special case ``x_t = w_t`` and are only stashed for reference.  ``eta_ref_err``/
    ``eta_tgt_err`` are accepted for API symmetry (reserved for a soft ratio
    prior) but do not enter this hard-ratio fit.
    """
    from scipy.optimize import least_squares

    dphi_slm = np.asarray(dphi_slm, dtype=float)
    g = np.asarray(g, dtype=float)
    fixed_bg = np.asarray(fixed_bg, dtype=float)
    y = np.asarray(y, dtype=float)
    sem = np.asarray(sem, dtype=float)

    A, B = float(eta_ref), float(eta_tgt)
    fix_scale = frac == 0.0                                   # box collapsed -> s pinned at 1

    def predict(s, dphi, d):
        s2 = s * s
        return (s2 * (A * A + B * B * g * g)
                + 2.0 * s2 * A * B * g * np.cos(dphi_slm + dphi) + fixed_bg + d)

    # phase seed: linear projection of the (background + self) subtracted signal
    w = 1.0 / sem**2
    r0 = y - fixed_bg - A**2 - B**2 * g**2
    P = float(np.sum(w * r0 * g * np.cos(dphi_slm)))
    Q = float(np.sum(w * r0 * g * np.sin(dphi_slm)))
    dphi0 = float(np.arctan2(-Q, P))

    if fix_scale:                  # 2 free params (dPhi_comb, d); a = A, b = B verbatim
        def resid(p):
            return (predict(1.0, p[0], p[1]) - y) / sem

        sol = least_squares(resid, [dphi0, 0.0], max_nfev=20000)
        s, dphi, d = 1.0, float(sol.x[0]), float(sol.x[1])
    else:                          # 3 free params; s boxed to [max(0, 1-frac), 1+frac]
        def resid(p):
            return (predict(p[0], p[1], p[2]) - y) / sem

        lo = [max(0.0, 1.0 - frac), -np.inf, -np.inf]
        hi = [1.0 + frac, np.inf, np.inf]
        sol = least_squares(resid, [1.0, dphi0, 0.0], bounds=(lo, hi), max_nfev=20000)
        s, dphi, d = (float(v) for v in sol.x)
    dphi = float(np.arctan2(np.sin(dphi), np.cos(dphi)))      # wrap to (-pi, pi]
    a, b = s * A, s * B                                       # ratio locked to A:B

    y_pred = predict(s, dphi, d)
    residuals = y - y_pred
    n_free = 2 if fix_scale else 3
    dof = max(len(y) - n_free, 1)
    chi2_red = float(np.sum((residuals / sem) ** 2) / dof)
    birge = max(1.0, np.sqrt(chi2_red))

    # covariance from the weighted Jacobian at the solution (resid already /sem)
    try:
        cov = np.linalg.inv(sol.jac.T @ sol.jac) * birge**2
    except np.linalg.LinAlgError:
        cov = np.full((n_free, n_free), np.nan)
    if fix_scale:
        s_err = 0.0                                           # s is a constant, not fitted
        dphi_err = float(np.sqrt(max(cov[0, 0], 0.0)))
        offset_err = float(np.sqrt(max(cov[1, 1], 0.0)))
        a_at_bound = b_at_bound = False
    else:
        s_err = float(np.sqrt(max(cov[0, 0], 0.0)))
        dphi_err = float(np.sqrt(max(cov[1, 1], 0.0)))
        offset_err = float(np.sqrt(max(cov[2, 2], 0.0)))

        def _hit(val, lower, upper) -> bool:
            span = max(abs(upper - lower), 1e-30)
            return bool(val - lower <= 1e-6 * span or upper - val <= 1e-6 * span)

        # a and b move together, so both share the single scale's bound state
        a_at_bound = b_at_bound = _hit(s, lo[0], hi[0])
    a_err, b_err = A * s_err, B * s_err                       # fully correlated via s
    amp = 2.0 * a * b                                         # = 2 s^2 A B
    amp_err = float(abs(4.0 * s * A * B) * s_err)             # d(amp)/ds = 4 s A B

    known = a * a + b * b * g * g + fixed_bg
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return PhaseFit(
        dphi_comb=dphi, dphi_comb_err=dphi_err,
        a=a, a_err=a_err, b=b, b_err=b_err,
        amp=amp, amp_err=amp_err,
        offset=d, offset_err=offset_err,
        chi2_red=chi2_red, dof=dof, birge=birge, r2=r2,
        eta_ref=eta_ref, eta_tgt=eta_tgt, bound_frac=frac,
        a_at_bound=a_at_bound, b_at_bound=b_at_bound,
        bg0=bg0, bg1=bg1, bg2=bg2,
        dphi_slm=dphi_slm, g=g, y=y, sem=sem, known=known,
        y_pred=y_pred, residuals=residuals,
    )


def fit_result(
    result: PhaseResult,
    tgt_model: PairModel,
    ref_model: PairModel,
    *,
    dark: float | None = None,
    frac: float | None = None,
    single_beam_bg: bool = False,
) -> PhaseFit:
    """Fit ``a``, ``b`` and ``dPhi_comb`` to the dark-subtracted Y.

    Per-row dark-subtracts and averages repeated trials per point (see
    :func:`_average_points`), then fits ``Y = a^2 + b^2 g^2 +
    2ab g cos(dPhi_SLM + dPhi_comb) + d``.

    ``frac`` selects how ``a``/``b`` are handled:

    * ``None`` (default) -- unconstrained closed-form fit (:func:`fit_phase`).
      This is the numerically clean, well-conditioned fit for ``dPhi_comb``.
    * a number -- lock the ratio ``a:b`` to the step-6 ``eta_ref:eta_tgt`` and
      float only a shared scale ``s`` boxed to ``+/- frac`` about 1, via the
      ratio-locked nonlinear fit (:func:`fit_phase_ratio`).  ``frac=0`` pins the
      scale exactly (``s = 1``: ``a``/``b`` ARE the step-6 etas; only
      ``dPhi_comb`` and ``d`` float).  ``single_beam_bg`` then also folds in both
      pairs' step-6 single-beam response as a FIXED additive background: the
      reference (held fully on) contributes a constant, the swept target
      contributes the ``~g`` ramp, so ``s``/``dPhi_comb`` are not forced to
      absorb it.

    ``dark`` (scalar) overrides the per-row dark uniformly.
    """
    x_t, w_t, x_r, w_r, y, sem = _average_points(result, dark_override=dark)

    g = np.sqrt(np.clip(x_t * w_t, 0.0, None))         # target field amplitude
    dphi_slm = slm_phase_diff(x_t, w_t, x_r, w_r)       # SLM phase difference

    result.tgt_model = tgt_model
    result.ref_model = ref_model
    if frac is None:
        fit = fit_phase(dphi_slm, g, y, sem)
    else:
        if single_beam_bg:
            # step-6 single-beam of both pairs as a fixed background (dark already out)
            fixed_bg = np.asarray(
                ref_model.single_beam(x_r, w_r) + tgt_model.single_beam(x_t, w_t),
                dtype=float,
            )
            bg0 = float(ref_model.single_beam(1.0, 1.0))
            bg1 = float(tgt_model.a_x + tgt_model.a_w)
            bg2 = float(tgt_model.q_x + tgt_model.q_w)
        else:
            fixed_bg = np.zeros_like(g)
            bg0 = bg1 = bg2 = 0.0

        fit = fit_phase_ratio(
            dphi_slm, g, fixed_bg, y, sem,
            eta_ref=ref_model.eta, eta_ref_err=ref_model.eta_err,
            eta_tgt=tgt_model.eta, eta_tgt_err=tgt_model.eta_err,
            bg0=bg0, bg1=bg1, bg2=bg2, frac=frac,
        )

    # stash the per-point commanded intensities so the plot can rebuild the sweep
    fit.x_t, fit.w_t, fit.x_r, fit.w_r = x_t, w_t, x_r, w_r
    result.fit = fit
    return fit


def swap_invariance(result: PhaseResult):
    """Table-2 diagnostic: |Z(x=a,w=b) - Z(x=b,w=a)| for each swap pair.

    The test runs on the CLEAN interference term, not raw Y, so the fitted self
    terms AND the step-6 single-beam background are removed first::

        Z(x,w) = Y(x,w) - [a^2 + b^2 (x w) + sb_ref + sb_tgt] - d
               = 2 a b sqrt(x w) cos(dPhi_SLM + dPhi_comb)

    Under the bilinear model the target amplitude ``sqrt(x w)`` and ``dPhi_SLM``
    (a channel *sum*) are swap-symmetric, so ``Z`` must be too; a residual well
    above the combined SEM flags a genuine channel asymmetry (unequal per-channel
    phase/amplitude law or crosstalk).  Returns ``(x_t, w_t, z, z_swapped,
    abs_diff, sem)`` for the off-diagonal cells.  Falls back to raw Y only if the
    fit is not attached.  ``fit.known`` already carries ``a^2 + b^2 g^2 + sb``.
    """
    x_t, w_t, x_r, w_r, y, sem = _average_points(result)   # y already dark-subtracted
    fit = result.fit
    if fit is not None and fit.known is not None and np.isfinite(fit.a):
        # clean interference: strip fitted self terms + step-6 single-beam + d
        sig = y - fit.known - fit.offset
    else:
        sig = y

    lut = {(round(a, 9), round(b, 9)): (zz, ss)
           for a, b, zz, ss in zip(x_t, w_t, sig, sem)}
    out = []
    for a, b, zz, ss in zip(x_t, w_t, sig, sem):
        if round(a, 9) == round(b, 9):
            continue
        swapped = lut.get((round(b, 9), round(a, 9)))
        if swapped is None:
            continue
        z_sw, s_sw = swapped
        out.append((float(a), float(b), float(zz), float(z_sw),
                    abs(float(zz) - float(z_sw)), float(np.hypot(ss, s_sw))))
    return out


# ======================================================================
# persistence
# ======================================================================

_CSV_HEADER = [
    "trial", "tgt_index", "ref_index",
    "phi_xt_deg", "phi_wt_deg", "x_t", "w_t", "x_r", "w_r",
    "dark_v", "voltage_mean_v", "voltage_std_v", "voltage_sem_v",
]


def write_phase_csv(result: PhaseResult, path: str | Path) -> str:
    """Raw rows: one line per (trial, point).  Round-trips via load.

    ``phi_xt_deg`` / ``phi_wt_deg`` are the target channel phases (for readable
    comparison with the sweep tables); the fit reloads from the canonical
    intensities.  ``dark_v`` is the per-row dark (that row's trial start) used for
    per-row subtraction; the run's mean dark is also stashed as a trailing comment.
    """
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_CSV_HEADER)
        for t, x_t, w_t, x_r, w_r, dark_v, mean_v, std_v, sem_v in zip(
            result.trial, result.x_t, result.w_t, result.x_r, result.w_r,
            result.dark_v, result.voltage_mean_v, result.voltage_std_v,
            result.voltage_sem_v,
        ):
            phi_xt = np.degrees(2.0 * float(phi_half(x_t)))
            phi_wt = np.degrees(2.0 * float(phi_half(w_t)))
            writer.writerow(
                [int(t), result.tgt_index, result.ref_index,
                 f"{phi_xt:.4g}", f"{phi_wt:.4g}",
                 f"{x_t:.6g}", f"{w_t:.6g}", f"{x_r:.6g}", f"{w_r:.6g}",
                 f"{dark_v:.9g}", f"{mean_v:.9g}", f"{std_v:.9g}", f"{sem_v:.9g}"]
            )
    with open(out, "a", newline="", encoding="utf-8") as f:
        f.write(f"# dark_mean_v,{result.dark:.9g}\n")
    result.csv_path = str(out)
    return str(out)


def load_phase_csv(
    path: str | Path,
    tgt_model: PairModel,
    ref_model: PairModel,
    *,
    dark: float | None = None,
    frac: float | None = None,
    single_beam_bg: bool = False,
    only_tgt: int | None = None,
) -> PhaseResult:
    """Load a raw phase-sweep CSV and re-fit dPhi_comb with the given step-6 models.

    The per-row ``dark_v`` column is used when present; otherwise the scalar
    ``# dark_mean_v`` comment, then the step-6 mean, is filled for every row.
    ``dark`` (scalar) overrides all of them uniformly.

    ``only_tgt`` keeps only rows whose ``tgt_index`` matches it; a collected file
    records every target pair vs the shared reference in one CSV, so pass the pair
    to fit (default None loads every row, for a single-target CSV).

    ``frac``/``single_beam_bg`` are forwarded to :func:`fit_result`: ``frac=None``
    (default) keeps the unconstrained closed-form fit; a number locks ``a:b`` to
    the step-6 ``eta_ref:eta_tgt`` ratio and floats a shared scale boxed to
    ``+/- frac`` (``frac=0`` pins ``a``/``b`` to the step-6 etas exactly).
    ``single_beam_bg`` additionally folds in both pairs' step-6 single-beam
    response as a fixed background.
    """
    file_dark: float | None = None
    with open(Path(path), newline="", encoding="utf-8") as f:
        for raw in f:
            if raw.startswith("#"):
                parts = raw.lstrip("#").strip().split(",")
                if len(parts) == 2 and parts[0].strip() == "dark_mean_v":
                    file_dark = float(parts[1])

    rows: list[tuple[int, float, float, float, float, float, float, float, float | None]] = []
    tgt_index, ref_index = tgt_model.index, ref_model.index
    with open(Path(path), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(line for line in f if not line.startswith("#")):
            row_tgt = int(float(row.get("tgt_index", tgt_index)))
            if only_tgt is not None and row_tgt != only_tgt:
                continue  # skip the other targets in a multi-pair CSV
            dv = row.get("dark_v")
            std_v = float(row.get("voltage_std_v", "nan") or "nan")
            sem_v = float(row["voltage_sem_v"])  # the fit weight; every CSV records it
            rows.append((
                int(float(row.get("trial", 0))),
                float(row["x_t"]), float(row["w_t"]),
                float(row["x_r"]), float(row["w_r"]),
                float(row["voltage_mean_v"]),
                std_v,
                sem_v,
                float(dv) if dv not in (None, "") else None,
            ))
            tgt_index = row_tgt
            ref_index = int(float(row.get("ref_index", ref_index)))

    trials = np.array([r[0] for r in rows], dtype=int)
    scalar_dark = (
        dark if dark is not None
        else file_dark if file_dark is not None
        else 0.5 * (tgt_model.d + ref_model.d)
    )
    # per-row dark: CSV column if present (and not overridden), else the scalar
    if dark is None and all(r[8] is not None for r in rows) and rows:
        dark_v = np.array([r[8] for r in rows], dtype=float)
    else:
        dark_v = np.full(len(rows), float(scalar_dark), dtype=float)

    result = PhaseResult(
        tgt_index=tgt_index, ref_index=ref_index,
        trial=trials,
        x_t=np.array([r[1] for r in rows], dtype=float),
        w_t=np.array([r[2] for r in rows], dtype=float),
        x_r=np.array([r[3] for r in rows], dtype=float),
        w_r=np.array([r[4] for r in rows], dtype=float),
        voltage_mean_v=np.array([r[5] for r in rows], dtype=float),
        voltage_std_v=np.array([r[6] for r in rows], dtype=float),
        voltage_sem_v=np.array([r[7] for r in rows], dtype=float),
        dark_v=dark_v,
        n_trials=int(trials.max()) + 1 if trials.size else 1,
        csv_path=str(Path(path).resolve()),
    )
    fit_result(result, tgt_model, ref_model, frac=frac, single_beam_bg=single_beam_bg)
    return result


def phase_fit_payload(fit: PhaseFit) -> dict:
    """JSON-ready summary of one dPhi_comb fit (shared by every phase saver)."""
    return {
        "dphi_comb_rad": fit.dphi_comb,
        "dphi_comb_deg": fit.dphi_comb_deg,
        "dphi_comb_err_rad": fit.dphi_comb_err,
        "dphi_comb_err_deg": float(np.degrees(fit.dphi_comb_err)),
        "a": fit.a,                 # reference amplitude R_1 (~ eta_ref)
        "a_err": fit.a_err,
        "a_at_bound": fit.a_at_bound,
        "b": fit.b,                 # target amplitude scale (~ eta_tgt)
        "b_err": fit.b_err,
        "b_at_bound": fit.b_at_bound,
        "eta_ref": fit.eta_ref,     # step-6 box centre for a
        "eta_tgt": fit.eta_tgt,     # step-6 box centre for b
        "bound_frac": fit.bound_frac,
        "amp_2ab": fit.amp,         # interference amplitude 2ab
        "amp_2ab_err": fit.amp_err,
        "dark_resid_v": fit.offset,  # residual DC after per-row dark subtraction
        "dark_resid_err_v": fit.offset_err,
        "chi2_red": fit.chi2_red,
        "dof": fit.dof,
        "birge": fit.birge,
        "r2": fit.r2,
    }


def save_phase_json(result: PhaseResult, path: str | Path) -> str:
    """Human-readable dPhi_comb summary (radians + degrees) and fit quality."""
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fit = result.fit
    per_trial = result.per_trial_darks()
    payload = {
        "tgt_index": result.tgt_index,
        "ref_index": result.ref_index,
        "dark_mean_v": result.dark,
        "dark_drift_std_v": float(per_trial.std(ddof=1)) if per_trial.size > 1 else 0.0,
        "n_trials": result.n_trials,
        "tgt_eta": result.tgt_model.eta if result.tgt_model else None,
        "ref_eta": result.ref_model.eta if result.ref_model else None,
        "fit": None if fit is None else phase_fit_payload(fit),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(out)


def save_comb_phase_json(
    fits: dict[tuple[int, str], PhaseFit],
    step6_path: str | Path,
    path: str | Path,
    *,
    ref_index: int,
    csv_path: str | None = None,
    single_beam_bg: bool | None = None,
) -> str:
    """Combined step-7 result JSON: ``{"step3": ..., "step6": ..., "step7": ...}``.

    ``fits`` maps ``(tgt_index, method)`` to a fitted :class:`PhaseFit` --
    ``method`` is the free-form amplitude-handling label the driver used
    (e.g. ``"bounded"`` / ``"fix"``); a target may carry one entry per method.
    The ``step3`` and ``step6`` payloads are carried over VERBATIM from the
    combined step-6 JSON at ``step6_path``, so this one file is a superset:
    channel layout (step3) + per-pair eta / single-beam / dark models (step6)
    + the comb-phase spectrum ``{Phi_k}`` vs ``ref_index`` (step7).  Downstream
    consumers (e.g. a multi-pair forward-model check) need nothing else.
    """
    payload6 = json.loads(Path(step6_path).read_text(encoding="utf-8"))
    if "step6" not in payload6:                        # bare save_tpa_pair_json summary
        payload6 = {"step3": payload6.get("step3"), "step6": payload6}
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step3": payload6.get("step3"),
        "step6": payload6.get("step6"),
        "step7": {
            "ref_index": int(ref_index),
            "csv": csv_path,
            "single_beam_bg": single_beam_bg,
            "step6_json": str(Path(step6_path).resolve()),
            "channels": [
                {"tgt_index": int(k), "method": str(m), "fit": phase_fit_payload(f)}
                for (k, m), f in sorted(fits.items())
            ],
        },
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(out)


def load_comb_phase_json(
    path: str | Path, *, method: str | None = None
) -> tuple[int, dict[int, dict]]:
    """Load a combined step-7 JSON -> ``(ref_index, {tgt_index: channel entry})``.

    Each returned entry is the stored ``{"tgt_index", "method", "fit": {...}}``
    dict (``fit["dphi_comb_rad"]`` is the comb phase vs the reference, which
    defines ``Phi = 0``).  ``method`` picks among multiple stored fits per
    target (e.g. ``"bounded"`` / ``"fix"``); with ``None`` a target must have
    exactly one stored fit, otherwise the choice is ambiguous and raises.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    step7 = payload.get("step7")
    if not isinstance(step7, dict):
        raise ValueError(f"{path} has no 'step7' section (not a combined step-7 result)")
    phases: dict[int, dict] = {}
    seen: dict[int, list[str]] = defaultdict(list)
    for ch in step7.get("channels", []):
        if not ch.get("fit"):
            continue
        k = int(ch["tgt_index"])
        m = str(ch.get("method", ""))
        seen[k].append(m)
        if method is None or m == method:
            if method is None and k in phases:
                raise ValueError(
                    f"pair {k} has several stored fits ({seen[k]}); pass method="
                )
            phases[k] = ch
    if method is not None:
        missing = [k for k, ms in seen.items() if k not in phases]
        if missing:
            raise ValueError(
                f"no '{method}' fit stored for pair(s) {sorted(missing)} in {path}"
            )
    return int(step7["ref_index"]), phases


__all__ = [
    "PairModel",
    "PhaseFit",
    "PhaseResult",
    "load_pair_models",
    "phi_half",
    "intensity_for_phase",
    "slm_phase_diff",
    "fit_phase",
    "fit_phase_ratio",
    "fit_result",
    "swap_invariance",
    "write_phase_csv",
    "load_phase_csv",
    "phase_fit_payload",
    "save_phase_json",
    "save_comb_phase_json",
    "load_comb_phase_json",
]
