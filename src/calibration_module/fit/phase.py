"""Comb-phase (dPhi_comb) calibration of each pair against a common reference.

Step 6 calibrates each pair's TPA efficiency ``eta`` one pair at a time, so
absolute optical phase never enters.  This step drives TWO pairs at once and
recovers, from their coherent TPA interference, the fixed comb phase a target
pair carries relative to a reference pair, which defines ``Phi = 0``.

In:  a :class:`PhaseResult` -- raw rows from
     :mod:`calibration_module.measure.phase_v2`, or a CSV through
     :func:`load_phase_csv` -- plus one :class:`PairModel` per pair from step 6.
Out: a :class:`PhaseFit` per target, stored as the run's ``step7`` section by
     :func:`save_step7_section`.  The rows round-trip through
     :func:`write_meas_csv` for an offline re-fit.

The model.  A channel commanded at normalised intensity ``x`` in [0, 1] sits at
panel phase ``phi = 2 asin(sqrt(x))`` with field ``sqrt(x) exp(i phi/2)``, so
``x = 1`` reaches ``phi = pi``, one half turn.  Per fitted point::

    g        = sqrt(x_t w_t)                        target field amplitude
    dPhi_SLM = phi_half(x_t) + phi_half(w_t) - phi_half(x_r) - phi_half(w_r)
    Y        = a^2 + b^2 g^2 + 2 a b g cos(dPhi_SLM + dPhi_comb)
               + step-6 single-beam background + d

``a``/``b`` are the reference/target amplitudes (step 6's etas).  ``g`` and
``dPhi_SLM`` come per row from the commanded intensities, so the fit is
geometry-general.  The SLM phase ADDS to the comb phase
(:func:`fringe_arg`): the one convention in this project, used by every fitter,
so a stored ``dPhi_comb`` IS ``Phi_k`` with no sign for a consumer to undo.

Three fitters, tightening in turn: :func:`fit_phase` (all four parameters free,
closed form), :func:`fit_phase_ratio` (``a:b`` locked to the step-6 eta ratio,
one shared scale floating) and :func:`fit_phase_fixed` -- step 7 v2 -- where
``dPhi_comb`` is the only free parameter, so a step-6 error shows up as a bad
``R^2`` instead of hiding in a nuisance term.

Fitting and IO only, no instrument: the sweep is in
:mod:`calibration_module.measure.phase_v2`, the runner in
``steps/calib_step7_v2.py``.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .outliers import BURST_MAX_V, flag_burst_traces, flag_over_range
from .sigma import STD_FLOOR_V, FloorSource, floor_std, measured_floor

# ======================================================================
# per-pair step-6 model  (background + eta, used to isolate the fringe)
# ======================================================================

@dataclass(frozen=True)
class PairModel:
    """One pair's step-6 fit: eta plus the single-beam / dark background terms.

    ``single_beam`` excludes the dark offset -- that is shared between pairs and
    handled once per run.
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
    def from_pair_v2(cls, fit) -> "PairModel":
        """Build from a :class:`calibration_module.fit.pair_v2.PairV2Fit`.

        The v2 estimator drops the quadratic columns, so ``q_x``/``q_w`` are
        absent from ``fit.bg`` and default to an exact 0.0 here, as on disk.
        """
        p = {k: v[0] for k, v in fit.bg.items()}
        return cls(
            index=fit.index, eta=fit.eta, eta_err=fit.eta_err,
            a_x=p["a_x"], q_x=p.get("q_x", 0.0),
            a_w=p["a_w"], q_w=p.get("q_w", 0.0), d=p["d"],
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

    In:  ``paths``, one or a sequence.  ``.json`` is step-6 output (a bare
         summary or a combined result); any other extension is a raw step-6 CSV,
         re-fit through :mod:`slm_module.tpa_pair`, for which ``layout`` is
         needed.
    Out: ``{pair index: PairModel}``.  Later paths win on collisions.
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
            from .pair import load_tpa_pair_csv
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


def fringe_arg(dphi_slm, dphi_comb) -> np.ndarray:
    """Cosine argument of the interference term: ``dPhi_SLM + dPhi_comb``.

    The SLM phase ADDS, a pair's field being
    ``eta sqrt(x w) exp(i[phi_half(x) + phi_half(w) + Phi_k])``.  Every fitter
    fits this argument, so a stored ``dPhi_comb`` IS ``Phi_k`` as
    :func:`calibration_module.fit.verify.predict` uses it.
    """
    dphi_slm = np.asarray(dphi_slm, dtype=float)
    return dphi_slm + dphi_comb


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
    """Recovery of dPhi_comb (+ pair amplitudes a, b) from the measured Y.

        Y = a^2 + b^2 g^2 + 2 a b g cos(dPhi_SLM + dPhi_comb)
          + fixed step-6 single-beam background + offset

    with ``g = sqrt(x_t w_t)`` and ``dPhi_SLM`` from :func:`slm_phase_diff`.
    ``a``/``b`` are the reference/target amplitudes, boxed to ``+/- bound_frac``
    where they float; ``a_at_bound``/``b_at_bound`` flag a box that bound.
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
    std: np.ndarray = field(repr=False)      # trace spread (+) STD_FLOOR_V: what the fit weighted by
    known: np.ndarray = field(repr=False)        # a^2 + b^2 g^2 + step-6 single-beam (no fringe/offset)
    y_pred: np.ndarray = field(repr=False)       # full model prediction
    residuals: np.ndarray = field(repr=False)
    # commanded intensities per fitted point (same order as g/dphi_slm), so the
    # plot can rebuild the swept geometry.  Optional -> None: old fits load.
    x_t: np.ndarray | None = field(default=None, repr=False)
    w_t: np.ndarray | None = field(default=None, repr=False)
    x_r: np.ndarray | None = field(default=None, repr=False)
    w_r: np.ndarray | None = field(default=None, repr=False)
    # Pinned step-6 amplitudes: their error, propagated into the phase.  Set by
    # fit_phase_fixed only -- a/b do not float there, so their step-6
    # uncertainty cannot appear in dphi_comb_err; it arrives through these.
    eta_ref_err: float = 0.0
    eta_tgt_err: float = 0.0
    dphi_deta_ref: float = 0.0        # d(dPhi_comb)/d(eta_ref), rad per unit eta
    dphi_deta_tgt: float = 0.0        # d(dPhi_comb)/d(eta_tgt)
    # The same sensitivities PER POINT, on the model: dm_j/d(eta) in volts per
    # unit eta.  They turn the step-6 eta error into :attr:`std_total`.  None
    # when a/b floated.
    dm_deta_ref: np.ndarray | None = field(default=None, repr=False)
    dm_deta_tgt: np.ndarray | None = field(default=None, repr=False)

    @property
    def dphi_comb_deg(self) -> float:
        return float(np.degrees(self.dphi_comb))

    @property
    def dphi_comb_err_eta(self) -> float:
        """Phase error from the PINNED step-6 etas alone (rad).

        Zero for every fit that floats a/b -- there the amplitude error is
        already inside dphi_comb_err via the covariance.
        """
        return float(np.hypot(self.dphi_deta_ref * self.eta_ref_err,
                              self.dphi_deta_tgt * self.eta_tgt_err))

    @property
    def dphi_comb_err_total(self) -> float:
        """Fringe noise and the pinned-eta term in quadrature (rad).

        The single-pair error bar.  Across pairs the eta_ref part is correlated,
        eta_ref being shared -- see :func:`phase_covariance`.
        """
        return float(np.hypot(self.dphi_comb_err, self.dphi_comb_err_eta))

    @property
    def std_model_eta(self) -> np.ndarray:
        """Per-point MODEL error from the pinned step-6 etas (volts).

            sigma_m,j = hypot(dm_j/d(eta_ref) * eta_ref_err,
                              dm_j/d(eta_tgt) * eta_tgt_err)

        All zeros when the fit floated a/b, or step 6 quoted no eta error.
        """
        if self.dm_deta_ref is None or self.dm_deta_tgt is None:
            return np.zeros_like(self.std)
        return np.hypot(self.dm_deta_ref * self.eta_ref_err,
                        self.dm_deta_tgt * self.eta_tgt_err)

    @property
    def std_total(self) -> np.ndarray:
        """Per-point sigma to judge a residual against: measurement (+) eta.

        ``std`` is how well the POINT is known; with a/b pinned the CURVE also
        carries step 6's eta error, which this adds.  Display only:
        :func:`fit_phase_fixed` still WEIGHTS by ``std``, the eta term being one
        number per pair.  Folding it into the weights would misdescribe it as
        independent noise.  For the same reason :attr:`pulls` move together and
        should be read as a set.
        """
        return np.hypot(self.std, self.std_model_eta)

    @property
    def pulls(self) -> np.ndarray:
        """``residuals / std_total`` -- the residual in sigmas of the RIGHT sigma."""
        return self.residuals / self.std_total

    def fringe_arg(self, dphi_slm=None):
        """This fit's cosine argument, ``dPhi_SLM + dPhi_comb``."""
        d = self.dphi_slm if dphi_slm is None else dphi_slm
        return fringe_arg(d, self.dphi_comb)


def fit_phase(
    dphi_slm: np.ndarray,
    g: np.ndarray,
    y: np.ndarray,
    std: np.ndarray,
) -> PhaseFit:
    """Weighted LS fit of ``Y = a^2 + b^2 g^2 + 2ab g cos(dPhi_SLM+dPhi_comb) + d``.

    In:  ``dphi_slm``, ``g``, ``y`` and ``std``, all per point.
    Out: a :class:`PhaseFit`, all four parameters free.

    Closed form: linear in the coefficients of
    ``[1, g^2, g cos(dPhi_SLM), g sin(dPhi_SLM)]``.
    """
    dphi_slm = np.asarray(dphi_slm, dtype=float)
    g = np.asarray(g, dtype=float)
    y = np.asarray(y, dtype=float)
    std = np.asarray(std, dtype=float)

    cols = [np.ones_like(g), g**2, g * np.cos(dphi_slm), g * np.sin(dphi_slm)]
    A = np.column_stack(cols)

    Aw = A / std[:, None]
    coeffs, *_ = np.linalg.lstsq(Aw, y / std, rcond=None)
    cov = np.linalg.inv(Aw.T @ Aw)

    y_pred = A @ coeffs
    residuals = y - y_pred

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
        r2=r2,
        eta_ref=a, eta_tgt=b, bound_frac=float("inf"),
        a_at_bound=False, b_at_bound=False,
        bg0=0.0, bg1=0.0, bg2=0.0,
        dphi_slm=dphi_slm, g=g, y=y, std=std, known=known,
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
    voltage_std_v: np.ndarray = field(repr=False)   # low-passed trace std -> the fit weight
    # per-row dark measured at that row's trial start; subtracted per row before
    # averaging so per-trial dark drift is removed row-by-row (not as a constant)
    dark_v: np.ndarray = field(repr=False)
    tgt_model: PairModel | None = None
    ref_model: PairModel | None = None
    n_trials: int = 1
    fit: PhaseFit | None = None
    csv_path: str | None = None
    # The PMT thermistor beside each row, as written: strings, blank where the
    # 2400 was absent, None when unlogged.  Here only so the CSV round-trips.
    pmt_ohm: np.ndarray | None = field(default=None, repr=False)
    pmt_c: np.ndarray | None = field(default=None, repr=False)
    # Burst acquisitions dropped on load, one dict each; empty on a clean
    # sweep.  Kept so the step can warn: a silent clean-up hides a bench fault.
    dropped_bursts: list[dict] = field(default_factory=list, repr=False)
    #: Sigma floor these points were weighted by, volts, and where it came from
    #: (a :class:`~calibration_module.fit.sigma.FloorSource` value).  Carried so
    #: the report and the step-7 section quote the floor actually used.
    sigma_floor_v: float = STD_FLOOR_V
    sigma_floor_source: str = FloorSource.FALLBACK.value
    #: Darks replaced because the recorded one was unusable, ``{tgt: {...}}``.
    dark_overrides: dict = field(default_factory=dict, repr=False)

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


def burst_mask(trace_stds, groups=None) -> np.ndarray:
    """Which acquisitions are bursts, grouped by target.

    In:  ``groups``, the per-row target (None = one sweep).  Judged per TARGET,
         not per file: a collected CSV holds every target against the shared
         reference, and a dim pair's honest points look like nothing beside a
         bright pair's.
    Out: the mask.  A group that is ALL bursts is left alone.
    """
    s = np.asarray(trace_stds, dtype=float)
    mask = np.zeros(s.size, dtype=bool)
    if groups is None:
        groups = [0] * s.size
    groups = list(groups)
    for g in set(groups):
        idx = [i for i, t in enumerate(groups) if t == g]
        sub = flag_burst_traces(s[idx])
        if sub.all():
            continue
        for i, bad in zip(idx, sub):
            mask[i] = bad
    return mask


def drop_bursts(result: PhaseResult) -> list[dict]:
    """Strip burst acquisitions from a PhaseResult in place; return what went.

    For the live path; :func:`load_phase_csv` shares :func:`burst_mask` instead.
    Appends to :attr:`PhaseResult.dropped_bursts` and applies the absolute
    ceiling (:data:`~calibration_module.fit.outliers.BURST_MAX_V`).  The DARK is
    not reached from here -- see :func:`guard_darks`.
    """
    keep = ~(burst_mask(result.voltage_std_v)
             | flag_over_range(result.voltage_mean_v))
    if keep.all():
        return []
    gone = [
        {"trial": int(result.trial[i]), "x_t": float(result.x_t[i]),
         "w_t": float(result.w_t[i]),
         "voltage_mean_v": float(result.voltage_mean_v[i]),
         "voltage_std_v": float(result.voltage_std_v[i])}
        for i in np.flatnonzero(~keep)
    ]
    for name in ("trial", "x_t", "w_t", "x_r", "w_r", "voltage_mean_v",
                 "voltage_std_v", "dark_v", "pmt_ohm", "pmt_c"):
        arr = getattr(result, name)
        if arr is not None:                    # pmt_* are None when unlogged
            setattr(result, name, np.asarray(arr)[keep])
    result.dropped_bursts = list(result.dropped_bursts) + gone
    return gone


def guard_darks(results: dict, *, limit: float = BURST_MAX_V) -> dict:
    """Replace any railed all-off dark with the median of the other targets'.

    For the live path, which has no CSV yet for :func:`median_dark_without`.

    In:  ``results``, ``{target index: PhaseResult}``; one dark per target,
         tested against ``limit``.
    Out: ``{tgt: {...}}`` for what was replaced, also recorded on each
         :attr:`PhaseResult.dark_overrides`.  Nothing when EVERY target is over.
    """
    darks: dict[int, float] = {}
    for k, r in results.items():
        dv = np.asarray(r.dark_v, dtype=float)
        if dv.size:
            darks[int(k)] = float(dv.flat[0])
    bad = {k for k, v in darks.items() if bool(flag_over_range([v], limit=limit)[0])}
    good = [v for k, v in darks.items() if k not in bad]
    if not bad or not good:
        return {}
    replacement = float(np.median(good))
    out: dict = {}
    for k in sorted(bad):
        entry = {
            "recorded_v": darks[k],
            "used_v": replacement,
            "reason": f"over the {limit*1e3:g} mV ceiling",
        }
        out[str(k)] = entry
        r = results[k]
        r.dark_v = np.full(np.asarray(r.dark_v).shape, replacement, dtype=float)
        r.dark_overrides = {**dict(r.dark_overrides), str(k): entry}
    return out


def _average_points(result: PhaseResult, dark_override: float | None = None):
    """Per-row dark-subtract, then average repeated trials per cell.

    Out: ``(x_t, w_t, x_r, w_r, y, std)`` per cell, ``y`` dark-subtracted.  The
    one place step 7 forms its weights.

    Each row's dark comes off BEFORE averaging (``dark_override`` replaces it).
    ``std`` is the across-trial standard deviation, NOT divided by sqrt(n),
    falling back to the row's ``voltage_std_v`` and then the median positive
    std, plus the sweep's :attr:`PhaseResult.sigma_floor_v` in quadrature.
    """
    y_raw = np.asarray(result.voltage_mean_v, dtype=float)
    if dark_override is None:
        dark_row = np.asarray(result.dark_v, dtype=float)
    else:
        dark_row = np.full(y_raw.shape, float(dark_override))
    y_sub = y_raw - dark_row
    std_row = np.asarray(result.voltage_std_v, dtype=float)

    cells: dict[tuple, list[float]] = defaultdict(list)
    scells: dict[tuple, list[float]] = defaultdict(list)
    key = np.column_stack([result.x_t, result.w_t, result.x_r, result.w_r])
    for row, y, s in zip(key, y_sub, std_row):
        rk = tuple(np.round(row, 9))
        cells[rk].append(float(y))
        scells[rk].append(float(s))

    keys, ys, std = [], [], []
    for k, vals in sorted(cells.items()):
        arr = np.asarray(vals, dtype=float)
        keys.append(k)
        ys.append(arr.mean())
        if arr.size > 1:
            std.append(arr.std(ddof=1))                         # across-trial spread
        else:
            rec = np.asarray(scells[k], dtype=float)            # recorded per-point std
            rec = rec[np.isfinite(rec) & (rec > 0)]
            std.append(float(rec.mean()) if rec.size else np.nan)

    keys = np.asarray(keys, dtype=float)
    ys = np.asarray(ys, dtype=float)
    std = np.asarray(std, dtype=float)
    finite = std[np.isfinite(std) & (std > 0)]
    fallback = float(np.median(finite)) if finite.size else 1.0
    std = np.where(np.isfinite(std) & (std > 0), std, fallback)
    # Systematic floor last, so it applies to the salvaged cells too.
    return (keys[:, 0], keys[:, 1], keys[:, 2], keys[:, 3], ys,
            floor_std(std, getattr(result, 'sigma_floor_v', None)))


def fit_phase_ratio(
    dphi_slm: np.ndarray,
    g: np.ndarray,
    fixed_bg: np.ndarray,
    y: np.ndarray,
    std: np.ndarray,
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

    Model per row, ``A := eta_ref``, ``B := eta_tgt``, ``s`` the shared scale::

        a = s A ,  b = s B
        Y = s^2 (A^2 + B^2 g^2 + 2 A B g cos(dPhi_SLM + dPhi_comb))
            + fixed_bg + d

    Three free parameters: ``s`` is boxed to ``[max(0,1-frac), 1+frac]``, so
    ``frac=1`` allows a 0..2x common gain drift between steps 6 and 7 with the
    relative efficiencies preserved, and ``frac=0`` pins it at 1, leaving only
    ``dPhi_comb`` and ``d`` free and ``a_err``/``b_err`` at 0.  ``fixed_bg`` is
    the per-row step-6 single-beam background, which the amplitudes do NOT
    scale.  ``eta_ref_err``/``eta_tgt_err`` do not enter this fit.
    """
    from scipy.optimize import least_squares

    dphi_slm = np.asarray(dphi_slm, dtype=float)
    g = np.asarray(g, dtype=float)
    fixed_bg = np.asarray(fixed_bg, dtype=float)
    y = np.asarray(y, dtype=float)
    std = np.asarray(std, dtype=float)

    A, B = float(eta_ref), float(eta_tgt)
    fix_scale = frac == 0.0                                   # box collapsed -> s pinned at 1

    def predict(s, dphi, d):
        s2 = s * s
        return (s2 * (A * A + B * B * g * g)
                + 2.0 * s2 * A * B * g * np.cos(dphi_slm + dphi) + fixed_bg + d)

    # phase seed: linear projection of the (background + self) subtracted signal
    w = 1.0 / std**2
    r0 = y - fixed_bg - A**2 - B**2 * g**2
    P = float(np.sum(w * r0 * g * np.cos(dphi_slm)))
    Q = float(np.sum(w * r0 * g * np.sin(dphi_slm)))
    dphi0 = float(np.arctan2(-Q, P))

    if fix_scale:                  # 2 free params (dPhi_comb, d); a = A, b = B verbatim
        def resid(p):
            return (predict(1.0, p[0], p[1]) - y) / std

        sol = least_squares(resid, [dphi0, 0.0], max_nfev=20000)
        s, dphi, d = 1.0, float(sol.x[0]), float(sol.x[1])
    else:                          # 3 free params; s boxed to [max(0, 1-frac), 1+frac]
        def resid(p):
            return (predict(p[0], p[1], p[2]) - y) / std

        lo = [max(0.0, 1.0 - frac), -np.inf, -np.inf]
        hi = [1.0 + frac, np.inf, np.inf]
        sol = least_squares(resid, [1.0, dphi0, 0.0], bounds=(lo, hi), max_nfev=20000)
        s, dphi, d = (float(v) for v in sol.x)
    dphi = float(np.arctan2(np.sin(dphi), np.cos(dphi)))      # wrap to (-pi, pi]
    a, b = s * A, s * B                                       # ratio locked to A:B

    y_pred = predict(s, dphi, d)
    residuals = y - y_pred
    n_free = 2 if fix_scale else 3

    # covariance from the weighted Jacobian at the solution (resid already /std)
    try:
        cov = np.linalg.inv(sol.jac.T @ sol.jac)
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
        r2=r2,
        eta_ref=eta_ref, eta_tgt=eta_tgt, bound_frac=frac,
        a_at_bound=a_at_bound, b_at_bound=b_at_bound,
        bg0=bg0, bg1=bg1, bg2=bg2,
        dphi_slm=dphi_slm, g=g, y=y, std=std, known=known,
        y_pred=y_pred, residuals=residuals,
    )


def fit_phase_fixed(
    dphi_slm: np.ndarray,
    g: np.ndarray,
    fixed_bg: np.ndarray,
    y: np.ndarray,
    std: np.ndarray,
    *,
    eta_ref: float,
    eta_tgt: float,
    eta_ref_err: float = 0.0,
    eta_tgt_err: float = 0.0,
    g_ref: np.ndarray | float = 1.0,
    bg0: float = 0.0,
    bg1: float = 0.0,
    bg2: float = 0.0,
) -> PhaseFit:
    """Fit dPhi_comb ALONE -- every amplitude and background pinned to step 6.

    One free parameter.  ``a = eta_ref g_ref`` and ``b = eta_tgt`` are step 6's
    amplitudes verbatim (no shared scale, unlike :func:`fit_phase_ratio`), the
    single-beam response enters as the fixed per-row ``fixed_bg``, the dark is
    already out of ``y`` row by row, and no residual offset is floated::

        Y = a^2 + b^2 g^2 + 2 a b g cos(dPhi_SLM + dPhi_comb) + fixed_bg

    ``g_ref = sqrt(x_r w_r)`` is the reference's own field amplitude, 1 when it
    is held fully on; the ``dphi_slm`` passed in already carries the matching
    ``-phi_half(x_r) - phi_half(w_r)``.  ``a_err``/``b_err``/``amp_err`` and
    ``offset``/``offset_err`` are 0, not being fitted, so ``dphi_comb_err`` is
    the FRINGE's error only.  Pass step 6's ``eta_ref_err``/``eta_tgt_err`` for
    the sensitivities ``dphi_deta_ref``/``dphi_deta_tgt``, which
    :attr:`PhaseFit.dphi_comb_err_total` and :attr:`PhaseFit.std_total` add back
    in.  The WEIGHTS stay ``1/std``.
    """
    from scipy.optimize import least_squares

    dphi_slm = np.asarray(dphi_slm, dtype=float)
    g = np.asarray(g, dtype=float)
    fixed_bg = np.asarray(fixed_bg, dtype=float)
    y = np.asarray(y, dtype=float)
    std = np.asarray(std, dtype=float)

    A, B = float(eta_ref), float(eta_tgt)
    gr = np.asarray(g_ref, dtype=float) * np.ones_like(g)   # reference field amplitude
    Aeff = A * gr                                 # the reference arm as actually driven
    known = Aeff * Aeff + B * B * g * g + fixed_bg  # everything that is not the fringe

    def predict(dphi):
        return known + 2.0 * Aeff * B * g * np.cos(fringe_arg(dphi_slm, dphi))

    # phase seed: project the fringe-only residual onto cos/sin(dPhi_SLM).  With
    # cos(dPhi_SLM + dPhi_comb) = cos dPhi_SLM cos dPhi_comb - sin dPhi_SLM sin dPhi_comb
    # the projections are (proportional to) +cos and -sin of dPhi_comb.
    w = 1.0 / std**2
    r0 = y - known
    P = float(np.sum(w * r0 * g * np.cos(dphi_slm)))
    Q = float(np.sum(w * r0 * g * np.sin(dphi_slm)))
    dphi0 = float(np.arctan2(-Q, P))

    def resid(p):
        return (predict(p[0]) - y) / std

    sol = least_squares(resid, [dphi0], max_nfev=20000)
    dphi = float(np.arctan2(np.sin(sol.x[0]), np.cos(sol.x[0])))   # wrap to (-pi, pi]

    y_pred = predict(dphi)
    residuals = y - y_pred
    try:
        cov = np.linalg.inv(sol.jac.T @ sol.jac)
        dphi_err = float(np.sqrt(max(cov[0, 0], 0.0)))
    except np.linalg.LinAlgError:
        dphi_err = float("nan")

    # The pinned amplitudes' error, propagated into the phase.  phi_hat solves
    # S(phi; A, B) = sum_j w_j (Y_j - m_j) dm_j/dphi = 0, so by the implicit
    # function theorem, dropping the Gauss-Newton second-derivative term:
    #
    #     dphi/dA = -(J_phi^T W J_A) / (J_phi^T W J_phi)
    #
    # The denominator is the Fisher information that sets dphi_err above.
    psi = fringe_arg(dphi_slm, dphi)
    # d/d(eta_ref) carries the chain factor g_ref, since a = eta_ref g_ref.
    j_phi = -2.0 * Aeff * B * g * np.sin(psi)                        # dm/d(dPhi_comb)
    j_A = 2.0 * A * gr * gr + 2.0 * B * g * gr * np.cos(psi)         # dm/d(eta_ref)
    j_B = 2.0 * B * g * g + 2.0 * Aeff * g * np.cos(psi)             # dm/d(eta_tgt)
    fisher = float(np.sum(w * j_phi * j_phi))            # = 1 / dphi_err^2
    if fisher > 0:
        dphi_deta_ref = -float(np.sum(w * j_phi * j_A)) / fisher
        dphi_deta_tgt = -float(np.sum(w * j_phi * j_B)) / fisher
    else:
        dphi_deta_ref = dphi_deta_tgt = float("nan")

    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # `a` / `amp` are the reference arm AS DRIVEN (eta_ref g_ref), so the plot's
    # curve stays exact; the bare eta is kept in `eta_ref`.  g_ref is constant
    # over a sweep, so take point 0.
    a_eff = float(Aeff[0]) if Aeff.size else A
    return PhaseFit(
        dphi_comb=dphi, dphi_comb_err=dphi_err,
        a=a_eff, a_err=0.0, b=B, b_err=0.0,
        amp=2.0 * a_eff * B, amp_err=0.0,
        offset=0.0, offset_err=0.0,
        r2=r2,
        eta_ref=A, eta_tgt=B, bound_frac=0.0,
        a_at_bound=False, b_at_bound=False,
        bg0=bg0, bg1=bg1, bg2=bg2,
        dphi_slm=dphi_slm, g=g, y=y, std=std, known=known,
        y_pred=y_pred, residuals=residuals,
        eta_ref_err=float(eta_ref_err), eta_tgt_err=float(eta_tgt_err),
        dphi_deta_ref=dphi_deta_ref, dphi_deta_tgt=dphi_deta_tgt,
        dm_deta_ref=j_A, dm_deta_tgt=j_B,
    )


def _step6_background(tgt_model, ref_model, x_t, w_t, x_r, w_r, enabled: bool):
    """Per-row step-6 single-beam background (dark already out) + its g-polynomial.

    Returns ``(fixed_bg, bg0, bg1, bg2)``.  ``bg0/bg1/bg2`` re-express the
    background as a polynomial in ``g`` for the special case ``x_t = w_t`` and
    are carried for reference only.  ``enabled=False`` -> an all-zero background.
    """
    g_shape = np.asarray(x_t, dtype=float)
    if not enabled:
        return np.zeros_like(g_shape), 0.0, 0.0, 0.0
    fixed_bg = np.asarray(
        ref_model.single_beam(x_r, w_r) + tgt_model.single_beam(x_t, w_t), dtype=float
    )
    # bg0 is the g-independent term: the reference's single beam AT ITS OWN
    # DRIVE, not at (1, 1) -- the reference is not always held fully on.
    return (fixed_bg,
            float(np.mean(np.atleast_1d(ref_model.single_beam(x_r, w_r)))),
            float(tgt_model.a_x + tgt_model.a_w),
            float(tgt_model.q_x + tgt_model.q_w))


def fit_result(
    result: PhaseResult,
    tgt_model: PairModel,
    ref_model: PairModel,
    *,
    dark: float | None = None,
    frac: float | None = None,
    single_beam_bg: bool = False,
    comb_only: bool = False,
) -> PhaseFit:
    """Fit ``a``, ``b`` and ``dPhi_comb`` to the dark-subtracted Y.

    The driver: dark-subtract, average repeats (:func:`_average_points`), pick a
    fitter.

    In:  a :class:`PhaseResult`.  ``frac=None`` (default) takes the
         unconstrained :func:`fit_phase`; a number locks ``a:b`` to the step-6
         eta ratio and floats a shared scale boxed to ``+/- frac``
         (:func:`fit_phase_ratio`, ``frac=0`` pinning it exactly), where
         ``single_beam_bg`` folds both pairs' step-6 single-beam response in as
         a fixed background.  ``comb_only=True`` overrides ``frac`` and runs
         :func:`fit_phase_fixed`.  ``dark`` overrides the per-row dark.
    Out: a :class:`PhaseFit`.
    """
    x_t, w_t, x_r, w_r, y, std = _average_points(result, dark_override=dark)

    g = np.sqrt(np.clip(x_t * w_t, 0.0, None))         # target field amplitude
    dphi_slm = slm_phase_diff(x_t, w_t, x_r, w_r)       # SLM phase difference

    result.tgt_model = tgt_model
    result.ref_model = ref_model
    if comb_only:
        # step-7 v2: nothing floats but dPhi_comb
        fixed_bg, bg0, bg1, bg2 = _step6_background(
            tgt_model, ref_model, x_t, w_t, x_r, w_r, single_beam_bg)
        fit = fit_phase_fixed(
            dphi_slm, g, fixed_bg, y, std,
            eta_ref=ref_model.eta, eta_ref_err=ref_model.eta_err,
            eta_tgt=tgt_model.eta, eta_tgt_err=tgt_model.eta_err,
            g_ref=np.sqrt(np.clip(x_r * w_r, 0.0, None)),   # reference field amplitude
            bg0=bg0, bg1=bg1, bg2=bg2,
        )
    elif frac is None:
        fit = fit_phase(dphi_slm, g, y, std)
    else:
        # step-6 single-beam of both pairs as a fixed background (dark already out)
        fixed_bg, bg0, bg1, bg2 = _step6_background(
            tgt_model, ref_model, x_t, w_t, x_r, w_r, single_beam_bg)

        fit = fit_phase_ratio(
            dphi_slm, g, fixed_bg, y, std,
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

    Out: ``(x_t, w_t, z, z_swapped, abs_diff, std)`` for the off-diagonal cells,
    or raw Y with no fit attached.  A difference well above the combined std is
    a channel asymmetry or crosstalk.

    On the CLEAN interference term, ``fit.known`` holding the self terms and the
    step-6 background::

        Z(x,w) = Y(x,w) - [a^2 + b^2 (x w) + sb_ref + sb_tgt] - d
               = 2 a b sqrt(x w) cos(dPhi_SLM + dPhi_comb)
    """
    x_t, w_t, x_r, w_r, y, std = _average_points(result)   # y already dark-subtracted
    fit = result.fit
    if fit is not None and fit.known is not None and np.isfinite(fit.a):
        # clean interference: strip fitted self terms + step-6 single-beam + d
        sig = y - fit.known - fit.offset
    else:
        sig = y

    lut = {(round(a, 9), round(b, 9)): (zz, ss)
           for a, b, zz, ss in zip(x_t, w_t, sig, std)}
    out = []
    for a, b, zz, ss in zip(x_t, w_t, sig, std):
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
    "dark_v", "voltage_mean_v", "voltage_std_v",
]


def write_phase_csv(result: PhaseResult, path: str | Path) -> str:
    """Raw rows, one line per (trial, point).  Round-trips via load.

    ``phi_xt_deg``/``phi_wt_deg`` are the target channel phases, for reading
    against the sweep tables; the fit reloads from the canonical intensities.
    ``dark_v`` is that row's dark, and the run's mean is stashed as a trailing
    comment.  ``voltage_std_v`` is the spread the fit weights by.
    """
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_CSV_HEADER)
        for t, x_t, w_t, x_r, w_r, dark_v, mean_v, std_v in zip(
            result.trial, result.x_t, result.w_t, result.x_r, result.w_r,
            result.dark_v, result.voltage_mean_v, result.voltage_std_v,
        ):
            phi_xt = np.degrees(2.0 * float(phi_half(x_t)))
            phi_wt = np.degrees(2.0 * float(phi_half(w_t)))
            writer.writerow(
                [int(t), result.tgt_index, result.ref_index,
                 f"{phi_xt:.4g}", f"{phi_wt:.4g}",
                 f"{x_t:.6g}", f"{w_t:.6g}", f"{x_r:.6g}", f"{w_r:.6g}",
                 f"{dark_v:.9g}", f"{mean_v:.9g}", f"{std_v:.9g}"]
            )
    with open(out, "a", newline="", encoding="utf-8") as f:
        f.write(f"# dark_mean_v,{result.dark:.9g}\n")
    result.csv_path = str(out)
    return str(out)


_MEAS_CSV_HEADER = [
    "trial", "tgt_index", "ref_index",
    "phi_xt_deg", "phi_wt_deg", "x_t", "w_t", "x_r", "w_r",
    "dark_v", "voltage_mean_v", "voltage_std_v", "std_ratio",
    # The PMT thermistor, blank when the sweep ran without it.  Trailing and
    # read back by name, so a v1 refit of this file is unaffected.
    "pmt_ohm_Ohm", "pmt_c",
]


def write_meas_csv(results: Sequence[PhaseResult], path: str | Path,
                   *, meta: Mapping[str, object] | None = None) -> str:
    """Raw rows for one or more target pairs, in a single CSV.

    :func:`write_phase_csv`'s columns plus a trailing ``std_ratio``
    (std/|mean|), every row carrying its own ``tgt_index`` beside the shared
    ``ref_index`` and read back with :func:`load_phase_csv`'s ``only_tgt``.
    ``meta`` is appended as trailing ``# key,value`` lines.
    """
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_MEAS_CSV_HEADER)
        for result in results:
            n = len(result.trial)
            ohms = result.pmt_ohm if result.pmt_ohm is not None else [""] * n
            cels = result.pmt_c if result.pmt_c is not None else [""] * n
            for t, x_t, w_t, x_r, w_r, dark_v, mean_v, std_v, ohm, celsius in zip(
                result.trial, result.x_t, result.w_t, result.x_r, result.w_r,
                result.dark_v, result.voltage_mean_v, result.voltage_std_v,
                ohms, cels,
            ):
                phi_xt = np.degrees(2.0 * float(phi_half(x_t)))
                phi_wt = np.degrees(2.0 * float(phi_half(w_t)))
                ratio = abs(std_v / mean_v) if mean_v else float("inf")
                writer.writerow(
                    [int(t), result.tgt_index, result.ref_index,
                     f"{phi_xt:.4g}", f"{phi_wt:.4g}",
                     f"{x_t:.6g}", f"{w_t:.6g}", f"{x_r:.6g}", f"{w_r:.6g}",
                     f"{dark_v:.9g}", f"{mean_v:.9g}", f"{std_v:.9g}",
                     f"{ratio:.6g}", ohm, celsius]
                )
        for key, value in (meta or {}).items():
            writer.writerow([f"# {key}", value])
    for result in results:
        result.csv_path = str(out)
    return str(out)


def targets_in_csv(path: str | Path, default=()) -> list[int]:
    """Distinct target-pair indices recorded in a CSV (sorted).

    A collected file stacks every target vs the shared reference, so its
    ``tgt_index`` column lists several pairs.  Falls back to ``default`` if the
    column is missing (an old single-target CSV).
    """
    seen: list[int] = []
    with open(Path(path), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(line for line in f if not line.startswith("#")):
            t = row.get("tgt_index")
            if t in (None, ""):
                continue
            k = int(float(t))
            if k not in seen:
                seen.append(k)
    return sorted(seen) if seen else sorted(default)


def reference_in_csv(path: str | Path) -> int | None:
    """The reference pair a recorded run was swept against, from its own rows.

    Every row carries ``ref_index``, so no caller has to supply it.  None when
    the column is absent (an old CSV); raises when the rows disagree, which
    means several runs were concatenated.
    """
    seen: set[int] = set()
    with open(Path(path), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(line for line in f if not line.startswith("#")):
            r = row.get("ref_index")
            if r not in (None, ""):
                seen.add(int(float(r)))
    if not seen:
        return None
    if len(seen) > 1:
        raise ValueError(
            f"{Path(path).name} mixes references {sorted(seen)}; a step-7 CSV "
            f"records one run against one reference"
        )
    return seen.pop()


def references_in_csv(path: str | Path) -> dict[int, int]:
    """``{tgt_index: ref_index}`` -- which reference EACH target was swept against.

    The cascaded form of :func:`reference_in_csv`, which refuses such a file.
    Feed the result to :func:`load_phase_csv` as the right ``ref_model`` per
    target.  Raises if one target appears against two references.
    """
    out: dict[int, int] = {}
    with open(Path(path), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(line for line in f if not line.startswith("#")):
            t, r = row.get("tgt_index"), row.get("ref_index")
            if t in (None, "") or r in (None, ""):
                continue
            k, ref = int(float(t)), int(float(r))
            if out.setdefault(k, ref) != ref:
                raise ValueError(
                    f"{Path(path).name}: pair {k} appears against references "
                    f"{out[k]} and {ref}; a step-7 CSV records each target once"
                )
    return out


def darks_in_csv(path: str | Path) -> dict[int, float]:
    """``{tgt_index: dark_v}`` -- the all-off read each target was corrected by.

    One dark per target, subtracted from every row of it.  It has no repeat and
    no burst guard of its own, so the other targets' darks are what judge it --
    see :func:`median_dark_without`.  Raises if one target carries two.
    """
    out: dict[int, float] = {}
    with open(Path(path), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(line for line in f if not line.startswith("#")):
            t, d = row.get("tgt_index"), row.get("dark_v")
            if t in (None, "") or d in (None, ""):
                continue
            k, dv = int(float(t)), float(d)
            if not np.isclose(out.setdefault(k, dv), dv, rtol=0, atol=1e-12):
                raise ValueError(
                    f"{Path(path).name}: pair {k} carries two darks "
                    f"{out[k]:g} and {dv:g}; step 7 records one per target"
                )
    return out


def median_dark_without(path: str | Path, pair: int) -> float:
    """Median of every OTHER target's dark in ``path`` -- the stand-in.

    ``pair`` is excluded, so a glitched read cannot vote on its own replacement.
    """
    others = [v for k, v in darks_in_csv(path).items() if k != int(pair)]
    if not others:
        raise ValueError(
            f"{Path(path).name} has no target other than {pair} to take a dark from"
        )
    return float(np.median(others))


def load_phase_csv(
    path: str | Path,
    tgt_model: PairModel,
    ref_model: PairModel,
    *,
    dark: float | None = None,
    frac: float | None = None,
    single_beam_bg: bool = False,
    comb_only: bool = False,
    only_tgt: int | None = None,
    drop_bursts: bool = True,
    floor: tuple[float, str] | None = None,
    rail_v: float = BURST_MAX_V,
) -> PhaseResult:
    """Load a raw phase-sweep CSV and re-fit dPhi_comb against step-6 models.

    In:  ``path`` and the two :class:`PairModel` objects.  ``only_tgt`` keeps
         just that ``tgt_index`` (None loads every row).  ``frac``,
         ``single_beam_bg`` and ``comb_only`` go to :func:`fit_result`.
         ``floor`` is the ``(volts, source)`` sigma floor to weight by,
         normally the run's step-6 one; omitted, the sweep's own quietest drive
         points are used, then the fallback constant.
    Out: a fitted :class:`PhaseResult`.

    The dark comes from the per-row ``dark_v`` column, else the
    ``# dark_mean_v`` comment, else the step-6 mean; a scalar ``dark`` wins.
    ``drop_bursts`` (default on) discards bursts into
    :attr:`PhaseResult.dropped_bursts`, leaving an all-burst sweep alone, and
    the absolute ceiling also reaches the DARK, replacing it from
    :func:`median_dark_without` into :attr:`PhaseResult.dark_overrides`.
    """
    file_dark: float | None = None
    with open(Path(path), newline="", encoding="utf-8") as f:
        for raw in f:
            if raw.startswith("#"):
                parts = raw.lstrip("#").strip().split(",")
                if len(parts) == 2 and parts[0].strip() == "dark_mean_v":
                    file_dark = float(parts[1])

    rows: list[tuple] = []
    row_tgts: list[int] = []
    tgt_index, ref_index = tgt_model.index, ref_model.index
    with open(Path(path), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(line for line in f if not line.startswith("#")):
            row_tgt = int(float(row.get("tgt_index", tgt_index)))
            if only_tgt is not None and row_tgt != only_tgt:
                continue  # skip the other targets in a multi-pair CSV
            dv = row.get("dark_v")
            std_v = float(row["voltage_std_v"])  # the fit weight; every CSV records it
            row_tgts.append(row_tgt)
            rows.append((
                int(float(row.get("trial", 0))),
                float(row["x_t"]), float(row["w_t"]),
                float(row["x_r"]), float(row["w_r"]),
                float(row["voltage_mean_v"]),
                std_v,
                float(dv) if dv not in (None, "") else None,
                row.get("pmt_ohm_Ohm") or "", row.get("pmt_c") or "",
            ))
            tgt_index = row_tgt
            ref_index = int(float(row.get("ref_index", ref_index)))

    dropped_bursts: list[dict] = []
    if drop_bursts and rows:
        mask = burst_mask([r[6] for r in rows], row_tgts)
        mask = mask | flag_over_range([r[5] for r in rows], limit=rail_v)
        for r, bad in zip(rows, mask):
            if bad:
                dropped_bursts.append({
                    "trial": r[0], "x_t": r[1], "w_t": r[2],
                    "voltage_mean_v": r[5], "voltage_std_v": r[6],
                })
        if mask.any():
            rows = [r for r, bad in zip(rows, mask) if not bad]

    trials = np.array([r[0] for r in rows], dtype=int)
    # The dark has neither a repeat nor a sweep median to be judged against and
    # is subtracted from every row of its target, so it gets the absolute
    # ceiling, which needs no reference but itself.
    dark_overrides: dict = {}
    if dark is None and drop_bursts and rows and all(r[7] is not None for r in rows):
        recorded = float(rows[0][7])
        if bool(flag_over_range([recorded], limit=rail_v)[0]):
            try:
                replacement = median_dark_without(path, tgt_index)
            except ValueError:
                replacement = None
            if replacement is not None:
                dark_overrides[str(tgt_index)] = {
                    "recorded_v": recorded,
                    "used_v": float(replacement),
                    "reason": f"over the {rail_v*1e3:g} mV ceiling",
                }
                dark = replacement
    scalar_dark = (
        dark if dark is not None
        else file_dark if file_dark is not None
        else 0.5 * (tgt_model.d + ref_model.d)
    )
    # per-row dark: CSV column if present (and not overridden), else the scalar
    if dark is None and all(r[7] is not None for r in rows) and rows:
        dark_v = np.array([r[7] for r in rows], dtype=float)
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
        dark_v=dark_v,
        n_trials=int(trials.max()) + 1 if trials.size else 1,
        csv_path=str(Path(path).resolve()),
        # only carried through when the file actually has them, so a CSV with no
        # thermistor round-trips back to None rather than to a column of blanks
        pmt_ohm=(np.array([r[8] for r in rows], dtype=object)
                 if any(r[8] for r in rows) else None),
        pmt_c=(np.array([r[9] for r in rows], dtype=object)
               if any(r[9] for r in rows) else None),
        dropped_bursts=dropped_bursts,
        dark_overrides=dark_overrides,
    )
    if floor is not None:
        result.sigma_floor_v, result.sigma_floor_source = float(floor[0]), str(floor[1])
    else:
        f_v, f_src = measured_floor(result.voltage_mean_v, result.voltage_std_v)
        result.sigma_floor_v, result.sigma_floor_source = float(f_v), f_src.value
    fit_result(result, tgt_model, ref_model, frac=frac,
               single_beam_bg=single_beam_bg, comb_only=comb_only)
    return result


def phase_fit_payload(fit: PhaseFit) -> dict:
    """JSON-ready summary of one dPhi_comb fit (shared by every phase saver)."""
    return {
        "dphi_comb_rad": fit.dphi_comb,
        "dphi_comb_deg": fit.dphi_comb_deg,
        "dphi_comb_err_rad": fit.dphi_comb_err,          # fringe noise only
        "dphi_comb_err_deg": float(np.degrees(fit.dphi_comb_err)),
        # pinned step-6 amplitudes (fit_phase_fixed): their error reaches the
        # phase through these sensitivities, never through dphi_comb_err
        "dphi_comb_err_eta_rad": fit.dphi_comb_err_eta,
        "dphi_comb_err_eta_deg": float(np.degrees(fit.dphi_comb_err_eta)),
        "dphi_comb_err_total_rad": fit.dphi_comb_err_total,
        "dphi_comb_err_total_deg": float(np.degrees(fit.dphi_comb_err_total)),
        "dphi_deta_ref": fit.dphi_deta_ref,   # d(dPhi_comb)/d(eta_ref), rad/eta
        "dphi_deta_tgt": fit.dphi_deta_tgt,
        "eta_ref_err": fit.eta_ref_err,
        "eta_tgt_err": fit.eta_tgt_err,
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
        "r2": fit.r2,
    }


def _chained_fit_payload(fit: PhaseFit, chain, root: int) -> dict:
    """:func:`phase_fit_payload`, with the phase referred to ``root`` if it hopped.

    Only the phase and its error bars move; the amplitudes, residual and R^2
    keep describing the hop as MEASURED.  ``dphi_deta_ref`` becomes the
    sensitivity to the ROOT's eta, and the hub's goes in the ``chained`` block
    with the rest of the path.  ``chain=None`` returns the payload unchanged.
    """
    payload = phase_fit_payload(fit)
    if chain is None or not chain.chained:
        return payload
    from .cascade import chained_payload      # local: cascade imports nothing here
    payload.update({
        "dphi_comb_rad": chain.dphi,
        "dphi_comb_deg": chain.dphi_deg,
        "dphi_comb_err_rad": chain.err_fringe,
        "dphi_comb_err_deg": float(np.degrees(chain.err_fringe)),
        "dphi_comb_err_eta_rad": chain.err_eta,
        "dphi_comb_err_eta_deg": float(np.degrees(chain.err_eta)),
        "dphi_comb_err_total_rad": chain.err_total,
        "dphi_comb_err_total_deg": float(np.degrees(chain.err_total)),
        "dphi_deta_ref": chain.deta.get(int(root), 0.0),
        "dphi_deta_tgt": chain.deta.get(int(chain.tgt), 0.0),
        "eta_ref_err": chain.eta_err.get(int(root), 0.0),
        "chained": chained_payload(chain, fit),
    })
    return payload


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


def save_step7_section(
    fits: dict[tuple[int, str], PhaseFit],
    run: str | Path,
    *,
    ref_index: int,
    csv: str | Path | Sequence[str | Path] | None = None,
    single_beam_bg: bool | None = None,
    chains: Mapping[int, object] | None = None,
    extra: dict | None = None,
    log=print,
):
    """Write the comb-phase spectrum as the run's ``step7`` section.

    In:  ``fits``, mapping ``(tgt_index, method)`` to a :class:`PhaseFit`,
         ``method`` being the driver's amplitude-handling label (``"bounded"``,
         ``"fix"``).  ``extra`` keys are merged into the section (v2 puts its
         cascade plan under ``"cascade"``).  ``chains``
         (:func:`~calibration_module.fit.cascade.chain_fits`) makes this the
         saver for a CASCADED run.
    Out: the written section, via
         :func:`calibration_module.calib_file.write_section`.

    With ``chains``, every stored ``dphi_comb_*`` is referred to the root
    ``ref_index`` whichever hub produced it, and a hopped pair also carries a
    ``"chained"`` block holding the hop as measured.  Step 3 and step 6 are not
    copied forward, being sections of the same file.
    """
    from calibration_module.calib_file import load_calib, write_section

    # Phases are measured against step 6's etas; without them the section is not
    # interpretable, so say so before the file is touched.
    load_calib(run, missing_ok=True).require(3, 6)
    section = {
        "ref_index": int(ref_index),
        "single_beam_bg": single_beam_bg,
        "channels": [
            {"tgt_index": int(k), "method": str(m),
             "fit": _chained_fit_payload(f, (chains or {}).get(k),
                                         int(ref_index))}
            for (k, m), f in sorted(fits.items())
        ],
    }
    if extra:
        section.update(extra)
    return write_section(run, 7, section, csv=csv, log=log)


def load_comb_phase_json(
    path: str | Path, *, method: str | None = None
) -> tuple[int, dict[int, dict]]:
    """Load a combined step-7 JSON -> ``(ref_index, {tgt_index: channel entry})``.

    Each entry is the stored ``{"tgt_index", "method", "fit": {...}}`` dict, in
    which ``fit["dphi_comb_rad"]`` is the phase against the reference that
    defines ``Phi = 0``.  ``method`` picks among several stored fits per target;
    with ``None`` a target must have exactly one, or the choice is ambiguous and
    it raises.
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



# ======================================================================
# phase covariance across targets
# ======================================================================

def phase_covariance(fits, pairs) -> np.ndarray:
    """Covariance of the fitted ``dPhi_comb`` across targets, step-6 eta included.

    In:  ``fits``, ``{target index: PhaseFit}`` from :func:`fit_phase_fixed`, the
         only fitter that pins the etas and so fills the sensitivities, and
         ``pairs``, the target order to build in.
    Out: the covariance matrix.  Three contributions, for targets ``i``, ``j``::

        fringe noise    diag(dphi_comb_err_i^2)                    independent
        eta_tgt,i       diag((dphi_deta_tgt,i * eta_tgt_err_i)^2)  independent
        eta_ref         dphi_deta_ref_i * dphi_deta_ref_j * s^2    COMMON

    The last term is rank 1, every target being pinned to the same reference
    amplitude.  Whiten residuals with the Cholesky factor before judging them;
    the diagonal is :attr:`PhaseFit.dphi_comb_err_total` squared.
    """
    pairs = list(pairs)
    if not pairs:
        raise ValueError("no target pairs to build a covariance over")
    # The rank-1 term below IS the assumption of one common reference.  A
    # cascaded run breaks it silently -- right shape, wrong weights -- and only
    # the eta_ref VALUES can say, the fits not recording which pair they are.
    refs = {round(float(fits[k].eta_ref), 12) for k in pairs}
    if len(refs) > 1:
        raise ValueError(
            f"these fits do not share one reference (eta_ref takes "
            f"{len(refs)} values), so the common-eta_ref term here would be "
            f"wrong; use calibration_module.fit.cascade.cascade_covariance, "
            f"which tracks each path's hops and pairs"
        )
    diag = np.array(
        [fits[k].dphi_comb_err**2 + (fits[k].dphi_deta_tgt * fits[k].eta_tgt_err) ** 2
         for k in pairs], dtype=float)
    c_ref = np.array([fits[k].dphi_deta_ref for k in pairs], dtype=float)
    s_ref = float(fits[pairs[0]].eta_ref_err)
    return np.diag(diag) + np.outer(c_ref, c_ref) * s_ref**2


# ======================================================================
# drive builders
# ======================================================================

def build_phase_sweep(
    *,
    n_points: int = 15,
    phi_start_deg: float = 0.0,
    phi_stop_deg: float = 180.0,
    ref_phase_deg: float = 180.0,
) -> list[tuple[float, float, float, float]]:
    """Symmetric target phase sweep vs a fixed reference (half fringe).

    The target is driven ``phi^x = phi^w = phi`` over
    ``[phi_start_deg, phi_stop_deg]`` (0..180 deg is the reachable half turn)
    with the reference fixed at ``ref_phase_deg`` on both channels.

    Out: commanded intensity tuples ``(x_t, w_t, x_r, w_r)``, target first, with
    ``x = sin(phi/2)^2``, so ``dPhi_SLM = phi - ref_phase`` sweeps the fringe.
    """
    phis = np.radians(np.linspace(phi_start_deg, phi_stop_deg, int(n_points)))
    x_r = float(intensity_for_phase(np.radians(ref_phase_deg)))
    x_t = intensity_for_phase(phis)
    return [(float(v), float(v), x_r, x_r) for v in x_t]


def build_symmetry_grid(
    *,
    phi_values_deg: Sequence[float] = (90.0, 135.0, 180.0),
    ref_phase_deg: float = 180.0,
) -> list[tuple[float, float, float, float]]:
    """3x3 grid on the target's individual channel phases (symmetry check).

    Sweeps the target's ``phi^x`` and ``phi^w`` INDEPENDENTLY over
    ``phi_values_deg`` with the reference fixed, so swapped and equal-sum cells
    can be compared (:func:`swap_invariance`).

    Out: commanded intensity tuples, target first.
    """
    x_r = float(intensity_for_phase(np.radians(ref_phase_deg)))
    out: list[tuple[float, float, float, float]] = []
    for px in phi_values_deg:
        xt = float(intensity_for_phase(np.radians(px)))
        for pw in phi_values_deg:
            wt = float(intensity_for_phase(np.radians(pw)))
            out.append((xt, wt, x_r, x_r))
    return out

__all__ = [
    "PairModel",
    "PhaseFit",
    "PhaseResult",
    "burst_mask",
    "drop_bursts",
    "load_pair_models",
    "build_phase_sweep",
    "build_symmetry_grid",
    "phi_half",
    "intensity_for_phase",
    "slm_phase_diff",
    "fringe_arg",
    "fit_phase",
    "fit_phase_ratio",
    "fit_phase_fixed",
    "fit_result",
    "swap_invariance",
    "write_phase_csv",
    "write_meas_csv",
    "targets_in_csv",
    "reference_in_csv",
    "references_in_csv",
    "load_phase_csv",
    "phase_fit_payload",
    "save_phase_json",
    "save_step7_section",
    "load_comb_phase_json",
]
