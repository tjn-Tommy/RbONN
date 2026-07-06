"""Comb-phase (dPhi_comb) calibration of each pair relative to a common reference.

Step 6 (:mod:`slm_module.tpa_pair`) calibrates each pair's two-photon efficiency
``eta`` *in isolation* -- one pair on at a time, so absolute optical phase never
enters.  This step drives **two pairs at once** and uses their coherent TPA
interference to recover the fixed comb phase offset ``dPhi_comb`` that a target
pair carries relative to a reference pair (pair 0 by convention).

Geometry.  A channel commanded at normalised INTENSITY ``x`` in [0, 1] (the
diffraction efficiency) sits at panel phase ``phi = 2*asin(sqrt(x))`` and has
field ``sqrt(x)*exp(i*phi/2)``.  The measured Step-3 transfer curve is monotonic
over the calibrated level range, so ``x`` in [0, 1] reaches ``phi`` in [0, pi]
only (``phi = pi`` is exactly ``x = 1``, fully on).  That is a *half* phase turn,
which is enough: with the reference fixed at ``phi = pi`` and the target swept
over ``phi in [0, pi]`` the relative SLM phase spans a full half fringe.

Sweep (Table 1).  Reference pair 0: ``phi^x_0 = phi^w_0 = pi`` (``x = w = 1``),
all other pairs off.  Target pair k swept symmetrically ``phi^x_k = phi^w_k =
phi`` over ``[0, pi]``.  Then::

    R_k    = eta_k * sin(phi/2)^2            (target amplitude; x_k = w_k = sin(phi/2)^2)
    R_0    = eta_0                           (reference amplitude, fixed)
    dPhi_SLM = 1/2*[(phi^x_k+phi^w_k) - (phi^x_0+phi^w_0)] = phi - pi   in [-pi, 0]
    Y      = R_0^2 + R_k^2 + 2*R_0*R_k*cos(dPhi_SLM + dPhi_comb)
             + (linear single-beam) + d

The reference-first turn (``phi = 0``, ``R_k = 0``) is a free baseline
``Y = R_0^2 + d``.  Because the target is calibrated *against* pair 0 and pair 0
defines ``Phi_0 == 0``, the fitted ``dPhi_comb`` IS pair k's phase in the
spectrum; running Table 1 for every k builds ``{Phi_k}``.

Background removal uses the step-6 per-pair fit (``a_x, q_x, a_w, q_w, d`` and
``eta``).  For each averaged point we subtract the linear/quadratic single-beam
response, the dark current and the two self-TPA pedestals ``R_k^2 + R_0^2``,
leaving the isolated interference term::

    Z := Y - d - single_beam_k - single_beam_0 - R_k^2 - R_0^2
       = 2*R_k*R_0*cos(dPhi_SLM + dPhi_comb)
       = A*[2*R_k*R_0*cos(dPhi_SLM)] + B*[2*R_k*R_0*sin(dPhi_SLM)]  (+ c)

which is LINEAR in ``A = cos(dPhi_comb)``, ``B = -sin(dPhi_comb)`` and solved by
weighted least squares.  ``dPhi_comb = atan2(-B, A)``; the fitted amplitude
``V = sqrt(A^2 + B^2)`` is the fringe *visibility* and should sit near 1 when the
step-6 etas are consistent and the two pairs are mutually coherent (V far from 1
flags an eta mismatch or partial coherence).  An optional constant ``c`` absorbs
any residual DC left by imperfect dark subtraction.

A second, one-time diagnostic (Table 2, :func:`build_symmetry_grid`) sweeps the
target's two channel phases *independently* on a 3x3 grid to check that phase
depends only on the sum ``phi^x + phi^w`` and amplitude only on the product
(swap invariance); see :func:`swap_invariance`.

The measurement is instrument-agnostic exactly like step 6: it drives an SLM
(``get_slm_info`` + ``display_array``) and reads whatever monitor exposes the
``ScopeController`` / ``DAQController`` shape.  Raw rows are persisted as a CSV
(one row per trial x point) so a run can be reloaded and re-fit offline.
"""
from __future__ import annotations

import csv
import json
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Fitted-parameter vector for the linear interference fit.
PARAMS: tuple[str, ...] = ("A", "B", "c")


class TPAPhaseAborted(Exception):
    """Raised when a stop_event interrupts a phase sweep."""


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
    step-6 ``save_tpa_pair_json`` output; any other extension is treated as a
    raw step-6 CSV and re-fit through :mod:`slm_module.tpa_pair` (so the fit is
    byte-identical to step 6).  ``layout`` is only needed for CSVs.  Later paths
    win on index collisions.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    models: dict[int, PairModel] = {}
    for path in paths:
        path = Path(path)
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
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

    Target (subscript t) minus reference (subscript r): for a symmetric target
    sweep against a fully-on reference this is ``phi - pi``.
    """
    return phi_half(x_t) + phi_half(w_t) - phi_half(x_r) - phi_half(w_r)


# ======================================================================
# fit  (linear least squares in A = cos dPhi_comb, B = -sin dPhi_comb, [c])
# ======================================================================

@dataclass
class PhaseFit:
    """Weighted-least-squares recovery of dPhi_comb from the interference fringe."""

    dphi_comb: float           # radians, wrapped to (-pi, pi]
    dphi_comb_err: float
    visibility: float          # sqrt(A^2 + B^2); ~1 when etas are consistent
    visibility_err: float
    offset: float              # fitted DC nuisance c (0 if not fit)
    offset_err: float
    chi2_red: float
    dof: int
    birge: float
    r2: float
    # point arrays the fit ran on (kept for plotting)
    dphi_slm: np.ndarray = field(repr=False)
    amp: np.ndarray = field(repr=False)          # 2*R_t*R_r at each point
    z: np.ndarray = field(repr=False)            # isolated interference term
    sem: np.ndarray = field(repr=False)
    z_pred: np.ndarray = field(repr=False)
    residuals: np.ndarray = field(repr=False)

    @property
    def dphi_comb_deg(self) -> float:
        return float(np.degrees(self.dphi_comb))


def fit_phase(
    dphi_slm: np.ndarray,
    amp: np.ndarray,
    z: np.ndarray,
    sem: np.ndarray,
    *,
    fit_offset: bool = True,
) -> PhaseFit:
    """Weighted LS fit of the isolated fringe ``z`` to ``A*g_c + B*g_s (+ c)``.

    ``amp`` is ``2*R_t*R_r`` per point; ``g_c = amp*cos(dPhi_SLM)``,
    ``g_s = amp*sin(dPhi_SLM)``.  Errors are Birge-scaled by ``sqrt(chi2/dof)``
    when chi2/dof > 1.  ``dPhi_comb = atan2(-B, A)`` with covariance-propagated
    error; ``visibility = sqrt(A^2 + B^2)``.
    """
    dphi_slm = np.asarray(dphi_slm, dtype=float)
    amp = np.asarray(amp, dtype=float)
    z = np.asarray(z, dtype=float)
    sem = np.asarray(sem, dtype=float)

    gc = amp * np.cos(dphi_slm)
    gs = amp * np.sin(dphi_slm)
    cols = [gc, gs] + ([np.ones_like(gc)] if fit_offset else [])
    A = np.column_stack(cols)

    Aw = A / sem[:, None]
    coeffs, *_ = np.linalg.lstsq(Aw, z / sem, rcond=None)
    cov = np.linalg.inv(Aw.T @ Aw)

    z_pred = A @ coeffs
    residuals = z - z_pred
    dof = max(len(z) - A.shape[1], 1)
    chi2_red = float(np.sum((residuals / sem) ** 2) / dof)
    birge = max(1.0, np.sqrt(chi2_red))
    cov_scaled = cov * birge**2

    Av, Bv = float(coeffs[0]), float(coeffs[1])
    var_a, var_b = float(cov_scaled[0, 0]), float(cov_scaled[1, 1])
    cov_ab = float(cov_scaled[0, 1])

    v2 = Av**2 + Bv**2
    dphi = float(np.arctan2(-Bv, Av))
    visibility = float(np.sqrt(v2))
    if v2 > 0:
        # theta = atan2(-B, A): d/dA = B/V^2, d/dB = -A/V^2
        dphi_var = (Bv**2 * var_a + Av**2 * var_b - 2 * Av * Bv * cov_ab) / v2**2
        # V = sqrt(A^2+B^2): d/dA = A/V, d/dB = B/V
        vis_var = (Av**2 * var_a + Bv**2 * var_b + 2 * Av * Bv * cov_ab) / v2
        dphi_err = float(np.sqrt(max(dphi_var, 0.0)))
        vis_err = float(np.sqrt(max(vis_var, 0.0)))
    else:
        dphi_err = vis_err = float("nan")

    if fit_offset:
        offset, offset_err = float(coeffs[2]), float(np.sqrt(cov_scaled[2, 2]))
    else:
        offset, offset_err = 0.0, 0.0

    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((z - z.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return PhaseFit(
        dphi_comb=dphi, dphi_comb_err=dphi_err,
        visibility=visibility, visibility_err=vis_err,
        offset=offset, offset_err=offset_err,
        chi2_red=chi2_red, dof=dof, birge=birge, r2=r2,
        dphi_slm=dphi_slm, amp=amp, z=z, sem=sem,
        z_pred=z_pred, residuals=residuals,
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
    voltage_std_v: np.ndarray = field(repr=False)
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
    Cells seen once inherit the median positive SEM so weighting stays finite.
    The returned ``y`` is therefore already dark-subtracted.
    """
    y_raw = np.asarray(result.voltage_mean_v, dtype=float)
    if dark_override is None:
        dark_row = np.asarray(result.dark_v, dtype=float)
    else:
        dark_row = np.full(y_raw.shape, float(dark_override))
    y_sub = y_raw - dark_row

    cells: dict[tuple, list[float]] = defaultdict(list)
    key = np.column_stack([result.x_t, result.w_t, result.x_r, result.w_r])
    for row, y in zip(key, y_sub):
        cells[tuple(np.round(row, 9))].append(float(y))

    keys, ys, sem = [], [], []
    for k, vals in sorted(cells.items()):
        arr = np.asarray(vals, dtype=float)
        keys.append(k)
        ys.append(arr.mean())
        sem.append(arr.std(ddof=1) / np.sqrt(arr.size) if arr.size > 1 else np.nan)

    keys = np.asarray(keys, dtype=float)
    ys = np.asarray(ys, dtype=float)
    sem = np.asarray(sem, dtype=float)
    finite = sem[np.isfinite(sem) & (sem > 0)]
    floor = float(np.median(finite)) if finite.size else 1.0
    sem = np.where(np.isfinite(sem) & (sem > 0), sem, floor)
    return keys[:, 0], keys[:, 1], keys[:, 2], keys[:, 3], ys, sem


def fit_result(
    result: PhaseResult,
    tgt_model: PairModel,
    ref_model: PairModel,
    *,
    dark: float | None = None,
    fit_offset: bool = True,
) -> PhaseFit:
    """Isolate the interference term with the step-6 models and fit dPhi_comb.

    Per-row dark-subtracts and averages repeated trials per point (see
    :func:`_average_points`), removes the single-beam response of both pairs and
    the two self-TPA pedestals, then fits the remaining fringe.  ``dark`` (scalar)
    overrides the per-row dark uniformly (e.g. to force the step-6 value).
    """
    x_t, w_t, x_r, w_r, y, sem = _average_points(result, dark_override=dark)

    r_t = tgt_model.amplitude(x_t, w_t)
    r_r = ref_model.amplitude(x_r, w_r)
    # dark was already removed per-row in _average_points; only the
    # cell-dependent single-beam + self-TPA terms remain to subtract
    background = (
        tgt_model.single_beam(x_t, w_t)
        + ref_model.single_beam(x_r, w_r)
        + r_t**2 + r_r**2
    )
    z = y - background
    dphi_slm = slm_phase_diff(x_t, w_t, x_r, w_r)
    amp = 2.0 * r_t * r_r

    result.tgt_model = tgt_model
    result.ref_model = ref_model
    result.fit = fit_phase(dphi_slm, amp, z, sem, fit_offset=fit_offset)
    return result.fit


def swap_invariance(result: PhaseResult):
    """Table-2 diagnostic: |Z(x=a,w=b) - Z(x=b,w=a)| for each swap pair.

    The test runs on the CLEAN interference term, not raw Y, so the known
    single-beam channel asymmetry is removed first (each ``a``/``q`` stays bolted
    to its own physical channel -- they are NOT swapped)::

        Z(x=a,w=b) = Y(a,b) - d - [a_x*a + q_x*a^2 + a_w*b + q_w*b^2] - R^2 - R_0^2
        Z(x=b,w=a) = Y(b,a) - d - [a_x*b + q_x*b^2 + a_w*a + q_w*a^2] - R^2 - R_0^2

    Under the bilinear TPA model ``R = eta*sqrt(x*w)`` and ``dPhi_SLM`` (a channel
    *sum*) are swap-symmetric, so ``Z`` must be too; a residual well above the
    combined SEM flags a genuine channel asymmetry (unequal per-channel
    phase/amplitude law or crosstalk) that survives the step-6 correction.
    Returns ``(x_t, w_t, z, z_swapped, abs_diff, sem)`` for the off-diagonal
    cells.  Falls back to raw Y only if the step-6 models are not attached.
    """
    x_t, w_t, x_r, w_r, y, sem = _average_points(result)   # y already dark-subtracted
    tgt, ref = result.tgt_model, result.ref_model
    if tgt is not None and ref is not None:
        r_t = tgt.amplitude(x_t, w_t)
        r_r = ref.amplitude(x_r, w_r)
        background = (tgt.single_beam(x_t, w_t)
                      + ref.single_beam(x_r, w_r) + r_t**2 + r_r**2)
        sig = y - background        # clean interference term Z
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
# drive builders
# ======================================================================

def build_phase_sweep(
    *,
    n_points: int = 15,
    phi_start_deg: float = 0.0,
    phi_stop_deg: float = 180.0,
    ref_phase_deg: float = 180.0,
) -> list[tuple[float, float, float, float]]:
    """Table 1: symmetric target phase sweep vs a fixed reference (half fringe).

    The target pair is driven symmetrically ``phi^x = phi^w = phi`` over
    ``[phi_start_deg, phi_stop_deg]`` (default 0..180 deg -- the full reachable
    half turn), the reference pair fixed at ``ref_phase_deg`` on both channels
    (default 180 deg == intensity 1, fully on).  Returns target-first commanded
    intensity tuples ``(x_t, w_t, x_r, w_r)`` with ``x = sin(phi/2)^2``, so
    ``dPhi_SLM = phi - ref_phase`` sweeps the fringe.
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
    """Table 2: 3x3 grid on the target's individual channel phases (symmetry check).

    Sweeps ``phi^x`` and ``phi^w`` of the target *independently* over
    ``phi_values_deg`` with the reference fixed, so swapped cells and equal-sum
    cells can be compared (see :func:`swap_invariance`).  Returns target-first
    commanded intensity tuples.
    """
    x_r = float(intensity_for_phase(np.radians(ref_phase_deg)))
    out: list[tuple[float, float, float, float]] = []
    for px in phi_values_deg:
        xt = float(intensity_for_phase(np.radians(px)))
        for pw in phi_values_deg:
            wt = float(intensity_for_phase(np.radians(pw)))
            out.append((xt, wt, x_r, x_r))
    return out


# ======================================================================
# measurement  (instrument-agnostic two-pair sweep)
# ======================================================================

@dataclass
class TPAPhaseProgress:
    step: int
    total: int
    message: str
    dphi_comb: float | None = None


ProgressCallback = Callable[["TPAPhaseProgress"], None]


def _read_mean_std(monitor, repeats: int, timeout: float) -> tuple[float, float]:
    """Averaged reading + the noise of the recorded waveform behind it."""
    means: list[float] = []
    variances: list[float] = []
    for _ in range(max(1, repeats)):
        sample = monitor.monitor_cycle(timeout=timeout)
        if sample is None:
            raise TPAPhaseAborted("monitor read aborted")
        means.append(float(sample.value))
        waveform = getattr(monitor, "last_values", None)
        if waveform is not None and np.size(waveform) > 1:
            variances.append(float(np.var(waveform)))
    mean_v = float(np.mean(means))
    std_v = float(np.sqrt(np.mean(variances))) if variances else 0.0
    return mean_v, std_v


def measure_phase_sweep(
    monitor,
    slm,
    layout,
    *,
    tgt_index: int,
    ref_index: int,
    drive: Sequence[tuple[float, float, float, float]],
    tgt_model: PairModel,
    ref_model: PairModel,
    n_trials: int = 1,
    repeats: int = 1,
    settle: float = 0.15,
    read_timeout: float = 30.0,
    measure_dark: bool = True,
    dark_per_trial: bool = True,
    stop_event: threading.Event | None = None,
    progress_callback: ProgressCallback | None = None,
) -> PhaseResult:
    """Drive target + reference over ``drive``, read Y at each point, fit dPhi_comb.

    ``monitor`` must already be configured (caller runs ``configure_monitor``);
    this only calls ``monitor_cycle``.  Only channels ``tgt_index`` and
    ``ref_index`` are driven; all others held off.  ``drive`` tuples are
    target-first ``(x_t, w_t, x_r, w_r)`` intensities.

    Dark handling: with ``measure_dark`` an all-off reading is taken and stored
    per row for per-row subtraction (drift removal).  ``dark_per_trial`` (default)
    takes a fresh all-off reading at the START OF EACH TRIAL, so slow dark drift
    over the run is tracked; set it False to take a single all-off reading once at
    the start.  If ``measure_dark`` is False the mean of the two step-6 darks is
    used for every row.  Raises :class:`TPAPhaseAborted` if ``stop_event`` is set.
    """
    n = layout.n_channels
    for name, idx in (("tgt_index", tgt_index), ("ref_index", ref_index)):
        if not (0 <= idx < n):
            raise ValueError(f"{name}={idx} out of range (layout has {n} pairs)")
    if tgt_index == ref_index:
        raise ValueError("tgt_index and ref_index must differ")

    zeros = np.zeros(n)
    slm_width, slm_height = slm.get_slm_info()
    from .encoding import encode_to_pattern

    def _check_stop() -> None:
        if stop_event is not None and stop_event.is_set():
            raise TPAPhaseAborted("phase sweep stopped by request")

    def _display(x_t, w_t, x_r, w_r) -> None:
        x_vals = zeros.copy()
        w_vals = zeros.copy()
        x_vals[tgt_index], w_vals[tgt_index] = x_t, w_t
        x_vals[ref_index], w_vals[ref_index] = x_r, w_r
        pattern = encode_to_pattern(x_vals, w_vals, layout, slm_width, slm_height)
        slm.display_array(pattern)
        if settle:
            time.sleep(settle)

    # dark handling: step-6 mean is the fallback; a measured all-off reading
    # overrides it (once, or per trial for drift tracking)
    fallback_dark = 0.5 * (tgt_model.d + ref_model.d)

    def _read_dark(trial: int, step: int, total: int) -> float:
        _check_stop()
        _display(0.0, 0.0, 0.0, 0.0)
        d, _ = _read_mean_std(monitor, repeats, read_timeout)
        if progress_callback is not None:
            progress_callback(TPAPhaseProgress(
                step=step, total=total,
                message=f"trial {trial} dark (all off) = {d*1000:.4f} mV"))
        return d

    drive = list(drive)
    reads_per_trial = len(drive) + (1 if measure_dark and dark_per_trial else 0)
    total = max(n_trials * reads_per_trial + (1 if measure_dark and not dark_per_trial else 0), 1)

    start_dark = fallback_dark
    step = 0
    if measure_dark and not dark_per_trial:
        step += 1
        start_dark = _read_dark(0, step, total)

    rows: list[tuple[int, float, float, float, float, float, float, float]] = []
    for trial in range(n_trials):
        if measure_dark and dark_per_trial:
            step += 1
            trial_dark = _read_dark(trial, step, total)
        elif measure_dark:
            trial_dark = start_dark
        else:
            trial_dark = fallback_dark
        for x_t, w_t, x_r, w_r in drive:
            _check_stop()
            _display(x_t, w_t, x_r, w_r)
            mean_v, std_v = _read_mean_std(monitor, repeats, read_timeout)
            rows.append((trial, x_t, w_t, x_r, w_r, mean_v, std_v, trial_dark))
            step += 1
            if progress_callback is not None:
                dphi_slm = float(slm_phase_diff(x_t, w_t, x_r, w_r))
                phi_t = float(np.degrees(2.0 * phi_half(x_t)))
                progress_callback(
                    TPAPhaseProgress(
                        step=step, total=total,
                        message=(
                            f"trial {trial} phi_t={phi_t:.1f}deg "
                            f"dPhi_SLM={np.degrees(dphi_slm):+.1f}deg "
                            f"-> {mean_v*1000:.4f} mV (dark {trial_dark*1000:.4f})"
                        ),
                    )
                )

    result = PhaseResult(
        tgt_index=tgt_index, ref_index=ref_index,
        trial=np.array([r[0] for r in rows], dtype=int),
        x_t=np.array([r[1] for r in rows], dtype=float),
        w_t=np.array([r[2] for r in rows], dtype=float),
        x_r=np.array([r[3] for r in rows], dtype=float),
        w_r=np.array([r[4] for r in rows], dtype=float),
        voltage_mean_v=np.array([r[5] for r in rows], dtype=float),
        voltage_std_v=np.array([r[6] for r in rows], dtype=float),
        dark_v=np.array([r[7] for r in rows], dtype=float),
        n_trials=n_trials,
    )
    fit_result(result, tgt_model, ref_model)
    if progress_callback is not None and result.fit is not None:
        progress_callback(
            TPAPhaseProgress(
                step=total, total=total,
                message=(
                    f"fit: dPhi_comb = {np.degrees(result.fit.dphi_comb):+.2f} deg "
                    f"(V = {result.fit.visibility:.3f})"
                ),
                dphi_comb=result.fit.dphi_comb,
            )
        )
    return result


# ======================================================================
# persistence
# ======================================================================

_CSV_HEADER = [
    "trial", "tgt_index", "ref_index",
    "phi_xt_deg", "phi_wt_deg", "x_t", "w_t", "x_r", "w_r",
    "dark_v", "voltage_mean_v", "voltage_std_v",
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


def load_phase_csv(
    path: str | Path,
    tgt_model: PairModel,
    ref_model: PairModel,
    *,
    dark: float | None = None,
) -> PhaseResult:
    """Load a raw phase-sweep CSV and re-fit dPhi_comb with the given step-6 models.

    The per-row ``dark_v`` column is used when present; otherwise the scalar
    ``# dark_mean_v`` (or legacy ``# dark_v``) comment, then the step-6 mean, is
    filled for every row.  ``dark`` (scalar) overrides all of them uniformly.
    """
    file_dark: float | None = None
    with open(Path(path), newline="", encoding="utf-8") as f:
        for raw in f:
            if raw.startswith("#"):
                parts = raw.lstrip("#").strip().split(",")
                if len(parts) == 2 and parts[0].strip() in ("dark_mean_v", "dark_v"):
                    file_dark = float(parts[1])

    rows: list[tuple[int, float, float, float, float, float, float, float | None]] = []
    tgt_index, ref_index = tgt_model.index, ref_model.index
    with open(Path(path), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(line for line in f if not line.startswith("#")):
            dv = row.get("dark_v")
            rows.append((
                int(float(row.get("trial", 0))),
                float(row["x_t"]), float(row["w_t"]),
                float(row["x_r"]), float(row["w_r"]),
                float(row["voltage_mean_v"]),
                float(row.get("voltage_std_v", "nan") or "nan"),
                float(dv) if dv not in (None, "") else None,
            ))
            tgt_index = int(float(row.get("tgt_index", tgt_index)))
            ref_index = int(float(row.get("ref_index", ref_index)))

    trials = np.array([r[0] for r in rows], dtype=int)
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
    )
    fit_result(result, tgt_model, ref_model)
    return result


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
        "fit": None if fit is None else {
            "dphi_comb_rad": fit.dphi_comb,
            "dphi_comb_deg": fit.dphi_comb_deg,
            "dphi_comb_err_rad": fit.dphi_comb_err,
            "dphi_comb_err_deg": float(np.degrees(fit.dphi_comb_err)),
            "visibility": fit.visibility,
            "visibility_err": fit.visibility_err,
            "offset_v": fit.offset,
            "chi2_red": fit.chi2_red,
            "dof": fit.dof,
            "birge": fit.birge,
            "r2": fit.r2,
        },
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(out)


__all__ = [
    "PARAMS",
    "TPAPhaseAborted",
    "TPAPhaseProgress",
    "PairModel",
    "PhaseFit",
    "PhaseResult",
    "load_pair_models",
    "phi_half",
    "intensity_for_phase",
    "slm_phase_diff",
    "fit_phase",
    "fit_result",
    "swap_invariance",
    "build_phase_sweep",
    "build_symmetry_grid",
    "measure_phase_sweep",
    "write_phase_csv",
    "load_phase_csv",
    "save_phase_json",
]
