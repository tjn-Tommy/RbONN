"""Synthetic data generator for the step-6 / step-7 calibration chain.

No hardware.  Builds ground-truth parameters for 5 pairs (= 10 SLM channels),
evaluates the full forward model, adds noise, writes CSVs in exactly the formats
the real scripts read, then optionally re-fits them with the real fitters and
prints fitted-vs-truth::

    python src/calibration_module/steps/calib_synth_v1.py             # generate + check
    python src/calibration_module/steps/calib_synth_v1.py --no-check  # generate only
    python src/calibration_module/steps/calib_synth_v1.py --seed 7    # another noise draw

In:  the TRUTH block below -- nothing else.
Out: into ``OUT_DIR``, ``synth_truth_*.json`` (the ground truth),
     ``synth_step6_*.csv`` (re-fits with ``calib_step6_v2.py <csv>`` or
     ``fit.pair.load_tpa_pair_csv``) and ``synth_step7_*.csv`` (re-fits with
     ``calib_step7_v2.py <csv>``).

Pairs are numbered from :data:`PAIR_INDEX_BASE`, so they are 1..5 with pair 1
the reference, and that number is what every CSV's ``pair_index`` /
``tgt_index`` column carries.  The truth arrays stay 0-based; :func:`_slot` maps
a pair number onto its slot.

The field model
---------------
Each SLM channel carries a comb line.  Pair ``n`` uses two, ``x`` and ``w``, and
the panel puts a phase depth ``theta`` on each::

    x_n = C_n^x sin(theta_n^x / 2) exp(+j theta_n^x / 2) exp(j Psi_n^x)
    w_n = C_n^w sin(theta_n^w / 2) exp(+j theta_n^w / 2) exp(j Psi_n^w)

``theta`` is what is tuned (the commanded intensity is ``I = sin^2(theta/2)``,
and the CSVs record ``I``), ``C`` is the channel's power, ``Psi`` the comb phase
it arrives with.  The panel phase ADDS, so a pair's field is
``eta sqrt(x w) exp(j[u + v + Phi])`` and the fringe is
``cos(dPhi_SLM + dPhi_comb)`` -- THE model, the one every fit in the project is
taken under, generator and fitters agreeing without a sign flip anywhere.

With the pair's two lines sitting SYMMETRICALLY about the two-photon centre
(``+n`` and ``-n``, ``Delta = 2 pi f_rep``)::

    Psi_i^w = beta1 (+m_i Delta) + beta2 (+m_i Delta)^2
    Psi_i^x = beta1 (-m_i Delta) + beta2 (-m_i Delta)^2
    -> Phi_i = Psi_i^x + Psi_i^w = 2 beta2 (m_i Delta)^2 = 2 beta2 Omega_i^2

so the group delay ``beta1`` CANCELS within a pair and only the second-order
dispersion survives.  Against the reference that is
``dPhi_comb,i = 2 beta2 (Omega_i^2 - Omega_1^2)``, the spectrum step 7 recovers
-- which needs the REAL comb lines, not the pair index; see the COMB_LINE block.

The measured signal
-------------------
Two-photon absorption is coherent across pairs, single-beam response is not::

    Y = | sum_n eta_n x_n w_n |^2            (TPA, interferes)
      + sum_channels (a_c I_c + q_c I_c^2)   (single beam, adds)
      + d                                    (dark)

One pair on its own collapses to step 6's
``Y = eta_fit^2 (x w) + a_x x + q_x x^2 + a_w w + q_w w^2 + d`` with
``eta_fit = eta_n C_n^x C_n^w``: the step-6 fit CANNOT separate the intrinsic
``eta`` from the channel powers, it only ever sees the product.  Two pairs on
gives step 7's fringe.

What is recoverable
-------------------
From the data: ``eta_fit = eta C^x C^w`` (5), ``a``/``q`` per channel (20),
``d`` (1), ``dPhi_comb`` per target (4), and from those the single slope
``beta2`` (1).  NOT recoverable: ``eta`` and ``C`` separately, ``beta1`` (it
cancels in every pair), and the absolute ``Phi_n`` (only differences against the
reference are observable, and only mod 2 pi).
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from calibration_module.fit.pair import load_tpa_pair_csv  # noqa: E402
from calibration_module.fit.phase import (  # noqa: E402
    PairModel,
    load_phase_csv,
)

# ======================================================================
# TRUTH  -- edit these; everything downstream is derived
# ======================================================================

N_PAIRS = 5                      # 5 pairs = 10 SLM channels (an x and a w each)
PAIR_INDEX_BASE = 1              # pairs are numbered 1..5 (what the CSVs record)
PAIRS = [PAIR_INDEX_BASE + s for s in range(N_PAIRS)]
OUT_DIR = REPO_ROOT / "src/calib_data"

# ---- comb + dispersion ----
F_REP_HZ = 80e6                  # comb repetition rate
DELTA = 2.0 * np.pi * F_REP_HZ   # rad/s per comb line
BETA1_S = 1e-9                   # group delay          (s)      -- cancels per pair
BETA2_S2 = 1e4 * (1e-15) ** 2    # 1e4 fs^2 of GDD      (s^2)    -- what step 7 sees

# Which comb line each pair sits on (its x line is the mirror, -n).  Set these
# to your real channel frequencies.  One line from centre gives Phi ~ 5e-9 rad
# at beta2 = 1e4 fs^2 -- unmeasurable -- so the pairs sit thousands of lines
# out, where beta2 bites: 1e4 lines is 0.8 THz (~1.6 nm) and Phi ~ 0.5 rad.
#
# Written as an OFFSET plus a per-pair step, the form step 7's verification
# config takes.  The offset MATTERS: the regressor is Omega^2, and
#     Phi_n - Phi_0 = 2 beta2 Delta^2 [dm^2 n^2 + 2 m0 dm n]
# carries a LINEAR-in-n term, so re-indexing to a bare pair index is not a
# relabelling -- a pure n^2 model cannot absorb it and beta2 comes out ~4x
# wrong.  Keep m0, or fit the n term too.
COMB_LINE_0 = 1_100              # comb line of the FIRST pair (index PAIR_INDEX_BASE)
COMB_LINE_STEP = 2_000           # dm: comb lines per pair index
# -> {1: 1100, 2: 3100, 3: 5100, 4: 7100, 5: 9100}
COMB_LINE = {i: COMB_LINE_0 + (i - PAIR_INDEX_BASE) * COMB_LINE_STEP for i in PAIRS}

# ---- per-pair TPA efficiency (intrinsic; the fit sees eta * C^x * C^w) ----
ETA = [0.170, 0.152, 0.166, 0.158, 0.148]          # V^0.5

# ---- per-channel power C (10 values: an x and a w per pair) ----
C_X = [1.00, 0.97, 0.94, 1.03, 0.99]
C_W = [0.96, 1.01, 0.95, 1.02, 0.98]

# ---- per-channel single-beam response a*I + q*I^2, in volts ----
# Real pairs sit at a few tens of uV against a ~25 mV cross term.
#
# The quadratic q is OFF to match calib_step6_v2's FIT_Q: fitting it saturates
# the 5-level background block and leaves no residual to check.  With q = 0 in
# the data the linear a is the whole truth and the 3-parameter fit has 2 dof.
# Turn one back on (e.g. Q_X[0] = 0.627e-3, what the real pair 1 showed) to
# check that step 6's "arm is curved" pull actually fires.
A_X = [0.029e-3, 0.040e-3, 0.012e-3, 0.006e-3, 0.021e-3]
Q_X = [0.0, 0.0, 0.0, 0.0, 0.0]
A_W = [0.011e-3, 0.025e-3, 0.033e-3, 0.040e-3, 0.009e-3]
Q_W = [0.0, 0.0, 0.0, 0.0, 0.0]

# ---- the one shared dark offset ----
D_V = -3.5e-5                    # -0.035 mV

# ---- noise (this is what the DAQ reports as the per-point trace std) ----
NOISE_ABS_V = 5e-5               # 0.05 mV detector/dark floor
NOISE_REL = 2e-3                 # + 0.2% of |Y| (laser/shot-like)
SEED = 0                         # RNG seed; --seed N overrides

# ---- grids (must match the real scripts, or the CSVs will not refit) ----
# Step 6: the calib_step6_v2.py GRID, (x, w, n_repeats) per pair.
STEP6_GRID: tuple[tuple[float, float, int], ...] = (
    (0.0, 0.00, 2),   # dark    -> d
    (0.5, 0.00, 2),   # x-only  -> a_x, q_x
    (1.0, 0.00, 3),   # x-only  -> a_x, q_x ; also the direct D(0) anchor
    (0.0, 0.50, 2),   # w-only  -> a_w, q_w
    (0.0, 1.00, 2),   # w-only  -> a_w, q_w
    (1.0, 0.20, 4),   # cross   -> eta^2
    (1.0, 0.45, 4),   # cross
    (1.0, 0.70, 4),   # cross
    (1.0, 0.90, 6),   # cross   (highest leverage -> most repeats)
    (1.0, 1.00, 2),   # cross, excluded from v2's slope fit
)

# Step 7: the calib_step7_v2.py sweep -- reference fully on, the target's two
# channels swept together over this ramp.
REF_INDEX = 1                    # pair 1 is the reference (Phi = 0)
TGT_INDICES = [2, 3, 4, 5]
SWEEP_MIN = 0.1
SWEEP_MAX = 1.0
N_SWEEP_POINTS = 10


# ======================================================================
# the forward model
# ======================================================================

def _slot(pair: int) -> int:
    """Pair number (1-based) -> its slot in the 0-based truth arrays."""
    return pair - PAIR_INDEX_BASE


def theta(intensity) -> np.ndarray:
    """Panel phase depth for a commanded intensity: ``theta = 2 asin(sqrt(I))``."""
    return 2.0 * np.arcsin(np.sqrt(np.clip(np.asarray(intensity, dtype=float), 0.0, 1.0)))


def psi(pair: int, side: str) -> float:
    """Comb phase of one channel: ``beta1 (m Delta) + beta2 (m Delta)^2``.

    The pair's two lines sit symmetrically about the two-photon centre, so the
    ``w`` channel is comb line ``+m`` and the ``x`` channel ``-m``.
    """
    m = COMB_LINE[pair] * (1.0 if side == "w" else -1.0)
    om = m * DELTA
    return BETA1_S * om + BETA2_S2 * om**2


def comb_phase(pair: int) -> float:
    """Pair comb phase ``Phi_n = Psi^x + Psi^w = 2 beta2 (m Delta)^2`` (beta1 cancels)."""
    return psi(pair, "x") + psi(pair, "w")


def omega(pair: int) -> float:
    """The pair's detuning from the comb centre, ``m Delta`` (rad/s)."""
    return COMB_LINE[pair] * DELTA


def eta_fit(pair: int) -> float:
    """What step 6 actually recovers: ``eta_i * C_i^x * C_i^w``."""
    s = _slot(pair)
    return ETA[s] * C_X[s] * C_W[s]


def signal(x_vals, w_vals) -> float:
    """Noise-free ``Y`` for one SLM pattern.

    In: commanded intensities in [0, 1], one per pair -- the shape
    ``encode_to_pattern`` takes.  TPA is coherent, so the pair terms are summed
    as FIELDS and squared; single-beam response adds incoherently.
    """
    x_vals = np.asarray(x_vals, dtype=float)
    w_vals = np.asarray(w_vals, dtype=float)
    th_x, th_w = theta(x_vals), theta(w_vals)

    field = 0.0 + 0.0j
    single = 0.0
    for i in PAIRS:
        s = _slot(i)
        # x_i and w_i exactly as written in the module docstring
        xi = C_X[s] * np.sin(th_x[s] / 2.0) * np.exp(1j * th_x[s] / 2.0 + 1j * psi(i, "x"))
        wi = C_W[s] * np.sin(th_w[s] / 2.0) * np.exp(1j * th_w[s] / 2.0 + 1j * psi(i, "w"))
        field += ETA[s] * xi * wi
        single += (A_X[s] * x_vals[s] + Q_X[s] * x_vals[s] ** 2
                   + A_W[s] * w_vals[s] + Q_W[s] * w_vals[s] ** 2)
    return float(abs(field) ** 2 + single + D_V)


def measure(x_vals, w_vals, rng) -> tuple[float, float]:
    """One noisy acquisition -> ``(voltage_mean_v, voltage_std_v)``.

    ``std`` is what the DAQ would report for the low-passed trace, and what
    every fit weights by, so it goes to the CSV as-is: a fixed detector floor in
    quadrature with a signal-proportional term.
    """
    y = signal(x_vals, w_vals)
    sigma = float(np.hypot(NOISE_ABS_V, NOISE_REL * abs(y)))
    return float(y + rng.normal(0.0, sigma)), sigma


def _drive(pairs_on: dict[int, tuple[float, float]]):
    """Commanded-intensity arrays with only ``{pair: (x, w)}`` driven, rest off."""
    x_vals = np.zeros(N_PAIRS)
    w_vals = np.zeros(N_PAIRS)
    for i, (x, w) in pairs_on.items():
        x_vals[_slot(i)], w_vals[_slot(i)] = x, w
    return x_vals, w_vals


# ======================================================================
# writers  (formats the real scripts read)
# ======================================================================

_STEP6_HEADER = ["trial", "pair_index", "x", "w", "product",
                 "voltage_mean_v", "voltage_std_v", "std_ratio"]

_STEP7_HEADER = ["trial", "tgt_index", "ref_index",
                 "phi_xt_deg", "phi_wt_deg", "x_t", "w_t", "x_r", "w_r",
                 "dark_v", "voltage_mean_v", "voltage_std_v", "std_ratio"]


def _ratio(mean_v: float, std_v: float) -> float:
    return abs(std_v / mean_v) if mean_v else float("inf")


def gen_step6_csv(path: Path, rng) -> Path:
    """Step-6 grid: one pair at a time over :data:`STEP6_GRID`, repeats included.

    Only that pair's two channels are driven, so the TPA term collapses to
    ``eta_fit^2 (x w)`` -- no comb phase, no interference, which is why step 6
    can calibrate eta without knowing any phase.
    """
    rows = []
    for i in PAIRS:
        for x, w, n_rep in STEP6_GRID:
            for rep in range(n_rep):
                mean_v, std_v = measure(*_drive({i: (x, w)}), rng)
                rows.append([rep, i, f"{x:.6g}", f"{w:.6g}", f"{x*w:.6g}",
                             f"{mean_v:.9g}", f"{std_v:.9g}",
                             f"{_ratio(mean_v, std_v):.6g}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_STEP6_HEADER)
        writer.writerows(rows)
    print(f"Step-6 CSV: {len(rows)} rows, pairs {PAIRS} -> {path}")
    return path


def gen_step7_csv(path: Path, rng) -> Path:
    """Step-7 sweep: reference fully on, each target's two channels swept together.

    One all-off dark per target (stored per row, as the hardware script does),
    then the ``SWEEP_MIN..SWEEP_MAX`` ramp with ``x_t = w_t``.  Both pairs are on
    at every sweep point, so the two pair fields interfere and the fringe carries
    ``dPhi_comb = Phi_t - Phi_r``.
    """
    rows = []
    values = np.round(np.linspace(SWEEP_MIN, SWEEP_MAX, N_SWEEP_POINTS), 6)
    for k in TGT_INDICES:
        dark_v, _ = measure(*_drive({}), rng)               # all off, once per target
        for v in values:
            x_t = w_t = float(v)
            x_r = w_r = 1.0
            mean_v, std_v = measure(*_drive({k: (x_t, w_t), REF_INDEX: (x_r, w_r)}), rng)
            th = np.degrees(theta(x_t))
            rows.append([0, k, REF_INDEX, f"{th:.4g}", f"{th:.4g}",
                         f"{x_t:.6g}", f"{w_t:.6g}", f"{x_r:.6g}", f"{w_r:.6g}",
                         f"{dark_v:.9g}", f"{mean_v:.9g}", f"{std_v:.9g}",
                         f"{_ratio(mean_v, std_v):.6g}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_STEP7_HEADER)
        writer.writerows(rows)
    print(f"Step-7 CSV: {len(rows)} rows, targets {TGT_INDICES} vs ref {REF_INDEX} -> {path}")
    return path


def truth_payload() -> dict:
    """Everything that was used to build the data, for the truth JSON + printing."""
    return {
        "n_pairs": N_PAIRS,
        "n_channels": 2 * N_PAIRS,
        "pair_index_base": PAIR_INDEX_BASE,
        "ref_index": REF_INDEX,
        "comb": {
            "f_rep_hz": F_REP_HZ, "delta_rad_per_s": DELTA,
            "beta1_s": BETA1_S, "beta2_s2": BETA2_S2,
            "beta2_fs2": BETA2_S2 / (1e-15) ** 2,
            "comb_line": {str(i): COMB_LINE[i] for i in PAIRS},
            "note": "x channel of pair i sits on comb line -m_i, w on +m_i",
        },
        "noise": {"abs_v": NOISE_ABS_V, "rel": NOISE_REL, "seed": SEED},
        "dark_v": D_V,
        "pairs": [
            {
                "index": i,
                "eta": ETA[_slot(i)], "C_x": C_X[_slot(i)], "C_w": C_W[_slot(i)],
                "eta_fit_expected": eta_fit(i),      # eta * C_x * C_w -- what step 6 sees
                "a_x": A_X[_slot(i)], "q_x": Q_X[_slot(i)],
                "a_w": A_W[_slot(i)], "q_w": Q_W[_slot(i)],
                "comb_line": COMB_LINE[i],
                "omega_rad_per_s": omega(i),
                "psi_x_rad": psi(i, "x"), "psi_w_rad": psi(i, "w"),
                "phi_rad": comb_phase(i),
                "dphi_comb_vs_ref_rad": comb_phase(i) - comb_phase(REF_INDEX),
                "dphi_comb_vs_ref_deg": float(np.degrees(comb_phase(i) - comb_phase(REF_INDEX))),
            }
            for i in PAIRS
        ],
        # Phi_t - Phi_r = 2 beta2 Delta^2 (m_t^2 - m_r^2), i.e. slope -2*beta2
        # against u = Omega_r^2 - Omega_t^2.  Recorded as truth only.
        "beta2_slope_vs_u_s2": -2.0 * BETA2_S2,
    }


def print_truth() -> None:
    """The parameter table: what was set, and what is recoverable from the data."""
    print(f"\n=== Ground truth: {N_PAIRS} pairs / {2*N_PAIRS} channels "
          f"(numbered {PAIRS[0]}..{PAIRS[-1]}, reference = pair {REF_INDEX}) ===")
    print(f"comb: f_rep = {F_REP_HZ/1e6:.0f} MHz -> Delta = {DELTA:.4g} rad/s;  "
          f"beta1 = {BETA1_S:.3g} s (cancels per pair);  "
          f"beta2 = {BETA2_S2/(1e-15)**2:.4g} fs^2")
    print(f"dark d = {D_V*1e3:+.4f} mV (one, shared);  "
          f"noise sigma = hypot({NOISE_ABS_V*1e3:.3f} mV, {NOISE_REL*100:.2f}% of Y)")
    print(f"{'pair':>4} {'eta':>7} {'C_x':>6} {'C_w':>6} {'eta*CxCw':>9} "
          f"{'a_x':>8} {'q_x':>8} {'a_w':>8} {'q_w':>8} {'line':>7} "
          f"{'Phi':>9} {'dPhi vs ref':>12}")
    for i in PAIRS:
        s = _slot(i)
        d = np.degrees(comb_phase(i) - comb_phase(REF_INDEX))
        print(f"{i:>4} {ETA[s]:>7.4f} {C_X[s]:>6.3f} {C_W[s]:>6.3f} {eta_fit(i):>9.5f} "
              f"{A_X[s]*1e3:>8.4f} {Q_X[s]*1e3:>8.4f} {A_W[s]*1e3:>8.4f} {Q_W[s]*1e3:>8.4f} "
              f"{COMB_LINE[i]:>7d} {comb_phase(i):>9.4f} {d:>+11.2f}deg")
    print(f"   (a/q in mV, Phi = 2*beta2*Omega_i^2, dPhi vs pair {REF_INDEX})")
    print("Recoverable: eta*CxCw (5), a/q per channel (20), d (1), dPhi_comb (4), beta2 (1).")
    print("NOT recoverable: eta and C separately, beta1, absolute Phi (only diffs, mod 2pi).")


# ======================================================================
# check  (refit the synthetic CSVs with the real fitters)
# ======================================================================

def _dev(fit: float, true: float) -> str:
    """'value (+x.x% vs truth)' -- percent when truth is non-zero, else absolute."""
    if true:
        return f"{fit:>10.5f} ({(fit/true - 1)*100:>+7.2f}%)"
    return f"{fit:>10.5f} ({fit - true:>+9.2e})"


def check_step6(csv_path: Path) -> dict[int, PairModel]:
    """Re-fit the synthetic step-6 CSV and compare every parameter to truth."""
    print("\n=== Check: step-6 fit vs truth ===")
    result = load_tpa_pair_csv(csv_path)
    models: dict[int, PairModel] = {}
    truth = {"a_x": A_X, "q_x": Q_X, "a_w": A_W, "q_w": Q_W}
    for grid in result.channels:
        i, fit = grid.index, grid.fit
        if fit is None:
            print(f"pair {i}: FIT FAILED")
            continue
        models[i] = PairModel.from_fit(i, fit)
        print(f"pair {i}  R^2 = {fit.r2:.6f}")
        esig = abs(fit.eta - eta_fit(i)) / fit.eta_err if fit.eta_err else float("nan")
        print(f"   eta      truth {eta_fit(i):>10.5f}  fit {_dev(fit.eta, eta_fit(i))}"
              f"  +/- {fit.eta_err:.5f} ({esig:.1f} sigma)")
        for name in ("a_x", "q_x", "a_w", "q_w"):
            val, err = fit.params[name]
            t = truth[name][_slot(i)]
            sig = abs(val - t) / err if err else float("nan")
            print(f"   {name:<8} truth {t*1e3:>+9.4f} mV  fit {val*1e3:>+9.4f} "
                  f"+/- {err*1e3:.4f} mV  ({sig:.1f} sigma)")
        val, err = fit.params["d"]
        sig = abs(val - D_V) / err if err else float("nan")
        print(f"   {'d':<8} truth {D_V*1e3:>+9.4f} mV  fit {val*1e3:>+9.4f} "
              f"+/- {err*1e3:.4f} mV  ({sig:.1f} sigma)")
    return models


def check_step7(csv_path: Path, models: dict[int, PairModel]) -> None:
    """Re-fit dPhi_comb per target with the step-7 v2 fit, compare against truth.

    Uses the models FITTED from the synthetic step-6 CSV, not the truth values,
    so this exercises the real chain with step-6 errors and all.  Which is why
    the quoted sigma must be ``dphi_comb_err_total``: the fit pins a and b to
    those models, so its own ``dphi_comb_err`` is fringe noise alone and step
    6's eta error is invisible in it, not absent.  Judging against truth on the
    fringe term alone reads several sigma off on data that is fine.
    """
    print("\n=== Check: step-7 v2 dPhi_comb vs truth ===")
    print("sigma = fringe noise (+) the PINNED step-6 eta, in quadrature "
          "(dphi_comb_err_total)")
    if REF_INDEX not in models:
        print(f"no step-6 model for reference pair {REF_INDEX}; skipping")
        return
    fits = {}
    for k in TGT_INDICES:
        if k not in models:
            continue
        res = load_phase_csv(csv_path, models[k], models[REF_INDEX],
                             comb_only=True, single_beam_bg=True, only_tgt=k)
        fit = res.fit
        fits[k] = fit
        true = comb_phase(k) - comb_phase(REF_INDEX)
        true = float(np.arctan2(np.sin(true), np.cos(true)))     # same (-pi, pi] branch
        err = fit.dphi_comb_err_total
        sig = abs(fit.dphi_comb - true) / err if err else float("nan")
        print(f"pair {k}:  truth {np.degrees(true):>+8.3f} deg   "
              f"fit {fit.dphi_comb_deg:>+8.3f} +/- {np.degrees(err):.3f} deg   "
              f"({sig:>4.1f} sigma)   R^2 = {fit.r2:.5f}")
        print(f"          sigma = {np.degrees(fit.dphi_comb_err):.3f} (fringe) "
              f"(+) {np.degrees(fit.dphi_comb_err_eta):.3f} (step-6 eta) deg   "
              f"[d(dPhi)/d(eta): ref {fit.dphi_deta_ref:+.2f}, "
              f"tgt {fit.dphi_deta_tgt:+.2f} rad per unit eta]")


# ======================================================================

def main(argv: list[str] | None = None) -> int:
    global SEED
    argv = sys.argv[1:] if argv is None else argv
    if "--seed" in argv:
        SEED = int(argv[argv.index("--seed") + 1])
    rng = np.random.default_rng(SEED)

    print_truth()
    stamp = time.strftime("%m%d_%H%M")
    truth_path = OUT_DIR / f"synth_truth_{stamp}.json"
    truth_path.parent.mkdir(parents=True, exist_ok=True)
    truth_path.write_text(json.dumps(truth_payload(), indent=2), encoding="utf-8")
    print(f"\nTruth JSON: {truth_path}")

    step6_csv = gen_step6_csv(OUT_DIR / f"synth_step6_{stamp}.csv", rng)
    step7_csv = gen_step7_csv(OUT_DIR / f"synth_step7_{stamp}.csv", rng)

    if "--no-check" in argv:
        print(f"\nFit with:  python calib_step6_v2.py {step6_csv}")
        print(f"           python calib_step7_v2.py {step7_csv}   "
              f"(point IN_STEP6 at the step-6 result first)")
        return 0
    models = check_step6(step6_csv)
    check_step7(step7_csv, models)
    return 0


if __name__ == "__main__":
    sys.exit(main())
