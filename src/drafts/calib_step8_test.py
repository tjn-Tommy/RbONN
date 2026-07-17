"""Manual smoke test: grid-sweep check of the multi-pair coherent forward model.

Steps 3-7 calibrated everything the forward model needs: the channel layout
(step 3), each pair's TPA efficiency eta + single-beam / dark backgrounds
(step 6) and each pair's comb phase Phi_k relative to the reference pair
(step 7).  This step closes the loop: drive several pairs AT ONCE over a
full-factorial GRID -- each channel swept over ``SWEEP_POINTS`` levels in
``[SWEEP_MIN, SWEEP_MAX]`` with ``x_k = w_k`` (both beams of a pair share the
level, as in step 7's target sweep), so ``CHANNELS = [1, 3, 5]`` gives a
5 x 5 x 5 = 125-point grid -- and compare the measured detector output against
the zero-free-parameter prediction::

    E      = sum_k  eta_k sqrt(x_k w_k) exp(i [phi_half(x_k) + phi_half(w_k) + Phi_k])
    Y_pred = |E|^2 + sum_k single_beam_k(x_k, w_k)          (+ measured dark)

with ``phi_half(x) = asin(sqrt(x))`` (the channel's field phase) and
``Phi_ref = 0``.  Every quantity comes from the calibrations -- nothing is
fitted to the new data.  Two a-posteriori drift diagnostics are then reported:
a GLOBAL gain ``alpha`` (all eta^2 scaled together) and a PER-PAIR scale refit
``{s_k}`` (each pair's field amplitude ``eta_k -> s_k eta_k``, phases and
background fixed) -- the per-pair analogue of the step-7 bounded-fit scale
``s`` (0715 data saw per-pair s of ~1.02 / 0.91 / 0.86).  If chi2/dof collapses
under {s_k} but not under alpha, amplitude drift is the dominant residual and
the phase spectrum itself is fine.

Not a pytest test (no mocks, needs real hardware) -- run it directly:

    python src/drafts/calib_step8_test.py            # COLLECT: drive the full
                                                     #   SWEEP_POINTS**K grid of
                                                     #   (x=w) vectors, write raw CSV,
                                                     #   then analyze it in place
    python src/drafts/calib_step8_test.py some.csv   # ANALYZE: recompute predictions
                                                     #   offline (no hw), compare + plot

``--bounded`` / ``--fix`` pick WHICH stored step-7 spectrum the prediction
uses (the combined JSON may carry both methods; no flag defaults to
``PHASE_METHOD``).  Give both flags to analyze against both back to back.
``--flip`` mirrors step 7's inverted-DAQ convention on a COLLECT: the raw
mean + dark are negated as they are read, so the CSV already carries the
positive light signal (re-analyze it WITHOUT --flip).

The ONE input is ``IN_STEP7``: a combined step-7 result JSON (written by a
``calib_step7_test.py`` REFIT) that embeds step 3 (-> layout), step 6
(-> eta + single-beam + dark per pair) and step 7 (-> the spectrum {Phi_k}).

What to look for.  If the pairwise-calibrated model composes, the measured
points sit on the prediction within their SEMs (pulls ~ +/-1) across the whole
grid -- including points where cross terms between two NON-reference pairs
matter (those probe ``Phi_j - Phi_k``, a combination step 7 never measured
directly).  The "incoh" column drops the cross terms (self TPA + background
only); its much larger residuals show how much of the signal the comb phases
actually predict.  A best-fit ``alpha != 1`` with small residual scatter flags
a common gain drift since calibration, not a wrong phase model.
"""
from __future__ import annotations

import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "drafts"))  # for draft_hw

from draft_hw import connect_daq, connect_slm, read_point  # noqa: E402
from slm_module.calibration.calibration_new import calibration_result_from_dict  # noqa: E402
from slm_module.encoding import channel_layout_from_calibration  # noqa: E402
from slm_module.tpa_phase import (  # noqa: E402
    load_comb_phase_json,
    load_pair_models,
    phi_half,
)

# ---- Edit these to match your setup ----
CALIB_PATH = REPO_ROOT / "src/calib_data"          # data directory: inputs + outputs live here

# The ONE input: a combined step-7 result JSON (calib_step7_test.py REFIT output).
IN_STEP7 = CALIB_PATH / "calib_step7_result_0715_1756.json"

CHANNELS = [1, 3, 5]        # pairs driven together; each needs a step-6 model, and a
                            # step-7 phase unless it IS the reference pair (Phi = 0)
PHASE_METHOD = "bounded"    # default stored step-7 fit to predict from (--bounded/--fix)

# ---- The grid drive ----
# Each channel is swept over SWEEP_POINTS levels in [SWEEP_MIN, SWEEP_MAX] with
# x_k = w_k (both beams of the pair share the level).  The drive is the full
# Cartesian product across CHANNELS, so the grid has SWEEP_POINTS**len(CHANNELS)
# points (e.g. 3 pairs x 5 levels = 125).
SWEEP_MIN, SWEEP_MAX = 0.1, 1.0
SWEEP_POINTS = 5

OUT_DIR = CALIB_PATH        # all step-8 outputs live in the data directory

SLM_DISPLAY_NO = None       # None -> auto-detect the LCOS-SLM display
USB_SLM_NO = 1              # SLM_Ctrl_* device index for the DVI-mode switch

DAQ_DEVICE = "Dev1"
DAQ_CHANNEL = "ai0"

# Fixed per-point acquisition (daq_module), same windows as steps 6/7: the
# all-off dark reads the longer T_single window; driven points (several pairs
# on -> bright) read T_both.
T_SINGLE_S = 5.0
T_BOTH_S = 3.0

SETTLE_S = 0.25             # wait after each SLM pattern change, before reading


# ======================================================================
# input loading  (layout + step-6 models + step-7 phases, all from IN_STEP7)
# ======================================================================

def load_inputs(method: str, channels):
    """Layout, per-pair step-6 models and step-7 comb phases from ``IN_STEP7``.

    Returns ``(layout, models, phases)`` with ``phases[k]`` in radians for every
    ``channels`` entry.  The reference pair defines ``Phi = 0`` and needs no
    stored step-7 fit; every other driven pair must carry a ``method`` fit.
    """
    payload = json.loads(IN_STEP7.read_text(encoding="utf-8"))
    step3 = payload.get("step3")
    if step3 is None:
        raise ValueError(
            f"{IN_STEP7} has no embedded 'step3' calibration; point IN_STEP7 at "
            f"a combined step-7 result (calib_step7_test.py REFIT output)"
        )
    layout = channel_layout_from_calibration(calibration_result_from_dict(step3))
    models = load_pair_models([IN_STEP7])              # reads the embedded "step6"
    ref_index, entries = load_comb_phase_json(IN_STEP7, method=method)
    phases = {ref_index: 0.0}
    phases.update({k: float(e["fit"]["dphi_comb_rad"]) for k, e in entries.items()})
    for k in channels:
        if not (0 <= k < layout.n_channels):
            raise ValueError(
                f"channel {k} out of range (layout has {layout.n_channels} pairs)"
            )
        if k not in models:
            raise ValueError(f"no step-6 model for pair {k} in {IN_STEP7}")
        if k not in phases:
            raise ValueError(
                f"no step-7 '{method}' phase for pair {k} in {IN_STEP7} "
                f"(reference is pair {ref_index})"
            )
    print(f"Step 7 [{method}] vs ref {ref_index}:  "
          + "  ".join(f"Phi[{k}] = {np.degrees(phases[k]):+7.2f} deg" for k in channels))
    print("Step 6:  " + "  ".join(f"eta[{k}] = {models[k].eta:.4g}" for k in channels))
    return layout, models, phases


# ======================================================================
# the forward model  (prediction is dark-subtracted, like the analyzed y)
# ======================================================================

def predict_parts(models, phases, channels, xs, ws) -> tuple[float, float]:
    """One drive point's dark-free prediction, split ``(coherent |E|^2, single-beam bg)``.

    ``E = sum_k eta_k sqrt(x_k w_k) exp(i [phi_half(x_k) + phi_half(w_k) + Phi_k])``;
    the single-beam response is detector background, not TPA field, so it adds
    outside the coherent sum.  Only the returned SUM is physical -- the split
    exists so the analysis can scale the TPA part alone (global gain alpha).
    """
    field = 0.0 + 0.0j
    bg = 0.0
    for k, x, w in zip(channels, xs, ws):
        m = models[k]
        g = float(np.sqrt(max(float(x) * float(w), 0.0)))
        ph = float(phi_half(x)) + float(phi_half(w)) + phases[k]
        field += m.eta * g * np.exp(1j * ph)
        bg += float(m.single_beam(x, w))
    return float(np.abs(field) ** 2), bg


def predict_incoherent(models, channels, xs, ws) -> float:
    """Cross-term-free baseline ``sum_k eta_k^2 x_k w_k`` (self TPA only, no bg)."""
    return float(sum(models[k].self_tpa(x, w) for k, x, w in zip(channels, xs, ws)))


def field_matrix(models, phases, channels, xs, ws) -> np.ndarray:
    """All points' per-pair complex field contributions, ``E[i, j]`` (n x K).

    Row i, column j is pair ``channels[j]``'s field at drive point i with unit
    scale, so ``|E @ ones|^2`` reproduces ``predict_parts``'s coherent term and
    ``|E @ s|^2`` is the per-pair-scaled one.
    """
    e = np.zeros((len(xs), len(channels)), dtype=complex)
    for i, (xv, wv) in enumerate(zip(xs, ws)):
        for j, (k, x, w) in enumerate(zip(channels, xv, wv)):
            m = models[k]
            g = float(np.sqrt(max(float(x) * float(w), 0.0)))
            ph = float(phi_half(x)) + float(phi_half(w)) + phases[k]
            e[i, j] = m.eta * g * np.exp(1j * ph)
    return e


def build_grid_drive() -> tuple[np.ndarray, np.ndarray]:
    """Full-factorial grid drive: (xs, ws), each ``SWEEP_POINTS**K x K``.

    Every channel is swept over ``SWEEP_POINTS`` levels in
    ``[SWEEP_MIN, SWEEP_MAX]``; the drive is the Cartesian product of those
    per-channel levels, and ``x_k = w_k`` at each point (both beams of a pair
    share the level, matching step 7's target sweep ``x_t = w_t = v``).  For
    ``CHANNELS = [1, 3, 5]`` that is a 5 x 5 x 5 = 125-point grid; the first
    channel varies slowest.
    """
    levels = np.linspace(SWEEP_MIN, SWEEP_MAX, SWEEP_POINTS)
    grids = np.meshgrid(*([levels] * len(CHANNELS)), indexing="ij")
    v = np.stack([g.ravel() for g in grids], axis=1)   # (SWEEP_POINTS**K, K)
    v = np.round(v, 6)
    return v, v.copy()


# ======================================================================
# persistence  (raw rows; channels encoded in the x_<k>/w_<k> column names)
# ======================================================================

def _csv_header(channels) -> list[str]:
    cols = ["trial"]
    for k in channels:
        cols += [f"x_{k}", f"w_{k}"]
    return cols + ["dark_v", "voltage_mean_v", "voltage_std_v", "voltage_sem_v",
                   "sem_ratio", "pred_v"]


def write_csv(path, channels, xs, ws, dark_v, mean_v, std_v, sem_v, pred_v,
              *, method: str) -> str:
    """One row per grid point; ``pred_v`` is the collect-time prediction
    (dark-subtracted, ``method`` spectrum) kept for eyeballing -- an ANALYZE
    recomputes it from ``IN_STEP7``.
    """
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_csv_header(channels))
        for i in range(len(mean_v)):
            row = [0]
            for j in range(len(channels)):
                row += [f"{xs[i, j]:.6g}", f"{ws[i, j]:.6g}"]
            ratio = abs(sem_v[i] / mean_v[i]) if mean_v[i] else float("inf")
            row += [f"{dark_v[i]:.9g}", f"{mean_v[i]:.9g}", f"{std_v[i]:.9g}",
                    f"{sem_v[i]:.9g}", f"{ratio:.6g}", f"{pred_v[i]:.9g}"]
            writer.writerow(row)
        f.write(f"# step7_json,{IN_STEP7}\n")
        f.write(f"# pred_method,{method}\n")
        f.write(f"# grid,min={SWEEP_MIN},max={SWEEP_MAX},points={SWEEP_POINTS},x_eq_w\n")
    return str(out)


def load_csv(path):
    """Reload a step-8 CSV -> ``(channels, xs, ws, dark_v, mean_v, std_v, sem_v)``.

    The driven pairs are recovered from the ``x_<k>`` column names, so the file
    is self-describing (CHANNELS may have changed since it was collected).
    """
    with open(Path(path), newline="", encoding="utf-8") as f:
        reader = csv.DictReader(line for line in f if not line.startswith("#"))
        rows = list(reader)
        fields = reader.fieldnames or []
    channels = sorted(int(m.group(1)) for c in fields
                      for m in [re.fullmatch(r"x_(\d+)", c)] if m)
    if not channels or not rows:
        raise ValueError(f"{path} has no x_<k> columns / data rows (not a step-8 CSV)")
    xs = np.array([[float(r[f"x_{k}"]) for k in channels] for r in rows])
    ws = np.array([[float(r[f"w_{k}"]) for k in channels] for r in rows])
    dark_v = np.array([float(r["dark_v"]) for r in rows])
    mean_v = np.array([float(r["voltage_mean_v"]) for r in rows])
    std_v = np.array([float(r.get("voltage_std_v", "nan") or "nan") for r in rows])
    sem_v = np.array([float(r["voltage_sem_v"]) for r in rows])
    return channels, xs, ws, dark_v, mean_v, std_v, sem_v


# ======================================================================
# analysis  (no fit -- compare, then diagnostic gains: global + per pair)
# ======================================================================

def fit_pair_scales(e_mat, y_coh, sem):
    """Per-pair diagnostic refit: ``y_coh ~= |sum_k s_k E_k|^2`` (K scales).

    Splits the single global ``alpha`` into one field-amplitude scale per
    driven pair (``eta_k -> s_k eta_k``; phases and background stay fixed) --
    the per-pair analogue of the step-7 bounded-fit ``s``.  Nonlinear weighted
    LS in the K scales, ``s_k`` boxed to [0, 2] like step 7's BOUND_FRAC = 1.
    Returns ``(s, s_err, chi2_red, dof)``; errors are Birge-scaled.
    """
    from scipy.optimize import least_squares

    n, kk = e_mat.shape

    def resid(s):
        return (y_coh - np.abs(e_mat @ s) ** 2) / sem

    res = least_squares(resid, np.ones(kk), bounds=(0.0, 2.0))
    dof = max(n - kk, 1)
    chi2_red = float(np.sum(res.fun**2)) / dof
    birge = max(1.0, float(np.sqrt(chi2_red)))
    try:
        cov = np.linalg.inv(res.jac.T @ res.jac)
        s_err = np.sqrt(np.clip(np.diag(cov), 0.0, None)) * birge
    except np.linalg.LinAlgError:
        s_err = np.full(kk, float("nan"))
    return res.x, s_err, chi2_red, dof

def analyze(channels, xs, ws, dark_v, mean_v, sem_v, models, phases,
            *, method: str, png_path) -> None:
    """Compare measured vs predicted Y point by point; report + PNG.

    ``y = mean - dark`` per row is compared against the zero-free-parameter
    prediction.  Two diagnostics are then fit to the coherent term only
    (background fixed): a single gain ``alpha`` (a COMMON eta^2 scale, so
    ``s = sqrt(alpha)`` compares directly with the step-7 bounded-fit scale)
    and the per-pair scales ``{s_k}`` (:func:`fit_pair_scales`).
    """
    y = mean_v - dark_v
    parts = [predict_parts(models, phases, channels, x, w) for x, w in zip(xs, ws)]
    coh = np.array([p[0] for p in parts])
    bg = np.array([p[1] for p in parts])
    pred = coh + bg
    inc = np.array([predict_incoherent(models, channels, x, w)
                    for x, w in zip(xs, ws)]) + bg
    resid = y - pred
    pull = resid / sem_v
    n = y.size

    # diagnostic global gain on the coherent part: y - bg ~= alpha * coh
    wgt = 1.0 / sem_v**2
    denom = float(np.sum(wgt * coh**2))
    alpha = float(np.sum(wgt * coh * (y - bg))) / denom
    resid_a = y - (alpha * coh + bg)
    chi2_red = float(np.sum(pull**2)) / n                      # alpha = 1: 0 free params
    chi2_red_a = float(np.sum((resid_a / sem_v) ** 2)) / max(n - 1, 1)
    birge = max(1.0, np.sqrt(chi2_red_a))
    alpha_err = float(np.sqrt(1.0 / denom)) * birge

    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0 else float("nan")

    print(f"\n=== Grid-sweep check [{method}]  "
          f"(pairs {channels}, {n} points, dark-subtracted, mV) ===")
    print("   i  " + "x=[" + " ".join(f"{k:>5}" for k in channels) + "]  "
          + "w=[" + " ".join(f"{k:>5}" for k in channels) + "]  "
          "   meas     pred    incoh     diff    pull")
    for i in range(n):
        xtxt = " ".join(f"{v:5.3f}" for v in xs[i])
        wtxt = " ".join(f"{v:5.3f}" for v in ws[i])
        print(f"  {i:2d}  x=[{xtxt}]  w=[{wtxt}]  "
              f"{y[i]*1e3:7.4f}  {pred[i]*1e3:7.4f}  {inc[i]*1e3:7.4f}  "
              f"{resid[i]*1e3:+7.4f}  {pull[i]:+6.2f}")

    rms = float(np.sqrt(np.mean(resid**2)))
    rms_inc = float(np.sqrt(np.mean((y - inc) ** 2)))
    print(f"\n  rms(meas - pred)      = {rms*1e3:.4f} mV   "
          f"(incoherent baseline: {rms_inc*1e3:.4f} mV)")
    print(f"  chi2/dof (alpha = 1)  = {chi2_red:.2f}  (dof={n})  ;  R^2 = {r2:.4f}")
    print(f"  max |pull|            = {float(np.max(np.abs(pull))):.2f}")
    print(f"  diagnostic gain alpha = {alpha:.4f} +/- {alpha_err:.4f}  "
          f"(s = sqrt(alpha) = {np.sqrt(max(alpha, 0.0)):.4f})  "
          f"-> chi2/dof = {chi2_red_a:.2f} after scaling")

    # per-pair scale refit: split alpha into one amplitude scale per pair
    e_mat = field_matrix(models, phases, channels, xs, ws)
    s_fit, s_err, chi2_red_s, dof_s = fit_pair_scales(e_mat, y - bg, sem_v)
    pred_s = np.abs(e_mat @ s_fit) ** 2 + bg
    rms_s = float(np.sqrt(np.mean((y - pred_s) ** 2)))
    print("  per-pair scales       : "
          + "  ".join(f"s[{k}] = {v:.4f} +/- {e:.4f}"
                      for k, v, e in zip(channels, s_fit, s_err)))
    print(f"    -> chi2/dof = {chi2_red_s:.2f}  (dof={dof_s})  ;  "
          f"rms = {rms_s*1e3:.4f} mV   "
          "(collapse vs alpha-scaled => amplitude drift, phases fine)")

    make_plot(channels, y, sem_v, pred, inc, pull, alpha, chi2_red, chi2_red_a,
              pred_s=pred_s, chi2_red_s=chi2_red_s,
              method=method, path=png_path)
    print(f"  Plot saved to {png_path}")


def make_plot(channels, y, sem, pred, inc, pull, alpha, chi2_red, chi2_red_a,
              *, pred_s=None, chi2_red_s=None, method: str, path) -> None:
    """Model curve (points sorted by predicted Y) + per-point pulls.

    The grid drive has no single physical sweep axis, so the "curve" is the
    model prediction itself: points are ordered by predicted Y, making the
    prediction a smooth monotone line that the measured points should hug.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless: write a PNG rather than open a window
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ---- left: the math model as a curve, measured points on top ----
    order = np.argsort(pred)           # sort by prediction -> the model is monotone
    rank = np.arange(y.size)
    ax1.plot(rank, pred[order] * 1e3, "-", color="tab:blue", lw=1.6, zorder=2,
             label=r"model  $|E|^2 + \mathrm{bg}$")
    if pred_s is not None:
        ax1.plot(rank, pred_s[order] * 1e3, "--", color="tab:green", lw=1.2,
                 zorder=2, label=r"per-pair scaled ($s_k$ refit)")
    ax1.plot(rank, inc[order] * 1e3, ":", color="gray", lw=1.3, zorder=1,
             label="incoherent (no cross terms)")
    ax1.errorbar(rank, y[order] * 1e3, yerr=sem[order] * 1e3, fmt="o", ms=3.5,
                 color="tab:orange", ecolor="lightgray", elinewidth=0.8, capsize=1.5,
                 zorder=3, label="measured")
    ax1.set_xlabel("grid point (sorted by predicted Y)")
    ax1.set_ylabel("Y, dark-subtracted  (mV)")
    ax1.set_title(f"Grid sweep, pairs {channels}  [{method}]")
    ax1.legend(loc="upper left", fontsize=8)

    # ---- right: pulls, same ordering so structure vs signal level is visible ----
    ax2.axhspan(-1, 1, color="tab:blue", alpha=0.12, label=r"$\pm1\sigma$")
    ax2.axhline(0, color="gray", ls="--", lw=1)
    ax2.scatter(rank, pull[order], c="tab:red", s=25, edgecolor="k", lw=0.3)
    ax2.set_xlabel("grid point (sorted by predicted Y)")
    ax2.set_ylabel("Pull = (meas - pred) / SEM")
    title = (f"Pulls  ($\\chi^2$/dof = {chi2_red:.2f}; "
             f"$\\alpha$-scaled {chi2_red_a:.2f}")
    if chi2_red_s is not None:
        title += f"; $s_k$-scaled {chi2_red_s:.2f}"
    ax2.set_title(title + ")")
    ax2.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)


def analyze_csv(path, *, methods: tuple[str, ...]) -> None:
    """Recompute the prediction(s) for an existing CSV and compare (no hardware)."""
    channels, xs, ws, dark_v, mean_v, _std_v, sem_v = load_csv(path)
    print(f"Loaded {mean_v.size} rows (pairs {channels}) from {path}")
    for method in methods:
        _, models, phases = load_inputs(method, channels)
        png = OUT_DIR / f"{Path(path).stem}_compare_{method}.png"
        analyze(channels, xs, ws, dark_v, mean_v, sem_v, models, phases,
                method=method, png_path=png)


# ======================================================================
# collect  (python calib_step8_test.py  ->  drive SLM, record CSV, analyze)
# ======================================================================

def measure_grid(*, flip: bool = False, methods: tuple[str, ...]) -> None:
    """Drive the full SWEEP_POINTS**K grid over CHANNELS; record, then analyze.

    An all-off dark is read once at the start (T_SINGLE_S window) and stored per
    row.  The live printout compares each read against the ``methods[0]``
    prediction; the CSV keeps only raw data (+ that prediction as ``pred_v``),
    so it can be re-analyzed offline against any stored step-7 spectrum.

    ``flip`` negates the raw mean/dark as read (inverted DAQ sign) so the CSV
    already holds the positive light signal -- re-analyze it WITHOUT --flip.
    """
    from slm_module.encoding import encode_to_pattern

    method = methods[0]
    layout, models, phases = load_inputs(method, CHANNELS)
    xs, ws = build_grid_drive()
    n_points = len(xs)
    print(f"Grid: {SWEEP_POINTS} levels in [{SWEEP_MIN}, {SWEEP_MAX}] per pair "
          f"(x = w), {len(CHANNELS)} pairs -> {n_points} points")
    if flip:
        print("Flip: negating voltage_mean_v + dark_v as read (inverted DAQ sign).")

    slm = connect_slm(SLM_DISPLAY_NO, USB_SLM_NO)
    daq = connect_daq(device=DAQ_DEVICE, channel=DAQ_CHANNEL,
                      t_both=T_BOTH_S, t_single=T_SINGLE_S)
    n = layout.n_channels
    zeros = np.zeros(n)
    slm_width, slm_height = slm.get_slm_info()

    def _display(xvec, wvec) -> None:
        x_vals = zeros.copy()
        w_vals = zeros.copy()
        for k, x, w in zip(CHANNELS, xvec, wvec):
            x_vals[k], w_vals[k] = x, w
        slm.display_array(encode_to_pattern(x_vals, w_vals, layout,
                                            slm_width, slm_height))
        if SETTLE_S:
            time.sleep(SETTLE_S)

    means, stds, sems, preds = [], [], [], []
    try:
        _display(np.zeros(len(CHANNELS)), np.zeros(len(CHANNELS)))   # all off
        dark, _, _ = read_point(daq, single=True)
        if flip:
            dark = -dark
        print(f"[0/{n_points}] dark (all off, {T_SINGLE_S:.0f}s) = {dark*1000:.4f} mV")
        for i in range(n_points):
            _display(xs[i], ws[i])
            mean_v, std_v, sem_v = read_point(daq)
            if flip:
                mean_v = -mean_v
            coh, bg = predict_parts(models, phases, CHANNELS, xs[i], ws[i])
            pred = coh + bg
            means.append(mean_v)
            stds.append(std_v)
            sems.append(sem_v)
            preds.append(pred)
            print(f"[{i+1}/{n_points}] "
                  f"x=[{' '.join(f'{v:.3f}' for v in xs[i])}] "
                  f"w=[{' '.join(f'{v:.3f}' for v in ws[i])}] "
                  f"-> {(mean_v-dark)*1000:.4f} mV  (pred {pred*1000:.4f}, "
                  f"diff {(mean_v-dark-pred)*1000:+.4f})")
    finally:
        slm.close_slm()
        daq.disconnect()

    dark_v = np.full(n_points, dark)
    csv_path = OUT_DIR / f"calib_step8_meas_{time.strftime('%m%d_%H%M')}.csv"
    write_csv(csv_path, CHANNELS, xs, ws, dark_v,
              np.asarray(means), np.asarray(stds), np.asarray(sems),
              np.asarray(preds), method=method)
    print(f"\nCSV (pairs {CHANNELS}, {n_points}-point grid) written to {csv_path}")
    print(f"Re-analyze with:  python {Path(__file__).name} {csv_path}")
    analyze_csv(csv_path, methods=methods)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    flip = "--flip" in argv     # inverted DAQ read (COLLECT only; CSVs are stored positive)
    methods = tuple(m for m in ("bounded", "fix") if f"--{m}" in argv) or (PHASE_METHOD,)
    positional = [a for a in argv if not a.startswith("-")]
    if positional:              # a CSV path -> offline analysis, no hardware
        if flip:
            print("Note: --flip only affects a COLLECT (CSVs already store the "
                  "positive signal); ignoring it.")
        analyze_csv(positional[0], methods=methods)
    else:                       # no arg -> collect a fresh grid run (drives SLM/DAQ)
        measure_grid(flip=flip, methods=methods)
    return 0


if __name__ == "__main__":
    sys.exit(main())
