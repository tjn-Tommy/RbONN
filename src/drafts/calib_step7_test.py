"""Manual smoke test: calibrate each pair's comb phase (dPhi_comb) vs a reference.

Not a pytest test (no mocks, needs real hardware) -- run it directly.  Two
invocations, no flags:

    python src/drafts/calib_step7_test.py            # COLLECT: sweep each target's
                                                     #   w channel, write a raw CSV
    python src/drafts/calib_step7_test.py some.csv   # REFIT:   fit dPhi_comb from an
                                                     #   existing CSV, offline (no hw)

What it measures.  Each target pair carries a fixed comb-phase offset ``dPhi_comb``
relative to a common reference pair (the reference defines ``Phi = 0``).  Driving
the two pairs at once makes them interfere; the interference fringe encodes
``dPhi_comb``.  Running every target builds the phase spectrum ``{Phi_k}``.

The drive (one geometry).  The reference pair AND the target's x channel are held
fully on (commanded intensity 1); only the target's w channel is swept over
``MEAS_W2_VALUES``.  A channel at intensity ``v`` sits at panel phase
``phi = 2*asin(sqrt(v))`` with field ``sqrt(v)*exp(i phi/2)``; intensity 1 is
exactly ``phi = pi`` (fully on).  So as ``w_t`` runs 0.1 -> 1.0 the target field
amplitude ``g = sqrt(x_t w_t) = sqrt(w_t)`` and the SLM phase difference
``dPhi_SLM`` both sweep, tracing part of the interference fringe.

The fit (in :mod:`slm_module.tpa_phase`).  Every point is reduced to
``(g, dPhi_SLM)`` from its commanded intensities, so the fit is geometry-general --
it never assumes how the SLM was driven (a REFIT of an older CSV that swept both
target channels together fits exactly the same way).  It floats ``dPhi_comb`` in

    Y = a^2 + b^2 g^2 + 2 a b g cos(dPhi_SLM + dPhi_comb) + step-6 background + d

with ``a`` := reference amplitude and ``b`` := target amplitude.  ``a`` and ``b``
are pinned to the step-6 ``eta_ref:eta_tgt`` ratio (only a shared gain ``s`` floats,
boxed to ``+/-BOUND_FRAC``), and each pair's step-6 single-beam response is folded
in as a FIXED background so the fringe need not absorb the single-beam ramp.

Prereq: every pair used here (reference + targets) must already have a step-6
(:mod:`slm_module.tpa_pair`) efficiency calibration -- that's where ``eta`` and the
single-beam / dark background terms come from.  Point ``STEP6_SOURCES`` at their
step-6 output(s); each may be a combined step-6 result JSON, a bare
``save_tpa_pair_json`` summary, or a raw step-6 CSV (re-fit here with the same
algorithm, so a JSON is not required).

All model / background removal / weighted fit / persistence live in
:mod:`slm_module.tpa_phase`; this file only wires up hardware and prints/plots.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from daq_module.controller import DAQController, DAQMonitorSettings  # noqa: E402
from slm_module.calibration.calibration_new import load_calibration_result  # noqa: E402
from slm_module.controller import SLMController  # noqa: E402
from slm_module.encoding import build_channel_layout  # noqa: E402
from slm_module.tpa_phase import (  # noqa: E402
    PhaseFit,
    PhaseResult,
    fit_result,
    load_pair_models,
    load_phase_csv,
    phi_half,
)

# ---- Edit these to match your setup ----
CALIB_PATH = REPO_ROOT / "src/calib_data"          # data directory: inputs + outputs live here
REF_INDEX = 1                                      # common reference pair (Phi = 0)
TGT_INDICES = [3, 4, 5]                             # target pairs measured vs the reference

IN_STEP3 = CALIB_PATH / "calib_step3_fast_channels.json"  # Step 3 two-pair calib -> channel layout

# Step-6 eta + background per pair.  Accepts a combined step-6 result JSON
# (save_combined_json: {"step3": ..., "step6": {channels:[...]}}), a bare
# save_tpa_pair_json summary, or a raw step-6 CSV (re-fit with the same
# algorithm, so a JSON is optional).  One combined JSON already carries every
# calibrated pair (the reference + all targets) in its channels list, so a single
# path is enough -- point it at the latest step-6 run.
STEP6_SOURCES = [
    CALIB_PATH / "calib_step6_result_0709_1439.json",  # pairs 1 (ref) + 3,4,5 (targets)
]

# ---- The w sweep ----
# Reference pair + the target's x channel are held fully on (intensity 1); only the
# target's w channel is swept over these commanded intensities.
MEAS_W2_VALUES = np.round(np.linspace(0.1, 1.0, 10), 6)   # 0.1, 0.2, ... 1.0

OUT_DIR = CALIB_PATH             # all step-7 outputs live in the data directory

SLM_DISPLAY_NO = None            # None -> auto-detect the LCOS-SLM display (like the GUI's Detect)
USB_SLM_NO = 1                   # SLM_Ctrl_* device index for the DVI-mode switch (USB link)

DAQ_DEVICE = "Dev1"
DAQ_CHANNEL = "ai0"
DAQ_F_CUT_HZ = 3.5               # DAQ low-pass 3 dB bandwidth (matches DAQMonitorSettings.f_cut)

# ---- Adaptive per-point averaging (DAQController) ----
# Each reading picks its own duration so its SEM meets
# max(TARGET_REL*|mean|, SEM_FLOOR), capped at T_MAX, and is recorded per point in
# voltage_sem_v.  The step-7 fringe sits at 7-22 mV, so most points hit the 1%
# relative target in a few seconds; the near-zero dark read stops at SEM_FLOOR.
TARGET_REL = 0.01                # target relative SEM (SEM/|mean|)
SEM_FLOOR = 60e-6                # absolute SEM floor (V) for near-zero-signal points
T_PROBE = 0.7                    # probe / minimum window per point (s)
T_MAX = 10.0                     # cap per point (s)

# Refitting OLD single-column CSVs (voltage_std_v held the RAW std, no
# voltage_sem_v column): the SEM is reconstructed as std/sqrt(n_eff),
# n_eff = 2*DAQ_DURATION_S*DAQ_F_CUT_HZ.  Auto-detected from the CSV header --
# new two-column CSVs carry voltage_sem_v and skip this entirely.
DAQ_DURATION_S = 5.0             # fixed window those legacy CSVs were recorded at

SETTLE_S = 0.25                  # wait after each SLM pattern change, before reading
REPEATS = 1                      # repeated monitor readings averaged per point
N_TRIALS = 1                     # times the whole sweep is repeated (statistics)

# Amplitude handling for the dPhi_comb fit.  None -> unconstrained closed-form
# fit; a number LOCKS the ratio a:b (= eta_ref:eta_tgt) from step 6 and floats a
# single shared scale s boxed to +/- this fraction about 1 (1.0 == s in [0, 2]),
# so a and b cannot diverge -- only a common gain drift between step 6 and 7 is
# allowed.  report()/make_plot() flag when s hits its box.
BOUND_FRAC = 1.0

# Fold in the step-6 single-beam response as a FIXED background.  The reference is
# held fully on -> its single-beam is a constant; only the swept target ramps with
# w.  Keeps the fringe from having to absorb the single-beam ramp.
SINGLE_BEAM_BG = True


# ======================================================================
# hardware wiring
# ======================================================================

def detect_slm_display() -> int:
    """Find the LCOS-SLM display number (the GUI's Detect step)."""
    probe = SLMController(display_no=1)
    for display_no, width, height, name in probe.detect_displays():
        print(f"  display {display_no}: {width}x{height} ({name})")
        if name.startswith("LCOS-SLM"):
            return display_no
    raise RuntimeError(
        "No LCOS-SLM display found. Check the SLM is connected as an extended "
        "display, or set SLM_DISPLAY_NO manually."
    )


def connect_slm() -> SLMController:
    display_no = SLM_DISPLAY_NO if SLM_DISPLAY_NO is not None else detect_slm_display()
    slm = SLMController(display_no=display_no)
    slm.open_slm()
    width, height = slm.get_slm_info()
    print(f"SLM: connected on display {display_no} ({width}x{height})")
    slm.set_dvi_mode(USB_SLM_NO)
    print(f"SLM: DVI mode set (USB device {USB_SLM_NO})")
    return slm


def connect_daq() -> DAQController:
    """DAQ is the Y-measurement instrument (hold=0: the sweep owns the settle)."""
    daq = DAQController(device=DAQ_DEVICE)
    daq.connect()
    daq.configure_monitor(
        DAQMonitorSettings(
            channel=DAQ_CHANNEL, hold=0.0,
            adaptive=True, target_rel=TARGET_REL, sem_floor=SEM_FLOOR,
            t_probe=T_PROBE, t_max=T_MAX,
        )
    )
    print(f"Monitor: DAQ ({DAQ_DEVICE}/{DAQ_CHANNEL}) adaptive: "
          f"rel<={TARGET_REL:.1%} or SEM<={SEM_FLOOR*1e6:.0f} uV, "
          f"{T_PROBE:.1f}-{T_MAX:.1f} s/point")
    return daq


def load_models(layout=None):
    """Load all step-6 models; require REF_INDEX and every TGT_INDICES entry."""
    models = load_pair_models(STEP6_SOURCES, layout=layout)
    needed = [("reference", REF_INDEX)] + [("target", k) for k in TGT_INDICES]
    for role, idx in needed:
        if idx not in models:
            raise ValueError(
                f"no step-6 model for {role} pair index {idx}; found "
                f"{sorted(models)} in {[str(p) for p in STEP6_SOURCES]}"
            )
    print(f"Step 6: eta[ref {REF_INDEX}] = {models[REF_INDEX].eta:.4g} ; "
          + " ".join(f"eta[{k}]={models[k].eta:.4g}" for k in TGT_INDICES))
    return models


# ======================================================================
# report + plot
# ======================================================================

def _sigma(value: float, err: float) -> float:
    return abs(value) / err if err else float("nan")


def _bound_note(value: float, eta: float, frac: float, at_bound: bool) -> str:
    """Deviation from the step-6 eta plus an '[AT +/-100% BOUND]' warning tag."""
    dev = (value / eta - 1.0) * 100.0 if eta else float("nan")
    tag = f"  [AT +/-{frac*100:.0f}% BOUND]" if at_bound else ""
    return f"  ({dev:+.0f}% vs eta {eta*1e3:.4f}){tag}"


def _wonly_sweep(fit: PhaseFit) -> bool:
    """True if this CSV came from the w-only sweep (x_t pinned on, only w_t swept),
    i.e. ``x_t != w_t``.

    The alternative (``x_t == w_t``) is an older CSV that swept the target's two
    channels together; a fit with no stored per-point intensities is treated as
    that older geometry.  The fit is identical either way -- only the printed model
    string and the plotted smooth curve differ, so the report adapts to whatever
    the CSV holds:

    * w-only  : g = sqrt(w_t) = sin(theta),  dPhi_SLM = theta - pi/2,  theta = asin(sqrt(w_t))
    * both    : g = sin^2(theta/2),          dPhi_SLM = theta - pi,    theta the shared phase
    """
    return fit.x_t is not None and not np.allclose(fit.x_t, fit.w_t)


def report(fit: PhaseFit, tgt: int, ref: int) -> None:
    """Print dPhi_comb (rad + deg), the ratio-locked amplitudes a/b and fit quality."""
    if _wonly_sweep(fit):
        print("Model:  Y = s^2 (a^2 + b^2 sin^2(theta) + 2ab sin(theta) cos(dPhi_comb - pi/2 + theta))")
        print("            [w-only sweep: ref + target-x on, only target-w swept; "
              "theta = asin(sqrt(w_t))]")
    else:
        print("Model:  Y = s^2 (a^2 + b^2 sin^4(theta/2) + 2ab sin^2(theta/2) cos(dPhi_comb - pi + theta))")
        print("            [both target channels swept together; theta the shared panel phase]")
    print("            + step6 single-beam + d     "
          f"(a:b locked to step-6 eta ratio; scale s boxed +/-{fit.bound_frac*100:.0f}%)")
    print(f"Pair {tgt} vs reference {ref}  (value +/- error, Birge-scaled):")
    print(f"  dPhi_comb = {fit.dphi_comb:+.4f} +/- {fit.dphi_comb_err:.4f} rad"
          f"   ( {fit.dphi_comb_deg:+.2f} +/- {np.degrees(fit.dphi_comb_err):.2f} deg )")
    print(f"  a (ref R_1)      = {fit.a*1e3:.4f} +/- {fit.a_err*1e3:.4f} mV^0.5"
          + _bound_note(fit.a, fit.eta_ref, fit.bound_frac, fit.a_at_bound))
    print(f"  b (tgt eta CxCw) = {fit.b*1e3:.4f} +/- {fit.b_err*1e3:.4f} mV^0.5"
          + _bound_note(fit.b, fit.eta_tgt, fit.bound_frac, fit.b_at_bound))
    print(f"  fringe amp 2ab   = {fit.amp*1e3:.4f} +/- {fit.amp_err*1e3:.4f} mV")
    print(f"  residual dark d  = {fit.offset*1e3:+.4f} +/- {fit.offset_err*1e3:.4f} mV"
          f"   (should be ~0 after per-row dark subtraction)")
    print(f"  chi2/dof = {fit.chi2_red:.2f}  (dof={fit.dof})  -> Birge x{fit.birge:.2f} "
          f"on errors ;  R^2 = {fit.r2:.4f}")


def make_plot(fit: PhaseFit, tgt: int, path) -> None:
    """Measured Y(dPhi_SLM) with the fitted a/b/dPhi_comb model curve + pulls, PNG."""
    import matplotlib

    matplotlib.use("Agg")  # headless: write a PNG rather than open a window
    import matplotlib.pyplot as plt

    dphi = np.degrees(fit.dphi_slm)             # dPhi_SLM at the measured points
    pulls = fit.residuals / fit.sem
    wonly = _wonly_sweep(fit)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Smooth model over the swept geometry.  Rebuild g/dPhi_SLM the same way the
    # fit did (from the per-point commanded intensities) so the curve tracks the
    # data.  w-only: hold x_t (= 1), sweep w_t in [0, 1].  both channels: sweep the
    # shared phase over the full 0..180 deg half turn (x_t = w_t = sin^2(phi/2)).
    if wonly:
        wt_s = np.linspace(0.0, 1.0, 400)                       # sweep target w only
        xt_s = np.full_like(wt_s, float(np.median(fit.x_t)))    # x_t held (= 1)
    else:
        wt_s = xt_s = np.sin(np.radians(np.linspace(0.0, 180.0, 400)) / 2.0) ** 2
    xr_c = float(np.median(fit.x_r)) if fit.x_r is not None else 1.0
    wr_c = float(np.median(fit.w_r)) if fit.w_r is not None else 1.0
    g_s = np.sqrt(np.clip(xt_s * wt_s, 0.0, None))
    dslm = phi_half(xt_s) + phi_half(wt_s) - phi_half(xr_c) - phi_half(wr_c)

    # Fixed step-6 single-beam background = fit.known - a^2 - b^2 g^2 at each
    # fitted point; interpolate it onto the smooth grid (exact at the points, and
    # geometry-agnostic, so no assumption about how the background splits in g).
    bg_pts = fit.known - fit.a**2 - fit.b**2 * fit.g**2
    order = np.argsort(fit.dphi_slm)
    bg_s = np.interp(dslm, fit.dphi_slm[order], bg_pts[order])

    model = (fit.a**2 + fit.b**2 * g_s**2
             + 2.0 * fit.a * fit.b * g_s * np.cos(dslm + fit.dphi_comb)
             + bg_s + fit.offset)
    label = (r"fit: $a^2+b^2\sin^2\theta+2ab\sin\theta\cos$" if wonly
             else r"fit: $a^2+b^2\sin^4+2ab\sin^2\cos$")
    ax1.plot(np.degrees(dslm), model * 1e3, "-", color="tab:blue", lw=1.6, label=label)
    ax1.errorbar(dphi, fit.y * 1e3, yerr=fit.sem * 1e3, fmt="o", ms=5, color="tab:orange",
                 ecolor="lightgray", elinewidth=1, capsize=2, zorder=3,
                 label="measured (dark-subtracted)")
    ax1.set_xlabel(r"$\Delta\Phi_{SLM}$  (deg)")
    ax1.set_ylabel(r"$Y$, dark-subtracted  (mV)")
    ax1.set_title(f"Pair {tgt} interference"
                  + (r"  (w-only sweep)" if wonly else "  (both channels, half fringe)"))
    ax1.legend(loc="best", fontsize=8)

    ax2.axhspan(-1, 1, color="tab:blue", alpha=0.12, label=r"$\pm1\sigma$")
    ax2.axhline(0, color="gray", ls="--", lw=1)
    ax2.scatter(dphi, pulls, c="tab:red", s=40, edgecolor="k", lw=0.4)
    ax2.set_xlabel(r"$\Delta\Phi_{SLM}$  (deg)")
    ax2.set_ylabel("Pull = residual / SEM")
    ax2.set_title(f"Pulls  ($\\chi^2$/dof = {fit.chi2_red:.2f})")
    ax2.legend(loc="upper right", fontsize=8)

    bflag = ("  [a@bound]" if fit.a_at_bound else "") + ("  [b@bound]" if fit.b_at_bound else "")
    txt = (
        f"$\\Delta\\Phi_{{comb}}$ = {fit.dphi_comb_deg:+.2f} $\\pm$ "
        f"{np.degrees(fit.dphi_comb_err):.2f} deg  "
        f"({_sigma(fit.dphi_comb, fit.dphi_comb_err):.0f}$\\sigma$)\n"
        f"a = {fit.a*1e3:.3f} ($\\eta$ {fit.eta_ref*1e3:.3f}), "
        f"b = {fit.b*1e3:.3f} ($\\eta$ {fit.eta_tgt*1e3:.3f}) mV$^{{1/2}}${bflag}\n"
        f"d = {fit.offset*1e3:+.3f} mV  (should be $\\approx$0)\n"
        f"$\\chi^2$/dof = {fit.chi2_red:.2f} (Birge x{fit.birge:.2f})"
    )
    ax1.text(0.05, 0.95, txt, transform=ax1.transAxes, va="top",
             bbox=dict(boxstyle="round", fc="white", alpha=0.85), fontsize=8)
    """Measured Y(theta2) with the fitted a/b/dPhi_comb model curve + pulls, PNG.

    Rendering lives in :mod:`slm_module.tpa_phase_report` (shared with the
    GUI); this wrapper only owns the headless figure + save-to-file part.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless: write a PNG rather than open a window
    from matplotlib.figure import Figure

    from slm_module.tpa_phase_report import plot_fringe

    fig = Figure(figsize=(12, 5))
    plot_fringe(fig, fit, tgt)
    fig.savefig(path, dpi=150)


# ======================================================================
# offline refit  (python calib_step7_test.py some.csv)
# ======================================================================

def _targets_in_csv(path, default) -> list[int]:
    """Distinct target-pair indices recorded in a CSV (sorted).

    A collected CSV (:func:`measure_only`) stacks every TGT_INDICES entry vs the
    shared reference in one file, so its ``tgt_index`` column lists several pairs.
    Falls back to ``default`` if the column is missing (an old single-target CSV).
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
    return sorted(seen) if seen else list(default)


def _csv_has_sem(path) -> bool:
    """True if the CSV header carries an explicit ``voltage_sem_v`` column.

    New two-column CSVs record the SEM directly; old single-column CSVs stored
    only the raw std, so their SEM must be reconstructed at refit time.
    """
    with open(Path(path), newline="", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            return "voltage_sem_v" in [c.strip() for c in line.split(",")]
    return False


def fit_csv(path) -> None:
    """Re-fit an already-recorded CSV offline (no hardware).

    Needs two inputs: this CSV plus the step-6 model JSON (``STEP6_SOURCES``) for
    each pair's eta + single-beam background.  The CSV may carry several target
    pairs -- a collected file records every TGT_INDICES entry vs the shared
    REF_INDEX -- so every target present (that has a step-6 model) is fit separately
    against the reference and gets its own refit PNG.
    """
    models = load_models()
    targets = _targets_in_csv(path, TGT_INDICES)
    fittable = [k for k in targets if k in models and k != REF_INDEX]
    if not fittable:
        raise ValueError(
            f"no fittable target in {path}: found targets {targets}, but have "
            f"step-6 models only for {sorted(models)} (reference is pair {REF_INDEX})"
        )
    print(f"Fitting pair(s) {fittable} vs reference {REF_INDEX} from {path}")
    # Old single-column CSVs (no voltage_sem_v column) stored the RAW std; new
    # two-column CSVs already carry the per-point SEM, so only the old ones need
    # SEM reconstructed as std/sqrt(n_eff).
    legacy = not _csv_has_sem(path)
    n_eff = max(2.0 * DAQ_DURATION_S * DAQ_F_CUT_HZ, 1.0)
    if legacy:
        print(f"Legacy single-column CSV: SEM = std/sqrt(n_eff={n_eff:.0f})")
    for k in fittable:
        print(f"\n=== Re-fit: pair {k} vs reference {REF_INDEX} ===")
        result = load_phase_csv(path, models[k], models[REF_INDEX],
                                frac=BOUND_FRAC, single_beam_bg=SINGLE_BEAM_BG,
                                only_tgt=k)
        if legacy:
            # raw waveform std -> SEM of the mean, then re-fit with SEM weights
            result.voltage_sem_v = np.asarray(result.voltage_std_v, dtype=float) / np.sqrt(n_eff)
            fit_result(result, models[k], models[REF_INDEX],
                       frac=BOUND_FRAC, single_beam_bg=SINGLE_BEAM_BG)
        dts = result.per_trial_darks()
        drift = f" +/- {dts.std(ddof=1)*1e3:.4f} drift" if dts.size > 1 else ""
        print(f"Loaded {result.trial.size} rows, "
              f"dark = {result.dark*1e3:.4f}{drift} mV")
        report(result.fit, result.tgt_index, result.ref_index)
        plot_path = OUT_DIR / f"calib_step7_pair{k}_refit.png"
        make_plot(result.fit, result.tgt_index, plot_path)
        print(f"Plot saved to {plot_path}")


# ======================================================================
# collect  (python calib_step7_test.py  ->  drive SLM, record raw CSV, no fit)
# ======================================================================

def build_w2_sweep() -> list[tuple[float, float, float, float]]:
    """Drive tuples for the w sweep: reference + target-x fully on, only target-w swept.

    Returns target-first commanded-intensity tuples ``(x_t, w_t, x_r, w_r)`` with
    ``x_t = x_r = w_r = 1`` and ``w_t`` stepping over ``MEAS_W2_VALUES``.  Raw data
    only -- the fit is done later, offline.
    """
    return [(1.0, float(w2), 1.0, 1.0) for w2 in MEAS_W2_VALUES]

    Rendering lives in :mod:`slm_module.tpa_phase_report` (shared with the
    GUI); this wrapper only owns the headless figure + save-to-file part.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless: write a PNG rather than open a window
    from matplotlib.figure import Figure

    from slm_module.tpa_phase_report import plot_report

    fig = Figure(figsize=(12, 5))
    plot_report(fig, result, tgt, ref, subtitle=subtitle)
    fig.savefig(path, dpi=150)

_MEAS_CSV_HEADER = [
    "trial", "tgt_index", "ref_index",
    "phi_xt_deg", "phi_wt_deg", "x_t", "w_t", "x_r", "w_r",
    "dark_v", "voltage_mean_v", "voltage_std_v",
]


def write_meas_csv(results, path) -> str:
    """Write raw rows for one or more target pairs into a single CSV.

    Same column layout as :func:`tpa_phase.write_phase_csv`, but concatenates
    several :class:`PhaseResult` objects so every row carries its own
    ``tgt_index`` (and the shared ``ref_index``) -- i.e. REF_INDEX and every
    TGT_INDICES entry are recorded in the file, per row.  Round-trips via
    :func:`tpa_phase.load_phase_csv` (used by the offline refit).
    """
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_MEAS_CSV_HEADER)
        for result in results:
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
    return str(out)


def _read_daq(daq, timeout: float) -> tuple[float, float, float]:
    """One averaged reading, its raw trace std, and the per-point SEM of the mean.

    ``std`` is the raw (low-passed) trace spread kept for diagnostics; ``sem`` is
    the DAQ's reported standard error of the mean (low-passed, effective-N) and is
    what the fit weights by.  With adaptive duration each point averages for a
    different time, so ``sem`` is recorded per point rather than reconstructed.
    """
    means: list[float] = []
    std_vars: list[float] = []
    sem_vars: list[float] = []
    for _ in range(max(1, REPEATS)):
        sample = daq.monitor_cycle(timeout=timeout)
        means.append(float(sample.value))
        std = getattr(sample, "std", None)
        waveform = getattr(daq, "last_values", None)
        raw_std = (
            float(std) if std is not None and np.isfinite(std)
            else (float(np.std(waveform)) if waveform is not None and np.size(waveform) > 1
                  else 0.0)
        )
        std_vars.append(raw_std ** 2)
        sem = getattr(sample, "sem", None)
        sem_vars.append(float(sem) ** 2 if sem is not None and np.isfinite(sem) else raw_std ** 2)
    mean_v = float(np.mean(means))
    std_v = float(np.sqrt(np.mean(std_vars))) if std_vars else 0.0
    sem_v = float(np.sqrt(np.mean(sem_vars))) if sem_vars else 0.0
    return mean_v, std_v, sem_v


def _measure_target(daq, slm, layout, k: int, drive) -> PhaseResult:
    """Drive pair ``k`` (vs REF_INDEX) over ``drive`` and read Y; PhaseResult, no fit.

    Only channels ``k`` and ``REF_INDEX`` are driven; all others held off.  A fresh
    all-off dark is read at the start of each trial and stored per row (matching
    the calibration sweep's per-trial dark).  Needs no step-6 model -- raw data
    only.
    """
    from slm_module.encoding import encode_to_pattern

    n = layout.n_channels
    zeros = np.zeros(n)
    slm_width, slm_height = slm.get_slm_info()
    read_timeout = max(30.0, T_MAX * 3.0 + 10.0)

    def _display(x_t, w_t, x_r, w_r) -> None:
        x_vals = zeros.copy()
        w_vals = zeros.copy()
        x_vals[k], w_vals[k] = x_t, w_t
        x_vals[REF_INDEX], w_vals[REF_INDEX] = x_r, w_r
        slm.display_array(encode_to_pattern(x_vals, w_vals, layout, slm_width, slm_height))
        if SETTLE_S:
            time.sleep(SETTLE_S)

    total = N_TRIALS * (len(drive) + 1)
    step = 0
    rows: list[tuple] = []
    for trial in range(N_TRIALS):
        _display(0.0, 0.0, 0.0, 0.0)                 # all-off dark, per trial
        dark_v, _, _ = _read_daq(daq, read_timeout)
        step += 1
        print(f"[{step}/{total}] pair {k} trial {trial} dark (all off) = {dark_v*1000:.4f} mV")
        for x_t, w_t, x_r, w_r in drive:
            _display(x_t, w_t, x_r, w_r)
            mean_v, std_v, sem_v = _read_daq(daq, read_timeout)
            rows.append((trial, x_t, w_t, x_r, w_r, mean_v, std_v, sem_v, dark_v))
            step += 1
            print(f"[{step}/{total}] pair {k} w_t={w_t:.3f} -> {mean_v*1000:.4f} mV "
                  f"(SEM {sem_v*1e6:.1f} uV, dark {dark_v*1000:.4f})")

    return PhaseResult(
        tgt_index=k, ref_index=REF_INDEX,
        trial=np.array([r[0] for r in rows], dtype=int),
        x_t=np.array([r[1] for r in rows], dtype=float),
        w_t=np.array([r[2] for r in rows], dtype=float),
        x_r=np.array([r[3] for r in rows], dtype=float),
        w_r=np.array([r[4] for r in rows], dtype=float),
        voltage_mean_v=np.array([r[5] for r in rows], dtype=float),
        voltage_std_v=np.array([r[6] for r in rows], dtype=float),
        voltage_sem_v=np.array([r[7] for r in rows], dtype=float),
        dark_v=np.array([r[8] for r in rows], dtype=float),
        n_trials=N_TRIALS,
    )


def measure_only() -> None:
    """Sweep w for every target pair vs the shared reference; write one raw CSV.

    Loops over TGT_INDICES (each vs REF_INDEX), holding x_ref = w_ref = x_tgt = 1
    and sweeping only that target's w channel over MEAS_W2_VALUES.  All rows go into
    a single timestamped CSV, tagged per row with ``tgt_index`` and ``ref_index`` so
    REF_INDEX and every TGT_INDICES entry are recorded.  Raw data only: no step-6
    models, no fit -- just drive the SLM and record the DAQ.  Refit later with
    ``python calib_step7_test.py <that csv>``.
    """
    calib = load_calibration_result(IN_STEP3)
    layout = build_channel_layout(calib)

    drive = build_w2_sweep()
    slm = connect_slm()
    daq = connect_daq()
    results = []
    try:
        for k in TGT_INDICES:
            print(f"\n=== Sweep: pair {k} vs reference {REF_INDEX}  "
                  f"(x{REF_INDEX}=w{REF_INDEX}=x{k}=1, sweep w{k} over "
                  f"{MEAS_W2_VALUES.tolist()}) ===")
            results.append(_measure_target(daq, slm, layout, k, drive))
    finally:
        daq.disconnect()
        slm.close_slm()

    csv_path = OUT_DIR / f"calib_step7_meas_{time.strftime('%m%d_%H%M')}.csv"
    write_meas_csv(results, csv_path)
    print(f"\nCSV (ref {REF_INDEX}, targets {TGT_INDICES}) written to {csv_path}")
    print(f"Refit with:  python {Path(__file__).name} {csv_path}")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    positional = [a for a in argv if not a.startswith("-")]
    if positional:              # a CSV path -> offline refit, no hardware
        fit_csv(positional[0])
    else:                       # no arg -> collect a fresh sweep (drives the SLM/DAQ)
        measure_only()
    return 0


if __name__ == "__main__":
    sys.exit(main())
