"""Manual smoke test: calibrate each pair's comb phase (dPhi_comb) vs pair 0.

Not a pytest test (no mocks, needs real hardware) -- run it directly::

    python tests/tpa_phase_calibration_test.py            # Table 1 sweep + fit + plot (all targets)
    python tests/tpa_phase_calibration_test.py --symmetry # Table 1, THEN also the 3x3 symmetry check (Table 2)
    python tests/tpa_phase_calibration_test.py some.csv    # re-fit an existing Table-1 CSV offline

Prereq: every pair used here (reference + targets) must already have a step-6
(:mod:`slm_module.tpa_pair`) efficiency calibration -- that's where ``eta`` and
the single-beam / dark background terms come from.  Point ``STEP6_SOURCES`` at
their step-6 outputs; each may be a ``save_tpa_pair_json`` summary or a raw
step-6 CSV (re-fit here with the same algorithm, so a JSON is not required).

Table 1 (the calibration): pair 0 is the common reference, held fully-on at
phi = 180 deg (intensity 1) on both its channels; all other pairs are off.  The
target pair k is swept symmetrically ``phi^x_k = phi^w_k = phi`` over [0, 180]
deg -- the full reachable half turn, since the measured Step-3 transfer curve is
monotonic (intensity 1 == phi = pi).  That traces ``dPhi_SLM = phi - 180 deg`` in
[-180, 0], and the fit of

    Y = R_k^2 + R_0^2 + 2*R_k*R_0*cos(dPhi_SLM + dPhi_comb)   (+ step-6 background)

returns pair k's phase relative to pair 0 (Phi_0 == 0 by definition).  Looping
over the targets builds the spectrum {Phi_k}.

Table 2 (--symmetry, one-time spot check): a 3x3 grid on the target's individual
channel phases {90, 135, 180} deg, verifying swap invariance (phase depends only
on phi^x+phi^w, amplitude only on the product).

All model / background removal / weighted fit / persistence live in
:mod:`slm_module.tpa_phase`; this file only wires up hardware and prints/plots.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from daq_module.controller import DAQController, DAQMonitorSettings  # noqa: E402
from slm_module.calibration.calibration_new import load_calibration_result  # noqa: E402
from slm_module.controller import SLMController  # noqa: E402
from slm_module.encoding import build_channel_layout  # noqa: E402
from slm_module.tpa_phase import (  # noqa: E402
    PhaseFit,
    build_phase_sweep,
    build_symmetry_grid,
    load_pair_models,
    load_phase_csv,
    measure_phase_sweep,
    save_phase_json,
    swap_invariance,
    write_phase_csv,
)

# ---- Edit these to match your setup ----
CALIB_PATH = REPO_ROOT / "calib_step3_2pairs.json"  # Step 3 two-pair calib -> channel layout
REF_INDEX = 0                                      # common reference pair (Phi_0 == 0)
TGT_INDICES = [3]                                  # far pair 3 vs near pair 0 (crosstalk test)

# Step-6 eta + background per pair. JSON (save_tpa_pair_json) or raw step-6 CSV;
# the CSV is re-fit with the same algorithm, so a JSON is optional.  This single
# two-pair ch-efficiency JSON carries both pair 0 and pair 3 in its channels list.
STEP6_SOURCES = [
    REPO_ROOT / "tpa_2pairs_chefficiency.json",             # pairs 0 (ref) + 3 (target)
]

SWEEP_POINTS = 15                # Table 1 points over phi in [PHI_START, PHI_STOP]
PHI_START_DEG = 0.0
PHI_STOP_DEG = 180.0             # capped at 180: the reachable half turn
REF_PHASE_DEG = 180.0            # reference held fully-on (intensity 1)

OUT_DIR = REPO_ROOT
SPECTRUM_JSON = REPO_ROOT / "tpa_phase_spectrum.json"
PLOT_PATH = REPO_ROOT / "tpa_phase_fit.png"

SLM_DISPLAY_NO = None            # None -> auto-detect the LCOS-SLM display (like the GUI's Detect)
USB_SLM_NO = 1                   # SLM_Ctrl_* device index for the DVI-mode switch (USB link)

DAQ_DEVICE = "Dev1"
DAQ_CHANNEL = "ai0"
DAQ_DURATION_S = 1.0             # DAQ averaging window per reading

SETTLE_S = 0.15                  # wait after each SLM pattern change, before reading
REPEATS = 1                      # repeated monitor readings averaged per point
N_TRIALS = 10                    # times the whole sweep is repeated (statistics)


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
        DAQMonitorSettings(channel=DAQ_CHANNEL, duration=DAQ_DURATION_S, hold=0.0)
    )
    print(f"Monitor: DAQ ({DAQ_DEVICE}/{DAQ_CHANNEL})")
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


def _sigma(value: float, err: float) -> float:
    return abs(value) / err if err else float("nan")


def report(fit: PhaseFit, tgt: int, ref: int) -> None:
    """Print dPhi_comb (rad + deg), the fringe visibility and the fit quality."""
    print(f"Model:  Y = R_k^2 + R_0^2 + 2*R_k*R_0*cos(dPhi_SLM + dPhi_comb)  (+ step-6 background)")
    print(f"Pair {tgt} vs reference {ref}  (value +/- error, Birge-scaled):")
    print(f"  dPhi_comb = {fit.dphi_comb:+.4f} +/- {fit.dphi_comb_err:.4f} rad"
          f"   ( {fit.dphi_comb_deg:+.2f} +/- {np.degrees(fit.dphi_comb_err):.2f} deg )")
    print(f"  visibility V = {fit.visibility:.4f} +/- {fit.visibility_err:.4f}"
          f"   (ideal ~1; far from 1 => eta mismatch / partial coherence)")
    print(f"  DC offset c  = {fit.offset*1e3:+.4f} mV")
    print(f"  chi2/dof = {fit.chi2_red:.2f}  (dof={fit.dof})  -> Birge x{fit.birge:.2f} "
          f"on errors ;  R^2 = {fit.r2:.4f}")


def make_plot(fit: PhaseFit, tgt: int, path) -> None:
    """Normalised half fringe (Z/2R_kR_0 vs dPhi_SLM) + pulls, headless PNG."""
    import matplotlib

    matplotlib.use("Agg")  # headless: write a PNG rather than open a window
    import matplotlib.pyplot as plt

    dphi = np.degrees(fit.dphi_slm)
    pulls = fit.residuals / fit.sem
    amax = np.max(np.abs(fit.amp)) if fit.amp.size else 1.0
    good = np.abs(fit.amp) > 0.15 * amax
    norm = np.where(good, fit.z / np.where(fit.amp == 0, np.nan, fit.amp), np.nan)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    grid = np.linspace(dphi.min() - 5, dphi.max() + 5, 400)
    ax1.plot(grid, np.cos(np.radians(grid) + fit.dphi_comb), "-", color="tab:blue",
             lw=1.6, label=r"fit: $\cos(\Delta\Phi_{SLM}+\Delta\Phi_{comb})$")
    sc = ax1.scatter(dphi[good], norm[good], c=fit.amp[good] * 1e6, cmap="viridis",
                     s=45, edgecolor="k", lw=0.4, zorder=3, label="measured")
    ax1.axhline(0, color="gray", ls="--", lw=0.8)
    ax1.set_xlabel(r"$\Delta\Phi_{SLM} = \phi - 180^\circ$  (deg)")
    ax1.set_ylabel(r"$Z / (2 R_k R_0)$  (normalised fringe)")
    ax1.set_title(f"Pair {tgt} interference (half fringe)")
    ax1.set_ylim(-1.4, 1.4)
    ax1.legend(loc="lower right", fontsize=8)
    fig.colorbar(sc, ax=ax1).set_label(r"$2 R_k R_0$  ($\mu$V)")

    ax2.axhspan(-1, 1, color="tab:blue", alpha=0.12, label=r"$\pm1\sigma$")
    ax2.axhline(0, color="gray", ls="--", lw=1)
    ax2.scatter(dphi, pulls, c="tab:red", s=40, edgecolor="k", lw=0.4)
    ax2.set_xlabel(r"$\Delta\Phi_{SLM}$  (deg)")
    ax2.set_ylabel("Pull = residual / SEM")
    ax2.set_title(f"Pulls  ($\\chi^2$/dof = {fit.chi2_red:.2f})")
    ax2.legend(loc="upper right", fontsize=8)

    txt = (
        f"$\\Delta\\Phi_{{comb}}$ = {fit.dphi_comb_deg:+.2f} $\\pm$ "
        f"{np.degrees(fit.dphi_comb_err):.2f} deg  "
        f"({_sigma(fit.dphi_comb, fit.dphi_comb_err):.0f}$\\sigma$)\n"
        f"V = {fit.visibility:.3f} $\\pm$ {fit.visibility_err:.3f}\n"
        f"$\\chi^2$/dof = {fit.chi2_red:.2f} (Birge x{fit.birge:.2f})"
    )
    ax1.text(0.05, 0.95, txt, transform=ax1.transAxes, va="top",
             bbox=dict(boxstyle="round", fc="white", alpha=0.85), fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)


def make_report(result, tgt: int, ref: int, path, *, subtitle: str = "") -> None:
    """Ch-efficiency-style report: measured-vs-predicted full voltage + pulls.

    Reconstructs the full model prediction (step-6 background + fitted fringe)
    per averaged cell and plots it against the measured voltage, plus the pull
    distribution.  Diagonal (phi^x = phi^w, swap-trivial) cells are drawn as
    squares and off-diagonal (phi^x != phi^w) cells as circles, so a symmetry
    breakdown is visible at a glance; both are coloured by x*w.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless: write a PNG rather than open a window
    import matplotlib.pyplot as plt

    from slm_module.tpa_phase import _average_points  # same cell averaging as the fit

    fit = result.fit
    tgt_m, ref_m = result.tgt_model, result.ref_model
    if fit is None or tgt_m is None or ref_m is None:
        raise ValueError("result has no fit/models attached; run the fit first")

    # rebuild per-cell arrays in the SAME sorted order the fit used
    x_t, w_t, x_r, w_r, y, sem = _average_points(result)
    r_t = tgt_m.amplitude(x_t, w_t)
    r_r = ref_m.amplitude(x_r, w_r)
    background = (tgt_m.single_beam(x_t, w_t) + ref_m.single_beam(x_r, w_r)
                 + r_t**2 + r_r**2)                    # dark already removed in y
    y_meas = y                                          # measured, dark-subtracted
    y_pred = background + fit.z_pred                    # full model (incl. fitted fringe + c)
    pulls = fit.residuals / sem
    diag = np.abs(x_t - w_t) < 1e-6                     # phi^x = phi^w (swap-trivial)
    off = ~diag
    xw = x_t * w_t
    vmin, vmax = float(np.min(xw)), float(np.max(xw))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ---- left: measured vs predicted --------------------------------------
    lims = [min(y_meas.min(), y_pred.min()) * 1e3, max(y_meas.max(), y_pred.max()) * 1e3]
    pad = 0.03 * ((lims[1] - lims[0]) or 1.0)
    lims = [lims[0] - pad, lims[1] + pad]
    ax1.plot(lims, lims, "--", color="gray", lw=1, label="ideal")
    ax1.errorbar(y_meas * 1e3, y_pred * 1e3, xerr=sem * 1e3, fmt="none",
                 ecolor="lightgray", elinewidth=1, zorder=1)
    sc = None
    if off.any():
        sc = ax1.scatter(y_meas[off] * 1e3, y_pred[off] * 1e3, c=xw[off], cmap="viridis",
                         vmin=vmin, vmax=vmax, marker="o", s=55, edgecolor="k", lw=0.4,
                         zorder=2, label=r"off-diagonal ($\phi^x\neq\phi^w$)")
    if diag.any():
        sc_d = ax1.scatter(y_meas[diag] * 1e3, y_pred[diag] * 1e3, c=xw[diag], cmap="viridis",
                           vmin=vmin, vmax=vmax, marker="s", s=75, edgecolor="k", lw=0.6,
                           zorder=3, label=r"diagonal ($\phi^x=\phi^w$)")
        sc = sc if sc is not None else sc_d
    ax1.set_xlim(lims); ax1.set_ylim(lims)
    ax1.set_xlabel("Measured voltage, trial-averaged (mV)")
    ax1.set_ylabel("Predicted voltage, full model (mV)")
    ax1.set_title(f"Joint fit  (R$^2$ = {fit.r2:.3f})")
    ax1.legend(loc="lower right", fontsize=8)
    fig.colorbar(sc, ax=ax1).set_label(r"$x\cdot w$")

    txt = (
        f"$\\Delta\\Phi_{{comb}}$ = {fit.dphi_comb_deg:+.1f} $\\pm$ "
        f"{np.degrees(fit.dphi_comb_err):.1f} deg\n"
        f"V = {fit.visibility:.2f} $\\pm$ {fit.visibility_err:.2f}  (ideal $\\approx$1)\n"
        f"offset c = {fit.offset*1e3:+.2f} mV  (should be $\\approx$0)\n"
        f"$\\chi^2$/dof = {fit.chi2_red:.1f} (Birge x{fit.birge:.2f})"
    )
    ax1.text(0.05, 0.95, txt, transform=ax1.transAxes, va="top",
             bbox=dict(boxstyle="round", fc="white", alpha=0.85), fontsize=8)

    # ---- right: pulls ------------------------------------------------------
    ax2.axhspan(-1, 1, color="tab:blue", alpha=0.12, label=r"$\pm1\sigma$")
    ax2.axhline(0, color="gray", ls="--", lw=1)
    if off.any():
        ax2.scatter(y_pred[off] * 1e3, pulls[off], c="tab:red", marker="o", s=50,
                    edgecolor="k", lw=0.4, label=r"off-diagonal ($\phi^x\neq\phi^w$)")
    if diag.any():
        ax2.scatter(y_pred[diag] * 1e3, pulls[diag], marker="s", s=70, facecolor="none",
                    edgecolor="tab:orange", lw=1.6, label=r"diagonal ($\phi^x=\phi^w$)")
    ax2.set_xlabel("Predicted voltage, full model (mV)")
    ax2.set_ylabel("Pull = residual / SEM")
    ax2.set_title(f"Pulls  ($\\chi^2$/dof = {fit.chi2_red:.1f})")
    ax2.legend(loc="upper left", fontsize=8)

    ok = fit.chi2_red < 3.0 and 0.7 <= fit.visibility <= 1.3
    verdict = "model OK" if ok else "model REJECTED"
    head = f"TPA comb-phase fit: pair {tgt} vs pair {ref}"
    if subtitle:
        head += f" -- {subtitle}"
    fig.suptitle(f"{head}  [{verdict}]", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)


def sweep_and_fit() -> None:
    """Table 1 for every target pair vs pair 0; save per-pair + spectrum, plot."""
    import json

    calib = load_calibration_result(CALIB_PATH)
    layout = build_channel_layout(calib)
    models = load_models(layout)

    drive = build_phase_sweep(
        n_points=SWEEP_POINTS, phi_start_deg=PHI_START_DEG,
        phi_stop_deg=PHI_STOP_DEG, ref_phase_deg=REF_PHASE_DEG,
    )
    slm = connect_slm()
    daq = connect_daq()
    spectrum: dict[int, dict] = {}
    try:
        for k in TGT_INDICES:
            print(f"\n=== Table 1: pair {k} vs reference {REF_INDEX} ===")
            result = measure_phase_sweep(
                daq, slm, layout,
                tgt_index=k, ref_index=REF_INDEX,
                drive=drive, tgt_model=models[k], ref_model=models[REF_INDEX],
                n_trials=N_TRIALS, repeats=REPEATS, settle=SETTLE_S,
                read_timeout=max(30.0, DAQ_DURATION_S * 3.0 + 10.0),
                progress_callback=lambda p: print(f"[{p.step}/{p.total}] {p.message}"),
            )
            csv_path = OUT_DIR / f"tpa_phase_pair{k}_vs{REF_INDEX}.csv"
            json_path = OUT_DIR / f"tpa_phase_pair{k}_vs{REF_INDEX}.json"
            write_phase_csv(result, csv_path)
            save_phase_json(result, json_path)
            report(result.fit, k, REF_INDEX)
            make_plot(result.fit, k, OUT_DIR / f"tpa_phase_pair{k}_fit.png")
            make_report(result, k, REF_INDEX,
                        OUT_DIR / f"tpa_phase_pair{k}_report.png",
                        subtitle="Table 1 (half-fringe sweep, phi_x = phi_w)")
            spectrum[k] = {
                "dphi_comb_deg": result.fit.dphi_comb_deg,
                "dphi_comb_err_deg": float(np.degrees(result.fit.dphi_comb_err)),
                "visibility": result.fit.visibility,
                "csv": str(csv_path),
            }
    finally:
        daq.disconnect()
        slm.close_slm()

    SPECTRUM_JSON.write_text(
        json.dumps({"ref_index": REF_INDEX, "phases": spectrum}, indent=2),
        encoding="utf-8",
    )
    print("\n=== Phase spectrum {Phi_k} (deg, referenced to pair 0) ===")
    print(f"  pair {REF_INDEX}: 0.00  (reference)")
    for k in TGT_INDICES:
        s = spectrum[k]
        print(f"  pair {k}: {s['dphi_comb_deg']:+.2f} +/- {s['dphi_comb_err_deg']:.2f}  "
              f"(V={s['visibility']:.3f})")
    print(f"Spectrum written to {SPECTRUM_JSON}")


def symmetry_check() -> None:
    """Table 2: one-time 3x3 symmetry / functional-form check on TGT_INDICES[0]."""
    calib = load_calibration_result(CALIB_PATH)
    layout = build_channel_layout(calib)
    models = load_models(layout)
    k = TGT_INDICES[0]

    drive = build_symmetry_grid(ref_phase_deg=REF_PHASE_DEG)
    slm = connect_slm()
    daq = connect_daq()
    try:
        print(f"\n=== Table 2: symmetry check, pair {k} vs reference {REF_INDEX} ===")
        result = measure_phase_sweep(
            daq, slm, layout,
            tgt_index=k, ref_index=REF_INDEX,
            drive=drive, tgt_model=models[k], ref_model=models[REF_INDEX],
            n_trials=N_TRIALS, repeats=REPEATS, settle=SETTLE_S,
            read_timeout=max(30.0, DAQ_DURATION_S * 3.0 + 10.0),
            progress_callback=lambda p: print(f"[{p.step}/{p.total}] {p.message}"),
        )
    finally:
        daq.disconnect()
        slm.close_slm()

    write_phase_csv(result, OUT_DIR / f"tpa_phase_pair{k}_symmetry.csv")
    make_report(result, k, REF_INDEX,
                OUT_DIR / f"tpa_phase_pair{k}_symmetry_report.png",
                subtitle="Table 2 (symmetry grid, phi_x vs phi_w)")
    sw = swap_invariance(result)
    n_asym = sum(1 for *_, diff, sem in sw if diff > 3 * sem)
    print("\nSwap invariance  |Z(x=a,w=b) - Z(x=b,w=a)|  on the CLEAN interference")
    print("term (step-6 dark + single-beam + self-TPA removed, a_x/a_w kept on their")
    print("own channels)  (should be <~ combined SEM):")
    for a, b, z, z_sw, diff, sem in sw:
        flag = "  <-- ASYMMETRIC" if diff > 3 * sem else ""
        print(f"  x={a:.3f} w={b:.3f}: Z={z*1e3:.4f} vs {z_sw*1e3:.4f} mV  "
              f"|d|={diff*1e3:.4f} mV (SEM {sem*1e3:.4f}){flag}")

    fit = result.fit
    print(f"\nFull-model fit over the {len(sw)+3} points:")
    print(f"  chi2/dof = {fit.chi2_red:.2f},  R^2 = {fit.r2:.4f},  visibility V = {fit.visibility:.2f}")
    if n_asym == 0 and fit.chi2_red < 3.0:
        print("  VERDICT: bilinear model holds -- swaps consistent, flat pulls.")
    else:
        print(f"  VERDICT: MODEL REJECTED -- {n_asym}/{len(sw)} swaps asymmetric and "
              f"chi2/dof >> 1.")
        print("  The two target channels are not interchangeable (unequal per-channel")
        print("  phase/amplitude law or crosstalk); a single eta*sin(px/2)sin(pw/2)")
        print("  does not describe ASYMMETRIC drive of this pair.")
        if fit.visibility > 1.3:
            print(f"  Also V = {fit.visibility:.2f} >> 1: the fringe is larger than the "
                  f"step-6 etas")
            print("  predict, so the amplitude/background scale differs from step 6.")
    print("\n  (Table 1 drives phi_x = phi_w, so it never probes this x/w asymmetry;")
    print("   but check Table 1's own visibility before trusting its dPhi_comb.)")


def fit_csv(path) -> None:
    """Re-fit an already-recorded Table-1 CSV offline (no hardware)."""
    models = load_models()
    k = TGT_INDICES[0]
    result = load_phase_csv(path, models[k], models[REF_INDEX])
    dts = result.per_trial_darks()
    drift = f" +/- {dts.std(ddof=1)*1e3:.4f} drift" if dts.size > 1 else ""
    print(f"Loaded {path}: {result.trial.size} rows, "
          f"dark = {result.dark*1e3:.4f}{drift} mV")
    report(result.fit, result.tgt_index, result.ref_index)
    make_plot(result.fit, result.tgt_index, PLOT_PATH)
    print(f"\nPlot saved to {PLOT_PATH}")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    flags = {"--symmetry", "-s"}
    positional = [a for a in argv if a not in flags]
    if positional:                        # a CSV path -> offline re-fit, no hardware
        fit_csv(positional[0])
        return 0
    sweep_and_fit()                       # Table 1: always run
    if any(a in flags for a in argv):     # Table 2: only when --symmetry is given
        symmetry_check()
    return 0


if __name__ == "__main__":
    sys.exit(main())
