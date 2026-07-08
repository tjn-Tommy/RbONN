"""Manual smoke test: calibrate one channel pair's TPA efficiency (eta) on a grid.

Not a pytest test (no mocks, needs real hardware) -- run it directly::

    python tests/tpa_pair_calibration_test.py             # sweep hardware, fit, plot
    python tests/tpa_pair_calibration_test.py some.csv     # re-fit an existing CSV offline

For channel pair ``PAIR_INDEX`` (x[PAIR_INDEX], w[PAIR_INDEX]) this walks the
reduced 1-D calibration curves built by ``tpa_pair.build_pair_points`` -- one
line per fit term rather than the full 2-D grid, ``N_SWEEP_POINTS`` points each:

  * x-only  (x=r, w=0)  -- only the x channel on -> pins a_x, q_x
  * w-only  (x=0, w=r)  -- only the w channel on -> pins a_w, q_w
  * cross   (x=1, w=r)  -- x pinned at 1, w swept -> pins eta
  * one shared dark (0, 0) point anchors the offset d

with every other channel held off, repeated ``N_TRIALS`` times, recording the
DAQ reading at each point.

The sweep, the weighted-least-squares fit of

    Y = eta^2*(x*w) + a_x*x + q_x*x^2 + a_w*w + q_w*w^2 + d

and the CSV persistence all live in :mod:`slm_module.tpa_pair` (the same code the
GUI's Step 6 TPA tab uses).  This file only wires up the hardware and prints /
plots the result, so there is a single source of truth for the processing.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from daq_module.controller import DAQController, DAQMonitorSettings  # noqa: E402
from slm_module.calibration.calibration_new import load_calibration_result  # noqa: E402
from slm_module.controller import SLMController  # noqa: E402
from slm_module.encoding import build_channel_layout  # noqa: E402
from slm_module.tpa_pair import (  # noqa: E402
    PairFit,
    average_cells,
    build_pair_points,
    build_sweep,
    load_tpa_pair_csv,
    measure_pair_grids,
    save_tpa_pair_json,
    write_tpa_pair_csv,
)

# ---- Edit these to match your setup ----
CALIB_PATH = REPO_ROOT / "src/calib_data"  # data directory: inputs + outputs live here

PAIR_INDICES = [0, 3]                           # near (cols 660/680) + far (cols 600/740)
SWEEP_MIN = 0.3                                 # min per-side intensity in the ramp (0..1)
SWEEP_MAX = 1.0                                 # max per-side intensity in the ramp (0..1)
N_SWEEP_POINTS = 5                              # points per 1-D curve (x-only / w-only / cross)
IN_STEP3 = CALIB_PATH / "calib_step3_pair0-3_meas.json"    # Step 3 calib (near pair 0 + far pair 3)
OUT_CSV = CALIB_PATH / "calib_step6_meas.csv"
OUT_JSON = CALIB_PATH / "calib_step6_pair3_result.json"
PLOT_PATH = CALIB_PATH / "calib_step6_pair3_result.png"    # per-pair plot: tpa_pair{i}_2pairs_fit.png

SLM_DISPLAY_NO = None           # None -> auto-detect the LCOS-SLM display (like the GUI's Detect)
USB_SLM_NO = 1                   # SLM_Ctrl_* device index for the DVI-mode switch (USB link)

DAQ_DEVICE = "Dev1"
DAQ_CHANNEL = "ai0"
DAQ_DURATION_S = 5.0            # DAQ averaging window per reading

SETTLE_S = 0.15                  # wait after each SLM pattern change, before reading
REPEATS = 1                      # repeated monitor readings averaged per grid point
N_TRIALS = 15                    # times the whole grid is repeated


def detect_slm_display() -> int:
    """Find the LCOS-SLM display number (the GUI's Detect step).

    Probing needs the DLL, so use a throwaway controller on display 1 just to
    run detect_displays(); the real controller is then built on the found no.
    Hardcoding display 1 is what dumped the pattern onto the main monitor.
    """
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
    # display_array() only writes the DVI-mode frame buffer; if the panel's
    # video interface is still set to Memory mode over USB, that write is
    # silently ignored by the hardware. Force DVI mode so what we send is
    # actually what the panel shows (mirrors the GUI's "Switch to DVI mode").
    slm.set_dvi_mode(USB_SLM_NO)
    print(f"SLM: DVI mode set (USB device {USB_SLM_NO})")
    return slm


def connect_daq() -> DAQController:
    """DAQ is the Y-measurement instrument for this sweep.

    ``hold=0`` because ``measure_pair_grids`` owns the settle (``SETTLE_S``)
    after each pattern change -- leaving the DAQ's own hold non-zero would
    double the wait.
    """
    daq = DAQController(device=DAQ_DEVICE)
    daq.connect()
    daq.configure_monitor(
        DAQMonitorSettings(channel=DAQ_CHANNEL, duration=DAQ_DURATION_S, hold=0.0)
    )
    print(f"Monitor: DAQ ({DAQ_DEVICE}/{DAQ_CHANNEL})")
    return daq


def _sigma(value: float, err: float) -> float:
    return abs(value) / err if err else float("nan")


def report(fit: PairFit) -> None:
    """Print eta, the single-beam terms, the dark offset and the fit quality."""
    p = fit.params
    print("Model:  Y = eta^2*(x*w) + a_x*x + q_x*x^2 + a_w*w + q_w*w^2 + d")
    print("Fitted parameters (value +/- error, Birge-scaled):")
    print(f"  eta = {fit.eta:.4e} +/- {fit.eta_err:.3e}   ({_sigma(fit.eta, fit.eta_err):.1f} sigma)")
    print(f"  a_x = {p['a_x'][0]:.4e} +/- {p['a_x'][1]:.3e}   ({_sigma(*p['a_x']):.1f} sigma)")
    print(f"  a_w = {p['a_w'][0]:.4e} +/- {p['a_w'][1]:.3e}   ({_sigma(*p['a_w']):.1f} sigma)")
    print(f"  d   = {p['d'][0]*1e3:.4f} +/- {p['d'][1]*1e3:.4f} mV   ({_sigma(*p['d']):.1f} sigma)")
    print(f"  (nuisance saturation terms: q_x = {p['q_x'][0]:.3e} +/- {p['q_x'][1]:.2e} , "
          f"q_w = {p['q_w'][0]:.3e} +/- {p['q_w'][1]:.2e} )")
    print(f"  chi2/dof = {fit.chi2_red:.2f}  (dof={fit.dof})  -> Birge x{fit.birge:.2f} "
          f"on errors ;  R^2 = {fit.r2:.4f}")


def make_plot(fit: PairFit, path: str | Path) -> None:
    """Measured data with the fitted TPA *model curve* overlaid, headless PNG.

    The model ``Y = eta^2*(x*w) + a_x*x + q_x*x^2 + a_w*w + q_w*w^2 + d`` is a
    2-D surface, so this shows it as 1-D slices: Y vs w at each fixed x level
    (left) and Y vs x at each fixed w level (right).  Every slice is a smooth
    *quadratic* curve evaluated from the fitted parameters over a fine grid --
    not a straight line between points -- overlaid on the trial-averaged
    measurements (with SEM error bars).  Curves are colored by the held-fixed
    level so the full surface is legible in two panels.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless: write a PNG rather than open a window
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize

    x, w = fit.x, fit.w
    y, sem = fit.y, fit.sem

    p = fit.params
    b = p["b"][0]
    a_x, q_x = p["a_x"][0], p["q_x"][0]
    a_w, q_w = p["a_w"][0], p["q_w"][0]
    d = p["d"][0]

    def model(xx: np.ndarray, ww: np.ndarray) -> np.ndarray:
        """Fitted TPA response Y(x, w) (b = eta^2)."""
        xx = np.asarray(xx, dtype=float)
        ww = np.asarray(ww, dtype=float)
        return b * (xx * ww) + a_x * xx + q_x * xx**2 + a_w * ww + q_w * ww**2 + d

    def _norm(levels: np.ndarray) -> Normalize:
        lo, hi = float(levels.min()), float(levels.max())
        return Normalize(vmin=lo, vmax=hi if hi > lo else lo + 1e-9)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # --- left: Y vs w, one fitted quadratic per fixed x level (colored by x) ---
    x_levels = np.unique(x)
    w_fine = np.linspace(0.0, float(w.max()), 200)
    norm_x = _norm(x_levels)
    for xl in x_levels:
        color = cm.viridis(norm_x(xl))
        m = x == xl
        order = np.argsort(w[m])
        ax1.plot(w_fine, model(xl, w_fine) * 1e3, "-", color=color, lw=1.4, zorder=2)
        ax1.errorbar(w[m][order] * 1.0, y[m][order] * 1e3, yerr=sem[m][order] * 1e3,
                     fmt="o", color=color, ms=5, capsize=2, lw=0.8,
                     mec="k", mew=0.3, zorder=3)
    ax1.set_xlabel("w  (per-side level)")
    ax1.set_ylabel("Voltage (mV)")
    ax1.set_title("Y vs w  (curves = fitted model, colored by x)")
    sm_x = cm.ScalarMappable(norm=norm_x, cmap="viridis")
    sm_x.set_array([])
    fig.colorbar(sm_x, ax=ax1).set_label("x level")

    # --- right: Y vs x, one fitted quadratic per fixed w level (colored by w) ---
    w_levels = np.unique(w)
    x_fine = np.linspace(0.0, float(x.max()), 200)
    norm_w = _norm(w_levels)
    for wl in w_levels:
        color = cm.plasma(norm_w(wl))
        m = w == wl
        order = np.argsort(x[m])
        ax2.plot(x_fine, model(x_fine, wl) * 1e3, "-", color=color, lw=1.4, zorder=2)
        ax2.errorbar(x[m][order] * 1.0, y[m][order] * 1e3, yerr=sem[m][order] * 1e3,
                     fmt="o", color=color, ms=5, capsize=2, lw=0.8,
                     mec="k", mew=0.3, zorder=3)
    ax2.set_xlabel("x  (per-side level)")
    ax2.set_title("Y vs x  (curves = fitted model, colored by w)")
    sm_w = cm.ScalarMappable(norm=norm_w, cmap="plasma")
    sm_w.set_array([])
    fig.colorbar(sm_w, ax=ax2).set_label("w level")

    txt = (
        f"eta = {fit.eta:.3g} $\\pm$ {fit.eta_err:.2g}  "
        f"({_sigma(fit.eta, fit.eta_err):.0f}$\\sigma$)\n"
        f"a_x = {a_x:.3g}   a_w = {a_w:.3g}\n"
        f"d   = {d*1e3:.3f} mV\n"
        f"R$^2$ = {fit.r2:.3f}   $\\chi^2$/dof = {fit.chi2_red:.2f} (Birge x{fit.birge:.2f})"
    )
    ax1.text(0.03, 0.97, txt, transform=ax1.transAxes, va="top",
             bbox=dict(boxstyle="round", fc="white", alpha=0.85), fontsize=8)

    fig.suptitle("TPA pair fit -- model curves vs measured data", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def sweep_and_fit() -> None:
    """Drive the hardware grid sweep, save the CSV, then fit + report + plot."""
    if not IN_STEP3.is_file():
        raise FileNotFoundError(
            f"Step-3 calibration not found: {IN_STEP3}\n"
            f"(CALIB_PATH is the calib_data directory; IN_STEP3 is the JSON in it.)"
        )
    calib = load_calibration_result(IN_STEP3)
    layout = build_channel_layout(calib)
    for pi in PAIR_INDICES:
        if not (0 <= pi < layout.n_channels):
            raise ValueError(
                f"pair index {pi} out of range (layout has {layout.n_channels} pairs)"
            )

    sweep = build_sweep(SWEEP_MIN, SWEEP_MAX, N_SWEEP_POINTS)  # ramp recorded on result
    points = build_pair_points(SWEEP_MIN, SWEEP_MAX, N_SWEEP_POINTS)
    print(f"Reduced sweep: {len(points)} points/pair "
          f"(x-only + w-only + cross @ {N_SWEEP_POINTS} each + dark)")
    slm = connect_slm()
    daq = connect_daq()
    try:
        result = measure_pair_grids(
            daq, slm, layout,
            pair_indices=list(PAIR_INDICES), sweep=sweep, points=points,
            n_trials=N_TRIALS, repeats=REPEATS, settle=SETTLE_S,
            read_timeout=max(30.0, DAQ_DURATION_S * 3.0 + 10.0),
            progress_callback=lambda p: print(f"[{p.step}/{p.total}] {p.message}"),
        )
    finally:
        daq.disconnect()
        slm.close_slm()

    write_tpa_pair_csv(result, OUT_CSV)
    save_tpa_pair_json(result, OUT_JSON)
    total_rows = sum(int(c.trial.size) for c in result.channels)
    print(f"\nSaved {total_rows} rows to {OUT_CSV}")
    print(f"Saved params JSON -> {OUT_JSON}")
    for grid in result.channels:
        print(f"\n=== pair {grid.index} ===")
        report(grid.fit)
        plot_path = PLOT_PATH.with_name(f"tpa_pair{grid.index}_2pairs_fit.png")
        make_plot(grid.fit, plot_path)
        print(f"Plot saved to {plot_path}")


def fit_csv(path: str | Path) -> None:
    """Re-fit an already-recorded pair-grid CSV offline (no hardware).

    Outputs are named by the pair index/indices found *inside the CSV*
    (``pair_index`` column), not the module-default OUT_JSON/PLOT_PATH.  So a
    refit of ``calib_step6_pair0_meas.csv`` writes ``calib_step6_pair0_result``
    files -- the export index tracks the input instead of always claiming pair3
    and clobbering the wrong file.
    """
    result = load_tpa_pair_csv(path)
    n_axis = int(sum(
        int(((c.fit.x == 0) | (c.fit.w == 0)).sum()) for c in result.channels if c.fit
    ))
    print(f"Loaded {path}: {len(result.channels)} pair(s), {n_axis} axis cells total")
    for c in result.channels:
        print(f"\npair {c.index}:")
        report(c.fit)

    # basename tracks the input's pair index (single: pair3; multi: pair0-3)
    out_dir = OUT_JSON.parent
    indices = [c.index for c in result.channels]
    tag = "-".join(str(i) for i in indices) if indices else "none"
    out_json = out_dir / f"calib_step6_pair{tag}_result.json"
    save_tpa_pair_json(result, out_json)
    print(f"\nSaved params JSON -> {out_json}")
    for c in result.channels:
        if c.fit is None:
            continue
        plot_path = out_dir / f"calib_step6_pair{c.index}_result.png"
        make_plot(c.fit, plot_path)
        print(f"Plot saved to {plot_path}")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv:
        fit_csv(argv[0])
    else:
        sweep_and_fit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
