"""Manual smoke test: calibrate one channel pair's TPA efficiency (eta) on a grid.

Not a pytest test (no mocks, needs real hardware) -- run it directly::

    python tests/tpa_pair_calibration_test.py             # sweep hardware, fit, plot
    python tests/tpa_pair_calibration_test.py some.csv     # re-fit an existing CSV offline

For channel pair ``PAIR_INDEX`` (x[PAIR_INDEX], w[PAIR_INDEX]) this sweeps both
sides' commanded intensity **independently** over the grid built by
``tpa_pair.build_sweep`` -- a leading 0 axis plus ``N_SWEEP_POINTS`` ramp values,
so the ``x=0`` / ``w=0`` axes are included -- with every other channel held off,
repeated ``N_TRIALS`` times, recording the DAQ reading at each point.

The sweep, the weighted-least-squares fit of

    Y = eta^2*(x*w) + a_x*x + q_x*x^2 + a_w*w + q_w*w^2 + d

and the CSV persistence all live in :mod:`slm_module.tpa_pair` (the same code the
GUI's Step 6 TPA tab uses).  This file only wires up the hardware and prints /
plots the result, so there is a single source of truth for the processing.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from daq_module.controller import DAQController, DAQMonitorSettings  # noqa: E402
from slm_module.calibration.calibration_new import load_calibration_result  # noqa: E402
from slm_module.controller import SLMController  # noqa: E402
from slm_module.encoding import build_channel_layout  # noqa: E402
from slm_module.tpa_pair import (  # noqa: E402
    PairFit,
    build_sweep,
    load_tpa_pair_csv,
    measure_pair_grids,
    write_tpa_pair_csv,
)

# ---- Edit these to match your setup ----
CALIB_PATH = REPO_ROOT / "calib_step333.json"   # Step 3 calibration result
PAIR_INDEX = 0                                  # which channel pair (x[i], w[i])
SWEEP_MIN = 0.3                                 # min per-side intensity in the ramp (0..1)
SWEEP_MAX = 1.0                                 # max per-side intensity in the ramp (0..1)
N_SWEEP_POINTS = 6                              # ramp points per side (+1 zero axis -> 7x7 grid)
OUT_CSV = REPO_ROOT / "tpa_pair0_calibration_linear.csv"
PLOT_PATH = REPO_ROOT / "tpa_fit.png"

SLM_DISPLAY_NO = None           # None -> auto-detect the LCOS-SLM display (like the GUI's Detect)
USB_SLM_NO = 1                   # SLM_Ctrl_* device index for the DVI-mode switch (USB link)

DAQ_DEVICE = "Dev1"
DAQ_CHANNEL = "ai0"
DAQ_DURATION_S = 1.0            # DAQ averaging window per reading

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
    """Joint-fit (measured vs predicted) + pulls, written to a headless PNG."""
    import matplotlib

    matplotlib.use("Agg")  # headless: write a PNG rather than open a window
    import matplotlib.pyplot as plt

    x, w = fit.x, fit.w
    y, sem, y_pred = fit.y, fit.sem, fit.y_pred
    pulls = fit.residuals / sem
    axis = (x == 0) | (w == 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    lims = [min(y.min(), y_pred.min()) * 1e3, max(y.max(), y_pred.max()) * 1e3]
    ax1.plot(lims, lims, "--", color="gray", lw=1, label="ideal")
    ax1.errorbar(y * 1e3, y_pred * 1e3, xerr=sem * 1e3, fmt="none",
                 ecolor="lightgray", elinewidth=1, zorder=1)
    ax1.scatter(y[axis] * 1e3, y_pred[axis] * 1e3, marker="s", s=55, facecolor="none",
                edgecolor="tab:orange", lw=1.4, zorder=3, label="axis (single-channel)")
    sc = ax1.scatter(y[~axis] * 1e3, y_pred[~axis] * 1e3, c=(x * w)[~axis], cmap="viridis",
                     s=50, edgecolor="k", lw=0.4, zorder=2, label="interior")
    ax1.set_xlabel("Measured voltage, trial-averaged (mV)")
    ax1.set_ylabel("Predicted voltage (mV)")
    ax1.set_title(f"Joint fit  (R$^2$ = {fit.r2:.3f})")
    ax1.legend(loc="lower right", fontsize=8)
    fig.colorbar(sc, ax=ax1).set_label("x * w (interior)")

    ax2.axhspan(-1, 1, color="tab:blue", alpha=0.12, label=r"$\pm1\sigma$")
    ax2.axhline(0, color="gray", ls="--", lw=1)
    ax2.scatter(y_pred[~axis] * 1e3, pulls[~axis], c="tab:red", s=45,
                edgecolor="k", lw=0.4, label="interior")
    ax2.scatter(y_pred[axis] * 1e3, pulls[axis], marker="s", facecolor="none",
                edgecolor="tab:orange", lw=1.4, s=55, label="axis")
    ax2.set_xlabel("Predicted voltage (mV)")
    ax2.set_ylabel("Pull = residual / SEM")
    ax2.set_title(f"Pulls  ($\\chi^2$/dof = {fit.chi2_red:.2f})")
    ax2.legend(loc="upper right", fontsize=8)

    p = fit.params
    txt = (
        f"eta = {fit.eta:.3g} $\\pm$ {fit.eta_err:.2g}  "
        f"({_sigma(fit.eta, fit.eta_err):.0f}$\\sigma$)\n"
        f"a_x = {p['a_x'][0]:.3g} $\\pm$ {p['a_x'][1]:.2g}\n"
        f"a_w = {p['a_w'][0]:.3g} $\\pm$ {p['a_w'][1]:.2g}\n"
        f"d   = {p['d'][0]*1e3:.3f} $\\pm$ {p['d'][1]*1e3:.3f} mV\n"
        f"$\\chi^2$/dof = {fit.chi2_red:.2f} (Birge x{fit.birge:.2f})"
    )
    ax1.text(0.05, 0.95, txt, transform=ax1.transAxes, va="top",
             bbox=dict(boxstyle="round", fc="white", alpha=0.85), fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)


def sweep_and_fit() -> None:
    """Drive the hardware grid sweep, save the CSV, then fit + report + plot."""
    calib = load_calibration_result(CALIB_PATH)
    layout = build_channel_layout(calib)
    if not (0 <= PAIR_INDEX < layout.n_channels):
        raise ValueError(
            f"PAIR_INDEX={PAIR_INDEX} out of range (layout has {layout.n_channels} pairs)"
        )

    sweep = build_sweep(SWEEP_MIN, SWEEP_MAX, N_SWEEP_POINTS)
    slm = connect_slm()
    daq = connect_daq()
    try:
        result = measure_pair_grids(
            daq, slm, layout,
            pair_indices=[PAIR_INDEX], sweep=sweep,
            n_trials=N_TRIALS, repeats=REPEATS, settle=SETTLE_S,
            read_timeout=max(30.0, DAQ_DURATION_S * 3.0 + 10.0),
            progress_callback=lambda p: print(f"[{p.step}/{p.total}] {p.message}"),
        )
    finally:
        daq.disconnect()
        slm.close_slm()

    write_tpa_pair_csv(result, OUT_CSV)
    grid = result.channels[0]
    print(f"\nSaved {grid.trial.size} rows to {OUT_CSV}")
    report(grid.fit)
    make_plot(grid.fit, PLOT_PATH)
    print(f"Plot saved to {PLOT_PATH}")


def fit_csv(path: str | Path) -> None:
    """Re-fit an already-recorded pair-grid CSV offline (no hardware)."""
    result = load_tpa_pair_csv(path)
    n_axis = int(sum(
        int(((c.fit.x == 0) | (c.fit.w == 0)).sum()) for c in result.channels if c.fit
    ))
    print(f"Loaded {path}: {len(result.channels)} pair(s), {n_axis} axis cells total")
    for c in result.channels:
        print(f"\npair {c.index}:")
        report(c.fit)
    if result.channels and result.channels[0].fit is not None:
        make_plot(result.channels[0].fit, PLOT_PATH)
        print(f"\nPlot saved to {PLOT_PATH}")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv:
        fit_csv(argv[0])
    else:
        sweep_and_fit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
