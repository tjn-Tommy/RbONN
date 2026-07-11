"""Manual smoke test: calibrate one channel pair's TPA efficiency (eta) on a grid.

Not a pytest test (no mocks, needs real hardware) -- run it directly::

    python drafts/calib_step6_test.py             # sweep hardware, fit, plot
    python drafts/calib_step6_test.py --meas       # raw meas CSV only: sweep + record, no fit
    python drafts/calib_step6_test.py some.csv     # re-fit an existing CSV offline

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

The result is written to a single ``calib_step6_result_MMDD_HHMM.json`` that
embeds the input Step-3 calibration JSON alongside every fitted pair result.
"""
from __future__ import annotations

import json
import random
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
from slm_module.tpa_pair import (  # noqa: E402
    ChannelPairGrid,
    PairFit,
    TPAPairResult,
    build_pair_points,
    build_sweep,
    fit_grid,
    load_tpa_pair_csv,
    measure_pair_grids,
    save_tpa_pair_json,
    write_tpa_pair_csv,
)

# ---- Edit these to match your setup ----
CALIB_PATH = REPO_ROOT / "src/calib_data"  # data directory: inputs + outputs live here

PAIR_INDICES = [1,3,4,5]                           # near (cols 660/680) + far (cols 600/740)
SWEEP_MIN = 0.1                                # min per-side intensity in the ramp (0..1)
SWEEP_MAX = 1.0                                 # max per-side intensity in the ramp (0..1)
N_SWEEP_POINTS = 10                              # points per 1-D curve (x-only / w-only / cross)
IN_STEP3 = CALIB_PATH / "calib_step3_0710.json"    # Step 3 calib (near pair 0 + far pair 3)

SLM_DISPLAY_NO = None           # None -> auto-detect the LCOS-SLM display (like the GUI's Detect)
USB_SLM_NO = 1                   # SLM_Ctrl_* device index for the DVI-mode switch (USB link)

DAQ_DEVICE = "Dev1"
DAQ_CHANNEL = "ai0"
DAQ_F_CUT_HZ = 3.5             # DAQ low-pass 3 dB bandwidth (matches DAQMonitorSettings.f_cut)

# ---- Adaptive per-point averaging (DAQController) ----
# Each reading picks its own duration so its SEM meets
# max(TARGET_REL*|mean|, SEM_FLOOR), capped at T_MAX.  Bright cross points finish
# fast; near-zero single-beam points stop at the absolute SEM_FLOOR instead of
# chasing an unreachable 1% relative target.  Recorded per-point in voltage_sem_v.
TARGET_REL = 0.01               # target relative SEM (SEM/|mean|)
SEM_FLOOR = 60e-6               # absolute SEM floor (V) for near-zero-signal points
T_PROBE = 1.0                   # probe / minimum window per point (s)
T_MAX = 10.0                    # cap per point (s)

# Refitting OLD single-column CSVs (voltage_std_v held the RAW std, no
# voltage_sem_v column): the SEM is reconstructed as std/sqrt(n_eff),
# n_eff = 2*DAQ_DURATION_S*DAQ_F_CUT_HZ.  Auto-detected from the CSV header --
# new two-column CSVs carry voltage_sem_v and skip this entirely.
DAQ_DURATION_S = 5.0            # fixed window those legacy CSVs were recorded at

SETTLE_S = 0.25                  # wait after each SLM pattern change, before reading
REPEATS = 1                      # repeated monitor readings averaged per grid point
N_TRIALS = 1                    # times the whole grid is repeated (single-trial sigma comes from voltage_sem_v)


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


def make_plot(fit: PairFit, path: str | Path | None = None) -> None:
    """Measured data with the fitted TPA *model curve* overlaid.

    The model ``Y = eta^2*(x*w) + a_x*x + q_x*x^2 + a_w*w + q_w*w^2 + d`` is a
    2-D surface, so this shows it as 1-D slices: Y vs w at each fixed x level
    (left) and Y vs x at each fixed w level (right).  Every slice is a smooth
    *quadratic* curve evaluated from the fitted parameters over a fine grid --
    not a straight line between points -- overlaid on the trial-averaged
    measurements (with SEM error bars).  Curves are colored by the held-fixed
    level so the full surface is legible in two panels.

    Writes a headless PNG to ``path``; if ``path`` is None, opens the figure in
    an interactive window instead (the offline refit uses this to eyeball a
    single random pair without writing any file).
    """
    import matplotlib

    if path is not None:
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
    if path is not None:
        fig.savefig(path, dpi=150)
    else:
        plt.show()


def _load_layout():
    """Load the Step-3 calibration -> channel layout, validating PAIR_INDICES.

    Shared by the fit run and the meas-only run: both need the same layout and the
    same in-range check on the configured pair indices.
    """
    if not IN_STEP3.is_file():
        raise FileNotFoundError(
            f"Step-3 calibration not found: {IN_STEP3}\n"
            f"(CALIB_PATH is the calib_data directory; IN_STEP3 is the JSON in it.)"
        )
    layout = build_channel_layout(load_calibration_result(IN_STEP3))
    for pi in PAIR_INDICES:
        if not (0 <= pi < layout.n_channels):
            raise ValueError(
                f"pair index {pi} out of range (layout has {layout.n_channels} pairs)"
            )
    return layout


def save_combined_json(result: TPAPairResult, out_path: str | Path) -> Path:
    """Write the Step-6 fitted params with the Step-3 calibration JSON embedded.

    Reuses :func:`tpa_pair.save_tpa_pair_json` for the Step-6 payload (single
    source of truth for the fit serialization), then rewrites the file wrapping
    that together with the raw input Step-3 calibration JSON -- so one output
    carries both the input calibration and every fitted pair result.
    """
    out_path = Path(out_path)
    save_tpa_pair_json(result, out_path)                       # step-6 fit payload
    step6 = json.loads(out_path.read_text(encoding="utf-8"))
    step3 = json.loads(IN_STEP3.read_text(encoding="utf-8"))
    out_path.write_text(
        json.dumps({"step3": step3, "step6": step6}, indent=2), encoding="utf-8"
    )
    return out_path


def sweep_and_fit() -> None:
    """Drive the hardware grid sweep, save the CSV + combined JSON, fit + report + plot."""
    layout = _load_layout()

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
            read_timeout=max(30.0, T_MAX * 3.0 + 10.0),
            progress_callback=lambda p: print(f"[{p.step}/{p.total}] {p.message}"),
        )
    finally:
        daq.disconnect()
        slm.close_slm()

    stamp = time.strftime("%m%d_%H%M")
    csv_path = CALIB_PATH / f"calib_step6_meas_{stamp}.csv"
    json_path = CALIB_PATH / f"calib_step6_result_{stamp}.json"
    write_tpa_pair_csv(result, csv_path)
    save_combined_json(result, json_path)
    total_rows = sum(int(c.trial.size) for c in result.channels)
    print(f"\nSaved {total_rows} rows to {csv_path}")
    print(f"Saved Step-3 calib + Step-6 fits -> {json_path}")
    for grid in result.channels:
        print(f"\n=== pair {grid.index} ===")
        report(grid.fit)
        plot_path = json_path.with_name(f"calib_step6_pair{grid.index}_{stamp}.png")
        make_plot(grid.fit, plot_path)
        print(f"Plot saved to {plot_path}")


def _csv_has_sem(path: str | Path) -> bool:
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


def fit_csv(path: str | Path) -> None:
    """Re-fit an already-recorded pair-grid CSV offline (no hardware).

    Writes the same combined ``calib_step6_result_MMDD_HHMM.json`` (input Step-3
    calib + every fitted pair) as the hardware run; the timestamp keeps a refit
    from clobbering earlier results.  No PNGs are written -- instead one random
    fitted pair's model plot is shown interactively to eyeball the refit.

    Old single-column CSVs (``voltage_std_v`` held the RAW waveform std, no
    ``voltage_sem_v`` column) are auto-detected and their SEM is reconstructed as
    std/sqrt(n_eff), n_eff = 2*duration*f_cut, before re-fitting.  New two-column
    CSVs already carry the per-point SEM, so nothing is converted.
    """
    result = load_tpa_pair_csv(path)
    if not _csv_has_sem(path):
        n_eff = max(2.0 * DAQ_DURATION_S * DAQ_F_CUT_HZ, 1.0)
        for c in result.channels:
            c.voltage_sem_v = np.asarray(c.voltage_std_v, dtype=float) / np.sqrt(n_eff)
            fit_grid(c)  # re-fit with SEM weights: eta unchanged, chi2/dof now meaningful
        print(f"Legacy single-column CSV: SEM = std/sqrt(n_eff={n_eff:.0f})")
    n_axis = int(sum(
        int(((c.fit.x == 0) | (c.fit.w == 0)).sum()) for c in result.channels if c.fit
    ))
    print(f"Loaded {path}: {len(result.channels)} pair(s), {n_axis} axis cells total")
    for c in result.channels:
        print(f"\npair {c.index}:")
        report(c.fit)

    stamp = time.strftime("%m%d_%H%M")
    json_path = CALIB_PATH / f"calib_step6_result_{stamp}.json"
    save_combined_json(result, json_path)
    print(f"\nSaved Step-3 calib + Step-6 fits -> {json_path}")

    fitted = [c for c in result.channels if c.fit is not None]
    if fitted:
        c = random.choice(fitted)
        print(f"\nShowing fit for pair {c.index} (random of {len(fitted)} fitted)")
        make_plot(c.fit)


def _read_daq(daq, timeout: float) -> tuple[float, float, float]:
    """One averaged reading, its raw trace std, and the per-point SEM of the mean.

    Mirrors :func:`tpa_pair._read_mean_std` (the calibration sweep's read), so a
    meas row is the same measurement as a real step-6 row -- it is just never fit.
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


def _measure_pair(daq, slm, layout, i: int, points) -> ChannelPairGrid:
    """Sweep pair ``i`` over ``points`` and read Y; raw ChannelPairGrid, no fit.

    Same drive as :func:`tpa_pair.measure_pair_grids` (only pair ``i`` on, every
    other channel off, ``SETTLE_S`` after each pattern change), but records raw
    rows only -- it never runs the least-squares fit, so a singular grid can't
    throw away the just-measured hardware data.  Repeated ``N_TRIALS`` times.
    """
    from slm_module.encoding import encode_to_pattern

    zeros = np.zeros(layout.n_channels)
    slm_width, slm_height = slm.get_slm_info()
    read_timeout = max(30.0, T_MAX * 3.0 + 10.0)
    x_ch = layout.x_channels[i]
    w_ch = layout.w_channels[i]

    total = N_TRIALS * len(points)
    step = 0
    rows: list[tuple[int, float, float, float, float, float]] = []
    for trial in range(N_TRIALS):
        for x_val, w_val in points:
            x_vals = zeros.copy()
            w_vals = zeros.copy()
            x_vals[i] = x_val
            w_vals[i] = w_val
            slm.display_array(
                encode_to_pattern(x_vals, w_vals, layout, slm_width, slm_height)
            )
            if SETTLE_S:
                time.sleep(SETTLE_S)
            mean_v, std_v, sem_v = _read_daq(daq, read_timeout)
            rows.append((trial, float(x_val), float(w_val), mean_v, std_v, sem_v))
            step += 1
            print(f"[{step}/{total}] pair {i} trial {trial} "
                  f"x={x_val:.3f} w={w_val:.3f} -> {mean_v*1000:.4f} mV "
                  f"(SEM {sem_v*1e6:.1f} uV)")

    return ChannelPairGrid(
        index=i,
        wl_x_nm=float(x_ch.wavelength_nm),
        wl_w_nm=float(w_ch.wavelength_nm),
        nominal_wl_nm=0.5 * (x_ch.wavelength_nm + w_ch.wavelength_nm),
        x_center_x=int(x_ch.x_center),
        x_center_w=int(w_ch.x_center),
        trial=np.array([r[0] for r in rows], dtype=int),
        x=np.array([r[1] for r in rows], dtype=float),
        w=np.array([r[2] for r in rows], dtype=float),
        voltage_mean_v=np.array([r[3] for r in rows], dtype=float),
        voltage_std_v=np.array([r[4] for r in rows], dtype=float),
        voltage_sem_v=np.array([r[5] for r in rows], dtype=float),
        fit=None,
    )


def measure_only() -> None:
    """Sweep every pair's reduced grid and write ONE raw CSV -- no fit, no plot.

    Mirrors :func:`sweep_and_fit`'s drive (the reduced x-only / w-only / cross
    curves from :func:`build_pair_points`, every other channel held off) but
    records raw rows only, into a timestamped ``calib_step6_meas_MMDD_HHMM`` CSV
    with the same column layout as the normal run.  Re-fit later offline with
    ``python src/drafts/calib_step6_test.py <that_csv>``.
    """
    layout = _load_layout()

    sweep = build_sweep(SWEEP_MIN, SWEEP_MAX, N_SWEEP_POINTS)   # recorded on the result
    points = build_pair_points(SWEEP_MIN, SWEEP_MAX, N_SWEEP_POINTS)
    print(f"Meas (no fit): {len(points)} points/pair "
          f"(x-only + w-only + cross @ {N_SWEEP_POINTS} each + dark), pairs {list(PAIR_INDICES)}")
    slm = connect_slm()
    daq = connect_daq()
    channels: list[ChannelPairGrid] = []
    try:
        for i in PAIR_INDICES:
            print(f"\n=== Meas: pair {i} ===")
            channels.append(_measure_pair(daq, slm, layout, i, points))
    finally:
        daq.disconnect()
        slm.close_slm()

    result = TPAPairResult(
        sweep=sweep, n_trials=N_TRIALS, channels=channels,
        center_wl=float(getattr(layout, "center_wl", 0.0)),
    )
    csv_path = CALIB_PATH / f"calib_step6_meas_{time.strftime('%m%d_%H%M')}.csv"
    write_tpa_pair_csv(result, csv_path)
    total_rows = sum(int(c.trial.size) for c in channels)
    print(f"\nMeas CSV (pairs {list(PAIR_INDICES)}, {total_rows} rows) written to {csv_path}")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    flags = {"--meas", "-m"}
    positional = [a for a in argv if a not in flags]
    if positional:                       # a CSV path -> offline re-fit, no hardware
        fit_csv(positional[0])
        return 0
    if any(a in ("--meas", "-m") for a in argv):   # raw meas CSV only: sweep + record, no fit
        measure_only()
        return 0
    sweep_and_fit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
