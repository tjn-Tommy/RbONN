"""Manual smoke test: calibrate one channel pair's TPA efficiency (eta) on a grid.

Not a pytest test (no mocks, needs real hardware) -- run it directly::

    python drafts/calib_step6_test.py             # sweep hardware, fit, plot
    python drafts/calib_step6_test.py --meas       # raw meas CSV only: sweep + record, no fit
    python drafts/calib_step6_test.py some.csv     # re-fit an existing CSV offline
    python drafts/calib_step6_test.py some.csv --flip  # re-fit sign-flipped (inverted read)

``--flip`` applies to a REFIT only (the measure path always records the raw
signal): when the photodiode/DAQ reads inverted (more light -> more negative
volts) it negates the loaded ``voltage_mean_v`` in memory (every row incl. the
(0,0) dark) and re-fits, so the fitted Y = eta^2*(x*w) + ... + d is the positive
light signal.  Nothing is written back -- the raw CSV on disk is untouched.

For channel pair ``PAIR_INDEX`` (x[PAIR_INDEX], w[PAIR_INDEX]) this walks the
reduced 1-D calibration curves built by ``tpa_pair.build_pair_points`` -- one
line per fit term rather than the full 2-D grid, ``N_SWEEP_POINTS`` points each:

  * x-only  (x=r, w=0)  -- only the x channel on -> pins a_x, q_x
  * w-only  (x=0, w=r)  -- only the w channel on -> pins a_w, q_w
  * cross   (x=1, w=r)  -- x pinned at 1, w swept -> pins eta
  * one shared dark (0, 0) point anchors the offset d

with every other channel held off, recording the DAQ reading at each point.

Each point is one fixed-duration ``daq_module`` acquisition (the same
``DAQController.monitor_cycle`` read the GUI pipeline uses): acquisition time
= ``T_SINGLE_S`` (5 s) if ``x == 0 or w == 0`` (the weak single-beam lines and
the shared dark point need the averaging), else ``T_BOTH_S`` (3 s) for the
bright both-beams cross points -- low-passed at the ``DAQMonitorSettings``
bandwidth.  The weighted-least-squares fit of

    Y = eta^2*(x*w) + a_x*x + q_x*x^2 + a_w*w + q_w*w^2 + d

and the CSV persistence live in :mod:`slm_module.tpa_pair` (the same code the
GUI's Step 6 TPA tab uses); every CSV row records the mean, its SEM and the
SEM ratio (sem/|mean|).

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
sys.path.insert(0, str(REPO_ROOT / "src" / "drafts"))  # for draft_hw

from draft_hw import connect_daq, connect_slm, read_point  # noqa: E402
from slm_module.calibration.calibration_new import load_calibration_result  # noqa: E402
from slm_module.encoding import channel_layout_from_calibration  # noqa: E402
from slm_module.tpa_pair import (  # noqa: E402
    ChannelPairGrid,
    PairFit,
    TPAPairResult,
    build_pair_points,
    build_sweep,
    fit_grid,
    load_tpa_pair_csv,
    save_tpa_pair_json,
    write_tpa_pair_csv,
)

# ---- Edit these to match your setup ----
CALIB_PATH = REPO_ROOT / "src/calib_data"  # data directory: inputs + outputs live here

PAIR_INDICES = [1,3,4,5]                           # near (cols 660/680) + far (cols 600/740)
SWEEP_MIN = 0.1                                # min per-side intensity in the ramp (0..1)
SWEEP_MAX = 1.0                                 # max per-side intensity in the ramp (0..1)
N_SWEEP_POINTS = 10                              # points per 1-D curve (x-only / w-only / cross)
IN_STEP3 = CALIB_PATH / "calib_step3b_0714_1534.json"    # Step 3 calib (near pair 0 + far pair 3)

SLM_DISPLAY_NO = None           # None -> auto-detect the LCOS-SLM display (like the GUI's Detect)
USB_SLM_NO = 1                   # SLM_Ctrl_* device index for the DVI-mode switch (USB link)

DAQ_DEVICE = "Dev1"
DAQ_CHANNEL = "ai0"

# ---- Fixed per-point acquisition (daq_module) ----
# Sample rate / range / low-pass bandwidth are the DAQMonitorSettings defaults
# (1 kS/s, +/-0.1 V DIFF, 20 Hz).  Acquisition time = T_SINGLE_S if x==0 or
# w==0 (weak single-beam / dark points), else T_BOTH_S.  Every CSV row records
# the per-point SEM (voltage_sem_v) and sem_ratio -- the per-point sigma.
T_SINGLE_S = 5.0                # at most one beam on (x==0 or w==0, incl. dark) (s)
T_BOTH_S = 3.0                  # both beams on (the bright cross points) (s)

SETTLE_S = 0.25                  # wait after each SLM pattern change, before reading


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
    same in-range check on the configured pair indices.  The Step-3b/3c rows ARE
    the channels, so the layout is loaded verbatim (the same
    ``channel_layout_from_calibration`` the GUI encoding page uses) -- no
    re-tiling, so pair indices here mean the same thing as in the UI.
    """
    if not IN_STEP3.is_file():
        raise FileNotFoundError(
            f"Step-3 calibration not found: {IN_STEP3}\n"
            f"(CALIB_PATH is the calib_data directory; IN_STEP3 is the JSON in it.)"
        )
    layout = channel_layout_from_calibration(load_calibration_result(IN_STEP3))
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
    slm = connect_slm(SLM_DISPLAY_NO, USB_SLM_NO)
    daq = connect_daq(device=DAQ_DEVICE, channel=DAQ_CHANNEL,
                      t_both=T_BOTH_S, t_single=T_SINGLE_S)
    channels: list[ChannelPairGrid] = []
    try:
        for i in PAIR_INDICES:
            print(f"\n=== Sweep: pair {i} ===")
            channels.append(_measure_pair(slm, daq, layout, i, points))
    finally:
        slm.close_slm()
        daq.disconnect()

    result = TPAPairResult(
        sweep=sweep, n_trials=1, channels=channels,
        center_wl=float(getattr(layout, "center_wl", 0.0)),
    )
    stamp = time.strftime("%m%d_%H%M")
    csv_path = CALIB_PATH / f"calib_step6_meas_{stamp}.csv"
    json_path = CALIB_PATH / f"calib_step6_result_{stamp}.json"
    write_tpa_pair_csv(result, csv_path)  # raw rows on disk BEFORE fitting
    total_rows = sum(int(c.trial.size) for c in result.channels)
    print(f"\nSaved {total_rows} rows to {csv_path}")
    for grid in result.channels:  # a singular fit can't lose the measured data now
        fit_grid(grid)
    save_combined_json(result, json_path)
    print(f"Saved Step-3 calib + Step-6 fits -> {json_path}")
    for grid in result.channels:
        print(f"\n=== pair {grid.index} ===")
        report(grid.fit)
        plot_path = json_path.with_name(f"calib_step6_pair{grid.index}_{stamp}.png")
        make_plot(grid.fit, plot_path)
        print(f"Plot saved to {plot_path}")


def fit_csv(path: str | Path, *, flip: bool = False) -> None:
    """Re-fit an already-recorded pair-grid CSV offline (no hardware).

    Writes the same combined ``calib_step6_result_MMDD_HHMM.json`` (input Step-3
    calib + every fitted pair) as the hardware run; the timestamp keeps a refit
    from clobbering earlier results.  No PNGs are written -- instead one random
    fitted pair's model plot is shown interactively to eyeball the refit.

    ``flip`` handles an inverted photodiode/DAQ read (more light -> more negative
    volts): the loaded ``voltage_mean_v`` is negated in memory on every row (incl.
    the (0,0) dark) and each pair is re-fit, so Y = eta^2*(x*w) + ... + d comes out
    as the positive light signal.  Nothing is written back -- the raw CSV on disk
    is untouched, and the spreads (std/SEM) are magnitudes so they stay as read.
    """
    result = load_tpa_pair_csv(path)
    if flip:
        for grid in result.channels:
            grid.voltage_mean_v = -grid.voltage_mean_v   # inverted read (same channel)
            fit_grid(grid)                               # re-fit the negated data
        print("Flip: negated voltage_mean_v in memory and re-fit (inverted read).")
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


def _read_point(daq, x_val: float, w_val: float) -> tuple[float, float, float, float]:
    """One fixed-duration DAQ read for a grid point; return ``(mean, std, sem, duration)``.

    Acquisition time = ``T_SINGLE_S`` if ``x == 0 or w == 0`` (at most one beam
    on, incl. the dark point), else ``T_BOTH_S``.  Filtering and
    the SEM (over ``n_eff = 2 * duration * f_cut``) happen inside
    ``DAQController.monitor_cycle`` -- the same read the GUI pipeline uses.
    ``std`` is the low-passed trace spread, so ``sem = std / sqrt(n_eff)``
    round-trips from the CSV.
    """
    single = x_val == 0.0 or w_val == 0.0
    mean_v, std_v, sem_v = read_point(daq, single=single)
    return mean_v, std_v, sem_v, (T_SINGLE_S if single else T_BOTH_S)


def _measure_pair(slm, daq, layout, i: int, points) -> ChannelPairGrid:
    """Sweep pair ``i`` over ``points`` and read Y; raw ChannelPairGrid, no fit.

    Only pair ``i`` on, every other channel off, ``SETTLE_S`` after each pattern
    change, then one fixed-duration read per point (see :func:`_read_point`).
    Records raw rows only -- the caller decides whether to fit, so a singular
    grid can't throw away the just-measured hardware data.
    """
    from slm_module.encoding import encode_to_pattern

    zeros = np.zeros(layout.n_channels)
    slm_width, slm_height = slm.get_slm_info()
    x_ch = layout.x_channels[i]
    w_ch = layout.w_channels[i]

    total = len(points)
    step = 0
    rows: list[tuple[int, float, float, float, float, float]] = []
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
        mean_v, std_v, sem_v, dur = _read_point(daq, x_val, w_val)
        rows.append((0, float(x_val), float(w_val), mean_v, std_v, sem_v))
        step += 1
        ratio = abs(sem_v / mean_v) if mean_v else float("inf")
        print(f"[{step}/{total}] pair {i} "
              f"x={x_val:.3f} w={w_val:.3f} ({dur:.0f}s) -> {mean_v*1000:.4f} mV "
              f"sem ratio {ratio*100:.2f}%")

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
    slm = connect_slm(SLM_DISPLAY_NO, USB_SLM_NO)
    daq = connect_daq(device=DAQ_DEVICE, channel=DAQ_CHANNEL,
                      t_both=T_BOTH_S, t_single=T_SINGLE_S)
    channels: list[ChannelPairGrid] = []
    try:
        for i in PAIR_INDICES:
            print(f"\n=== Meas: pair {i} ===")
            channels.append(_measure_pair(slm, daq, layout, i, points))
    finally:
        slm.close_slm()
        daq.disconnect()

    result = TPAPairResult(
        sweep=sweep, n_trials=1, channels=channels,
        center_wl=float(getattr(layout, "center_wl", 0.0)),
    )
    csv_path = CALIB_PATH / f"calib_step6_meas_{time.strftime('%m%d_%H%M')}.csv"
    write_tpa_pair_csv(result, csv_path)
    total_rows = sum(int(c.trial.size) for c in channels)
    print(f"\nMeas CSV (pairs {list(PAIR_INDICES)}, {total_rows} rows) written to {csv_path}")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    flags = {"--meas", "-m", "--flip"}
    flip = "--flip" in argv               # refit only: negate voltage_mean_v (inverted read)
    positional = [a for a in argv if a not in flags]
    if positional:                       # a CSV path -> offline re-fit, no hardware
        fit_csv(positional[0], flip=flip)
        return 0
    if any(a in ("--meas", "-m") for a in argv):   # raw meas CSV only: sweep + record, no fit
        measure_only()
        return 0
    sweep_and_fit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
