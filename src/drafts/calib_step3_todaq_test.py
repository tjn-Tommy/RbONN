"""Manual smoke test: sweep the SLM level on a fixed pixel window, read the DAQ,
and fit the DAQ-vs-level trace to a sin^2 curve plus a DC offset.

Not a pytest test (needs real hardware) -- run it directly::

    python tests/slm_sin2_level_sweep_test.py            # sweep + read + fit + plot
    python tests/slm_sin2_level_sweep_test.py trace.csv  # re-fit an existing CSV offline (no hardware)

What it does, per the request:
  * SLM: turn on a 16-px vertical window; all other pixels held at level 420.
  * Sweep the grayscale level applied to that window from 400 to 900, step 5.
  * DAQ: for each level, read channel ai1 at 200 MS/s, wait ``HOLD`` after the
    pattern change, then average over a 1 s window -> (mean, std) in volts.
  * Plot mean +/- std vs level and fit to  y = offset + amp * sin^2(pi*(x-x0)/period).

All the tunable knobs are the CONSTANTS just below.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from daq_module.controller import DAQController, DAQMonitorSettings  # noqa: E402
from slm_module.controller import SLMController  # noqa: E402

# ---- Edit these to match your setup ----------------------------------------
# SLM pixel window to turn "on" (columns 645..660 inclusive = 16 px).
WINDOW_X_START = 645
WINDOW_PX = 16                    # 660 - 645 + 1
BACKGROUND_LEVEL = 420           # all other SLM pixels held at level 420
SLM_INTERVAL_S = 0.2            # DVI settle passed to the SLM driver on each frame

# Level sweep (grayscale / phase level applied to the window), inclusive of stop.
LEVEL_START = 400
LEVEL_STOP = 900
LEVEL_STEP = 5

# DAQ readout.  NOTE: 200 MS/s over a 1 s window is 2e8 samples (~1.6 GB of
# float64) per point -- well above a USB-6251's ~1.25 MS/s ceiling.  It is set
# here exactly as requested; drop SAMPLE_RATE (e.g. 200e3) if the board rejects
# it or you hit a memory wall.
DAQ_DEVICE = "Dev1"
DAQ_CHANNEL = "ai1"
SAMPLE_RATE = 100e3              # 200 MS/s
AVERAGE_S = 1.0                  # averaging window per level
HOLD_S = 0.15                    # settle after the SLM pattern change (150 ms)
MIN_VAL = -0.1                   # DAQ input range, volts -- set to your detector's
MAX_VAL = 0.1

SLM_DISPLAY_NO = None            # None -> auto-detect the LCOS-SLM display
USB_SLM_NO = 1                   # SLM_Ctrl_* device index for the DVI-mode switch

OUT_DIR = REPO_ROOT
CSV_PATH = OUT_DIR / "slm_sin2_level_sweep.csv"
PLOT_PATH = OUT_DIR / "slm_sin2_level_sweep.png"


# ---- hardware wiring (mirrors tpa_phase_calib_test.py) ----------------------
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
    """DAQ is the measurement instrument; hold owns the settle after each frame."""
    daq = DAQController(device=DAQ_DEVICE)
    daq.connect()
    daq.configure_monitor(
        DAQMonitorSettings(
            channel=DAQ_CHANNEL,
            sample_rate=SAMPLE_RATE,
            duration=AVERAGE_S,
            hold=HOLD_S,
            min_val=MIN_VAL,
            max_val=MAX_VAL,
        )
    )
    print(f"DAQ: {DAQ_DEVICE}/{DAQ_CHANNEL} @ {SAMPLE_RATE:g} Sa/s, "
          f"avg {AVERAGE_S:g}s, hold {HOLD_S*1e3:.0f}ms")
    return daq


# ---- sweep ------------------------------------------------------------------
def run_sweep() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sweep the window level; return (levels, mean_V, std_V) as arrays."""
    levels = np.arange(LEVEL_START, LEVEL_STOP + 1, LEVEL_STEP, dtype=int)
    read_timeout = max(30.0, AVERAGE_S * 3.0 + 10.0)

    slm = connect_slm()
    daq = connect_daq()
    means = np.full(levels.size, np.nan)
    stds = np.full(levels.size, np.nan)
    try:
        for i, level in enumerate(levels):
            slm.display_vertical_window(
                x_start=WINDOW_X_START,
                level=int(level),
                window_px=WINDOW_PX,
                background_level=BACKGROUND_LEVEL,
                interval=SLM_INTERVAL_S,
            )
            sample = daq.monitor_cycle(index=i, timeout=read_timeout)
            means[i] = sample.value
            stds[i] = sample.std
            print(f"[{i + 1}/{levels.size}] level={level:4d}  "
                  f"mean={sample.value * 1e3:+.4f} mV  std={sample.std * 1e3:.4f} mV")
    finally:
        daq.disconnect()
        slm.close_slm()

    save_csv(levels, means, stds, CSV_PATH)
    print(f"Trace saved to {CSV_PATH}")
    return levels.astype(float), means, stds


def save_csv(levels, means, stds, path: Path) -> None:
    header = "level,mean_V,std_V"
    data = np.column_stack([np.asarray(levels, float), means, stds])
    np.savetxt(path, data, delimiter=",", header=header, comments="",
               fmt=["%d", "%.10e", "%.10e"])


def load_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[None, :]
    return data[:, 0], data[:, 1], data[:, 2]


# ---- fit: y = offset + amp * sin^2(pi*(x - x0)/period) ----------------------
def sin2_model(x, offset, amp, period, x0):
    return offset + amp * np.sin(np.pi * (x - x0) / period) ** 2


def _period_guess(x: np.ndarray, y: np.ndarray) -> float:
    """Estimate the fringe period (level units) from the dominant FFT component.

    sin^2(pi*x/P) = 0.5 - 0.5*cos(2*pi*x/P), so the trace oscillates with
    period P; the strongest non-DC rfft bin gives 1/P.
    """
    n = x.size
    if n < 4:
        return float(np.ptp(x)) or 1.0
    dx = float(np.mean(np.diff(x)))
    spectrum = np.abs(np.fft.rfft(y - y.mean()))
    freqs = np.fft.rfftfreq(n, d=dx)
    k = int(np.argmax(spectrum[1:])) + 1  # skip DC
    if freqs[k] <= 0:
        return float(np.ptp(x)) or 1.0
    return 1.0 / freqs[k]


def fit_sin2(x: np.ndarray, y: np.ndarray, sigma: np.ndarray | None = None):
    """Least-squares fit to the sin^2-plus-offset model; returns (popt, pcov, r2)."""
    from scipy.optimize import curve_fit

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    period0 = _period_guess(x, y)
    p0 = [float(y.min()), float(np.ptp(y)) or 1.0, period0, float(x.min())]

    kw = {}
    if sigma is not None and np.all(np.asarray(sigma) > 0):
        kw = {"sigma": np.asarray(sigma, float), "absolute_sigma": True}

    popt, pcov = curve_fit(
        sin2_model, x, y, p0=p0,
        bounds=([-np.inf, -np.inf, 1e-6, -np.inf], [np.inf, np.inf, np.inf, np.inf]),
        maxfev=20000, **kw,
    )
    resid = y - sin2_model(x, *popt)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - np.sum(resid ** 2) / ss_tot if ss_tot > 0 else float("nan")
    return popt, pcov, r2


def report_fit(popt, pcov, r2) -> None:
    offset, amp, period, x0 = popt
    perr = np.sqrt(np.diag(pcov))
    print("\nFit:  y = offset + amp * sin^2(pi*(x - x0)/period)")
    print(f"  offset = {offset * 1e3:+.4f} +/- {perr[0] * 1e3:.4f} mV")
    print(f"  amp    = {amp * 1e3:+.4f} +/- {perr[1] * 1e3:.4f} mV")
    print(f"  period = {period:.3f} +/- {perr[2]:.3f} levels  (2*pi phase span)")
    print(f"  x0     = {x0:.3f} +/- {perr[3]:.3f} levels")
    print(f"  R^2    = {r2:.5f}")


def make_plot(x, y, sigma, popt, r2, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")  # headless: write a PNG rather than open a window
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(x, y * 1e3, yerr=np.asarray(sigma) * 1e3, fmt="o", ms=4,
                color="tab:blue", ecolor="lightgray", elinewidth=1, capsize=2,
                label="DAQ mean +/- std")
    grid = np.linspace(x.min(), x.max(), 1000)
    ax.plot(grid, sin2_model(grid, *popt) * 1e3, "-", color="tab:red", lw=1.6,
            label=r"fit: offset + amp$\cdot\sin^2(\pi(x-x_0)/P)$")

    offset, amp, period, x0 = popt
    txt = (f"amp = {amp * 1e3:.3f} mV\n"
           f"offset = {offset * 1e3:.3f} mV\n"
           f"period = {period:.1f} levels\n"
           f"$R^2$ = {r2:.4f}")
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", fc="white", alpha=0.85), fontsize=9)

    ax.set_xlabel("SLM level (window 645-660)")
    ax.set_ylabel("DAQ ai1 (mV)")
    ax.set_title("SLM level sweep -- sin$^2$ fit")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Plot saved to {path}")


# ---- entry point ------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv:                              # a CSV path -> offline re-fit, no hardware
        levels, means, stds = load_csv(argv[0])
        print(f"Loaded {argv[0]}: {levels.size} points")
    else:
        levels, means, stds = run_sweep()

    popt, pcov, r2 = fit_sin2(levels, means, stds)
    report_fit(popt, pcov, r2)
    make_plot(levels, means, stds, popt, r2, PLOT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
