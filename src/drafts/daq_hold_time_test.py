"""Manual smoke test: measure the DAQ "hold time" after an SLM pattern change.

Not a pytest test (no mocks, needs real hardware) -- run it directly::

    python src/drafts/daq_hold_time_test.py

`hold` in DAQMonitorSettings is the settle time we wait after changing the SLM
pattern before trusting a DAQ reading. This script measures how long that
actually needs to be. It settles the SLM on LEVEL_A first, then records the
DAQ continuously across a single LEVEL_A -> LEVEL_B step,

    0. SLM -> all LEVEL_A, hold PRE_RECORD_HOLD_S (settle before recording)
    1. start recording (t0)
    2. hold HOLD_A_S at LEVEL_A (clean baseline)
    3. SLM -> all LEVEL_B
    4. hold HOLD_B_S
    5. stop recording

then overlays the SLM-change instant on the waveform and reports the settling
time of the step (time until the trace stays inside a tolerance band around its
final value). That settling time is the hold time you should configure.

Because the acquisition blocks for its whole window, it runs in a background
thread; the main thread drives the SLM and timestamps each change against the
acquisition-start instant ``t0`` so the events line up with the samples.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import nidaqmx  # noqa: E402
from nidaqmx.constants import AcquisitionType, TerminalConfiguration  # noqa: E402

from slm_module.controller import SLMController  # noqa: E402

# ---- DAQ (edit to match your setup; see NI-MAX for the device name) ----
DEVICE = "Dev1"
CHANNEL = "ai0"
SAMPLE_RATE_HZ = 1_000       # 1 kS/s -> 1 ms resolution, plenty for a ~0.2 s settle
MIN_VAL_V = -2
MAX_VAL_V =2

# ---- SLM ----
SLM_DISPLAY_NO = None        # None -> auto-detect the LCOS-SLM display (like the GUI's Detect)
USB_SLM_NO = 1               # SLM_Ctrl_* device index for the DVI-mode switch (USB link)
LEVEL_A = 470                # first uniform grayscale level
LEVEL_B = 420                # second uniform grayscale level (the step we time)

# ---- Sequence timing ----
PRE_RECORD_HOLD_S = 0.5      # settle LEVEL_A before recording starts (off the clock)
HOLD_A_S = 1.0               # baseline dwell at LEVEL_A once recording has started
HOLD_B_S = 2.0               # dwell after switching to LEVEL_B
MARGIN_S = 0.5               # extra recording tail so the LEVEL_B settle is fully captured
SETTLE_FRAC = 0.05           # settling band = SETTLE_FRAC * step (widened by tail noise)
SMOOTH_MS = 20.0             # moving-average window for settle detection (<< the settle itself)

# Total finite acquisition length. Must cover t0 -> last SLM dwell end.
RECORD_S = HOLD_A_S + HOLD_B_S + MARGIN_S


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
    # display writes only hit the DVI frame buffer; if the panel is still in
    # Memory mode over USB the write is silently ignored. Force DVI mode so the
    # panel actually shows what we send (mirrors the GUI's "Switch to DVI mode").
    slm.set_dvi_mode(USB_SLM_NO)
    print(f"SLM: DVI mode set (USB device {USB_SLM_NO})")
    return slm


def record_waveform(result: dict, armed: threading.Event) -> None:
    """Background finite acquisition; stores t0 + voltages in ``result``.

    ``t0`` is stamped right after task.start() -- as close as we can get to the
    first sample -- and ``armed`` is set so the main thread starts the SLM
    sequence on the same clock reference.
    """
    n_samples = int(round(SAMPLE_RATE_HZ * RECORD_S))
    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_voltage_chan(
            f"{DEVICE}/{CHANNEL}",
            terminal_config=TerminalConfiguration.RSE,  # single-ended vs AI GND
            min_val=MIN_VAL_V,
            max_val=MAX_VAL_V,
        )
        task.timing.cfg_samp_clk_timing(
            SAMPLE_RATE_HZ, sample_mode=AcquisitionType.FINITE, samps_per_chan=n_samples
        )
        task.start()
        result["t0"] = time.perf_counter()
        armed.set()
        values = task.read(number_of_samples_per_channel=n_samples, timeout=RECORD_S + 10.0)
        task.stop()
    result["voltages"] = np.asarray(values, dtype=float)


def run_sequence(slm: SLMController) -> tuple[np.ndarray, np.ndarray, list[tuple[float, int]]]:
    """Record the DAQ across a LEVEL_A -> LEVEL_B step; return (times, voltages, events).

    LEVEL_A is set and allowed to settle *before* recording starts, so the
    recording opens on a clean LEVEL_A baseline and the only transition on the
    clock is the LEVEL_A -> LEVEL_B step. ``events`` holds (t_seconds_from_t0,
    level) for that step so the caller can align it with the samples.
    """
    # interval=0.0 throughout: issue each write immediately and let our own
    # sleeps own the schedule (so the dwell isn't double-counted by the driver).
    # Settle LEVEL_A off the clock, before recording starts.
    slm.display_grayscale(LEVEL_A, interval=0.0)
    time.sleep(PRE_RECORD_HOLD_S)

    result: dict = {}
    armed = threading.Event()
    worker = threading.Thread(target=record_waveform, args=(result, armed), daemon=True)
    worker.start()
    armed.wait()  # acquisition is running and t0 is stamped
    t0 = result["t0"]

    time.sleep(HOLD_A_S)  # clean LEVEL_A baseline, on the clock

    events: list[tuple[float, int]] = []
    events.append((time.perf_counter() - t0, LEVEL_B))
    slm.display_grayscale(LEVEL_B, interval=0.0)
    time.sleep(HOLD_B_S)

    worker.join(timeout=RECORD_S + 15.0)
    if "voltages" not in result:
        raise RuntimeError("DAQ acquisition did not finish")
    voltages = result["voltages"]
    times = np.arange(voltages.size) / SAMPLE_RATE_HZ
    return times, voltages, events


def _smooth(v: np.ndarray, fs: float, window_ms: float) -> np.ndarray:
    """Centered moving average; window kept far below the settle so it barely
    smears the step but rejects the noise spikes that would otherwise count as
    'not settled' and inflate the settling time. Edge-replicating (``nearest``)
    so the settled tail isn't biased toward zero at the array boundary."""
    from scipy.ndimage import uniform_filter1d

    n = max(1, int(round(window_ms * 1e-3 * fs)))
    if n <= 1:
        return v
    return uniform_filter1d(v, size=n, mode="nearest")


def settling_time(
    times: np.ndarray, v: np.ndarray, t_event: float, tail_s: float = 0.3
) -> tuple[float, float, float, float]:
    """Settling time of the step starting at ``t_event``.

    Detection runs on a lightly smoothed copy (see ``_smooth``). Final value is
    the mean of the last ``tail_s`` seconds; the band is ``SETTLE_FRAC * |step|``
    widened to at least 3x the (smoothed) tail noise so sample noise alone
    doesn't count as "not settled". Returns (settle_seconds, final_v, step_v,
    band_v); settle is NaN if never settled.
    """
    fs = 1.0 / float(np.median(np.diff(times)))
    vs = _smooth(v, fs, SMOOTH_MS)
    pre = vs[times < t_event]
    start_v = float(pre[-1]) if pre.size else float(vs[0])
    tail = vs[times >= times[-1] - tail_s]
    final_v = float(tail.mean())
    step_v = final_v - start_v
    band = max(SETTLE_FRAC * abs(step_v), 3.0 * float(tail.std()))

    mask = times >= t_event
    tt, vv = times[mask], vs[mask]
    outside = np.abs(vv - final_v) > band
    if not outside.any():
        return 0.0, final_v, step_v, band
    last_out = int(np.max(np.nonzero(outside)))
    if last_out + 1 >= tt.size:
        return float("nan"), final_v, step_v, band
    return float(tt[last_out + 1] - t_event), final_v, step_v, band


def main() -> None:
    import matplotlib.pyplot as plt

    slm = connect_slm()
    try:
        times, voltages, events = run_sequence(slm)
    finally:
        slm.close_slm()

    print(f"Recorded {voltages.size} samples over {times[-1]:.3f} s")
    for t_ev, level in events:
        print(f"  SLM -> {level:>4} at t = {t_ev*1000:8.1f} ms")

    t_step = events[-1][0]  # the LEVEL_A -> LEVEL_B change we time
    settle, final_v, step_v, band = settling_time(times, voltages, t_step)
    print(
        f"\nStep {LEVEL_A} -> {LEVEL_B}: "
        f"final = {final_v*1000:.4f} mV, step = {step_v*1000:.4f} mV, "
        f"band = +/-{band*1000:.4f} mV"
    )
    if np.isnan(settle):
        print(f"  did not settle within the {times[-1]-t_step:.2f} s recorded tail")
    else:
        print(f"  HOLD TIME (settling) = {settle*1000:.0f} ms")

    plt.figure(figsize=(10, 6))
    plt.plot(times, voltages * 1000.0, linewidth=0.8, label="DAQ")
    for t_ev, level in events:
        plt.axvline(t_ev, color="k", linestyle="--", linewidth=1.0, alpha=0.6)
        plt.text(t_ev, plt.ylim()[1], f" {level}", va="top", fontsize=8)
    plt.axhline(final_v * 1000.0, color="C1", linewidth=1.0, alpha=0.7, label="final")
    plt.axhspan((final_v - band) * 1000.0, (final_v + band) * 1000.0,
                color="C1", alpha=0.12, label="settling band")
    if not np.isnan(settle):
        plt.axvline(t_step + settle, color="C2", linewidth=1.2,
                    label=f"settled (+{settle*1000:.0f} ms)")
    plt.title(f"{DEVICE}/{CHANNEL}  DAQ hold-time test  ({LEVEL_A} -> {LEVEL_B})")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (mV)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
