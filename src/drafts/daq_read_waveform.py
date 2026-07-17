"""Read one waveform from an NI-DAQ analog input -- diagnostic script *and* module.

Needs real hardware (no mocks), so it is not a pytest test.  The acquisition
itself lives in :mod:`daq_module` (``NIDAQDriver`` -> ``DAQController``): DIFF
input, one untriggered finite window, Butterworth low-pass at the detector
bandwidth, SEM over the effective sample count ``n_eff = 2*T*f_cut``.  This
draft drives that same path directly to eyeball a raw trace and its amplitude
spectrum -- the bring-up view the production monitor doesn't plot.

Two draft-local corrections sit on top of that shared path (neither touches
``daq_module``): every read is sign-inverted -- our TIA outputs negative volts
for positive light -- and the first ``SETTLE_CYCLES / f_cut`` seconds are
discarded after filtering as a turn-on transient before any statistics.

* Run it directly -- read once with the module defaults, print stats, plot
  time + frequency domain::

      python src/drafts/daq_read_waveform.py

* Import it from another draft and grab one read, overriding any acquisition
  parameter per call::

      from daq_read_waveform import measure, read_waveform
      mean_v, sem_v = measure(duration_s=1.0)         # low-passed mean + its SEM
      times, voltages = read_waveform()               # raw trace, if you want it

The module-level constants mirror the :class:`daq_module.DAQMonitorSettings`
defaults (the values the step-6/7 calibration drafts validated), so this
diagnostic can never drift from what the calibrations actually use.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from daq_module import DAQMonitorSettings, NIDAQDriver, lowpass  # noqa: E402

# Defaults come from daq_module (single source of truth); DEVICE names the
# board (see NI-MAX).
_DEFAULTS = DAQMonitorSettings()
DEVICE = "Dev1"
CHANNEL = _DEFAULTS.channel

F_CUT = _DEFAULTS.f_cut                  # detector 3 dB bandwidth (Hz)
FILTER_ORDER = _DEFAULTS.filter_order    # digital Butterworth low-pass order
SAMPLE_RATE_HZ = _DEFAULTS.sample_rate
# SAMPLE_RATE_HZ = 1_000
# DURATION_S = _DEFAULTS.duration
DURATION_S = 5
# Input range is quantized: the board only offers +/-0.1, 0.2, 0.5, 1, 2, 5, 10 V
# and rounds any request UP to the next one -- +/-0.1 V is the most sensitive.
# MIN_VAL_V = _DEFAULTS.min_val
# MAX_VAL_V = _DEFAULTS.max_val
MAX_VAL_V = 0.1
MIN_VAL_V = -0.1
# Our transimpedance amplifier outputs a NEGATIVE voltage for positive light, so
# every read is inverted to recover a positive light signal (more light -> more
# positive volts).
INVERT = True
# Leading guard discarded after filtering: the raw acquisition settles and the
# zero-phase low-pass anchors its output to the first sample, so the first
# ``SETTLE_CYCLES / f_cut`` seconds are a turn-on transient, not steady state.
SETTLE_CYCLES = 3.0


def read_waveform(
    *,
    device: str = DEVICE,
    channel: str = CHANNEL,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
    duration_s: float = DURATION_S,
    min_val_v: float = MIN_VAL_V,
    max_val_v: float = MAX_VAL_V,
    invert: bool = INVERT,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """One untriggered finite acquisition via NIDAQDriver; return ``(times, voltages)``.

    Every parameter defaults to the module-level constant, so a direct run needs
    no arguments while an importer can override any of them per call.  With
    ``invert`` (default ``INVERT``) the trace is negated so the TIA's
    negative-for-light output reads as a positive light signal.
    """
    driver = NIDAQDriver(device=device)
    driver.connect()
    voltages = driver.read_waveform(
        channel=channel, sample_rate=sample_rate_hz, duration=duration_s,
        min_val=min_val_v, max_val=max_val_v, timeout=duration_s + 10.0,
    )
    if invert:
        voltages = -np.asarray(voltages, dtype=float)   # TIA: -volts -> +light
    if verbose:
        print(f"read {voltages.size} samples ({duration_s:g} s @ {sample_rate_hz:g} S/s)")
    times = np.arange(voltages.size) / sample_rate_hz
    return times, voltages


def amplitude_spectrum(v: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Single-sided amplitude spectrum ``(freqs_hz, |V|_volts)``.

    A Hann window (with coherent-gain correction) tames spectral leakage from
    the finite record; the DC bin is dropped so the mean doesn't swamp the plot
    and the noise floor / low-pass roll-off are what you see.
    """
    n = v.size
    win = np.hanning(n)
    scale = 2.0 / np.sum(win)  # coherent gain -> single-sided amplitude in volts
    spec = np.abs(np.fft.rfft(v * win)) * scale
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs[1:], spec[1:]  # drop DC bin


def stats(v: np.ndarray, n_eff: float) -> tuple[float, float]:
    """Return ``(mean, sem)`` for a trace, sem = std / sqrt(n_eff)."""
    return float(v.mean()), float(v.std() / np.sqrt(n_eff))


def report(label: str, v: np.ndarray, n_eff: float) -> tuple[float, float]:
    """Print mean, SEM (= std / sqrt(n_eff)) and the SEM ratio; return ``(mean, sem)``."""
    mean, sem = stats(v, n_eff)
    print(
        f"{label:>8}: mean={abs(mean)*1000:.4f} mV, sem={sem*1000:.4f} mV, "
        f"sem ratio={abs(sem/mean)*100:.4f}%"
    )
    return mean, sem


def settle_samples(fs: float, f_cut: float, cycles: float = SETTLE_CYCLES) -> int:
    """Leading samples to discard as filter/detector turn-on transient.

    The raw acquisition settles over the first few detector cycles and the
    zero-phase low-pass anchors its output to the first sample, so the leading
    ``cycles / f_cut`` seconds are not steady-state signal.  Returns 0 when the
    sample rate or cutoff is non-positive.
    """
    if fs <= 0.0 or f_cut <= 0.0:
        return 0
    return int(round(cycles / f_cut * fs))


def measure(
    *,
    f_cut: float = F_CUT,
    filter_order: int = FILTER_ORDER,
    device: str = DEVICE,
    channel: str = CHANNEL,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
    duration_s: float = DURATION_S,
    min_val_v: float = MIN_VAL_V,
    max_val_v: float = MAX_VAL_V,
    invert: bool = INVERT,
    verbose: bool = False,
) -> tuple[float, float]:
    """Read one waveform and return ``(mean_v, sem_v)`` of the low-passed trace.

    Like ``DAQController.monitor_cycle`` -- band-limit to ``f_cut`` (the detector
    bandwidth) and report the mean plus its standard error -- but with two
    draft-local corrections: the read is sign-inverted (``invert``) and the first
    ``SETTLE_CYCLES / f_cut`` s are dropped after filtering as a turn-on
    transient, so the mean and its SEM (over the retained ``n_eff = 2 * T_kept *
    f_cut``) see only steady-state signal.
    """
    _, voltages = read_waveform(
        device=device, channel=channel, sample_rate_hz=sample_rate_hz,
        duration_s=duration_s, min_val_v=min_val_v, max_val_v=max_val_v,
        invert=invert, verbose=verbose,
    )
    filtered = lowpass(voltages, sample_rate_hz, f_cut, filter_order)
    n_settle = settle_samples(sample_rate_hz, f_cut)
    kept = filtered[n_settle:] if filtered.size > n_settle else filtered
    n_eff = max(2.0 * (kept.size / sample_rate_hz) * f_cut, 1.0)
    return stats(kept, n_eff)


__all__ = [
    "read_waveform",
    "lowpass",
    "amplitude_spectrum",
    "stats",
    "settle_samples",
    "measure",
    "report",
]


def main() -> None:
    import matplotlib.pyplot as plt

    times, voltages = read_waveform(verbose=True)   # already sign-inverted
    filtered = lowpass(voltages, SAMPLE_RATE_HZ, F_CUT, FILTER_ORDER)

    # Drop the leading turn-on transient before any statistics / spectrum.
    n_settle = settle_samples(SAMPLE_RATE_HZ, F_CUT)
    settle_s = n_settle / SAMPLE_RATE_HZ
    v_kept, f_kept = voltages[n_settle:], filtered[n_settle:]
    n_eff = max(2.0 * (v_kept.size / SAMPLE_RATE_HZ) * F_CUT, 1.0)
    print(
        f"Read {voltages.size} samples over {times[-1]:.3f} s; "
        f"dropped first {settle_s*1000:.0f} ms warmup ({n_settle} samples); "
        f"effective samples: {n_eff:.0f}"
    )
    report("raw", v_kept, n_eff)
    report("filtered", f_kept, n_eff)

    fig, (ax_t, ax_f) = plt.subplots(2, 1, figsize=(10, 8))

    # --- time domain (full trace shown; shaded span = discarded warmup) ---
    ax_t.plot(times, voltages * 1000.0, linewidth=0.8, alpha=0.5, label="raw")
    ax_t.plot(times, filtered * 1000.0, linewidth=1.5, label=f"low-pass {F_CUT:.1f} Hz")
    if n_settle > 0:
        ax_t.axvspan(0.0, settle_s, color="gray", alpha=0.15,
                     label=f"discarded warmup ({settle_s*1000:.0f} ms)")
    ax_t.set_title(f"{DEVICE}/{CHANNEL}  ({SAMPLE_RATE_HZ/1000:.0f} kS/s, {DURATION_S:.1f} s)")
    ax_t.set_xlabel("Time (s)")
    ax_t.set_ylabel("Voltage (mV)")
    ax_t.legend()
    ax_t.grid(True)

    # --- frequency domain (single-sided amplitude spectrum, steady-state only) ---
    f_raw, s_raw = amplitude_spectrum(v_kept, SAMPLE_RATE_HZ)
    f_filt, s_filt = amplitude_spectrum(f_kept, SAMPLE_RATE_HZ)
    ax_f.loglog(f_raw, s_raw * 1e6, linewidth=0.8, alpha=0.5, label="raw")
    ax_f.loglog(f_filt, s_filt * 1e6, linewidth=1.5, label=f"low-pass {F_CUT:.1f} Hz")
    ax_f.axvline(F_CUT, color="k", linestyle="--", linewidth=1.0, alpha=0.6,
                 label=f"{F_CUT:.1f} Hz cutoff")
    ax_f.set_title("Amplitude spectrum")
    ax_f.set_xlabel("Frequency (Hz)")
    ax_f.set_ylabel("Amplitude (uV)")
    ax_f.legend()
    ax_f.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
