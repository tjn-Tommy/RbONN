"""Manual smoke test: read one waveform from an NI-DAQ analog input and plot it.

Not a pytest test (no mocks, needs real hardware) -- run it directly::

    python tests/daq_read_waveform.py

Records ``DURATION_S`` seconds on a single channel with no trigger: the PC
arms the task, blocks until the buffer fills, then stops it. Prints the
device-reported ADC resolution so you can confirm it's the expected 16 bits.
"""
from __future__ import annotations

import nidaqmx
import numpy as np
from nidaqmx.constants import AcquisitionType, TerminalConfiguration
from scipy.signal import butter, sosfiltfilt

# Edit these to match your setup (see NI-MAX for the device name).
DEVICE = "Dev1"
CHANNEL = "ai0"
F_CUT = 10 # hardware 3dB bandwidth
FILTER_ORDER = 4  # digital Butterworth low-pass order
SAMPLE_RATE_HZ = 1_000  # 100 kS/s
DURATION_S = 2
# Input range is quantized: the board only offers +/-0.1, 0.2, 0.5, 1, 2, 5, 10 V
# and rounds any request UP to the next one. Keep +/-0.1 V (most sensitive) for mV signals.
MIN_VAL_V = -0.1
MAX_VAL_V = 0.1
EXPECTED_RESOLUTION_BITS = 16


def read_waveform() -> tuple[np.ndarray, np.ndarray]:
    n_samples = int(SAMPLE_RATE_HZ * DURATION_S)
    with nidaqmx.Task() as task:
        chan = task.ai_channels.add_ai_voltage_chan(
            f"{DEVICE}/{CHANNEL}",
            terminal_config=TerminalConfiguration.RSE,  # single-ended vs AI GND
            min_val=MIN_VAL_V,
            max_val=MAX_VAL_V,
        )
        task.timing.cfg_samp_clk_timing(
            SAMPLE_RATE_HZ, sample_mode=AcquisitionType.FINITE, samps_per_chan=n_samples
        )

        resolution = chan.ai_resolution
        print(f"ADC resolution: {resolution:.0f} bits (expected {EXPECTED_RESOLUTION_BITS})")
        if int(resolution) != EXPECTED_RESOLUTION_BITS:
            print("WARNING: resolution does not match the expected 16 bits")

        task.start()
        values = task.read(number_of_samples_per_channel=n_samples, timeout=DURATION_S + 10.0)
        task.stop()

    voltages = np.asarray(values, dtype=float)
    times = np.arange(voltages.size) / SAMPLE_RATE_HZ
    return times, voltages


def lowpass(v: np.ndarray, fs: float, f_cut: float, order: int = FILTER_ORDER) -> np.ndarray:
    """Zero-phase Butterworth low-pass (SOS form) at ``f_cut`` Hz.

    Returns ``v`` unchanged if the cutoff is at/above Nyquist (nothing to do).
    """
    nyq = 0.5 * fs
    if f_cut >= nyq:
        return v
    sos = butter(order, f_cut / nyq, btype="low", output="sos")
    return sosfiltfilt(sos, v)


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


def report(label: str, v: np.ndarray, n_eff: float) -> None:
    """Print mean, SEM (= std / sqrt(n_eff)), and the SEM ratio for a trace."""
    mean = v.mean()
    sem = v.std() / np.sqrt(n_eff)
    print(
        f"{label:>8}: mean={mean*1000:.4f} mV, sem={sem*1000:.4f} mV, "
        f"sem ratio={sem/mean*100:.4f}%"
    )


def main() -> None:
    import matplotlib.pyplot as plt

    times, voltages = read_waveform()
    n_eff = 2 * DURATION_S * F_CUT
    print(f"Read {voltages.size} samples over {times[-1]:.3f} s, Effective samples: {n_eff}")

    filtered = lowpass(voltages, SAMPLE_RATE_HZ, F_CUT)
    report("raw", voltages, n_eff)
    report("filtered", filtered, n_eff)

    fig, (ax_t, ax_f) = plt.subplots(2, 1, figsize=(10, 8))

    # --- time domain ---
    ax_t.plot(times, voltages * 1000.0, linewidth=0.8, alpha=0.5, label="raw")
    ax_t.plot(times, filtered * 1000.0, linewidth=1.5, label=f"low-pass {F_CUT:.1f} Hz")
    ax_t.set_title(f"{DEVICE}/{CHANNEL}  ({SAMPLE_RATE_HZ/1000:.0f} kS/s, {DURATION_S:.1f} s)")
    ax_t.set_xlabel("Time (s)")
    ax_t.set_ylabel("Voltage (mV)")
    ax_t.legend()
    ax_t.grid(True)

    # --- frequency domain (single-sided amplitude spectrum) ---
    f_raw, s_raw = amplitude_spectrum(voltages, SAMPLE_RATE_HZ)
    f_filt, s_filt = amplitude_spectrum(filtered, SAMPLE_RATE_HZ)
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
