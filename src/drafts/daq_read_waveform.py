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
from nidaqmx.constants import AcquisitionType

# Edit these to match your setup (see NI-MAX for the device name).
DEVICE = "Dev1"
CHANNEL = "ai0"
SAMPLE_RATE_HZ = 100_000  # 100 kS/s
DURATION_S = 2
MIN_VAL_V = -0.010
MAX_VAL_V = 0.050
EXPECTED_RESOLUTION_BITS = 16


def read_waveform() -> tuple[np.ndarray, np.ndarray]:
    n_samples = int(SAMPLE_RATE_HZ * DURATION_S)
    with nidaqmx.Task() as task:
        chan = task.ai_channels.add_ai_voltage_chan(
            f"{DEVICE}/{CHANNEL}", min_val=MIN_VAL_V, max_val=MAX_VAL_V
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


def main() -> None:
    import matplotlib.pyplot as plt

    times, voltages = read_waveform()
    print(f"Read {voltages.size} samples over {times[-1]:.3f} s")
    print(f"min={voltages.min()*1000:.4f} mV  max={voltages.max()*1000:.4f} mV  "
          f"mean={voltages.mean()*1000:.4f} mV")

    plt.figure(figsize=(10, 6))
    plt.plot(times, voltages * 1000.0, linewidth=0.8)
    plt.title(f"{DEVICE}/{CHANNEL}  ({SAMPLE_RATE_HZ/1000:.0f} kS/s, {DURATION_S:.1f} s)")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (mV)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
