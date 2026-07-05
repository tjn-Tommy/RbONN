"""NI-DAQmx analog-input driver (e.g. USB-6251) over the ``nidaqmx`` package."""
from __future__ import annotations

import numpy as np


class DAQError(RuntimeError):
    """Base error for DAQ driver failures."""


class DAQConnectionError(DAQError):
    """Raised when the device is not connected or cannot be found."""


class NIDAQDriver:
    """Single-device NI-DAQmx analog-input driver.

    Mirrors RTO6_Driver in role: connect() verifies the device is reachable,
    read_mean() performs one untriggered finite acquisition -- the PC arms the
    task, blocks for ``duration`` seconds, then stops it -- and returns the
    window's (mean, std) in volts. There is no waveform-download / hardware
    trigger support yet; this only serves the single-averaged-reading role
    ScopeController.monitor_cycle() plays for the TPA encoder page.
    """

    name = "NI-DAQ"

    def __init__(self, device: str = "Dev1"):
        self.device = str(device)
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        try:
            from nidaqmx.system import System
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise DAQError(
                "nidaqmx is required for the DAQ driver; install with `pip install nidaqmx`"
            ) from exc
        try:
            devices = [d.name for d in System.local().devices]
        except Exception as exc:
            raise DAQConnectionError(f"failed to query NI-DAQmx system: {exc}") from exc
        if self.device not in devices:
            raise DAQConnectionError(
                f"{self.device!r} not found (available: {', '.join(devices) or 'none'})"
            )
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def _ensure_connected(self) -> None:
        if not self._connected:
            raise DAQConnectionError(f"{self.name} is not connected; call connect() first")

    def identify(self) -> str:
        self._ensure_connected()
        from nidaqmx.system import System

        dev = System.local().devices[self.device]
        return f"{self.device} ({dev.product_type})"

    def read_waveform(
        self,
        *,
        channel: str,
        sample_rate: float,
        duration: float,
        min_val: float,
        max_val: float,
        timeout: float,
    ) -> np.ndarray:
        """One untriggered finite acquisition; returns the raw voltage samples."""
        self._ensure_connected()
        import nidaqmx
        from nidaqmx.constants import AcquisitionType

        n_samples = max(1, int(round(sample_rate * duration)))
        try:
            with nidaqmx.Task() as task:
                task.ai_channels.add_ai_voltage_chan(
                    f"{self.device}/{channel}", min_val=min_val, max_val=max_val
                )
                task.timing.cfg_samp_clk_timing(
                    sample_rate,
                    sample_mode=AcquisitionType.FINITE,
                    samps_per_chan=n_samples,
                )
                task.start()
                values = task.read(number_of_samples_per_channel=n_samples, timeout=timeout)
                task.stop()
        except DAQError:
            raise
        except Exception as exc:
            raise DAQError(f"DAQ read failed: {exc}") from exc

        return np.asarray(values, dtype=float)
