"""High-level NI-DAQ orchestration, mirroring ScopeController's monitor role.

Only the single-averaged-reading path is implemented here -- the part
ScopeController's configure_monitor()/monitor_cycle() plays for the TPA
encoder page's "read after send" feedback. ``MonitorSample`` is reused as-is
from scope_module so the encoder page can treat a scope reading and a DAQ
reading identically.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from scope_module.controller import MonitorSample

from .driver import DAQConnectionError, DAQError, NIDAQDriver


@dataclass(frozen=True)
class DAQMonitorSettings:
    """Parameters for one untriggered averaged DAQ readout.

    No trigger: the PC arms the acquisition directly and stops it after
    ``duration`` seconds (see NIDAQDriver.read_waveform), mirroring the
    scope's software/AUTO free-run read path used for the same page.
    """

    channel: str = "ai0"
    sample_rate: float = 100.0   # Sa/s
    duration: float = 0.05           # averaging window, seconds
    hold: float = 0.1                # settle time after the SLM pattern change, seconds
    min_val: float = -0.010          # V
    max_val: float = 0.050           # V


class DAQController:
    """Connect/read wrapper around NIDAQDriver (injectable for testing)."""

    def __init__(self, device: str | None = None, *, driver: Any | None = None):
        if driver is not None:
            self.driver = driver
        elif device is not None:
            self.driver = NIDAQDriver(device=device)
        else:
            raise ValueError("either device or an explicit driver is required")
        self._settings: DAQMonitorSettings | None = None
        self.last_times: np.ndarray | None = None
        self.last_values: np.ndarray | None = None

    @property
    def is_connected(self) -> bool:
        return self.driver.is_connected

    def connect(self) -> None:
        self.driver.connect()

    def disconnect(self) -> None:
        self.driver.disconnect()

    def identify(self) -> str:
        return self.driver.identify()

    def configure_monitor(self, settings: DAQMonitorSettings) -> None:
        self._settings = settings

    def monitor_cycle(
        self,
        *,
        index: int = 0,
        timeout: float = 30.0,
        stop_event: threading.Event | None = None,
    ) -> MonitorSample | None:
        """One settle-then-read averaged sample, shaped like ScopeController's.

        Returns None if ``stop_event`` is already set. Assumes
        configure_monitor() has already run (falls back to defaults otherwise).
        The raw waveform behind the average is cached on ``last_times`` /
        ``last_values`` so callers can plot it (e.g. a "current waveform" view).
        """
        if stop_event is not None and stop_event.is_set():
            return None
        settings = self._settings or DAQMonitorSettings()
        if settings.hold:
            time.sleep(settings.hold)
        values = self.driver.read_waveform(
            channel=settings.channel,
            sample_rate=settings.sample_rate,
            duration=settings.duration,
            min_val=settings.min_val,
            max_val=settings.max_val,
            timeout=timeout,
        )
        self.last_values = values
        self.last_times = (
            np.arange(values.size, dtype=float) / settings.sample_rate
            if settings.sample_rate else np.zeros_like(values)
        )
        return MonitorSample(value=float(values.mean()), index=index, timestamp=time.time())

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.disconnect()


__all__ = [
    "DAQController",
    "DAQMonitorSettings",
    "MonitorSample",
    "DAQError",
    "DAQConnectionError",
]
