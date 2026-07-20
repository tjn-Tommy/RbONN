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
from typing import Any, Callable

import numpy as np

from scope_module.controller import MonitorSample

from .driver import DAQConnectionError, DAQError, NIDAQDriver


def lowpass(v: np.ndarray, fs: float, f_cut: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth low-pass at ``f_cut`` Hz.

    The one low-pass every DAQ consumer shares (controller, drafts, GUI).
    Returns ``v`` unchanged when the cutoff is at/above Nyquist, when the trace
    is too short to filter, or when SciPy is unavailable -- so the caller always
    gets a usable trace back.
    """
    v = np.asarray(v, dtype=float)
    if not fs or f_cut <= 0.0:
        return v
    nyq = 0.5 * fs
    if f_cut >= nyq:
        return v
    try:
        from scipy.signal import butter, sosfiltfilt

        sos = butter(order, f_cut / nyq, btype="low", output="sos")
        return np.asarray(sosfiltfilt(sos, v), dtype=float)
    except Exception:
        # scipy missing or trace too short for the filter's padding -> raw trace
        return v


@dataclass(frozen=True)
class DAQMonitorSettings:
    """Parameters for one untriggered averaged DAQ readout.

    No trigger: the PC arms the acquisition directly and stops it after a fixed
    window (see NIDAQDriver.read_waveform), mirroring the scope's software/AUTO
    free-run read path used for the same page.  Two fixed windows: ``duration``
    (T_both -- both beams of a pair on, bright signal) and the longer
    ``single_duration`` (T_single -- at most one beam on, i.e. ``x == 0 or
    w == 0`` including the all-off dark point) -- a weak signal needs the extra
    averaging while bright points do not (``monitor_cycle(single=True)``
    selects it).

    Defaults are the values validated on hardware by the step-6/7 calibration
    runs (``src/drafts/calib_step6_test.py`` / ``calib_step7_test.py``).
    """

    channel: str = "ai0"
    sample_rate: float = 1_000.0     # Sa/s
    duration: float = 3.0            # T_both: window per read, seconds
    single_duration: float = 5.0     # T_single: window for single-beam / dark reads, seconds
    hold: float = 0.1                # settle time after the SLM pattern change, seconds
    # Input range is quantized: the board only offers +/-0.1, 0.2, 0.5, 1, 2,
    # 5, 10 V and rounds any request UP -- +/-0.1 V is the most sensitive.
    min_val: float = -0.1            # V
    max_val: float = 0.1             # V
    f_cut: float = 20.0              # Hz, hardware 3 dB bandwidth (low-pass + effective-N)
    filter_order: int = 4            # digital Butterworth low-pass order


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
        self._sample_listeners: list[Callable[[MonitorSample], None]] = []
        self._sample_listener_lock = threading.Lock()

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

    def add_sample_listener(
        self, listener: Callable[[MonitorSample], None]
    ) -> None:
        """Register a callback fired once per completed monitor_cycle().

        The listener runs on the *calling* (worker) thread with the returned
        MonitorSample, so it must only hand the sample off (e.g. emit a queued
        Qt signal), never touch widgets. Listener exceptions are swallowed so
        a display bug can never abort a measurement.
        """
        with self._sample_listener_lock:
            if listener not in self._sample_listeners:
                self._sample_listeners.append(listener)

    def remove_sample_listener(
        self, listener: Callable[[MonitorSample], None]
    ) -> None:
        """Unregister a sample listener (no-op if it is not registered)."""
        with self._sample_listener_lock:
            try:
                self._sample_listeners.remove(listener)
            except ValueError:
                pass

    def _notify_sample_listeners(self, sample: MonitorSample) -> None:
        with self._sample_listener_lock:
            listeners = tuple(self._sample_listeners)
        for listener in listeners:
            try:
                listener(sample)
            except Exception:
                pass  # a monitor must never break the measurement

    def monitor_cycle(
        self,
        *,
        index: int = 0,
        timeout: float = 30.0,
        stop_event: threading.Event | None = None,
        single: bool = False,
    ) -> MonitorSample | None:
        """One settle-then-read averaged sample, shaped like ScopeController's.

        Returns None if ``stop_event`` is already set. Assumes
        configure_monitor() has already run (falls back to defaults otherwise).
        The raw waveform behind the average is cached on ``last_times`` /
        ``last_values`` so callers can plot it (e.g. a "current waveform" view).

        The reported ``value`` / ``std`` / ``sem`` are all computed on the
        *low-passed* trace (band-limited to ``settings.f_cut``, the hardware
        bandwidth), so out-of-band noise doesn't inflate the spread.  ``sem`` is
        the standard error of the mean taken over the *effective* independent-
        sample count ``n_eff = 2 * duration * f_cut`` (Nyquist for a
        ``f_cut``-bandwidth signal), not the raw sample count -- oversampling
        past ``2 * f_cut`` adds no new information about the mean.

        ``single=True`` marks a weak point -- at most one beam on (``x == 0 or
        w == 0``, including the all-off dark): it reads the longer
        ``settings.single_duration`` window (T_single), since a weak signal
        needs the extra averaging.  Every other read uses the fixed
        ``settings.duration`` window (T_both).
        """
        if stop_event is not None and stop_event.is_set():
            return None
        settings = self._settings or DAQMonitorSettings()
        if settings.hold:
            time.sleep(settings.hold)
        duration = settings.single_duration if single else settings.duration
        values = self._read_raw(settings, duration, timeout)
        return self._sample_from(settings, values, index)

    def _read_raw(self, settings: DAQMonitorSettings, duration: float,
                  timeout: float) -> np.ndarray:
        """One untriggered finite acquisition of ``duration`` seconds."""
        return np.asarray(
            self.driver.read_waveform(
                channel=settings.channel,
                sample_rate=settings.sample_rate,
                duration=duration,
                min_val=settings.min_val,
                max_val=settings.max_val,
                timeout=timeout,
            ),
            dtype=float,
        )

    def _sample_from(self, settings: DAQMonitorSettings, values: np.ndarray,
                     index: int) -> MonitorSample:
        """Build a MonitorSample from a raw trace: low-pass, then mean/std/sem.

        ``std`` is the spread of the (low-passed) trace; ``sem`` divides it by
        sqrt(n_eff), n_eff = 2 * duration * f_cut.  Also caches the raw trace on
        ``last_values`` / ``last_times``.
        """
        self.last_values = values
        self.last_times = (
            np.arange(values.size, dtype=float) / settings.sample_rate
            if settings.sample_rate else np.zeros_like(values)
        )
        filtered = lowpass(values, settings.sample_rate, settings.f_cut, settings.filter_order)
        mean = float(filtered.mean())
        std = float(filtered.std())
        duration = values.size / settings.sample_rate if settings.sample_rate else 0.0
        n_eff = max(2.0 * duration * settings.f_cut, 1.0)
        sem = std / float(np.sqrt(n_eff))
        sem_ratio = sem / mean if mean else float("nan")
        sample = MonitorSample(
            value=mean, std=std, sem=sem, sem_ratio=sem_ratio,
            index=index, timestamp=time.time(),
        )
        self._notify_sample_listeners(sample)
        return sample

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
    "lowpass",
]
