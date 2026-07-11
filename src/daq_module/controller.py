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


def _lowpass(v: np.ndarray, fs: float, f_cut: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth low-pass at ``f_cut`` Hz (ported from the
    ``daq_read_waveform`` smoke test).

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

    No trigger: the PC arms the acquisition directly and stops it after
    ``duration`` seconds (see NIDAQDriver.read_waveform), mirroring the
    scope's software/AUTO free-run read path used for the same page.
    """

    channel: str = "ai0"
    sample_rate: float = 100.0   # Sa/s
    duration: float = 0.05           # fixed averaging window, seconds (adaptive=False)
    hold: float = 0.1                # settle time after the SLM pattern change, seconds
    min_val: float = -0.010          # V
    max_val: float = 0.050           # V
    f_cut: float = 3.5               # Hz, hardware 3 dB bandwidth (low-pass + effective-N)
    filter_order: int = 4            # digital Butterworth low-pass order

    # ---- Adaptive per-point averaging (see DAQController._adaptive_read) ----
    # When ``adaptive`` is set, ``duration`` is ignored: each reading probes for
    # ``t_probe`` seconds, then extends until its SEM meets
    # ``max(target_rel*|mean|, sem_floor)``, capped at ``t_max``.  Bright points
    # finish fast; near-zero-signal points stop at the absolute ``sem_floor``
    # instead of chasing an unreachable relative target.
    adaptive: bool = False           # per-point dynamic duration to hit a SEM target
    target_rel: float = 0.01         # target relative SEM (SEM/|mean|)
    sem_floor: float = 60e-6         # absolute SEM floor (V) for near-zero signals
    t_probe: float = 1.0             # probe / minimum window per point, seconds
    t_max: float = 10.0              # cap per point, seconds


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

        With ``settings.adaptive`` set, the fixed ``duration`` window is replaced
        by a per-point dynamic one (see :meth:`_adaptive_read`).
        """
        if stop_event is not None and stop_event.is_set():
            return None
        settings = self._settings or DAQMonitorSettings()
        if settings.hold:
            time.sleep(settings.hold)
        if settings.adaptive:
            return self._adaptive_read(settings, index=index, timeout=timeout,
                                       stop_event=stop_event)
        values = self._read_raw(settings, settings.duration, timeout)
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
        filtered = _lowpass(values, settings.sample_rate, settings.f_cut, settings.filter_order)
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

    def _adaptive_read(
        self,
        settings: DAQMonitorSettings,
        *,
        index: int,
        timeout: float,
        stop_event: threading.Event | None,
    ) -> MonitorSample | None:
        """Per-point dynamic duration: probe once, then extend to hit a SEM target.

        The low-passed waveform std is (to first order) independent of the window
        length, so a short probe of ``t_probe`` seconds predicts the SEM at any
        duration via ``SEM(T) = std / sqrt(2*T*f_cut)``.  Inverting that for the
        SEM target ``max(target_rel*|mean|, sem_floor)`` gives the duration needed;
        it is clamped to ``[t_probe, t_max]``.

        The probe samples are kept and pooled with the extension read, so no
        acquisition time is wasted -- and the returned mean/std/sem are recomputed
        on the *full pooled trace*, so the recorded uncertainty is the true
        achieved SEM even if the probe's duration estimate was slightly off.  Near-
        zero-signal points hit the absolute ``sem_floor`` (and its bounded
        duration) instead of an unreachable relative target.
        """
        f_cut = settings.f_cut
        t_probe = settings.t_probe if settings.t_probe > 0 else settings.duration
        values = self._read_raw(settings, t_probe, timeout)

        filtered = _lowpass(values, settings.sample_rate, f_cut, settings.filter_order)
        mean = float(filtered.mean())
        std = float(filtered.std())
        sem_target = max(settings.target_rel * abs(mean), settings.sem_floor)
        if f_cut > 0.0 and sem_target > 0.0:
            t_need = std * std / (2.0 * f_cut * sem_target * sem_target)
        else:
            t_need = settings.t_max
        t_total = min(max(t_need, t_probe), settings.t_max)

        if t_total > t_probe * 1.001:
            if stop_event is not None and stop_event.is_set():
                return None
            # Back-to-back finite reads with a ~ms re-arm gap; for an f_cut-limited
            # signal the seam is negligible, so treat the concatenation as one trace.
            extra = self._read_raw(settings, t_total - t_probe, timeout)
            values = np.concatenate([values, extra])

        return self._sample_from(settings, values, index)

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
