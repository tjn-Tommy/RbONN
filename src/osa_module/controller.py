from __future__ import annotations

import csv
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .driver import AQ637X_Driver, OSAError


@dataclass(frozen=True)
class MeasurementSettings:
    """Immutable snapshot of the OSA measurement parameters.

    Values are passed straight through to the instrument, so they use the
    device's own unit-suffixed string format (e.g. "1550nm", "10uW"), matching
    read_trace.ipynb.
    """

    center_wl: str = "778nm"
    span: str = "8nm"
    sensitivity: str = "HIGH2"          # NORM, MID, HIGH1, HIGH2, HIGH3
    sampling_points: str = "AUTO"       # "AUTO" or a count like "1001"
    y_unit: str = "LINear"              # LINear (W) or LOGarithmic (dBm)
    reference_level: str = "10uW"
    trace_id: str = "TRA"               # TRA..TRG
    trace_mode: str = "WRITe"


@dataclass
class TraceData:
    """A downloaded spectrum: wavelengths (m) paired with power levels."""

    wavelengths: np.ndarray              # meters
    powers: np.ndarray                   # W (LINear) or dBm (LOGarithmic)
    trace_id: str
    y_unit: str
    center_wl: str = ""
    span: str = ""
    averages: int = 1                    # number of sweeps averaged into this trace

    @property
    def n_points(self) -> int:
        return int(self.wavelengths.size)

    @property
    def wavelengths_nm(self) -> np.ndarray:
        return self.wavelengths * 1e9

    @property
    def power_label(self) -> str:
        return "power_dBm" if self.y_unit.upper().startswith("LOG") else "power_W"

    def to_csv(self, path: str | Path) -> Path:
        out = Path(path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["wavelength_m", self.power_label])
            for wl, power in zip(self.wavelengths, self.powers):
                writer.writerow([wl, power])
        return out


class OSAController:
    """High-level OSA orchestration: configure, sweep, download.

    Mirrors SLMController -- it owns a driver (injectable for testing), keeps
    the public surface task-oriented, and leaves the wire protocol to the
    driver.

    Example::

        with OSAController(host="192.168.1.11") as osa:
            trace = osa.measure(MeasurementSettings(center_wl="1550nm",
                                                    span="10nm"))
            trace.to_csv("spectrum.csv")
    """

    def __init__(
        self,
        host: str | None = None,
        port: int = 10001,
        driver: Any | None = None,
        **driver_kwargs: Any,
    ):
        if driver is not None:
            self.driver = driver
        elif host is not None:
            self.driver = AQ637X_Driver(host, port, **driver_kwargs)
        else:
            raise ValueError("either host or an explicit driver is required")
        self._settings: MeasurementSettings | None = None
        self._trace_listeners: list[Callable[[TraceData], None]] = []
        self._trace_listener_lock = threading.Lock()

    @property
    def is_connected(self) -> bool:
        return self.driver.is_connected

    @property
    def settings(self) -> MeasurementSettings | None:
        return self._settings

    def connect(self) -> None:
        self.driver.connect()

    def disconnect(self) -> None:
        self.driver.disconnect()

    def add_trace_listener(self, listener: Callable[[TraceData], None]) -> None:
        """Register a callback fired with every trace :meth:`measure` returns.

        Listeners run on whatever (worker) thread called measure(), so a GUI
        listener must only hand the trace off (e.g. emit a queued Qt signal),
        never touch widgets. Listener exceptions are swallowed so a display bug
        can never abort an acquisition.
        """
        with self._trace_listener_lock:
            if listener not in self._trace_listeners:
                self._trace_listeners.append(listener)

    def remove_trace_listener(self, listener: Callable[[TraceData], None]) -> None:
        """Unregister a trace listener (no-op if it is not registered)."""
        with self._trace_listener_lock:
            try:
                self._trace_listeners.remove(listener)
            except ValueError:
                pass

    def _notify_trace_listeners(self, trace: TraceData) -> None:
        with self._trace_listener_lock:
            listeners = tuple(self._trace_listeners)
        for listener in listeners:
            try:
                listener(trace)
            except Exception:
                pass  # a monitor must never break the measurement

    def identify(self) -> str:
        return self.driver.identify()

    def configure(self, settings: MeasurementSettings) -> None:
        """Apply every measurement parameter (order follows read_trace.ipynb)."""
        driver = self.driver
        driver.set_center_wavelength(settings.center_wl)
        driver.set_span(settings.span)
        driver.set_sensitivity(settings.sensitivity)
        driver.set_sampling_points(settings.sampling_points)
        driver.set_y_unit(settings.y_unit)
        driver.set_reference_level(settings.reference_level)
        driver.select_trace(settings.trace_id, settings.trace_mode)
        self._settings = settings

    def run_single_sweep(
        self,
        *,
        timeout: float = 180.0,
        poll_interval: float = 0.5,
        stop_event: threading.Event | None = None,
    ) -> bool:
        """Trigger one sweep and poll until it completes.

        Returns True on completion, False if stop_event fires first; raises
        OSAError on timeout. Each poll is a separate locked query, so the
        wait never holds the io lock between polls.
        """
        self.driver.initiate_single_sweep()
        deadline = time.monotonic() + timeout
        while True:
            if stop_event is not None and stop_event.is_set():
                return False
            if self.driver.is_sweep_complete():
                return True
            if time.monotonic() >= deadline:
                raise OSAError(f"sweep did not complete within {timeout:.0f} s")
            time.sleep(poll_interval)

    def read_trace(self, trace_id: str | None = None) -> TraceData:
        """Download the X/Y arrays of a trace (defaults to the active one)."""
        settings = self._settings
        if trace_id is None:
            trace_id = settings.trace_id if settings is not None else "TRA"
        x = self.driver.read_trace_x(trace_id)
        y = self.driver.read_trace_y(trace_id)
        # guard against a length mismatch between the two downloads
        count = min(x.size, y.size)
        return TraceData(
            wavelengths=x[:count],
            powers=y[:count],
            trace_id=trace_id,
            y_unit=settings.y_unit if settings is not None else "",
            center_wl=settings.center_wl if settings is not None else "",
            span=settings.span if settings is not None else "",
        )

    def measure(
        self,
        settings: MeasurementSettings | None = None,
        *,
        averages: int = 1,
        timeout: float = 180.0,
        poll_interval: float = 0.5,
        stop_event: threading.Event | None = None,
    ) -> TraceData:
        """configure (optional) -> sweep(s) -> download, in one call.

        With averages > 1 the OSA is swept that many times and the power
        spectra are averaged (in software). Averaging is done in the linear
        power domain even when the display unit is dBm, so the result is a true
        power average rather than a log-domain mean; the wavelength axis comes
        from the first sweep. `timeout` applies per sweep.

        Registered trace listeners (see :meth:`add_trace_listener`) are
        notified exactly once per call with the returned trace (the averaged
        one when averages > 1), on the calling thread.
        """
        if averages < 1:
            raise ValueError("averages must be >= 1")
        if settings is not None:
            self.configure(settings)

        traces: list[TraceData] = []
        for index in range(averages):
            completed = self.run_single_sweep(
                timeout=timeout, poll_interval=poll_interval, stop_event=stop_event
            )
            if not completed:
                raise OSAError(
                    f"sweep {index + 1}/{averages} aborted before completion"
                )
            traces.append(self.read_trace())

        result = traces[0] if len(traces) == 1 else self._average_traces(traces)
        self._notify_trace_listeners(result)
        return result

    @staticmethod
    def _average_traces(traces: list[TraceData]) -> TraceData:
        """Average several spectra of the same settings into one trace."""
        first = traces[0]
        # align on the shortest sweep in case a point count ever differs
        count = min(trace.n_points for trace in traces)
        powers = np.vstack([trace.powers[:count] for trace in traces])
        if first.y_unit.upper().startswith("LOG"):
            # dBm -> mW, average, back to dBm (true power average)
            mean_power = 10.0 * np.log10((10.0 ** (powers / 10.0)).mean(axis=0))
        else:
            mean_power = powers.mean(axis=0)
        return TraceData(
            wavelengths=first.wavelengths[:count],
            powers=mean_power,
            trace_id=first.trace_id,
            y_unit=first.y_unit,
            center_wl=first.center_wl,
            span=first.span,
            averages=len(traces),
        )

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.disconnect()
