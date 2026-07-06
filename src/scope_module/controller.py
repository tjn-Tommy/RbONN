from __future__ import annotations

import csv
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .driver import RTO6_Driver, ScopeError

# Map a SCPI FORMat:DATA token to the struct code query_binary_values needs.
_DATATYPE = {"REAL,32": "f", "REAL": "f", "INT,16": "h", "INT,8": "b"}


@dataclass(frozen=True)
class ScopeSettings:
    """Immutable snapshot of the channel-1 acquisition parameters.

    Vertical fields are passed straight through to the instrument in its own
    unit format (volts), so leave them None to keep the scope's current setup.
    The defaults capture ~1 s on ch1 in peak-detect mode at a reduced record
    length -- small to transfer, but the min/max decimation still preserves the
    true pulse peaks of the 80 MHz signal (its "signal strength").
    """

    channel: int = 1
    vertical_scale: str | None = None      # V/div, e.g. "0.1"
    vertical_offset: str | None = None     # V
    coupling: str | None = None            # e.g. "DC", "DCLimit", "AC"
    time_range: str = "1.0"                # total acquisition window, seconds
    record_length: int | None = 1_000_000  # points; None -> scope decides
    decimation: str = "PDETect"            # SAMPle | PDETect | HRESolution | RMS
    arithmetics: str = "OFF"               # OFF (raw) | AVERage | ENVelope
    data_format: str = "REAL,32"           # REAL,32 (volts) or INT,16 (raw)
    bandwidth_limit: str | None = None     # FULL | B800 | B200 | B20 (None=keep)
    digital_filter_cutoff: float | None = None  # Hz low-pass (None=off)

    @property
    def datatype(self) -> str:
        return _DATATYPE.get(self.data_format.upper().replace(" ", ""), "f")


@dataclass(frozen=True)
class MonitorSettings:
    """Parameters for the triggered per-event mean readout (monitor loop).

    Each SLM trigger starts one acquisition; after ``hold`` seconds of settling
    the scope averages the signal (MEAN measurement) over ``duration`` seconds
    and returns one value. HRESolution + a bandwidth/digital-filter low-pass
    maximise the accuracy of that mean for a smooth, low-frequency signal.
    """

    channel: int = 1
    trigger_mode: str = "NORMal"           # NORMal (hardware edge) | FREerun (software/immediate)
    trigger_source: str = "CHANnel1"       # CHANnel1..4 or EXTernanalog (NORMal only)
    trigger_level: float | None = 1.5      # volts; SLM pulse is 0->3 V (NORMal only)
    trigger_slope: str = "POSitive"        # rising edge
    hold: float = 0.1                      # settle time after trigger, seconds
    duration: float = 1.0                  # averaging window, seconds
    record_length: int | None = 1_000_000  # points across hold+duration
    decimation: str = "HRESolution"        # best SNR for a smooth signal
    coupling: str | None = None
    vertical_scale: str | None = None
    bandwidth_limit: str | None = None     # e.g. "B20"
    digital_filter_cutoff: float | None = None  # Hz

    @property
    def total_window(self) -> float:
        return float(self.hold) + float(self.duration)


@dataclass
class MonitorSample:
    """One triggered reading: the scope-computed mean of the settled window."""

    value: float                           # volts (gated MEAN of the channel)
    std: float | None = None               # volts (gated STDDev over the window)
    index: int = 0                         # sequence number in the monitor run
    timestamp: float = 0.0                 # time.time() when read
    waveform: "Waveform | None" = None     # optional captured trace


@dataclass
class Waveform:
    """A downloaded ch1 record: time axis paired with sample values.

    For SAMPle/HRESolution decimation ``values_per_sample`` is 1 and ``values``
    holds one level per time point. For PDETect it is 2: ``values`` holds
    interleaved (min, max) pairs, exposed split via ``mins``/``maxs``. The
    envelope peak (``maxs``) is the per-interval signal strength.
    """

    times: np.ndarray                      # seconds
    values: np.ndarray                     # volts (REAL,32) or raw counts (INT)
    values_per_sample: int = 1
    channel: int = 1
    sample_rate: float = 0.0               # Sa/s, as reported by the scope
    time_range: str = ""

    @property
    def n_points(self) -> int:
        return int(self.times.size)

    @property
    def mins(self) -> np.ndarray:
        if self.values_per_sample == 2:
            return self.values.reshape(-1, 2)[:, 0]
        return self.values

    @property
    def maxs(self) -> np.ndarray:
        if self.values_per_sample == 2:
            return self.values.reshape(-1, 2)[:, 1]
        return self.values

    def to_csv(self, path: str | Path) -> Path:
        out = Path(path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            if self.values_per_sample == 2:
                writer.writerow(["time_s", "min_V", "max_V"])
                for t, lo, hi in zip(self.times, self.mins, self.maxs):
                    writer.writerow([t, lo, hi])
            else:
                writer.writerow(["time_s", "value_V"])
                for t, v in zip(self.times, self.values):
                    writer.writerow([t, v])
        return out

    def to_npz(self, path: str | Path) -> Path:
        out = Path(path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            times=self.times,
            values=self.values,
            values_per_sample=self.values_per_sample,
            channel=self.channel,
            sample_rate=self.sample_rate,
        )
        return out


class ScopeController:
    """High-level RTO6 orchestration: configure, acquire, download.

    Mirrors OSAController -- it owns a driver (injectable for testing), keeps
    the public surface task-oriented, and leaves the wire protocol to the
    driver.

    Example::

        with ScopeController(host="192.168.1.2") as scope:
            wf = scope.acquire(ScopeSettings(time_range="1.0",
                                             record_length=1_000_000))
            wf.to_npz("ch1.npz")
    """

    def __init__(
        self,
        host: str | None = None,
        *,
        driver: Any | None = None,
        **driver_kwargs: Any,
    ):
        if driver is not None:
            self.driver = driver
        elif host is not None:
            self.driver = RTO6_Driver(host=host, **driver_kwargs)
        else:
            raise ValueError("either host or an explicit driver is required")
        self._settings: ScopeSettings | None = None

    @property
    def is_connected(self) -> bool:
        return self.driver.is_connected

    @property
    def settings(self) -> ScopeSettings | None:
        return self._settings

    def connect(self) -> None:
        self.driver.connect()

    def disconnect(self) -> None:
        self.driver.disconnect()

    def identify(self) -> str:
        return self.driver.identify()

    def configure(self, settings: ScopeSettings) -> None:
        """Apply every acquisition parameter for the chosen channel."""
        driver = self.driver
        driver.configure_channel(
            settings.channel,
            state=True,
            scale=settings.vertical_scale,
            offset=settings.vertical_offset,
            coupling=settings.coupling,
        )
        driver.set_decimation(settings.channel, settings.decimation)
        driver.set_arithmetics(settings.channel, settings.arithmetics)
        if settings.bandwidth_limit is not None:
            driver.set_bandwidth_limit(settings.channel, settings.bandwidth_limit)
        if settings.digital_filter_cutoff is not None:
            driver.set_digital_filter(settings.channel, settings.digital_filter_cutoff)
        driver.set_time_range(settings.time_range)
        driver.set_record_length(settings.record_length)
        self._settings = settings

    def run_single_acquisition(
        self,
        *,
        timeout: float = 120.0,
        poll_interval: float = 0.1,
        stop_event: threading.Event | None = None,
    ) -> bool:
        """Arm one acquisition and poll until it completes.

        Returns True on completion, False if stop_event fires first (the scope
        is told to STOP); raises ScopeError on timeout. Each poll is a separate
        locked query, so the wait never holds the io lock between polls.
        """
        self.driver.single_acquisition()
        deadline = time.monotonic() + timeout
        while True:
            if stop_event is not None and stop_event.is_set():
                self.driver.stop()
                return False
            if self.driver.is_acquisition_complete():
                return True
            if time.monotonic() >= deadline:
                self.driver.stop()
                raise ScopeError(
                    f"acquisition did not complete within {timeout:.0f} s"
                )
            time.sleep(poll_interval)

    def download(self, channel: int | None = None) -> Waveform:
        """Download the active record of a channel (defaults to the configured one)."""
        settings = self._settings
        if channel is None:
            channel = settings.channel if settings is not None else 1
        datatype = settings.datatype if settings is not None else "f"

        x_start, x_stop, record_length, values_per_sample = (
            self.driver.read_waveform_header(channel)
        )
        values = self.driver.read_waveform(channel, datatype)

        # guard against a header/payload mismatch by trusting the shorter one
        if values_per_sample >= 1:
            usable = (values.size // values_per_sample) * values_per_sample
            values = values[:usable]
            n_points = usable // values_per_sample
        else:
            n_points = values.size
        record_length = min(record_length, n_points) if record_length else n_points
        values = values[: record_length * max(values_per_sample, 1)]

        times = np.linspace(x_start, x_stop, record_length) if record_length else (
            np.empty(0, dtype=float)
        )
        try:
            sample_rate = float(self.driver.sample_rate())
        except ScopeError:
            sample_rate = 0.0
        return Waveform(
            times=times,
            values=values,
            values_per_sample=values_per_sample,
            channel=channel,
            sample_rate=sample_rate,
            time_range=settings.time_range if settings is not None else "",
        )

    def acquire(
        self,
        settings: ScopeSettings | None = None,
        *,
        timeout: float = 120.0,
        poll_interval: float = 0.1,
        stop_event: threading.Event | None = None,
    ) -> Waveform:
        """configure (optional) -> single acquisition -> download, in one call."""
        if settings is not None:
            self.configure(settings)
        completed = self.run_single_acquisition(
            timeout=timeout, poll_interval=poll_interval, stop_event=stop_event
        )
        if not completed:
            raise ScopeError("acquisition aborted before completion")
        channel = settings.channel if settings is not None else None
        return self.download(channel)

    # --- triggered monitor (per-event averaged readout) ------------------
    def configure_monitor(self, settings: MonitorSettings) -> None:
        """Set up the scope once for the triggered mean-readout loop.

        Arms a NORMal edge trigger, places the whole record after the trigger,
        applies the vertical/decimation/filter setup for a clean average, and
        gates an on-scope MEAN measurement to the settled window
        [hold, hold+duration]. After this, each monitor_cycle() waits for one
        trigger and returns the scope-computed mean -- one scalar, no waveform
        transfer.
        """
        driver = self.driver
        ch = settings.channel
        driver.configure_channel(
            ch, state=True, scale=settings.vertical_scale, coupling=settings.coupling
        )
        driver.set_decimation(ch, settings.decimation)
        driver.set_arithmetics(ch, "OFF")
        if settings.bandwidth_limit is not None:
            driver.set_bandwidth_limit(ch, settings.bandwidth_limit)
        if settings.digital_filter_cutoff is not None:
            driver.set_digital_filter(ch, settings.digital_filter_cutoff)
        driver.set_acquisition_count(1)   # exactly one acquisition per SINGle
        driver.set_record_length(settings.record_length)
        driver.set_post_trigger_window()
        if settings.trigger_mode.upper().startswith("NORM"):
            # hardware edge: arm the trigger, capture hold+duration, gate the MEAN
            # to the settled part
            driver.set_trigger(
                source=settings.trigger_source,
                level=settings.trigger_level,
                slope=settings.trigger_slope,
                mode="NORMal",
            )
            driver.set_time_range(settings.total_window)
            # group 1 = gated MEAN, group 2 = gated STDDev over the same window
            driver.setup_mean_measurement(
                ch, gate_start=settings.hold, gate_stop=settings.total_window
            )
            driver.setup_stddev_measurement(
                ch, gate_start=settings.hold, gate_stop=settings.total_window
            )
        else:
            # immediate (software) read: AUTO free-run with NO armed edge, so the
            # SINGle self-triggers and completes. Caller sleeps the hold before
            # arming, so the whole `duration` record is the settled window.
            driver.set_trigger_mode("AUTO")
            driver.set_time_range(settings.duration)
            driver.setup_mean_measurement(ch)
            driver.setup_stddev_measurement(ch)
        self._monitor_settings = settings

    def monitor_cycle(
        self,
        *,
        index: int = 0,
        timeout: float = 30.0,
        poll_interval: float = 0.05,
        stop_event: threading.Event | None = None,
        want_waveform: bool = False,
    ) -> MonitorSample | None:
        """Wait for one trigger, then return the gated MEAN (and STDDev).

        Returns None if stop_event fires before a trigger arrives. Assumes
        configure_monitor() has already run. ``timeout`` must exceed the worst
        case wait for a trigger plus the acquisition window. Both the mean
        (group 1) and its within-window standard deviation (group 2) come from
        the one completed acquisition.
        """
        settings = getattr(self, "_monitor_settings", None)
        completed = self.run_single_acquisition(
            timeout=timeout, poll_interval=poll_interval, stop_event=stop_event
        )
        if not completed:
            return None
        value = self.driver.read_measurement(group=1)
        std = self.driver.read_measurement(group=2)
        waveform = None
        if want_waveform and settings is not None:
            waveform = self.download(settings.channel)
        return MonitorSample(
            value=value, std=std, index=index, timestamp=time.time(), waveform=waveform
        )

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.disconnect()
