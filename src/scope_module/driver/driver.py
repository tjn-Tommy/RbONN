from __future__ import annotations

import numpy as np

from .base_scope import (
    BaseScope,
    ScopeConnectionError,
    ScopeError,
    ScopeTimeoutError,
)

# Bit 0 of the standard event status register (*ESR?) is the Operation
# Complete bit, set once a "SINGle;*OPC" acquisition has finished.
_OPC_BIT = 0x01


class RTO6_Driver(BaseScope):
    """Rohde & Schwarz RTO6-series oscilloscope over PyVISA / HiSLIP.

    Mirrors AQ637X_Driver in shape: the transport hooks wrap a single VISA
    resource, every public method is serialized by the inherited io lock, and
    the vendor command set lives here. Waveforms are downloaded as IEEE-488.2
    binary blocks (FORMat REAL,32) for speed.

    SCPI note: every command string used here is verified against the RTO6 SCPI
    manual; see scpi_reference.json (next to this package) for the annotated
    command list and manual page numbers.

    Example::

        with RTO6_Driver(host="192.168.1.2") as scope:
            scope.configure_channel(1, scale="0.1", offset="0")
            scope.set_acquisition(record_length=1_000_000, mode="PDETect",
                                  time_range="1.0")
            scope.single_acquisition()
            header = scope.read_waveform_header(1)
            values = scope.read_waveform(1)
    """

    name = "RTO6"

    def __init__(
        self,
        host: str | None = None,
        *,
        resource: str | None = None,
        timeout: float = 30.0,
        chunk_size: int = 1 << 20,
        visa_library: str = "@py",
    ):
        super().__init__()
        if resource is None:
            if host is None:
                raise ValueError("either host or an explicit resource is required")
            # HiSLIP is R&S's recommended high-throughput LAN protocol.
            resource = f"TCPIP::{host}::hislip0::INSTR"
        self.host = host
        self.resource = str(resource)
        self.timeout = float(timeout)
        self.chunk_size = int(chunk_size)
        self.visa_library = str(visa_library)
        self._rm = None
        self._inst = None

    # --- transport hooks --------------------------------------------------
    def _open_transport(self) -> None:
        try:
            import pyvisa
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ScopeError(
                "pyvisa is required for the RTO6 driver; "
                "install with `conda install -c conda-forge pyvisa pyvisa-py`"
            ) from exc

        try:
            self._rm = pyvisa.ResourceManager(self.visa_library)
            inst = self._rm.open_resource(self.resource)
        except Exception as exc:  # pyvisa raises a variety of errors
            self._discard()
            raise ScopeConnectionError(
                f"failed to open scope at {self.resource}: {exc}"
            ) from exc

        # VISA timeout is in milliseconds; long acquisitions need a generous one.
        inst.timeout = self.timeout * 1000.0
        inst.chunk_size = self.chunk_size
        inst.read_termination = "\n"
        inst.write_termination = "\n"
        self._inst = inst

        try:
            self._write("*CLS")
            self._write("FORMat:DATA REAL,32")    # 32-bit float waveform samples
            self._write("FORMat:BORDer LSBFirst")  # little-endian binary blocks
            # CHANnel:DATA? honours INCXvalues; keep it OFF so we get Y-only and
            # build the time axis from the header (RTO6 manual 27.8.6).
            self._write("EXPort:WAVeform:INCXvalues OFF")
        except ScopeError:
            self._discard()
            raise

    def _close_transport(self) -> None:
        self._discard()

    def _discard(self) -> None:
        for obj in (self._inst, self._rm):
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass
        self._inst = None
        self._rm = None

    def _write(self, command: str) -> None:
        if self._inst is None:
            raise ScopeConnectionError("scope is not open")
        try:
            self._inst.write(command)
        except Exception as exc:
            raise ScopeConnectionError(f"failed sending {command!r}: {exc}") from exc

    def _query(self, command: str) -> str:
        if self._inst is None:
            raise ScopeConnectionError("scope is not open")
        try:
            return self._inst.query(command).strip()
        except Exception as exc:
            if self._is_timeout(exc):
                raise ScopeTimeoutError(f"timed out querying {command!r}") from exc
            raise ScopeConnectionError(f"failed querying {command!r}: {exc}") from exc

    def _query_binary(self, command: str, datatype: str) -> np.ndarray:
        if self._inst is None:
            raise ScopeConnectionError("scope is not open")
        try:
            data = self._inst.query_binary_values(
                command,
                datatype=datatype,
                is_big_endian=False,
                container=np.ndarray,
            )
        except Exception as exc:
            if self._is_timeout(exc):
                raise ScopeTimeoutError(
                    f"timed out reading waveform via {command!r}"
                ) from exc
            raise ScopeConnectionError(
                f"failed reading waveform via {command!r}: {exc}"
            ) from exc
        return np.asarray(data, dtype=float)

    @staticmethod
    def _is_timeout(exc: Exception) -> bool:
        try:
            import pyvisa.errors

            return isinstance(exc, pyvisa.errors.VisaIOError) and (
                exc.error_code == pyvisa.errors.StatusCode.error_timeout
            )
        except Exception:
            return "timeout" in str(exc).lower()

    # --- identification ---------------------------------------------------
    def identify(self) -> str:
        return self.query("*IDN?")

    # --- channel configuration -------------------------------------------
    def configure_channel(
        self,
        channel: int,
        *,
        state: bool = True,
        scale: str | None = None,
        offset: str | None = None,
        coupling: str | None = None,
    ) -> None:
        """Enable a channel and set its vertical parameters (only those given).

        ``scale`` is volts/division, ``offset`` is volts, ``coupling`` is the
        RTO coupling token (e.g. DC, DCLimit, AC).
        """
        ch = int(channel)
        self.write(f"CHANnel{ch}:STATe {'ON' if state else 'OFF'}")
        if scale is not None:
            self.write(f"CHANnel{ch}:SCALe {scale}")          # V/div
        if offset is not None:
            self.write(f"CHANnel{ch}:OFFSet {offset}")        # V
        if coupling is not None:
            self.write(f"CHANnel{ch}:COUPling {coupling}")    # DC | DCLimit | AC

    def set_decimation(self, channel: int, mode: str) -> None:
        """Set the per-channel decimation / waveform type.

        SAMPle keeps raw samples; PDETect (peak detect) keeps the min/max of
        each decimation interval (two values per sample), so reduced records
        still capture true pulse peaks; HRESolution boxcar-averages.
        """
        # RTO6 decimation: CHANnel<n>:TYPE {SAMPle|PDETect|HRESolution|RMS}
        self.write(f"CHANnel{int(channel)}:TYPE {mode}")

    def set_arithmetics(self, channel: int, mode: str) -> None:
        """Set how consecutive acquisitions are combined.

        OFF gives raw single-shot data; AVERage means across-acquisition
        averaging; ENVelope accumulates min/max. Kept explicit so a capture is
        deterministically raw regardless of the scope's prior front-panel state.
        """
        # RTO6: CHANnel<n>:ARIThmetics {OFF|AVERage|ENVelope}
        self.write(f"CHANnel{int(channel)}:ARIThmetics {mode}")

    def set_bandwidth_limit(self, channel: int, limit: str) -> None:
        """Limit the channel's analog bandwidth to cut wideband noise.

        RTO6: CHANnel<n>:BANDwidth {FULL|B800|B200|B20}. B20 (20 MHz) is the
        lowest hardware limit -- a large SNR win for a low-frequency signal.
        """
        self.write(f"CHANnel{int(channel)}:BANDwidth {limit}")

    def set_digital_filter(self, channel: int, cutoff: float | None) -> None:
        """Enable/disable the per-channel digital low-pass filter.

        RTO6: CHANnel<n>:DIGFilter:STATe / :CUToff <Hz>. Passing None turns the
        filter off. For a smooth <=kHz signal a cutoff ~10x the signal bandwidth
        rejects almost all noise.
        """
        ch = int(channel)
        if cutoff is None:
            self.write(f"CHANnel{ch}:DIGFilter:STATe OFF")
        else:
            self.write(f"CHANnel{ch}:DIGFilter:STATe ON")
            self.write(f"CHANnel{ch}:DIGFilter:CUToff {cutoff}")

    # --- trigger ----------------------------------------------------------
    # Source token -> the LEVel<n> index (RTO6: 1..4 channels, 5 = ext input).
    _TRIG_LEVEL_INDEX = {
        "CHANNEL1": 1, "CHAN1": 1, "C1": 1,
        "CHANNEL2": 2, "CHAN2": 2, "C2": 2,
        "CHANNEL3": 3, "CHAN3": 3, "C3": 3,
        "CHANNEL4": 4, "CHAN4": 4, "C4": 4,
        "EXTERNANALOG": 5, "EXT": 5,
    }

    def set_trigger_mode(self, mode: str) -> None:
        """Set only the trigger mode (AUTO free-runs, NORMal waits for an edge).

        Used for the software/immediate read: MODE AUTO with no armed edge lets
        SINGle self-trigger and complete, matching the proven free-run capture.
        """
        self.write(f"TRIGger1:MODE {mode}")

    def set_trigger(
        self,
        *,
        source: str = "CHANnel1",
        level: float | None = None,
        slope: str = "POSitive",
        mode: str = "NORMal",
    ) -> None:
        """Configure a single edge trigger (A-event).

        mode: AUTO | NORMal | FREerun (NORMal waits for a real trigger).
        source: CHANnel1..4 or EXTernanalog. slope: POSitive (rising) etc.
        The level is written to the LEVel index that matches the source.
        """
        self.write(f"TRIGger1:MODE {mode}")
        self.write(f"TRIGger1:SOURce {source}")
        self.write("TRIGger1:TYPE EDGE")
        self.write(f"TRIGger1:EDGE:SLOPe {slope}")
        if level is not None:
            n = self._TRIG_LEVEL_INDEX.get(source.upper(), 1)
            self.write(f"TRIGger1:LEVel{n} {level}")

    # --- acquisition / horizontal ----------------------------------------
    def set_time_range(self, seconds: str | float) -> None:
        """Set the full acquisition time window (TIMebase:RANGe, in seconds)."""
        self.write(f"TIMebase:RANGe {seconds}")

    def set_acquisition_count(self, count: int = 1) -> None:
        """Set how many acquisitions one SINGle runs (ACQuire:COUNt).

        Must be 1 for a single-shot capture: a stale count (left high by a prior
        averaging run) makes SINGle acquire many records, so the OPC-complete bit
        never latches within the poll timeout and the read wrongly times out.
        """
        self.write(f"ACQuire:COUNt {int(count)}")

    def set_record_length(self, points: int | None) -> None:
        """Fix the record length, or let the scope keep resolution constant.

        ACQuire:POINts:AUTO selects *which* quantity stays constant when the
        time range changes: RECLength pins the record length (so resolution
        adapts -- what we want for a fixed point budget over 1 s), RESolution
        hands record length back to the scope.
        """
        if points is None:
            self.write("ACQuire:POINts:AUTO RESolution")
        else:
            self.write("ACQuire:POINts:AUTO RECLength")
            self.write(f"ACQuire:POINts {int(points)}")

    def set_post_trigger_window(self) -> None:
        """Place the trigger at the record's left edge (all data post-trigger).

        REFerence 0 puts the reference point at 0% of the screen and
        HORizontal:POSition 0 makes it coincide with the trigger (the zero
        point), so the acquisition spans [0, TIMebase:RANGe] after the trigger.
        """
        self.write("TIMebase:REFerence 0")
        self.write("TIMebase:HORizontal:POSition 0")

    def sample_rate(self) -> float:
        return float(self.query("ACQuire:SRATe?"))

    def record_length(self) -> int:
        return int(float(self.query("ACQuire:POINts?")))

    # --- single-acquisition control --------------------------------------
    def stop(self) -> None:
        self.write("STOP")

    def single_acquisition(self) -> None:
        """Arm one acquisition and flag operation-complete for polling.

        Uses the classic pollable-OPC handshake: stop any running acquisition
        for a clean start, clear the status registers, then ``SINGle;*OPC`` so
        that *ESR? bit 0 latches when this one acquisition finishes -- this
        stays abortable, unlike a blocking ``*OPC?``.
        """
        self.write("STOP")
        self.write("*CLS")
        self.write("SINGle;*OPC")

    def event_status(self) -> int:
        """Read (and clear) the standard event status register (*ESR?)."""
        return int(float(self.query("*ESR?")))

    def is_acquisition_complete(self) -> bool:
        return bool(self.event_status() & _OPC_BIT)

    # --- waveform download ------------------------------------------------
    def read_waveform_header(self, channel: int) -> tuple[float, float, int, int]:
        """Return (x_start_s, x_stop_s, record_length, values_per_sample).

        RTO6 returns "XStart,XStop,RecordLength,ValuesPerSample" -- values per
        sample is 2 for peak-detect/envelope waveforms, 1 otherwise.
        """
        raw = self.query(f"CHANnel{int(channel)}:DATA:HEADer?")
        parts = [p for p in raw.replace(";", ",").split(",") if p != ""]
        if len(parts) < 4:
            raise ScopeError(f"unexpected waveform header: {raw!r}")
        x_start = float(parts[0])
        x_stop = float(parts[1])
        record_length = int(float(parts[2]))
        values_per_sample = int(float(parts[3]))
        return x_start, x_stop, record_length, values_per_sample

    def read_waveform(self, channel: int, datatype: str = "f") -> np.ndarray:
        """Download the raw Y values of a channel as a float array.

        CHANnel<n>:DATA? is the [:WAVeform1] shorthand; INCXvalues is forced OFF
        at connect so only Y-values come back.
        """
        return self.query_binary(f"CHANnel{int(channel)}:DATA?", datatype)

    # --- automatic measurement (on-scope averaging over the record) -------
    def _setup_amptime_measurement(
        self,
        channel: int,
        main: str,
        *,
        group: int,
        gate_start: float | None = None,
        gate_stop: float | None = None,
    ) -> None:
        """Configure measurement group <group> for ch<channel> <main> (AMPTime).

        <main> is an amplitude/time measurement type (MEAN, STDDev, ...); the
        scope computes it over the waveform window for us, so we read one scalar
        instead of transferring the whole record. An optional absolute time gate
        [gate_start, gate_stop] (seconds after the trigger) restricts it to the
        settled part of the acquisition.
        """
        mg = int(group)
        self.write(f"MEASurement{mg}:ENABle ON")
        self.write(f"MEASurement{mg}:CATegory AMPTime")
        self.write(f"MEASurement{mg}:SOURce C{int(channel)}W1")
        self.write(f"MEASurement{mg}:MAIN {main}")
        if gate_start is not None and gate_stop is not None:
            self.write(f"MEASurement{mg}:GATE:MODE ABS")
            self.write(f"MEASurement{mg}:GATE:ABSolute:STARt {gate_start}")
            self.write(f"MEASurement{mg}:GATE:ABSolute:STOP {gate_stop}")
            self.write(f"MEASurement{mg}:GATE:STATe ON")
        else:
            self.write(f"MEASurement{mg}:GATE:STATe OFF")

    def setup_mean_measurement(
        self,
        channel: int,
        *,
        group: int = 1,
        gate_start: float | None = None,
        gate_stop: float | None = None,
    ) -> None:
        """Configure measurement group <group> to return ch<channel> MEAN.

        The MEAN measurement is the average of the waveform over its window --
        i.e. the scope computes the averaged signal level for us, so we read one
        scalar instead of transferring the whole record. An optional absolute
        time gate [gate_start, gate_stop] (seconds after the trigger) restricts
        the mean to the settled part of the acquisition.
        """
        self._setup_amptime_measurement(
            channel, "MEAN", group=group, gate_start=gate_start, gate_stop=gate_stop
        )

    def setup_stddev_measurement(
        self,
        channel: int,
        *,
        group: int = 2,
        gate_start: float | None = None,
        gate_stop: float | None = None,
    ) -> None:
        """Configure measurement group <group> to return ch<channel> STDDev.

        STDDev is the standard deviation of the waveform samples over the (gated)
        window -- the within-shot spread/noise of the signal, read as one scalar
        alongside the MEAN. Uses the same absolute time gate so it covers the
        identical settled window the mean is averaged over.
        """
        self._setup_amptime_measurement(
            channel, "STDDev", group=group, gate_start=gate_start, gate_stop=gate_stop
        )

    def read_measurement(self, group: int = 1) -> float:
        """Return the current main-measurement result of a group as a float."""
        return float(self.query(f"MEASurement{int(group)}:RESult:ACTual?"))
