"""Low-level serial transport for the Korad KWR103 programmable DC supply.

Same shape as :class:`heater_module.driver.TC300Driver`: one command per call
over a pyserial port, with ``serial_factory`` injection so everything above it
can be tested without hardware.

Protocol facts probed on the unit at this bench (``KORAD KWR103 V2.3
SN:000000047439``, COM4):

* 115200 8-N-1; every command ends in a bare ``"\n"``, as does every reply.
* The KWR dialect takes **no channel suffix**: ``VSET?`` answers, ``VSET1?`` is
  silent. (The KA/KD-series drivers found online use the suffixed form -- they
  do not work here.)
* Queries reply with one line; **sets reply with nothing at all**, so a set can
  only be confirmed by reading the value back -- see
  :meth:`psu_module.controller.PSUController.set_voltage`. The colon set forms
  ``VSET:``/``ISET:``/``OVP:`` are confirmed against the unit (0.900 V, 0.040 A
  and 1.150 V all read back exactly); ``OUT:1`` is the same dialect but has not
  been exercised, since proving it means energising the load.
* Replies are fixed-width, zero-padded ASCII (``"01.000"``), so ``float()``
  them; ``STATUS?`` answers a single raw byte instead.
* No inter-command gap is needed: 20 back-to-back queries answered 20/20 in
  40 ms (~2 ms each), so even a 10 Hz monitor loop is nowhere near the limit.
* There is no error query (``ERR?`` is silent) and an unrecognised command is
  simply ignored, which is why :meth:`connect` verifies ``*IDN?`` -- on the
  wrong port you get silence, not a complaint.

One command per reply and no framing on the wire means concurrent callers would
interleave into nonsense, so every exchange holds a re-entrant port lock. A
caller that must keep several exchanges together -- a set plus its readback,
say -- takes the same lock through :meth:`KWR103Driver.transaction`. That is
what lets the GUI poll a live readout on a worker thread while the user applies
a new setpoint from the GUI thread.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator

try:
    import serial  # pyserial
except ImportError:  # keep the module importable without the dependency
    serial = None  # type: ignore[assignment]

BAUD = 115200
TERM = "\n"
# *IDN? on this family answers "KORAD KWR103 V2.3 SN:...". Match the model so a
# suffixed-dialect sibling (KA3005P etc.) on the same PC is rejected rather than
# half-driven.
IDN_TOKEN = "KWR"


class PSUError(Exception):
    """Any KWR103 command/transport failure."""


class PSUConnectionError(PSUError):
    """The serial port could not be opened, or *IDN? did not name a KWR unit."""


class KWR103Driver:
    """Serial transport for the Korad KWR103 (COM port, 115200 8-N-1).

    ``serial_factory`` lets tests inject a fake ``Serial``; it defaults to
    ``serial.Serial``. Everything above the raw line level lives in
    :class:`psu_module.controller.PSUController`.
    """

    def __init__(
        self,
        port: str = "COM4",
        baud: int = BAUD,
        *,
        timeout: float = 0.4,
        write_delay: float = 0.02,
        serial_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.port = port
        self.baud = baud
        self.timeout = timeout
        # A set answers nothing, so there is no reply to pace against; this is
        # the one place a small fixed wait earns its keep.
        self.write_delay = write_delay
        self._serial_factory = serial_factory
        self._ser: Any | None = None
        self._idn: str | None = None
        # Re-entrant so a transaction can wrap query/write without deadlocking.
        self._lock = threading.RLock()

    @property
    def is_connected(self) -> bool:
        return self._ser is not None

    @property
    def idn(self) -> str | None:
        """Identity string captured at :meth:`connect` (None while closed)."""
        return self._idn

    # ------------------------------------------------------------------ conn
    def connect(self) -> None:
        if self._ser is not None:
            return
        factory = self._serial_factory
        if factory is None:
            if serial is None:
                raise PSUConnectionError(
                    "pyserial not installed. Run:  pip install pyserial"
                )
            factory = serial.Serial
        try:
            self._ser = factory(
                self.port, self.baud, bytesize=8, parity="N", stopbits=1,
                timeout=self.timeout,
            )
        except Exception as exc:  # serial.SerialException and friends
            self._ser = None
            raise PSUConnectionError(
                f"Could not open {self.port}: {exc}. "
                "Close the Korad GUI / other scripts holding the port first."
            ) from exc

        try:
            idn = self.query("*IDN?")
        except PSUError as exc:      # silence: wrong port, or wrong baud
            self.disconnect()
            raise PSUConnectionError(
                f"No response to *IDN? on {self.port} -- is this the KWR103 port?"
            ) from exc
        if IDN_TOKEN not in idn.upper():
            self.disconnect()
            raise PSUConnectionError(
                f"{self.port} answered {idn!r}, which is not a Korad KWR unit."
            )
        self._idn = idn

    def disconnect(self) -> None:
        ser, self._ser = self._ser, None
        self._idn = None
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass

    # ------------------------------------------------------------------- I/O
    def _require(self) -> Any:
        if self._ser is None:
            raise PSUError("KWR103 not connected")
        return self._ser

    @contextmanager
    def transaction(self) -> Iterator["KWR103Driver"]:
        """Hold the port across several exchanges (e.g. a set and its readback).

        Without it a poll on another thread can land between the two and the
        readback verifies the wrong thing.
        """
        with self._lock:
            yield self

    def query(self, cmd: str) -> str:
        """Send a query and return its one-line reply (raises on silence)."""
        with self._lock:
            ser = self._require()
            ser.reset_input_buffer()
            ser.write((cmd + TERM).encode("ascii"))
            raw = ser.read_until(TERM.encode("ascii"))
        reply = raw.decode("ascii", errors="replace").strip("\r\n")
        if not reply:
            raise PSUError(
                f"No reply to {cmd!r} on {self.port}. The firmware ignores "
                "commands it does not recognise, so check the spelling "
                "(KWR takes no channel suffix: VSET?, not VSET1?)."
            )
        return reply

    def write(self, cmd: str) -> None:
        """Send a set. The firmware acknowledges nothing -- read back to verify."""
        with self._lock:
            ser = self._require()
            ser.write((cmd + TERM).encode("ascii"))
            if self.write_delay:
                time.sleep(self.write_delay)

    def identify(self) -> str:
        return self.query("*IDN?")

    def __enter__(self) -> "KWR103Driver":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disconnect()


__all__ = ["KWR103Driver", "PSUError", "PSUConnectionError", "BAUD", "TERM",
           "IDN_TOKEN"]
