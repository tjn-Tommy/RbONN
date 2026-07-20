"""Low-level serial transport for the Thorlabs TC300B thermal controller.

Faithful to ``src/drafts/heat_controller.py``'s ``send()`` / ``probe_terminator()``:
one command per call -- flush the input buffer, write ``"<cmd><terminator>"``,
wait a beat, read the reply back and strip the echoed command / ``>`` prompt /
line endings. The reply terminator (``\\r`` on this firmware) is auto-probed at
:meth:`connect`.

Split out from the controller so it can be swapped for a fake in tests, exactly
like ``NIDAQDriver`` under ``DAQController``.
"""
from __future__ import annotations

import time
from typing import Any, Callable

try:
    import serial  # pyserial
except ImportError:  # keep the module importable without the dependency
    serial = None  # type: ignore[assignment]

BAUD = 115200
# Response terminators to try at connect, in the order the TC300 is most likely
# to use (this firmware answers to a bare CR).
_TERMINATORS = ("\r", "\r\n", "\n")


class HeaterError(Exception):
    """Any TC300 command/transport failure."""


class HeaterConnectionError(HeaterError):
    """The serial port could not be opened, or the TC300 did not answer IDN?."""


class TC300Driver:
    """Serial transport for the TC300B (COM port, 115200 8-N-1).

    ``serial_factory`` lets tests inject a fake ``Serial`` without real hardware;
    it defaults to ``serial.Serial``. Everything above the raw byte level lives
    in :class:`heater_module.controller.TC300Controller`.
    """

    def __init__(
        self,
        port: str = "COM3",
        baud: int = BAUD,
        *,
        read_delay: float = 0.08,
        timeout: float = 0.2,
        serial_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.port = port
        self.baud = baud
        self.read_delay = read_delay
        self.timeout = timeout
        self._serial_factory = serial_factory
        self._ser: Any | None = None
        self._term: str | None = None

    @property
    def is_connected(self) -> bool:
        return self._ser is not None

    @property
    def terminator(self) -> str | None:
        return self._term

    def connect(self) -> None:
        if self._ser is not None:
            return
        factory = self._serial_factory
        if factory is None:
            if serial is None:
                raise HeaterConnectionError(
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
            raise HeaterConnectionError(
                f"Could not open {self.port}: {exc}. "
                "Close the Thorlabs GUI / other scripts holding the port first."
            ) from exc

        self._term = self._probe_terminator()
        if self._term is None:
            self.disconnect()
            raise HeaterConnectionError(
                f"No response to IDN? on {self.port} -- is this the TC300 port?"
            )

    def disconnect(self) -> None:
        ser, self._ser = self._ser, None
        self._term = None
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass

    def _probe_terminator(self) -> str | None:
        for term in _TERMINATORS:
            resp = self._raw_command("IDN?", term)
            if "TC300" in resp.upper() or "THORLABS" in resp.upper():
                return term
        return None

    def _raw_command(self, cmd: str, term: str) -> str:
        ser = self._ser
        if ser is None:
            raise HeaterError("TC300 not connected")
        ser.reset_input_buffer()
        ser.write((cmd + term).encode("ascii"))
        time.sleep(self.read_delay)
        raw = ser.read(256).decode("ascii", errors="replace")
        return (
            raw.replace(cmd, "").replace(">", "")
            .replace("\r", " ").replace("\n", " ").strip()
        )

    def command(self, cmd: str) -> str:
        """Send one command and return the stripped reply string."""
        if self._ser is None:
            raise HeaterError("TC300 not connected")
        return self._raw_command(cmd, self._term or "\r")

    def identify(self) -> str:
        return self.command("IDN?")

    def __enter__(self) -> "TC300Driver":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disconnect()


__all__ = ["TC300Driver", "HeaterError", "HeaterConnectionError", "BAUD"]
