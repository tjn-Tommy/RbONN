"""High-level Korad KWR103 orchestration for scripts and the GUI.

Same shape as :class:`daq_module.controller.DAQController` /
:class:`heater_module.controller.TC300Controller`: connect, set, read, plus a
read-only :meth:`PSUController.monitor` loop a background thread can drive.

Two facts about this supply shape the whole class:

* **Sets are never acknowledged** (see :mod:`psu_module.driver`), so every write
  here is read back and compared. A dropped byte would otherwise leave the
  supply sitting at a silently different voltage, and the caller gets the value
  the hardware actually took rather than the one it asked for.
* **It has real rails behind it.** ``max_voltage`` / ``max_current`` are soft
  caps checked before every write, and :meth:`PSUController.enable` re-checks
  the live setpoints against them, so a setpoint left on the front panel cannot
  be handed to a load by a script that believes it set 5 V. Over-cap requests
  raise; nothing is silently clamped.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .driver import KWR103Driver, PSUConnectionError, PSUError

# Model rating (KWR103: 60 V / 15 A / 300 W). This is the ceiling of last
# resort -- the supply clamps writes to it by itself, so a cap left at the
# default protects nothing. Pass max_voltage/max_current for whatever is
# actually wired to the terminals.
KWR103_V_MAX = 60.0
KWR103_I_MAX = 15.0

# What this supply is actually for on this bench: the PMT module gain-control
# input, specified to 1.1 V. Gain rises steeply with control voltage, so the
# ceiling is a hardware limit, not a preference -- the GUI and any script should
# pass these rather than the model rating.
#
# The control input is high impedance (microamps), so the current limit is a
# fault detector rather than an operating parameter: in normal use the supply
# never leaves CV, and CC means something is shorted or mis-wired.
PMT_CONTROL_V_MAX = 1.1
PMT_CONTROL_I_MAX = 0.05

# A readback must land this close to the request or the write counts as failed.
# Loose enough for the supply quantisation (setpoints report to 1 mV / 1 mA),
# tight enough that a mis-spelled command -- which this firmware answers with
# silence rather than an error -- surfaces as a raise.
SET_TOL_V = 0.02
SET_TOL_A = 0.02


@dataclass(frozen=True)
class PSUReading:
    """One measured snapshot: what the terminals are actually doing."""

    volts: float                                   # VOUT?, measured
    amps: float                                    # IOUT?, measured
    output_on: bool                                # OUT?
    t: float = field(default_factory=time.time)    # wall clock, for logging

    @property
    def watts(self) -> float:
        return self.volts * self.amps


@dataclass(frozen=True)
class PSUState:
    """Everything the supply will tell us in one pass: settings + measurement."""

    vset: float          # VSET?, commanded voltage
    iset: float          # ISET?, commanded current limit
    volts: float         # VOUT?, measured
    amps: float          # IOUT?, measured
    ovp: float           # OVP?, over-voltage trip level
    ocp: float           # OCP?, over-current trip level
    output_on: bool      # OUT?
    status_raw: int      # STATUS? byte -- see read_status_byte
    t: float = field(default_factory=time.time)

    @property
    def watts(self) -> float:
        return self.volts * self.amps

    @property
    def current_limited(self) -> bool:
        """True when the load is pulling the current limit (CC rather than CV).

        Derived from IOUT vs ISET, not from the STATUS byte, whose bit layout is
        undocumented for this firmware (see :meth:`read_status_byte`).
        """
        return self.output_on and self.amps >= self.iset - SET_TOL_A


def to_float(s: Any) -> float:
    """Parse a KWR reply like ``'01.000'``; raises PSUError on anything else."""
    try:
        return float(str(s).strip())
    except ValueError as exc:
        raise PSUError(f"Expected a number from the KWR103, got {s!r}") from exc


class PSUController:
    """Connect / set / read wrapper around :class:`KWR103Driver` (injectable)."""

    def __init__(
        self,
        port: str | None = None,
        *,
        driver: Any | None = None,
        max_voltage: float = KWR103_V_MAX,
        max_current: float = KWR103_I_MAX,
    ) -> None:
        if driver is not None:
            self.driver = driver
        elif port is not None:
            self.driver = KWR103Driver(port=port)
        else:
            raise ValueError("either port or an explicit driver is required")
        self.max_voltage = min(max_voltage, KWR103_V_MAX)
        self.max_current = min(max_current, KWR103_I_MAX)

    # ------------------------------------------------------------------ conn
    @property
    def is_connected(self) -> bool:
        return self.driver.is_connected

    def connect(self) -> None:
        self.driver.connect()

    def disconnect(self) -> None:
        """Close the port, leaving the output exactly as it is.

        Closing a serial port must not change what a load is being fed; call
        :meth:`disable` first if that is what you mean.
        """
        self.driver.disconnect()

    def identify(self) -> str:
        return self.driver.identify()

    # ------------------------------------------------------------- primitives
    def _query_float(self, cmd: str) -> float:
        return to_float(self.driver.query(cmd))

    def _check_limit(self, value: float, limit: float, unit: str,
                     what: str, knob: str) -> None:
        if value < 0.0:
            raise ValueError(f"{what} cannot be negative (got {value:g} {unit})")
        if value > limit:
            raise ValueError(
                f"{what} {value:g} {unit} exceeds the configured limit "
                f"{limit:g} {unit}. Raise {knob} deliberately if the load can "
                f"really take it."
            )

    def _set_verified(self, cmd: str, query: str, value: float, tol: float,
                      unit: str) -> float:
        """Write a setpoint, read it back, return what the hardware took."""
        with self.driver.transaction():   # nothing may land between the two
            self.driver.write(f"{cmd}:{value:.3f}")
            actual = self._query_float(query)
        if abs(actual - value) > tol:
            raise PSUError(
                f"{cmd} did not take: asked {value:.3f} {unit}, supply reports "
                f"{actual:.3f} {unit}. This firmware never acknowledges a set, "
                f"so a rejected or mis-spelled command looks exactly like this."
            )
        return actual

    # -------------------------------------------------------------- setpoints
    def set_voltage(self, volts: float) -> float:
        """Set the output voltage; returns the value the supply reports back."""
        self._check_limit(volts, self.max_voltage, "V", "voltage", "max_voltage")
        return self._set_verified("VSET", "VSET?", volts, SET_TOL_V, "V")

    def set_current(self, amps: float) -> float:
        """Set the current limit; returns the value the supply reports back."""
        self._check_limit(amps, self.max_current, "A", "current", "max_current")
        return self._set_verified("ISET", "ISET?", amps, SET_TOL_A, "A")

    def get_voltage_setpoint(self) -> float:
        return self._query_float("VSET?")

    def get_current_limit(self) -> float:
        return self._query_float("ISET?")

    def set_ovp(self, volts: float) -> float:
        """Over-voltage trip level (bounded by the model rating, not the soft cap)."""
        self._check_limit(volts, KWR103_V_MAX, "V", "OVP level", "the model rating")
        return self._set_verified("OVP", "OVP?", volts, SET_TOL_V, "V")

    def set_ocp(self, amps: float) -> float:
        """Over-current trip level (bounded by the model rating, not the soft cap)."""
        self._check_limit(amps, KWR103_I_MAX, "A", "OCP level", "the model rating")
        return self._set_verified("OCP", "OCP?", amps, SET_TOL_A, "A")

    # ----------------------------------------------------------------- output
    def is_output_on(self) -> bool:
        """Output state straight from ``OUT?`` -- no status-byte guessing."""
        return self.driver.query("OUT?").strip() not in ("0", "OFF", "off")

    def enable(self) -> None:
        """Turn the output on, but only if the live setpoints are within the caps.

        The setpoints are re-read first because the front panel and other
        scripts can move them behind our back; handing a stale 55 V to a 6 V
        load is exactly the accident the caps exist to prevent.
        """
        with self.driver.transaction():
            vset, iset = self.get_voltage_setpoint(), self.get_current_limit()
            self._check_limit(vset, self.max_voltage, "V", "voltage setpoint",
                              "max_voltage")
            self._check_limit(iset, self.max_current, "A", "current limit",
                              "max_current")
            self.set_output(True)

    def disable(self) -> None:
        self.set_output(False)

    def set_output(self, on: bool) -> None:
        """Drive the output and confirm it moved (sets are unacknowledged)."""
        with self.driver.transaction():
            self.driver.write(f"OUT:{1 if on else 0}")
            actual = self.is_output_on()
        if actual != on:
            raise PSUError(
                f"Output stayed {'on' if actual else 'off'} after asking for "
                f"{'on' if on else 'off'}. Check that the front panel is not "
                f"locked and that no protection has tripped."
            )

    # --------------------------------------------------------------- readback
    def read(self) -> PSUReading:
        """Fast measured snapshot (3 queries, ~6 ms) for a monitor loop."""
        with self.driver.transaction():   # one coherent snapshot, not three
            return PSUReading(
                volts=self._query_float("VOUT?"),
                amps=self._query_float("IOUT?"),
                output_on=self.is_output_on(),
            )

    def read_state(self) -> PSUState:
        """Full snapshot: setpoints, protections, measurement and status byte."""
        with self.driver.transaction():
            return PSUState(
                vset=self._query_float("VSET?"),
                iset=self._query_float("ISET?"),
                volts=self._query_float("VOUT?"),
                amps=self._query_float("IOUT?"),
                ovp=self._query_float("OVP?"),
                ocp=self._query_float("OCP?"),
                output_on=self.is_output_on(),
                status_raw=self.read_status_byte(),
            )

    def read_status_byte(self) -> int:
        """``STATUS?`` as an int, carried through raw on purpose.

        This firmware answers a single non-printable byte and its bit layout is
        undocumented; the one datum in hand is 0x05 with the output off, which
        already rules out the KA-series layout. Output state comes from ``OUT?``
        and CV/CC from IOUT vs ISET (:attr:`PSUState.current_limited`), so
        nothing here depends on decoding it.
        """
        reply = self.driver.query("STATUS?")
        return ord(reply[0]) if reply else 0

    # ------------------------------------------------------------------ loops
    def monitor(
        self,
        *,
        sample_cb: Callable[[Any], None],
        stop_event: threading.Event,
        period: float = 0.5,
        full: bool = False,
    ) -> None:
        """Read-only live loop until ``stop_event``.

        Issues no writes at all, so it is safe to watch a supply that a front
        panel or another script is driving. ``full`` sends a whole
        :class:`PSUState` per pass instead of a :class:`PSUReading` -- eight
        queries rather than three, which buys the live setpoints, the protection
        levels and the CV/CC verdict for a display to show.
        """
        while not stop_event.is_set():
            sample_cb(self.read_state() if full else self.read())
            stop_event.wait(period)

    def __enter__(self) -> "PSUController":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disconnect()


__all__ = [
    "PSUController",
    "PSUReading",
    "PSUState",
    "PSUError",
    "PSUConnectionError",
    "KWR103_V_MAX",
    "KWR103_I_MAX",
    "PMT_CONTROL_V_MAX",
    "PMT_CONTROL_I_MAX",
    "SET_TOL_V",
    "SET_TOL_A",
    "to_float",
]
