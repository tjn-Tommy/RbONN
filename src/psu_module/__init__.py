"""Korad KWR103 programmable DC power-supply package.

A serial :class:`KWR103Driver` plus a :class:`PSUController` that verifies every
setpoint by reading it back (this firmware acknowledges nothing) and refuses
writes above a configurable soft limit, in the same shape as the DAQ and heater
modules.
"""

from .controller import (
    KWR103_I_MAX,
    KWR103_V_MAX,
    PMT_CONTROL_I_MAX,
    PMT_CONTROL_V_MAX,
    PSUController,
    PSUReading,
    PSUState,
    SET_TOL_A,
    SET_TOL_V,
    to_float,
)
from .driver import KWR103Driver, PSUConnectionError, PSUError

__all__ = [
    "PSUController",
    "KWR103Driver",
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
