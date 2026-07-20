"""Thorlabs TC300B thermal-controller package.

Reusable, GUI-friendly mirror of ``src/drafts/heat_controller.py``: a serial
:class:`TC300Driver` and a high-level :class:`TC300Controller` that runs the
same watchdog-safe staircase ramp/hold and a read-only live monitor loop.
"""

from .controller import (
    HeaterCycle,
    HeaterSample,
    PID_DEFAULTS,
    StaircaseSettings,
    TC300Controller,
    pid_int,
    to_float,
)
from .driver import HeaterConnectionError, HeaterError, TC300Driver

__all__ = [
    "TC300Controller",
    "TC300Driver",
    "HeaterSample",
    "HeaterCycle",
    "StaircaseSettings",
    "PID_DEFAULTS",
    "HeaterError",
    "HeaterConnectionError",
    "to_float",
    "pid_int",
]
