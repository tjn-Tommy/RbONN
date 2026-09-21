"""The PMT thermistor (Keithley 2400) as a per-row column for any calibration step.

Split out of ``steps/system_accuracy.py`` so steps 6, 7 and the accuracy run all
stamp their rows from one implementation -- a temperature logged beside a step-6
eta and one logged beside an accuracy read must be the same measurement, or
comparing them across steps is meaningless.

The instrument configuration is *not* defined here: it is imported from
``drafts/daq_pmt_monitor.py``, which owns it, so this can never drift from the
long-term monitor's setup.  The resistance is turned into degrees by
``drafts/th10k.py`` (the TH10K spec-sheet curve -- wrong for any other
thermistor).

One GPIB bus, one 2400: nothing that reads the thermistor may run while
``daq_pmt_monitor.py`` is running.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]

PMT_PERIOD_S = 60.0             # resample the 2400 at most this often

__all__ = ["PMT_PERIOD_S", "PmtTemperature", "open_pmt"]


class PmtTemperature:
    """The PMT thermistor on the Keithley 2400, as a per-read row hook.

    The instrument configuration is *not* redefined here: it is imported from
    ``drafts/daq_pmt_monitor.py``, which owns it, so the temperature logged
    beside an accuracy run and the one logged by the long-term monitor can
    never come from two different setups.  The read itself is
    ``monitor_lib.read_ohm_2400``: a fresh GPIB session per reading with the
    output ON only while it measures, so the bead is not self-heated in
    between.

    Readings are cached for ``period_s``.  A step-6 sweep is ~350 reads but the
    PMT's thermal time constant is minutes, so sampling once a minute costs
    ~20 GPIB reads instead of 350 and adds seconds, not minutes, to a sweep.
    Each sample fills the row it lands on and every row until the next one.

    Two ways to use it: :meth:`sample` returns the cells (for callers that build
    rows as tuples, like steps 6 and 7), and ``__call__(row)`` stamps a row dict
    in place (for callers with a dict row hook, like the accuracy run).

    Never raises: the 2400 failing blanks the two cells and prints why, once
    per failure streak, and the run goes on without it -- losing the
    temperature must not lose the night.
    """

    COLUMNS = ("pmt_ohm_Ohm", "pmt_c")

    def __init__(self, resource: str | None = None, period_s: float = PMT_PERIOD_S):
        drafts = REPO_ROOT / "src" / "drafts"
        if str(drafts) not in sys.path:
            sys.path.insert(0, str(drafts))
        import daq_pmt_monitor as PMT
        from monitor_lib import read_ohm_2400
        from th10k import th10k_celsius_one

        self._read_ohm = read_ohm_2400
        self._celsius = th10k_celsius_one
        self.resource = resource or PMT.K2400_RESOURCE
        if self.resource is None:
            raise ValueError(
                "no 2400 resource: daq_pmt_monitor.K2400_RESOURCE is None and "
                "--pmt was given no address")
        self.kwargs = dict(four_wire=PMT.K2400_FOUR_WIRE, nplc=PMT.K2400_NPLC,
                           current=PMT.K2400_CURRENT,
                           compliance=PMT.K2400_COMPLIANCE,
                           timeout_s=PMT.K2400_TIMEOUT_S)
        self.period_s = float(period_s)
        self._cells: dict[str, str] = {c: "" for c in self.COLUMNS}
        self._t_last = -np.inf
        self._failing = False
        self.n_reads = 0

    def read(self) -> dict[str, str]:
        """Read the 2400 now, cache the result and return the CSV cells."""
        self._t_last = time.monotonic()
        try:
            ohm = float(self._read_ohm(self.resource, **self.kwargs))
            if not np.isfinite(ohm):
                raise ValueError("over-range (thermistor open?)")
            celsius = self._celsius(ohm)
            if not np.isfinite(celsius):
                raise ValueError(f"{ohm:.6g} Ohm is off the TH10K curve")
        except Exception as exc:          # noqa: BLE001 - see the class docstring
            if not self._failing:
                self._failing = True
                print(f"\nPMT temperature unavailable ({type(exc).__name__}: {exc}); "
                      f"leaving its columns blank until it comes back")
            self._cells = {c: "" for c in self.COLUMNS}
            return dict(self._cells)
        if self._failing:
            self._failing = False
            print(f"\nPMT temperature back: {celsius:.3f} C")
        self.n_reads += 1
        self._cells = {"pmt_ohm_Ohm": f"{ohm:.6g}", "pmt_c": f"{celsius:.4f}"}
        return dict(self._cells)

    def sample(self) -> dict[str, str]:
        """The cached cells, re-reading the 2400 once ``period_s`` has passed."""
        if time.monotonic() - self._t_last >= self.period_s:
            return self.read()
        return dict(self._cells)

    def cells(self) -> tuple[str, str]:
        """:meth:`sample` as a ``(ohm, celsius)`` pair, in ``COLUMNS`` order."""
        s = self.sample()
        return s["pmt_ohm_Ohm"], s["pmt_c"]

    def __call__(self, row: dict) -> None:
        """Row hook: stamp this read with the current temperature."""
        row.update(self.sample())


def open_pmt(resource: str | None = None, period_s: float = PMT_PERIOD_S, *,
             enabled: bool = True, log=print) -> PmtTemperature | None:
    """A :class:`PmtTemperature`, or ``None`` if it is off or cannot be opened.

    Steps 6 and 7 log the thermistor by DEFAULT, so a missing 2400 -- no VISA,
    no GPIB adapter, ``K2400_RESOURCE = None`` -- must cost two blank columns
    and not the calibration.  This is where that is decided; once the object
    exists it never raises again (see the class docstring).
    """
    if not enabled:
        return None
    try:
        return PmtTemperature(resource, period_s=period_s)
    except Exception as exc:          # noqa: BLE001 - a missing 2400 is not fatal
        if log is not None:
            log(f"PMT temperature off ({type(exc).__name__}: {exc}); "
                f"its columns will be blank -- pass --no-pmt to silence this")
        return None
