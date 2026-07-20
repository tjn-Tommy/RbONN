from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heater_module.controller import (
    HeaterCycle,
    PID_DEFAULTS,
    StaircaseSettings,
    TC300Controller,
    pid_int,
    to_float,
)


class FakeDriver:
    """Records the commands TC300Controller sends; returns canned replies.

    Emulates a TC300 already at 79.5 C drawing ~6.5 V -- so the staircase lands
    immediately and never rails (no emergency rest, no stepping past target).
    """

    def __init__(self) -> None:
        self.is_connected = False
        self.calls: list[str] = []

    def connect(self) -> None:
        self.is_connected = True

    def disconnect(self) -> None:
        self.is_connected = False

    def identify(self) -> str:
        return "THORLABS TC300B HV 3.20 FV 4.04 (fake)"

    def command(self, cmd: str) -> str:
        self.calls.append(cmd)
        if cmd == "IDN?":
            return self.identify()
        if cmd == "ERR?":
            return "0"
        if cmd.endswith("?"):
            if cmd.startswith("MOD"):
                return "0"                      # already in Heater mode
            if cmd.startswith("VMAX"):
                return "24.0"                   # volts (query returns real V)
            if cmd.startswith("TACT"):
                return "79.5"                   # already at target
            if cmd.startswith("VOLT"):
                return "6.5"                     # linear regime, not railed
            if cmd.startswith("CURR"):
                return "47"
            if cmd.startswith("TSET"):
                return "79.500"
            if cmd.startswith("PID"):
                return "0.50 20.00 2.00 100"
        return ""                                # setters ack empty


class HelperTests(unittest.TestCase):
    def test_pid_int_scales_by_100(self) -> None:
        self.assertEqual(pid_int(0.5), 50)
        self.assertEqual(pid_int(20.0), 2000)
        self.assertEqual(pid_int(2.0), 200)

    def test_to_float_extracts_first_number(self) -> None:
        self.assertAlmostEqual(to_float("79.5"), 79.5)
        self.assertAlmostEqual(to_float("0.50 20.00 2.00 100"), 0.50)
        self.assertIsNone(to_float("Out-Of-Range!"))


class ControllerTests(unittest.TestCase):
    def test_requires_port_or_driver(self) -> None:
        with self.assertRaises(ValueError):
            TC300Controller()

    def test_read_channel_snapshot(self) -> None:
        ctrl = TC300Controller(driver=FakeDriver())
        s = ctrl.read_channel(1)
        self.assertEqual(s.channel, 1)
        self.assertAlmostEqual(s.temp, 79.5)
        self.assertAlmostEqual(s.volt, 6.5)
        self.assertAlmostEqual(s.curr, 47.0)

    def test_hold_sets_mode_pid_target_enable(self) -> None:
        drv = FakeDriver()
        TC300Controller(driver=drv).hold(2, 79.5)
        d = PID_DEFAULTS[2]
        self.assertIn(f"KP2={pid_int(d['kp'])}", drv.calls)
        self.assertIn(f"TI2={pid_int(d['ti'])}", drv.calls)
        self.assertIn(f"TD2={pid_int(d['td'])}", drv.calls)
        self.assertIn("TSET2=79500", drv.calls)
        self.assertIn("EN2=1", drv.calls)

    def test_run_staircase_enables_then_disables_on_exit(self) -> None:
        drv = FakeDriver()
        ctrl = TC300Controller(driver=drv)
        stop = threading.Event()
        cycles: list[HeaterCycle] = []

        def on_sample(cyc: HeaterCycle) -> None:
            cycles.append(cyc)
            stop.set()   # one iteration is enough for the assertions

        st = ctrl.run_staircase(
            [1],
            StaircaseSettings(target=79.5, step=2.0, period=0.0),
            sample_cb=on_sample,
            stop_event=stop,
            disable_on_exit=True,
        )

        self.assertEqual(len(cycles), 1)
        self.assertIn(1, cycles[0].channels)
        self.assertAlmostEqual(cycles[0].channels[1].temp, 79.5)
        # tuned PID applied, enabled during the ramp, disabled cleanly on exit
        self.assertIn("EN1=1", drv.calls)
        self.assertIn("EN1=0", drv.calls)
        self.assertLess(drv.calls.index("EN1=1"), drv.calls.index("EN1=0"))
        self.assertEqual(st[1]["steps"], 0)   # already at target: no stepping

    def test_run_staircase_can_leave_channel_enabled(self) -> None:
        drv = FakeDriver()
        ctrl = TC300Controller(driver=drv)
        stop = threading.Event()

        ctrl.run_staircase(
            [1],
            StaircaseSettings(target=79.5, period=0.0),
            sample_cb=lambda _c: stop.set(),
            stop_event=stop,
            disable_on_exit=False,
        )
        self.assertIn("EN1=1", drv.calls)
        self.assertNotIn("EN1=0", drv.calls)

    def test_monitor_is_read_only(self) -> None:
        drv = FakeDriver()
        ctrl = TC300Controller(driver=drv)
        stop = threading.Event()

        ctrl.monitor(
            [1, 2],
            sample_cb=lambda _c: stop.set(),
            stop_event=stop,
            period=0.0,
        )
        # never touches EN / TSET / PID / MOD setters
        self.assertFalse(any("=" in c and not c.endswith("?") for c in drv.calls))


if __name__ == "__main__":
    unittest.main()
