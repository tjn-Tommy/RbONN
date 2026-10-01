from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from psu_module.controller import (
    KWR103_I_MAX,
    KWR103_V_MAX,
    PMT_CONTROL_I_MAX,
    PMT_CONTROL_V_MAX,
    PSUController,
    to_float,
)
from psu_module.driver import KWR103Driver, PSUConnectionError, PSUError


class FakeKWR103Serial:
    """Byte-level stand-in for the supply, mirroring the dialect probed on COM4.

    Faithful to the real unit in the ways that matter: ``\\n``-terminated
    traffic, queries answer one zero-padded line, **sets answer nothing**, and
    an unrecognised command answers nothing at all (rather than complaining).
    Ohmic load so the current limit can be exercised: at 10 ohm a 1 V setpoint
    draws 100 mA, so an ISET below that puts the supply in CC.
    """

    def __init__(self, port, baud, *, bytesize=8, parity="N", stopbits=1,
                 timeout=0.4, idn="KORAD KWR103 V2.3 SN:000000047439",
                 load_ohms=10.0, ignore_sets=False, lock_output=False,
                 io_delay=0.0) -> None:
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.idn = idn
        self.load_ohms = load_ohms
        self.ignore_sets = ignore_sets      # a set that silently does not take
        self.lock_output = lock_output       # front panel locked / protection
        # Seconds a reply waits on the wire. The real supply takes ~2 ms, which
        # is the window another thread could reset the input buffer in.
        self.io_delay = io_delay
        self.closed = False
        self.written: list[str] = []
        self.levels = {"VSET": 1.0, "ISET": 0.05, "OVP": 0.5, "OCP": 15.5}
        self.out = False
        self._rx = b""

    # ------------------------------------------------------------ pyserial API
    def reset_input_buffer(self) -> None:
        self._rx = b""

    def read_until(self, expected=b"\n", size=None) -> bytes:
        idx = self._rx.find(expected)
        if idx < 0:
            out, self._rx = self._rx, b""      # nothing pending: a real port
            return out                          # would just time out empty
        cut = idx + len(expected)
        out, self._rx = self._rx[:cut], self._rx[cut:]
        return out

    def close(self) -> None:
        self.closed = True

    def write(self, data: bytes) -> int:
        cmd = data.decode("ascii").strip()
        self.written.append(cmd)
        reply = self._respond(cmd)
        if reply is not None:
            self._rx += (reply + "\n").encode("ascii")
        if self.io_delay:
            time.sleep(self.io_delay)
        return len(data)

    # ----------------------------------------------------------------- device
    def _measured(self) -> tuple[float, float]:
        if not self.out:
            return 0.0, 0.0
        vset, iset = self.levels["VSET"], self.levels["ISET"]
        amps = vset / self.load_ohms
        if amps > iset:                         # CC: the limit pulls V down
            return iset * self.load_ohms, iset
        return vset, amps

    def _respond(self, cmd: str) -> str | None:
        """The reply, or None for the silence a set (or a typo) gets."""
        if cmd == "*IDN?":
            return self.idn
        if cmd == "STATUS?":
            # Only ever observed as 0x05 (output off); the bit layout is
            # undocumented, so the fake claims nothing more than was measured.
            return "\x05"
        if cmd == "OUT?":
            return "1" if self.out else "0"
        volts, amps = self._measured()
        if cmd == "VOUT?":
            return f"{volts:06.3f}"
        if cmd == "IOUT?":
            return f"{amps:06.3f}"
        if cmd.endswith("?") and cmd[:-1] in self.levels:
            return f"{self.levels[cmd[:-1]]:06.3f}"
        if cmd.startswith("OUT:"):
            if not self.lock_output:
                self.out = cmd.endswith("1")
            return None
        key, _, value = cmd.partition(":")
        if value and key in self.levels:
            if not self.ignore_sets:
                self.levels[key] = float(value)
            return None
        return None                              # unknown command: silence


def make_psu(**kwargs) -> tuple[PSUController, FakeKWR103Serial]:
    """A connected controller over a fake port, plus the fake for assertions."""
    caps = {k: kwargs.pop(k) for k in ("max_voltage", "max_current")
            if k in kwargs}
    holder: dict[str, FakeKWR103Serial] = {}

    def factory(port, baud, **kw):
        holder["ser"] = FakeKWR103Serial(port, baud, **{**kw, **kwargs})
        return holder["ser"]

    driver = KWR103Driver(port="COM_FAKE", serial_factory=factory,
                          write_delay=0.0)
    psu = PSUController(driver=driver, **caps)
    psu.connect()
    return psu, holder["ser"]


class DriverTests(unittest.TestCase):
    def test_connect_captures_the_identity(self) -> None:
        psu, ser = make_psu()
        self.assertTrue(psu.is_connected)
        self.assertIn("KWR103", psu.driver.idn or "")
        self.assertEqual(ser.written, ["*IDN?"])

    def test_connect_rejects_a_foreign_instrument(self) -> None:
        # A suffixed-dialect sibling on the same PC must be refused, not
        # half-driven: KA3005P ignores every command this module sends.
        with self.assertRaises(PSUConnectionError) as ctx:
            make_psu(idn="KORAD KA3005P V2.0 SN:1")
        self.assertIn("KA3005P", str(ctx.exception))

    def test_connect_reports_a_silent_port(self) -> None:
        with self.assertRaises(PSUConnectionError) as ctx:
            make_psu(idn=None)                  # None -> no reply at all
        self.assertIn("*IDN?", str(ctx.exception))

    def test_query_raises_on_an_unrecognised_command(self) -> None:
        # The firmware answers a typo with silence, so silence must not be
        # mistaken for a value.
        psu, _ = make_psu()
        with self.assertRaises(PSUError) as ctx:
            psu.driver.query("VSET1?")           # the KA/KD suffixed form
        self.assertIn("VSET1?", str(ctx.exception))

    def test_commands_before_connect_raise(self) -> None:
        driver = KWR103Driver(port="COM_FAKE", serial_factory=FakeKWR103Serial)
        with self.assertRaises(PSUError):
            driver.query("VSET?")
        with self.assertRaises(PSUError):
            driver.write("VSET:1.000")

    def test_disconnect_closes_the_port(self) -> None:
        psu, ser = make_psu()
        psu.disconnect()
        self.assertTrue(ser.closed)
        self.assertFalse(psu.is_connected)
        self.assertIsNone(psu.driver.idn)


class SetpointTests(unittest.TestCase):
    def test_set_voltage_uses_the_colon_dialect_and_reads_back(self) -> None:
        psu, ser = make_psu(max_voltage=12.0)
        self.assertAlmostEqual(psu.set_voltage(5.0), 5.0)
        self.assertIn("VSET:5.000", ser.written)
        self.assertAlmostEqual(psu.get_voltage_setpoint(), 5.0)

    def test_set_current_uses_the_colon_dialect_and_reads_back(self) -> None:
        psu, ser = make_psu(max_current=1.0)
        self.assertAlmostEqual(psu.set_current(0.25), 0.25)
        self.assertIn("ISET:0.250", ser.written)
        self.assertAlmostEqual(psu.get_current_limit(), 0.25)

    def test_over_cap_request_refuses_without_writing(self) -> None:
        psu, ser = make_psu(max_voltage=6.0)
        before = list(ser.written)
        with self.assertRaises(ValueError) as ctx:
            psu.set_voltage(24.0)
        self.assertIn("max_voltage", str(ctx.exception))
        self.assertEqual(ser.written, before)    # nothing reached the supply
        self.assertAlmostEqual(psu.get_voltage_setpoint(), 1.0)

    def test_negative_request_refuses(self) -> None:
        psu, _ = make_psu()
        with self.assertRaises(ValueError):
            psu.set_voltage(-1.0)
        with self.assertRaises(ValueError):
            psu.set_current(-0.1)

    def test_a_set_that_does_not_take_raises(self) -> None:
        # The whole point of reading back: this firmware acknowledges nothing,
        # so a refused write is indistinguishable from a successful one until
        # the value is queried.
        psu, _ = make_psu(max_voltage=12.0, ignore_sets=True)
        with self.assertRaises(PSUError) as ctx:
            psu.set_voltage(5.0)
        self.assertIn("did not take", str(ctx.exception))

    def test_caps_never_exceed_the_model_rating(self) -> None:
        psu, _ = make_psu(max_voltage=999.0, max_current=999.0)
        self.assertEqual(psu.max_voltage, KWR103_V_MAX)
        self.assertEqual(psu.max_current, KWR103_I_MAX)

    def test_protection_levels_are_set_and_verified(self) -> None:
        psu, ser = make_psu(max_voltage=6.0)
        self.assertAlmostEqual(psu.set_ovp(7.5), 7.5)   # above the soft cap:
        self.assertIn("OVP:7.500", ser.written)          # a trip level, not an
        self.assertAlmostEqual(psu.set_ocp(2.0), 2.0)    # output request
        self.assertIn("OCP:2.000", ser.written)


class OutputTests(unittest.TestCase):
    def test_enable_and_disable_confirm_the_output_moved(self) -> None:
        psu, ser = make_psu(max_voltage=12.0)
        self.assertFalse(psu.is_output_on())
        psu.enable()
        self.assertTrue(ser.out)
        self.assertTrue(psu.is_output_on())
        psu.disable()
        self.assertFalse(psu.is_output_on())
        self.assertIn("OUT:1", ser.written)
        self.assertIn("OUT:0", ser.written)

    def test_enable_refuses_a_setpoint_left_above_the_cap(self) -> None:
        # The front panel (or another script) can move VSET behind our back;
        # enabling must not hand 55 V to a 6 V load.
        psu, ser = make_psu(max_voltage=6.0)
        ser.levels["VSET"] = 55.0
        with self.assertRaises(ValueError) as ctx:
            psu.enable()
        self.assertIn("voltage setpoint", str(ctx.exception))
        self.assertFalse(ser.out)

    def test_an_output_that_will_not_move_raises(self) -> None:
        psu, _ = make_psu(max_voltage=12.0, lock_output=True)
        with self.assertRaises(PSUError) as ctx:
            psu.enable()
        self.assertIn("stayed off", str(ctx.exception))

    def test_disconnect_leaves_the_output_alone(self) -> None:
        psu, ser = make_psu(max_voltage=12.0)
        psu.enable()
        psu.disconnect()
        self.assertTrue(ser.out)                 # closing a port feeds no load


class ReadbackTests(unittest.TestCase):
    def test_read_reports_the_measured_terminals(self) -> None:
        psu, _ = make_psu(max_voltage=12.0, max_current=2.0)
        psu.set_voltage(5.0)
        psu.set_current(1.0)                     # 5 V / 10 ohm = 0.5 A, so CV
        self.assertEqual(psu.read().volts, 0.0)  # output still off
        psu.enable()
        reading = psu.read()
        self.assertAlmostEqual(reading.volts, 5.0)
        self.assertAlmostEqual(reading.amps, 0.5)
        self.assertTrue(reading.output_on)
        self.assertAlmostEqual(reading.watts, 2.5)

    def test_state_snapshot_carries_settings_and_status(self) -> None:
        psu, _ = make_psu(max_voltage=12.0, max_current=2.0)
        psu.set_voltage(5.0)
        psu.enable()
        state = psu.read_state()
        self.assertAlmostEqual(state.vset, 5.0)
        self.assertAlmostEqual(state.ovp, 0.5)
        self.assertAlmostEqual(state.ocp, 15.5)
        self.assertTrue(state.output_on)
        self.assertEqual(state.status_raw, 0x05)  # as measured on the real unit

    def test_current_limited_flag_tracks_cc(self) -> None:
        psu, _ = make_psu(max_voltage=12.0, max_current=2.0)
        psu.set_voltage(5.0)
        psu.set_current(1.0)                     # limit above the 0.5 A draw
        psu.enable()
        self.assertFalse(psu.read_state().current_limited)
        psu.set_current(0.2)                     # now the limit is binding
        state = psu.read_state()
        self.assertTrue(state.current_limited)
        self.assertAlmostEqual(state.amps, 0.2)
        self.assertAlmostEqual(state.volts, 2.0)  # folded back by the limit

    def test_to_float_rejects_junk(self) -> None:
        self.assertAlmostEqual(to_float("01.000"), 1.0)
        with self.assertRaises(PSUError):
            to_float("\x05")


class MonitorTests(unittest.TestCase):
    def test_monitor_streams_until_stopped_and_writes_nothing(self) -> None:
        psu, ser = make_psu(max_voltage=12.0, max_current=2.0)
        psu.set_voltage(3.0)
        psu.set_current(1.0)                     # CV: 3 V / 10 ohm = 0.3 A
        psu.enable()
        stop = threading.Event()
        seen: list[float] = []

        def sample_cb(reading) -> None:
            seen.append(reading.volts)
            if len(seen) == 3:
                stop.set()

        ser.written.clear()
        psu.monitor(sample_cb=sample_cb, stop_event=stop, period=0.0)
        self.assertEqual(len(seen), 3)
        self.assertTrue(all(abs(v - 3.0) < 1e-9 for v in seen))
        self.assertFalse([c for c in ser.written if not c.endswith("?")])

    def test_monitor_returns_immediately_if_already_stopped(self) -> None:
        psu, ser = make_psu()
        stop = threading.Event()
        stop.set()
        ser.written.clear()
        psu.monitor(sample_cb=lambda r: self.fail("should not sample"),
                    stop_event=stop, period=0.0)
        self.assertEqual(ser.written, [])


class PMTControlTests(unittest.TestCase):
    """The bench configuration: CV into the PMT gain-control input."""

    def psu(self):
        return make_psu(max_voltage=PMT_CONTROL_V_MAX,
                        max_current=PMT_CONTROL_I_MAX,
                        load_ohms=1e6)      # control input draws microamps

    def test_the_ceiling_is_the_modules_spec_not_the_supply_rating(self) -> None:
        psu, _ = self.psu()
        self.assertAlmostEqual(psu.set_voltage(1.1), 1.1)
        with self.assertRaises(ValueError):
            psu.set_voltage(1.2)

    def test_a_high_impedance_input_never_leaves_cv(self) -> None:
        psu, _ = self.psu()
        psu.set_voltage(0.9)
        psu.set_current(PMT_CONTROL_I_MAX)
        psu.enable()
        state = psu.read_state()
        self.assertAlmostEqual(state.volts, 0.9)
        self.assertFalse(state.current_limited)   # CC here would mean a fault


class TransactionTests(unittest.TestCase):
    def test_monitor_full_sends_the_whole_state(self) -> None:
        psu, _ = make_psu(max_voltage=1.1, max_current=0.05)
        psu.set_voltage(0.8)
        stop = threading.Event()
        seen: list[object] = []

        def sample_cb(sample) -> None:
            seen.append(sample)
            stop.set()

        psu.monitor(sample_cb=sample_cb, stop_event=stop, period=0.0, full=True)
        self.assertAlmostEqual(seen[0].vset, 0.8)       # settings, not just
        self.assertAlmostEqual(seen[0].ocp, 15.5)        # the measurement
        self.assertFalse(seen[0].current_limited)

    def test_a_transaction_is_reentrant(self) -> None:
        psu, _ = make_psu()
        with psu.driver.transaction():
            self.assertIsNotNone(psu.read())             # would deadlock on a
            self.assertFalse(psu.is_output_on())         # plain Lock

    def test_polling_while_setting_does_not_interleave(self) -> None:
        # The GUI polls a live readout on a worker thread while the user applies
        # a setpoint from the GUI thread. One command per reply and no framing on
        # the wire means an unlocked port loses replies to the other thread.
        psu, _ = make_psu(max_voltage=1.1, max_current=0.05, io_delay=0.001)
        stop = threading.Event()
        errors: list[Exception] = []

        def poll() -> None:
            try:
                while not stop.is_set():
                    psu.read()
            except Exception as exc:            # a lost reply surfaces here
                errors.append(exc)

        poller = threading.Thread(target=poll)
        poller.start()
        try:
            for i in range(50):
                volts = 0.500 + 0.001 * (i % 10)
                self.assertAlmostEqual(psu.set_voltage(volts), volts, places=3)
        finally:
            stop.set()
            poller.join(timeout=2.0)
        self.assertEqual(errors, [])
        self.assertFalse(poller.is_alive())


class ConstructionTests(unittest.TestCase):
    def test_port_or_driver_is_required(self) -> None:
        with self.assertRaises(ValueError):
            PSUController()

    def test_port_builds_a_real_driver(self) -> None:
        psu = PSUController(port="COM4")
        self.assertIsInstance(psu.driver, KWR103Driver)
        self.assertEqual(psu.driver.port, "COM4")
        self.assertFalse(psu.is_connected)        # constructing opens nothing


if __name__ == "__main__":
    unittest.main()
