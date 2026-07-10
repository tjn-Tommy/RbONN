from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scope_module.controller import (
    MonitorSample,
    MonitorSettings,
    ScopeController,
    ScopeSettings,
    Waveform,
)
from scope_module.driver import RTO6_Driver, ScopeError


class FakeDriver:
    """Records the calls ScopeController makes, without any I/O."""

    def __init__(self) -> None:
        self.is_connected = False
        self.calls: list[str] = []
        self._polls = 0
        self.complete_after = 3
        self.stopped = False
        self.values_per_sample = 2
        # interleaved (min, max) pairs for 3 time points
        self.values = np.array([-1.0, 1.0, -0.5, 1.5, -0.2, 2.0])

    def connect(self) -> None:
        self.is_connected = True

    def disconnect(self) -> None:
        self.is_connected = False

    def configure_channel(self, ch, *, state=True, scale=None, offset=None, coupling=None):
        self.calls.append(f"chan={ch},state={state},scale={scale},off={offset},coup={coupling}")

    def set_decimation(self, ch, mode): self.calls.append(f"decim={ch},{mode}")
    def set_arithmetics(self, ch, mode): self.calls.append(f"arith={ch},{mode}")
    def set_time_range(self, v): self.calls.append(f"trange={v}")
    def set_record_length(self, v): self.calls.append(f"rlen={v}")
    def set_bandwidth_limit(self, ch, v): self.calls.append(f"bw={ch},{v}")
    def set_digital_filter(self, ch, v): self.calls.append(f"digf={ch},{v}")
    def set_post_trigger_window(self): self.calls.append("posttrig")
    def set_acquisition_count(self, count=1): self.calls.append(f"acount={count}")
    def set_trigger_mode(self, mode): self.calls.append(f"trigmode={mode}")

    def set_trigger(self, *, source="CHANnel1", level=None, slope="POSitive", mode="NORMal"):
        self.calls.append(f"trig={source},lvl={level},slope={slope},mode={mode}")

    def setup_mean_measurement(self, ch, *, group=1, gate_start=None, gate_stop=None):
        self.calls.append(f"meas={ch},g={group},gate={gate_start},{gate_stop}")

    def setup_stddev_measurement(self, ch, *, group=2, gate_start=None, gate_stop=None):
        self.calls.append(f"std={ch},g={group},gate={gate_start},{gate_stop}")

    def read_measurement(self, group=1):
        return 0.0425 if group == 1 else 0.0012  # group 1 = MEAN, group 2 = STDDev

    def single_acquisition(self) -> None:
        self.calls.append("single")
        self._polls = 0

    def is_acquisition_complete(self) -> bool:
        self._polls += 1
        return self._polls >= self.complete_after

    def stop(self) -> None:
        self.stopped = True
        self.calls.append("stop")

    def sample_rate(self) -> float:
        return 1.0e6

    def read_waveform_header(self, ch):
        return (0.0, 1.0, 3, self.values_per_sample)

    def read_waveform(self, ch, datatype="f"):
        return self.values


class ControllerTests(unittest.TestCase):
    def test_configure_issues_expected_calls(self):
        driver = FakeDriver()
        scope = ScopeController(driver=driver)
        scope.configure(ScopeSettings(
            channel=1, vertical_scale="0.1", coupling="DC",
            time_range="1.0", record_length=1_000_000, decimation="PDETect",
        ))
        self.assertIn("chan=1,state=True,scale=0.1,off=None,coup=DC", driver.calls)
        self.assertIn("decim=1,PDETect", driver.calls)
        self.assertIn("arith=1,OFF", driver.calls)  # raw single-shot by default
        self.assertIn("trange=1.0", driver.calls)
        self.assertIn("rlen=1000000", driver.calls)

    def test_acquire_returns_peak_detect_waveform(self):
        driver = FakeDriver()
        scope = ScopeController(driver=driver)
        wf = scope.acquire(ScopeSettings(), poll_interval=0.0)
        self.assertIsInstance(wf, Waveform)
        self.assertEqual(wf.n_points, 3)
        self.assertEqual(wf.values_per_sample, 2)
        np.testing.assert_allclose(wf.mins, [-1.0, -0.5, -0.2])
        np.testing.assert_allclose(wf.maxs, [1.0, 1.5, 2.0])
        np.testing.assert_allclose(wf.times, [0.0, 0.5, 1.0])
        self.assertEqual(wf.sample_rate, 1.0e6)
        self.assertIn("single", driver.calls)

    def test_single_decimation_waveform(self):
        driver = FakeDriver()
        driver.values_per_sample = 1
        driver.values = np.array([0.1, 0.2, 0.3])
        scope = ScopeController(driver=driver)
        wf = scope.acquire(ScopeSettings(decimation="SAMPle"), poll_interval=0.0)
        self.assertEqual(wf.values_per_sample, 1)
        np.testing.assert_allclose(wf.mins, wf.maxs)
        np.testing.assert_allclose(wf.values, [0.1, 0.2, 0.3])

    def test_stop_event_aborts(self):
        driver = FakeDriver()
        driver.complete_after = 1000  # never completes on its own
        scope = ScopeController(driver=driver)
        stop = threading.Event()
        stop.set()
        with self.assertRaises(ScopeError):
            scope.acquire(ScopeSettings(), poll_interval=0.0, stop_event=stop)
        self.assertTrue(driver.stopped)

    def test_timeout_raises_and_stops(self):
        driver = FakeDriver()
        driver.complete_after = 1000
        scope = ScopeController(driver=driver)
        with self.assertRaises(ScopeError):
            scope.run_single_acquisition(timeout=0.0, poll_interval=0.0)
        self.assertTrue(driver.stopped)

    def test_requires_host_or_driver(self):
        with self.assertRaises(ValueError):
            ScopeController()

    def test_waveform_roundtrip_csv_npz(self):
        wf = Waveform(
            times=np.array([0.0, 0.5, 1.0]),
            values=np.array([-1.0, 1.0, -0.5, 1.5, -0.2, 2.0]),
            values_per_sample=2,
            channel=1,
            sample_rate=1e6,
        )
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = wf.to_csv(Path(tmp) / "wf.csv")
            npz_path = wf.to_npz(Path(tmp) / "wf.npz")
            self.assertTrue(csv_path.exists())
            with np.load(npz_path) as loaded:
                np.testing.assert_allclose(loaded["times"], wf.times)
                np.testing.assert_allclose(loaded["values"], wf.values)


class MonitorTests(unittest.TestCase):
    def test_configure_monitor_issues_trigger_and_gate(self):
        driver = FakeDriver()
        scope = ScopeController(driver=driver)
        scope.configure_monitor(MonitorSettings(
            channel=1, trigger_source="CHANnel3", trigger_level=1.5,
            hold=0.1, duration=1.0, decimation="HRESolution",
            bandwidth_limit="B20", digital_filter_cutoff=10e3,
        ))
        calls = driver.calls
        self.assertIn("trig=CHANnel3,lvl=1.5,slope=POSitive,mode=NORMal", calls)
        self.assertIn("decim=1,HRESolution", calls)
        self.assertIn("bw=1,B20", calls)
        self.assertIn("digf=1,10000.0", calls)
        self.assertIn("posttrig", calls)
        self.assertIn("acount=1", calls)             # single acquisition per SINGle
        self.assertIn("trange=1.1", calls)          # hold + duration
        self.assertIn("meas=1,g=1,gate=0.1,1.1", calls)  # gate = [hold, hold+dur]
        self.assertIn("std=1,g=2,gate=0.1,1.1", calls)   # STDDev over the same gate

    def test_configure_monitor_auto_mode_no_gate(self):
        driver = FakeDriver()
        scope = ScopeController(driver=driver)
        scope.configure_monitor(MonitorSettings(
            channel=1, trigger_mode="AUTO", hold=0.1, duration=1.0,
        ))
        calls = driver.calls
        # free-run read: AUTO mode only, no armed edge trigger
        self.assertIn("trigmode=AUTO", calls)
        self.assertNotIn("trig=CHANnel1,lvl=1.5,slope=POSitive,mode=AUTO", calls)
        self.assertIn("acount=1", calls)
        self.assertIn("trange=1.0", calls)               # duration only (no hold)
        self.assertIn("meas=1,g=1,gate=None,None", calls)  # whole-record mean
        self.assertIn("std=1,g=2,gate=None,None", calls)   # whole-record stddev

    def test_monitor_cycle_returns_scope_mean(self):
        driver = FakeDriver()
        scope = ScopeController(driver=driver)
        scope.configure_monitor(MonitorSettings())
        sample = scope.monitor_cycle(index=5, poll_interval=0.0)
        self.assertIsInstance(sample, MonitorSample)
        self.assertEqual(sample.index, 5)
        self.assertAlmostEqual(sample.value, 0.0425)  # group 1 (MEAN)
        self.assertAlmostEqual(sample.std, 0.0012)    # group 2 (STDDev)
        self.assertIsNone(sample.waveform)

    def test_monitor_cycle_aborts_on_stop(self):
        driver = FakeDriver()
        driver.complete_after = 10_000  # never triggers on its own
        scope = ScopeController(driver=driver)
        scope.configure_monitor(MonitorSettings())
        stop = threading.Event()
        stop.set()
        self.assertIsNone(scope.monitor_cycle(poll_interval=0.0, stop_event=stop))

    def test_total_window(self):
        self.assertAlmostEqual(MonitorSettings(hold=0.1, duration=1.0).total_window, 1.1)


class SampleListenerTests(unittest.TestCase):
    def _scope(self) -> ScopeController:
        scope = ScopeController(driver=FakeDriver())
        scope.configure_monitor(MonitorSettings())
        return scope

    def test_monitor_cycle_notifies_listener_with_returned_sample(self):
        scope = self._scope()
        seen: list[MonitorSample] = []
        scope.add_sample_listener(seen.append)
        sample = scope.monitor_cycle(index=2, poll_interval=0.0)
        self.assertEqual(seen, [sample])

    def test_aborted_cycle_does_not_notify(self):
        scope = self._scope()
        scope.driver.complete_after = 10_000
        seen: list[MonitorSample] = []
        scope.add_sample_listener(seen.append)
        stop = threading.Event()
        stop.set()
        self.assertIsNone(scope.monitor_cycle(poll_interval=0.0, stop_event=stop))
        self.assertEqual(seen, [])

    def test_listener_exception_never_breaks_the_cycle(self):
        scope = self._scope()
        seen: list[MonitorSample] = []

        def bad(_sample) -> None:
            raise RuntimeError("display bug")

        scope.add_sample_listener(bad)
        scope.add_sample_listener(seen.append)
        sample = scope.monitor_cycle(poll_interval=0.0)
        self.assertIsInstance(sample, MonitorSample)
        self.assertEqual(seen, [sample])

    def test_duplicate_add_and_remove_semantics(self):
        scope = self._scope()
        seen: list[MonitorSample] = []
        scope.add_sample_listener(seen.append)
        scope.add_sample_listener(seen.append)      # dedupe: still one call
        scope.monitor_cycle(poll_interval=0.0)
        self.assertEqual(len(seen), 1)
        scope.remove_sample_listener(seen.append)
        scope.remove_sample_listener(seen.append)   # no-op when absent
        scope.monitor_cycle(poll_interval=0.0)
        self.assertEqual(len(seen), 1)


class DriverConstructionTests(unittest.TestCase):
    def test_default_resource_is_hislip(self):
        drv = RTO6_Driver(host="192.168.1.2")
        self.assertEqual(drv.resource, "TCPIP::192.168.1.2::hislip0::INSTR")

    def test_explicit_resource_overrides_host(self):
        drv = RTO6_Driver(resource="TCPIP::1.2.3.4::5025::SOCKET")
        self.assertEqual(drv.resource, "TCPIP::1.2.3.4::5025::SOCKET")

    def test_requires_host_or_resource(self):
        with self.assertRaises(ValueError):
            RTO6_Driver()


if __name__ == "__main__":
    unittest.main()
