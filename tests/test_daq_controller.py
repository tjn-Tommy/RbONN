from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from daq_module.controller import DAQController, DAQMonitorSettings, MonitorSample
from daq_module.driver import DAQConnectionError


class FakeDriver:
    """Records the calls DAQController makes, without any real hardware."""

    def __init__(self) -> None:
        self.is_connected = False
        self.calls: list[str] = []
        self.values = np.array([0.0119, 0.0121, 0.0125, 0.0127])

    def connect(self) -> None:
        self.is_connected = True

    def disconnect(self) -> None:
        self.is_connected = False

    def identify(self) -> str:
        return "Dev1 (fake)"

    def read_waveform(self, *, channel, sample_rate, duration, min_val, max_val, timeout):
        self.calls.append(
            f"read ch={channel} rate={sample_rate} dur={duration} "
            f"range=[{min_val},{max_val}] timeout={timeout}"
        )
        return self.values


class DAQControllerTests(unittest.TestCase):
    def test_requires_device_or_driver(self) -> None:
        with self.assertRaises(ValueError):
            DAQController()

    def test_monitor_cycle_returns_sample_with_configured_settings(self) -> None:
        driver = FakeDriver()
        daq = DAQController(driver=driver)
        daq.connect()
        daq.configure_monitor(
            DAQMonitorSettings(
                channel="ai0", sample_rate=100_000.0, duration=0.05,
                hold=0.0, min_val=-0.01, max_val=0.05,
            )
        )
        sample = daq.monitor_cycle(index=3, timeout=5.0)

        self.assertIsInstance(sample, MonitorSample)
        self.assertEqual(sample.index, 3)
        self.assertAlmostEqual(sample.value, float(driver.values.mean()))
        self.assertIn(
            "read ch=ai0 rate=100000.0 dur=0.05 range=[-0.01,0.05] timeout=5.0",
            driver.calls,
        )
        np.testing.assert_allclose(daq.last_values, driver.values)
        self.assertEqual(daq.last_times.size, driver.values.size)
        np.testing.assert_allclose(daq.last_times, np.arange(4) / 100_000.0)

    def test_monitor_cycle_uses_default_settings_when_unconfigured(self) -> None:
        driver = FakeDriver()
        daq = DAQController(driver=driver)
        sample = daq.monitor_cycle()
        self.assertIsInstance(sample, MonitorSample)

    def test_monitor_cycle_aborts_on_stop_event(self) -> None:
        driver = FakeDriver()
        daq = DAQController(driver=driver)
        stop = threading.Event()
        stop.set()
        self.assertIsNone(daq.monitor_cycle(stop_event=stop))
        self.assertEqual(driver.calls, [])

    def test_disconnect_reflects_driver_state(self) -> None:
        driver = FakeDriver()
        daq = DAQController(driver=driver)
        daq.connect()
        self.assertTrue(daq.is_connected)
        daq.disconnect()
        self.assertFalse(daq.is_connected)


class SampleListenerTests(unittest.TestCase):
    def test_monitor_cycle_notifies_listener_with_returned_sample(self) -> None:
        daq = DAQController(driver=FakeDriver())
        daq.configure_monitor(DAQMonitorSettings(hold=0.0))
        seen: list[MonitorSample] = []
        daq.add_sample_listener(seen.append)
        sample = daq.monitor_cycle(index=7)
        self.assertEqual(seen, [sample])

    def test_aborted_cycle_does_not_notify(self) -> None:
        daq = DAQController(driver=FakeDriver())
        daq.configure_monitor(DAQMonitorSettings(hold=0.0))
        seen: list[MonitorSample] = []
        daq.add_sample_listener(seen.append)
        stop = threading.Event()
        stop.set()
        self.assertIsNone(daq.monitor_cycle(stop_event=stop))
        self.assertEqual(seen, [])

    def test_listener_exception_never_breaks_the_cycle(self) -> None:
        daq = DAQController(driver=FakeDriver())
        daq.configure_monitor(DAQMonitorSettings(hold=0.0))
        seen: list[MonitorSample] = []

        def bad(_sample) -> None:
            raise RuntimeError("display bug")

        daq.add_sample_listener(bad)
        daq.add_sample_listener(seen.append)
        sample = daq.monitor_cycle()
        self.assertIsInstance(sample, MonitorSample)
        self.assertEqual(seen, [sample])

    def test_duplicate_add_and_remove_semantics(self) -> None:
        daq = DAQController(driver=FakeDriver())
        daq.configure_monitor(DAQMonitorSettings(hold=0.0))
        seen: list[MonitorSample] = []
        daq.add_sample_listener(seen.append)
        daq.add_sample_listener(seen.append)      # dedupe: still one call
        daq.monitor_cycle()
        self.assertEqual(len(seen), 1)
        daq.remove_sample_listener(seen.append)
        daq.remove_sample_listener(seen.append)   # no-op when absent
        daq.monitor_cycle()
        self.assertEqual(len(seen), 1)


class DriverConstructionTests(unittest.TestCase):
    def test_unconnected_read_raises(self) -> None:
        from daq_module.driver import NIDAQDriver

        drv = NIDAQDriver(device="Dev1")
        with self.assertRaises(DAQConnectionError):
            drv.read_waveform(
                channel="ai0", sample_rate=100_000.0, duration=0.01,
                min_val=-0.01, max_val=0.05, timeout=5.0,
            )


if __name__ == "__main__":
    unittest.main()
