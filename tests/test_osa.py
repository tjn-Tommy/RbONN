from __future__ import annotations

import socket
import sys
import threading
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from osa_module.controller import MeasurementSettings, OSAController, TraceData
from osa_module.driver import AQ637X_Driver, OSAConnectionError


class FakeDriver:
    """Records the calls OSAController makes, without any I/O."""

    def __init__(self) -> None:
        self.is_connected = False
        self.calls: list[str] = []
        self._sweep_polls = 0
        self.complete_after = 3
        self.sweep_count = 0
        self.y_per_sweep: list = []  # if set, read_trace_y returns one per sweep

    def connect(self) -> None:
        self.is_connected = True

    def disconnect(self) -> None:
        self.is_connected = False

    def set_center_wavelength(self, v): self.calls.append(f"cwl={v}")
    def set_span(self, v): self.calls.append(f"span={v}")
    def set_sensitivity(self, v): self.calls.append(f"sens={v}")
    def set_sampling_points(self, v): self.calls.append(f"pts={v}")
    def set_y_unit(self, v): self.calls.append(f"yunit={v}")
    def set_reference_level(self, v): self.calls.append(f"ref={v}")
    def select_trace(self, t, m): self.calls.append(f"trace={t},{m}")

    def initiate_single_sweep(self) -> None:
        self.calls.append("init")
        self._sweep_polls = 0
        self.sweep_count += 1

    def is_sweep_complete(self) -> bool:
        self._sweep_polls += 1
        return self._sweep_polls >= self.complete_after

    def read_trace_x(self, t):
        return np.array([1.549e-6, 1.550e-6, 1.551e-6])

    def read_trace_y(self, t):
        if self.y_per_sweep:
            return self.y_per_sweep[(self.sweep_count - 1) % len(self.y_per_sweep)]
        return np.array([1e-6, 2e-6, 1.5e-6])


class ControllerTests(unittest.TestCase):
    def test_requires_host_or_driver(self) -> None:
        with self.assertRaises(ValueError):
            OSAController()

    def test_measure_configures_sweeps_and_reads(self) -> None:
        fake = FakeDriver()
        with OSAController(driver=fake) as osa:
            self.assertTrue(osa.is_connected)
            trace = osa.measure(
                MeasurementSettings(center_wl="1550nm", span="2nm"),
                poll_interval=0.0,
            )
        self.assertFalse(fake.is_connected)
        self.assertIn("cwl=1550nm", fake.calls)
        self.assertIn("init", fake.calls)
        self.assertIsInstance(trace, TraceData)
        self.assertEqual(trace.n_points, 3)
        self.assertAlmostEqual(trace.wavelengths_nm[1], 1550.0, places=6)
        self.assertEqual(trace.power_label, "power_W")

    def test_measure_rejects_bad_average_count(self) -> None:
        with self.assertRaises(ValueError):
            OSAController(driver=FakeDriver()).measure(averages=0)

    def test_linear_averaging_means_power(self) -> None:
        fake = FakeDriver()
        fake.y_per_sweep = [
            np.array([1.0, 2.0, 3.0]),
            np.array([3.0, 4.0, 5.0]),
        ]
        osa = OSAController(driver=fake)
        osa.configure(MeasurementSettings(y_unit="LINear"))
        trace = osa.measure(averages=2, poll_interval=0.0)
        self.assertEqual(fake.sweep_count, 2)
        self.assertEqual(trace.averages, 2)
        np.testing.assert_allclose(trace.powers, [2.0, 3.0, 4.0])

    def test_log_averaging_happens_in_linear_domain(self) -> None:
        # 0 dBm (1 mW) and -10 dBm (0.1 mW) -> mean 0.55 mW -> ~-2.596 dBm,
        # which differs from the naive dBm mean of -5 dBm.
        fake = FakeDriver()
        fake.y_per_sweep = [np.array([0.0]), np.array([-10.0])]
        osa = OSAController(driver=fake)
        osa.configure(MeasurementSettings(y_unit="LOGarithmic"))
        trace = osa.measure(averages=2, poll_interval=0.0)
        expected = 10.0 * np.log10((1.0 + 0.1) / 2.0)
        self.assertAlmostEqual(float(trace.powers[0]), expected, places=6)
        self.assertNotAlmostEqual(float(trace.powers[0]), -5.0, places=2)

    def test_run_single_sweep_honours_stop_event(self) -> None:
        fake = FakeDriver()
        fake.complete_after = 10_000  # never completes on its own
        stop = threading.Event()
        stop.set()
        osa = OSAController(driver=fake)
        self.assertFalse(osa.run_single_sweep(poll_interval=0.0, stop_event=stop))

    def test_trace_to_csv_roundtrip(self) -> None:
        import csv
        import tempfile

        fake = FakeDriver()
        osa = OSAController(driver=fake)
        osa.configure(MeasurementSettings(y_unit="LOGarithmic"))
        trace = osa.read_trace()
        with tempfile.TemporaryDirectory() as tmp:
            path = trace.to_csv(Path(tmp) / "spectrum.csv")
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle))
        self.assertEqual(rows[0], ["wavelength_m", "power_dBm"])
        self.assertEqual(len(rows), trace.n_points + 1)


def _fake_osa_server(ready: threading.Event, port_box: dict) -> None:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port_box["port"] = srv.getsockname()[1]
    ready.set()
    conn, _ = srv.accept()
    conn.settimeout(5.0)
    buf = b""

    def readline():
        nonlocal buf
        while b"\r\n" not in buf:
            chunk = conn.recv(4096)
            if not chunk:
                return None
            buf += chunk
        line, buf = buf.split(b"\r\n", 1)
        return line.decode("ascii")

    assert readline().startswith('open "anonymous"')
    conn.sendall(b"prompt\r\n")
    readline()  # empty password line
    conn.sendall(b"ready\r\n")

    while True:
        line = readline()
        if line is None:
            break
        if line.startswith("CFORM"):
            continue
        if line == "*IDN?":
            conn.sendall(b"YOKOGAWA,AQ6370,0,1.0\r\n")
        elif line.startswith(":TRACe:X?"):
            conn.sendall(b"3,1.549e-6,1.550e-6,1.551e-6\r\n")
        elif line.startswith(":TRACe:Y?"):
            conn.sendall(b"3,1e-6,2e-6,1.5e-6\r\n")
        elif line.endswith("?"):
            conn.sendall(b"0\r\n")
    conn.close()
    srv.close()


class DriverTransportTests(unittest.TestCase):
    def test_handshake_and_trace_download(self) -> None:
        ready = threading.Event()
        port_box: dict = {}
        threading.Thread(
            target=_fake_osa_server, args=(ready, port_box), daemon=True
        ).start()
        self.assertTrue(ready.wait(5.0))

        drv = AQ637X_Driver(
            "127.0.0.1", port_box["port"], write_delay=0.0, timeout=5.0
        )
        drv.connect()
        try:
            self.assertTrue(drv.is_connected)
            self.assertIn("AQ6370", drv.identify())
            x = drv.read_trace_x("TRA")
            y = drv.read_trace_y("TRA")
            self.assertEqual(x.size, 3)  # leading count header dropped
            self.assertEqual(y.size, 3)
            self.assertAlmostEqual(float(x[1]), 1.550e-6, places=12)
        finally:
            drv.disconnect()
        self.assertFalse(drv.is_connected)

    def test_query_before_connect_raises(self) -> None:
        drv = AQ637X_Driver("127.0.0.1", 1, timeout=0.5)
        with self.assertRaises(OSAConnectionError):
            drv.query("*IDN?")

    def test_connect_to_dead_port_raises(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()  # nothing listening now
        drv = AQ637X_Driver("127.0.0.1", port, timeout=1.0)
        with self.assertRaises(OSAConnectionError):
            drv.connect()


class TraceListenerTests(unittest.TestCase):
    def test_measure_notifies_listener_with_returned_trace(self) -> None:
        osa = OSAController(driver=FakeDriver())
        seen: list = []
        osa.add_trace_listener(seen.append)
        trace = osa.measure(MeasurementSettings(), poll_interval=0.0)
        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0], trace)

    def test_averaged_measure_notifies_once_with_averaged_trace(self) -> None:
        fake = FakeDriver()
        fake.y_per_sweep = [np.array([1.0, 2.0, 3.0]), np.array([3.0, 4.0, 5.0])]
        osa = OSAController(driver=fake)
        osa.configure(MeasurementSettings(y_unit="LINear"))
        seen: list = []
        osa.add_trace_listener(seen.append)
        trace = osa.measure(averages=2, poll_interval=0.0)
        self.assertEqual(len(seen), 1)          # once per measure(), not per sweep
        self.assertIs(seen[0], trace)
        self.assertEqual(seen[0].averages, 2)

    def test_raising_listener_does_not_break_measure(self) -> None:
        osa = OSAController(driver=FakeDriver())
        seen: list = []

        def bad_listener(trace) -> None:
            raise RuntimeError("display bug")

        osa.add_trace_listener(bad_listener)
        osa.add_trace_listener(seen.append)     # later listener still fires
        trace = osa.measure(MeasurementSettings(), poll_interval=0.0)
        self.assertIsInstance(trace, TraceData)
        self.assertEqual(len(seen), 1)

    def test_add_remove_semantics(self) -> None:
        osa = OSAController(driver=FakeDriver())
        seen: list = []
        listener = seen.append
        osa.add_trace_listener(listener)
        osa.add_trace_listener(listener)        # duplicate registration ignored
        osa.measure(MeasurementSettings(), poll_interval=0.0)
        self.assertEqual(len(seen), 1)
        osa.remove_trace_listener(listener)
        osa.remove_trace_listener(listener)     # extra removal is a no-op
        osa.measure(MeasurementSettings(), poll_interval=0.0)
        self.assertEqual(len(seen), 1)          # unhooked -> no new traces


if __name__ == "__main__":
    unittest.main()
