from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slm_module.keepalive import SLMKeepAlive


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class KeepAliveTests(unittest.TestCase):
    def test_pings_periodically_and_stops(self) -> None:
        pings = []
        keepalive = SLMKeepAlive(
            ping=lambda: pings.append(time.monotonic()),
            interval_seconds=0.01,
        )

        keepalive.start()
        self.assertTrue(keepalive.is_running)
        self.assertTrue(_wait_until(lambda: len(pings) >= 3))
        keepalive.stop()

        self.assertFalse(keepalive.is_running)
        count = len(pings)
        time.sleep(0.05)
        self.assertEqual(len(pings), count)

    def test_suspend_and_resume(self) -> None:
        pings = []
        keepalive = SLMKeepAlive(
            ping=lambda: pings.append(1),
            interval_seconds=0.01,
        )
        keepalive.start()
        self.assertTrue(_wait_until(lambda: len(pings) >= 1))

        keepalive.suspend()
        time.sleep(0.05)
        count = len(pings)
        time.sleep(0.05)
        self.assertLessEqual(len(pings) - count, 1)

        keepalive.resume()
        self.assertTrue(_wait_until(lambda: len(pings) > count + 1))
        keepalive.stop()

    def test_failed_ping_reports_status_and_keeps_running(self) -> None:
        statuses = []
        attempts = []
        report = threading.Event()

        def flaky_ping() -> None:
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("usb glitch")

        def on_status(ok: bool, message: str) -> None:
            statuses.append((ok, message))
            if len(statuses) >= 2:
                report.set()

        keepalive = SLMKeepAlive(
            ping=flaky_ping, interval_seconds=0.01, on_status=on_status
        )
        keepalive.start()
        self.assertTrue(report.wait(5.0))
        keepalive.stop()

        self.assertEqual(statuses[0][0], False)
        self.assertIn("usb glitch", statuses[0][1])
        self.assertEqual(statuses[1][0], True)

    def test_rejects_invalid_interval(self) -> None:
        with self.assertRaises(ValueError):
            SLMKeepAlive(ping=lambda: None, interval_seconds=0)


if __name__ == "__main__":
    unittest.main()
