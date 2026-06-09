from __future__ import annotations

import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slm_module.controller import ScanParams, SLMController, validate_slm_csv
from slm_module.detector import SimulatedDetector
from slm_module.driver import MODE_DVI, MODE_MEMORY


class FakeDriver:
    def __init__(self, size: tuple[int, int] = (10, 4)):
        self.size = size
        self.loaded_csv: list[tuple[str, float]] = []
        self.grayscale: list[tuple[int, float]] = []
        self.opened = False
        self.video_mode = MODE_DVI
        self.pings: list[tuple[int, bool]] = []

    def slm_info(self) -> tuple[int, int]:
        return self.size

    def open_slm(self) -> None:
        self.opened = True

    def close_slm(self) -> None:
        self.opened = False

    def load_grayscale(self, grayscale: int, interval: float = 0.2) -> None:
        self.grayscale.append((grayscale, interval))

    def load_csv(self, csv_path: str, interval: float = 0.2) -> None:
        self.loaded_csv.append((csv_path, interval))

    def ping(self, slm_number: int = 1, verify_video_mode: bool = False) -> int | None:
        self.pings.append((slm_number, verify_video_mode))
        return self.video_mode if verify_video_mode else None


class ControllerTests(unittest.TestCase):
    def test_display_grayscale_validates_level(self) -> None:
        fake = FakeDriver()
        controller = SLMController(driver=fake)

        controller.display_grayscale(123, interval=0.01)

        self.assertEqual(fake.grayscale, [(123, 0.01)])
        with self.assertRaises(ValueError):
            controller.display_grayscale(2000)

    def test_display_vertical_window_writes_and_loads_csv(self) -> None:
        fake = FakeDriver(size=(8, 3))
        controller = SLMController(driver=fake)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "window.csv"
            result = controller.display_vertical_window(
                x_start=2,
                level=400,
                window_px=5,
                csv_path=path,
                interval=0.01,
            )

            self.assertEqual(result, path.resolve())
            self.assertEqual(len(fake.loaded_csv), 1)
            validate_slm_csv(result, expected_width=8, expected_height=3)

    def test_display_center_scan_stops_when_event_is_set(self) -> None:
        fake = FakeDriver(size=(10, 2))
        controller = SLMController(driver=fake)
        seen: list[Path] = []

        def progress(index: int, path: Path) -> None:
            seen.append(path)
            if index == 0:
                stop_event.set()

        import threading

        stop_event = threading.Event()
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = controller.display_center_scan(
                level=200,
                window_px=5,
                step_px=5,
                output_dir=temp_dir,
                stop_event=stop_event,
                progress_callback=progress,
                dwell_seconds=0.01,
            )

        self.assertEqual(len(paths), 1)
        self.assertEqual(len(seen), 1)
        self.assertEqual(len(fake.loaded_csv), 1)


class RunCenterScanTests(unittest.TestCase):
    def test_collects_detector_samples_and_center(self) -> None:
        fake = FakeDriver(size=(20, 2))
        controller = SLMController(driver=fake)
        params = ScanParams(200, window_px=2, step_px=2, dwell_seconds=0.001)
        detector = SimulatedDetector(center_x=11.0, sigma_px=4.0, noise=0.0)

        with tempfile.TemporaryDirectory() as temp_dir:
            result = controller.run_center_scan(
                params, output_dir=temp_dir, detector=detector
            )

            self.assertTrue(result.samples_path is not None and result.samples_path.exists())

        self.assertEqual(len(result.frames), 10)
        self.assertEqual(len(result.samples), 10)
        self.assertIsNotNone(result.center)
        self.assertAlmostEqual(result.center.peak_x, 11.0)

    def test_step_change_applies_on_next_frame(self) -> None:
        fake = FakeDriver(size=(10, 2))
        controller = SLMController(driver=fake)
        params = ScanParams(100, window_px=2, step_px=2, dwell_seconds=0.001)
        positions: list[int] = []

        def progress(index: int, x: int, path: Path) -> None:
            positions.append(x)
            if index == 0:
                params.update(step_px=4)

        with tempfile.TemporaryDirectory() as temp_dir:
            controller.run_center_scan(
                params, output_dir=temp_dir, progress_callback=progress
            )

        self.assertEqual(positions, [0, 4, 8])

    def test_pause_blocks_frames_until_resume(self) -> None:
        fake = FakeDriver(size=(10, 2))
        controller = SLMController(driver=fake)
        params = ScanParams(100, window_px=2, step_px=2, dwell_seconds=0.001)
        stop_event = threading.Event()
        pause_event = threading.Event()
        pause_event.set()
        holder: dict[str, object] = {}

        def run() -> None:
            with tempfile.TemporaryDirectory() as temp_dir:
                holder["result"] = controller.run_center_scan(
                    params,
                    output_dir=temp_dir,
                    stop_event=stop_event,
                    pause_event=pause_event,
                )

        thread = threading.Thread(target=run)
        thread.start()
        time.sleep(0.2)
        self.assertEqual(len(fake.loaded_csv), 0)

        pause_event.clear()
        thread.join(5.0)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(holder["result"].frames), 5)

    def test_stop_while_paused_exits(self) -> None:
        fake = FakeDriver(size=(10, 2))
        controller = SLMController(driver=fake)
        params = ScanParams(100, window_px=2, step_px=2, dwell_seconds=0.001)
        stop_event = threading.Event()
        pause_event = threading.Event()
        pause_event.set()
        holder: dict[str, object] = {}

        def run() -> None:
            with tempfile.TemporaryDirectory() as temp_dir:
                holder["result"] = controller.run_center_scan(
                    params,
                    output_dir=temp_dir,
                    stop_event=stop_event,
                    pause_event=pause_event,
                )

        thread = threading.Thread(target=run)
        thread.start()
        time.sleep(0.1)
        stop_event.set()
        thread.join(5.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(len(holder["result"].frames), 0)

    def test_scan_params_validates_updates(self) -> None:
        params = ScanParams(100)
        with self.assertRaises(ValueError):
            params.update(level=5000)
        with self.assertRaises(ValueError):
            params.update(step_px=0)
        params.update(dwell_seconds=1.5)
        self.assertEqual(params.snapshot().dwell_seconds, 1.5)


class RefreshDisplayTests(unittest.TestCase):
    def test_returns_false_when_nothing_displayed(self) -> None:
        controller = SLMController(driver=FakeDriver())

        self.assertFalse(controller.refresh_display())

    def test_resends_last_grayscale(self) -> None:
        fake = FakeDriver()
        controller = SLMController(driver=fake)
        controller.display_grayscale(300, interval=0.01)

        self.assertTrue(controller.refresh_display())

        self.assertEqual(fake.grayscale, [(300, 0.01), (300, 0.0)])

    def test_resends_last_scan_frame(self) -> None:
        fake = FakeDriver(size=(10, 2))
        controller = SLMController(driver=fake)
        params = ScanParams(100, window_px=2, step_px=2, dwell_seconds=0.001)

        with tempfile.TemporaryDirectory() as temp_dir:
            result = controller.run_center_scan(params, output_dir=temp_dir)
            frames_sent = len(fake.loaded_csv)

            self.assertTrue(controller.refresh_display())

            self.assertEqual(len(fake.loaded_csv), frames_sent + 1)
            self.assertEqual(fake.loaded_csv[-1], (str(result.frames[-1]), 0.0))

    def test_returns_false_after_close(self) -> None:
        fake = FakeDriver()
        controller = SLMController(driver=fake)
        controller.display_grayscale(300, interval=0.01)
        controller.close_slm()

        self.assertFalse(controller.refresh_display())


class PingTests(unittest.TestCase):
    def test_ping_passes_through(self) -> None:
        fake = FakeDriver()
        controller = SLMController(driver=fake)

        controller.ping(slm_number=2)

        self.assertEqual(fake.pings, [(2, False)])

    def test_ping_verify_dvi_raises_on_memory_mode(self) -> None:
        fake = FakeDriver()
        controller = SLMController(driver=fake)
        controller.ping(verify_dvi=True)

        fake.video_mode = MODE_MEMORY
        with self.assertRaises(RuntimeError):
            controller.ping(verify_dvi=True)


if __name__ == "__main__":
    unittest.main()
