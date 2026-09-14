from __future__ import annotations

import json
import os

# Render Qt/matplotlib headless before either is imported by the app module.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gui.app import _format_duration


class FormatDurationTests(unittest.TestCase):
    def test_minutes_and_seconds(self) -> None:
        self.assertEqual(_format_duration(0), "0:00")
        self.assertEqual(_format_duration(7), "0:07")
        self.assertEqual(_format_duration(67), "1:07")
        self.assertEqual(_format_duration(750), "12:30")

    def test_rolls_over_to_hours(self) -> None:
        self.assertEqual(_format_duration(3725), "1:02:05")

    def test_non_finite_or_negative_is_dash(self) -> None:
        self.assertEqual(_format_duration(-1), "—")
        self.assertEqual(_format_duration(float("nan")), "—")
        self.assertEqual(_format_duration(float("inf")), "—")


class FormatVoltsTests(unittest.TestCase):
    def test_scales_to_readable_units(self) -> None:
        from gui.live_readout import _format_volts

        self.assertEqual(_format_volts(1.5), "1.5 V")
        self.assertEqual(_format_volts(0.0123), "12.3 mV")
        self.assertEqual(_format_volts(-0.0004), "-400 \N{MICRO SIGN}V")
        self.assertEqual(_format_volts(0.0), "0 V")


class CalibrationDialogEtaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PyQt5 import QtWidgets

        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_eta_estimates_remaining_from_pace(self) -> None:
        from calibration_module.measure.slm_response import CalibrationProgress
        from gui.app import CalibrationProgressDialog

        dialog = CalibrationProgressDialog()
        try:
            # first update enters the phase and starts the clock
            dialog.update_progress(
                CalibrationProgress("intensity", step=0, total=10, message="start")
            )
            # pin the phase start 10 s in the past: 1/10 done -> ~90 s remaining
            dialog._phase_start = time.perf_counter() - 10.0
            dialog.update_progress(
                CalibrationProgress("intensity", step=0, total=10, message="tick")
            )
            text = dialog.eta_label.text()
            self.assertTrue(text.startswith("Elapsed 0:1"), text)
            self.assertIn("ETA 1:", text)
        finally:
            dialog.close()


class MainWindowStartupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PyQt5 import QtWidgets

        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_navigation_has_one_page_per_item(self) -> None:
        from gui.app import MainWindow

        window = MainWindow()
        try:
            self.assertEqual(window.nav.count(), window.stack.count())
            labels = [window.nav.item(i).text() for i in range(window.nav.count())]
            self.assertEqual(sum("TPA Encoding" in label for label in labels), 1)
            self.assertFalse(any("Scope Monitor" in label for label in labels))
        finally:
            window.close()

    def test_encoding_shape_defaults_to_off(self) -> None:
        from gui.app import MainWindow

        window = MainWindow()
        try:
            self.assertFalse(window.shape_enabled_check.isChecked())
            self.assertFalse(window.edge_table.isEnabled())
            self.assertIsNone(window._active_col_ratio())
        finally:
            window.close()

    def test_pipeline_loads_stage1_result_profile_key(self) -> None:
        from gui.app import MainWindow

        window = MainWindow()
        try:
            expected = np.linspace(0.2, 0.9, 8)
            with tempfile.TemporaryDirectory() as temp_dir:
                path = Path(temp_dir) / "stage1_result.json"
                path.write_text(
                    json.dumps({"l": expected.tolist(), "skipped": True}),
                    encoding="utf-8",
                )
                parsed = window._load_pipeline_initial_profile(path)
            np.testing.assert_allclose(parsed, expected)
        finally:
            window.close()

    def test_encoding_page_loads_layout_verbatim_from_calibration(self) -> None:
        from calibration_module.fit.slm_response import CalibrationResult
        from gui.app import MainWindow

        # mirror-symmetric channel grid: centre 510 px, pitch 20, one guard skip
        coords = np.array([420.0, 440.0, 480.0, 500.0, 520.0, 540.0, 580.0, 600.0])
        wavelengths = -0.005 * coords + 781.0
        # the encoding page loads with method="fit", so the fixture curve has
        # to be sin^2-shaped -- see tests/test_slm_encoding.sin2_curve
        levels = np.arange(380, 901, 20, dtype=int)
        intensity = np.tile(
            np.sin(np.pi * (levels - 390.0) / (2.0 * (860.0 - 390.0))) ** 2,
            (coords.size, 1),
        )
        calib = CalibrationResult(
            wavelength=wavelengths,
            coordinates=coords,
            max_level=1023,
            min_level=0,
            level_range=levels,
            intensity_levels=intensity,
        )

        window = MainWindow()
        try:
            window._enc_calib_override = calib
            window._enc_build_layout()
            layout = window.encoding_layout
            self.assertIsNotNone(layout)
            self.assertEqual(layout.n_channels, 4)
            # channels sit exactly on the calibrated coordinates, no re-tiling
            self.assertEqual(
                [ch.x_center for ch in layout.x_channels], [500, 480, 440, 420]
            )
            self.assertEqual(
                [ch.x_center for ch in layout.w_channels], [520, 540, 580, 600]
            )
            self.assertAlmostEqual(layout.center_x, 510.0)
            self.assertEqual(layout.pitch_px, 20)
            self.assertEqual(layout.channel_width_px, 15)   # pitch - default gap
            self.assertEqual(window.enc_val_table.rowCount(), 4)
            self.assertTrue(window.enc_generate_button.isEnabled())
        finally:
            window.close()

    def test_live_readout_dock_receives_bridged_samples(self) -> None:
        from scope_module.controller import MonitorSample
        from gui.app import MainWindow

        window = MainWindow()
        try:
            dock = window.live_readout_dock
            view = dock.view
            self.assertEqual(view.sample_count, 0)
            # same-thread emit is delivered synchronously by Qt
            dock.bridge.on_sample(
                MonitorSample(
                    value=0.0123, std=0.0004, std_ratio=0.0325, index=0, timestamp=1.0
                )
            )
            dock.bridge.on_sample(
                MonitorSample(value=0.0125, std=None, index=1, timestamp=2.0)
            )
            self.assertEqual(view.sample_count, 2)
            view._draw_samples()   # render path handles a None std/std_ratio
            self.assertIn("reading 2", view.status_label.text())
            view.clear()
            self.assertEqual(view.sample_count, 0)
            self.assertTrue(window.mon_dock_button.isCheckable())
            self.assertFalse(dock.isVisible())
        finally:
            window.close()

    def test_live_readout_dock_watch_unwatch(self) -> None:
        from gui.app import MainWindow

        class _FakeMonitorController:
            def __init__(self) -> None:
                self.listeners: list = []

            def add_sample_listener(self, listener) -> None:
                self.listeners.append(listener)

            def remove_sample_listener(self, listener) -> None:
                self.listeners.remove(listener)

        window = MainWindow()
        try:
            monitor = _FakeMonitorController()
            window.live_readout_dock.watch(monitor)
            self.assertEqual(len(monitor.listeners), 1)
            window.live_readout_dock.unwatch(monitor)
            self.assertEqual(monitor.listeners, [])
        finally:
            window.close()

    def test_step_settings_include_sampling_points(self) -> None:
        from gui.app import MainWindow

        window = MainWindow()
        try:
            window.step_widgets[2]["sampling_points"].setText("2001")
            settings = window._step_settings(2)
            self.assertEqual(settings.sampling_points, "2001")
            self.assertEqual(window._step_settings(1).sampling_points, "AUTO")
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
