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

from slm_module.gui.app import _format_duration


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
        from slm_module.gui.daq_monitor import _format_volts

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
        from slm_module.calibration.calibration_new import CalibrationProgress
        from slm_module.gui.app import CalibrationProgressDialog

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
        from slm_module.gui.app import MainWindow

        window = MainWindow()
        try:
            self.assertEqual(window.nav.count(), window.stack.count())
            labels = [window.nav.item(i).text() for i in range(window.nav.count())]
            self.assertEqual(sum("TPA Encoding" in label for label in labels), 1)
            self.assertFalse(any("Scope Monitor" in label for label in labels))
        finally:
            window.close()

    def test_encoding_shape_defaults_to_off(self) -> None:
        from slm_module.gui.app import MainWindow

        window = MainWindow()
        try:
            self.assertFalse(window.shape_enabled_check.isChecked())
            self.assertFalse(window.edge_table.isEnabled())
            self.assertIsNone(window._active_col_ratio())
        finally:
            window.close()

    def test_pipeline_loads_stage1_result_profile_key(self) -> None:
        from slm_module.gui.app import MainWindow

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

    def test_unified_pipeline_page_dependency_toggle(self) -> None:
        from slm_module.gui.app import MainWindow

        window = MainWindow()
        try:
            page = window.pipeline_page
            wl_row = page.rows["wl_map"]
            int_row = page.rows["intensity"]
            wl_row.group.setChecked(True)
            int_row.group.setChecked(True)
            combo = int_row.input_combos["wl_map"]
            combo.setCurrentText("From memory")
            self.assertEqual(combo.currentText(), "From memory")
            # disabling the producer forces the dependent input back to file
            wl_row.group.setChecked(False)
            self.assertEqual(combo.currentText(), "From file…")
        finally:
            window.close()

    def test_unified_pipeline_request_building(self) -> None:
        from slm_module.gui.app import MainWindow
        from slm_module.pipeline import validate_request

        window = MainWindow()
        try:
            page = window.pipeline_page
            for row in page.rows.values():
                row.group.setChecked(False)
            page.rows["wl_map"].group.setChecked(True)
            page.rows["intensity"].group.setChecked(True)
            page.rows["intensity"].input_combos["wl_map"].setCurrentText(
                "From memory"
            )
            page.lay_center_gap_check.setChecked(True)
            page.lay_center_gap.setValue(10)
            page.wl_osa.points.setText("501")

            request = page._build_request(0.15)
            validate_request(request)

            self.assertEqual(
                [plan.stage_id for plan in request.stages],
                ["wl_map", "intensity"],
            )
            self.assertEqual(
                request.stages[1].inputs["wl_map"].source, "memory"
            )
            self.assertEqual(request.layout.center_gap_px, 10)
            settings = request.stages[0].config.osa.to_measurement_settings()
            self.assertEqual(settings.sampling_points, "501")
            self.assertIsNotNone(request.stages[0].config.outlier_policy)
        finally:
            window.close()

    def test_encoding_page_center_gap_control(self) -> None:
        from slm_module.gui.app import MainWindow

        window = MainWindow()
        try:
            self.assertIsNone(window._enc_center_gap())    # legacy default
            window.enc_center_gap_check.setChecked(True)
            window.enc_center_gap_spin.setValue(12)
            self.assertEqual(window._enc_center_gap(), 12)
        finally:
            window.close()

    def test_daq_monitor_dock_receives_bridged_samples(self) -> None:
        from scope_module.controller import MonitorSample
        from slm_module.gui.app import MainWindow

        window = MainWindow()
        try:
            view = window.daq_monitor_view
            self.assertEqual(view.sample_count, 0)
            # same-thread emit is delivered synchronously by Qt
            window._monitor_bridge.on_sample(
                MonitorSample(value=0.0123, std=0.0004, index=0, timestamp=1.0)
            )
            window._monitor_bridge.on_sample(
                MonitorSample(value=0.0125, std=None, index=1, timestamp=2.0)
            )
            self.assertEqual(view.sample_count, 2)
            view._draw_samples()   # render path handles a None std
            self.assertIn("reading 2", view.status_label.text())
            view.clear()
            self.assertEqual(view.sample_count, 0)
            self.assertTrue(window.mon_dock_button.isCheckable())
            self.assertFalse(window.daq_monitor_dock.isVisible())
        finally:
            window.close()

    def test_step_settings_include_sampling_points(self) -> None:
        from slm_module.gui.app import MainWindow

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
