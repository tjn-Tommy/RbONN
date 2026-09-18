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


class StepDefaultsTests(unittest.TestCase):
    """The values the calibration pages come up with.

    These are what actually gets run when nobody retypes anything, so they are
    worth pinning: a default that drifts is not a visible bug, it is a run
    measured under settings nobody chose. Change the page and change these in
    the same commit -- that is the point of the test, not that any particular
    number is sacred.
    """

    @classmethod
    def setUpClass(cls) -> None:
        from gui.app import MainWindow
        cls.win = MainWindow()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.win.close()

    def test_step_2_sweep_defaults(self) -> None:
        w = self.win.step_widgets[2]
        self.assertEqual(w["window"].value(), 20)
        self.assertEqual(w["stride"].value(), 30)
        self.assertAlmostEqual(w["peak_nm"].value(), 0.2)
        self.assertAlmostEqual(w["sweep_nm"].value(), 1.0)
        self.assertAlmostEqual(w["min_wl"].value(), 775.0)
        self.assertAlmostEqual(w["max_wl"].value(), 781.0)

    def test_step_3c_channel_grid_defaults(self) -> None:
        w = self.win.step_widgets["3c"]
        self.assertAlmostEqual(w["target"].value(), 778.04)
        # 15 px window + 35 px gap: the geometry the 0916 ten-pair run used
        self.assertEqual(w["window"].value(), 15)
        self.assertEqual(w["pad"].value(), 35)
        self.assertEqual(w["count"].value(), 10)
        self.assertTrue(w["guard_check"].isChecked())
        self.assertAlmostEqual(w["guard_nm"].value(), 0.06)

    def test_step_3c_sweeps_the_usable_middle_of_the_level_range(self) -> None:
        """Not 0-1023: 3c starts well above the floor and stops below the top."""
        w = self.win.step_widgets["3c"]
        self.assertEqual(
            (w["level_start"].value(), w["level_stop"].value(),
             w["level_step"].value()),
            (350, 950, 25),
        )

    def test_step_1_still_sweeps_the_whole_range(self) -> None:
        """Step 1 finds the min and max, so it cannot start inside them."""
        w = self.win.step_widgets[1]
        self.assertEqual(
            (w["level_start"].value(), w["level_stop"].value(),
             w["level_step"].value()),
            (0, 1023, 64),
        )

    def test_the_daq_sweep_defaults(self) -> None:
        w = self.win.step_widgets["3c"]
        self.assertEqual(w["daq_channel"].text(), "ai1")
        self.assertAlmostEqual(w["daq_sample_rate"].value(), 1_000.0)
        self.assertAlmostEqual(w["daq_hold"].value(), 250.0)
        self.assertAlmostEqual(w["daq_duration"].value(), 1.0)

    def test_the_region_limit_is_off_but_pre_filled(self) -> None:
        """Unchecked, so the default span only applies once it is switched on."""
        w = self.win.step_widgets[2]
        self.assertFalse(w["region_check"].isChecked())
        self.assertFalse(w["region_start"].isEnabled())
        self.assertEqual(w["region_start"].value(), 600)
        self.assertEqual(w["region_end"].value(), 1700)
        self.assertIsNone(self.win._step_region(2))

    def test_the_region_reaches_the_sweep_once_it_is_switched_on(self) -> None:
        w = self.win.step_widgets[2]
        w["region_check"].setChecked(True)
        try:
            self.assertTrue(w["region_start"].isEnabled())
            self.assertEqual(self.win._step_region(2), (600, 1700))
        finally:
            w["region_check"].setChecked(False)

    def test_connecting_an_slm_leaves_the_step_2_region_alone(self) -> None:
        """The page's region is a choice; the panel width must not replace it.

        ``_on_info_read`` used to widen ``region_end`` to the full width
        whenever the limit was off, which reset the default to 1919 while
        leaving the start at 600.
        """
        from gui.app import MainWindow

        window = MainWindow()
        try:
            w = window.step_widgets[2]
            window._on_info_read((1920, 1200))
            self.assertEqual((w["region_start"].value(), w["region_end"].value()),
                             (600, 1700))
            w["region_check"].setChecked(True)
            self.assertEqual(window._step_region(2), (600, 1700))
        finally:
            window.close()

    def test_a_region_wider_than_the_panel_is_pulled_in(self) -> None:
        """Bounding it to the real width is the part that has to stay."""
        from gui.app import MainWindow

        window = MainWindow()
        try:
            w = window.step_widgets[2]
            w["region_check"].setChecked(True)
            w["region_end"].setValue(8000)
            window._on_info_read((1920, 1200))
            self.assertEqual(w["region_end"].value(), 1919)
            self.assertEqual(w["region_end"].maximum(), 1919)
        finally:
            window.close()

    def test_the_slm_control_page_still_scans_the_whole_panel(self) -> None:
        """Step 2's region default must not narrow the SLM Control scan."""
        from gui.app import MainWindow

        window = MainWindow()
        try:
            self.assertEqual(window.start_x_spin.value(), 0)
            window._on_info_read((1920, 1200))
            self.assertEqual(window.start_x_spin.value(), 0)
            self.assertEqual(window.end_x_spin.value(), 1919)
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
