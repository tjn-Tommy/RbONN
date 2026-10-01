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


class FakeDAQDriver:
    """Records what the page asked the card for, and hands back a flat trace."""

    def __init__(self, level: float = -0.010) -> None:
        self.is_connected = True
        self.level = level
        self.calls: list[dict] = []

    def read_waveform(self, **kwargs) -> np.ndarray:
        self.calls.append(kwargs)
        n = max(1, int(round(kwargs["sample_rate"] * kwargs["duration"])))
        return np.full(n, self.level)


class DAQMonitorContinuousTests(unittest.TestCase):
    """The DAQ Monitor page's readout-only continuous mode."""

    @classmethod
    def setUpClass(cls) -> None:
        from PyQt5 import QtWidgets

        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def _window(self):
        from gui.app import MainWindow

        return MainWindow()

    def test_history_defaults_to_one_second_and_starts_idle(self) -> None:
        window = self._window()
        try:
            self.assertEqual(window.daqmon_history.value(), 1.0)
            self.assertFalse(window.daqmon_cont_button.isChecked())
            self.assertFalse(window.daqmon_stop_button.isEnabled())
            self.assertTrue(window.daqmon_read_button.isEnabled())
        finally:
            window.close()

    def test_a_reading_acquires_history_plus_the_guard(self) -> None:
        """A 1 s History must average a full 1 s, so the guard is acquired on top."""
        from daq_module.controller import DAQController, settle_samples
        from gui.app import MainWindow

        window = self._window()
        try:
            driver = FakeDAQDriver(level=-0.010)
            daq = DAQController(driver=driver)
            cfg = window._daqmon_cfg_snapshot()
            payload = MainWindow._daqmon_acquire(daq, cfg, 1.0)

            n_guard = settle_samples(cfg["fs"], cfg["f_hw"], cfg["f_dig"])
            self.assertEqual(len(driver.calls), 1)
            self.assertAlmostEqual(driver.calls[0]["duration"],
                                   1.0 + n_guard / cfg["fs"])
            self.assertEqual(payload["n_settle"], n_guard)
            self.assertEqual(payload["kept"], 1.0)
            # kept span == History, to the sample
            self.assertEqual(payload["raw"].size - payload["n_settle"],
                             int(round(cfg["fs"] * 1.0)))
            self.assertAlmostEqual(payload["mean"], 0.010)   # inverted TIA sign
        finally:
            window.close()

    def test_a_retuned_history_reaches_the_next_reading(self) -> None:
        """The loop reads the republished snapshot, not the widgets."""
        window = self._window()
        try:
            window.daqmon_history.setValue(0.5)
            self.assertEqual(window._daqmon_stream_cfg["history"], 0.5)
            window.daqmon_history.setValue(2.0)
            self.assertEqual(window._daqmon_stream_cfg["history"], 2.0)
        finally:
            window.close()

    def test_buttons_follow_the_running_loop(self) -> None:
        window = self._window()
        try:
            window.daq_stream_stop_event = threading.Event()
            window._daqmon_set_running(continuous=True)
            self.assertFalse(window.daqmon_read_button.isEnabled())
            self.assertTrue(window.daqmon_cont_button.isChecked())
            self.assertTrue(window.daqmon_stop_button.isEnabled())
            # parameters stay editable: each reading re-reads them
            self.assertTrue(window.daqmon_cfg.isEnabled())

            window._daq_stream_stop()
            self.assertTrue(window.daq_stream_stop_event.is_set())

            window._daq_stream_done({"readings": 7})
            self.assertIn("7 readings", window.daqmon_status.text())
            self.assertTrue(window.daqmon_read_button.isEnabled())
            self.assertFalse(window.daqmon_stop_button.isEnabled())
            self.assertFalse(window.daqmon_cont_button.isChecked())
        finally:
            window.close()

    def test_a_one_shot_read_is_refused_while_the_loop_runs(self) -> None:
        window = self._window()
        try:
            window.daq_stream_stop_event = threading.Event()
            window._daq_waveform_read()
            self.assertIn("Continuous mode is running", window.daqmon_status.text())
            self.assertFalse(window.daq_stream_stop_event.is_set())
        finally:
            window.close()

    def test_a_starting_calibration_stops_the_loop(self) -> None:
        """A calibration owns the card, so the reading loop must let go."""
        window = self._window()
        try:
            window.daq_stream_stop_event = threading.Event()
            window._mark_tpa_calibration("step6", True)
            self.assertTrue(window.daq_stream_stop_event.is_set())
            window._mark_tpa_calibration("step6", False)
        finally:
            window.close()

    def test_disconnecting_the_card_stops_the_loop(self) -> None:
        window = self._window()
        try:
            window.daq_stream_stop_event = threading.Event()
            window._disconnect_daq()
            self.assertTrue(window.daq_stream_stop_event.is_set())
        finally:
            window.close()

    def test_a_reading_paints_the_readout_and_the_summary(self) -> None:
        from daq_module.controller import DAQController
        from gui.app import MainWindow

        window = self._window()
        try:
            daq = DAQController(driver=FakeDAQDriver(level=-0.010))
            payload = MainWindow._daqmon_acquire(daq, window._daqmon_cfg_snapshot(), 1.0)
            payload["count"] = 3
            for trace in ("times", "raw", "filtered"):
                payload.pop(trace)    # as the loop emits it: a reading, no traces
            window._on_daq_reading(payload)

            self.assertIn("+10.0000", window.daqmon_readout.text())
            self.assertIn("reading 3", window.daqmon_status.text())
            self.assertIn("warmup", window.daqmon_status.text())
        finally:
            window.close()


class FakePSUDriver:
    """Controller-level stand-in for the KWR103: answers queries, records sets.

    High-impedance load by default, as the PMT gain-control input is: the
    current limit never binds unless a test asks for it.
    """

    def __init__(self, load_ohms: float = 1e6, **levels) -> None:
        self.is_connected = True
        self.levels = {"VSET": 1.0, "ISET": 0.05, "OVP": 0.5, "OCP": 15.5}
        self.levels.update(levels)
        self.load_ohms = load_ohms
        self.out = False
        self.written: list[str] = []

    def transaction(self):
        from contextlib import nullcontext

        return nullcontext(self)

    def connect(self) -> None:
        self.is_connected = True

    def disconnect(self) -> None:
        self.is_connected = False

    def identify(self) -> str:
        return "KORAD KWR103 V2.3 SN:fake"

    def _measured(self) -> tuple[float, float]:
        if not self.out:
            return 0.0, 0.0
        vset, iset = self.levels["VSET"], self.levels["ISET"]
        amps = vset / self.load_ohms
        if amps > iset:
            return iset * self.load_ohms, iset
        return vset, amps

    def query(self, cmd: str) -> str:
        from psu_module.driver import PSUError

        if cmd == "*IDN?":
            return self.identify()
        if cmd == "OUT?":
            return "1" if self.out else "0"
        if cmd == "STATUS?":
            return "\x05"
        volts, amps = self._measured()
        if cmd == "VOUT?":
            return f"{volts:06.3f}"
        if cmd == "IOUT?":
            return f"{amps:06.3f}"
        if cmd.endswith("?") and cmd[:-1] in self.levels:
            return f"{self.levels[cmd[:-1]]:06.3f}"
        raise PSUError(f"no reply to {cmd!r}")

    def write(self, cmd: str) -> None:
        self.written.append(cmd)
        key, _, value = cmd.partition(":")
        if key == "OUT":
            self.out = value == "1"
        elif key in self.levels:
            self.levels[key] = float(value)


class PSUPageTests(unittest.TestCase):
    """The PMT Gain page: constant voltage into the gain-control input."""

    @classmethod
    def setUpClass(cls) -> None:
        from PyQt5 import QtWidgets

        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def _window(self):
        from gui.app import MainWindow

        return MainWindow()

    def _connected(self, window, **driver_kwargs):
        """Attach a fake supply exactly as a successful connect would."""
        from psu_module.controller import (
            PMT_CONTROL_I_MAX,
            PMT_CONTROL_V_MAX,
            PSUController,
        )

        driver = FakePSUDriver(**driver_kwargs)
        psu = PSUController(driver=driver, max_voltage=PMT_CONTROL_V_MAX,
                            max_current=PMT_CONTROL_I_MAX)
        window._on_psu_connected((psu, psu.identify(), psu.read_state()))
        return psu, driver

    def _pump(self, done, timeout: float = 5.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline and not done():
            self.app.processEvents()
        self.assertTrue(done(), "the PSU task did not finish in time")

    def _close(self, window) -> None:
        """Let this test's workers finish before the window goes away.

        _run_task delivers its result through a queued signal, so a worker still
        in flight when the window is destroyed lands in freed memory and takes
        the interpreter down in whichever test happens to pump events next.
        """
        deadline = time.time() + 5.0
        while time.time() < deadline and window._workers:
            self.app.processEvents()
        window.close()
        self.app.processEvents()

    def test_the_page_starts_idle_and_capped_at_the_pmt_spec(self) -> None:
        window = self._window()
        try:
            self.assertEqual(window.psu_vset.maximum(), 1.1)
            self.assertEqual(window.psu_iset.maximum(), 0.05)
            window.psu_vset.setValue(2.0)          # the spinbox will not go past
            self.assertEqual(window.psu_vset.value(), 1.1)
            for button in (window.psu_apply_button, window.psu_on_button,
                           window.psu_off_button, window.psu_monitor_button,
                           window.psu_monitor_stop_button):
                self.assertFalse(button.isEnabled())
        finally:
            self._close(window)

    def test_connecting_adopts_the_supplys_own_setpoints(self) -> None:
        # The PMT may already be running at a gain someone chose; the page must
        # show that, not overwrite it with a default.
        window = self._window()
        try:
            window.psu_vset.setValue(0.200)
            self._connected(window, VSET=0.750, ISET=0.030)
            self.assertAlmostEqual(window.psu_vset.value(), 0.750)
            self.assertAlmostEqual(window.psu_iset.value(), 0.030)
            self.assertIn("0.750 V", window.psu_page_status.text())
            self.assertTrue(window.psu_apply_button.isEnabled())
        finally:
            self._close(window)

    def test_apply_sets_the_limit_before_the_voltage(self) -> None:
        window = self._window()
        try:
            _, driver = self._connected(window)
            driver.written.clear()
            window.psu_vset.setValue(0.850)
            window.psu_iset.setValue(0.020)
            window._psu_apply()
            self._pump(lambda: not window._psu_busy)
            self.assertEqual(driver.written, ["ISET:0.020", "VSET:0.850"])
            self.assertAlmostEqual(driver.levels["VSET"], 0.850)
            self.assertIn("0.850 V", window.psu_page_status.text())
        finally:
            self._close(window)

    def test_output_on_is_gated_by_the_confirmation(self) -> None:
        from PyQt5 import QtWidgets

        window = self._window()
        try:
            _, driver = self._connected(window)
            with patch.object(QtWidgets.QMessageBox, "question",
                              return_value=QtWidgets.QMessageBox.No):
                window._psu_output_on()
            self.assertFalse(driver.out)
            self.assertIn("left off", window.psu_page_status.text())

            with patch.object(QtWidgets.QMessageBox, "question",
                              return_value=QtWidgets.QMessageBox.Yes):
                window._psu_output_on()
                self._pump(lambda: not window._psu_busy)
            self.assertTrue(driver.out)
            self.assertIn("Output: ON", window.psu_output_label.text())
        finally:
            self._close(window)

    def test_the_confirmation_warns_about_an_ovp_below_the_setpoint(self) -> None:
        # The supply shipped with OVP 0.5 V under a 1.0 V setpoint, which would
        # trip the moment the output went live.
        from PyQt5 import QtWidgets

        window = self._window()
        try:
            self._connected(window, VSET=1.000, OVP=0.500)
            self.assertIn("would trip", window.psu_prot_label.text())
            with patch.object(QtWidgets.QMessageBox, "question",
                              return_value=QtWidgets.QMessageBox.No) as ask:
                window._psu_output_on()
            self.assertIn("OVP is 0.500 V", ask.call_args[0][2])
        finally:
            self._close(window)

    def test_the_confirmation_flags_an_unapplied_spinbox(self) -> None:
        from PyQt5 import QtWidgets

        window = self._window()
        try:
            self._connected(window, VSET=0.400)
            window.psu_vset.setValue(0.900)       # typed but never applied
            with patch.object(QtWidgets.QMessageBox, "question",
                              return_value=QtWidgets.QMessageBox.No) as ask:
                window._psu_output_on()
            text = ask.call_args[0][2]
            self.assertIn("0.400 V", text)         # what will actually be applied
            self.assertIn("not applied", text)
        finally:
            self._close(window)

    def test_cc_is_reported_as_a_fault(self) -> None:
        # A high-impedance control input in CC means shorted or mis-wired.
        window = self._window()
        try:
            psu, driver = self._connected(window, load_ohms=1.0)   # short
            driver.levels["VSET"] = 0.500
            driver.out = True
            window._psu_show_state(psu.read_state())
            self.assertIn("CC", window.psu_mode_label.text())
            self.assertEqual(window.psu_mode_label.property("status"), "error")
        finally:
            self._close(window)

    def test_a_poll_never_moves_a_spinbox_the_user_is_editing(self) -> None:
        window = self._window()
        try:
            psu, driver = self._connected(window, VSET=0.400)
            window.psu_vset.setValue(0.900)        # user is mid-edit
            driver.out = True
            window._on_psu_sample(psu.read_state())
            self.assertAlmostEqual(window.psu_vset.value(), 0.900)
            self.assertIn("0.400 V", window.psu_readout.text())  # but the readout
        finally:                                                  # shows the truth
            self._close(window)

    def test_setpoints_stay_usable_while_the_monitor_runs(self) -> None:
        # The port is locked per exchange, so a poll cannot swallow a set.
        window = self._window()
        try:
            self._connected(window)
            window.psu_stop_event = threading.Event()
            window._psu_refresh_buttons()
            self.assertTrue(window.psu_apply_button.isEnabled())
            self.assertTrue(window.psu_on_button.isEnabled())
            self.assertTrue(window.psu_monitor_stop_button.isEnabled())
            self.assertFalse(window.psu_monitor_button.isEnabled())
            self.assertFalse(window.psu_period.isEnabled())

            window._psu_monitor_stop()
            self.assertTrue(window.psu_stop_event.is_set())
            window._psu_monitor_done()
            self.assertIsNone(window.psu_stop_event)
            self.assertTrue(window.psu_monitor_button.isEnabled())
            self.assertFalse(window.psu_monitor_stop_button.isEnabled())
        finally:
            self._close(window)

    def test_disconnect_waits_for_the_monitor_to_unwind(self) -> None:
        window = self._window()
        try:
            self._connected(window)
            window.psu_stop_event = threading.Event()
            window._disconnect_psu()
            self.assertTrue(window.psu_stop_event.is_set())
            self.assertIsNotNone(window.psu_controller)   # not yet: the loop owns it

            window._psu_monitor_done()                    # loop finished
            self.assertIsNone(window.psu_controller)
            self.assertTrue(window.psu_connect_button.isEnabled())
            self.assertIn("left exactly as it was", window.psu_page_status.text())
        finally:
            self._close(window)

    def test_closing_the_window_stops_the_monitor(self) -> None:
        window = self._window()
        stop = threading.Event()
        window.psu_stop_event = stop
        self._close(window)
        self.assertTrue(stop.is_set())

class Step8ProbeTests(unittest.TestCase):
    """The Step 8 tab's sweep band, its levels read-out and its one-level probe.

    None of this touches hardware: ``full_scale`` is a prediction, so the panel
    can say where Y_max lands and what the ladder will do with it the moment a
    step-7 file is named. The button is what needs the bench, and it stays
    disabled until there is a forward model to drive -- otherwise the only
    failure mode left is a dialog after the click.

    The load-bearing test is :meth:`test_the_predicted_millivolts_are_shown_
    before_the_press`. The probe can be asked for any alpha in [0, 1], and at
    the present gain ``alpha = 1`` is ~1534 mV into a 1 V amplifier; the caption
    is what stops that being one keystroke away unannounced.
    """

    REPO = Path(__file__).resolve().parents[1]
    CANDIDATES = (
        "src/calib_data/run_0928/calib.json",
        "src/calib_data/run_0916_10pairs_w15g35/calib.json",
        "src/calib_data/run_0911_10pairs_w20g30/calib.json",
    )

    @classmethod
    def setUpClass(cls) -> None:
        from PyQt5 import QtWidgets

        from gui.app import MainWindow

        cls.step7 = next((cls.REPO / c for c in cls.CANDIDATES
                          if (cls.REPO / c).is_file()), None)
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls.win = MainWindow()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.win.close()

    def tearDown(self) -> None:
        self.win.step8_step7_edit.setText("")

    def test_the_button_is_dead_until_a_step_7_is_named(self) -> None:
        self.assertFalse(self.win.step8_probe_button.isEnabled())

    def test_the_old_headroom_button_is_gone(self) -> None:
        """It could only drive alpha = 1, which now clips by construction."""
        self.assertFalse(hasattr(self.win, "step8_headroom_button"))
        self.assertFalse(hasattr(self.win, "step8_headroom_note"))

    def test_the_page_shows_no_prose(self) -> None:
        """The tab carries numbers, and nothing that is not a number.

        Every caption here was one somebody had to scroll past on every visit.
        The reasoning still exists -- in calib_step8's docstring and in this
        page's own comments -- where it costs no screen.
        """
        self.assertEqual(self.win.step8_levels_note.text(), "")
        self.assertEqual(self.win.step8_probe_pred.text(), "")
        self.assertEqual(self.win.step8_result_text.placeholderText(), "")
        self.assertEqual(self.win.step8_plot_label.text(), "")
        self.assertFalse(hasattr(self.win, "step8_daq_note"))
        self.assertFalse(hasattr(self.win, "step8_step7_note"))

    def test_the_button_says_it_reads(self) -> None:
        """It displays the pattern AND takes a reading; the label must say so."""
        self.assertIn("Read", self.win.step8_probe_button.text())
        self.assertIn("Display", self.win.step8_probe_button.text())

    def test_the_alpha_box_spans_the_whole_scale(self) -> None:
        box = self.win.step8_probe_alpha
        self.assertAlmostEqual(box.minimum(), 0.0)
        self.assertAlmostEqual(box.maximum(), 1.0)

    def test_a_path_that_is_not_a_file_leaves_it_dead(self) -> None:
        self.win.step8_step7_edit.setText("no/such/calib_step7_result.json")
        self.assertFalse(self.win.step8_probe_button.isEnabled())

    def test_a_json_that_is_not_a_forward_model_says_so(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "calib_step7_result_bogus.json"
            bad.write_text(json.dumps({"hello": "world"}), encoding="utf-8")
            self.win.step8_step7_edit.setText(str(bad))
            self.assertFalse(self.win.step8_probe_button.isEnabled())
            # Feedback goes to the status line; the note stays numeric.
            self.assertIn("Cannot read the forward model",
                          self.win.step8_status.text())
            self.assertEqual(self.win.step8_levels_note.text(), "")

    def test_a_real_step_7_enables_it_and_quotes_y_max(self) -> None:
        if self.step7 is None:
            self.skipTest("no step-7 result in the tree")
        self.win.step8_step7_edit.setText(str(self.step7))
        self.assertTrue(self.win.step8_probe_button.isEnabled())
        note = self.win.step8_levels_note.text()
        self.assertIn("Y_max", note)
        self.assertIn("4-bit step", note)

    def test_the_note_matches_the_module(self) -> None:
        """The panel must not compute its own Y_max -- one number, one owner."""
        if self.step7 is None:
            self.skipTest("no step-7 result in the tree")
        from calibration_module.fit.verify import full_scale, load_forward_model
        from calibration_module.steps import calib_step8 as STEP8

        models, phases, _ = load_forward_model(self.step7)
        fs = full_scale(models, phases, sorted(models)).fs
        self.win.step8_step7_edit.setText(str(self.step7))
        self.assertAlmostEqual(self.win.step8_y_max, fs)
        note = self.win.step8_levels_note.text()
        self.assertIn(f"{fs * 1e3:.1f} mV", note)
        lv = STEP8.ladder_levels(fs)
        self.assertIn(f"{lv['cap']:.4f}", self.win.step8_sweep_max_alpha.text())

    def test_the_default_alpha_is_the_ladders_own_cap(self) -> None:
        """The brightest read the sweep will take is the one worth probing."""
        if self.step7 is None:
            self.skipTest("no step-7 result in the tree")
        from calibration_module.fit.verify import full_scale, load_forward_model
        from calibration_module.steps import calib_step8 as STEP8

        models, phases, _ = load_forward_model(self.step7)
        fs = full_scale(models, phases, sorted(models)).fs
        self.win.step8_step7_edit.setText(str(self.step7))
        self.assertAlmostEqual(self.win.step8_probe_alpha.value(),
                               round(STEP8.ladder_levels(fs)["cap"], 4),
                               places=4)

    def test_the_predicted_millivolts_are_shown_before_the_press(self) -> None:
        """alpha IS the fraction of Y_max, so this needs no solve and no bench.

        And it has to warn, because the box goes to 1: at the present gain that
        is ~1534 mV into a 1 V amplifier.
        """
        if self.step7 is None:
            self.skipTest("no step-7 result in the tree")
        from calibration_module.fit.verify import full_scale, load_forward_model
        from calibration_module.steps import calib_step8 as STEP8

        models, phases, _ = load_forward_model(self.step7)
        fs = full_scale(models, phases, sorted(models)).fs
        self.win.step8_step7_edit.setText(str(self.step7))
        for alpha in (0.05, 0.25, 0.5, 1.0):
            self.win.step8_probe_alpha.setValue(alpha)
            text = self.win.step8_probe_pred.text()
            self.assertIn(f"{alpha * fs * 1e3:.1f} mV", text)
            v = alpha * fs
            if v > STEP8.AMP_LIMIT_V:
                self.assertIn("OVER", text)
            elif v > STEP8.LADDER_CEILING_V:
                self.assertIn("above Sweep max", text)

    def test_a_running_sweep_takes_the_probe_away(self) -> None:
        if self.step7 is None:
            self.skipTest("no step-7 result in the tree")
        self.win.step8_step7_edit.setText(str(self.step7))
        self.assertTrue(self.win.step8_probe_button.isEnabled())
        self.win._step8_set_running(True)
        try:
            self.assertFalse(self.win.step8_probe_button.isEnabled())
            # and the alpha cannot be moved under a running sweep either
            self.assertFalse(self.win.step8_probe_alpha.isEnabled())
        finally:
            self.win._step8_set_running(False)
        self.assertTrue(self.win.step8_probe_button.isEnabled())

    # ---- the sweep band ------------------------------------------------

    def test_the_band_boxes_start_at_the_modules_own_constants(self) -> None:
        """One owner for the defaults: a GUI copy is a second place to be wrong."""
        from calibration_module.steps import calib_step8 as STEP8

        for box, volts in ((self.win.step8_sweep_max, STEP8.LADDER_CEILING_V),
                           (self.win.step8_band_floor, STEP8.LADDER_FLOOR_V),
                           (self.win.step8_sweep_min, STEP8.LADDER_LOW_V)):
            self.assertAlmostEqual(box.value(), volts * 1e3, places=3)
            self.assertIn("mV", box.suffix())
        # and no box can ask for more than the amplifier can hold
        self.assertAlmostEqual(self.win.step8_sweep_max.maximum(),
                               STEP8.AMP_LIMIT_V * 1e3, places=3)

    def test_the_alpha_column_is_blank_until_a_step_7_is_loaded(self) -> None:
        """alpha is mV / Y_max, and there is no Y_max before the file."""
        for cap in (self.win.step8_sweep_max_alpha,
                    self.win.step8_band_floor_alpha,
                    self.win.step8_sweep_min_alpha):
            self.assertEqual(cap.text(), "")

    def test_the_alpha_column_follows_the_boxes(self) -> None:
        if self.step7 is None:
            self.skipTest("no step-7 result in the tree")
        from calibration_module.steps import calib_step8 as STEP8

        self.win.step8_step7_edit.setText(str(self.step7))
        fs = self.win.step8_y_max
        try:
            for mv in (450.0, 300.0, 700.0):
                self.win.step8_sweep_max.setValue(mv)
                lv = STEP8.ladder_levels(fs, ceiling_v=mv / 1e3,
                                         floor_v=self.win.step8_band_floor.value() / 1e3,
                                         low_v=self.win.step8_sweep_min.value() / 1e3)
                self.assertIn(f"{lv['cap']:.4f}",
                              self.win.step8_sweep_max_alpha.text())
        finally:
            self.win.step8_sweep_max.setValue(1e3 * STEP8.LADDER_CEILING_V)

    def test_a_clamped_box_says_what_will_actually_be_driven(self) -> None:
        """ladder_levels clamps rather than refuses, so the clamp has to show.

        A band floor above the ceiling is not an error -- it is 30 rungs at a
        level nobody asked for, and a box that still reads 900 mV.
        """
        if self.step7 is None:
            self.skipTest("no step-7 result in the tree")
        from calibration_module.steps import calib_step8 as STEP8

        self.win.step8_step7_edit.setText(str(self.step7))
        fs = self.win.step8_y_max
        try:
            self.win.step8_band_floor.setValue(900.0)
            text = self.win.step8_band_floor_alpha.text()
            lv = STEP8.ladder_levels(fs, floor_v=0.900)
            self.assertLess(lv["floor"] * fs, 0.900)          # it was clamped
            self.assertIn(f"{lv['floor'] * fs * 1e3:.1f} mV", text)
            self.assertIn("clamped", text)
        finally:
            self.win.step8_band_floor.setValue(1e3 * STEP8.LADDER_FLOOR_V)

    def test_a_dim_model_is_not_called_clamped_at_the_ceiling(self) -> None:
        """Under the ceiling the atoms own the top; that is not a clamp.

        Y_max is set directly because no step-7 in the tree is dim enough: at
        the present gain it is ~1 V, past the amplifier, which is the whole
        reason the ceiling exists.
        """
        saved = getattr(self.win, "step8_y_max", None)
        try:
            self.win.step8_y_max = 0.300
            self.win._step8_refresh_levels()
            text = self.win.step8_sweep_max_alpha.text()
            self.assertIn("1.0000", text)
            self.assertIn("Y_max is below it", text)
            self.assertNotIn("clamped", text)
            self.assertIn("not capped", self.win.step8_levels_note.text())
        finally:
            self.win.step8_y_max = saved
            self.win._step8_refresh_levels()

    def test_clearing_the_path_takes_the_alphas_with_it(self) -> None:
        """A stale alpha beside a live box is the one thing this cannot show.

        Blanking the note is not enough: alpha is mV / Y_max, so dropping Y_max
        drops the whole column.
        """
        if self.step7 is None:
            self.skipTest("no step-7 result in the tree")
        self.win.step8_step7_edit.setText(str(self.step7))
        self.assertNotEqual(self.win.step8_sweep_max_alpha.text(), "")
        self.win.step8_step7_edit.setText("")
        self.assertIsNone(self.win.step8_y_max)
        for cap in (self.win.step8_sweep_max_alpha,
                    self.win.step8_band_floor_alpha,
                    self.win.step8_sweep_min_alpha):
            self.assertEqual(cap.text(), "")
        self.assertEqual(self.win.step8_levels_note.text(), "")

    def test_sweep_min_goes_away_with_the_model_checks(self) -> None:
        """It is the checks' floor; with no checks it bounds nothing."""
        from calibration_module.steps import calib_step8 as STEP8

        try:
            self.win.step8_low.setValue(0)
            self.assertFalse(self.win.step8_sweep_min.isEnabled())
            self.assertEqual(self.win.step8_sweep_min_alpha.text(), "")
            self.win.step8_low.setValue(STEP8.LADDER_LOW)
            self.assertTrue(self.win.step8_sweep_min.isEnabled())
        finally:
            self.win.step8_low.setValue(STEP8.LADDER_LOW)

    def test_a_running_sweep_freezes_the_band(self) -> None:
        self.win._step8_set_running(True)
        try:
            for box in (self.win.step8_ladder, self.win.step8_low,
                        self.win.step8_sweep_max, self.win.step8_band_floor,
                        self.win.step8_sweep_min):
                self.assertFalse(box.isEnabled())
        finally:
            self.win._step8_set_running(False)
        self.assertTrue(self.win.step8_sweep_max.isEnabled())

    def test_the_volts_reach_the_module_as_the_module_spells_them(self) -> None:
        """_step8_levels_v is what build() and probe_verdict are given."""
        from calibration_module.steps import calib_step8 as STEP8

        volts = self.win._step8_levels_v()
        self.assertEqual(set(volts), {"ceiling_v", "floor_v", "low_v"})
        self.assertAlmostEqual(volts["ceiling_v"], STEP8.LADDER_CEILING_V)
        # spelled so they can be splatted straight into the module's own call
        STEP8.ladder_levels(1.0, **volts)

    def test_a_zero_alpha_is_refused_without_touching_the_bench(self) -> None:
        """A zero drive is the dark, which every read already takes first."""
        if self.step7 is None:
            self.skipTest("no step-7 result in the tree")
        self.win.step8_step7_edit.setText(str(self.step7))
        self.win.step8_probe_alpha.setValue(0.0)
        self.win._step8_probe()
        self.assertIn("above 0", self.win.step8_status.text())


class Step8SaveTests(unittest.TestCase):
    """Where a step-8 sweep goes, and when.

    Steps 6 and 7 write nothing until Save.  Step 8 used to write its CSV, its
    result JSON and its plot into ``calib_data`` on every press, which left a
    run somebody abandoned indistinguishable from one they meant to keep.  It
    stages instead, and Save is the only thing that reaches a directory anybody
    chose.

    The fit is not re-tested here -- ``test_calib_step8`` owns it.  What is
    tested is the contract this page hands it: which CSV, which step-7 model,
    and which directory.
    """

    REPO = Path(__file__).resolve().parents[1]

    @classmethod
    def setUpClass(cls) -> None:
        from PyQt5 import QtWidgets

        from gui.app import MainWindow

        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls.win = MainWindow()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.win.close()

    def setUp(self) -> None:
        self.win.step8_last = None
        self.win._step8_refresh_buttons()

    def tearDown(self) -> None:
        self.win.step8_step7_edit.setText("")
        self.win.step8_last = None

    @staticmethod
    def _result(csv: Path) -> dict:
        """A fit, with every key the page reads off one."""
        stem = csv.stem.replace("calib_step8_", "")
        return {"y_sat_v": 5.06, "sigma_v": 0.12, "lo_v": 4.9, "hi_v": 5.2,
                "unbounded": False, "n": 38, "n_ladder": 30,
                "rmse_v": 2.0e-3, "rmse_linear_v": 8.0e-3,
                "gain_lever": 15.0, "full_scale": {},
                "png": str(csv.with_name(f"{csv.stem}_ysat.png")),
                "result_json": str(csv.with_name(
                    f"calib_step8_result_{stem}.json"))}

    def _payload(self, csv: Path, step7: Path | None = None) -> dict:
        return {"status": "complete", "csv": csv,
                "step7": Path("step7.json") if step7 is None else step7,
                "result": self._result(csv)}

    def _staged(self, name: str = "calib_step8_0930_1209.csv") -> Path:
        """A CSV sitting in staging, as a finished sweep leaves one."""
        csv = self.win._step8_staging() / name
        csv.write_text("block,driven,mean_v\n0,(1, 2),0.42\n", encoding="utf-8")
        return csv

    # ---- the buttons ---------------------------------------------------
    def test_the_refit_button_became_a_load_button(self) -> None:
        """Re-fitting a CSV is loading one; steps 6 and 7 already call it that."""
        self.assertFalse(hasattr(self.win, "step8_refit_button"))
        self.assertFalse(hasattr(self.win, "_step8_refit"))
        self.assertIn("Load", self.win.step8_load_button.text())
        self.assertIn("Save", self.win.step8_save_button.text())

    def test_save_is_dead_until_a_sweep_has_been_fitted(self) -> None:
        self.assertFalse(self.win.step8_save_button.isEnabled())
        self.win._step8_finished(self._payload(self._staged()))
        self.assertTrue(self.win.step8_save_button.isEnabled())

    def test_a_running_sweep_takes_save_away(self) -> None:
        self.win._step8_finished(self._payload(self._staged()))
        self.win._step8_set_running(True)
        self.assertFalse(self.win.step8_save_button.isEnabled())
        self.assertFalse(self.win.step8_load_button.isEnabled())
        self.win._step8_set_running(False)
        self.assertTrue(self.win.step8_save_button.isEnabled())

    def test_a_run_that_read_nothing_leaves_nothing_to_save(self) -> None:
        self.win._step8_finished({"status": "partial (stopped)", "csv": None})
        self.assertIsNone(self.win.step8_last)
        self.assertFalse(self.win.step8_save_button.isEnabled())
        self.assertIn("nothing written", self.win.step8_status.text())

    # ---- staging -------------------------------------------------------
    def test_the_staging_directory_is_outside_the_repo(self) -> None:
        """A run nobody kept must not have to be recognised and deleted later."""
        from calibration_module.steps import calib_step8 as STEP8

        out = self.win._step8_staging()
        self.assertTrue(out.is_dir())
        self.assertNotEqual(out.resolve(), Path(STEP8.OUT_DIR).resolve())
        self.assertNotIn(self.REPO.resolve(), out.resolve().parents)

    def test_a_finished_run_says_it_was_only_staged(self) -> None:
        self.win._step8_finished(self._payload(self._staged()))
        text = self.win.step8_status.text()
        self.assertIn("staged as calib_step8_0930_1209.csv", text)
        self.assertIn("press Save", text)

    def test_a_csv_from_anywhere_else_is_not_called_staged(self) -> None:
        """A loaded file is already somewhere; only a fresh sweep is staged."""
        with tempfile.TemporaryDirectory() as tmp:
            csv = Path(tmp) / "calib_step8_0101_0101.csv"
            csv.write_text("x\n", encoding="utf-8")
            self.win._step8_finished(self._payload(csv))
        text = self.win.step8_status.text()
        self.assertIn("calib_step8_0101_0101.csv", text)
        self.assertNotIn("staged", text)

    def test_a_load_fits_into_staging_and_not_beside_the_csv(self) -> None:
        """Opening a CSV is not permission to write next to it."""
        from calibration_module.steps import calib_step8 as STEP8

        seen: dict = {}

        def fake_fit(csv, step7, *, out_dir=None, log=print, **kw):
            seen.update(csv=Path(csv), step7=Path(step7), out_dir=Path(out_dir))
            return self._result(Path(csv))

        with tempfile.TemporaryDirectory() as tmp:
            step7 = Path(tmp) / "calib_step7_result_0101_0101.json"
            step7.write_text("{}", encoding="utf-8")
            csv = Path(tmp) / "calib_step8_0101_0101.csv"
            csv.write_text("x\n", encoding="utf-8")
            self.win.step8_step7_edit.setText(str(step7))
            with patch("PyQt5.QtWidgets.QFileDialog.getOpenFileName",
                       return_value=(str(csv), "")), \
                 patch.object(STEP8, "fit_and_write", fake_fit), \
                 patch.object(self.win, "_run_task",
                              lambda _l, work, done, _f: done(work())):
                self.win._step8_load()
            self.assertEqual(seen["csv"], csv)
            self.assertEqual(seen["step7"], step7)
            self.assertEqual(seen["out_dir"].resolve(),
                             self.win._step8_staging().resolve())
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()),
                             sorted([csv.name, step7.name]))

    # ---- the save ------------------------------------------------------
    def test_save_copies_the_csv_and_writes_the_fit_beside_it(self) -> None:
        """The JSON is WRITTEN at the destination, not copied from staging.

        It records the CSV and the PNG it came from by absolute path, so a
        copied one would point back into a temp directory.
        """
        from calibration_module.steps import calib_step8 as STEP8

        src = self._staged()
        self.win._step8_finished(self._payload(src, step7=Path("model.json")))
        seen: dict = {}

        def fake_fit(csv, step7, *, out_dir=None, log=print, **kw):
            seen.update(csv=Path(csv), step7=Path(step7), out_dir=Path(out_dir))
            return self._result(Path(csv))

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "run_0930" / src.name
            with patch("PyQt5.QtWidgets.QFileDialog.getSaveFileName",
                       return_value=(str(dest), "")), \
                 patch.object(STEP8, "fit_and_write", fake_fit):
                self.win._step8_save()
            self.assertTrue(dest.is_file())
            self.assertEqual(dest.read_text(encoding="utf-8"),
                             src.read_text(encoding="utf-8"))
            self.assertEqual(seen["csv"], dest)
            self.assertEqual(seen["out_dir"], dest.parent)
            # not whatever the box holds -- the model this fit was made against
            self.assertEqual(seen["step7"], Path("model.json"))
            text = self.win.step8_status.text()
            self.assertIn("Saved", text)
            stamp = dest.stem.replace("calib_step8_", "")
            for name in (dest.name, f"calib_step8_result_{stamp}.json",
                         f"{dest.stem}_ysat.png"):
                self.assertIn(name, text)

    def test_save_ignores_a_step_7_box_edited_after_the_fit(self) -> None:
        """What was fitted is what gets recorded, whatever the box says now."""
        from calibration_module.steps import calib_step8 as STEP8

        src = self._staged()
        self.win._step8_finished(self._payload(src, step7=Path("fitted.json")))
        self.win.step8_step7_edit.setText("something_else.json")
        seen: dict = {}

        def fake_fit(csv, step7, *, out_dir=None, log=print, **kw):
            seen["step7"] = Path(step7)
            return self._result(Path(csv))

        with tempfile.TemporaryDirectory() as tmp:
            with patch("PyQt5.QtWidgets.QFileDialog.getSaveFileName",
                       return_value=(str(Path(tmp) / src.name), "")), \
                 patch.object(STEP8, "fit_and_write", fake_fit):
                self.win._step8_save()
        self.assertEqual(seen["step7"], Path("fitted.json"))

    def test_a_second_save_copies_nothing_because_it_is_already_there(self) -> None:
        """After a save the run LIVES at the destination; staging is history."""
        from calibration_module.steps import calib_step8 as STEP8

        src = self._staged()
        self.win._step8_finished(self._payload(src))

        def fake_fit(csv, step7, *, out_dir=None, log=print, **kw):
            return self._result(Path(csv))

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / src.name
            with patch("PyQt5.QtWidgets.QFileDialog.getSaveFileName",
                       return_value=(str(dest), "")), \
                 patch.object(STEP8, "fit_and_write", fake_fit):
                self.win._step8_save()
                self.assertEqual(Path(self.win.step8_last["csv"]), dest)
                src.unlink()                  # the staged copy is gone now
                self.win._step8_save()        # and this must still work
            self.assertIn("Saved", self.win.step8_status.text())
            self.assertTrue(dest.is_file())

    def test_a_staged_run_is_offered_calib_data(self) -> None:
        """Where a sweep goes if nobody says otherwise."""
        from calibration_module.steps import calib_step8 as STEP8

        src = self._staged()
        self.win._step8_finished(self._payload(src))
        with patch("PyQt5.QtWidgets.QFileDialog.getSaveFileName",
                   return_value=("", "")) as dialog:
            self.win._step8_save()
        self.assertEqual(Path(dialog.call_args[0][2]),
                         Path(STEP8.OUT_DIR) / src.name)

    def test_a_run_that_already_lives_somewhere_is_offered_back_there(self) -> None:
        """A loaded CSV saves beside itself, not into calib_data."""
        with tempfile.TemporaryDirectory() as tmp:
            csv = Path(tmp) / "calib_step8_0101_0101.csv"
            csv.write_text("x\n", encoding="utf-8")
            self.win._step8_finished(self._payload(csv))
            with patch("PyQt5.QtWidgets.QFileDialog.getSaveFileName",
                       return_value=("", "")) as dialog:
                self.win._step8_save()
            self.assertEqual(Path(dialog.call_args[0][2]), csv)

    def test_a_cancelled_save_writes_nothing(self) -> None:
        from calibration_module.steps import calib_step8 as STEP8

        self.win._step8_finished(self._payload(self._staged()))

        def fail(*_a, **_kw):
            raise AssertionError("a cancelled dialog must not write")

        with patch("PyQt5.QtWidgets.QFileDialog.getSaveFileName",
                   return_value=("", "")), \
             patch.object(STEP8, "fit_and_write", fail):
            self.win._step8_save()
        self.assertTrue(self.win.step8_save_button.isEnabled())

    def test_a_failed_save_says_so_and_leaves_the_button_alive(self) -> None:
        """A full disk or a read-only folder must not swallow the run."""
        from calibration_module.steps import calib_step8 as STEP8

        self.win._step8_finished(self._payload(self._staged()))

        def boom(*_a, **_kw):
            raise OSError("no space left on device")

        with tempfile.TemporaryDirectory() as tmp:
            with patch("PyQt5.QtWidgets.QFileDialog.getSaveFileName",
                       return_value=(str(Path(tmp) / "x.csv"), "")), \
                 patch.object(STEP8, "fit_and_write", boom):
                self.win._step8_save()
        self.assertIn("Save failed", self.win.step8_status.text())
        self.assertTrue(self.win.step8_save_button.isEnabled())


if __name__ == "__main__":
    unittest.main()
