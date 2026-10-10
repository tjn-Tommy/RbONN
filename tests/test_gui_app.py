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

    #: Everything setUp has to blank, so a stale label cannot pass a later test.
    _LABELS = ("psu_page_status", "psu_readout", "psu_mode_label",
               "psu_prot_label", "psu_output_label")

    @classmethod
    def setUpClass(cls) -> None:
        from PyQt5 import QtWidgets

        from gui.app import MainWindow

        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        # One window for the whole class: building and closing one costs ~1.2 s,
        # which was most of this file's runtime. setUp below puts the page back
        # to disconnected-and-blank, so the tests stay independent.
        cls.win = MainWindow()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._drain(cls.win, cls.app)
        cls.win.close()
        cls.app.processEvents()

    @staticmethod
    def _drain(window, app) -> None:
        """Let any worker still in flight finish.

        _run_task delivers its result through a queued signal, so a worker still
        in flight when the window is destroyed lands in freed memory and takes
        the interpreter down in whichever test happens to pump events next.
        """
        deadline = time.time() + 5.0
        while time.time() < deadline and window._workers:
            app.processEvents()

    def setUp(self) -> None:
        w = self.win
        w.psu_stop_event = None
        w._psu_disconnect_pending = False
        w._psu_busy = False
        if w.psu_controller is not None:
            w._psu_do_disconnect()      # the direct path: no monitor to unwind
        self._drain(w, self.app)
        w.psu_vset.setValue(0.0)
        w.psu_iset.setValue(0.0)
        for name in self._LABELS:
            getattr(w, name).setText("")
        w._psu_refresh_buttons()

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

    def test_the_page_starts_idle_and_capped_at_the_pmt_spec(self) -> None:
        window = self.win
        self.assertEqual(window.psu_vset.maximum(), 1.1)
        self.assertEqual(window.psu_iset.maximum(), 0.05)
        window.psu_vset.setValue(2.0)          # the spinbox will not go past
        self.assertEqual(window.psu_vset.value(), 1.1)
        for button in (window.psu_apply_button, window.psu_on_button,
                       window.psu_off_button, window.psu_monitor_button,
                       window.psu_monitor_stop_button):
            self.assertFalse(button.isEnabled())

    def test_connecting_adopts_the_supplys_own_setpoints(self) -> None:
        # The PMT may already be running at a gain someone chose; the page must
        # show that, not overwrite it with a default.
        window = self.win
        window.psu_vset.setValue(0.200)
        self._connected(window, VSET=0.750, ISET=0.030)
        self.assertAlmostEqual(window.psu_vset.value(), 0.750)
        self.assertAlmostEqual(window.psu_iset.value(), 0.030)
        self.assertIn("0.750 V", window.psu_page_status.text())
        self.assertTrue(window.psu_apply_button.isEnabled())

    def test_apply_sets_the_limit_before_the_voltage(self) -> None:
        window = self.win
        _, driver = self._connected(window)
        driver.written.clear()
        window.psu_vset.setValue(0.850)
        window.psu_iset.setValue(0.020)
        window._psu_apply()
        self._pump(lambda: not window._psu_busy)
        self.assertEqual(driver.written, ["ISET:0.020", "VSET:0.850"])
        self.assertAlmostEqual(driver.levels["VSET"], 0.850)
        self.assertIn("0.850 V", window.psu_page_status.text())

    def test_output_on_is_gated_by_the_confirmation(self) -> None:
        from PyQt5 import QtWidgets

        window = self.win
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

    def test_the_confirmation_warns_about_an_ovp_below_the_setpoint(self) -> None:
        # The supply shipped with OVP 0.5 V under a 1.0 V setpoint, which would
        # trip the moment the output went live.
        from PyQt5 import QtWidgets

        window = self.win
        self._connected(window, VSET=1.000, OVP=0.500)
        self.assertIn("would trip", window.psu_prot_label.text())
        with patch.object(QtWidgets.QMessageBox, "question",
                          return_value=QtWidgets.QMessageBox.No) as ask:
            window._psu_output_on()
        self.assertIn("OVP is 0.500 V", ask.call_args[0][2])

    def test_the_confirmation_flags_an_unapplied_spinbox(self) -> None:
        from PyQt5 import QtWidgets

        window = self.win
        self._connected(window, VSET=0.400)
        window.psu_vset.setValue(0.900)       # typed but never applied
        with patch.object(QtWidgets.QMessageBox, "question",
                          return_value=QtWidgets.QMessageBox.No) as ask:
            window._psu_output_on()
        text = ask.call_args[0][2]
        self.assertIn("0.400 V", text)         # what will actually be applied
        self.assertIn("not applied", text)

    def test_cc_is_reported_as_a_fault(self) -> None:
        # A high-impedance control input in CC means shorted or mis-wired.
        window = self.win
        psu, driver = self._connected(window, load_ohms=1.0)   # short
        driver.levels["VSET"] = 0.500
        driver.out = True
        window._psu_show_state(psu.read_state())
        self.assertIn("CC", window.psu_mode_label.text())
        self.assertEqual(window.psu_mode_label.property("status"), "error")

    def test_a_poll_never_moves_a_spinbox_the_user_is_editing(self) -> None:
        window = self.win
        psu, driver = self._connected(window, VSET=0.400)
        window.psu_vset.setValue(0.900)        # user is mid-edit
        driver.out = True
        window._on_psu_sample(psu.read_state())
        self.assertAlmostEqual(window.psu_vset.value(), 0.900)
        self.assertIn("0.400 V", window.psu_readout.text())  # but the readout
                                                             # shows the truth

    def test_setpoints_stay_usable_while_the_monitor_runs(self) -> None:
        # The port is locked per exchange, so a poll cannot swallow a set.
        window = self.win
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

    def test_disconnect_waits_for_the_monitor_to_unwind(self) -> None:
        window = self.win
        self._connected(window)
        window.psu_stop_event = threading.Event()
        window._disconnect_psu()
        self.assertTrue(window.psu_stop_event.is_set())
        self.assertIsNotNone(window.psu_controller)   # not yet: the loop owns it

        window._psu_monitor_done()                    # loop finished
        self.assertIsNone(window.psu_controller)
        self.assertTrue(window.psu_connect_button.isEnabled())
        self.assertIn("left exactly as it was", window.psu_page_status.text())

    def test_closing_the_window_stops_the_monitor(self) -> None:
        # Its own window: it is the close itself that is under test.
        window = self._window()
        stop = threading.Event()
        window.psu_stop_event = stop
        self._drain(window, self.app)
        window.close()
        self.app.processEvents()
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
        "src/calib_data/archive/2026-09/run_0928/calib.json",
        "src/calib_data/archive/2026-09/run_0916_10pairs_w15g35/calib.json",
        "src/calib_data/archive/2026-09/run_0911_10pairs_w20g30/calib.json",
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
            self.assertIn("Cannot read the forward model",
                          self.win.step8_status.text())

    def test_a_real_step_7_enables_it_and_quotes_y_max(self) -> None:
        """The file's one line goes beside the title, Y_max included."""
        if self.step7 is None:
            self.skipTest("no step-7 result in the tree")
        self.win.step8_step7_edit.setText(str(self.step7))
        self.assertTrue(self.win.step8_probe_button.isEnabled())
        said = self.win.step8_status.text()
        self.assertIn("Y_max", said)
        self.assertIn("4-bit step", said)
        self.assertIn("pairs", said)

    def test_the_status_line_matches_the_module(self) -> None:
        """The page must not compute its own Y_max -- one number, one owner."""
        if self.step7 is None:
            self.skipTest("no step-7 result in the tree")
        from calibration_module.fit.verify import full_scale, load_forward_model

        models, phases, _ = load_forward_model(self.step7)
        fs = full_scale(models, phases, sorted(models)).fs
        self.win.step8_step7_edit.setText(str(self.step7))
        self.assertAlmostEqual(self.win.step8_y_max, fs)
        self.assertIn(f"{fs * 1e3:.1f} mV", self.win.step8_status.text())
        # The band's top in mV, which is the panel's derived column.
        hi = self.win._step8_alphas()["alpha_hi"]
        self.assertIn(f"{hi * fs * 1e3:.1f} mV",
                      self.win.step8_alpha_hi_mv.text())

    def test_the_default_alpha_is_the_bands_own_top(self) -> None:
        """The brightest read the sweep will take is the one worth probing."""
        if self.step7 is None:
            self.skipTest("no step-7 result in the tree")
        self.win.step8_step7_edit.setText(str(self.step7))
        self.assertAlmostEqual(
            self.win.step8_probe_alpha.value(),
            round(self.win._step8_alphas()["alpha_hi"], 4), places=4)

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
        top = self.win._step8_alphas()["alpha_hi"]
        for alpha in (0.05, 0.25, 0.5, 1.0):
            self.win.step8_probe_alpha.setValue(alpha)
            text = self.win.step8_probe_pred.text()
            self.assertIn(f"{alpha * fs * 1e3:.1f} mV", text)
            peak = alpha * fs * STEP8.PEAK_OVER_MEAN
            if peak >= STEP8.AMP_LIMIT_V:
                # The peak is what rails; the mean is averaged back under it.
                self.assertIn("raw peak", text)
                self.assertIn("flat-top", text)
            elif peak > 0.85 * STEP8.AMP_LIMIT_V:
                self.assertIn("raw peak", text)
            elif alpha > top:
                self.assertIn("above alpha max", text)

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

        self.assertAlmostEqual(self.win.step8_alpha_lo.value(), STEP8.ALPHA_LO,
                               places=4)
        self.assertAlmostEqual(self.win.step8_points.value(), STEP8.LADDER)
        # Both ends are alphas, so neither box can ask past full scale.
        for box in (self.win.step8_alpha_lo, self.win.step8_alpha_hi):
            self.assertAlmostEqual(box.maximum(), 1.0, places=6)
            self.assertGreater(box.minimum(), 0.0)

    def test_the_millivolt_column_is_blank_until_a_calib_is_loaded(self) -> None:
        """The mV are alpha x Y_max, and there is no Y_max before the file."""
        for cap in (self.win.step8_alpha_hi_mv, self.win.step8_alpha_lo_mv):
            self.assertEqual(cap.text(), "")

    def test_the_millivolt_column_follows_the_boxes(self) -> None:
        if self.step7 is None:
            self.skipTest("no step-7 result in the tree")
        from calibration_module.steps import calib_step8 as STEP8

        self.win.step8_step7_edit.setText(str(self.step7))
        fs = self.win.step8_y_max
        saved = self.win.step8_alpha_hi.value()
        try:
            for alpha in (0.10, 0.20, 0.2945):
                self.win.step8_alpha_hi.setValue(alpha)
                self.assertIn(f"{alpha * fs * 1e3:.1f} mV",
                              self.win.step8_alpha_hi_mv.text())
        finally:
            self.win.step8_alpha_hi.setValue(saved)

    def test_a_band_top_near_the_rail_says_so_without_refusing(self) -> None:
        """The peak is what reaches the rail, and the mean cannot show it.

        Said in the caption rather than in a dialog: where this amplifier starts
        to compress has never been measured, so it is a judgement call about the
        electronics and the recorded peak_v is what settles it afterwards.
        """
        from calibration_module.steps import calib_step8 as STEP8

        saved_fs = getattr(self.win, "step8_y_max", None)
        saved = self.win.step8_alpha_hi.value()
        try:
            self.win.step8_y_max = 1.534
            self.win.step8_alpha_hi.setValue(0.6)      # 920 mV mean, 1.2 V peak
            self.win._step8_refresh_levels()
            text = self.win.step8_alpha_hi_mv.text()
            peak = 0.6 * 1.534 * STEP8.PEAK_OVER_MEAN
            self.assertGreater(peak, STEP8.AMP_LIMIT_V)
            self.assertIn("raw peak", text)
            self.assertIn(f"{peak * 1e3:.0f} mV", text)
            # a band well under the rail says nothing about it
            self.win.step8_alpha_hi.setValue(0.2945)
            self.win._step8_refresh_levels()
            self.assertNotIn("raw peak", self.win.step8_alpha_hi_mv.text())
        finally:
            self.win.step8_alpha_hi.setValue(saved)
            self.win.step8_y_max = saved_fs
            self.win._step8_refresh_levels()

    def test_an_empty_band_says_so(self) -> None:
        """alpha_lo above alpha_hi is a typo, and build() refuses it."""
        saved_fs = getattr(self.win, "step8_y_max", None)
        saved = self.win.step8_alpha_lo.value()
        try:
            self.win.step8_y_max = 1.534
            self.win.step8_alpha_lo.setValue(0.9)
            self.win._step8_refresh_levels()
            self.assertIn("above alpha max", self.win.step8_alpha_lo_mv.text())
        finally:
            self.win.step8_alpha_lo.setValue(saved)
            self.win.step8_y_max = saved_fs
            self.win._step8_refresh_levels()

    def test_clearing_the_path_takes_the_millivolts_with_it(self) -> None:
        """A stale mV beside a live box is the one thing this cannot show.

        Blanking the note is not enough: the mV are alpha x Y_max, so dropping
        Y_max drops the whole column.
        """
        if self.step7 is None:
            self.skipTest("no step-7 result in the tree")
        self.win.step8_step7_edit.setText(str(self.step7))
        self.assertNotEqual(self.win.step8_alpha_hi_mv.text(), "")
        self.win.step8_step7_edit.setText("")
        self.assertIsNone(self.win.step8_y_max)
        for cap in (self.win.step8_alpha_hi_mv, self.win.step8_alpha_lo_mv):
            self.assertEqual(cap.text(), "")
        self.assertIn("No calibration file", self.win.step8_status.text())

    def test_a_running_sweep_freezes_the_band(self) -> None:
        self.win._step8_set_running(True)
        try:
            for box in (self.win.step8_points, self.win.step8_alpha_lo,
                        self.win.step8_alpha_hi):
                self.assertFalse(box.isEnabled())
        finally:
            self.win._step8_set_running(False)
        self.assertTrue(self.win.step8_alpha_hi.isEnabled())

    def test_the_alphas_reach_the_module_as_the_module_spells_them(self) -> None:
        """_step8_alphas is what build() is given, splatted."""
        from calibration_module.fit.verify import load_forward_model
        from calibration_module.steps import calib_step8 as STEP8

        alphas = self.win._step8_alphas()
        self.assertEqual(set(alphas), {"alpha_lo", "alpha_hi"})
        self.assertAlmostEqual(alphas["alpha_lo"], STEP8.ALPHA_LO, places=4)
        if self.step7 is None:
            self.skipTest("no step-7 result in the tree")
        # spelled so they can be splatted straight into the module's own call
        models, phases, _ = load_forward_model(self.step7)
        STEP8.build(models, phases, sorted(models), seed=1, n=4, **alphas)

    def test_a_zero_alpha_is_refused_without_touching_the_bench(self) -> None:
        """A zero drive is the dark, which every read already takes first."""
        if self.step7 is None:
            self.skipTest("no step-7 result in the tree")
        self.win.step8_step7_edit.setText(str(self.step7))
        self.win.step8_probe_alpha.setValue(0.0)
        self.win._step8_probe()
        self.assertIn("above 0", self.win.step8_status.text())

    def test_display_is_refused_at_zero_too(self) -> None:
        """Same drive, same objection -- there is nothing to put on the panel."""
        if self.step7 is None:
            self.skipTest("no step-7 result in the tree")
        self.win.step8_step7_edit.setText(str(self.step7))
        self.win.step8_probe_alpha.setValue(0.0)
        self.win._step8_display()
        self.assertIn("above 0", self.win.step8_status.text())

    def test_display_needs_only_the_slm(self) -> None:
        """Its whole point: a known level on the panel before the DAQ is up.

        The probe asks for the DAQ first; display must not, so the only thing
        left to complain about is the SLM.
        """
        if self.step7 is None:
            self.skipTest("no step-7 result in the tree")
        self.win.step8_step7_edit.setText(str(self.step7))
        self.win.step8_probe_alpha.setValue(0.2)
        self.win._step8_display()
        self.assertIn("SLM", self.win.step8_status.text())
        self.assertNotIn("DAQ", self.win.step8_status.text())

    def test_display_is_live_with_the_probe(self) -> None:
        if self.step7 is None:
            self.skipTest("no step-7 result in the tree")
        self.win.step8_step7_edit.setText(str(self.step7))
        self.assertTrue(self.win.step8_display_button.isEnabled())
        self.win._step8_set_running(True)
        try:
            self.assertFalse(self.win.step8_display_button.isEnabled())
        finally:
            self.win._step8_set_running(False)
        self.assertTrue(self.win.step8_display_button.isEnabled())


class Step8WritesItselfTests(unittest.TestCase):
    """Where a step-8 sweep goes, and when.

    It used to stage itself in a temp directory and wait for a Save, because
    the page had no way of knowing which run it belonged to.  It does now: the
    calib file is named before the bench is touched, and it is the forward model
    this Y_sat measures -- so a finished sweep writes itself beside it, and Save
    is gone.  A Load, the other way round, writes nothing at all.

    The fit is not re-tested here -- ``test_calib_step8`` owns it.  What is
    tested is the contract this page hands it: which CSV, which calib file, and
    which directory.
    """

    REPO = Path(__file__).resolve().parents[1]

    @classmethod
    def setUpClass(cls) -> None:
        from PyQt5 import QtWidgets

        from gui.app import MainWindow

        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls.win = MainWindow()
        cls._tmp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()
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
                "unbounded": False, "n": 30, "n_ladder": 30,
                "rmse_v": 2.0e-3, "rmse_linear_v": 8.0e-3,
                "gain_lever": 15.0, "y_max_pred_v": 1.534,
                "png": str(csv.with_name(f"{csv.stem}_ysat.png")),
                "result_json": str(csv.with_name(
                    f"calib_step8_result_{stem}.json"))}

    def _payload(self, csv: Path, *, written: bool,
                 step7: Path | None = None) -> dict:
        return {"status": "complete", "csv": csv, "written": written,
                "step7": Path("calib.json") if step7 is None else step7,
                "result": self._result(csv)}

    def _sweep_csv(self, name: str = "calib_step8_0930_1209.csv") -> Path:
        """A CSV where a finished sweep leaves one: beside the calib file."""
        csv = Path(self._tmp.name) / name
        csv.write_text("block,driven,mean_v\n0,(1, 2),0.42\n", encoding="utf-8")
        return csv

    # ---- what a finished run says --------------------------------------
    def test_a_sweep_says_what_it_wrote(self) -> None:
        """There is no Save, so the status line is where "kept" is said."""
        self.win._step8_finished(self._payload(self._sweep_csv(), written=True))
        text = self.win.step8_status.text()
        self.assertIn("Y_sat = 5.0600", text)
        self.assertIn("wrote step8 + calib_step8_0930_1209.csv", text)

    def test_a_load_says_it_wrote_nothing(self) -> None:
        """Looking at a recorded sweep must not read as having re-filed it."""
        self.win._step8_finished(self._payload(self._sweep_csv(), written=False))
        text = self.win.step8_status.text()
        self.assertIn("re-fit calib_step8_0930_1209.csv", text)
        self.assertIn("nothing written", text)

    def test_a_run_that_read_nothing_leaves_nothing(self) -> None:
        self.win._step8_finished({"status": "partial (stopped)", "csv": None})
        self.assertIsNone(self.win.step8_last)
        self.assertIn("nothing written", self.win.step8_status.text())

    def test_a_running_sweep_takes_load_away(self) -> None:
        self.win._step8_set_running(True)
        self.assertFalse(self.win.step8_load_button.isEnabled())
        self.win._step8_set_running(False)

    # ---- Load ----------------------------------------------------------
    def test_the_render_directory_is_outside_the_repo(self) -> None:
        """A Load draws a PNG to look at; that is not a calibration artefact."""
        out = self.win._step8_render_dir()
        self.assertTrue(out.is_dir())
        self.assertNotEqual(out.resolve(),
                            self.win._default_calib_dir().resolve())
        self.assertNotIn(self.REPO.resolve(), out.resolve().parents)

    def test_load_re_fits_the_csv_the_section_names_and_writes_nothing(self) -> None:
        """ONE input: the calib file. Its step8 section says which sweep."""
        import calibration_module.fit.step8_report as REPORT

        run = self.REPO / "src/calib_data/archive/2026-09/run_0930"
        if not (run / "calib.json").is_file():          # pragma: no cover
            self.skipTest("run_0930 is not in this checkout")
        from calibration_module.calib_file import load_calib
        wanted = load_calib(run).csv_path(8)
        before = sorted(p.name for p in run.iterdir())

        seen: dict = {}

        def fake_report(csv, calib, *, out_dir=None, log=print, **kw):
            seen.update(csv=Path(csv), calib=Path(calib), out_dir=Path(out_dir))
            return self._result(Path(csv))

        # Load ASKS which calibration to re-fit; the page's own field is where
        # a sweep would write and must not decide what gets looked at.
        self.win.step8_step7_edit.setText("")
        with patch("PyQt5.QtWidgets.QFileDialog.getOpenFileName",
                   return_value=(str(run / "calib.json"), "")), \
             patch.object(REPORT, "report_csv", fake_report), \
             patch.object(self.win, "_run_task",
                          lambda _l, work, done, _f: done(work())):
            self.win._step8_load()

        self.assertEqual(seen["csv"], wanted)
        self.assertEqual(seen["calib"], (run / "calib.json"))
        self.assertEqual(seen["out_dir"].resolve(),
                         self.win._step8_render_dir().resolve())
        self.assertEqual(sorted(p.name for p in run.iterdir()), before,
                         "a Load leaves the calibration exactly as it was")
        self.assertIn("nothing written", self.win.step8_status.text())

    def test_load_without_a_step8_section_says_which_step_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run_thin"
            run.mkdir()
            (run / "calib.json").write_text(
                '{"schema": "rbonn_calib_v3"}', encoding="utf-8")
            with patch("PyQt5.QtWidgets.QFileDialog.getOpenFileName",
                       return_value=(str(run / "calib.json"), "")):
                self.win._step8_load()
        text = self.win.step8_status.text()
        self.assertIn("Load failed", text)
        self.assertIn("step8", text)

    def test_a_cancelled_load_does_nothing_at_all(self) -> None:
        """Dismissing the dialog is not an error, and is not a fit either."""
        self.win.step8_status.setText("Idle")
        with patch("PyQt5.QtWidgets.QFileDialog.getOpenFileName",
                   return_value=("", "")):
            self.win._step8_load()
        self.assertEqual(self.win.step8_status.text(), "Idle")


class AcquisitionPanelTests(unittest.TestCase):
    """The four acquisition settings steps 6, 7 and 8 actually show.

    The panel used to carry nine controls on steps 6 and 7, of which five have
    had the same value on every run this bench has taken.  They are still live
    widgets -- the sweep reads them, and a hidden one that stopped carrying its
    default would change what a run measures without showing it -- so these
    pin both halves: what is on the panel, and what the hidden ones hold.
    """

    @classmethod
    def setUpClass(cls) -> None:
        from PyQt5 import QtWidgets

        from gui.app import MainWindow

        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls.win = MainWindow()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.win.close()

    def _daq_panel(self, edit):
        """The DAQ group box of the page holding ``edit``."""
        from PyQt5 import QtWidgets

        col = edit
        while not isinstance(col.parentWidget(), QtWidgets.QSplitter):
            col = col.parentWidget()
        for i in range(col.layout().count()):
            panel = col.layout().itemAt(i).widget()
            if isinstance(panel, QtWidgets.QGroupBox) and \
                    panel.title().startswith("DAQ"):
                return panel
        raise AssertionError("no DAQ panel on this page")

    def _visible_rows(self, panel):
        from PyQt5 import QtWidgets

        out = []
        lay = panel.layout()
        for i in range(lay.count()):
            w = lay.itemAt(i).widget()
            if w is None or w.isHidden():
                continue
            if isinstance(w, QtWidgets.QCheckBox):
                out.append(w.text())
            elif isinstance(w, QtWidgets.QLabel):
                out.append(w.text())
        return out

    def test_every_page_shows_the_same_four(self) -> None:
        for name, edit in (("step 6", self.win.tpa_step3_edit),
                           ("step 7", self.win.tpa_phase_step6_edit),
                           ("step 8", self.win.step8_step7_edit)):
            with self.subTest(page=name):
                self.assertEqual(
                    self._visible_rows(self._daq_panel(edit)),
                    ["Channel", "Range", "T_meas", "Invert sign (TIA)"])

    def test_the_hidden_ones_still_carry_the_bench_defaults(self) -> None:
        """Hidden is not dropped: the sweep still reads every one of these."""
        w = self.win
        for name, widget in (("step6 rate", w.tpa_daq_rate),
                             ("step7 rate", w.tpa_phase_daq_rate),
                             ("step8 rate", w.step8_daq_rate)):
            with self.subTest(widget=name):
                self.assertTrue(widget.isHidden())
                self.assertEqual(widget.value(), 1000.0)
        for name, widget in (("step6 low-pass", w.tpa_daq_fcut),
                             ("step7 low-pass", w.tpa_phase_daq_fcut),
                             ("step8 low-pass", w.step8_daq_fcut)):
            with self.subTest(widget=name):
                self.assertTrue(widget.isHidden())
                self.assertEqual(widget.value(), 20.0)
        for name, widget in (("step6 settle", w.tpa_settle),
                             ("step7 settle", w.tpa_phase_settle)):
            with self.subTest(widget=name):
                self.assertTrue(widget.isHidden())
                self.assertEqual(widget.value(), 0.25)
        # The PMT log is on for steps 6 and 7 and off for step 8, which is what
        # each page came up with before it was hidden.
        self.assertTrue(w.tpa_pmt.isHidden())
        self.assertTrue(w.tpa_pmt.isChecked())
        self.assertTrue(w.tpa_phase_pmt.isHidden())
        self.assertTrue(w.tpa_phase_pmt.isChecked())
        self.assertTrue(w.step8_pmt.isHidden())
        self.assertFalse(w.step8_pmt.isChecked())

    def test_a_hidden_settle_still_reaches_the_sweep(self) -> None:
        """The proof that these are controls and not decoration."""
        w = self.win
        self.assertEqual(w._tpa_acq().settle_s, 0.25)
        w.tpa_settle.setValue(0.5)
        try:
            self.assertEqual(w._tpa_acq().settle_s, 0.5)
        finally:
            w.tpa_settle.setValue(0.25)

    def test_step8_reads_its_own_page_not_a_module_constant(self) -> None:
        """Step 8 declared no acquisition of its own; now it does.

        It took the window, the range and the sign from STEP8.ACQ, which made
        the one step whose fit is a saturation curve the one step that could
        not say what it read at.
        """
        from calibration_module.measure.pair_v2 import DEFAULT_ACQ

        w = self.win
        self.assertEqual(w._step8_acq(), DEFAULT_ACQ)       # untouched == CLI

        w.step8_tmeas.setValue(2.0)
        w.step8_invert.setChecked(False)
        try:
            acq = w._step8_acq()
            self.assertEqual(acq.t_meas_s, 2.0)
            self.assertFalse(acq.invert)
            # and the run-time estimate is quoted under the page, not the module
            self.assertIn("2 s every read", w.step8_estimate.text())
        finally:
            w.step8_tmeas.setValue(DEFAULT_ACQ.t_meas_s)
            w.step8_invert.setChecked(DEFAULT_ACQ.invert)

    def test_no_control_column_scrolls(self) -> None:
        """A setting you have to scroll to find is a setting you will not check."""
        from PyQt5 import QtWidgets

        for name, edit in (("step 6", self.win.tpa_step3_edit),
                           ("step 7", self.win.tpa_phase_step6_edit),
                           ("step 8", self.win.step8_step7_edit)):
            with self.subTest(page=name):
                w = edit
                while w is not None:
                    self.assertNotIsInstance(w, QtWidgets.QScrollArea)
                    w = w.parentWidget()


class PageStatusLineTests(unittest.TestCase):
    """One line of feedback, beside the page title, with no history."""

    @classmethod
    def setUpClass(cls) -> None:
        from PyQt5 import QtWidgets

        from gui.app import MainWindow

        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls.win = MainWindow()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.win.close()

    def _statuses(self):
        return (("step 6", self.win.tpa_status),
                ("step 7", self.win.tpa_phase_status),
                ("step 8", self.win.step8_status))

    def test_it_sits_in_the_title_row(self) -> None:
        from gui.app import StatusLine

        for name, status in self._statuses():
            with self.subTest(page=name):
                self.assertIsInstance(status, StatusLine)
                header = status.parentWidget().layout().itemAt(0).layout()
                # item 0 is the page title, item 1 is this
                self.assertEqual(header.indexOf(status), 1)

    def test_it_keeps_the_whole_sentence_however_narrow_the_window(self) -> None:
        """It is painted elided, not stored elided.

        What the page said is not a function of how wide the window happens to
        be -- and every reader of it, the tests included, wants the sentence.
        """
        long = "Loaded run_0930/calib.json \N{MIDDLE DOT} " + "x" * 400
        for name, status in self._statuses():
            with self.subTest(page=name):
                status.setText(long)
                self.assertEqual(status.text(), long)
                self.assertEqual(status.toolTip(), long)

    def test_it_is_one_line_and_keeps_nothing(self) -> None:
        for name, status in self._statuses():
            with self.subTest(page=name):
                self.assertFalse(status.wordWrap())
                status.setText("first")
                status.setText("second")
                self.assertEqual(status.text(), "second")


if __name__ == "__main__":
    unittest.main()
