"""GUI Step 6 on the v2 estimator -- page wiring, no hardware.

These build a real MainWindow offscreen and drive the page's own methods, so
they cover the wiring the unit tests in ``test_pair_v2.py`` cannot: that the
table shows what the fit produced, that a row selection reaches the panels, and
that the file the Save button writes is the one Step 7 reads.

Nothing here touches an instrument.  The sweep path needs an SLM and a DAQ and
is not exercised; everything downstream of the rows is, by feeding the page the
same committed measurement CSV the offline script refits.
"""

from __future__ import annotations

import json
import os

# Render Qt/matplotlib headless before either is imported by the app module.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import numpy as np
from PyQt5 import QtWidgets

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from calibration_module.fit.pair_v2 import load_meas_csv  # noqa: E402
from calibration_module.fit.phase import PairModel, load_pair_models  # noqa: E402

MEAS_CSV = REPO / "src/calib_data/archive/2026-09/run_0907_1724/calib_step6v2_meas_0907_1757.csv"
STEP3 = REPO / "src/calib_data/archive/2026-09/run_0908_1444/calib_step3c_0907_1358_pad10.json"

#: What the 0907 run published, in the order the table lists them.
RECORDED_ETA = [0.17760529803227112, 0.17954179839571333, 0.1920940148157429,
                0.2142402286038278, 0.25500301123102215]

_app: QtWidgets.QApplication | None = None


def setUpModule() -> None:
    global _app
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _WindowCase(unittest.TestCase):
    """One MainWindow per class -- building it is by far the slow part."""

    @classmethod
    def setUpClass(cls) -> None:
        from gui.app import MainWindow
        cls.win = MainWindow()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.win.close()

    def _load(self):
        w = self.win
        failed = w._tpa_fit_rows(load_meas_csv(MEAS_CSV), w._tpa_config(), None)
        self.assertEqual(failed, [])
        w._tpa_fill_table()
        w._tpa_redraw()
        return w


class PairListTests(_WindowCase):
    def test_ranges_lists_and_mixes(self) -> None:
        parse = self.win._tpa_parse_pairs
        self.assertEqual(parse("2-6"), [2, 3, 4, 5, 6])
        self.assertEqual(parse("1,3,5"), [1, 3, 5])
        self.assertEqual(parse("1, 3-5 ; 8"), [1, 3, 4, 5, 8])

    def test_duplicates_collapse_and_order_is_normalised(self) -> None:
        self.assertEqual(self.win._tpa_parse_pairs("5,2,5,3-4"), [2, 3, 4, 5])

    def test_bad_input_raises(self) -> None:
        for text in ("", "  ", "abc", "5-2"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.win._tpa_parse_pairs(text)


class SweepRampTests(_WindowCase):
    """The three ramp controls build the cross line -- the levels eta is fitted to."""

    def tearDown(self) -> None:
        self.win.tpa_sweep_min.setValue(0.20)
        self.win.tpa_sweep_max.setValue(0.90)
        self.win.tpa_points.setValue(4)

    @staticmethod
    def _cross(cfg):
        return [(w, n) for x, w, n in cfg.grid if x and w]

    def test_defaults_give_the_37_acquisition_grid(self) -> None:
        from calibration_module.measure.pair_v2 import build_schedule

        cfg = self.win._tpa_config()
        self.assertEqual(cfg.fit_w_range, (0.2, 0.9))
        self.assertEqual(len(build_schedule(cfg)), 37)

    def test_the_ramp_reaches_the_grid_and_the_fit_window(self) -> None:
        w = self.win
        w.tpa_sweep_min.setValue(0.30)
        w.tpa_sweep_max.setValue(0.90)
        w.tpa_points.setValue(3)
        cfg = w._tpa_config()
        self.assertEqual(cfg.fit_w_range, (0.3, 0.9))
        self.assertEqual([lv[0] for lv in self._cross(cfg)][:3], [0.3, 0.6, 0.9])

    def test_last_ramp_point_gets_the_extra_repeats(self) -> None:
        """The top of the window carries the most leverage on the slope."""
        cross = self._cross(self.win._tpa_config())
        in_window = [lv for lv in cross if lv[0] <= 0.9]
        self.assertEqual(in_window[-1][1], 6)
        self.assertTrue(all(n == 4 for _w, n in in_window[:-1]))

    def test_max_below_one_keeps_the_top_drive_diagnostic(self) -> None:
        cross = self._cross(self.win._tpa_config())
        self.assertEqual(cross[-1], (1.0, 2))          # measured, excluded from the fit

    def test_max_at_one_gives_up_that_diagnostic(self) -> None:
        """Compression then sits inside the fit -- the cost of the wider window."""
        w = self.win
        w.tpa_sweep_max.setValue(1.00)
        cfg = w._tpa_config()
        self.assertEqual(cfg.fit_w_range[1], 1.0)
        self.assertEqual(self._cross(cfg)[-1], (1.0, 6))    # the ramp end, not excluded
        # Refitting the recorded run under this window pulls w = 1.0 in, so
        # nothing is left to report as the compression diagnostic.
        w._tpa_fit_rows(load_meas_csv(MEAS_CSV), cfg, None)
        self.assertEqual(w.tpa_fits[0].excluded, [])

    def test_a_half_typed_ramp_falls_back_instead_of_raising(self) -> None:
        """A spinbox mid-edit must not stop the page redrawing."""
        w = self.win
        w.tpa_sweep_min.setValue(0.90)
        w.tpa_sweep_max.setValue(0.90)                 # min == max: not a ramp
        cfg = w._tpa_config()
        self.assertEqual(cfg.fit_w_range, (0.2, 0.9))  # the validated default
        w._tpa_redraw()                                 # must not raise


class AcquisitionSettingsTests(_WindowCase):
    def test_invert_is_on_by_default(self) -> None:
        """Off gives b < 0 and eta = NaN, so this is a precondition not a taste."""
        self.assertTrue(self.win.tpa_invert.isChecked())
        self.assertTrue(self.win._tpa_acq().invert)

    def test_checkboxes_reach_the_acq_config(self) -> None:
        w = self.win
        w.tpa_invert.setChecked(False)
        w.tpa_autorange.setChecked(False)
        acq = w._tpa_acq()
        self.assertFalse(acq.invert)
        self.assertFalse(acq.autorange)
        w.tpa_invert.setChecked(True)
        w.tpa_autorange.setChecked(True)

    def test_default_windows_are_the_v2_validated_ones(self) -> None:
        self.assertAlmostEqual(self.win.tpa_tboth.value(), 8.0)
        self.assertAlmostEqual(self.win.tpa_tsingle.value(), 10.0)


class StepThreeInputTests(_WindowCase):
    """The Step-3 calibration is named on the page and required to run.

    It is not inherited from the Encoding page: every level of the sweep is
    encoded through it, so a run against the wrong one does not fail, it
    calibrates a different aperture. Naming it per run is also what makes a GUI
    run and a ``--step3`` terminal run the same run.
    """

    def tearDown(self) -> None:
        self.win.tpa_step3_edit.setText("")

    def test_run_refuses_without_one(self) -> None:
        w = self.win
        w.tpa_step3_edit.setText("")
        w._tpa_run()
        self.assertIn("Step 3", w.tpa_status.text())
        self.assertIn("no Step-3 calibration", w.tpa_status.text())

    def test_run_refuses_a_path_that_is_not_there(self) -> None:
        w = self.win
        w.tpa_step3_edit.setText(str(STEP3.parent / "no_such_file.json"))
        w._tpa_run()
        self.assertIn("not found", w.tpa_status.text())

    def test_a_valid_file_is_described_before_the_run(self) -> None:
        """An hour of bench time deserves a look at the file first."""
        w = self.win
        w.tpa_step3_edit.setText(str(STEP3))
        text = w.tpa_step3_label.text()
        self.assertIn(STEP3.name, text)
        self.assertRegex(text, r"\d+ pairs")
        self.assertRegex(text, r"\d+ px window")
        self.assertEqual(w.tpa_step3_label.property("status"), "ok")

    def test_a_bad_file_is_flagged_not_silently_ignored(self) -> None:
        w = self.win
        w.tpa_step3_edit.setText(str(MEAS_CSV))       # a CSV, not a calibration
        self.assertEqual(w.tpa_step3_label.property("status"), "error")

    def test_layout_matches_the_offline_loader(self) -> None:
        """Same file, same loader, same method -> the same pixels get driven."""
        from calibration_module.fit.slm_response import load_calibration_result
        from slm_module.encoding import channel_layout_from_calibration

        w = self.win
        w.tpa_step3_edit.setText(str(STEP3))
        _calib, gui_layout = w._tpa_load_step3()
        offline = channel_layout_from_calibration(
            load_calibration_result(STEP3), method=w._tpa_config().encoding_method
        )
        self.assertEqual(gui_layout.n_channels, offline.n_channels)
        self.assertEqual(gui_layout.channel_width_px, offline.channel_width_px)
        self.assertEqual(gui_layout.pitch_px, offline.pitch_px)


class ResultsTableTests(_WindowCase):
    def test_table_shows_the_published_etas(self) -> None:
        w = self._load()
        self.assertEqual(w.tpa_table.rowCount(), 5)
        for row, expected in enumerate(RECORDED_ETA):
            with self.subTest(row=row):
                self.assertEqual(w.tpa_table.item(row, 0).text(), str(row + 2))
                self.assertAlmostEqual(
                    float(w.tpa_table.item(row, 1).text()), expected, places=5
                )

    def test_background_columns_are_in_millivolts(self) -> None:
        """a_x/a_w/d are volts on the fit and mV in the table -- easy to get wrong."""
        w = self._load()
        fit = w.tpa_fits[3]                       # pair 5
        for col, name in ((3, "a_x"), (4, "a_w"), (5, "d")):
            with self.subTest(name=name):
                self.assertAlmostEqual(
                    float(w.tpa_table.item(3, col).text()),
                    fit.bg[name][0] * 1e3, places=4,
                )

    def test_all_checks_pass_on_this_run(self) -> None:
        w = self._load()
        for row in range(w.tpa_table.rowCount()):
            self.assertEqual(w.tpa_table.item(row, 8).text(), "OK")

    def test_row_selection_drives_the_panels(self) -> None:
        """The table IS the pair selector; there is no separate combo box."""
        w = self._load()
        w.tpa_table.selectRow(3)
        self.assertEqual(w._tpa_selected_fit().index, 5)
        w.tpa_table.selectRow(0)
        self.assertEqual(w._tpa_selected_fit().index, 2)
        self.assertFalse(hasattr(w, "tpa_pair_combo"))

    def test_panels_render_for_every_pair(self) -> None:
        """Two panels, not the PNG's six -- and no verification prose panel."""
        w = self._load()
        self.assertFalse(hasattr(w, "tpa_verify_label"))
        for row in range(w.tpa_table.rowCount()):
            with self.subTest(row=row):
                w.tpa_table.selectRow(row)
                # estimator over its own pulls, sharing the w axis
                self.assertEqual(len(w.tpa_est_fig.axes), 2)
                self.assertEqual(len(w.tpa_check_fig.axes), 1)

    def test_panels_are_safe_with_no_result(self) -> None:
        from gui.app import MainWindow
        fresh = MainWindow()
        try:
            self.assertIsNone(fresh._tpa_selected_fit())
            fresh._tpa_redraw()                    # must not raise
            self.assertEqual(len(fresh.tpa_est_fig.axes), 2)
        finally:
            fresh.close()


class SaveRoundTripTests(_WindowCase):
    def test_saved_json_is_what_step_7_reads(self) -> None:
        """The point of the combined file: step 7 needs this one file, not two."""
        w = self._load()
        w.tpa_step3_edit.setText(str(STEP3))
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "calib_step6v2_meas_0910_1938.csv"
            with unittest.mock.patch.object(
                QtWidgets.QFileDialog, "getSaveFileName",
                staticmethod(lambda *a, **k: (str(target), "")),
            ):
                w._tpa_save()

            self.assertTrue(target.is_file(), w.tpa_status.text())
            js_path = target.with_name("calib_step6v2_result_0910_1938.json")
            self.assertTrue(js_path.is_file(), w.tpa_status.text())
            self.assertEqual(len(list(target.parent.glob("*.json"))), 1,
                             "the JSON is named like the script's, not the CSV")

            payload = json.loads(js_path.read_text(encoding="utf-8"))
            self.assertEqual(sorted(payload), ["encoding", "step3", "step6"])
            self.assertIn("transfer_fits", payload["step3"])

            models = load_pair_models(str(js_path))
            self.assertEqual(sorted(models), [2, 3, 4, 5, 6])
            self.assertAlmostEqual(models[5].eta, RECORDED_ETA[3], places=12)

            self.assertEqual(
                len(list(target.parent.glob("calib_step6v2_pair*_0910_1938.png"))), 5,
                "one diagnostic PNG per pair, same renderer as the offline script",
            )

    def test_save_without_a_calibration_still_writes_the_rows(self) -> None:
        """The measurement must never be lost to a missing step-3 file."""
        w = self._load()
        w.tpa_step3_edit.setText("")
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "rows_only.csv"
            with unittest.mock.patch.object(
                QtWidgets.QFileDialog, "getSaveFileName",
                staticmethod(lambda *a, **k: (str(target), "")),
            ):
                w._tpa_save()
            self.assertTrue(target.is_file())
            self.assertFalse(list(target.parent.glob("*.json")))
            self.assertIn("no Step-3 calibration", w.tpa_status.text())


class ProgressDialogTests(_WindowCase):
    """A run takes tens of minutes, so it reports into the shared popup.

    Steps 1-3 already pop ``CalibrationProgressDialog``; reusing it gets step 6
    the same window with elapsed/ETA and a live trace, rather than a second
    progress widget with its own idea of how to estimate time.
    """

    def tearDown(self) -> None:
        if self.win.calibration_dialog is not None:
            self.win.calibration_dialog.close()
            self.win.calibration_dialog = None

    def _progress(self, step: int, total: int = 37):
        from calibration_module.measure.pair_v2 import PairV2Progress

        return PairV2Progress(
            step=step, total=total, pair_index=2, repeat=0, x=1.0, w=0.9,
            mean_v=0.0412, std_v=2e-4, range_v=0.1, single=False, duration_s=8.0,
        )

    def test_no_inline_progress_bar_on_the_page(self) -> None:
        self.assertFalse(hasattr(self.win, "tpa_progress_bar"))

    def test_dialog_shows_the_step_6_phase_and_tracks_the_bar(self) -> None:
        w = self.win
        w._open_calibration_dialog(on_stop=w._tpa_stop)
        w._on_tpa_progress(self._progress(1))
        self.assertEqual(w.calibration_dialog.phase_label.text(),
                         "Step 6 \N{MIDDLE DOT} TPA pair efficiency")
        w._on_tpa_progress(self._progress(19))
        self.assertEqual(w.calibration_dialog.progress_bar.value(), 19)
        self.assertEqual(w.calibration_dialog.progress_bar.maximum(), 37)
        self.assertIn("ETA", w.calibration_dialog.eta_label.text())
        self.assertIn("pair 2", w.calibration_dialog.status_label.text())

    def test_stop_button_reaches_this_page_not_the_full_calibration(self) -> None:
        w = self.win
        w.tpa_stop_event = __import__("threading").Event()
        w._open_calibration_dialog(on_stop=w._tpa_stop)
        w.calibration_dialog.stop_button.click()
        self.assertTrue(w.tpa_stop_event.is_set())
        w.tpa_stop_event = None

    def test_finish_freezes_the_window_but_leaves_it_open(self) -> None:
        """On a run this long, the log and trace must not vanish when it ends."""
        w = self.win
        w._open_calibration_dialog(on_stop=w._tpa_stop)
        w._on_tpa_progress(self._progress(37))
        w._tpa_close_dialog(True, "Done · 5 pairs")
        self.assertFalse(w.calibration_dialog.stop_button.isEnabled())
        self.assertTrue(w.calibration_dialog.close_button.isEnabled())
        self.assertIn("Done", w.calibration_dialog.status_label.text())

    def test_a_rejected_run_pops_no_dialog(self) -> None:
        w = self.win
        w.calibration_dialog = None
        w.tpa_step3_edit.setText("")
        w._tpa_run()
        self.assertIsNone(w.calibration_dialog)


class Step7HandoffTests(_WindowCase):
    def test_step_7_loads_the_saved_file_not_this_page_s_memory(self) -> None:
        """The handoff is a named file, not the tab's last result.

        Step 7 pins its fringe amplitudes to these etas AND builds its layout
        from the step-3 payload beside them, so it takes one file it can record
        in its own output -- there is deliberately no path by which a step-7 run
        is driven by something with no file on disk.
        """
        w = self._load()
        w.tpa_step3_edit.setText(str(STEP3))
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "handoff.csv"
            with unittest.mock.patch.object(
                QtWidgets.QFileDialog, "getSaveFileName",
                staticmethod(lambda *a, **k: (str(target), "")),
            ):
                w._tpa_save()
            w.tpa_phase_step6_edit.setText(str(target.with_name("handoff_result.json")))
            layout, models = w._tpa_phase_load_step6()
            self.assertEqual(sorted(models), [2, 3, 4, 5, 6])
            self.assertAlmostEqual(models[5].eta, RECORDED_ETA[3], places=12)
            self.assertEqual(layout.n_channels, 6)
        w.tpa_phase_step6_edit.setText("")

    def test_dropped_q_columns_arrive_as_exact_zeros(self) -> None:
        """v2 does not fit q_x/q_w; PairModel needs them, and 0.0 is the contract."""
        w = self._load()
        model = PairModel.from_pair_v2(w.tpa_fits[3])
        self.assertEqual(model.q_x, 0.0)
        self.assertEqual(model.q_w, 0.0)
        self.assertNotIn("q_x", w.tpa_fits[3].bg)


if __name__ == "__main__":
    unittest.main()
