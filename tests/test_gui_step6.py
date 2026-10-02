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

import contextlib
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibration_module.fit.pair_v2 import load_meas_csv  # noqa: E402
from calibration_module.fit.phase import PairModel, load_pair_models  # noqa: E402
from calibration_module.measure.pair_v2 import DEFAULT_ACQ  # noqa: E402
from calib_fixtures import seed_run  # noqa: E402

MEAS_CSV = REPO / "src/calib_data/archive/2026-09/run_0907_1724/calib_step6v2_meas_0907_1757.csv"
STEP6 = REPO / "src/calib_data/archive/2026-09/run_0907_1724/calib_step6v2_result_0907_1757.json"
STEP3 = REPO / "src/calib_data/archive/2026-09/run_0908_1444/calib_step3c_0907_1358_pad10.json"

#: What the 0907 CSV fits to (see test_pair_v2.RECORDED_ETA), table order.
RECORDED_ETA = [0.17723508222881532, 0.17949741792999532, 0.19180164107087086,
                0.21419773519244129, 0.2549718270732447]

_app: QtWidgets.QApplication | None = None


def setUpModule() -> None:
    global _app
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@contextlib.contextmanager
def _picks(path):
    """Answer the Load dialog with ``path``.

    Load asks which calibration to re-fit rather than taking the page's field:
    looking at another night is the usual reason to press it, and the field is
    where a SWEEP writes.  So every Load here has to say what was picked.
    """
    with unittest.mock.patch.object(
            QtWidgets.QFileDialog, "getOpenFileName",
            staticmethod(lambda *a, **k: (str(path), ""))):
        yield



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
        failed = w._tpa_fit_rows(load_meas_csv(MEAS_CSV), None)
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

    def test_defaults_give_the_34_acquisition_grid(self) -> None:
        from calibration_module.measure.pair_v2 import build_schedule

        self.assertEqual(len(build_schedule(self.win._tpa_config())), 34)

    def test_the_ramp_reaches_the_grid(self) -> None:
        w = self.win
        w.tpa_sweep_min.setValue(0.30)
        w.tpa_sweep_max.setValue(0.90)
        w.tpa_points.setValue(3)
        self.assertEqual([lv[0] for lv in self._cross(w._tpa_config())], [0.3, 0.6, 0.9])

    def test_last_ramp_point_gets_the_extra_repeats(self) -> None:
        """The top of the ramp carries the most leverage on the slope."""
        cross = self._cross(self.win._tpa_config())
        self.assertEqual(cross[-1][1], 6)
        self.assertTrue(all(n == 4 for _w, n in cross[:-1]))

    def test_max_below_one_appends_nothing(self) -> None:
        """No (1, 1.0) compression level: the sweep is the ramp and only the ramp."""
        cross = self._cross(self.win._tpa_config())
        self.assertEqual(cross[-1], (0.9, 6))
        self.assertNotIn(1.0, [w for w, _n in cross])

    def test_a_loaded_csv_is_fitted_over_its_own_cross_line(self) -> None:
        """The ramp controls describe the NEXT sweep; a loaded file is fitted as
        it was swept.  Spinboxes at 0.2-0.9, file swept to 1.0: window 0.2-1.0."""
        w = self.win
        w.tpa_sweep_min.setValue(0.20)
        w.tpa_sweep_max.setValue(0.90)
        w._tpa_fit_rows(load_meas_csv(MEAS_CSV), None)
        self.assertEqual(w.tpa_fits[0].fit_w_range, (0.2, 1.0))

    def test_a_half_typed_ramp_falls_back_instead_of_raising(self) -> None:
        """A spinbox mid-edit must not stop the page redrawing."""
        from calibration_module.fit.pair_v2 import PairV2Config

        w = self.win
        w.tpa_sweep_min.setValue(0.90)
        w.tpa_sweep_max.setValue(0.90)                 # min == max: not a ramp
        self.assertEqual(w._tpa_config(), PairV2Config())  # the validated default
        w._tpa_redraw()                                 # must not raise


class AcquisitionSettingsTests(_WindowCase):
    def test_invert_is_on_by_default(self) -> None:
        """Off gives b < 0 and eta = NaN, so this is a precondition not a taste."""
        self.assertTrue(self.win.tpa_invert.isChecked())
        self.assertTrue(self.win._tpa_acq().invert)

    def test_the_invert_checkbox_reaches_the_acq_config(self) -> None:
        w = self.win
        w.tpa_invert.setChecked(False)
        self.assertFalse(w._tpa_acq().invert)
        w.tpa_invert.setChecked(True)
        self.assertTrue(w._tpa_acq().invert)

    def test_there_is_one_window_box_and_it_is_the_bench_s(self) -> None:
        """One window, dark included, and it is not this page's to choose.

        The dark used to run 10 s against the sweep's 8 on the argument that a
        near-zero read needs the averaging.  It does, but the dark is subtracted
        from every point of the sweep, so a dark averaged differently puts its
        own noise into all of them.
        """
        self.assertAlmostEqual(self.win.tpa_tmeas.value(), DEFAULT_ACQ.t_meas_s)
        self.assertAlmostEqual(self.win._tpa_acq().t_meas_s, DEFAULT_ACQ.t_meas_s)
        self.assertFalse(hasattr(self.win, "tpa_tsingle"))


class StepThreeInputTests(_WindowCase):
    """The RUN FOLDER is named on the page and required to run.

    Its step-3 section is not inherited from the Encoding page: every level of
    the sweep is encoded through it, so a run against the wrong one does not
    fail, it calibrates a different aperture. Naming it per run is also what
    makes a GUI run and a ``--run`` terminal run the same run.
    """

    def setUp(self) -> None:
        super().setUp()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.run = seed_run(Path(tmp.name) / "run_x", step3=STEP3)

    def tearDown(self) -> None:
        self.win.tpa_step3_edit.setText("")

    def test_run_refuses_without_one(self) -> None:
        w = self.win
        w.tpa_step3_edit.setText("")
        w._tpa_run()
        self.assertIn("no calibration file", w.tpa_status.text())

    def test_run_refuses_a_path_that_is_not_there(self) -> None:
        w = self.win
        w.tpa_step3_edit.setText(str(self.run.parent / "no_such_run"))
        w._tpa_run()
        self.assertIn("is not there", w.tpa_status.text())

    def test_a_valid_run_is_described_before_the_run(self) -> None:
        """An hour of bench time deserves a look at the folder first."""
        w = self.win
        w.tpa_step3_edit.setText(str(self.run))
        text = w.tpa_step3_label.text()
        self.assertIn(self.run.name, text)        # the folder, not just calib.json
        self.assertIn("step3", text)                 # the stamp chain
        self.assertRegex(text, r"\d+ pairs")
        self.assertRegex(text, r"\d+ px window")
        # It lands in the page's status line, beside the title, and plain: the
        # green pill under the field said only that the field had a path in it.
        self.assertIs(w.tpa_step3_label, w.tpa_status)
        self.assertIsNone(w.tpa_step3_label.property("status"))

    def test_a_folder_without_a_calib_json_is_flagged(self) -> None:
        w = self.win
        w.tpa_step3_edit.setText(str(MEAS_CSV.parent))   # no calib.json in there
        self.assertIn("calib.json", w.tpa_status.text())
        self.assertFalse(w.tpa_run_button.isEnabled())

    def test_layout_matches_the_offline_loader(self) -> None:
        """Same run, same loader -> the same pixels get driven."""
        from calibration_module.calib_file import load_calib
        from slm_module.encoding import channel_layout_from_calibration

        w = self.win
        w.tpa_step3_edit.setText(str(self.run))
        _calib, gui_layout = w._tpa_load_step3()
        offline = channel_layout_from_calibration(load_calib(self.run).step3_result())
        self.assertEqual(gui_layout.n_channels, offline.n_channels)
        self.assertEqual(gui_layout.channel_width_px, offline.channel_width_px)
        self.assertEqual(gui_layout.pitch_px, offline.pitch_px)


class ResultsTableTests(_WindowCase):
    def test_table_shows_the_published_etas(self) -> None:
        """Each row is its own pair's eta, to within what the weights can move it.

        Not equality, so a reweighting cannot break it.  The point of this test
        is that row ``i`` carries pair ``i + 2``'s eta -- a transposition or an
        off-by-one would be wrong by percent, not by the 0.1% a reweighting
        moves it.
        """
        w = self._load()
        self.assertEqual(w.tpa_table.rowCount(), 5)
        for row, expected in enumerate(RECORDED_ETA):
            with self.subTest(row=row):
                self.assertEqual(w.tpa_table.item(row, 0).text(), str(row + 2))
                got = float(w.tpa_table.item(row, 1).text())
                self.assertAlmostEqual(got / expected, 1.0, places=2)

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
    def setUp(self) -> None:
        super().setUp()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def test_the_save_writes_the_step6_section_step_7_reads(self) -> None:
        """One directory, one JSON: the save merges into the run it saved into."""
        w = self._load()
        run = seed_run(self.tmp / "run_y", step3=STEP3)
        w.tpa_step3_edit.setText(str(run))
        w._tpa_autosave()
        written = sorted(run.glob("calib_step6v2_meas_*.csv"))
        self.assertEqual(len(written), 1, w.tpa_status.text())
        target = written[0]

        self.assertTrue(target.is_file(), w.tpa_status.text())
        jsons = list(run.glob("*.json"))
        self.assertEqual([p.name for p in jsons], ["calib.json"],
                         "a run directory holds exactly one JSON")

        payload = json.loads(jsons[0].read_text(encoding="utf-8"))
        self.assertEqual(sorted(payload), ["schema", "step3", "step6"])
        self.assertIn("transfer_fits", payload["step3"])
        # The section points back at the sweep it was fitted from.
        self.assertEqual(payload["step6"]["csv"], target.name)

        models = load_pair_models(str(jsons[0]))
        self.assertEqual(sorted(models), [2, 3, 4, 5, 6])
        # What the page showed is exactly what it wrote; see
        # test_table_shows_the_published_etas for why this is not RECORDED_ETA.
        self.assertAlmostEqual(models[5].eta, w.tpa_fits[3].eta, places=12)

        self.assertEqual(
            list(run.glob("*.png")), [],
            "no PNGs: Load re-fits the CSV the section names and draws it",
        )

    def test_save_without_a_run_parks_the_rows_rather_than_losing_them(self) -> None:
        """The measurement must never be lost to a missing calibration.

        With the dialog gone there is no path to type, so a sweep whose page
        named no run goes to a dated folder under the calibration directory --
        no section (nothing there to interpret it against), but the hour of
        bench time survives.
        """
        w = self._load()
        parked = self.tmp / "unfiled_home"
        parked.mkdir()
        w.tpa_step3_edit.setText("")
        with unittest.mock.patch.object(
            type(w), "_default_calib_dir", lambda _self: parked
        ):
            w._tpa_autosave()
        rows = list(parked.glob("unfiled_*/calib_step6v2_meas_*.csv"))
        self.assertEqual(len(rows), 1, w.tpa_status.text())
        self.assertFalse(list(rows[0].parent.glob("*.json")))
        self.assertIn("no usable calibration file", w.tpa_status.text())


class LoadAsksTests(_WindowCase):
    """Load asks which calibration to re-fit; the page's own file stays put.

    The field is where a SWEEP writes.  Looking at another night is the usual
    reason to press Load, and it must not quietly re-point that.
    """

    def test_it_fits_the_file_it_was_handed_not_the_one_on_the_page(self) -> None:
        w = self.win
        stale = str(Path("C:/nowhere/at/all/calib.json"))
        w.tpa_step3_edit.setText(stale)
        self.addCleanup(w.tpa_step3_edit.setText, "")
        with tempfile.TemporaryDirectory() as tmp:
            other = seed_run(Path(tmp) / "other_night", step3=STEP3,
                             step6=STEP6, step6_csv=MEAS_CSV)
            with _picks(other / "calib.json"):
                w._tpa_load()
            self.assertTrue(w.tpa_fits, w.tpa_status.text())
            self.assertIn("other_night", w.tpa_status.text())
        self.assertEqual(w.tpa_step3_edit.text(), stale,
                         "a Load must not move where a sweep would write")

    def test_a_cancelled_load_changes_nothing(self) -> None:
        w = self.win
        w.tpa_status.setText("–")
        with _picks(""):
            w._tpa_load()
        self.assertEqual(w.tpa_status.text(), "–")


class RailTests(_WindowCase):
    """The rail ceiling is a page setting now, and it reaches the fit.

    It was the constant ``outliers.BURST_MAX_V``, which is a property of this
    bench at this gain: it has to clear the brightest honest read of the sweep,
    and nobody but the operator knows what that will be.
    """

    def test_the_default_is_the_modules_constant(self) -> None:
        from calibration_module.fit.outliers import BURST_MAX_V

        self.assertAlmostEqual(self.win._tpa_rail_v(), BURST_MAX_V, places=9)

    def test_lowering_it_rejects_reads_the_default_keeps(self) -> None:
        """A rail under the sweep's own cross line takes the cross line out."""
        w = self.win
        self.addCleanup(w.tpa_rail.setValue, w.tpa_rail.value())
        rows = load_meas_csv(MEAS_CSV)

        w.tpa_rail.setValue(300.0)
        self.assertEqual(w._tpa_fit_rows(rows, None), [])
        self.assertEqual(len(w.tpa_fits), len(rows))
        kept = {f.index: len(f.levels) for f in w.tpa_fits}

        # 1 mV is under this sweep's drive levels (~8 mV) and over its dark
        # (~0.09 mV), so the bright half of every pair is railed away.
        w.tpa_rail.setValue(1.0)
        failed = w._tpa_fit_rows(rows, None)
        self.assertTrue(failed, "a rail under the drive must cost reads")
        self.assertLess(len(w.tpa_fits), len(rows))
        for fit in w.tpa_fits:
            self.assertLess(len(fit.levels), kept[fit.index])


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
    def test_step_7_loads_the_saved_run_not_this_page_s_memory(self) -> None:
        """The handoff is a run folder, not the tab's last result.

        Step 7 pins its fringe amplitudes to these etas AND builds its layout
        from the step-3 section beside them, so it takes the one folder it can
        record in its own output -- there is deliberately no path by which a
        step-7 run is driven by something with nothing on disk.
        """
        w = self._load()
        with tempfile.TemporaryDirectory() as tmp:
            run = seed_run(Path(tmp) / "run_h", step3=STEP3)
            w.tpa_step3_edit.setText(str(run))
            w._tpa_autosave()
            w.tpa_phase_step6_edit.setText(str(run))
            layout, models = w._tpa_phase_load_step6()
            self.assertEqual(sorted(models), [2, 3, 4, 5, 6])
            # The etas step 7 pins to are the ones step 6 just wrote, not a
            # historical constant -- the floor is measured per sweep now.
            self.assertAlmostEqual(models[5].eta, w.tpa_fits[3].eta, places=12)
            self.assertEqual(layout.n_channels, 6)
        w.tpa_phase_step6_edit.setText("")

    def test_dropped_q_columns_arrive_as_exact_zeros(self) -> None:
        """v2 does not fit q_x/q_w; PairModel needs them, and 0.0 is the contract."""
        w = self._load()
        model = PairModel.from_pair_v2(w.tpa_fits[3])
        self.assertEqual(model.q_x, 0.0)
        self.assertEqual(model.q_w, 0.0)
        self.assertNotIn("q_x", w.tpa_fits[3].bg)


class PmtCheckboxTests(_WindowCase):
    """The Step-6 tab's "Log PMT temperature" switch.

    On by default, and the sweep must ask it rather than the module default --
    a GUI run that silently recorded nothing is the failure this is guarding.
    """

    def test_it_is_on_by_default(self) -> None:
        self.assertTrue(self.win.tpa_pmt.isChecked())

    def test_the_meta_helpers_bracket_a_sweep(self) -> None:
        from gui.app import _pmt_meta_start, _pmt_meta_end

        class _Pmt:
            resource, period_s = "GPIB0::21::INSTR", 60.0

            def __init__(self):
                self.n = 0

            def read(self):
                self.n += 1
                return {"pmt_ohm_Ohm": f"907{self.n}", "pmt_c": f"27.{self.n}"}

        pmt = _Pmt()
        meta = _pmt_meta_start(pmt)
        self.assertEqual(meta["pmt_c_start"], "27.1")
        self.assertEqual(meta["pmt_resource"], "GPIB0::21::INSTR")
        _pmt_meta_end(pmt, meta)
        self.assertEqual(meta["pmt_c_end"], "27.2")

    def test_no_thermistor_means_no_meta_at_all(self) -> None:
        from gui.app import _pmt_meta_start, _pmt_meta_end

        meta = _pmt_meta_start(None)
        _pmt_meta_end(None, meta)
        self.assertEqual(meta, {})

    def test_a_load_takes_the_meta_from_the_file_not_the_session(self) -> None:
        """Otherwise Save would stamp loaded rows with this session's sweep."""
        w = self.win
        with tempfile.TemporaryDirectory() as tmp:
            run = seed_run(Path(tmp) / "run_meta", step3=STEP3, step6=STEP6,
                           step6_csv=MEAS_CSV)
            w.tpa_meta = {"pmt_c_start": "99.9"}    # left over from a sweep here
            with _picks(run / "calib.json"):
                w._tpa_load()
        self.assertNotIn("pmt_c_start", w.tpa_meta)


if __name__ == "__main__":
    unittest.main()
