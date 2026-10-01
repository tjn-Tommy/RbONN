"""GUI Step 7 on the v2 fit -- page wiring, no hardware.

These build a real MainWindow offscreen and drive the page's own methods, so
they cover the wiring that ``fit/phase.py``'s unit tests cannot: that the sweep
controls build the drive the offline runner builds, that the table shows what
the fit produced, that a row selection reaches the fringe, and that the file the
Save button writes is the one step 8 reads.

Nothing here touches an instrument.  The sweep path needs an SLM and a DAQ and
is not exercised; everything downstream of the rows is, by feeding the page the
same committed measurement CSV the offline runner refits.
"""

from __future__ import annotations

import json
import os

# Render Qt/matplotlib headless before either is imported by the app module.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

import sys
import tempfile
import threading
import unittest
import unittest.mock
from pathlib import Path

import numpy as np
from PyQt5 import QtWidgets

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from calibration_module.measure.phase_v2 import (  # noqa: E402
    DEFAULT_ACQ,
    DEFAULT_CONFIG,
    PhaseV2Config,
    PhaseV2Progress,
    build_xw_sweep,
    run_seconds,
)

RUN = REPO / "src/calib_data/archive/2026-09/run_0907_1724"
STEP6 = RUN / "calib_step6v2_result_0907_1757.json"
MEAS_CSV = RUN / "calib_step7_meas_0907_1808.csv"
RESULT_JSON = RUN / "calib_step7_result_0907_1808.json"

#: That run swept pairs 3-6 against reference 2.
REF_INDEX = 2

#: What the 0907 run published, keyed by target pair.
# The archived step-7 JSONs in src/calib_data predate the project's single
# sign convention (2026-09-25): they store -Phi_k, the cosine having been
# fitted as cos(dPhi_comb - dPhi_SLM).  Nothing in src/ negates anything any
# more -- a fresh fit stores Phi_k outright -- so a fixture built from one of
# those files has to negate it here to mean what it meant when it was written.
RECORDED_DPHI = {
    c["tgt_index"]: -c["fit"]["dphi_comb_rad"]
    for c in json.loads(RESULT_JSON.read_text(encoding="utf-8"))["step7"]["channels"]
}

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
        """Point the page at the recorded run and re-fit its CSV.

        The reference is deliberately NOT set here: a recorded CSV carries its
        own, and the page is supposed to take it from the file.
        """
        w = self.win
        w.tpa_phase_step6_edit.setText(str(STEP6))
        with unittest.mock.patch.object(
            QtWidgets.QFileDialog, "getOpenFileName",
            staticmethod(lambda *a, **k: (str(MEAS_CSV), "")),
        ):
            w._tpa_phase_load()
        self.assertTrue(w.tpa_phase_results, w.tpa_phase_status.text())
        return w


class SweepControlsTests(_WindowCase):
    """The four sweep controls build the drive, and it is the runner's drive."""

    def _cfg(self, lo: float, hi: float, n: int, ref: float) -> PhaseV2Config:
        w = self.win
        w.tpa_phase_sweep_min.setValue(lo)
        w.tpa_phase_sweep_max.setValue(hi)
        w.tpa_phase_points.setValue(n)
        w.tpa_phase_ref_level.setValue(ref)
        return w._tpa_phase_config()

    def tearDown(self) -> None:
        self._cfg(0.10, 0.90, 10, 0.90)

    def test_defaults_are_the_validated_drive(self) -> None:
        cfg = self.win._tpa_phase_config()
        self.assertEqual(
            (cfg.sweep_min, cfg.sweep_max, cfg.n_points, cfg.ref_level),
            (DEFAULT_CONFIG.sweep_min, DEFAULT_CONFIG.sweep_max,
             DEFAULT_CONFIG.n_points, DEFAULT_CONFIG.ref_level),
        )

    def test_the_controls_reach_the_ramp(self) -> None:
        drive = build_xw_sweep(self._cfg(0.20, 0.80, 4, 0.75))
        self.assertEqual([round(x, 6) for x, _, _, _ in drive],
                         [0.2, 0.4, 0.6, 0.8])
        self.assertTrue(all(x == w for x, w, _, _ in drive),
                        "the target's two channels are swept together")
        self.assertEqual({(xr, wr) for _, _, xr, wr in drive}, {(0.75, 0.75)})

    def test_the_recorded_run_is_reproduced_by_the_defaults(self) -> None:
        """The committed CSV's ramp is what these controls build, untouched."""
        import csv as _csv

        with open(MEAS_CSV, newline="", encoding="utf-8") as f:
            rows = [r for r in _csv.DictReader(f)
                    if int(float(r["tgt_index"])) == 3]
        drive = build_xw_sweep(self.win._tpa_phase_config())
        self.assertEqual([round(float(r["x_t"]), 6) for r in rows],
                         [round(x, 6) for x, _, _, _ in drive])
        self.assertEqual({round(float(r["x_r"]), 6) for r in rows},
                         {DEFAULT_CONFIG.ref_level})

    def test_a_half_typed_ramp_falls_back_instead_of_raising(self) -> None:
        """A spinbox mid-edit must not stop the page redrawing."""
        w = self.win
        w.tpa_phase_sweep_min.setValue(0.90)
        w.tpa_phase_sweep_max.setValue(0.20)     # min > max, momentarily
        cfg = w._tpa_phase_config()
        self.assertEqual(cfg.sweep_min, DEFAULT_CONFIG.sweep_min)
        self.assertEqual(cfg.sweep_max, DEFAULT_CONFIG.sweep_max)

    def test_the_ramp_rejects_a_drive_the_fit_cannot_use(self) -> None:
        for kw in ({"n_points": 1}, {"sweep_min": 0.9, "sweep_max": 0.2},
                   {"sweep_min": 0.0}, {"sweep_max": 1.5}, {"ref_level": 0.0}):
            with self.subTest(**kw), self.assertRaises(ValueError):
                PhaseV2Config(**kw)


class AcquisitionSettingsTests(_WindowCase):
    def test_invert_is_on_by_default(self) -> None:
        self.assertTrue(self.win.tpa_phase_invert.isChecked())
        self.assertTrue(self.win._tpa_phase_acq().invert)

    def test_the_invert_checkbox_reaches_the_acq_config(self) -> None:
        w = self.win
        w.tpa_phase_invert.setChecked(False)
        self.assertFalse(w._tpa_phase_acq().invert)
        w.tpa_phase_invert.setChecked(True)
        self.assertTrue(w._tpa_phase_acq().invert)

    def test_default_window_is_the_bench_s(self) -> None:
        acq = self.win._tpa_phase_acq()
        self.assertEqual(acq.t_meas_s, DEFAULT_ACQ.t_meas_s)
        self.assertEqual(acq.settle_s, DEFAULT_ACQ.settle_s)

    def test_range_is_step_6_s(self) -> None:
        """One range for both steps -- and for step 8, and for the dark.

        Step 7 used to start a range above step 6 because it is the brightest
        step: the reference is on for every point.  Brightness no longer picks
        the range.  It is fixed at the transimpedance amplifier's own 1 V limit,
        so a read too big for it is a flat-top rather than a range problem, and
        step 7's numbers are pinned to step 6's etas -- which means the two have
        to have been read the same way.
        """
        self.assertEqual(self.win._tpa_phase_acq().range_v, DEFAULT_ACQ.range_v)
        self.assertEqual(self.win._tpa_phase_acq().range_v,
                         self.win._tpa_acq().range_v)

    def test_the_two_pages_still_have_their_own_boxes(self) -> None:
        """Same DEFAULT, separate widgets: step 7 is overridable on its own.

        The two agree by default, which is the point -- but a bench needs to be
        able to depart from it on one page without silently moving the other.
        """
        w = self.win
        before = w.tpa_tmeas.value()
        w.tpa_phase_tmeas.setValue(3.5)
        self.assertEqual(w.tpa_tmeas.value(), before)
        w.tpa_phase_tmeas.setValue(DEFAULT_ACQ.t_meas_s)


class StepSixInputTests(_WindowCase):
    def tearDown(self) -> None:
        self.win.tpa_phase_step6_edit.setText("")

    def test_run_refuses_without_one(self) -> None:
        w = self.win
        w.tpa_phase_step6_edit.setText("")
        w._tpa_phase_run()
        self.assertIn("Step-6", w.tpa_phase_status.text())

    def test_run_refuses_a_path_that_is_not_there(self) -> None:
        w = self.win
        w.tpa_phase_step6_edit.setText(str(RUN / "nope.json"))
        w._tpa_phase_run()
        self.assertIn("not found", w.tpa_phase_status.text())

    def test_a_valid_file_is_described_before_the_run(self) -> None:
        w = self.win
        w.tpa_phase_step6_edit.setText(str(STEP6))
        text = w.tpa_phase_step6_label.text()
        self.assertIn(STEP6.name, text)
        self.assertIn("6 pairs", text)
        self.assertIn("2-6", text)          # the etas it actually carries

    def test_a_step_6_json_without_step_3_is_refused(self) -> None:
        """A bare pair summary cannot build a layout, so it cannot drive a run."""
        w = self.win
        with tempfile.TemporaryDirectory() as tmp:
            bare = Path(tmp) / "bare.json"
            payload = json.loads(STEP6.read_text(encoding="utf-8"))
            bare.write_text(json.dumps({"step6": payload["step6"]}),
                            encoding="utf-8")
            w.tpa_phase_step6_edit.setText(str(bare))
            self.assertIn("step3", w.tpa_phase_step6_label.text())
            w._tpa_phase_run()
            self.assertIn("step3", w.tpa_phase_status.text())

    def test_the_layout_is_the_one_step_6_measured_under(self) -> None:
        """Same file, same loader, same encoding marker as the offline runner."""
        w = self.win
        w.tpa_phase_step6_edit.setText(str(STEP6))
        layout, models = w._tpa_phase_load_step6()
        sys.path.insert(0, str(REPO / "src/calibration_module/steps"))
        import calib_step7_v2 as runner
        self.assertEqual(layout.n_channels,
                         runner._load_layout(STEP6).n_channels)
        self.assertEqual(sorted(models), [2, 3, 4, 5, 6])


class ResultsTableTests(_WindowCase):
    def test_table_reproduces_the_published_phases(self) -> None:
        w = self._load()
        t = w.tpa_phase_table
        self.assertEqual(t.rowCount(), 4)
        for row in range(t.rowCount()):
            k = int(t.item(row, 0).text())
            self.assertEqual(t.item(row, 1).text(),
                             f"{np.degrees(RECORDED_DPHI[k]):+.2f}")

    def test_the_fit_itself_reproduces_the_recorded_run(self) -> None:
        """To 1e-12 rad, not bit-for-bit -- the archive was fitted mirrored.

        It used to be bit-identical.  The 0907 run minimised
        ``cos(dPhi_comb - dPhi_SLM)`` and stored ``-Phi_k``; the fitter now
        minimises ``cos(dPhi_SLM + dPhi_comb)``, the project's one convention.
        The two cost functions are equal for all ``dPhi``, so the recovered
        phase is the same number, but ``least_squares`` walks a mirrored
        parameter and stops a couple of ulps away: ~2e-10 rad, or 1e-8 deg.
        """
        w = self._load()
        for k, result in w.tpa_phase_results.items():
            self.assertAlmostEqual(result.fit.dphi_comb, RECORDED_DPHI[k],
                                   places=8)

    def test_the_error_is_split_into_its_two_sources(self) -> None:
        """a and b are pinned, so the fitter's own error cannot see the eta term."""
        w = self._load()
        t = w.tpa_phase_table
        for row in range(t.rowCount()):
            total = float(t.item(row, 2).text())
            fringe = float(t.item(row, 3).text())
            eta = float(t.item(row, 4).text())
            self.assertGreater(total, fringe)
            self.assertAlmostEqual(total, np.hypot(fringe, eta), places=1)

    def test_the_reference_arm_is_common_to_every_target(self) -> None:
        w = self._load()
        t = w.tpa_phase_table
        a_col = {t.item(r, 5).text() for r in range(t.rowCount())}
        self.assertEqual(len(a_col), 1, "a = eta_ref * g_ref, one reference")

    def test_row_selection_drives_the_fringe(self) -> None:
        w = self._load()
        for row, k in enumerate(sorted(w.tpa_phase_results)):
            w.tpa_phase_table.selectRow(row)
            w._tpa_phase_redraw()
            self.assertIn(f"Pair {k}", w.tpa_phase_fig.axes[0].get_title())

    def test_the_fringe_is_safe_with_no_result(self) -> None:
        w = self.win
        w.tpa_phase_results = {}
        w.tpa_phase_table.setRowCount(0)
        w._tpa_phase_redraw()          # must not raise
        self.assertTrue(w.tpa_phase_fig.axes)

    def test_the_reference_is_never_fit_against_itself(self) -> None:
        w = self._load()
        self.assertNotIn(REF_INDEX, w.tpa_phase_results)


class RecordedReferenceTests(_WindowCase):
    """A CSV records the reference it was swept against; that reference wins.

    The spinbox is a setting for the NEXT sweep. Letting it decide how an
    existing file is read is how a re-fit ends up asking step 6 for a pair that
    was never the reference -- which fails for every target at once, with an
    error that looks like arithmetic rather than like a mismatched input.
    """

    def test_the_page_does_not_need_to_be_told_the_reference(self) -> None:
        w = self.win
        w.tpa_phase_ref.setValue(1)          # wrong: this run used 2
        w.tpa_phase_step6_edit.setText(str(STEP6))
        with unittest.mock.patch.object(
            QtWidgets.QFileDialog, "getOpenFileName",
            staticmethod(lambda *a, **k: (str(MEAS_CSV), "")),
        ):
            w._tpa_phase_load()
        self.assertEqual(sorted(w.tpa_phase_results), [3, 4, 5, 6])
        self.assertIn("vs ref 2", w.tpa_phase_status.text())

    def test_the_spinbox_is_moved_to_what_was_used(self) -> None:
        """The page must never show a reference the results are not against."""
        w = self.win
        w.tpa_phase_ref.setValue(1)
        w.tpa_phase_step6_edit.setText(str(STEP6))
        with unittest.mock.patch.object(
            QtWidgets.QFileDialog, "getOpenFileName",
            staticmethod(lambda *a, **k: (str(MEAS_CSV), "")),
        ):
            w._tpa_phase_load()
        self.assertEqual(w.tpa_phase_ref.value(), REF_INDEX)

    def test_a_missing_reference_eta_is_one_error_naming_the_cause(self) -> None:
        w = self.win
        with tempfile.TemporaryDirectory() as tmp:
            payload = json.loads(STEP6.read_text(encoding="utf-8"))
            payload["step6"]["channels"] = [
                c for c in payload["step6"]["channels"]
                if c["index"] != REF_INDEX
            ]
            bad = Path(tmp) / "no_reference.json"
            bad.write_text(json.dumps(payload), encoding="utf-8")
            w.tpa_phase_step6_edit.setText(str(bad))
            with unittest.mock.patch.object(
                QtWidgets.QFileDialog, "getOpenFileName",
                staticmethod(lambda *a, **k: (str(MEAS_CSV), "")),
            ):
                w._tpa_phase_load()
        text = w.tpa_phase_status.text()
        self.assertIn(f"reference {REF_INDEX}", text)
        self.assertIn("no η", text)
        self.assertNotIn("fit(s) failed", text)
        w.tpa_phase_step6_edit.setText(str(STEP6))

    def test_the_runner_reads_the_same_reference_off_the_same_file(self) -> None:
        from calibration_module.fit.phase import reference_in_csv

        self.assertEqual(reference_in_csv(MEAS_CSV), REF_INDEX)


class SaveRoundTripTests(_WindowCase):
    def test_saved_json_carries_the_whole_chain(self) -> None:
        """Step 8 needs this one file: layout, etas, and the phases."""
        w = self._load()
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "calib_step7_meas_0910_2025.csv"
            with unittest.mock.patch.object(
                QtWidgets.QFileDialog, "getSaveFileName",
                staticmethod(lambda *a, **k: (str(target), "")),
            ):
                w._tpa_phase_save()

            self.assertTrue(target.is_file(), w.tpa_phase_status.text())
            js_path = target.with_name("calib_step7_result_0910_2025.json")
            self.assertTrue(js_path.is_file(), w.tpa_phase_status.text())
            self.assertEqual(len(list(target.parent.glob("*.json"))), 1,
                             "the JSON is named like the script's, not the CSV")

            payload = json.loads(js_path.read_text(encoding="utf-8"))
            self.assertEqual(sorted(payload),
                             ["encoding", "step3", "step6", "step7"])
            self.assertIn("transfer_fits", payload["step3"])
            self.assertEqual(payload["step7"]["ref_index"], REF_INDEX)
            self.assertTrue(payload["step7"]["single_beam_bg"])

            got = {c["tgt_index"]: c["fit"]["dphi_comb_rad"]
                   for c in payload["step7"]["channels"]}
            self.assertEqual(sorted(got), sorted(RECORDED_DPHI))
            for k, want in RECORDED_DPHI.items():
                # places=8, not equality: see
                # test_the_fit_itself_reproduces_the_recorded_run.
                self.assertAlmostEqual(got[k], want, places=8)

            self.assertEqual(
                len(list(target.parent.glob("calib_step7_pair*_0910_2025.png"))), 4,
                "one fringe PNG per target, same renderer as the page",
            )

    def test_the_rows_survive_a_missing_step_6_file(self) -> None:
        """An hour of bench time must never be lost to an unreadable input."""
        w = self._load()
        w.tpa_phase_step6_edit.setText("")
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "rows_only.csv"
            with unittest.mock.patch.object(
                QtWidgets.QFileDialog, "getSaveFileName",
                staticmethod(lambda *a, **k: (str(target), "")),
            ):
                w._tpa_phase_save()
            self.assertTrue(target.is_file())
            self.assertFalse(list(target.parent.glob("*.json")))
            self.assertIn("no Step-6 result", w.tpa_phase_status.text())
        w.tpa_phase_step6_edit.setText(str(STEP6))

    def test_the_saved_csv_re_fits_to_the_same_phases(self) -> None:
        w = self._load()
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "roundtrip.csv"
            with unittest.mock.patch.object(
                QtWidgets.QFileDialog, "getSaveFileName",
                staticmethod(lambda *a, **k: (str(target), "")),
            ):
                w._tpa_phase_save()
            with unittest.mock.patch.object(
                QtWidgets.QFileDialog, "getOpenFileName",
                staticmethod(lambda *a, **k: (str(target), "")),
            ):
                w._tpa_phase_load()
        for k, result in w.tpa_phase_results.items():
            self.assertAlmostEqual(result.fit.dphi_comb, RECORDED_DPHI[k],
                                   places=6)


class ProgressDialogTests(_WindowCase):
    """A run takes tens of minutes, so it reports into the shared popup."""

    def tearDown(self) -> None:
        if self.win.calibration_dialog is not None:
            self.win.calibration_dialog.close()
            self.win.calibration_dialog = None

    def _progress(self, step: int, total: int = 11, single: bool = False):
        return PhaseV2Progress(
            step=step, total=total, tgt_index=3, ref_index=REF_INDEX,
            x_t=0.5, w_t=0.5, x_r=0.9, w_r=0.9,
            mean_v=0.0252, std_v=7e-4, range_v=0.2, single=single,
            duration_s=10.0,
        )

    def test_no_inline_progress_bar_on_the_page(self) -> None:
        self.assertFalse(hasattr(self.win, "tpa_phase_progress_bar"))

    def test_dialog_shows_the_step_7_phase_and_tracks_the_bar(self) -> None:
        w = self.win
        w._open_calibration_dialog(on_stop=w._tpa_phase_stop)
        w._on_tpa_phase_progress(self._progress(1, single=True))
        self.assertEqual(w.calibration_dialog.phase_label.text(),
                         "Step 7 \N{MIDDLE DOT} Comb phase sweep")
        self.assertIn("dark", w.calibration_dialog.status_label.text())
        w._on_tpa_phase_progress(self._progress(7))
        self.assertEqual(w.calibration_dialog.progress_bar.value(), 7)
        self.assertEqual(w.calibration_dialog.progress_bar.maximum(), 11)
        self.assertIn("ETA", w.calibration_dialog.eta_label.text())
        self.assertIn("pair 3", w.calibration_dialog.status_label.text())

    def test_stop_button_reaches_this_page_not_the_full_calibration(self) -> None:
        w = self.win
        w.tpa_phase_stop_event = threading.Event()
        w._open_calibration_dialog(on_stop=w._tpa_phase_stop)
        w.calibration_dialog.stop_button.click()
        self.assertTrue(w.tpa_phase_stop_event.is_set())
        w.tpa_phase_stop_event = None

    def test_finish_freezes_the_window_but_leaves_it_open(self) -> None:
        w = self.win
        w._open_calibration_dialog(on_stop=w._tpa_phase_stop)
        w._on_tpa_phase_progress(self._progress(11))
        w._tpa_phase_close_dialog(True, "Done · 4 targets")
        self.assertFalse(w.calibration_dialog.stop_button.isEnabled())
        self.assertTrue(w.calibration_dialog.close_button.isEnabled())
        self.assertIn("Done", w.calibration_dialog.status_label.text())

    def test_a_rejected_run_pops_no_dialog(self) -> None:
        w = self.win
        w.calibration_dialog = None
        w.tpa_phase_step6_edit.setText("")
        w._tpa_phase_run()
        self.assertIsNone(w.calibration_dialog)


class RunPlanTests(_WindowCase):
    """What Run would drive, checked without driving it."""

    def test_the_reference_is_dropped_from_the_target_list(self) -> None:
        w = self.win
        w.tpa_phase_ref.setValue(2)
        targets = [k for k in w._tpa_parse_pairs("2-6") if k != 2]
        self.assertEqual(targets, [3, 4, 5, 6])

    def test_run_refuses_when_only_the_reference_is_targeted(self) -> None:
        w = self.win
        w.tpa_phase_step6_edit.setText(str(STEP6))
        w.tpa_phase_ref.setValue(2)
        w.tpa_phase_targets.setText("2")
        w._tpa_phase_run()
        self.assertIn("not the reference", w.tpa_phase_status.text())
        w.tpa_phase_targets.setText("2-6")

    def test_run_refuses_a_pair_the_step_6_file_has_no_eta_for(self) -> None:
        w = self.win
        w.tpa_phase_step6_edit.setText(str(STEP6))
        w.tpa_phase_ref.setValue(2)
        w.tpa_phase_targets.setText("1,3")     # the file covers 2-6
        w._tpa_phase_run()
        self.assertIn("No step-6", w.tpa_phase_status.text())
        self.assertIn("2-6", w.tpa_phase_status.text())
        w.tpa_phase_targets.setText("2-6")

    def test_run_refuses_a_pair_the_layout_does_not_have(self) -> None:
        w = self.win
        w.tpa_phase_step6_edit.setText(str(STEP6))
        w.tpa_phase_ref.setValue(2)
        w.tpa_phase_targets.setText("3,9")     # the layout has 6 pairs
        w._tpa_phase_run()
        self.assertIn("out of range", w.tpa_phase_status.text())
        w.tpa_phase_targets.setText("2-6")

    def test_the_run_length_is_the_runner_s(self) -> None:
        w = self.win
        drive = build_xw_sweep(w._tpa_phase_config())
        acq = w._tpa_phase_acq()
        secs = run_seconds(drive, acq)
        self.assertAlmostEqual(
            secs, acq.t_meas_s + 10 * acq.t_meas_s + 11 * acq.settle_s,
            places=6)


class CascadeTests(_WindowCase):
    """A run whose target list includes the reference's neighbours.

    The 0916 ten-pair run is the case: pairs 1 and 3 cannot be swept against
    reference 2, so the page has to plan two stages, drive each against its own
    reference, and report everything against 2.  It was done by hand at the
    time, and ``calib_step7_result_0916_merged_ref2.json`` is what that produced
    -- the numbers here are checked against it.
    """

    RUN10 = REPO / "src/calib_data/run_0916_10pairs_w15g35"
    STEP6_10 = RUN10 / "calib.json"
    MERGED = RUN10 / "calib.json"

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._tmp = tempfile.TemporaryDirectory()
        # The two committed stage CSVs are one cascaded run; the GUI writes
        # them as a single file now, so stitch them into one to load it back.
        head = (cls.RUN10 / "calib_step7_meas_0916_2015.csv").read_text(
            encoding="utf-8").splitlines()
        tail = (cls.RUN10 / "calib_step7_meas_0916_2028.csv").read_text(
            encoding="utf-8").splitlines()
        cls.CASCADE_CSV = Path(cls._tmp.name) / "calib_step7_meas_cascade.csv"
        cls.CASCADE_CSV.write_text("\n".join(head + tail[1:]) + "\n",
                                   encoding="utf-8")
        # Negated: see RECORDED_DPHI -- this archive predates the single
        # sign convention and stores -Phi_k.
        cls.MERGED_DPHI = {
            c["tgt_index"]: -c["fit"]["dphi_comb_rad"]
            for c in json.loads(cls.MERGED.read_text(encoding="utf-8"))
            ["step7"]["channels"]
        }

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()
        super().tearDownClass()

    def _page(self, targets: str, ref: int = 2):
        w = self.win
        w.tpa_phase_step6_edit.setText(str(self.STEP6_10))
        w.tpa_phase_ref.setValue(ref)
        w.tpa_phase_targets.setText(targets)
        return w

    def tearDown(self) -> None:
        self.win.tpa_phase_targets.setText("2-6")
        self.win.tpa_phase_ref.setValue(2)
        self.win.tpa_phase_step6_edit.setText(str(STEP6))

    # ---- planning ------------------------------------------------------
    def test_the_page_plans_the_two_stages_the_run_actually_used(self) -> None:
        w = self._page("1-10")
        _cfg, _layout, _models, stages = w._tpa_phase_plan()
        self.assertEqual([(s.ref, s.targets) for s in stages],
                         [(2, (4, 5, 6, 7, 8, 9, 10)), (5, (1, 3))])

    def test_a_list_with_no_neighbours_still_plans_one_stage(self) -> None:
        _cfg, _layout, _models, stages = self._page("4-10")._tpa_phase_plan()
        self.assertEqual(len(stages), 1)
        self.assertEqual(stages[0].ref, 2)

    def test_run_refuses_a_list_no_hub_can_bridge(self) -> None:
        w = self._page("1-3")
        w._tpa_phase_run()
        text = w.tpa_phase_status.text()
        self.assertIn("1, 3", text)
        self.assertIn("5-10", text)            # says what would fix it

    def test_a_neighbour_is_fine_once_something_can_bridge_it(self) -> None:
        _cfg, _layout, _models, stages = self._page("1,2,5")._tpa_phase_plan()
        self.assertEqual([(s.ref, s.targets) for s in stages],
                         [(2, (5,)), (5, (1,))])

    def test_a_typo_is_named_as_a_typo_not_as_an_unreachable_pair(self) -> None:
        """Out-of-range and missing-η come first: they are the likelier fault."""
        w = self._page("1,3,99")
        w._tpa_phase_run()
        self.assertIn("out of range", w.tpa_phase_status.text())

    # ---- loading a recorded cascade ------------------------------------
    def _load_cascade(self):
        w = self._page("1-10")
        with unittest.mock.patch.object(
            QtWidgets.QFileDialog, "getOpenFileName",
            staticmethod(lambda *a, **k: (str(self.CASCADE_CSV), "")),
        ):
            w._tpa_phase_load()
        self.assertTrue(w.tpa_phase_results, w.tpa_phase_status.text())
        return w

    def test_one_file_carrying_two_references_loads_as_one_run(self) -> None:
        w = self._load_cascade()
        self.assertEqual(sorted(w.tpa_phase_results), list(range(1, 11))[:1]
                         + [3, 4, 5, 6, 7, 8, 9, 10])
        self.assertEqual(w.tpa_phase_root, 2)
        self.assertEqual(w.tpa_phase_ref.value(), 2)

    def test_each_target_is_re_fit_against_the_reference_it_was_swept_on(self) -> None:
        w = self._load_cascade()
        self.assertEqual(w.tpa_phase_results[1].ref_index, 5)
        self.assertEqual(w.tpa_phase_results[4].ref_index, 2)

    def test_the_loaded_phases_are_the_hand_merged_ones(self) -> None:
        w = self._load_cascade()
        for k, chain in sorted(w.tpa_phase_chains.items()):
            self.assertAlmostEqual(chain.dphi, self.MERGED_DPHI[k], places=6,
                                   msg=f"pair {k}")

    def test_the_table_shows_the_root_referred_phase_not_the_hub_one(self) -> None:
        w = self._load_cascade()
        row = sorted(w.tpa_phase_results).index(1)
        shown = float(w.tpa_phase_table.item(row, 1).text())
        self.assertAlmostEqual(shown, np.degrees(self.MERGED_DPHI[1]), places=2)
        # and NOT the hop it was measured over, which is a different number
        self.assertNotAlmostEqual(
            shown, np.degrees(w.tpa_phase_results[1].fit.dphi_comb), places=1)

    def test_the_quality_columns_still_describe_the_measured_hop(self) -> None:
        w = self._load_cascade()
        row = sorted(w.tpa_phase_results).index(1)
        fit = w.tpa_phase_results[1].fit
        self.assertAlmostEqual(float(w.tpa_phase_table.item(row, 7).text()),
                               fit.r2, places=5)
        self.assertAlmostEqual(float(w.tpa_phase_table.item(row, 5).text()),
                               fit.a * 1e3, places=3)

    def test_a_chained_row_says_where_it_came_from(self) -> None:
        w = self._load_cascade()
        tip = w.tpa_phase_table.item(sorted(w.tpa_phase_results).index(1),
                                     0).toolTip()
        self.assertIn("hub 5", tip)
        direct = w.tpa_phase_table.item(sorted(w.tpa_phase_results).index(4),
                                        0).toolTip()
        self.assertIn("directly", direct)

    def test_the_error_bars_are_the_chained_ones(self) -> None:
        w = self._load_cascade()
        row = sorted(w.tpa_phase_results).index(1)
        chain, fit = w.tpa_phase_chains[1], w.tpa_phase_results[1].fit
        self.assertAlmostEqual(float(w.tpa_phase_table.item(row, 2).text()),
                               np.degrees(chain.err_total), places=2)
        # bigger than the hop alone: the hub's own fringe noise is in there
        self.assertGreater(chain.err_total, fit.dphi_comb_err_total)

    def test_the_run_path_fits_each_stage_against_its_own_reference(self) -> None:
        """What _tpa_phase_finished does with a sweep's results, minus the sweep.

        The load path rebuilds its stages from the file; this is the other
        caller, where the stages come from the planner and each target must
        still be fit against the reference its stage drove.
        """
        w = self._load_cascade()
        results = dict(w.tpa_phase_results)
        for r in results.values():             # discard the fits, keep the rows
            r.fit = None
        _cfg, _layout, models, stages = self._page("1-10")._tpa_phase_plan()
        failed = w._tpa_phase_fit_results(results, models, 2, stages)
        self.assertEqual(failed, [])
        self.assertEqual(w.tpa_phase_chains[1].via, 5)
        self.assertEqual(w.tpa_phase_chains[4].via, 2)
        for k, chain in sorted(w.tpa_phase_chains.items()):
            self.assertAlmostEqual(chain.dphi, self.MERGED_DPHI[k], places=6,
                                   msg=f"pair {k}")

    def test_a_pair_is_not_chained_onto_a_hub_that_never_fit(self) -> None:
        """A stage-2 pair with a dead hub has no route home, and says so."""
        w = self._load_cascade()
        results = dict(w.tpa_phase_results)
        for r in results.values():
            r.fit = None
        _cfg, _layout, models, stages = self._page("1-10")._tpa_phase_plan()
        broken = {k: v for k, v in models.items() if k != 5}
        failed = w._tpa_phase_fit_results(results, broken, 2, stages)
        self.assertIn(5, [int(s.split()[1].rstrip(":")) for s in failed])
        self.assertNotIn(1, w.tpa_phase_chains)
        self.assertIn(4, w.tpa_phase_chains)

    # ---- saving --------------------------------------------------------
    def test_the_saved_json_is_rooted_at_the_reference_and_keeps_the_path(self) -> None:
        from calibration_module.fit.phase import load_comb_phase_json
        w = self._load_cascade()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "calib_step7_meas_x.csv"
            with unittest.mock.patch.object(
                QtWidgets.QFileDialog, "getSaveFileName",
                staticmethod(lambda *a, **k: (str(out), "")),
            ):
                w._tpa_phase_save()
            js = out.with_name("calib_step7_result_x.json")
            self.assertTrue(js.is_file(), w.tpa_phase_status.text())
            payload = json.loads(js.read_text(encoding="utf-8"))

            step7 = payload["step7"]
            self.assertEqual(step7["ref_index"], 2)
            self.assertEqual(
                [(s["ref"], tuple(s["targets"])) for s in step7["cascade"]["stages"]],
                [(2, (4, 5, 6, 7, 8, 9, 10)), (5, (1, 3))])

            by_pair = {c["tgt_index"]: c["fit"] for c in step7["channels"]}
            for k, want in self.MERGED_DPHI.items():
                self.assertAlmostEqual(by_pair[k]["dphi_comb_rad"], want,
                                       places=6, msg=f"pair {k}")
            self.assertEqual(by_pair[1]["chained"]["via"], 5)
            self.assertNotIn("chained", by_pair[4])

            # the file step 8 reads, read the way step 8 reads it
            ref, entries = load_comb_phase_json(js, method="fixed_comb_only")
            self.assertEqual(ref, 2)
            self.assertEqual(sorted(entries), [1, 3, 4, 5, 6, 7, 8, 9, 10])

    def test_the_saved_csv_still_records_the_reference_per_target(self) -> None:
        from calibration_module.fit.phase import references_in_csv
        w = self._load_cascade()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "calib_step7_meas_x.csv"
            with unittest.mock.patch.object(
                QtWidgets.QFileDialog, "getSaveFileName",
                staticmethod(lambda *a, **k: (str(out), "")),
            ):
                w._tpa_phase_save()
            self.assertEqual(references_in_csv(out)[1], 5)
            self.assertEqual(references_in_csv(out)[4], 2)

    def test_a_single_reference_run_records_no_cascade_block(self) -> None:
        """Nothing changes for a run that never needed a hub."""
        w = self._load()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "calib_step7_meas_x.csv"
            with unittest.mock.patch.object(
                QtWidgets.QFileDialog, "getSaveFileName",
                staticmethod(lambda *a, **k: (str(out), "")),
            ):
                w._tpa_phase_save()
            payload = json.loads(
                out.with_name("calib_step7_result_x.json").read_text("utf-8"))
            self.assertNotIn("cascade", payload["step7"])
            for ch in payload["step7"]["channels"]:
                self.assertNotIn("chained", ch["fit"])


class PmtCheckboxTests(_WindowCase):
    """The Step-7 tab has its own switch, on by default, beside its own DAQ box."""

    def test_it_is_on_by_default(self) -> None:
        self.assertTrue(self.win.tpa_phase_pmt.isChecked())

    def test_it_is_not_the_step6_checkbox(self) -> None:
        """Step 7 has its own DAQ settings, so it has its own switch too."""
        w = self.win
        self.assertIsNot(w.tpa_phase_pmt, w.tpa_pmt)
        w.tpa_phase_pmt.setChecked(False)
        self.assertTrue(w.tpa_pmt.isChecked())
        w.tpa_phase_pmt.setChecked(True)


class LiveSweepBurstTests(_WindowCase):
    """A live sweep must drop bursts too, not only a loaded CSV.

    _tpa_phase_fit_results is the path a finished sweep takes: it fits the
    in-memory PhaseResult from measure_target and never sees load_phase_csv.
    Without the guard there, the same sweep would fit one way on the bench and
    another way when refit from the CSV it had just written.
    """

    BURST_RUN = REPO / "src/calib_data/run_0929"
    BURST_CSV = BURST_RUN / "calib_step7_meas_0930_0236.csv"
    BURST_S6 = BURST_RUN / "calib.json"

    def _swept(self, tgt: int, ref: int):
        """The sweep as measure_target hands it over: bursts still in."""
        from calibration_module.fit.phase import load_pair_models, load_phase_csv
        if not (self.BURST_CSV.is_file() and self.BURST_S6.is_file()):
            self.skipTest(f"missing run_0929 fixture: {self.BURST_CSV}")
        models = load_pair_models(self.BURST_S6)
        raw = load_phase_csv(self.BURST_CSV, models[tgt], models[ref],
                             comb_only=True, single_beam_bg=True,
                             only_tgt=tgt, drop_bursts=False)
        return raw, models

    def test_the_finished_sweep_path_drops_them(self) -> None:
        w = self.win
        raw, models = self._swept(10, 2)
        n = raw.trial.size
        failed = w._tpa_phase_fit_results({10: raw}, models, 2)
        self.assertEqual(failed, [])
        kept = w.tpa_phase_results[10]
        self.assertEqual(len(kept.dropped_bursts), 2)
        self.assertEqual(kept.trial.size, n - 2)
        # the point of the whole exercise: the fit now reads as the good fit it is
        self.assertGreater(kept.fit.r2, 0.95)

    def test_it_agrees_with_the_same_sweep_refit_from_its_csv(self) -> None:
        from calibration_module.fit.phase import load_phase_csv
        w = self.win
        raw, models = self._swept(6, 2)
        w._tpa_phase_fit_results({6: raw}, models, 2)
        live = w.tpa_phase_results[6]
        viacsv = load_phase_csv(self.BURST_CSV, models[6], models[2],
                                comb_only=True, single_beam_bg=True, only_tgt=6)
        self.assertAlmostEqual(live.fit.dphi_comb, viacsv.fit.dphi_comb, places=9)
        self.assertAlmostEqual(live.fit.r2, viacsv.fit.r2, places=9)

    def test_a_clean_sweep_is_untouched(self) -> None:
        w = self.win
        raw, models = self._swept(4, 2)
        n = raw.trial.size
        w._tpa_phase_fit_results({4: raw}, models, 2)
        self.assertEqual(w.tpa_phase_results[4].dropped_bursts, [])
        self.assertEqual(w.tpa_phase_results[4].trial.size, n)

    def test_the_table_and_fringe_still_fill(self) -> None:
        """The panel must repopulate after a sweep that dropped rows."""
        w = self.win
        raw, models = self._swept(10, 2)
        w._tpa_phase_fit_results({10: raw}, models, 2)
        w._tpa_phase_fill_table()
        w._tpa_phase_redraw()
        self.assertEqual(w.tpa_phase_table.rowCount(), 1)
        self.assertEqual(w.tpa_phase_table.item(0, 0).text(), "10")
        self.assertEqual(len(w.tpa_phase_fig.axes), 2)


if __name__ == "__main__":
    unittest.main()
