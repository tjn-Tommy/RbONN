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
RECORDED_DPHI = {
    c["tgt_index"]: c["fit"]["dphi_comb_rad"]
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

    def test_checkboxes_reach_the_acq_config(self) -> None:
        w = self.win
        for box, field in ((w.tpa_phase_invert, "invert"),
                           (w.tpa_phase_autorange, "autorange")):
            box.setChecked(False)
            self.assertFalse(getattr(w._tpa_phase_acq(), field))
            box.setChecked(True)
            self.assertTrue(getattr(w._tpa_phase_acq(), field))

    def test_default_windows_are_the_v2_validated_ones(self) -> None:
        acq = self.win._tpa_phase_acq()
        self.assertEqual(acq.t_both_s, DEFAULT_ACQ.t_both_s)
        self.assertEqual(acq.t_single_s, DEFAULT_ACQ.t_single_s)
        self.assertEqual(acq.settle_s, DEFAULT_ACQ.settle_s)

    def test_range_starts_a_step_above_step_6(self) -> None:
        """Step 7 is the brightest step: the reference is on for every point."""
        self.assertEqual(self.win._tpa_phase_acq().range_v, DEFAULT_ACQ.range_v)
        self.assertGreater(self.win._tpa_phase_acq().range_v,
                           self.win._tpa_acq().range_v)

    def test_windows_are_not_shared_with_step_6(self) -> None:
        """Step 6's T_both is bound to the DAQ Monitor page; step 7's is its own."""
        w = self.win
        before = w.tpa_tboth.value()
        w.tpa_phase_tboth.setValue(3.5)
        self.assertEqual(w.tpa_tboth.value(), before)
        w.tpa_phase_tboth.setValue(DEFAULT_ACQ.t_both_s)


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

    def test_the_fit_itself_is_bit_identical_to_the_recorded_run(self) -> None:
        w = self._load()
        for k, result in w.tpa_phase_results.items():
            self.assertEqual(result.fit.dphi_comb, RECORDED_DPHI[k])

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
            self.assertEqual(got, RECORDED_DPHI)

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
        secs = run_seconds(drive, w._tpa_phase_acq())
        self.assertAlmostEqual(secs, 10.0 + 10 * 10.0 + 11 * 0.25, places=6)


if __name__ == "__main__":
    unittest.main()
