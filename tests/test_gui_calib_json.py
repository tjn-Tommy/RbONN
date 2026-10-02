"""Every GUI page that loads a calibration must take a run's calib.json.

The schema migration moved each run's calibration into one ``calib.json`` and
left these paths reading the pre-v3 standalone files, which failed in three
different ways at once: ``load_calibration_result`` raised a bare ``TypeError``
on a v3 file, the encoder's auto-discovery pattern did not match the name
``calib.json``, and the "is this usable?" test was ``intensity_levels is not
None`` -- which a v3 step-3 section never has, its raw sweep being in the CSV.

So these drive the real pages against the real committed runs.  They are cheap
(no hardware, no sweep) and they are the regression guard for a class of bug
whose symptom is a page that silently does nothing.
"""

from __future__ import annotations

import os

# Render Qt/matplotlib headless before either is imported by the app module.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

import sys
import unittest
from pathlib import Path

from PyQt5 import QtWidgets

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from calibration_module.fit.slm_response import (  # noqa: E402
    load_calibration_result,
    load_step2_map,
)

#: A committed run with every section, step 2 through step 8.
RUN = REPO / "src/calib_data/run_0930"

_app: QtWidgets.QApplication | None = None


def setUpModule() -> None:
    global _app
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class LoaderTests(unittest.TestCase):
    """The loader under all of them: a directory, its file, or a legacy file."""

    def test_a_run_directory_and_its_calib_json_load_the_same(self) -> None:
        from_dir = load_calibration_result(RUN)
        from_file = load_calibration_result(RUN / "calib.json")
        self.assertEqual(from_dir.coordinates.size, from_file.coordinates.size)
        self.assertGreater(from_dir.coordinates.size, 0)

    def test_the_step3_section_can_drive_the_encoder_without_its_sweep(self) -> None:
        calib = load_calibration_result(RUN)
        self.assertTrue(calib.encodable)
        self.assertFalse(calib.has_curves, "the sweep belongs in step3_*.csv")
        self.assertEqual(len(calib.transfer_fits), calib.coordinates.size)

    def test_a_calib_json_with_no_step3_says_so(self) -> None:
        import tempfile

        from calibration_module.calib_file import write_section

        with tempfile.TemporaryDirectory() as tmp:
            write_section(tmp, 2, {"min_level": 0, "max_level": 1023}, log=None)
            with self.assertRaisesRegex(ValueError, "no step3 section"):
                load_calibration_result(tmp)

    def test_the_step2_map_comes_back_whole(self) -> None:
        """Its line is in the CSV and its levels in the section; both arrive."""
        m = load_step2_map(RUN)
        self.assertGreater(m.coordinates.size, 1000)
        self.assertEqual(m.coordinates.size, m.wavelength.size)
        self.assertGreater(int(m.max_level), int(m.min_level))


class _WindowCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from gui.app import MainWindow
        cls.win = MainWindow()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.win.close()


class EncodingPageTests(_WindowCase):
    def test_the_default_calib_dir_is_the_one_the_runs_are_in(self) -> None:
        """It pointed at <repo>/calib_data until 2026-09-30, which never existed."""
        directory = self.win._default_calib_dir()
        self.assertTrue(directory.is_dir(), directory)
        self.assertEqual(directory.resolve(), (REPO / "src/calib_data").resolve())

    def test_auto_discovery_finds_a_run_s_calib_json(self) -> None:
        found = self.win._enc_local_calib_path()
        self.assertIsNotNone(found, "no calibration discovered under src/calib_data")
        self.assertEqual(found.name, "calib.json")

    def test_loading_one_builds_the_layout(self) -> None:
        w = self.win
        w._enc_calib_override = load_calibration_result(RUN / "calib.json")
        try:
            w._enc_build_layout()
            self.assertIsNotNone(w.encoding_layout, w.enc_layout_status.text())
            self.assertEqual(w.encoding_layout.n_channels, 10)
        finally:
            w._enc_calib_override = None


class StepPageTests(_WindowCase):
    def test_the_step2_source_accepts_a_run_folder(self) -> None:
        m = self.win._load_step2_source(str(RUN), 0, 1023)
        self.assertGreater(m.coordinates.size, 1000)

    def test_steps_6_and_7_describe_the_run_they_are_pointed_at(self) -> None:
        w = self.win
        w.tpa_step3_edit.setText(str(RUN))
        self.addCleanup(w.tpa_step3_edit.setText, "")
        text = w.tpa_step3_label.text()
        # Beside the title, not a pill under the field: _STEP_STATUS routes it
        # to the page's own status line and _set_status leaves it unstyled.
        self.assertIs(w.tpa_step3_label, w.tpa_status)
        self.assertIsNone(w.tpa_step3_label.property("status"), text)
        self.assertIn("step3", text)            # the stamp chain
        self.assertIn("pairs", text)

        w.tpa_phase_step6_edit.setText(str(RUN))
        self.addCleanup(w.tpa_phase_step6_edit.setText, "")
        self.assertNotIn("Cannot", w.tpa_phase_step6_label.text())
        self.assertNotIn("no step", w.tpa_phase_step6_label.text())

    def test_the_page_resolves_its_calib_file(self) -> None:
        """The file named here IS what gets the section.

        There is no Save dialog any more -- a finished sweep writes itself into
        this file and its folder -- so this is the only thing standing between a
        sweep and the wrong calibration.  A typed DIRECTORY resolves to the
        calib.json inside it, exactly as the runners' --calib does.
        """
        w = self.win
        w.tpa_step3_edit.setText(str(RUN))
        self.addCleanup(w.tpa_step3_edit.setText, "")
        self.assertEqual(Path(w._tpa_step3_path()).resolve(),
                         (RUN / "calib.json").resolve())

        w.tpa_step3_edit.setText(str(RUN / "calib.json"))
        self.assertEqual(Path(w._tpa_step3_path()).resolve(),
                         (RUN / "calib.json").resolve())

        w.tpa_step3_edit.setText("")
        self.assertIsNone(w._tpa_step3_path())

    def test_step_8_reads_the_forward_model_out_of_a_run_folder(self) -> None:
        """A directory is not is_file(), which left this panel silently idle."""
        w = self.win
        w.step8_step7_edit.setText(str(RUN))
        self.addCleanup(w.step8_step7_edit.setText, "")
        self.assertIsNotNone(w.step8_y_max, w.step8_status.text())
        self.assertGreater(float(w.step8_y_max), 0.0)
        self.assertIn("Y_max", w.step8_status.text())


class PerPageCalibFileTests(_WindowCase):
    """Every step page names its own calib.json, and gates on that one.

    There was a single shared field in the calibration tab's header until
    2026-10-01, built to stop step 7's phases being pinned to one night's etas
    while step 8 drove another night's model.  It did stop that, by making six
    pages agree by force -- which also meant that looking at last night's step 6
    re-pointed step 8, and that is the common case, not the mix-up.  The file is
    the argument now, as it is on the command line, and each page takes one.
    """

    def test_each_page_has_its_own_field(self) -> None:
        w = self.win
        edits = [w.tpa_step3_edit, w.tpa_phase_step6_edit, w.step8_step7_edit]
        self.assertEqual(len({id(e) for e in edits}), 3)

        w.tpa_step3_edit.setText(str(RUN))
        self.addCleanup(w.tpa_step3_edit.setText, "")
        self.assertEqual(w.tpa_phase_step6_edit.text(), "")
        self.assertEqual(w.step8_step7_edit.text(), "")

    def test_a_step_is_disabled_until_its_file_has_what_it_needs(self) -> None:
        """Said before the click: a step 7 with no step 6 used to spend the
        sweep and only then refuse to fit it."""
        import json
        import tempfile

        w = self.win
        payload = json.loads((RUN / "calib.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            thin = Path(tmp) / "run_thin"
            thin.mkdir()
            (thin / "calib.json").write_text(json.dumps(
                {k: v for k, v in payload.items()
                 if not k.startswith("step") or k in ("step2", "step3")}),
                encoding="utf-8")
            for edit in (w.tpa_step3_edit, w.tpa_phase_step6_edit,
                         w.step8_step7_edit):
                edit.setText(str(thin))
                self.addCleanup(edit.setText, "")

            self.assertTrue(w.tpa_run_button.isEnabled(), "step 3 is there")
            self.assertFalse(w.tpa_phase_run_button.isEnabled())
            self.assertIn("step6", w.tpa_phase_run_button.toolTip())
            self.assertFalse(w.step8_run_button.isEnabled())
            self.assertIn("step7", w.step8_run_button.toolTip())

    def test_an_empty_field_disables_its_own_page_and_says_so(self) -> None:
        w = self.win
        for edit in (w.tpa_step3_edit, w.tpa_phase_step6_edit,
                     w.step8_step7_edit):
            edit.setText("")
        for button in (w.tpa_run_button, w.tpa_phase_run_button,
                       w.step8_run_button):
            self.assertFalse(button.isEnabled())
        self.assertIn("No calibration file", w.tpa_step3_label.text())

    def test_an_empty_field_leaves_load_alone(self) -> None:
        """Load asks which file to re-fit, so the field cannot be its gate.

        Gating it meant you could not open a calibration until you had already
        named a different one -- and the field is where a SWEEP writes, which is
        a question Load never asks.
        """
        w = self.win
        for edit in (w.tpa_step3_edit, w.tpa_phase_step6_edit,
                     w.step8_step7_edit):
            edit.setText("")
        for button in (w.tpa_load_button, w.tpa_phase_load_button,
                       w.step8_load_button):
            self.assertTrue(button.isEnabled())

    def test_browse_starts_in_the_calibration_directory(self) -> None:
        """src/calib_data, where the folders are: the dialog is how one is picked."""
        import unittest.mock

        w = self.win
        seen = {}

        def open_spy(_parent, _title, start, _filt):
            seen["open"] = start
            return ("", "")

        def save_spy(_parent, _title, start, _filt):
            seen["save"] = start
            return ("", "")

        edit = QtWidgets.QLineEdit()
        with unittest.mock.patch.object(QtWidgets.QFileDialog,
                                        "getOpenFileName",
                                        staticmethod(open_spy)):
            w._browse_calib_into(edit)
        with unittest.mock.patch.object(QtWidgets.QFileDialog,
                                        "getSaveFileName",
                                        staticmethod(save_spy)):
            w._browse_new_calib_into(edit)

        home = w._default_calib_dir().resolve()
        self.assertEqual(Path(seen["open"]).resolve(), home)
        # Step 1 CREATES the file, so its dialog offers a name in that folder.
        self.assertEqual(Path(seen["save"]).resolve(), home / "calib.json")


if __name__ == "__main__":
    unittest.main()
