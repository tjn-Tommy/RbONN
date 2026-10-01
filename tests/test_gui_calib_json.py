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
        self.assertEqual(w.tpa_step3_label.property("status"), "ok", text)
        self.assertIn("step3", text)            # the stamp chain
        self.assertIn("pairs", text)

        w.tpa_phase_step6_edit.setText(str(RUN))
        self.addCleanup(w.tpa_phase_step6_edit.setText, "")
        self.assertNotEqual(w.tpa_phase_step6_label.property("status"), "error",
                            w.tpa_phase_step6_label.text())

    def test_step_8_reads_the_forward_model_out_of_a_run_folder(self) -> None:
        """A directory is not is_file(), which left this panel silently idle."""
        w = self.win
        w.step8_step7_edit.setText(str(RUN))
        self.addCleanup(w.step8_step7_edit.setText, "")
        self.assertIsNotNone(w.step8_y_max, w.step8_status.text())
        self.assertGreater(float(w.step8_y_max), 0.0)
        self.assertIn("Y_max", w.step8_levels_note.text())


if __name__ == "__main__":
    unittest.main()
