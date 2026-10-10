"""GUI Step 3 -- page wiring, no hardware.

These build a real MainWindow offscreen and drive the page's own methods, so
they cover what the transfer-fit unit tests in ``test_slm_transfer_fit.py``
cannot: that the table shows one row per channel of the sweep that was loaded,
that the points marked on the plot are the ones the fit dropped, that a row
selection reaches the plot, and that Load writes nothing.

Nothing here touches an instrument.  The sweep path needs an SLM and a DAQ and
is not exercised; everything downstream of the CSV is, by feeding the page the
same committed step-3 sweep the offline script refits
(``drafts/plot_step3_transfer.py``).
"""

from __future__ import annotations

import contextlib
import os

# Render Qt/matplotlib headless before either is imported by the app module.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from PyQt5 import QtWidgets

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibration_module.fit.slm_response import (  # noqa: E402
    CalibrationResult,
    load_intensity_calibration_csv,
)
from calib_fixtures import seed_run  # noqa: E402

#: The 0907 Step-3c sweep: 12 channels x 45 levels, and the one in the repo with
#: a spike in it -- channel 11 carries the 25x cell the fitter has to drop.
STEP3_JSON = REPO / "src/calib_data/archive/2026-09/run_0908_1444/calib_step3c_0907_1358_pad10.json"
STEP3_CSV = REPO / "src/calib_data/archive/2026-09/run_0907_productcheck_fit/calib_step3c_0907_1358.csv"

#: What that sweep's clip takes, by channel row.  Measured, not chosen: these
#: are the cells the stored calibration was built without.
CLIPPED = {0: 1, 11: 4}

CH, X_PX, WL, CONTRAST, OFF, ON, RMS, MAXABS, CURV, N, FLAGS = range(11)

_app: QtWidgets.QApplication | None = None


def setUpModule() -> None:
    global _app
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@contextlib.contextmanager
def _picks(path):
    """Answer the Load dialog with ``path``.

    Load asks which calibration to re-fit rather than taking the page's field:
    looking at another night is the usual reason to press it, and the field is
    where a SWEEP writes.
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


class EmptyPageTests(_WindowCase):
    """What the page says before anything has been run or loaded."""

    def test_the_plot_is_safe_with_no_sweep(self) -> None:
        self.win._step3_redraw()
        axes = self.win.step3_fig.axes[0]
        self.assertEqual([text.get_text() for text in axes.texts],
                         ["Run or load a sweep"])
        self.assertEqual(self.win.step3_table.rowCount(), 0)

    def test_the_page_has_one_status_line_and_the_table_is_the_selector(self) -> None:
        from gui.app import MainWindow
        self.assertEqual(MainWindow._STEP_STATUS["3c"], "step3_status")
        self.assertIs(self.win.step_widgets["3c"]["status"], self.win.step3_status)
        self.assertIs(self.win._calib_labels["3c"], self.win.step3_status)

    def test_the_legacy_fit_from_csv_panel_is_gone(self) -> None:
        """Its job is this page now: one fit per channel, on the live model."""
        for attribute in ("calibration_path_edit", "wavelength_combo",
                          "save_fit_button", "fit_table", "map_canvas",
                          "map_kind_combo", "calibration_fits", "_fit_backing"):
            self.assertFalse(hasattr(self.win, attribute), attribute)


class LoadTests(_WindowCase):
    """Load re-fits the sweep a calib file names, and only that."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._tmp = tempfile.TemporaryDirectory()
        cls.run_dir = seed_run(Path(cls._tmp.name) / "run_0907",
                               step3=STEP3_JSON, step3_csv=STEP3_CSV)
        cls.coords, cls.waves, cls.levels, cls.curves = (
            load_intensity_calibration_csv(cls.run_dir / STEP3_CSV.name))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()
        super().tearDownClass()

    def _load(self):
        with _picks(self.run_dir / "calib.json"):
            self.win._step3_load()
        return self.win

    def _cell(self, row: int, column: int) -> str:
        return self.win.step3_table.item(row, column).text()

    def test_every_channel_of_the_sweep_gets_a_row(self) -> None:
        w = self._load()
        self.assertEqual(w.step3_table.rowCount(), self.coords.size)
        self.assertEqual(len(w.step3_fits), 12)
        self.assertNotIn(None, w.step3_fits)

    def test_the_coordinate_and_wavelength_columns_come_from_the_csv(self) -> None:
        self._load()
        for row in (0, 5, 11):
            self.assertEqual(self._cell(row, CH), str(row))
            self.assertEqual(self._cell(row, X_PX), f"{self.coords[row]:.1f}")
            self.assertEqual(self._cell(row, WL), f"{self.waves[row]:.3f}")

    def test_the_clip_is_reported_per_channel(self) -> None:
        w = self._load()
        for row, fit in enumerate(w.step3_fits):
            self.assertEqual(fit.n_clipped, CLIPPED.get(row, 0), f"ch{row}")
            flags = self._cell(row, FLAGS)
            if row in CLIPPED:
                self.assertIn(f"{CLIPPED[row]} clipped", flags)
            else:
                self.assertNotIn("clipped", flags)

    def test_the_status_line_counts_the_file_the_channels_and_the_clip(self) -> None:
        w = self._load()
        text = w.step3_status.text()
        self.assertIn("run_0907/calib.json", text)
        self.assertIn(STEP3_CSV.name, text)
        self.assertIn("12 channel(s) re-fit over 45 levels", text)
        self.assertIn(f"{sum(CLIPPED.values())} point(s) clipped", text)

    def test_the_marked_points_are_the_ones_the_fit_dropped(self) -> None:
        """n_clipped and the red markers are the same cells, by construction."""
        w = self._load()
        for row, fit in enumerate(w.step3_fits):
            self.assertEqual(int(w.step3_clipped[row].sum()), fit.n_clipped,
                             f"ch{row}")
        w.step3_table.selectRow(11)
        marked = [line for line in w.step3_fig.axes[0].lines
                  if line.get_marker() == "x"]
        self.assertEqual(len(marked), 1)
        self.assertEqual(len(marked[0].get_xdata()), CLIPPED[11])

    def test_a_row_selection_redraws_for_that_channel(self) -> None:
        w = self._load()
        for row in (3, 7):
            w.step3_table.selectRow(row)
            title = w.step3_fig.axes[0].get_title()
            self.assertIn(f"ch {row}", title)
            self.assertIn(f"{self.coords[row]:.1f} px", title)
            self.assertIn(f"{self.waves[row]:.3f} nm", title)

    def test_the_fitted_curve_is_drawn_over_the_measured_points(self) -> None:
        w = self._load()
        w.step3_table.selectRow(5)
        lines = w.step3_fig.axes[0].lines
        points = [line for line in lines if line.get_marker() == "o"]
        self.assertEqual(len(points), 1)
        self.assertEqual(len(points[0].get_xdata()), self.levels.size)
        curve = [line for line in lines
                 if line.get_marker() in ("", "None") and line.get_linestyle() == "-"]
        self.assertEqual(len(curve), 1)
        self.assertGreater(len(curve[0].get_xdata()), self.levels.size)

    def test_loading_writes_nothing(self) -> None:
        """Opening last week's calibration to look at it must not re-write it."""
        before = {path: path.read_bytes()
                  for path in sorted(self.run_dir.iterdir()) if path.is_file()}
        self._load()
        after = {path: path.read_bytes()
                 for path in sorted(self.run_dir.iterdir()) if path.is_file()}
        self.assertEqual(before, after)

    def test_a_sweep_whose_csv_is_missing_is_reported_not_raised(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = seed_run(Path(tmp) / "run_gone",
                           step3=STEP3_JSON, step3_csv=STEP3_CSV)
            (run / STEP3_CSV.name).unlink()
            with _picks(run / "calib.json"):
                self.win._step3_load()
        self.assertIn("copy the CSV in beside it", self.win.step3_status.text())


class FinishedRunTests(_WindowCase):
    """A finished sweep draws the fits it just stored, not a second fit of them."""

    def test_a_finished_run_fills_the_table_and_says_what_it_wrote(self) -> None:
        w = self.win
        coords, waves, levels, curves = load_intensity_calibration_csv(STEP3_CSV)
        fits, clipped, failed = w._step3_fit_curves(levels, curves)
        self.assertEqual(failed, [])
        result = CalibrationResult(
            wavelength=waves, coordinates=coords, max_level=880, min_level=400,
            level_range=levels, intensity_levels=curves, transfer_fits=fits,
        )
        w._on_step_finished({
            "status": "ok", "step": "3c", "result": result, "saved": None,
            "csv": None, "summary": "12 channels, pitch 20 px",
            "levels": levels, "curves": curves, "clipped": clipped,
        })
        self.assertEqual(w.step3_table.rowCount(), coords.size)
        # The very objects the section was written from.
        self.assertIs(w.step3_fits[0], fits[0])
        self.assertIn("Done", w.step3_status.text())
        self.assertIn("wrote step3", w.step3_status.text())


if __name__ == "__main__":
    unittest.main()
