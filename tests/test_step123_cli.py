"""Steps 1-3 from the command line: one ``--calib FILE``, and what it supplies.

Every runner takes the calib.json itself -- a directory is accepted as shorthand
for the file inside it -- reads its own input out of that one file and writes its
section back into it.  What is exercised here is that plumbing, not the sweeps:
steps 1 and 2 need an OSA and step 3 a DAQ, and none of them is touched.

Step 1 has no input to resolve (it CREATES the file), so what there is to test
about it is that the flag is required.  Step 2's levels are the interesting case:
the file's ``step1`` section, or ``--min-max``, and the second is not a fallback
-- a calibration seeded by hand is complete and never has a step 1 at all.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from calibration_module.calib_file import load_calib, write_section  # noqa: E402

STEPS = REPO / "src/calibration_module/steps"


def _load(name: str, path: Path):
    """Import a step script by path -- they live outside any package."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _quiet_main(mod, argv):
    """``mod.main(argv)`` with argparse's usage text swallowed."""
    with contextlib.redirect_stderr(io.StringIO()), \
            contextlib.redirect_stdout(io.StringIO()):
        return mod.main(argv)


class _TmpCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.calib = self.tmp / "run_x" / "calib.json"


class TheFlagIsRequiredTests(_TmpCase):
    """No default path, for the reason steps 6-8 have none.

    A constant edited by hand drifts out from under the script the moment a run
    is filed away, and a step that writes into the wrong calibration does not
    fail -- it leaves a file that reads as valid.
    """

    def test_every_step_refuses_to_run_without_it(self) -> None:
        for name, script in (("_s1_flag", "calib_step1.py"),
                             ("_s2_flag", "calib_step2.py"),
                             ("_s3_flag", "calib_step3.py")):
            mod = _load(name, STEPS / script)
            with self.subTest(script=script):
                with self.assertRaises(SystemExit) as caught:
                    _quiet_main(mod, [])
                self.assertEqual(caught.exception.code, 2)


class Step2LevelsTests(_TmpCase):
    """Where step 2's off/on levels come from, and what happens when nowhere."""

    def setUp(self) -> None:
        super().setUp()
        self.s2 = _load("_s2_levels", STEPS / "calib_step2.py")

    def test_the_levels_come_out_of_the_files_step1_section(self) -> None:
        write_section(self.calib, 1,
                      {"min_level": 120, "max_level": 880}, log=None)
        seed = self.s2._seed(self.calib, None)
        self.assertEqual((int(seed.min_level), int(seed.max_level)), (120, 880))

    def test_min_max_wins_over_the_section(self) -> None:
        """An override, not a fallback: a typed pair is a decision, and the
        file's step 1 may be the very thing being distrusted."""
        write_section(self.calib, 1,
                      {"min_level": 120, "max_level": 880}, log=None)
        seed = self.s2._seed(self.calib, "400,900")
        self.assertEqual((int(seed.min_level), int(seed.max_level)), (400, 900))

    def test_min_max_stands_alone_with_no_file_at_all(self) -> None:
        """A calibration seeded by hand is complete, and starts here."""
        seed = self.s2._seed(self.calib, "400,900")
        self.assertEqual((int(seed.min_level), int(seed.max_level)), (400, 900))
        self.assertFalse(self.calib.exists(), "resolving a seed writes nothing")

    def test_no_step1_and_no_flag_says_both_ways_out(self) -> None:
        write_section(self.calib, 3, {"marker": "x"}, log=None)
        with self.assertRaises(SystemExit) as caught:
            self.s2._seed(self.calib, None)
        text = str(caught.exception)
        self.assertIn("no step1 section", text)
        self.assertIn("--min-max", text)

    def test_a_malformed_pair_is_rejected_by_name(self) -> None:
        for bad in ("400", "400,", "a,b", "900,400"):
            with self.subTest(given=bad), self.assertRaises(SystemExit):
                self.s2._seed(self.calib, bad)

    def test_only_the_two_levels_are_taken(self) -> None:
        """The sweep reads nothing else off the seed, which is why a hand-given
        pair is as good as a measured one -- see wavelength_calibration."""
        write_section(self.calib, 1,
                      {"min_level": 0, "max_level": 1023,
                       "level_range": [0, 64, 128]}, log=None)
        seed = self.s2._seed(self.calib, None)
        self.assertEqual(seed.coordinates.size, 0)
        self.assertEqual(seed.wavelength.size, 0)


class Step3InputTests(_TmpCase):
    """Step 3's map is the file's step-2 section plus the CSV it points at."""

    def setUp(self) -> None:
        super().setUp()
        self.s3 = _load("_s3_input", STEPS / "calib_step3.py")

    def test_a_file_with_no_step2_is_refused_by_name(self) -> None:
        write_section(self.calib, 1, {"min_level": 0, "max_level": 1023},
                      log=None)
        with self.assertRaises(Exception) as caught:
            self.s3.load_step2(self.calib)
        self.assertIn("step2", str(caught.exception))

    def test_a_directory_means_the_calib_json_inside_it(self) -> None:
        """The runners' one concession to the old run-folder habit."""
        run = REPO / "src/calib_data/archive/2026-09/run_0930"
        if not (run / "calib.json").is_file():           # pragma: no cover
            self.skipTest("run_0930 is not in this checkout")
        here = self.tmp / "copied"
        here.mkdir()
        shutil.copy2(run / "calib.json", here / "calib.json")
        for csv in load_calib(run).csv_paths(2):
            shutil.copy2(csv, here / csv.name)

        by_dir = self.s3.load_step2(here)
        by_file = self.s3.load_step2(here / "calib.json")
        self.assertGreater(by_dir.coordinates.size, 0)
        self.assertEqual(by_dir.coordinates.size, by_file.coordinates.size)


if __name__ == "__main__":
    unittest.main()
