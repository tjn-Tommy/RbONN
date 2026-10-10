"""Steps 6 and 7 v2 from the command line.

``--calib`` names the one calib.json every step reads and writes back into --
a directory is accepted as shorthand for the file inside it -- and the pair
lists are required flags (step 6 ``--pairs``, step 7 ``--targets`` and
``--ref``).  The 6 -> 7 -> 8 runner passes those through.  Refits only -- no
hardware.

The archived fixtures are frozen in the pre-v3 shape on purpose, so the folders
here are seeded from them (see ``calib_fixtures.seed_run``).
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibration_module.measure.phase_v2 import PhaseV2Config  # noqa: E402
from calib_fixtures import seed_run  # noqa: E402

RUN = REPO / "src/calib_data/archive/2026-09/run_0908_1444"
STEP3 = RUN / "calib_step3c_0907_1358_pad10.json"
STEP6_CSV = RUN / "calib_step6v2_meas_0908_1517.csv"
STEP6 = RUN / "calib_step6v2_result_0908_1517.json"
STEP7_CSV = RUN / "calib_step7_meas_0908_1528.csv"
STEPS = REPO / "src/calibration_module/steps"


def _load(name: str, path: Path):
    """A fresh copy of a script, so one test's rebound constants never leak."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@contextlib.contextmanager
def _quiet():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


class _TmpCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def run3(self) -> Path:
        """A run directory carrying only the archived step-3 section."""
        return seed_run(self.tmp / "run3", step3=STEP3)

    def run6(self) -> Path:
        """A run directory carrying the archived step-3 AND step-6 sections."""
        return seed_run(self.tmp / "run6", step3=STEP6, step6=STEP6)


class Step6CliTests(_TmpCase):
    def setUp(self):
        super().setUp()
        self.s6 = _load("_s6_cli", STEPS / "calib_step6_v2.py")

    def test_it_will_not_run_without_pairs(self):
        with _quiet(), self.assertRaises(SystemExit) as cm:
            self.s6.main(["--calib", str(self.tmp)])
        self.assertEqual(cm.exception.code, 2)

    def test_a_malformed_pair_list_is_rejected(self):
        for bad in ("2,x", ","):
            with _quiet(), self.assertRaises(SystemExit):
                self.s6.main(["--calib", str(self.tmp), "--pairs", bad])

    def test_a_sweep_with_no_pairs_stops_before_the_hardware(self):
        with mock.patch.object(self.s6, "connect_slm", side_effect=AssertionError), \
                self.assertRaises(ValueError):
            self.s6._run_sweep(self.run3(), fit_after=False)

    def test_refit_writes_the_section_beside_the_calib_file(self):
        run = self.run3()
        with _quiet():
            rc = self.s6.main(["--calib", str(run), "--pairs", "2,3,4,5,6",
                               str(STEP6_CSV)])
        self.assertEqual(rc, 0)
        self.assertEqual(self.s6.PAIR_INDICES, [2, 3, 4, 5, 6])
        self.assertEqual([p.name for p in run.glob("*.json")], ["calib.json"])
        self.assertEqual(list(run.glob("*.png")), [],
                         "no pictures: the GUI's Load draws the fit from the "
                         "CSV the section names")

        from calibration_module.calib_file import load_calib
        calib = load_calib(run)
        self.assertEqual(calib.steps(), [3, 6])
        self.assertEqual(calib.csv_path(6), STEP6_CSV.resolve())


class Step7CliTests(_TmpCase):
    def setUp(self):
        super().setUp()
        self.s7 = _load("_s7_cli", STEPS / "calib_step7_v2.py")

    def test_it_will_not_run_without_targets_or_ref(self):
        for argv in (["--calib", str(self.tmp), "--ref", "2"],
                     ["--calib", str(self.tmp), "--targets", "2,3"]):
            with _quiet(), self.assertRaises(SystemExit) as cm:
                self.s7.main(argv)
            self.assertEqual(cm.exception.code, 2)

    def test_a_sweep_with_no_targets_stops_before_the_hardware(self):
        with mock.patch.object(self.s7, "connect_slm", side_effect=AssertionError), \
                self.assertRaises(ValueError):
            self.s7._run_sweep(self.run6(), fit_after=False)

    def test_refit_writes_the_section_beside_the_calib_file(self):
        run = self.run6()
        with _quiet():
            rc = self.s7.main(["--calib", str(run), "--targets", "2,3,4,5,6",
                               "--ref", "2", str(STEP7_CSV)])
        self.assertEqual(rc, 0)
        self.assertEqual(self.s7.TGT_INDICES, [2, 3, 4, 5, 6])
        self.assertEqual(self.s7.CONFIG.ref_index, 2)
        self.assertEqual([p.name for p in run.glob("*.json")], ["calib.json"])
        self.assertEqual(list(run.glob("*.png")), [],
                         "no pictures: the GUI's Load draws the fit from the "
                         "CSV the section names")

        from calibration_module.calib_file import load_calib
        self.assertEqual(load_calib(run).steps(), [3, 6, 7])


class RunnerPassesTheFlagsTests(_TmpCase):
    def setUp(self):
        super().setUp()
        self.runner = _load("_runner_cli", REPO / "src/drafts/calib_step6-8_v2.py")
        self.argv: dict[int, list[str]] = {}

        def fake(n, writes_section):
            """A step that records its argv and writes the section it would.

            It has to write a real section: the chain checks for one after each
            step, because a sweep can succeed while its fit fails and that used
            to surface only as a confusing error one step later.
            """
            from calibration_module.calib_file import write_section

            def main(argv):
                self.argv.setdefault(n, []).extend(argv)
                if writes_section:
                    run = Path(argv[argv.index("--calib") + 1])
                    write_section(run, n, {"fake": True}, log=None)
                return 0
            return main

        self.mods = {
            "calib_step6_v2": SimpleNamespace(
                PAIR_INDICES=[], main=fake(6, True)),
            "calib_step7_v2": SimpleNamespace(
                TGT_INDICES=[], CONFIG=PhaseV2Config(ref_index=1),
                main=fake(7, True)),
            "calib_step8": SimpleNamespace(PAIRS=[], main=fake(8, True)),
        }
        patch = mock.patch.object(self.runner, "_load_step", lambda stem: self.mods[stem])
        patch.start()
        self.addCleanup(patch.stop)

    def _run(self, **kw):
        """One run folder across calls, so --start-at resumes a real one."""
        if not hasattr(self, "_run_dir"):
            self._run_dir = self.run3()
        with _quiet():
            return self.runner.run_sequence(run_dir=self._run_dir,
                                            pairs=[2, 3, 4], ref_index=2, **kw)

    def test_steps_6_and_7_get_the_run_folder_the_pairs_and_the_reference(self):
        self.assertEqual(self._run(stop_at=7), 0)
        calib = str(self._run_dir / "calib.json")
        self.assertEqual(self.argv[6], ["--calib", calib, "--pairs", "2,3,4"])
        self.assertEqual(self.argv[7], ["--calib", calib,
                                        "--targets", "2,3,4", "--ref", "2"])

    def test_stop_at_7_skips_the_verify_and_8_resumes_from_the_sections(self):
        self.assertEqual(self._run(stop_at=7), 0)
        self.assertNotIn(8, self.argv)
        # Steps 6 and 7 left their sections behind, which is the only thing
        # --start-at 8 needs; nothing was carried between the two calls.
        self.assertEqual(self._run(start_at=8, verify_n=(1,)), 0)
        self.assertIn(8, self.argv)


class PmtFlagTests(_TmpCase):
    """--no-pmt and --pmt-period on both steps.

    Logging is ON by default, so the flag under test is the one that turns it
    off; what a sweep then hands ``_run_sweep`` is the thing worth asserting,
    because everything downstream keys off ``pmt is None``.
    """

    def setUp(self):
        super().setUp()
        self.s6 = _load("_s6_pmt", STEPS / "calib_step6_v2.py")
        self.s7 = _load("_s7_pmt", STEPS / "calib_step7_v2.py")
        self.r3 = self.run3()
        self.r6 = self.run6()

    def _sweep_kwargs(self, mod, argv):
        """Run main() far enough to see what it would pass to _run_sweep."""
        seen = {}

        def fake_sweep(*a, **kw):
            seen.update(kw)

        with _quiet(), mock.patch.object(mod, "_run_sweep", fake_sweep):
            mod.main(argv)
        return seen

    def test_step6_logs_by_default_and_no_pmt_turns_it_off(self):
        argv = ["--calib", str(self.r3), "--pairs", "2,3"]
        with mock.patch.object(self.s6, "open_pmt") as op:
            self._sweep_kwargs(self.s6, argv)
        self.assertTrue(op.call_args.kwargs["enabled"])

        with mock.patch.object(self.s6, "open_pmt") as op:
            self._sweep_kwargs(self.s6, argv + ["--no-pmt"])
        self.assertFalse(op.call_args.kwargs["enabled"])

    def test_step7_logs_by_default_and_no_pmt_turns_it_off(self):
        argv = ["--calib", str(self.r6), "--targets", "2,3", "--ref", "1"]
        with mock.patch.object(self.s7, "open_pmt") as op:
            self._sweep_kwargs(self.s7, argv)
        self.assertTrue(op.call_args.kwargs["enabled"])

        with mock.patch.object(self.s7, "open_pmt") as op:
            self._sweep_kwargs(self.s7, argv + ["--no-pmt"])
        self.assertFalse(op.call_args.kwargs["enabled"])

    def test_the_period_and_the_resource_reach_open_pmt(self):
        for mod, argv in (
            (self.s6, ["--calib", str(self.r3), "--pairs", "2,3"]),
            (self.s7, ["--calib", str(self.r6), "--targets", "2,3", "--ref", "1"]),
        ):
            with self.subTest(mod=mod.__name__):
                with mock.patch.object(mod, "open_pmt") as op:
                    self._sweep_kwargs(mod, argv + ["--pmt-period", "5",
                                                    "--pmt-resource", "GPIB0::9::INSTR"])
                self.assertEqual(op.call_args.args, ("GPIB0::9::INSTR", 5.0))

    def test_a_missing_2400_does_not_stop_a_sweep(self):
        """open_pmt swallows it, so _run_sweep is still reached with pmt=None."""
        # patched where open_pmt LOOKS it up, not where the step imported it
        with mock.patch("calibration_module.measure.pmt_temp.PmtTemperature",
                        side_effect=RuntimeError("no VISA")):
            seen = self._sweep_kwargs(
                self.s6, ["--calib", str(self.r3), "--pairs", "2,3"])
        self.assertIsNone(seen["pmt"])


if __name__ == "__main__":
    unittest.main()
