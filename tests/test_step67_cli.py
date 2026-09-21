"""Steps 6 and 7 v2 from the command line.

The pair lists are required flags (step 6 ``--pairs``, step 7 ``--targets`` and
``--ref``), ``--out`` moves every output, and the 6 -> 7 -> 8 runner passes
those flags through.  Refits only -- no hardware.
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

from calibration_module.measure.phase_v2 import PhaseV2Config  # noqa: E402

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


class Step6CliTests(_TmpCase):
    def setUp(self):
        super().setUp()
        self.s6 = _load("_s6_cli", STEPS / "calib_step6_v2.py")

    def test_it_will_not_run_without_pairs(self):
        with _quiet(), self.assertRaises(SystemExit) as cm:
            self.s6.main(["--step3", str(STEP3)])
        self.assertEqual(cm.exception.code, 2)

    def test_a_malformed_pair_list_is_rejected(self):
        for bad in ("2,x", ","):
            with _quiet(), self.assertRaises(SystemExit):
                self.s6.main(["--step3", str(STEP3), "--pairs", bad])

    def test_a_sweep_with_no_pairs_stops_before_the_hardware(self):
        with mock.patch.object(self.s6, "connect_slm", side_effect=AssertionError), \
                self.assertRaises(ValueError):
            self.s6._run_sweep(STEP3, fit_after=False)

    def test_refit_takes_the_pairs_and_writes_into_out(self):
        out = self.tmp / "nested" / "w25g20"
        with _quiet():
            rc = self.s6.main(["--step3", str(STEP3), "--pairs", "2,3,4,5,6",
                               "--out", str(out), str(STEP6_CSV)])
        self.assertEqual(rc, 0)
        self.assertEqual(self.s6.PAIR_INDICES, [2, 3, 4, 5, 6])
        self.assertEqual(len(list(out.glob("calib_step6v2_result_*.json"))), 1)
        self.assertEqual(len(list(out.glob("calib_step6v2_pair*_*.png"))), 5)


class Step7CliTests(_TmpCase):
    def setUp(self):
        super().setUp()
        self.s7 = _load("_s7_cli", STEPS / "calib_step7_v2.py")

    def test_it_will_not_run_without_targets_or_ref(self):
        for argv in (["--step6", str(STEP6), "--ref", "2"],
                     ["--step6", str(STEP6), "--targets", "2,3"]):
            with _quiet(), self.assertRaises(SystemExit) as cm:
                self.s7.main(argv)
            self.assertEqual(cm.exception.code, 2)

    def test_a_sweep_with_no_targets_stops_before_the_hardware(self):
        with mock.patch.object(self.s7, "connect_slm", side_effect=AssertionError), \
                self.assertRaises(ValueError):
            self.s7._run_sweep(STEP6, fit_after=False)

    def test_refit_takes_targets_and_ref_and_writes_into_out(self):
        out = self.tmp / "w20g25"
        with _quiet():
            rc = self.s7.main(["--step6", str(STEP6), "--targets", "2,3,4,5,6",
                               "--ref", "2", "--out", str(out), str(STEP7_CSV)])
        self.assertEqual(rc, 0)
        self.assertEqual(self.s7.TGT_INDICES, [2, 3, 4, 5, 6])
        self.assertEqual(self.s7.CONFIG.ref_index, 2)
        self.assertEqual(len(list(out.glob("calib_step7_result_*.json"))), 1)
        self.assertEqual(len(list(out.glob("calib_step7v2_pair*_*.png"))), 4)


class RunnerPassesTheFlagsTests(_TmpCase):
    def setUp(self):
        super().setUp()
        self.runner = _load("_runner_cli", REPO / "src/drafts/calib_step6-8_v2.py")
        self.argv: dict[int, list[str]] = {}

        def fake(n, result):
            def main(argv):
                self.argv.setdefault(n, []).extend(argv)
                if result:
                    (self.tmp / result).write_text("{}", encoding="utf-8")
                return 0
            return main

        self.mods = {
            "calib_step6_v2": SimpleNamespace(
                CALIB_PATH=None, PAIR_INDICES=[],
                main=fake(6, "calib_step6v2_result_0000_0000.json")),
            "calib_step7_v2": SimpleNamespace(
                CALIB_PATH=None, TGT_INDICES=[], CONFIG=PhaseV2Config(ref_index=1),
                main=fake(7, "calib_step7_result_0000_0000.json")),
            "calib_npair_verify": SimpleNamespace(OUT_DIR=None, PAIRS=[], main=fake(8, None)),
        }
        patch = mock.patch.object(self.runner, "_load_step", lambda stem: self.mods[stem])
        patch.start()
        self.addCleanup(patch.stop)

    def _run(self, **kw):
        with _quiet():
            return self.runner.run_sequence(run_dir=self.tmp, step3=STEP3,
                                            pairs=[2, 3, 4], ref_index=2, **kw)

    def test_steps_6_and_7_get_the_pairs_and_the_reference(self):
        self.assertEqual(self._run(stop_at=7), 0)
        self.assertEqual(self.argv[6][2:], ["--pairs", "2,3,4"])
        self.assertEqual(self.argv[7][2:], ["--targets", "2,3,4", "--ref", "2"])

    def test_stop_at_7_skips_the_verify(self):
        self._run(stop_at=7)
        self.assertNotIn(8, self.argv)
        self._run(start_at=8, verify_n=(1,))
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

    def _sweep_kwargs(self, mod, argv):
        """Run main() far enough to see what it would pass to _run_sweep."""
        seen = {}

        def fake_sweep(*a, **kw):
            seen.update(kw)

        with _quiet(), mock.patch.object(mod, "_run_sweep", fake_sweep):
            mod.main(argv)
        return seen

    def test_step6_logs_by_default_and_no_pmt_turns_it_off(self):
        argv = ["--step3", str(STEP3), "--pairs", "2,3"]
        with mock.patch.object(self.s6, "open_pmt") as op:
            self._sweep_kwargs(self.s6, argv)
        self.assertTrue(op.call_args.kwargs["enabled"])

        with mock.patch.object(self.s6, "open_pmt") as op:
            self._sweep_kwargs(self.s6, argv + ["--no-pmt"])
        self.assertFalse(op.call_args.kwargs["enabled"])

    def test_step7_logs_by_default_and_no_pmt_turns_it_off(self):
        argv = ["--step6", str(STEP6), "--targets", "2,3", "--ref", "1"]
        with mock.patch.object(self.s7, "open_pmt") as op:
            self._sweep_kwargs(self.s7, argv)
        self.assertTrue(op.call_args.kwargs["enabled"])

        with mock.patch.object(self.s7, "open_pmt") as op:
            self._sweep_kwargs(self.s7, argv + ["--no-pmt"])
        self.assertFalse(op.call_args.kwargs["enabled"])

    def test_the_period_and_the_resource_reach_open_pmt(self):
        for mod, argv in (
            (self.s6, ["--step3", str(STEP3), "--pairs", "2,3"]),
            (self.s7, ["--step6", str(STEP6), "--targets", "2,3", "--ref", "1"]),
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
                self.s6, ["--step3", str(STEP3), "--pairs", "2,3"])
        self.assertIsNone(seen["pmt"])


if __name__ == "__main__":
    unittest.main()
