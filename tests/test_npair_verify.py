"""Step 8 n-pair verification: forward model, full scale, CSV, and the collect loop.

Everything runs off the committed 0908_1300 run -- its step-7 result and the
x = w = v step-8 CSV collected against it -- plus fake instruments for the loop.
"""
import contextlib
import io
import sys
import tempfile
import unittest
from unittest import mock
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from calibration_module.fit import verify as V  # noqa: E402
from calibration_module.fit.phase import PairModel  # noqa: E402
from calibration_module.steps import calib_npair_verify as S8  # noqa: E402

RUN = REPO / "src/calib_data/archive/2026-09/run_0908_1300"
STEP7 = RUN / "calib_step7_result_0908_1344.json"
OLD_CSV = RUN / "calib_step8_simple_0908_1351.csv"
PAIRS = [2, 3, 4, 5, 6]


def _quiet():
    return contextlib.redirect_stdout(io.StringIO())


def _bare(eta: float) -> PairModel:
    """A pair with no single-beam background, so its extremes are known exactly."""
    return PairModel(index=0, eta=eta, a_x=0.0, q_x=0.0, a_w=0.0, q_w=0.0, d=0.0)


class ForwardModelTests(unittest.TestCase):
    def setUp(self):
        self.models, self.phases, self.ref = V.load_forward_model(STEP7)

    def test_reads_every_pair_the_run_calibrated(self):
        self.assertEqual(sorted(self.models), PAIRS)
        self.assertEqual(self.ref, 2)
        self.assertEqual(self.phases[2], 0.0)

    def test_v2_phase_is_flipped_into_the_field_convention(self):
        import json
        chans = json.loads(STEP7.read_text(encoding="utf-8"))["step7"]["channels"]
        stored = {c["tgt_index"]: c["fit"]["dphi_comb_rad"] for c in chans}
        for k, dphi in stored.items():
            self.assertEqual(self.phases[k], -dphi)

    def test_a_pair_without_a_phase_is_refused(self):
        with self.assertRaisesRegex(ValueError, r"\[1\]"):
            V.load_forward_model(STEP7, pairs=[1, 3])

    def test_reproduces_the_predictions_step_8_recorded(self):
        # The old step 8 drove x = w = v and wrote its prediction per row; the
        # general model has to give the same numbers on the same drives.
        _, blocks, _ = V.load_verify_csv(OLD_CSV)
        for b in blocks:
            pred = V.predict(self.models, self.phases, b.driven, b.x, b.w)
            np.testing.assert_allclose(pred, b.pred_v, rtol=1e-7)


THREE = {1: _bare(0.2), 2: _bare(0.15), 3: _bare(0.1)}


class FullScaleTests(unittest.TestCase):
    def test_one_pair_reaches_eta_squared(self):
        # A lone pair has no relative phase to lose, so its ceiling is eta^2 and
        # the free-phase bound is exact.
        s = V.full_scale({1: _bare(0.2)}, {1: 0.0}, (1,))
        self.assertAlmostEqual(s.fs, 0.04, places=12)
        self.assertAlmostEqual(s.headroom, 1.0, places=12)

    def test_equal_phases_reach_the_free_phase_bound(self):
        s = V.full_scale(THREE, {1: 0.3, 2: 0.3, 3: 0.3}, (1, 2, 3))
        self.assertAlmostEqual(s.fs, s.bound, places=9)
        self.assertAlmostEqual(s.headroom, 1.0, places=9)
        self.assertAlmostEqual(s.bound, 0.45 ** 2, places=12)

    def test_disagreeing_phases_cannot_reach_it(self):
        s = V.full_scale(THREE, {1: 0.0, 2: 1.3, 3: -2.2}, (1, 2, 3))
        self.assertLess(s.fs, s.bound)
        self.assertAlmostEqual(s.bound, 0.45 ** 2, places=12)

    def test_full_scale_is_the_reachable_ceiling(self):
        # Both halves matter: no pattern may exceed FS, and the drive FS reports
        # must attain it -- checked through predict, which knows nothing about
        # the support scan that produced it.
        phases = {1: 0.0, 2: 1.3, 3: -2.2}
        s = V.full_scale(THREE, phases, (1, 2, 3))
        rng = np.random.default_rng(0)
        x, w = rng.uniform(0, 1, (20000, 3)), rng.uniform(0, 1, (20000, 3))
        y = V.predict(THREE, phases, (1, 2, 3), x, w)
        self.assertLessEqual(y.max(), s.fs + 1e-12)
        drive = np.array(s.drive)
        top = V.predict(THREE, phases, (1, 2, 3), drive, drive)
        self.assertAlmostEqual(float(top), s.fs, places=12)

    def test_the_real_run_ceiling_sits_well_below_s_squared(self):
        models, phases, _ = V.load_forward_model(STEP7)
        s = V.full_scale(models, phases, PAIRS)
        self.assertEqual(s.amplitudes, tuple(models[k].eta for k in PAIRS))
        self.assertAlmostEqual(s.bound, sum(models[k].eta for k in PAIRS) ** 2,
                               places=15)
        # 0908_1300's phases span ~280 deg, so the reachable ceiling is a third
        # of the free-phase bound -- quoting NRMSE against S^2 would flatter
        # every block by ~3x here.
        self.assertLess(s.headroom, 0.4)
        self.assertGreater(s.headroom, 0.3)


class DriveTests(unittest.TestCase):
    def test_one_block_per_n_subset(self):
        blocks = V.blocks_for(range(1, 11), 5)
        self.assertEqual(len(blocks), 252)
        self.assertEqual(blocks[0], (1, 2, 3, 4, 5))
        self.assertEqual(len(V.blocks_for(PAIRS, 2)), 10)

    def test_n_outside_1_to_N_is_refused(self):
        for n in (0, 6):
            with self.assertRaisesRegex(ValueError, r"1\.\.5"):
                V.blocks_for(PAIRS, n)

    def test_patterns_stay_in_the_box_and_repeat_with_the_seed(self):
        x, w = V.random_patterns(10, 2, 8, 0.1, 0.9, seed=7)
        self.assertEqual(x.shape, (10, 8, 2))
        self.assertTrue(((x >= 0.1) & (x <= 0.9) & (w >= 0.1) & (w <= 0.9)).all())
        self.assertFalse(np.array_equal(x, w))
        x2, w2 = V.random_patterns(10, 2, 8, 0.1, 0.9, seed=7)
        np.testing.assert_array_equal(x, x2)
        np.testing.assert_array_equal(w, w2)

    def test_range_follows_the_prediction(self):
        self.assertEqual(S8.range_for(0.0), (0.1, 0.2))
        self.assertEqual(S8.range_for(0.05), (0.1, 0.2))
        self.assertEqual(S8.range_for(0.08), (0.2, 0.5))
        self.assertEqual(S8.range_for(50.0), (10.0, 10.0))


def _rows(sign=1.0):
    x, w = V.random_patterns(2, 2, 3, 0.1, 0.9, seed=1)
    rows = []
    for b, driven in enumerate([(2, 3), (4, 6)]):
        for j in range(3):
            rows.append({"block": b, "driven": driven, "x": x[b, j], "w": w[b, j],
                         "dark_v": 1e-4 * sign, "mean_v": sign * (0.01 + j * 1e-3),
                         "std_v": 2e-4, "range_v": 0.1, "pred_v": 0.01})
    return rows


class CsvTests(unittest.TestCase):
    def test_round_trip(self):
        rows = _rows()
        with tempfile.TemporaryDirectory() as tmp:
            path = V.write_verify_csv(Path(tmp) / "v.csv", PAIRS, rows,
                                      meta={"drive": "min=0.2,max=0.8,patterns=3,seed=1"})
            pairs, blocks, meta = V.load_verify_csv(path)
        self.assertEqual(pairs, PAIRS)
        self.assertEqual([b.driven for b in blocks], [(2, 3), (4, 6)])
        self.assertEqual(V.drive_bounds(meta, (0.0, 1.0)), (0.2, 0.8))
        np.testing.assert_array_equal(blocks[1].x, np.array([r["x"] for r in rows[3:]]))
        np.testing.assert_array_equal(blocks[1].w, np.array([r["w"] for r in rows[3:]]))
        np.testing.assert_allclose(blocks[0].y, [r["mean_v"] - r["dark_v"] for r in rows[:3]])

    def test_old_step_8_csv_loads_block_by_block(self):
        pairs, blocks, meta = V.load_verify_csv(OLD_CSV)
        self.assertEqual(pairs, PAIRS)
        self.assertEqual([b.n for b in blocks], [1] * 5 + [2] * 10)
        self.assertEqual(V.drive_bounds(meta, (0.0, 1.0)), (0.1, 0.9))
        self.assertTrue(all(np.array_equal(b.x, b.w) for b in blocks))
        self.assertTrue(all(np.isnan(b.range_v).all() for b in blocks))

    def test_a_negated_csv_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = V.write_verify_csv(Path(tmp) / "neg.csv", PAIRS, _rows(sign=-1.0), meta={})
            with self.assertRaisesRegex(ValueError, "NEGATIVE"):
                V.load_verify_csv(path)


class CompareTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._out = S8.OUT_DIR
        S8.OUT_DIR = Path(self._tmp.name)

    def tearDown(self):
        S8.OUT_DIR = self._out
        self._tmp.cleanup()

    def test_old_step_8_run_compares_offline(self):
        with _quiet():
            out = S8.compare_csv(OLD_CSV, STEP7)
        self.assertEqual({k: len(v) for k, v in out.items()}, {1: 5, 2: 10})
        for errors in out.values():
            for e in errors:
                self.assertTrue(np.isfinite(e.rms_v))
                self.assertLess(e.rms_v, 2e-3)      # 0908_1300 stays under ~1.2 mV a block
        self.assertEqual(sorted(p.name for p in S8.OUT_DIR.glob("*.png")),
                         [f"{OLD_CSV.stem}_1pair_compare_auto.png", f"{OLD_CSV.stem}_2pair_compare_auto.png"])

    def test_n_selects_one_block_size(self):
        with _quiet():
            out = S8.compare_csv(OLD_CSV, STEP7, n=2)
        self.assertEqual(list(out), [2])
        with _quiet(), self.assertRaisesRegex(ValueError, "no 3-pair blocks"):
            S8.compare_csv(OLD_CSV, STEP7, n=3)

    def test_out_puts_the_pngs_in_a_new_directory(self):
        out = S8.OUT_DIR / "nested" / "dir"
        with _quiet():
            S8.main(["--step7", str(STEP7), str(OLD_CSV), "--n", "1", "--out", str(out)])
        self.assertEqual([p.name for p in out.glob("*.png")],
                         [f"{OLD_CSV.stem}_1pair_compare_auto.png"])
        self.assertEqual(list(S8.OUT_DIR.glob("*.png")), [])

    def test_step7_is_required(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            S8.main([str(OLD_CSV)])

    def test_collecting_needs_n(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            S8.main(["--step7", str(STEP7)])


class _FakeSLM:
    def get_slm_info(self):
        return (64, 8)

    def display_array(self, pattern):
        self.shown = pattern


class _FakeDAQ:
    """Reads the model's own prediction for whatever the SLM shows, TIA-inverted."""

    def __init__(self, slm, models, phases, *, sign=-1.0, dark=1e-4, fail_on=None):
        self.slm, self.models, self.phases = slm, models, phases
        self.sign, self.dark, self.fail_on, self.calls = sign, dark, fail_on, 0

    def monitor_cycle(self, timeout=None, single=False):
        self.calls += 1
        if self.calls == self.fail_on:
            raise KeyboardInterrupt
        x_vals, w_vals = self.slm.shown
        pairs = sorted(self.models)
        slots = [k - S8.PAIR_INDEX_BASE for k in pairs]
        y = float(V.predict(self.models, self.phases, pairs, x_vals[slots], w_vals[slots]))
        return SimpleNamespace(value=self.sign * (y + self.dark), std=1e-5)


def _encode(x_vals, w_vals, layout, width, height):
    return (x_vals.copy(), w_vals.copy())


class CollectLoopTests(unittest.TestCase):
    def setUp(self):
        self.models, self.phases, _ = V.load_forward_model(STEP7)
        self.blocks = V.blocks_for(PAIRS, 2)
        self.x, self.w = V.random_patterns(len(self.blocks), 2, 3, 0.1, 0.9, seed=5)
        self.layout = SimpleNamespace(n_channels=6)
        self.acq = replace(S8.ACQ, settle_s=0.0)

    def _run(self, daq, slm, rows, encode=_encode):
        S8.run_patterns(daq, slm, self.layout, self.blocks, self.x, self.w,
                        self.models, self.phases, rows=rows, acq=self.acq,
                        encode=encode, log=lambda _: None)

    def _errors(self, rows):
        with tempfile.TemporaryDirectory() as tmp:
            path = V.write_verify_csv(Path(tmp) / "run.csv", PAIRS, rows, meta={})
            _, blocks, _ = V.load_verify_csv(path)
        return [V.evaluate(b, self.models, self.phases) for b in blocks]

    def test_a_perfect_bench_reads_zero_error(self):
        slm = _FakeSLM()
        rows = []
        self._run(_FakeDAQ(slm, self.models, self.phases), slm, rows)
        self.assertEqual(len(rows), 30)
        errors = self._errors(rows)
        self.assertEqual([e.driven for e in errors], self.blocks)
        self.assertLess(max(e.nrmse_pct for e in errors), 1e-6)

    def test_the_zero_above_is_not_trivially_zero(self):
        # The perfect-bench zero is only worth something if a wiring slip is not
        # also zero.  Swapping x and w is the hardest slip to see: the TPA term is
        # symmetric in them, so only the single-beam asymmetry (a_x != a_w)
        # moves -- ~0.02 %FS here, far below bench noise, but 10^4 above the
        # perfect-bench floor, which is what this test can honestly claim.
        slm = _FakeSLM()
        rows = []
        swapped = lambda x_vals, w_vals, *a: (w_vals.copy(), x_vals.copy())  # noqa: E731
        self._run(_FakeDAQ(slm, self.models, self.phases), slm, rows, encode=swapped)
        self.assertGreater(max(e.nrmse_pct for e in self._errors(rows)), 1e-3)

    def test_a_double_inverted_signal_stops_the_run(self):
        slm = _FakeSLM()
        with self.assertRaisesRegex(RuntimeError, "BELOW the"):
            self._run(_FakeDAQ(slm, self.models, self.phases, sign=+1.0), slm, [])

    def test_an_interrupted_run_keeps_what_it_read(self):
        slm = _FakeSLM()
        rows = []
        with self.assertRaises(KeyboardInterrupt):
            self._run(_FakeDAQ(slm, self.models, self.phases, fail_on=5), slm, rows)
        self.assertEqual(len(rows), 3)      # read 1 is the dark; 2-4 are patterns


class CollectWritesTheMeasurementCsvTests(unittest.TestCase):
    """collect() end to end on fake instruments: the CSV is the deliverable."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.out = Path(tmp.name)
        models, phases, _ = V.load_forward_model(STEP7)
        self.slm = _FakeSLM()
        self.slm.close_slm = lambda: None
        self.daq = _FakeDAQ(self.slm, models, phases)
        self.daq.disconnect = lambda: None
        for patch in (
            mock.patch.object(S8, "OUT_DIR", self.out),
            mock.patch.object(S8, "PATTERNS", 2),
            mock.patch.object(S8, "SEED", 11),
            mock.patch.object(S8, "connect_slm", lambda *a, **k: self.slm),
            mock.patch.object(S8, "connect_daq", lambda *a, **k: self.daq),
            mock.patch.object(S8, "load_layout", lambda *a, **k: SimpleNamespace(n_channels=6)),
            mock.patch("slm_module.encoding.encode_to_pattern", _encode),
            mock.patch("time.sleep", lambda _: None),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def test_a_run_writes_one_row_per_read(self):
        with _quiet():
            path = S8.collect(STEP7, 2)
        self.assertTrue(Path(path).name.startswith("calib_verify_2pair_"))
        pairs, blocks, meta = V.load_verify_csv(path)
        self.assertEqual(pairs, PAIRS)
        self.assertEqual([b.driven for b in blocks], V.blocks_for(PAIRS, 2))
        self.assertEqual(sum(b.y.size for b in blocks), 10 * 2)
        self.assertEqual(meta["status"], "complete")
        self.assertIn("seed=11", meta["drive"])
        self.assertTrue((self.out / f"{Path(path).stem}_compare_auto.png").is_file())

    def test_out_dir_takes_the_csv_and_png(self):
        out = self.out / "elsewhere"
        with _quiet():
            path = S8.collect(STEP7, 2, out_dir=out)
        self.assertEqual(Path(path).parent, out)
        self.assertTrue((out / f"{Path(path).stem}_compare_auto.png").is_file())
        self.assertEqual(list(self.out.glob("calib_verify_*")), [])

    def test_a_stopped_run_still_writes_what_it_read(self):
        self.daq.fail_on = 6        # read 1 is the dark; reads 2-5 are patterns
        with _quiet(), self.assertRaises(KeyboardInterrupt):
            S8.collect(STEP7, 2)
        (path,) = self.out.glob("calib_verify_2pair_*.csv")
        _, blocks, meta = V.load_verify_csv(path)
        self.assertEqual(sum(b.y.size for b in blocks), 4)
        self.assertTrue(meta["status"].startswith("partial"))


if __name__ == "__main__":
    unittest.main()
