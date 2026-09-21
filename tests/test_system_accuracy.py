"""The system-accuracy run end to end, against a fake bench.

The perfect-bench test is the load-bearing one: a DAQ that hands back exactly
what the model predicts must produce a report whose residuals are zero.  Anything
that quietly corrupts the chain -- the coherent/total mix-up, a single-beam
subtraction applied twice, a forced range that changes the reading, the extra CSV
columns coming back misaligned -- shows up there as a non-zero residual.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calibration_module.fit import levels, levels_report as R, verify as V  # noqa: E402
from calibration_module.steps import calib_npair_verify as S8  # noqa: E402
from calibration_module.steps import system_accuracy as SA  # noqa: E402
from calibration_module.fit import pair_v2 as P6, phase as PH  # noqa: E402
from calibration_module.fit.pair import load_tpa_pair_csv  # noqa: E402
from calibration_module.measure.pmt_temp import open_pmt  # noqa: E402

RUN = ROOT / "src" / "calib_data" / "run_0911_10pairs_w20g30"
STEP7 = RUN / "calib_step7_result_0911_2037.json"


def model():
    if not STEP7.exists():
        raise unittest.SkipTest(f"{STEP7} not in the tree")
    models, phases, _ = V.load_forward_model(STEP7)
    return models, phases, sorted(models)


class _FakeSLM:
    def get_slm_info(self):
        return (64, 8)

    def display_array(self, pattern):
        self.shown = pattern


class _FakeDAQ:
    """Hands back the model's own FULL prediction, TIA-inverted, plus a dark."""

    def __init__(self, slm, models, phases, *, dark=1e-4, noise=0.0, seed=0):
        self.slm, self.models, self.phases = slm, models, phases
        self.dark, self.noise = dark, noise
        self.rng = np.random.default_rng(seed)

    def monitor_cycle(self, timeout=None, single=False):
        x_vals, w_vals = self.slm.shown
        pairs = sorted(self.models)
        slots = [k - S8.PAIR_INDEX_BASE for k in pairs]
        y = float(V.predict(self.models, self.phases, pairs,
                            x_vals[slots], w_vals[slots]))
        if self.noise:
            y += self.noise * self.rng.standard_normal()
        return SimpleNamespace(value=-(y + self.dark), std=1e-5)


def _encode(x_vals, w_vals, layout, width, height):
    return (x_vals.copy(), w_vals.copy())


class BuildTests(unittest.TestCase):
    """The run plan, before any hardware."""

    @classmethod
    def setUpClass(cls):
        # Solving the ladder takes seconds, and every test here reads the same
        # plan, so it is built once.
        cls.models, cls.phases, cls.pairs = model()
        cls.scale, cls.patterns, cls.preds = SA.build(
            cls.models, cls.phases, cls.pairs, seed=101)

    def counts(self):
        return {k: sum(1 for p in self.patterns if p.kind == k) for k in "ABC"}

    def test_every_family_is_present(self):
        c = self.counts()
        self.assertEqual(c["A"], SA.LADDER)
        self.assertEqual(c["B"], len(SA.REPEAT_ALPHAS) * SA.REPEATS)
        self.assertEqual(c["C"], 2 * len(levels.PSI_OFFSETS_DEG))

    def test_the_scan_brackets_the_run(self):
        n = len(levels.PSI_OFFSETS_DEG)
        self.assertTrue(all(p.kind == "C" for p in self.patterns[:n]))
        self.assertTrue(all(p.kind == "C" for p in self.patterns[-n:]))
        self.assertTrue(all(p.label.endswith("@start") for p in self.patterns[:n]))
        self.assertTrue(all(p.label.endswith("@end") for p in self.patterns[-n:]))
        self.assertFalse(any(p.kind == "C" for p in self.patterns[n:-n]))

    def test_the_ladder_is_not_in_level_order(self):
        """Ascending order would alias drift onto the level axis."""
        alpha = [p.alpha for p in self.patterns if p.kind == "A"]
        self.assertFalse(alpha == sorted(alpha))
        # and not merely reversed either
        self.assertGreater(np.mean(np.abs(np.diff(alpha))), 0.2)

    def test_the_repeats_are_spread_through_the_run(self):
        pos = {}
        for i, p in enumerate(self.patterns):
            if p.kind == "B":
                pos.setdefault(p.target_alpha, []).append(i)
        for alpha, where in pos.items():
            self.assertEqual(len(where), SA.REPEATS)
            gaps = np.diff(where)
            # evenly spread: no repeat may sit adjacent to its neighbour
            self.assertGreater(gaps.min(), len(self.patterns) / (2 * SA.REPEATS))

    def test_the_ladder_covers_the_scale(self):
        alpha = np.array([p.alpha for p in self.patterns if p.kind == "A"])
        self.assertLess(alpha.min(), 0.05)
        self.assertGreater(alpha.max(), 0.95)
        self.assertGreater(np.median(alpha), 0.4)
        occupied = len(set(np.clip((alpha * 15).astype(int), 0, 15).tolist()))
        self.assertGreaterEqual(occupied, 15)

    def test_seed_reproduces_the_whole_plan(self):
        _, again, _ = SA.build(self.models, self.phases, self.pairs, seed=101)
        self.assertEqual([p.label for p in again], [p.label for p in self.patterns])
        for a, b in zip(again, self.patterns):
            np.testing.assert_array_equal(a.x, b.x)


class CollectTests(unittest.TestCase):
    """The loop, the CSV and the report, against a bench that cannot be wrong."""

    @classmethod
    def setUpClass(cls):
        cls.models, cls.phases, cls.pairs = model()
        # The fake bench reads instantly, so the whole plan runs here -- a
        # truncated one would cut B down to a single repeat per level.
        cls.scale, cls.patterns, cls.preds = SA.build(
            cls.models, cls.phases, cls.pairs, seed=7)

    def setUp(self):
        self.layout = SimpleNamespace(n_channels=len(self.pairs))
        self.acq = replace(S8.ACQ, settle_s=0.0, t_both_s=0.0, t_single_s=0.0)

    def _collect(self, noise=0.0):
        slm = _FakeSLM()
        daq = _FakeDAQ(slm, self.models, self.phases, noise=noise)
        pats = self.patterns
        x = np.stack([p.x for p in pats])[None]
        w = np.stack([p.w for p in pats])[None]
        coh = [float(V.predict(self.models, self.phases, self.pairs, p.x, p.w,
                               coherent=True)) for p in pats]
        full = [float(V.predict(self.models, self.phases, self.pairs, p.x, p.w))
                for p in pats]
        extra = [[{"kind": p.kind, "label": p.label, "seq": i,
                   "target_alpha": f"{p.target_alpha:.6f}", "alpha": f"{p.alpha:.6f}",
                   "coherent_v": f"{c:.9g}", "sb_pred_v": f"{q - c:.9g}"}
                  for i, (p, c, q) in enumerate(zip(pats, coh, full))]]
        rows: list[dict] = []
        S8.run_patterns(daq, slm, self.layout, [tuple(self.pairs)], x, w,
                        self.models, self.phases, rows=rows, acq=self.acq,
                        encode=_encode, log=lambda _: None, extra=extra)
        return rows

    def _report(self, rows, tmp):
        path = V.write_verify_csv(
            Path(tmp) / "system_accuracy_test.csv", self.pairs, rows,
            meta={"step7_json": STEP7, "y_max_v": self.scale.fs, "status": "complete"},
            extra=["kind", "label", "seq", "target_alpha", "alpha",
                   "coherent_v", "sb_pred_v"])
        return R.load_reads(path, STEP7)

    def test_one_row_per_pattern(self):
        rows = self._collect()
        self.assertEqual(len(rows), len(self.patterns))

    def test_a_perfect_bench_reads_zero_residual(self):
        """Coherent target, single-beam subtraction, CSV columns -- all of it."""
        rows = self._collect()
        with tempfile.TemporaryDirectory() as tmp:
            reads, _ = self._report(rows, tmp)
        self.assertEqual(len(reads), len(self.patterns))
        self.assertLess(np.max(np.abs(reads.resid)), 1e-9)

    def test_the_zero_above_is_not_trivially_zero(self):
        """Leaving the single beam in must NOT also read zero, or the test is empty."""
        rows = self._collect()
        for r in rows:
            r["sb_pred_v"] = "0"
        with tempfile.TemporaryDirectory() as tmp:
            reads, _ = self._report(rows, tmp)
        resid = np.max(np.abs(reads.resid))
        self.assertGreater(resid, 1e-4)      # the background is ~1 mV
        self.assertLess(resid, 2e-3)

    def test_the_families_survive_the_round_trip(self):
        rows = self._collect()
        with tempfile.TemporaryDirectory() as tmp:
            reads, _ = self._report(rows, tmp)
        self.assertEqual(list(reads.kind), [p.kind for p in self.patterns])
        self.assertEqual(list(reads.label), [p.label for p in self.patterns])

    def test_each_read_takes_the_range_its_prediction_calls_for(self):
        """The range follows the level: dim reads stay sensitive, bright ones do
        not clip.  Nothing clips on a perfect bench, so no read escalates and
        every one keeps the range ``range_for`` picked."""
        rows = self._collect()
        for row in rows:
            want, _ = S8.range_for(row["pred_v"])
            self.assertEqual(row["range_v"], want)
            self.assertLessEqual(abs(row["pred_v"]),
                                 S8.RANGE_FILL * row["range_v"])
        spanned = {row["range_v"] for row in rows}
        self.assertGreater(len(spanned), 1, "the ladder should span several ranges")

    def test_repeats_of_one_drive_are_one_group(self):
        rows = self._collect(noise=1e-4)
        with tempfile.TemporaryDirectory() as tmp:
            reads, _ = self._report(rows, tmp)
        for alpha, ys in R.repeat_spread(reads):
            self.assertGreater(ys.size, 1)
            self.assertGreater(ys.std(ddof=1), 0.0)   # noise made them differ

    def test_a_perfect_bench_puts_full_scale_where_the_model_says(self):
        """C's fit must recover shift 0 and scale 1 when nothing is wrong."""
        rows = self._collect()
        with tempfile.TemporaryDirectory() as tmp:
            reads, _ = self._report(rows, tmp)
        peak = R.psi_peak(reads)
        for tag in ("start", "end"):
            self.assertAlmostEqual(peak[tag]["peak_deg"], 0.0, places=6)
            self.assertAlmostEqual(peak[tag]["scale"], 1.0, places=6)
            self.assertAlmostEqual(peak[tag]["fs_meas"], self.scale.fs, places=9)

    def test_a_phase_error_shows_up_as_a_shifted_peak(self):
        """The zero above is only worth something if a real error moves it."""
        rows = self._collect()
        with tempfile.TemporaryDirectory() as tmp:
            reads, _ = self._report(rows, tmp)
        c = reads.kind == "C"
        off = np.array([float(l[1:l.index("deg")]) for l in reads.label[c]])
        shifted = reads[c]
        # re-read the scan as if psi* were 10 deg away from where the model put it
        order = np.argsort(off)
        shifted.y[order] = np.interp(off[order] - 10.0, off[order],
                                     shifted.pred[order])
        peak = R.psi_peak(shifted)
        self.assertAlmostEqual(peak["start"]["peak_deg"], 10.0, places=1)

    def test_the_report_writes_a_png(self):
        rows = self._collect(noise=1e-4)
        with tempfile.TemporaryDirectory() as tmp:
            path = V.write_verify_csv(
                Path(tmp) / "system_accuracy_test.csv", self.pairs, rows,
                meta={"step7_json": STEP7, "status": "complete"},
                extra=["kind", "label", "seq", "target_alpha", "alpha",
                       "coherent_v", "sb_pred_v"])
            R.report_csv(path, STEP7, out_dir=tmp)
            self.assertTrue((Path(tmp) / "system_accuracy_test_report.png").exists())


class CsvCompatibilityTests(unittest.TestCase):
    """The extra columns must not break anything that already reads these files."""

    def test_a_step_8_csv_still_loads_with_no_extra_columns(self):
        models, phases, pairs = model()
        x, w = V.random_patterns(1, 2, 3, 0.1, 0.9, seed=1)
        rows = [{"block": 0, "driven": (1, 2), "x": x[0, j], "w": w[0, j],
                 "dark_v": 0.0, "mean_v": 0.01, "std_v": 1e-4,
                 "range_v": 0.1, "pred_v": 0.01} for j in range(3)]
        with tempfile.TemporaryDirectory() as tmp:
            path = V.write_verify_csv(Path(tmp) / "old.csv", [1, 2], rows, meta={})
            _, blocks, _ = V.load_verify_csv(path)
        self.assertEqual(blocks[0].extra, {})

    def test_extra_columns_round_trip_by_row(self):
        rows = [{"block": 0, "driven": (1, 2), "x": np.array([0.5, 0.5]),
                 "w": np.array([0.5, 0.5]), "dark_v": 0.0, "mean_v": 0.01,
                 "std_v": 1e-4, "range_v": 0.1, "pred_v": 0.01,
                 "kind": k, "seq": i, "label": f"L{i}"}
                for i, k in enumerate("ABC")]
        with tempfile.TemporaryDirectory() as tmp:
            path = V.write_verify_csv(Path(tmp) / "new.csv", [1, 2], rows, meta={},
                                      extra=["kind", "seq", "label"])
            _, blocks, _ = V.load_verify_csv(path)
        extra = blocks[0].extra
        self.assertEqual(list(extra["kind"]), ["A", "B", "C"])
        self.assertEqual(list(extra["label"]), ["L0", "L1", "L2"])
        np.testing.assert_array_equal(extra["seq"], [0.0, 1.0, 2.0])

    def test_a_system_accuracy_csv_still_compares_as_a_step_8_block(self):
        """The leading columns are unchanged, so the old reader keeps working."""
        models, phases, pairs = model()
        pats = levels.level_patterns(models, phases, pairs, 4, seed=3)
        rows = [{"block": 0, "driven": tuple(pairs), "x": p.x, "w": p.w,
                 "dark_v": 0.0,
                 "mean_v": float(V.predict(models, phases, pairs, p.x, p.w)),
                 "std_v": 1e-4, "range_v": 0.5,
                 "pred_v": float(V.predict(models, phases, pairs, p.x, p.w)),
                 "kind": "A", "label": p.label}
                for p in pats]
        with tempfile.TemporaryDirectory() as tmp:
            path = V.write_verify_csv(Path(tmp) / "sa.csv", pairs, rows, meta={},
                                      extra=["kind", "label"])
            _, blocks, _ = V.load_verify_csv(path)
        self.assertEqual(len(blocks), 1)
        err = V.evaluate(blocks[0], models, phases)
        self.assertLess(err.nrmse_pct, 1e-6)


class LooBuildTests(unittest.TestCase):
    """The leave-one-out plan: padding, matched strata, and the interleave."""

    @classmethod
    def setUpClass(cls):
        cls.models, cls.phases, cls.pool = model()
        cls.subs, cls.scales, cls.flat = levels.loo_patterns(
            cls.models, cls.phases, cls.pool, 4, seed=3)

    def test_the_pool_and_every_leave_one_out(self):
        self.assertEqual(len(self.subs), len(self.pool) + 1)
        self.assertEqual(self.subs[0], tuple(self.pool))
        self.assertEqual({levels.subset_tag(s, self.pool) for s in self.subs},
                         {"all"} | {f"drop{k}" for k in self.pool})

    def test_padding_does_not_change_the_prediction(self):
        """The whole single-block design rests on this."""
        for sub in self.subs:
            solved = levels.level_patterns(self.models, self.phases, sub, 3, seed=3)
            for p in solved:
                on_sub = V.predict(self.models, self.phases, sub, p.x, p.w)
                padded = levels.pad_to_pool(p, sub, self.pool)
                on_pool = V.predict(self.models, self.phases, self.pool,
                                    padded.x, padded.w)
                self.assertAlmostEqual(float(on_sub), float(on_pool), places=12)

    def test_an_excluded_pair_is_driven_to_nothing(self):
        idx = {k: i for i, k in enumerate(self.pool)}
        for p in self.flat:
            tag = p.label.split("/")[1]
            for k in self.pool:
                if tag == f"drop{k}":
                    self.assertEqual(p.x[idx[k]], 0.0)
                    self.assertEqual(p.w[idx[k]], 0.0)

    def test_every_subset_gets_the_same_stratum_grid(self):
        """Matched alphas, so a difference between subsets is not sampling."""
        by_tag: dict[str, list[float]] = {}
        for p in self.flat:
            by_tag.setdefault(p.label.split("/")[1], []).append(p.target_alpha)
        grids = list(by_tag.values())
        self.assertEqual(len(grids), len(self.pool) + 1)
        for g in grids[1:]:
            np.testing.assert_allclose(g, grids[0])

    def test_a_subsets_full_scale_is_its_own(self):
        full = self.scales["all"].fs
        for sub in self.subs[1:]:
            tag = levels.subset_tag(sub, self.pool)
            self.assertLess(self.scales[tag].fs, full)
            self.assertAlmostEqual(
                self.scales[tag].fs,
                V.full_scale({k: self.models[k] for k in sub},
                             {k: self.phases[k] for k in sub}, sub).fs)

    def test_the_subsets_are_interleaved_not_run_back_to_back(self):
        """Eleven ladders in a row would put drift straight on the subset axis."""
        rng = np.random.default_rng(3 + 3)
        order = [self.flat[i].label.split("/")[1]
                 for i in rng.permutation(len(self.flat))]
        runs = 1 + sum(1 for a, b in zip(order, order[1:]) if a != b)
        self.assertGreater(runs, 0.7 * len(order))


class LooCollectTests(unittest.TestCase):
    """The sweep through the CSV and the report, against a fake bench."""

    LADDER = 5

    @classmethod
    def setUpClass(cls):
        cls.models, cls.phases, cls.pool = model()
        cls.subs, cls.scales, flat = levels.loo_patterns(
            cls.models, cls.phases, cls.pool, cls.LADDER, seed=11)
        rng = np.random.default_rng(11 + 3)
        cls.patterns = [flat[i] for i in rng.permutation(len(flat))]

    def _report(self, tmp, truth_phases=None, noise=0.0):
        slm = _FakeSLM()
        daq = _FakeDAQ(slm, self.models, truth_phases or self.phases, noise=noise)
        pats = self.patterns
        x = np.stack([p.x for p in pats])[None]
        w = np.stack([p.w for p in pats])[None]
        coh = [float(V.predict(self.models, self.phases, self.pool, p.x, p.w,
                               coherent=True)) for p in pats]
        full = [float(V.predict(self.models, self.phases, self.pool, p.x, p.w))
                for p in pats]
        extra = [[{"kind": p.kind, "label": p.label, "seq": i,
                   "subset": p.label.split("/")[1],
                   "target_alpha": f"{p.target_alpha:.6f}", "alpha": f"{p.alpha:.6f}",
                   "coherent_v": f"{c:.9g}", "sb_pred_v": f"{q - c:.9g}"}
                  for i, (p, c, q) in enumerate(zip(pats, coh, full))]]
        rows: list[dict] = []
        S8.run_patterns(daq, slm, SimpleNamespace(n_channels=len(self.pool)),
                        [tuple(self.pool)], x, w, self.models, self.phases,
                        rows=rows, acq=replace(S8.ACQ, settle_s=0.0, t_both_s=0.0,
                                               t_single_s=0.0),
                        encode=_encode, log=lambda _: None, extra=extra)
        path = V.write_verify_csv(
            Path(tmp) / "system_accuracy_loo_test.csv", self.pool, rows,
            meta={"step7_json": STEP7, "mode": "loo", "status": "complete"},
            extra=["kind", "label", "seq", "target_alpha", "alpha",
                   "coherent_v", "sb_pred_v", "subset"])
        return R.loo_table(path, STEP7)

    def test_one_row_per_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows, meta = self._report(tmp)
        self.assertEqual([r["subset"] for r in rows],
                         ["all"] + [f"drop{k}" for k in self.pool])
        self.assertTrue(all(r["N"] == self.LADDER for r in rows))
        self.assertEqual(meta["pool"], list(self.pool))

    def test_a_perfect_bench_reads_zero_everywhere(self):
        """Catches a padding, subset or full-scale mix-up anywhere in the chain."""
        with tempfile.TemporaryDirectory() as tmp:
            rows, _ = self._report(tmp)
        for r in rows:
            self.assertLess(abs(r["rms_lsb"]), 1e-6, r["subset"])

    def test_each_subset_is_scaled_by_its_own_full_scale(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows, _ = self._report(tmp)
        for r in rows:
            tag = r["subset"]
            self.assertAlmostEqual(r["y_max_v"], self.scales[tag].fs)
            self.assertAlmostEqual(r["level_step_v"], self.scales[tag].fs / 15.0)

    def test_a_miscalibrated_pair_is_the_subset_that_drops_it(self):
        """The load-bearing one: does the sweep find the pair it is looking for?

        The bench runs with pair 1's phase 25 deg off what the model believes.
        Every subset that still drives pair 1 inherits that error; the one that
        leaves it out does not, so ``drop1`` has to come out best.
        """
        truth = dict(self.phases)
        truth[self.pool[0]] += np.radians(25.0)
        with tempfile.TemporaryDirectory() as tmp:
            rows, _ = self._report(tmp, truth_phases=truth)
        by = {r["subset"]: r["rms_lsb"] for r in rows}
        best = min((r for r in rows if r["subset"] != "all"),
                   key=lambda r: r["rms_lsb"])["subset"]
        self.assertEqual(best, f"drop{self.pool[0]}")
        self.assertLess(by[f"drop{self.pool[0]}"], 1e-6)
        self.assertGreater(by["all"], 1e-3)

    def test_the_report_writes_a_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows, meta = self._report(tmp)
            R.print_loo(rows, meta)
            png = Path(tmp) / "loo.png"
            R.plot_loo(rows, png)
            self.assertTrue(png.exists() and png.stat().st_size > 0)

    def test_a_run_without_the_subset_column_is_refused(self):
        """A plain system-accuracy CSV must not be read as a sweep."""
        with tempfile.TemporaryDirectory() as tmp:
            path = V.write_verify_csv(
                Path(tmp) / "plain.csv", self.pool,
                [{"block": 0, "driven": tuple(self.pool),
                  "x": [0.5] * len(self.pool), "w": [0.5] * len(self.pool),
                  "dark_v": 0.0, "mean_v": 0.0, "std_v": 0.0, "range_v": 0.1,
                  "pred_v": 0.0, "kind": "A"}],
                meta={"step7_json": STEP7}, extra=["kind"])
            with self.assertRaises(ValueError):
                R.loo_table(path, STEP7)


class _FakeK2400:
    """Stands in for the GPIB stack: hands back scripted ohms, counts the reads."""

    def __init__(self, *values):
        self.values = list(values)
        self.calls = 0

    def __call__(self, resource, **kwargs):
        self.calls += 1
        value = self.values[min(self.calls - 1, len(self.values) - 1)]
        if isinstance(value, Exception):
            raise value
        return value


class PmtTemperatureTests(unittest.TestCase):
    """The --pmt row hook: caching, alignment and failure isolation.

    The 2400 and its VISA stack are replaced wholesale, so this runs anywhere;
    what is under test is the caching and the never-raise contract, which is
    what an unattended overnight run depends on.
    """

    def _pmt(self, fake, **kwargs):
        """A PmtTemperature whose instrument layer is ``fake``."""
        pmt_mod = SimpleNamespace(
            K2400_RESOURCE="GPIB0::21::INSTR", K2400_FOUR_WIRE=False,
            K2400_NPLC=10.0, K2400_CURRENT=None, K2400_COMPLIANCE=21.0,
            K2400_TIMEOUT_S=10.0)
        stubs = {"daq_pmt_monitor": pmt_mod,
                 "monitor_lib": SimpleNamespace(read_ohm_2400=fake)}
        saved = {k: sys.modules.get(k) for k in stubs}
        sys.modules.update(stubs)
        try:
            return SA.PmtTemperature(**kwargs)
        finally:
            for k, v in saved.items():
                if v is None:
                    del sys.modules[k]
                else:
                    sys.modules[k] = v

    def test_10_kohm_is_25_c(self):
        pmt = self._pmt(_FakeK2400(10_000.0))
        cells = pmt.read()
        self.assertAlmostEqual(float(cells["pmt_c"]), 25.0, places=3)
        self.assertEqual(float(cells["pmt_ohm_Ohm"]), 10_000.0)

    def test_the_reading_is_cached_between_samples(self):
        """175 reads must not cost 175 GPIB round trips -- that is the whole point."""
        fake = _FakeK2400(10_000.0, 12_000.0)
        pmt = self._pmt(fake, period_s=3600.0)
        first = pmt.sample()
        for _ in range(20):
            self.assertEqual(pmt.sample(), first)
        self.assertEqual(fake.calls, 1)

        pmt._t_last -= 3601.0                       # the period has now passed
        second = pmt.sample()
        self.assertEqual(fake.calls, 2)
        self.assertNotEqual(second["pmt_ohm_Ohm"], first["pmt_ohm_Ohm"])
        self.assertGreater(float(first["pmt_c"]), float(second["pmt_c"]))

    def test_a_dead_2400_blanks_the_cells_and_does_not_raise(self):
        pmt = self._pmt(_FakeK2400(RuntimeError("GPIB timeout")))
        row = {"pred_v": 0.0}
        pmt(row)                                    # must not raise
        self.assertEqual(row["pmt_ohm_Ohm"], "")
        self.assertEqual(row["pmt_c"], "")

    def test_over_range_and_off_curve_readings_are_blank_not_wrong(self):
        for value in (float("inf"), 1e9):           # open circuit, off the TH10K table
            with self.subTest(value=value):
                pmt = self._pmt(_FakeK2400(value))
                self.assertEqual(pmt.read(), {"pmt_ohm_Ohm": "", "pmt_c": ""})

    def test_the_hook_stamps_every_read_and_survives_the_csv(self):
        models, phases, pairs = model()
        _, patterns, _ = SA.build(models, phases, pairs, seed=3, ladder=4)
        patterns = patterns[:6]
        slm, layout = _FakeSLM(), SimpleNamespace(n_channels=len(pairs))
        daq = _FakeDAQ(slm, models, phases)
        x = np.stack([p.x for p in patterns])[None]
        w = np.stack([p.w for p in patterns])[None]
        pmt = self._pmt(_FakeK2400(10_000.0), period_s=3600.0)

        rows: list[dict] = []
        S8.run_patterns(daq, slm, layout, [tuple(pairs)], x, w, models, phases,
                        rows=rows, acq=replace(S8.ACQ, settle_s=0.0, t_both_s=0.0,
                                               t_single_s=0.0),
                        encode=_encode, log=lambda _: None, row_hook=pmt)

        self.assertEqual(len(rows), len(patterns))
        self.assertTrue(all(r["pmt_c"] for r in rows))
        with tempfile.TemporaryDirectory() as tmp:
            path = V.write_verify_csv(Path(tmp) / "pmt.csv", pairs, rows,
                                      meta={"step7_json": STEP7},
                                      extra=list(SA.PmtTemperature.COLUMNS))
            header, first = Path(path).read_text().splitlines()[:2]
        self.assertEqual(header.split(",")[-2:], list(SA.PmtTemperature.COLUMNS))
        self.assertAlmostEqual(float(first.split(",")[-1]), 25.0, places=3)

    def test_run_patterns_without_a_hook_is_unchanged(self):
        models, phases, pairs = model()
        _, patterns, _ = SA.build(models, phases, pairs, seed=3, ladder=4)
        patterns = patterns[:3]
        slm, layout = _FakeSLM(), SimpleNamespace(n_channels=len(pairs))
        x = np.stack([p.x for p in patterns])[None]
        w = np.stack([p.w for p in patterns])[None]
        rows: list[dict] = []
        S8.run_patterns(_FakeDAQ(slm, models, phases), slm, layout, [tuple(pairs)],
                        x, w, models, phases, rows=rows,
                        acq=replace(S8.ACQ, settle_s=0.0, t_both_s=0.0, t_single_s=0.0),
                        encode=_encode, log=lambda _: None)
        self.assertTrue(all("pmt_c" not in r for r in rows))


class PmtInStepsSixAndSevenTests(unittest.TestCase):
    """Steps 6 and 7 carry the thermistor in two trailing columns.

    The contract that matters is compatibility: the columns and the meta lines
    must not change what either step's fit produces, and a v1 re-fit of a v2
    step-6 CSV -- which that CSV's docstring promises -- must still work with
    both present.
    """

    S6_CSV = ROOT / "src/calib_data/calib_0917_1617/calib_step6v2_meas_0917_1811.csv"
    S7_CSV = ROOT / "src/calib_data/calib_0917_1617/calib_step7_meas_0917_1831.csv"
    S6_JSON = ROOT / "src/calib_data/calib_0917_1617/calib_step6v2_result_0917_1811.json"
    META = {"pmt_resource": "GPIB0::21::INSTR", "pmt_period_s": "60",
            "pmt_c_start": "27.2379", "pmt_c_end": "27.2146"}

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    # ---- step 6 ----------------------------------------------------------
    def test_step6_rows_round_trip_with_and_without_the_thermistor(self):
        rows = {2: [(0, 0.5, 0.5, 0.01, 1e-4, "9070.1", "27.2379")],
                3: [(1, 0.9, 0.9, 0.03, 2e-4)]}          # an old 5-tuple row
        path = P6.write_meas_csv(rows, self.tmp / "s6.csv", meta=self.META)
        back = P6.load_meas_csv(path)
        self.assertEqual(back[2][0], (0, 0.5, 0.5, 0.01, 1e-4, "9070.1", "27.2379"))
        self.assertEqual(back[3][0][5:], ("", ""))       # blank, not missing
        self.assertEqual(P6.meas_csv_meta(path)["pmt_c_end"], "27.2146")

    def test_step6_average_levels_takes_5_and_7_tuples_alike(self):
        seven = [(0, 0.5, 0.5, 0.01, 1e-4, "9070", "27.2"),
                 (1, 0.5, 0.5, 0.02, 1e-4, "9071", "27.1")]
        five = [r[:5] for r in seven]
        a, b = P6.average_levels(seven), P6.average_levels(five)
        self.assertEqual(len(a), 1)
        np.testing.assert_allclose(a[0].means, b[0].means)

    def test_a_v1_refit_survives_the_new_columns_and_the_meta_lines(self):
        """The v2 CSV promises v1 can still re-fit it; that must stay true."""
        before = {c.index: c.fit.eta
                  for c in load_tpa_pair_csv(self.S6_CSV).channels if c.fit}
        rows = P6.load_meas_csv(self.S6_CSV)
        stamped = {k: [r[:5] + ("9070.1", "27.2379") for r in v]
                   for k, v in rows.items()}
        path = P6.write_meas_csv(stamped, self.tmp / "s6_pmt.csv", meta=self.META)
        after = {c.index: c.fit.eta
                 for c in load_tpa_pair_csv(path).channels if c.fit}
        self.assertEqual(before.keys(), after.keys())
        self.assertEqual(len(before), 10)
        for k in before:                    # bit-identical, not merely close
            self.assertEqual(before[k], after[k], f"pair {k} moved")

    # ---- step 7 ----------------------------------------------------------
    def _models(self):
        return PH.load_pair_models(self.S6_JSON)

    def test_step7_columns_round_trip_and_do_not_move_the_fit(self):
        models = self._models()
        r0 = PH.load_phase_csv(self.S7_CSV, models[4], models[2], only_tgt=4,
                               single_beam_bg=True, comb_only=True)
        self.assertIsNone(r0.pmt_c, "a CSV with no thermistor must load as None")

        n = r0.trial.size
        r0.pmt_ohm = np.array(["9070.1"] * n, dtype=object)
        r0.pmt_c = np.array([f"{27.2 - i * 0.001:.4f}" for i in range(n)],
                            dtype=object)
        path = PH.write_meas_csv([r0], self.tmp / "s7.csv", meta=self.META)
        r1 = PH.load_phase_csv(path, models[4], models[2], only_tgt=4,
                               single_beam_bg=True, comb_only=True)
        self.assertEqual(r1.fit.dphi_comb, r0.fit.dphi_comb)
        self.assertEqual(list(r1.pmt_c), list(r0.pmt_c))
        self.assertEqual(list(r1.pmt_ohm), list(r0.pmt_ohm))

    def test_a_step7_sweep_without_the_thermistor_stays_none(self):
        """Not a column of blanks: None is what says the sweep never logged it."""
        models = self._models()
        r0 = PH.load_phase_csv(self.S7_CSV, models[4], models[2], only_tgt=4,
                               comb_only=True)
        path = PH.write_meas_csv([r0], self.tmp / "s7_bare.csv")
        r1 = PH.load_phase_csv(path, models[4], models[2], only_tgt=4,
                               comb_only=True)
        self.assertIsNone(r1.pmt_c)
        self.assertIsNone(r1.pmt_ohm)

    # ---- the on-by-default switch ---------------------------------------
    def test_open_pmt_returns_none_instead_of_raising(self):
        """Steps 6 and 7 log by default, so a missing 2400 must cost two columns.

        Not the calibration: PmtTemperature's constructor is the one place that
        still raises, and open_pmt is what stands between it and an hour of
        bench time.
        """
        said: list[str] = []
        with mock.patch("calibration_module.measure.pmt_temp.PmtTemperature",
                        side_effect=RuntimeError("no VISA backend")):
            self.assertIsNone(open_pmt(log=said.append))
        self.assertTrue(any("no VISA backend" in m for m in said), said)

    def test_open_pmt_off_never_touches_the_instrument(self):
        with mock.patch("calibration_module.measure.pmt_temp.PmtTemperature",
                        side_effect=AssertionError("must not be constructed")):
            self.assertIsNone(open_pmt(enabled=False))


if __name__ == "__main__":
    unittest.main()
