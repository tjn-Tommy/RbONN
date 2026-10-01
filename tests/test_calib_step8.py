"""The system-accuracy run end to end, against a fake bench.

The perfect-bench test is the load-bearing one: a DAQ that hands back exactly
what the model predicts must produce a report whose residuals are zero.  Anything
that quietly corrupts the chain -- the coherent/total mix-up, a single-beam
subtraction applied twice, a forced range that changes the reading, the extra CSV
columns coming back misaligned -- shows up there as a non-zero residual.
"""
from __future__ import annotations

import json
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
from calibration_module.measure import patterns as S8  # noqa: E402
from calibration_module.steps import calib_step8 as SA  # noqa: E402
from calibration_module.fit import pair_v2 as P6, phase as PH  # noqa: E402
from calibration_module.fit.pair import load_tpa_pair_csv  # noqa: E402
from calibration_module.measure.pmt_temp import open_pmt  # noqa: E402
from daq_module import DAQMonitorSettings  # noqa: E402

RUN = ROOT / "src" / "calib_data" / "run_0911_10pairs_w20g30"
STEP7 = RUN / "calib.json"


def model():
    if not STEP7.exists():
        raise unittest.SkipTest(f"{STEP7} not in the tree")
    models, phases, _ = V.load_forward_model(STEP7)
    return models, phases, sorted(models)


def hot_model(y_max_v=1.534):
    """The same model scaled to a given Y_max -- a brighter bench than the file.

    The file on disk tops out near 317 mV, which is UNDER
    :data:`LADDER_CEILING_V`, so it exercises the uncapped branch and cannot say
    anything about the capped one.  ``Y_max = |sum eta|^2``, so scaling every
    ``eta`` by ``sqrt(k)`` scales ``Y_max`` by ``k``: this is the present bench,
    where the amplifier and not the atoms decides where the ladder stops.
    """
    models, phases, pairs = model()
    k = (float(y_max_v) / V.full_scale(models, phases, pairs).fs) ** 0.5
    hot = {i: replace(m, eta=m.eta * k) for i, m in models.items()}
    return hot, phases, pairs


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
        return {k: sum(1 for p in self.patterns if p.kind == k) for k in "ABCF"}

    def test_the_sweep_is_the_two_bands_and_the_bracket_and_nothing_else(self):
        c = self.counts()
        self.assertEqual(c["A"], SA.LADDER + SA.LADDER_LOW)
        self.assertEqual(c["F"], 2)
        self.assertEqual(c["B"], 0)
        self.assertEqual(c["C"], 0)

    def test_the_ceiling_brackets_the_run(self):
        self.assertEqual(self.patterns[0].kind, "F")
        self.assertEqual(self.patterns[-1].kind, "F")
        self.assertTrue(self.patterns[0].label.endswith("@start"))
        self.assertTrue(self.patterns[-1].label.endswith("@end"))
        self.assertFalse(any(p.kind == "F" for p in self.patterns[1:-1]))

    def test_the_bracket_is_the_top_of_the_ladder(self):
        """The fit lives at the top, so the bracket has to BE the top."""
        cap = SA.ladder_levels(self.scale.fs)["cap"]
        alpha = max(p.alpha for p in self.patterns if p.kind == "A")
        for p in (self.patterns[0], self.patterns[-1]):
            self.assertAlmostEqual(p.alpha, cap, places=6)
            self.assertGreaterEqual(p.alpha + 1e-9, alpha)
        # and both ends drive the identical frame, or the drift bracket is not one
        np.testing.assert_array_equal(self.patterns[0].x, self.patterns[-1].x)
        np.testing.assert_array_equal(self.patterns[0].w, self.patterns[-1].w)

    def test_the_ladder_is_not_in_level_order(self):
        """Ascending order would alias drift onto the level axis."""
        alpha = [p.alpha for p in self.patterns if p.kind == "A"]
        self.assertFalse(alpha == sorted(alpha))
        # Scale-free, because the band's width is a bench number now: a sorted
        # ladder steps by span/(n-1) between neighbours, a shuffled one by
        # O(span).  Anything past a few times the sorted step is not an order.
        span = max(alpha) - min(alpha)
        self.assertGreater(np.mean(np.abs(np.diff(alpha))),
                           5 * span / (len(alpha) - 1))

    def test_the_bright_band_holds_the_leverage(self):
        """LADDER rungs above the floor, LADDER_LOW below, and no strays."""
        lv = SA.ladder_levels(self.scale.fs)
        alpha = np.array([p.alpha for p in self.patterns if p.kind == "A"])
        self.assertEqual(int((alpha >= lv["floor"] - 1e-9).sum()), SA.LADDER)
        self.assertEqual(int((alpha < lv["floor"] - 1e-9).sum()), SA.LADDER_LOW)

    def test_the_bright_band_is_covered_evenly(self):
        """Stratified, so 30 rungs cannot leave a tenth of the band empty."""
        lv = SA.ladder_levels(self.scale.fs)
        alpha = np.array([p.alpha for p in self.patterns if p.kind == "A"])
        bright = alpha[alpha >= lv["floor"] - 1e-9]
        bins = np.clip(((bright - lv["floor"]) / (lv["cap"] - lv["floor"])
                        * 10).astype(int), 0, 9)
        self.assertEqual(len(set(bins.tolist())), 10)

    def test_the_low_rungs_reach_down_to_the_check_floor(self):
        """They are not leverage; they are the only reads near the ONN median."""
        lv = SA.ladder_levels(self.scale.fs)
        alpha = np.array([p.alpha for p in self.patterns if p.kind == "A"])
        low = alpha[alpha < lv["floor"] - 1e-9]
        self.assertGreaterEqual(low.min(), lv["low"] - 1e-9)
        # inside the first stratum of the low band, so genuinely near the floor
        width = (lv["floor"] - lv["low"]) / SA.LADDER_LOW
        self.assertLess(low.min(), lv["low"] + width)

    def test_a_bright_bench_stops_at_the_ceiling_in_VOLTS(self):
        """The whole point of the cap: 450 mV, whatever Y_max happens to be.

        The file on disk is dimmer than the ceiling, so this uses a scaled model
        -- the present bench, where Y_max is 1534 mV against a 1 V amplifier.
        """
        hot, phases, pairs = hot_model(1.534)
        scale, patterns, preds = SA.build(hot, phases, pairs, seed=5)
        self.assertAlmostEqual(scale.fs, 1.534, places=6)
        lv = SA.ladder_levels(scale.fs)
        self.assertTrue(lv["capped"])
        coh = [float(V.predict(hot, phases, pairs, p.x, p.w, coherent=True))
               for p in patterns]
        self.assertAlmostEqual(max(coh), SA.LADDER_CEILING_V, places=4)
        # and nothing anywhere near the amplifier: that is what buys the margin
        self.assertLess(max(preds), 0.55 * SA.AMP_LIMIT_V)

    def test_a_dim_bench_is_not_capped_at_all(self):
        """Under the ceiling the atoms own the top, and alpha reaches 1."""
        lv = SA.ladder_levels(0.300)
        self.assertFalse(lv["capped"])
        self.assertEqual(lv["cap"], 1.0)

    def test_the_levels_are_the_constants_divided_by_y_max(self):
        lv = SA.ladder_levels(1.534)
        self.assertAlmostEqual(lv["cap"] * 1.534, SA.LADDER_CEILING_V, places=9)
        self.assertAlmostEqual(lv["floor"] * 1.534, SA.LADDER_FLOOR_V, places=9)
        self.assertAlmostEqual(lv["low"] * 1.534, SA.LADDER_LOW_V, places=9)

    def test_the_three_voltages_are_arguments_and_still_volts(self):
        """The GUI and the CLI can move the band without editing the module.

        In VOLTS at the boundary too, for the same reason they are volts in the
        module: the same three numbers have to mean the same three reads after
        the PMT gain moves.
        """
        for fs in (1.534, 0.900):
            with self.subTest(y_max_v=fs):
                lv = SA.ladder_levels(fs, ceiling_v=0.700, floor_v=0.350,
                                      low_v=0.040)
                self.assertAlmostEqual(lv["cap"] * fs, 0.700, places=9)
                self.assertAlmostEqual(lv["floor"] * fs, 0.350, places=9)
                self.assertAlmostEqual(lv["low"] * fs, 0.040, places=9)

    def test_a_moved_ceiling_moves_the_brightest_read_of_the_sweep(self):
        """Not just the arithmetic: the solved drives land on the new volts.

        This is the measurement the probe exists to enable -- find where the
        amplifier really compresses, then put it in this box.
        """
        hot, phases, pairs = hot_model(1.534)
        scale, patterns, _preds = SA.build(hot, phases, pairs, seed=7,
                                           ceiling_v=0.700, floor_v=0.350,
                                           low_v=0.040)
        coh = np.array([float(V.predict(hot, phases, pairs, p.x, p.w,
                                        coherent=True)) for p in patterns])
        self.assertAlmostEqual(coh.max(), 0.700, places=4)
        bright = coh[coh >= 0.350 - 1e-9]
        self.assertEqual(len(bright), SA.LADDER + 2)      # the bracket is up here
        self.assertAlmostEqual(coh.min(), 0.040, delta=0.055)

    def test_a_raised_band_floor_crowds_the_rungs_without_moving_the_top(self):
        hot, phases, pairs = hot_model(1.534)
        _scale, patterns, _preds = SA.build(hot, phases, pairs, seed=7,
                                            floor_v=0.350)
        coh = np.array([float(V.predict(hot, phases, pairs, p.x, p.w,
                                        coherent=True)) for p in patterns])
        self.assertAlmostEqual(coh.max(), SA.LADDER_CEILING_V, places=4)
        bright = np.sort(coh)[-(SA.LADDER + 2):]
        self.assertGreaterEqual(bright.min(), 0.350 - 1e-6)
        # and the checks still reach down to their own floor, unmoved
        self.assertLess(coh.min(), 0.06)

    def test_the_bands_stay_ordered_for_an_absurdly_dim_model(self):
        """Clamped, so a bad step 7 cannot produce an empty or inverted band."""
        for fs in (1e-4, 1e-3, 0.01, 0.05, 0.3, 1.0, 5.0):
            with self.subTest(y_max_v=fs):
                lv = SA.ladder_levels(fs)
                self.assertLess(lv["low"], lv["floor"])
                self.assertLess(lv["floor"], lv["cap"])
                self.assertLessEqual(lv["cap"], 1.0)
                self.assertGreater(lv["low"], 0.0)

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
        # The fake bench reads instantly, so the whole plan runs here.
        cls.scale, cls.patterns, cls.preds = SA.build(
            cls.models, cls.phases, cls.pairs, seed=7)
        # Step 8 drives A and F only, but levels_report.py still has to read the
        # A+B+C runs already on disk, so those paths are exercised against a
        # plan built here rather than against one the step can still produce.
        scan = levels.psi_scan_patterns(cls.models, cls.phases, cls.pairs)
        cls.legacy = (
            [replace(q, label=q.label + "@start") for q in scan]
            + levels.level_patterns(cls.models, cls.phases, cls.pairs, 12, seed=7)
            + levels.repeat_patterns(cls.models, cls.phases, cls.pairs,
                                     (0.25, 0.75), 4, seed=8)
            + [replace(q, label=q.label + "@end") for q in scan])

    def setUp(self):
        self.layout = SimpleNamespace(n_channels=len(self.pairs))
        self.acq = replace(S8.DEFAULT_ACQ, settle_s=0.0, t_meas_s=0.0)

    def _collect(self, noise=0.0, pats=None):
        slm = _FakeSLM()
        daq = _FakeDAQ(slm, self.models, self.phases, noise=noise)
        pats = self.patterns if pats is None else pats
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

    def _report(self, rows, tmp, name="calib_step8_test"):
        path = V.write_verify_csv(
            Path(tmp) / f"{name}.csv", self.pairs, rows,
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

    def test_every_read_takes_the_one_fixed_range(self):
        """One range for the whole ladder, whatever each read is predicted at.

        This is the opposite of what the step used to do -- it picked each
        read's range from its prediction, so a 90-rung ladder spanned three or
        four of them and the dark sat on a fifth.  A ladder is read rung against
        rung, and the board's offset and gain error are per-range, so that put a
        range-shaped step into the exact axis ``fit_y_sat`` reads.
        """
        rows = self._collect()
        self.assertEqual({row["range_v"] for row in rows}, {self.acq.range_v})
        # and the range the old policy would have chosen really does vary, so
        # this is a fact about the run rather than about a flat ladder
        wanted = {S8.range_for(row["pred_v"])[0] for row in rows}
        self.assertGreater(len(wanted), 1)

    def test_repeats_of_one_drive_are_one_group(self):
        rows = self._collect(noise=1e-4, pats=self.legacy)
        with tempfile.TemporaryDirectory() as tmp:
            reads, _ = self._report(rows, tmp)
        for alpha, ys in R.repeat_spread(reads):
            self.assertGreater(ys.size, 1)
            self.assertGreater(ys.std(ddof=1), 0.0)   # noise made them differ

    def test_a_perfect_bench_puts_full_scale_where_the_model_says(self):
        """C's fit must recover shift 0 and scale 1 when nothing is wrong."""
        rows = self._collect(pats=self.legacy)
        with tempfile.TemporaryDirectory() as tmp:
            reads, _ = self._report(rows, tmp)
        peak = R.psi_peak(reads)
        for tag in ("start", "end"):
            self.assertAlmostEqual(peak[tag]["peak_deg"], 0.0, places=6)
            self.assertAlmostEqual(peak[tag]["scale"], 1.0, places=6)
            self.assertAlmostEqual(peak[tag]["fs_meas"], self.scale.fs, places=9)

    def test_a_phase_error_shows_up_as_a_shifted_peak(self):
        """The zero above is only worth something if a real error moves it."""
        rows = self._collect(pats=self.legacy)
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
        rows = self._collect(noise=1e-4, pats=self.legacy)
        with tempfile.TemporaryDirectory() as tmp:
            path = V.write_verify_csv(
                Path(tmp) / "system_accuracy_test.csv", self.pairs, rows,
                meta={"step7_json": STEP7, "status": "complete"},
                extra=["kind", "label", "seq", "target_alpha", "alpha",
                       "coherent_v", "sb_pred_v"])
            R.report_csv(path, STEP7, out_dir=tmp)
            self.assertTrue((Path(tmp) / "system_accuracy_test_report.png").exists())


class YSatFitTests(unittest.TestCase):
    """The one number step 8 exists to produce."""

    @classmethod
    def setUpClass(cls):
        cls.models, cls.phases, cls.pairs = model()
        cls.scale, cls.patterns, cls.preds = SA.build(
            cls.models, cls.phases, cls.pairs, seed=11, ladder=40)
        cls.pred = np.array([float(V.predict(cls.models, cls.phases, cls.pairs,
                                             p.x, p.w, coherent=True))
                             for p in cls.patterns])

    def test_it_recovers_a_y_sat_it_was_given(self):
        """Saturate a perfect bench by hand; the fit must find the constant back."""
        for truth in (3.0, 5.0, 12.0):
            got = V.fit_y_sat(V.saturate(self.pred, truth), self.pred)
            self.assertAlmostEqual(got["y_sat_v"], truth, places=2)
            self.assertLess(got["rmse_v"], 1e-5)      # 0.01 mV: grid resolution
            self.assertFalse(got["unbounded"])

    def test_noise_widens_the_interval_but_keeps_the_answer(self):
        rng = np.random.default_rng(0)
        y = V.saturate(self.pred, 5.0) + rng.normal(0, 2e-3, self.pred.size)
        got = V.fit_y_sat(y, self.pred, np.full(self.pred.size, 2e-3))
        self.assertLess(abs(got["y_sat_v"] - 5.0), 5 * got["sigma_v"])
        self.assertGreater(got["sigma_v"], 0.0)
        self.assertLess(got["lo_v"], 5.0)
        self.assertGreater(got["hi_v"], 5.0)

    def test_a_dim_sweep_cannot_bound_it_and_says_so(self):
        """The whole reason the ladder now runs to alpha = 1.

        The curvature grows with |E|^2, so at the bottom of the scale it sits
        under the noise and no fit there can bound Y_sat.  Same noise, same
        model, same number of reads -- only the brightness differs, and that
        alone decides whether the answer is a measurement or a limit.
        """
        rng = np.random.default_rng(3)
        noise = 2e-3                            # what the bench actually delivers
        for scale, want_unbounded in ((0.02, True), (1.0, False)):
            dim = self.pred * scale
            y = V.saturate(dim, 5.0) + rng.normal(0, noise, dim.size)
            got = V.fit_y_sat(y, dim, np.full(dim.size, noise))
            self.assertEqual(got["unbounded"], want_unbounded,
                             f"at {100 * scale:g} % of full scale")

    def test_the_gain_lever_is_reported(self):
        """A g=1 Y_sat is levered by Y_sat/Y_max; the caller has to be told."""
        got = V.fit_y_sat(V.saturate(self.pred, 5.0), self.pred)
        self.assertAlmostEqual(got["gain_lever"],
                               got["y_sat_v"] / self.pred.max(), places=9)
        self.assertAlmostEqual(got["gain_lever"], 5.0 / self.pred.max(), places=2)
        self.assertGreater(got["gain_lever"], 5.0)

    def test_the_ceiling_reads_give_measured_full_scale(self):
        """F is what turns a predicted Y_max into a measured one."""
        truth = 5.0
        slm = _FakeSLM()
        daq = _FakeDAQ(slm, self.models, self.phases, noise=0.0)
        layout = SimpleNamespace(n_channels=len(self.pairs))
        acq = replace(S8.DEFAULT_ACQ, settle_s=0.0, t_meas_s=0.0)
        x = np.stack([p.x for p in self.patterns])[None]
        w = np.stack([p.w for p in self.patterns])[None]
        full = [float(V.predict(self.models, self.phases, self.pairs, p.x, p.w))
                for p in self.patterns]
        extra = [[{"kind": p.kind, "label": p.label, "seq": i,
                   "target_alpha": f"{p.target_alpha:.6f}", "alpha": f"{p.alpha:.6f}",
                   "coherent_v": f"{c:.9g}", "sb_pred_v": f"{q - c:.9g}"}
                  for i, (p, c, q) in enumerate(zip(self.patterns, self.pred, full))]]
        rows: list[dict] = []
        S8.run_patterns(daq, slm, layout, [tuple(self.pairs)], x, w,
                        self.models, self.phases, rows=rows, acq=acq,
                        encode=_encode, log=lambda _: None, extra=extra)
        # the fake bench is linear, so saturate its coherent reads by hand
        for row, c in zip(rows, self.pred):
            row["mean_v"] = row["dark_v"] + V.saturate(c, truth) + (row["pred_v"] - c)

        with tempfile.TemporaryDirectory() as tmp:
            path = V.write_verify_csv(
                Path(tmp) / "calib_step8_0000_0000.csv", self.pairs, rows,
                meta={"step7_json": STEP7, "y_max_v": self.scale.fs,
                      "status": "complete"},
                extra=["kind", "label", "seq", "target_alpha", "alpha",
                       "coherent_v", "sb_pred_v"])
            result = SA.fit_and_write(path, STEP7, out_dir=tmp, log=lambda *_: None)
            written = json.loads(
                (Path(tmp) / "calib_step8_result_0000_0000.json").read_text())
            self.assertTrue((Path(tmp) / "calib_step8_0000_0000_ysat.png").exists())
            # the caller that chose out_dir should not have to re-derive either
            # name from the CSV: the fit reports where it put both files
            self.assertEqual(Path(result["result_json"]),
                             Path(tmp) / "calib_step8_result_0000_0000.json")
            self.assertEqual(Path(result["png"]),
                             Path(tmp) / "calib_step8_0000_0000_ysat.png")

        self.assertAlmostEqual(result["y_sat_v"], truth, places=2)
        fs = result["full_scale"]
        self.assertEqual(result["n_full_scale"], 2)
        # the ceiling was driven, so measured |E|^2 must come back at Y_max
        self.assertAlmostEqual(fs["y_max_meas_v"], self.scale.fs, places=4)
        self.assertAlmostEqual(fs["y_max_pred_v"], self.scale.fs, places=9)
        # a perfect bench does not drift between the two ends
        self.assertAlmostEqual(fs["drift_frac"], 0.0, places=9)
        # and the JSON downstream reads carries the anchor it was defined by
        self.assertAlmostEqual(written["y_sat_v"], result["y_sat_v"])
        self.assertEqual(Path(written["gain_reference"]["csv"]).name,
                         "calib_step8_0000_0000.csv")


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
                        rows=rows, acq=replace(S8.DEFAULT_ACQ, settle_s=0.0, t_meas_s=0.0),
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
                        acq=replace(S8.DEFAULT_ACQ, settle_s=0.0, t_meas_s=0.0),
                        encode=_encode, log=lambda _: None)
        self.assertTrue(all("pmt_c" not in r for r in rows))

class _RangeSpyDAQ(_FakeDAQ):
    """A ``_FakeDAQ`` that ``can_autorange`` accepts, and that records set_range.

    ``can_autorange`` looks for ``_settings`` and ``last_values``; the plain
    fake has neither, so it could never exercise the range path either way.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        from daq_module import DAQMonitorSettings
        self._settings = DAQMonitorSettings()
        self.last_values = np.zeros(4)
        self.ranges: list[float] = []

    def configure_monitor(self, settings):
        self._settings = settings
        self.ranges.append(float(settings.max_val))


class FixedRangeTests(unittest.TestCase):
    """Nothing in a run touches the board's input range.

    There used to be two mechanisms here, both gone.  ``range_for(pred)`` picked
    each read's range and ``set_range`` applied it, so a 90-rung ladder spanned
    three or four ranges and took its dark on a fifth -- putting the board's
    per-range offset straight into the dark subtraction.  On top of that a
    near-rail raw peak triggered a re-read one range up, which in the 0928 gain
    sweep fired on a spike 1700x the trace's own spread and latched the wider
    range for the rest of the run.

    A ``_RangeSpyDAQ`` that records every ``configure_monitor`` is the test: the
    list has to stay empty.
    """

    def _run(self, acq):
        models, phases, pairs = model()
        _, patterns, _ = SA.build(models, phases, pairs, seed=3, ladder=6)
        slm, layout = _FakeSLM(), SimpleNamespace(n_channels=len(pairs))
        daq = _RangeSpyDAQ(slm, models, phases)
        x = np.stack([p.x for p in patterns])[None]
        w = np.stack([p.w for p in patterns])[None]
        rows: list[dict] = []
        S8.run_patterns(daq, slm, layout, [tuple(pairs)], x, w, models, phases,
                        rows=rows, acq=acq, encode=_encode, log=lambda _: None)
        return daq, rows

    def test_the_range_is_never_reconfigured(self):
        acq = replace(S8.DEFAULT_ACQ, settle_s=0.0, t_meas_s=0.0)
        daq, rows = self._run(acq)
        self.assertEqual(daq.ranges, [])
        self.assertEqual({r["range_v"] for r in rows}, {acq.range_v})

    def test_the_old_policy_really_would_have_moved_it(self):
        """So the test above is about the code, not about a flat ladder."""
        acq = replace(S8.DEFAULT_ACQ, settle_s=0.0, t_meas_s=0.0, range_v=0.1)
        _, rows = self._run(acq)
        self.assertGreater(len({S8.range_for(r["pred_v"])[0] for r in rows}), 1)


class ProgressCallbackTests(unittest.TestCase):
    """``run_patterns`` announces every read before it takes it.

    The point is not the numbers -- the row already carries those -- but the
    ANNOUNCEMENT: a step-8 read is a whole averaging window behind a ten second
    opening dark, and the GUI used to show nothing at all until the first one
    landed.  So the contract under test is that an event arrives BEFORE each
    read, that ``step`` counts finished pattern reads (never the darks), and
    that a caller passing nothing still gets the old behaviour.
    """

    ACQ = property(lambda self: replace(S8.DEFAULT_ACQ, settle_s=0.0,
                                        t_meas_s=0.0))

    def _run(self, n=4, acq=None, **kw):
        models, phases, pairs = model()
        _, patterns, _ = SA.build(models, phases, pairs, seed=3, ladder=4)
        patterns = patterns[:n]
        slm, layout = _FakeSLM(), SimpleNamespace(n_channels=len(pairs))
        x = np.stack([p.x for p in patterns])[None]
        w = np.stack([p.w for p in patterns])[None]
        rows: list[dict] = []
        S8.run_patterns(_FakeDAQ(slm, models, phases), slm, layout, [tuple(pairs)],
                        x, w, models, phases, rows=rows,
                        acq=self.ACQ if acq is None else acq,
                        encode=_encode, log=lambda _: None, **kw)
        return rows, len(patterns)

    def test_every_read_is_announced_before_it_lands(self):
        seen: list = []
        rows, n = self._run(progress_callback=seen.append)

        self.assertEqual(len(rows), n)
        # one dark (announced + landed) plus two per pattern
        self.assertEqual(len(seen), 2 + 2 * n)
        darks = [p for p in seen if p.single]
        self.assertEqual([p.mean_v is None for p in darks], [True, False])
        self.assertTrue(all(p.total == n for p in seen))

        reads = [p for p in seen if not p.single]
        # announcement, landing, announcement, landing, ...
        self.assertEqual([p.mean_v is None for p in reads],
                         [True, False] * n)
        # step counts FINISHED pattern reads, so it is 0 through the dark and
        # through the first announcement, and n once the last read lands.
        self.assertEqual([p.step for p in darks], [0, 0])
        self.assertEqual([p.step for p in reads],
                         [k for i in range(n) for k in (i, i + 1)])

    def test_a_landed_read_carries_its_row_and_its_net_signal(self):
        seen: list = []
        rows, _ = self._run(progress_callback=seen.append)
        landed = [p for p in seen if not p.single and p.mean_v is not None]

        self.assertEqual([p.row for p in landed], rows)
        for p, row in zip(landed, rows):
            self.assertAlmostEqual(p.net_v, row["mean_v"] - row["dark_v"])
            self.assertAlmostEqual(p.pred_v, row["pred_v"])
            self.assertEqual(p.range_v, row["range_v"])
        self.assertTrue(all(p.net_v is None for p in seen if p.mean_v is None))

    def test_the_announcement_carries_the_window_it_is_about_to_average(self):
        acq = replace(self.ACQ, t_meas_s=4.0)
        seen: list = []
        self._run(acq=acq, progress_callback=seen.append)
        for p in seen:
            self.assertEqual(p.t_read_s,
                             acq.t_meas_s)

    def test_without_a_callback_nothing_changes(self):
        rows, n = self._run()
        self.assertEqual(len(rows), n)


class PmtInStepsSixAndSevenTests(unittest.TestCase):
    """Steps 6 and 7 carry the thermistor in two trailing columns.

    The contract that matters is compatibility: the columns and the meta lines
    must not change what either step's fit produces, and a v1 re-fit of a v2
    step-6 CSV -- which that CSV's docstring promises -- must still work with
    both present.
    """

    S6_CSV = ROOT / "src/calib_data/calib_0917_1617/calib_step6v2_meas_0917_1811.csv"
    S7_CSV = ROOT / "src/calib_data/calib_0917_1617/calib_step7_meas_0917_1831.csv"
    S6_JSON = ROOT / "src/calib_data/calib_0917_1617/calib.json"
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

class ProbeTests(unittest.TestCase):
    """One displayed level, one read: does the readout still look linear?

    The load-bearing tests are the two peak ones.  Clipping, amplifier
    compression and atomic saturation all bend the top of a ladder down the same
    way, and ``fit_y_sat`` has one free parameter, so any of them returns a
    clean-looking wrong ``Y_sat``.  The RAW TRACE PEAK is the only thing in the
    chain that sees the first two: the 20 Hz low-pass pulls a flat-top back under
    the rail, so the reported mean stays perfectly reasonable.
    """

    FS_HOT = 1.010509        # the 0928 ten-pair ceiling, past the 1 V limit
    FS_COLD = 0.454          # the 0917 ceiling, comfortably under it

    def _verdict(self, y_sat, *, fs=None, alpha=1.0, sb=1.2e-3, dark=-1.2e-4,
                 peak_extra=6e-2, mean_override=None, peak_override=None,
                 ceiling_v=SA.LADDER_CEILING_V):
        fs = self.FS_HOT if fs is None else fs
        coh = alpha * fs
        meas = V.saturate(coh, y_sat) + sb
        if mean_override is not None:
            meas = mean_override
        peak = meas + peak_extra if peak_override is None else peak_override
        return SA.probe_verdict(
            alpha=alpha, y_max_v=fs, coherent_v=coh, sb_pred_v=sb,
            mean_v=meas + dark, dark_v=dark, std_v=2.2e-4,
            peak_v=peak, range_v=1.0, ceiling_v=ceiling_v)

    def test_one_read_recovers_the_y_sat_it_was_taken_under(self):
        for truth in (3.0, 4.82, 12.0):
            with self.subTest(y_sat=truth):
                d = self._verdict(truth)
                self.assertAlmostEqual(d["y_sat_implied_v"], truth, places=6)

    def test_it_recovers_y_sat_from_a_level_that_is_not_the_top(self):
        """The probe's whole point is that alpha is the caller's."""
        for alpha in (0.1, 0.293, 0.55, 1.0):
            with self.subTest(alpha=alpha):
                d = self._verdict(4.82, alpha=alpha)
                self.assertAlmostEqual(d["y_sat_implied_v"], 4.82, places=5)
                self.assertAlmostEqual(d["alpha"], alpha)

    def test_a_flat_top_the_mean_hides_is_caught_by_the_peak(self):
        # A mean at 780 mV is 22 % clear of the limit, and the trace is railed.
        d = self._verdict(4.82, mean_override=0.780, peak_override=0.9995)
        self.assertLess(d["measured_v"], 0.8 * d["limit_v"])
        self.assertTrue(d["clipped"])
        self.assertEqual(d["verdict"], "CLIPPED")

    def test_a_peak_approaching_the_rail_is_tight_though_the_mean_is_not(self):
        """Compression starts before the rail, and the peak gets there first.

        This is why LADDER_CEILING_V is 450 mV and not 700: the raw trace runs
        ~1.2x the mean on this bench, so a mean with 16 % of margin has a peak
        with 10 %.
        """
        d = self._verdict(4.82)                       # 835 mV mean, 895 mV peak
        self.assertLess(d["measured_v"], 0.85 * d["limit_v"])
        self.assertGreater(d["peak_fraction"], 0.85)
        self.assertFalse(d["clipped"])
        self.assertEqual(d["verdict"], "TIGHT")

    def test_a_comfortable_read_is_ok(self):
        d = self._verdict(4.82, fs=self.FS_COLD)
        self.assertFalse(d["clipped"])
        self.assertEqual(d["verdict"], "OK")
        self.assertLess(d["peak_fraction"], 0.85)

    def test_a_read_that_beats_the_linear_model_reports_no_saturation(self):
        d = self._verdict(None, fs=self.FS_COLD, peak_extra=0.0)
        self.assertFalse(np.isfinite(d["y_sat_implied_v"]))

    def test_a_moved_ceiling_is_what_the_probe_judges_against(self):
        """Raise the box and a read that WAS over the ceiling stops being over it.

        The probe is how the ceiling gets measured, so it has to answer about
        the ceiling being tried, not about the module's default.
        """
        d = self._verdict(4.82, alpha=0.550, ceiling_v=0.700)
        self.assertAlmostEqual(d["ceiling_v"], 0.700)
        self.assertAlmostEqual(d["alpha_cap"] * self.FS_HOT, 0.700, places=9)
        self.assertGreater(d["measured_v"], SA.LADDER_CEILING_V)
        self.assertFalse(d["over_ceiling"])
        # and the default still calls that same read over the ceiling
        self.assertTrue(self._verdict(4.82, alpha=0.550)["over_ceiling"])

    def test_it_reports_where_the_ladder_would_stop(self):
        """So the panel can say "you are above the sweep" without guessing."""
        d = self._verdict(4.82)
        self.assertAlmostEqual(d["alpha_cap"],
                               SA.ladder_levels(self.FS_HOT)["cap"])
        self.assertAlmostEqual(d["ceiling_v"], SA.LADDER_CEILING_V)
        # 835 mV against a 450 mV ceiling
        self.assertTrue(d["over_ceiling"])

    def test_a_read_inside_the_ladder_is_not_flagged_over_the_ceiling(self):
        d = self._verdict(4.82, alpha=SA.ladder_levels(self.FS_HOT)["cap"])
        self.assertFalse(d["over_ceiling"])
        self.assertEqual(d["verdict"], "OK")


class _SaturatingDAQ(_FakeDAQ):
    """A bench whose atoms saturate, and which exposes a raw trace."""

    def __init__(self, *args, y_sat, rail=None, **kw):
        super().__init__(*args, **kw)
        self.y_sat, self.rail = y_sat, rail
        # Both attributes together are what can_autorange() looks for, so this
        # fake goes down the same range-and-window path the real board does --
        # which is the point: range_for() sends a ~1 V ceiling to +/-2 V, where
        # the near-rail escalation can no longer see a 1 V flat top.
        self.last_values = np.zeros(4)
        self._settings = DAQMonitorSettings()

    def monitor_cycle(self, timeout=None, single=False):
        x_vals, w_vals = self.slm.shown
        pairs = sorted(self.models)
        slots = [k - S8.PAIR_INDEX_BASE for k in pairs]
        y = float(V.predict(self.models, self.phases, pairs,
                            x_vals[slots], w_vals[slots], y_sat=self.y_sat))
        v = y + self.dark
        if self.rail is not None:
            v = min(v, self.rail)
        # The TIA is negative for light; the raw trace is what the board sampled.
        self.last_values = np.full(4, -v)
        return SimpleNamespace(value=-v, std=1e-5)

    def configure_monitor(self, settings):
        self._settings = settings


class ProbeReadTests(unittest.TestCase):
    """One probe read through the real drive loop, against a fake bench."""

    ALPHA = 1.0

    @classmethod
    def setUpClass(cls):
        cls.models, cls.phases, cls.pairs = model()
        cls.scale = V.full_scale(cls.models, cls.phases, cls.pairs)
        # The GUI solves the probe pattern with ceiling_pattern, so the test
        # drives what the bench drives.
        cls.pattern = SA.ceiling_pattern(cls.models, cls.phases, cls.pairs,
                                         cls.ALPHA, seed=3)

    def _read(self, y_sat, rail=None, limit_v=None):
        # The 0911 file on disk tops out at ~317 mV, well under the real 1 V
        # amplifier limit, so a test that wants to see a railed read has to say
        # where the rail is -- and the plan must be told the same number, or the
        # verdict would be judging the read against a limit it never met.
        limit_v = SA.AMP_LIMIT_V if limit_v is None else limit_v
        slm = _FakeSLM()
        daq = _SaturatingDAQ(slm, self.models, self.phases, y_sat=y_sat, rail=rail)
        layout = SimpleNamespace(n_channels=len(self.pairs))
        acq = replace(S8.DEFAULT_ACQ, settle_s=0.0, t_meas_s=0.0)
        rows: list[dict] = []
        S8.run_patterns(daq, slm, layout, [tuple(self.pairs)],
                        self.pattern.x[None, None], self.pattern.w[None, None],
                        self.models, self.phases, rows=rows, acq=acq,
                        encode=_encode, log=lambda _: None, y_sat=None)
        coh = float(V.predict(self.models, self.phases, self.pairs,
                              self.pattern.x, self.pattern.w, coherent=True))
        tot = float(V.predict(self.models, self.phases, self.pairs,
                              self.pattern.x, self.pattern.w))
        r = rows[0]
        return r, SA.probe_verdict(
            alpha=self.ALPHA, y_max_v=self.scale.fs, limit_v=limit_v,
            coherent_v=coh, sb_pred_v=tot - coh,
            mean_v=r["mean_v"], dark_v=r["dark_v"], std_v=r["std_v"],
            peak_v=r["peak_v"], range_v=r["range_v"])

    def test_the_loop_stamps_the_raw_peak_on_the_row(self):
        row, _ = self._read(6.0)
        self.assertTrue(np.isfinite(row["peak_v"]))
        # The peak is the |trace| max, so the TIA's sign must not reach it.
        self.assertGreater(row["peak_v"], 0.0)

    def test_the_read_recovers_the_benchs_own_y_sat(self):
        _, d = self._read(6.0)
        self.assertAlmostEqual(d["y_sat_implied_v"], 6.0, places=4)
        self.assertFalse(d["clipped"])

    def test_a_railed_bench_is_reported_clipped(self):
        _, d = self._read(30.0, rail=0.300, limit_v=0.300)
        self.assertTrue(d["clipped"])
        self.assertEqual(d["verdict"], "CLIPPED")

    def test_the_peak_column_survives_the_csv(self):
        row, _ = self._read(6.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = V.write_verify_csv(
                Path(tmp) / "calib_step8_probe.csv", self.pairs, [row],
                meta={"status": "complete"}, extra=["peak_v"])
            _, blocks, _ = V.load_verify_csv(path)
        self.assertAlmostEqual(float(blocks[0].extra["peak_v"][0]),
                               row["peak_v"], places=9)


if __name__ == "__main__":
    unittest.main()
