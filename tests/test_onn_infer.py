"""The inference run's instrument correction: the anchors, g, and the dark.

Three things are load-bearing here and none of them is the optics.

``build_anchors`` must hand back distinct, reproducible drives, and
``measure_gain`` must fit its first block against the model (``g0``) and every
later one as a ratio against THOSE reads -- the whole point of taking a ratio
on an identical SLM frame is that no model enters it.

``fit_gain`` must be a slope through the origin, and its rms must be non-zero
when the reads are NOT a pure rescaling of the reference, because that residual
is the only thing in the run that can say "something other than the gain moved".

``apply_gain`` must interpolate.  The drift is continuous and the anchors sample
it every fifteen minutes, so holding the earlier value leaves a sawtooth whose
amplitude is the full drift across the interval -- which is exactly the error
the anchors were added to remove.
"""
from __future__ import annotations

import contextlib
import csv
import json
import sys
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calibration_module.fit import verify as V  # noqa: E402
from calibration_module.fit.phase import PairModel  # noqa: E402
from calibration_module.measure import patterns as P  # noqa: E402
from onn_module import onn_infer as OI  # noqa: E402

PAIRS = [1, 2, 3]


def _has_step3(path: Path) -> bool:
    """A step-7 result the layout loader can actually use."""
    return '"step3"' in path.read_text(encoding="utf-8")


# ======================================================================
# the anchors, solved not read
# ======================================================================

def _three_pair_model(eta=0.1):
    """A flat three-pair forward model: eta per pair, no background, phases 0."""
    models = {k: PairModel(index=k, eta=eta, a_x=0.0, q_x=0.0,
                           a_w=0.0, q_w=0.0, d=0.0) for k in PAIRS}
    return models, {k: 0.0 for k in PAIRS}


class AnchorTests(unittest.TestCase):
    """``build_anchors``: what the ladder is, and that it is not yet a reference."""

    def test_rungs_come_back_in_alpha_order_inside_the_band(self):
        models, phases = _three_pair_model()
        got = OI.build_anchors(models, phases, PAIRS, seed=3, n=6, band=(0.5, 0.9))
        self.assertEqual(got["n"], 6)
        self.assertTrue(np.all(np.diff(got["alpha"]) > 0))
        self.assertGreaterEqual(got["alpha"].min(), 0.5 - 0.01)
        self.assertLessEqual(got["alpha"].max(), 0.9 + 0.01)

    def test_every_rung_is_a_different_drive(self):
        """Ten repeats of one frame would be a mean with extra steps."""
        models, phases = _three_pair_model()
        got = OI.build_anchors(models, phases, PAIRS, seed=3)
        frames = {tuple(np.r_[x, w]) for x, w in zip(got["x"], got["w"])}
        self.assertEqual(len(frames), got["n"])

    def test_the_same_seed_gives_the_same_ladder(self):
        """The seed is written to the CSV so the frames can be rebuilt."""
        models, phases = _three_pair_model()
        a = OI.build_anchors(models, phases, PAIRS, seed=11)
        b = OI.build_anchors(models, phases, PAIRS, seed=11)
        np.testing.assert_array_equal(a["x"], b["x"])
        np.testing.assert_array_equal(a["w"], b["w"])

    def test_a_bright_model_is_capped_at_step_8_s_ceiling(self):
        """eta = 0.25 per pair puts Y_max at 562 mV, over the 450 mV ceiling.

        The band must stop at the ceiling, as step 8's does, so the brightest
        anchor never exceeds what step 8 was willing to drive.
        """
        models, phases = _three_pair_model(eta=0.25)
        got = OI.build_anchors(models, phases, PAIRS, seed=3, band=(0.5, 1.0))
        self.assertTrue(got["capped"])
        self.assertAlmostEqual(got["band"][1], OI.LADDER_CEILING_V / got["y_max_v"],
                               places=9)
        self.assertLessEqual(got["pred_v"].max(), OI.LADDER_CEILING_V * 1.01)

    def test_a_dim_model_runs_the_band_as_asked(self):
        models, phases = _three_pair_model(eta=0.1)
        got = OI.build_anchors(models, phases, PAIRS, seed=3, band=(0.5, 0.9))
        self.assertFalse(got["capped"])
        self.assertEqual(got["band"], (0.5, 0.9))

    def test_the_reference_volts_do_not_exist_until_read(self):
        models, phases = _three_pair_model()
        got = OI.build_anchors(models, phases, PAIRS, seed=3)
        self.assertIsNone(got["v_ref"])


class _FakeRunPatterns:
    """``run_patterns`` as a pure function of a gain: no SLM, no DAQ.

    Each rung reads ``g * v_true + dark``; ``dark_v`` is what was passed in, or
    a fixed value when none was (the first block measures one).  Rungs are told
    apart by the ``alpha`` their row carries, so a call for a subset of the
    ladder (a re-read, a ladder with a rung turned off) reads the right ones.

    ``spikes`` is a queue, one dict per CALL, of ``{rung: multiplier}`` applied
    to that call's reads only -- a level shift lasting one window, which is
    what the bench actually does (trace std stays ordinary).
    """

    def __init__(self, v_true, alphas, *, g=1.0, dark=1e-4, spikes=()):
        self.v_true, self.alphas = np.asarray(v_true, float), np.asarray(alphas, float)
        self.g, self.dark = g, dark
        self.spikes = list(spikes)
        self.calls, self.read = 0, []

    def __call__(self, daq, slm, layout, blocks, x, w, models, phases, *,
                 rows, extra, acq, y_sat, dark, log, row_hook):
        self.calls += 1
        spike = self.spikes.pop(0) if self.spikes else {}
        d = self.dark if dark is None else dark
        for e in extra[0]:
            i = int(np.argmin(np.abs(self.alphas - float(e["alpha"]))))
            self.read.append(i)
            row = dict(e, mean_v=self.g * self.v_true[i] * spike.get(i, 1.0) + d,
                       dark_v=d)
            row_hook(row)
            rows.append(row)


class _CountingPMT:
    COLUMNS = ("pmt_ohm_Ohm", "pmt_c")

    def __init__(self):
        self.reads = 0

    def read(self):
        self.reads += 1

    def __call__(self, row):
        row["pmt_c"] = "25.0"


class MeasureGainTests(unittest.TestCase):
    """The first block fits g0 to the model; every later one is a ratio to it."""

    ALPHA = np.array([0.5, 0.6, 0.7, 0.8, 0.9])

    def setUp(self):
        n = len(self.ALPHA)
        self.v_true = np.array([0.25, 0.30, 0.35, 0.40, 0.45])
        # the model predicts the truth to 1 % per rung: inside GAIN_TOL
        pred = self.v_true * np.array([1.01, 0.99, 1.0, 1.01, 0.99])
        self.anchors = {"n": n, "alpha": self.ALPHA, "pred_v": pred,
                        "x": np.zeros((n, 3)), "w": np.zeros((n, 3)),
                        "v_ref": None, "keep": np.ones(n, dtype=bool)}
        self.bench = {k: None for k in ("daq", "slm", "layout", "models",
                                        "phases", "y_sat", "acq")}
        self.bench["pairs"] = PAIRS

    def _fake(self, **kw):
        return _FakeRunPatterns(self.v_true, self.ALPHA, **kw)

    def _g0(self, g: float = 1.0, keep=None) -> float:
        """What the first block must report: the origin slope of its volts
        against ``pred_v`` over the rungs it kept."""
        pred = self.anchors["pred_v"]
        k = np.ones(len(pred), dtype=bool) if keep is None else np.asarray(keep)
        return OI.fit_gain(g * self.v_true[k], pred[k])[0]

    def _measure(self, fake, rows, dark=None, pmt=None):
        with mock.patch.object(OI, "run_patterns", fake):
            return OI.measure_gain(self.bench, self.anchors, seq0=-1, t0=0.0,
                                   rows=rows, dark=dark, pmt=pmt,
                                   log=lambda _m: None)

    def test_the_first_block_fits_g0_to_the_model_and_keeps_its_volts(self):
        rows: list[dict] = []
        g, rms, dark = self._measure(self._fake(g=1.37), rows)
        g0 = self._g0(1.37)
        self.assertAlmostEqual(g, g0, places=12)
        self.assertAlmostEqual(g / 1.37, 1.0, delta=0.01)   # the model is good to 1 %
        self.assertGreater(rms, 0.0)                         # ... and not perfect
        self.assertEqual(self.anchors["g0"], g)
        np.testing.assert_allclose(self.anchors["v_ref"], 1.37 * self.v_true)
        self.assertTrue(self.anchors["keep"].all())
        self.assertEqual({r["g"] for r in rows}, {f"{g0:.9g}"})
        self.assertEqual({r["kind"] for r in rows}, {"gain"})
        self.assertAlmostEqual(dark, 1e-4)

    def test_a_first_block_that_reads_the_model_exactly_is_g0_1(self):
        self.anchors["pred_v"] = self.v_true.copy()
        g, rms, _ = self._measure(self._fake(g=1.0), [])
        self.assertAlmostEqual(g, 1.0, places=12)
        self.assertLess(rms, 1e-12)

    def test_a_later_block_measures_the_change_since_the_first(self):
        """g = g0 * ratio: the ratio sees the bench, g0 the model."""
        rows: list[dict] = []
        self._measure(self._fake(g=1.0), rows)
        g0 = self._g0()
        g, rms, _ = self._measure(self._fake(g=1.07), rows, dark=1e-4)
        self.assertAlmostEqual(g, 1.07 * g0, places=9)
        self.assertLess(rms, 1e-12)
        self.assertEqual(rows[-1]["g"], f"{1.07 * g0:.9g}")

    def test_a_first_block_at_the_dark_is_refused(self):
        with self.assertRaises(SystemExit):
            self._measure(_FakeRunPatterns(np.zeros(5), self.ALPHA), [])
        self.assertIsNone(self.anchors["v_ref"])

    def test_the_pmt_is_read_once_per_block(self):
        pmt = _CountingPMT()
        rows: list[dict] = []
        self._measure(self._fake(), rows, pmt=pmt)
        self._measure(self._fake(), rows, dark=1e-4, pmt=pmt)
        self.assertEqual(pmt.reads, 2)
        self.assertTrue(all(r["pmt_c"] == "25.0" for r in rows))

    # -- the +/- GAIN_TOL rule ------------------------------------------

    def test_a_spiked_rung_in_a_later_block_is_dropped_from_the_fit(self):
        """Rung 2 reads 8 % high for one window: g must not see it."""
        rows: list[dict] = []
        self._measure(self._fake(), rows)
        fake = self._fake(g=1.02, spikes=[{2: 1.08}])
        g, rms, _ = self._measure(fake, rows, dark=1e-4)
        self.assertAlmostEqual(g, 1.02 * self._g0(), places=9)
        self.assertLess(rms, 1e-12)
        block = rows[-5:]
        self.assertEqual([r["kind"] for r in block],
                         ["gain", "gain", "gain_drop", "gain", "gain"])
        self.assertEqual(block[2]["g"], f"{g:.9g}")        # stamped, as evidence

    def test_a_rung_inside_the_tolerance_is_kept(self):
        rows: list[dict] = []
        self._measure(self._fake(), rows)
        self._measure(self._fake(g=1.02, spikes=[{2: 1.02}]), rows, dark=1e-4)
        self.assertEqual({r["kind"] for r in rows[-5:]}, {"gain"})

    def test_a_spiked_rung_in_the_first_block_is_re_read_once(self):
        """Judged against the model; the re-read replaces the spike."""
        rows: list[dict] = []
        fake = self._fake(spikes=[{3: 1.08}])
        g, _, _ = self._measure(fake, rows)
        self.assertEqual(fake.calls, 2)
        self.assertEqual(fake.read, [0, 1, 2, 3, 4, 3])
        np.testing.assert_allclose(self.anchors["v_ref"], self.v_true)
        self.assertTrue(self.anchors["keep"].all())
        self.assertAlmostEqual(g, self._g0(), places=12)   # fitted on the re-read
        self.assertEqual([r["kind"] for r in rows],
                         ["gain", "gain", "gain", "gain_drop", "gain", "gain"])
        self.assertEqual([r["seq"] for r in rows], [-1, -2, -3, -4, -5, -6])

    def test_a_rung_off_twice_in_the_first_block_leaves_the_ladder(self):
        rows: list[dict] = []
        fake = self._fake(spikes=[{3: 1.08}, {3: 0.92}])
        self._measure(fake, rows)
        np.testing.assert_array_equal(self.anchors["keep"],
                                      [True, True, True, False, True])
        self.assertEqual([r["kind"] for r in rows][3::2], ["gain_drop", "gain_drop"])
        # g0 is fitted over the four rungs that stayed
        self.assertAlmostEqual(self.anchors["g0"],
                               self._g0(keep=self.anchors["keep"]), places=12)
        # ... and is never displayed again
        fake2 = self._fake(g=1.03)
        g, rms, _ = self._measure(fake2, rows, dark=1e-4)
        self.assertEqual(fake2.read, [0, 1, 2, 4])
        self.assertAlmostEqual(g, 1.03 * self.anchors["g0"], places=9)
        self.assertEqual(len(rows), 6 + 4)

    def test_a_block_with_every_rung_off_is_fitted_whole(self):
        """Five rungs scattered +/- 10 %: dropping all would leave no fit."""
        rows: list[dict] = []
        self._measure(self._fake(), rows)
        wild = {0: 0.90, 1: 1.10, 2: 0.90, 3: 1.10, 4: 0.90}
        g, rms, _ = self._measure(self._fake(spikes=[wild]), rows, dark=1e-4)
        self.assertTrue(np.isfinite(g))
        self.assertEqual({r["kind"] for r in rows[-5:]}, {"gain"})


class FlagOffRungsTests(unittest.TestCase):

    def test_a_pure_rescaling_flags_nothing(self):
        ref = np.array([0.2, 0.3, 0.4, 0.5])
        self.assertFalse(OI.flag_off_rungs(1.08 * ref, ref).any())

    def test_the_rung_off_the_slope_is_the_one_flagged(self):
        """One rung 8 % high pulls the slope ~2 %; only that rung is off."""
        ref = np.array([0.2, 0.3, 0.4, 0.5])
        v = ref * np.array([1.0, 1.0, 1.08, 1.0])
        np.testing.assert_array_equal(OI.flag_off_rungs(v, ref),
                                      [False, False, True, False])

    def test_the_tolerance_is_relative_and_visible(self):
        ref = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
        v = ref * np.array([1.0, 1.0, 1.08, 1.0, 1.0])
        g, _ = OI.fit_gain(v, ref)
        dev = abs(v[2] / (g * ref[2]) - 1)
        self.assertTrue(OI.flag_off_rungs(v, ref, tol=dev - 1e-9)[2])
        self.assertFalse(OI.flag_off_rungs(v, ref, tol=dev + 1e-9).any())
        self.assertLess(OI.GAIN_TOL, dev)          # the default catches it

    def test_a_reference_at_zero_is_off_not_a_crash(self):
        ref = np.array([0.0, 0.3, 0.4])
        v = np.array([0.0, 0.3, 0.4])
        np.testing.assert_array_equal(OI.flag_off_rungs(v, ref), [True, False, False])


# ======================================================================
# the fit
# ======================================================================

class FitGainTests(unittest.TestCase):

    def test_recovers_a_known_scale(self):
        ref = np.array([0.19, 0.24, 0.26, 0.28, 0.34])
        g, rms = OI.fit_gain(1.0732 * ref, ref)
        self.assertAlmostEqual(g, 1.0732, places=9)
        self.assertAlmostEqual(rms, 0.0, places=12)

    def test_is_through_the_origin(self):
        """An offset must land in the residual, not be absorbed by the slope.

        With a free intercept this fit would report g = 1 and rms = 0 for a
        shifted set of reads.  It must not: every anchor is bright, so an
        intercept there is fitted to noise (see the module docstring).
        """
        ref = np.array([0.19, 0.24, 0.26, 0.28, 0.34])
        g, rms = OI.fit_gain(ref + 0.01, ref)
        self.assertGreater(g, 1.0)
        self.assertGreater(rms, 1e-4)

    def test_a_pure_rescaling_leaves_no_residual_at_any_level(self):
        """The rms is the run's only "something else moved" detector."""
        ref = np.array([0.05, 0.19, 0.34])
        for g_true in (0.88, 1.0, 1.15):
            _, rms = OI.fit_gain(g_true * ref, ref)
            self.assertAlmostEqual(rms, 0.0, places=12)


# ======================================================================
# the interpolation
# ======================================================================

def onn_row(t, *, sb=0.0, y_pred=0.0, mean=0.0, dark=0.0, g=1.0):
    return {"kind": "onn", "t_s": f"{t:g}", "g": f"{g:g}", "mean_v": mean,
            "dark_v": dark, "sb_pred_v": f"{sb:g}", "y_pred_v": f"{y_pred:g}"}


def gain_rows(t, g, n=5):
    return [{"kind": "gain", "t_s": f"{t + i:g}", "g": f"{g:g}"} for i in range(n)]


class ApplyGainTests(unittest.TestCase):

    def test_interpolates_between_two_measurements(self):
        rows = (gain_rows(0, 1.0) + [onn_row(50, mean=0.1)] + gain_rows(100, 1.2))
        OI.apply_gain(rows, log=lambda _m: None)
        # marks land at the mean of each block's timestamps: 2 and 102.
        want = 0.1 / np.interp(50.0, [2.0, 102.0], [1.0, 1.2])
        self.assertAlmostEqual(float(rows[5]["y_meas_v"]), want, places=9)

    def test_clamps_outside_the_bracketing_measurements(self):
        """There is no anchor before the first one; the nearest is all there is."""
        rows = gain_rows(100, 1.5) + [onn_row(0, mean=0.3), onn_row(9999, mean=0.3)]
        OI.apply_gain(rows, log=lambda _m: None)
        self.assertAlmostEqual(float(rows[-2]["y_meas_v"]), 0.3 / 1.5, places=9)
        self.assertAlmostEqual(float(rows[-1]["y_meas_v"]), 0.3 / 1.5, places=9)

    def test_divides_by_g_before_subtracting_the_background(self):
        """Step 6 fits a_x/a_w where g is 1 by definition; order is not free."""
        rows = gain_rows(0, 2.0) + [onn_row(0, mean=0.5, dark=0.1, sb=0.02)]
        OI.apply_gain(rows, log=lambda _m: None)
        self.assertAlmostEqual(float(rows[-1]["y_meas_v"]),
                               (0.5 - 0.1) / 2.0 - 0.02, places=9)

    def test_residual_is_against_y_pred(self):
        rows = gain_rows(0, 1.0) + [onn_row(0, mean=0.25, y_pred=0.2)]
        OI.apply_gain(rows, log=lambda _m: None)
        self.assertAlmostEqual(float(rows[-1]["resid_v"]), 0.05, places=9)

    def test_is_idempotent(self):
        rows = gain_rows(0, 1.0) + [onn_row(50, mean=0.1)] + gain_rows(100, 1.2)
        OI.apply_gain(rows, log=lambda _m: None)
        once = rows[5]["y_meas_v"]
        OI.apply_gain(rows, log=lambda _m: None)
        self.assertEqual(rows[5]["y_meas_v"], once)

    def test_no_anchors_leaves_the_rows_alone(self):
        rows = [onn_row(0, mean=0.1)]
        self.assertEqual(OI.apply_gain(rows, log=lambda _m: None), 0)
        self.assertNotIn("y_meas_v", rows[0])


# ======================================================================
# the dark
# ======================================================================

class _FakeSLM:
    def get_slm_info(self):
        return (64, 8)

    def display_array(self, pattern):
        self.shown = pattern


class _CountingDAQ:
    """Returns a fixed level and counts the single-beam (dark) reads."""

    def __init__(self, level=0.05, dark=1e-4):
        self.level, self.dark, self.singles, self.reads = level, dark, 0, 0

    def monitor_cycle(self, timeout=None, single=False):
        self.reads += 1
        self.singles += int(single)
        value = self.dark if single else self.level + self.dark
        return SimpleNamespace(value=-value, std=1e-5)


def _encode(x_vals, w_vals, layout, width, height):
    return (x_vals.copy(), w_vals.copy())


class DarkTests(unittest.TestCase):
    """``run_patterns(dark=...)``: the one kwarg onn_infer needs from step 6's loop."""

    def setUp(self):
        self.models = {k: PairModel(index=k, eta=0.1, a_x=0.0, q_x=0.0,
                                    a_w=0.0, q_w=0.0, d=0.0) for k in PAIRS}
        self.layout = SimpleNamespace(n_channels=len(PAIRS))
        self.acq = replace(P.DEFAULT_ACQ, t_meas_s=0.0, settle_s=0.0)

    def _run(self, dark):
        slm, daq = _FakeSLM(), _CountingDAQ()
        x = np.full((1, 4, len(PAIRS)), 0.5)
        rows: list[dict] = []
        P.run_patterns(daq, slm, self.layout, [tuple(PAIRS)], x, x,
                       self.models, {k: 0.0 for k in PAIRS}, rows=rows,
                       acq=self.acq, encode=_encode, log=lambda _m: None,
                       dark=dark)
        return daq, rows

    def test_a_given_dark_is_never_measured(self):
        daq, rows = self._run(dark=-3.2e-5)
        self.assertEqual(daq.singles, 0)
        self.assertEqual(daq.reads, len(rows))

    def test_a_given_dark_is_the_one_every_row_carries(self):
        _, rows = self._run(dark=-3.2e-5)
        self.assertTrue(all(r["dark_v"] == -3.2e-5 for r in rows))

    def test_the_default_still_measures_one(self):
        """Steps 6, 7 and 8 must be untouched by the kwarg existing."""
        daq, _ = self._run(dark=None)
        self.assertEqual(daq.singles, 1)


# ======================================================================
# the Y_sat handover
# ======================================================================

class Step8HandoverTests(unittest.TestCase):
    """A step-8 result carries y_sat_v and a path; eta/phi come from the path."""

    def setUp(self):
        try:
            import onn_module.params as params       # noqa: F401
        except ImportError as exc:                   # torch lives in rbonn-ml
            raise unittest.SkipTest(f"onn_module.params needs torch ({exc})")
        self.params = params
        self.step7 = next(
            (p for p in sorted((ROOT / "src" / "calib_data").rglob(
                "calib.json"), reverse=True)), None)
        if self.step7 is None:
            raise unittest.SkipTest("no step-7 result in the tree")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def _result(self, step7_path, y_sat=2.4053784):
        import json
        path = self.dir / "calib_step8_result_test.json"
        path.write_text(json.dumps({"step7_json": str(step7_path),
                                    "csv": "nowhere.csv", "y_sat_v": y_sat}),
                        encoding="utf-8")
        return path

    def test_resolves_eta_and_phi_through_step7_json(self):
        p = self.params.ONNParams.from_json(self._result(self.step7))
        self.assertAlmostEqual(p.y_sat, 2.4053784, places=6)
        self.assertGreater(p.n_channels, 0)
        self.assertTrue(np.all(np.isfinite(p.eta.numpy())))

    def test_falls_back_to_the_step7_beside_it(self):
        """The recorded path is absolute; a moved calib_data must still load."""
        import shutil
        beside = self.dir / self.step7.name
        shutil.copy(self.step7, beside)
        moved = self.dir / "gone" / self.step7.name
        p = self.params.ONNParams.from_json(self._result(moved))
        self.assertAlmostEqual(p.y_sat, 2.4053784, places=6)

    def test_an_explicit_y_sat_overrides_the_file(self):
        p = self.params.ONNParams.from_json(self._result(self.step7), y_sat=9.0)
        self.assertAlmostEqual(p.y_sat, 9.0, places=9)

    def test_a_result_with_neither_eta_nor_a_path_is_an_error(self):
        import json
        path = self.dir / "bad.json"
        path.write_text(json.dumps({"y_sat_v": 5.0, "csv": "x"}), encoding="utf-8")
        with self.assertRaises(KeyError):
            self.params.ONNParams.from_step8(path)




# ======================================================================
# the loop, end to end
# ======================================================================

class _SteppingDAQ:
    """A bench whose gain steps upward once per anchor block.

    Not a clock-driven ramp.  The gain must be CONSTANT across the five rungs of
    one measurement, or "a pure gain change leaves no residual in the slope fit"
    stops being a statement about the code and becomes a race against how fast
    the machine gets through the reads.  So it steps when a block STARTS, which
    it recognises by the lowest anchor's drive -- ``build_anchors`` returns them
    sorted by alpha, so that frame opens every block and appears nowhere else
    (the inference drives are random).  The first block is then read at g = 1,
    and since this bench IS the model, that is also the g0 it fits.

    Dark is constant and non-zero so a dropped dark subtraction cannot pass.
    """

    def __init__(self, slm, models, phases, first_drive, *, dark=2e-4, rate=0.01):
        self.slm, self.models, self.phases = slm, models, phases
        self.first, self.dark, self.rate = np.asarray(first_drive), dark, rate
        self.blocks, self.singles = -1, 0

    @property
    def g(self) -> float:
        return 1.0 + self.rate * max(self.blocks, 0)

    def monitor_cycle(self, timeout=None, single=False):
        self.singles += int(single)
        if single:
            return SimpleNamespace(value=-self.dark, std=1e-6)
        x_vals, w_vals = self.slm.shown
        pairs = sorted(self.models)
        slots = [k - P.PAIR_INDEX_BASE for k in pairs]
        x, w = x_vals[slots], w_vals[slots]
        if np.allclose(x, self.first, atol=1e-9):
            self.blocks += 1
        y = float(V.predict(self.models, self.phases, pairs, x, w))
        return SimpleNamespace(value=-(self.g * y + self.dark), std=1e-6)


class RunLoopTests(unittest.TestCase):
    """main() against a fake bench: the wiring the unit tests cannot reach.

    Worth an integration test because this loop runs unattended for hours.  The
    anchor/chunk interleave, the single dark threaded through every call, the
    negative seq counter and the interpolation pass are all wiring, and wiring
    fails silently.

    The run happens ONCE for the class and has to occupy real wall clock: ``t_s``
    is stamped to 0.1 s, so a run that finishes inside one tick gives every read
    the same timestamp and there is nothing for the interpolation to do.  Hence
    the 0.05 s settle -- the only knob here that costs the suite anything.
    """

    @classmethod
    def setUpClass(cls):
        found = sorted((ROOT / "src" / "calib_data").rglob(
            "calib.json"), reverse=True)
        cls.step7 = next((p for p in found if _has_step3(p)), None)
        if cls.step7 is None:
            raise unittest.SkipTest("no step-7 result with an embedded step 3")
        cls.models, cls.phases, _ = V.load_forward_model(cls.step7)
        cls.pairs = sorted(cls.models)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.dir = Path(cls.tmp.name)
        cls.rows = cls._run()

    @classmethod
    def _npz(cls, n=2, tiles=3, outs=2) -> Path:
        k = len(cls.pairs)
        rng = np.random.default_rng(0)
        path = cls.dir / "ref.npz"
        np.savez(path,
                 idx=np.arange(n), labels=np.zeros(n, dtype=np.int64),
                 x=rng.uniform(0.1, 0.9, (n, tiles * k)),
                 W=rng.uniform(0.1, 0.9, (outs, tiles * k)),
                 y_block=np.zeros((n, outs, tiles)),
                 logits_ref=np.zeros((n, outs)),
                 channels=np.asarray(cls.pairs, dtype=np.int64),
                 y_sat=np.float64(np.inf), n_tiles=np.int64(tiles),
                 n_channels=np.int64(k), n_out=np.int64(outs),
                 n_banks=np.int64(1), step7_json=str(cls.step7),
                 meta=json.dumps({"checkpoint": "none"}))
        return path

    SEED = 5

    @classmethod
    def _run(cls) -> list[dict]:
        slm = _FakeSLM()
        # The same ladder main() will build: its lowest rung opens every block.
        first = OI.build_anchors(cls.models, cls.phases, cls.pairs, seed=cls.SEED)
        cls.daq = _SteppingDAQ(slm, cls.models, cls.phases, first["x"][0])
        layout = SimpleNamespace(n_channels=len(cls.pairs))
        argv = ["onn_infer", "--npz", str(cls._npz()), "--calib", str(cls.step7),
                "--gain-seed", str(cls.SEED), "--no-pmt", "--t-meas", "0.0",
                "--gain-every", "0.4", "--out", str(cls.dir)]
        patches = [
            mock.patch.object(OI, "connect_slm", return_value=slm),
            mock.patch.object(OI, "connect_daq", return_value=cls.daq),
            mock.patch.object(OI, "ACQ", replace(OI.ACQ, settle_s=0.05)),
            mock.patch.object(OI, "load_layout", return_value=layout),
            mock.patch("slm_module.encoding.encode_to_pattern", _encode),
            mock.patch.object(sys, "argv", argv),
        ]
        with contextlib.ExitStack() as stack:
            for q in patches:
                stack.enter_context(q)
            OI.main()
        path = next(cls.dir.glob("onn_infer_*.csv"))
        text = [ln for ln in path.read_text(encoding="utf-8").splitlines()
                if not ln.startswith("#")]
        return list(csv.DictReader(text))

    def _of(self, kind):
        return [r for r in self.rows if r["kind"] == kind]

    def test_the_dark_is_read_exactly_once_for_the_whole_run(self):
        self.assertEqual(self.daq.singles, 1)

    def test_every_row_carries_that_one_dark(self):
        self.assertEqual(len({r["dark_v"] for r in self.rows}), 1)

    def test_both_kinds_of_read_reach_the_csv(self):
        self.assertEqual(len(self._of("onn")), 2 * 3 * 2)
        self.assertGreaterEqual(len(self._of("gain")), 2 * OI.GAIN_RUNGS)   # >= 2 blocks

    def test_a_bench_that_only_drifts_drops_no_rung(self):
        self.assertEqual(self._of("gain_drop"), [])

    def test_anchor_rows_never_collide_with_inference_seq(self):
        gain = [int(r["seq"]) for r in self._of("gain")]
        onn = {int(r["seq"]) for r in self._of("onn")}
        self.assertTrue(all(s < 0 for s in gain))
        self.assertFalse(set(gain) & onn)
        self.assertEqual(len(set(gain)), len(gain))          # all distinct

    def test_g_tracks_the_drifting_bench(self):
        """The bench is the model at g = 1, so g0 = 1; it steps 1 % per block."""
        g = [float(r["g"]) for r in self._of("gain")]
        self.assertEqual(len(set(g[:OI.GAIN_RUNGS])), 1)
        self.assertAlmostEqual(g[0], 1.0, places=9)
        self.assertAlmostEqual(g[OI.GAIN_RUNGS], 1.0 + self.daq.rate, places=6)
        self.assertGreater(g[-1], g[0])

    def test_the_csv_meta_records_the_reference(self):
        """Seed, band, g0 and the first block's volts: enough to rebuild it."""
        path = next(self.dir.glob("onn_infer_*.csv"))
        meta = dict(ln[2:].split(",", 1) for ln in
                    path.read_text(encoding="utf-8").splitlines()
                    if ln.startswith("# "))
        self.assertEqual(meta["gain_seed"], str(self.SEED))
        self.assertEqual(meta["g_reference"],
                         "first gain block of this run, fitted to the model (g0)")
        self.assertAlmostEqual(float(meta["g0"]), 1.0, places=9)
        self.assertEqual(len(meta["g_reference_v"].split()), OI.GAIN_RUNGS)
        self.assertEqual(meta["gain_tol"], f"{OI.GAIN_TOL:g}")
        self.assertEqual(meta["gain_rungs_out"], "")
        self.assertTrue(meta["calib"].endswith("calib.json"))

    def test_a_pure_gain_change_leaves_no_anchor_residual(self):
        """rms is the "something other than gain moved" detector; nothing did."""
        for r in self._of("gain"):
            self.assertLess(float(r["g_rms_v"]), 1e-9)

    def test_every_inference_read_is_corrected(self):
        self.assertTrue(all(r["y_meas_v"] not in ("", None)
                            for r in self._of("onn")))

    def test_inference_reads_get_interpolated_gains_not_held_ones(self):
        """A held g would only ever repeat the anchors' own values."""
        anchors = {float(r["g"]) for r in self._of("gain")}
        used = {float(r["g"]) for r in self._of("onn")}
        self.assertGreater(len(used), len(anchors))
        self.assertTrue(used - anchors)

    def test_the_correction_recovers_the_model_s_own_y(self):
        """The whole point: undo the gain and Y comes back.

        The fake bench's only departure from the model is the gain the anchors
        measured, so every corrected read must land on its own prediction.
        ``pred_v`` is the FULL prediction and ``y_meas_v`` has had the
        single-beam background taken off, so the coherent part is what it must
        match.  The tolerance is what one interpolation interval of a step
        change leaves behind, not a measurement claim.
        """
        onn = self._of("onn")
        coh = [float(r["pred_v"]) - float(r["sb_pred_v"]) for r in onn]
        rel = [abs(float(r["y_meas_v"]) - c) / c for r, c in zip(onn, coh)]
        self.assertLess(max(rel), 0.02)


if __name__ == "__main__":
    unittest.main()
