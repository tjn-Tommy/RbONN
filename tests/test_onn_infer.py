"""The inference run's instrument correction: the anchors, g, and the dark.

Three things are load-bearing here and none of them is the optics.

``read_step8_anchors`` must return the drives step 8 actually displayed, not a
recomputation of them -- the whole point of taking a ratio on an identical SLM
frame is that no model enters it, and a re-solved drive is a different frame.

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
# a step-8 CSV, without a step 8
# ======================================================================

def write_step8_csv(path: Path, alphas, *, dark=-7.5e-5, kind="A") -> Path:
    """A minimal step-8 CSV: one ladder rung per alpha, drives that differ.

    The drive is a deterministic function of alpha so a test can assert that
    what came back is the row it asked for rather than a neighbour.
    """
    cols = (["kind", "alpha"]
            + [f"x_{k}" for k in PAIRS] + [f"w_{k}" for k in PAIRS]
            + ["voltage_mean_v", "dark_v"])
    with open(path, "w", newline="", encoding="utf-8") as fh:
        fh.write("# step7_json,nowhere.json\n")
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        for a in alphas:
            row = {"kind": kind, "alpha": f"{a:.6f}",
                   "voltage_mean_v": f"{0.4 * a + dark:.9g}", "dark_v": f"{dark:.9g}"}
            for j, k in enumerate(PAIRS):
                row[f"x_{k}"] = f"{a * (j + 1) / 10:.6f}"
                row[f"w_{k}"] = f"{a * (j + 2) / 10:.6f}"
            wr.writerow(row)
    return path


class AnchorTests(unittest.TestCase):
    """Which rungs get picked, and whether their drives survive the trip."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def test_picks_the_nearest_rung_to_each_target(self):
        # Ladder topping at exactly 1, so a fraction of the top IS that alpha.
        csv_path = write_step8_csv(self.dir / "s8.csv", np.arange(0.05, 1.001, 0.05))
        got = OI.read_step8_anchors(csv_path, PAIRS, (0.55, 0.75, 0.95))
        np.testing.assert_allclose(got["alpha"], [0.55, 0.75, 0.95], atol=1e-6)
        self.assertAlmostEqual(got["top"], 1.0)

    def test_the_drive_is_the_row_asked_for_not_a_neighbour(self):
        """Verbatim, per column -- a ratio on a re-solved frame measures nothing."""
        csv_path = write_step8_csv(self.dir / "s8.csv", np.arange(0.05, 1.001, 0.05))
        got = OI.read_step8_anchors(csv_path, PAIRS, (0.85,))
        np.testing.assert_allclose(got["x"][0], [0.085, 0.170, 0.255], atol=1e-6)
        np.testing.assert_allclose(got["w"][0], [0.170, 0.255, 0.340], atol=1e-6)

    def test_the_targets_follow_a_capped_ladder_down(self):
        """Step 8 caps itself in VOLTS, so its top alpha moves with the gain.

        A ladder capped at 0.293 -- the present bench, 450 mV of a 1534 mV Y_max
        -- has nothing at all above 0.3.  The targets have to come down with it.
        """
        alphas = np.linspace(0.15, 0.293, 30)
        csv_path = write_step8_csv(self.dir / "s8.csv", alphas)
        got = OI.read_step8_anchors(csv_path, PAIRS)
        self.assertAlmostEqual(got["top"], 0.293, places=6)
        self.assertEqual(got["n"], len(OI.ANCHOR_FRACTIONS))
        self.assertEqual(len(set(got["alpha"].tolist())), len(got["alpha"]))
        for frac, picked in zip(OI.ANCHOR_FRACTIONS, got["alpha"]):
            self.assertAlmostEqual(picked, frac * 0.293, delta=0.01)

    def test_a_capped_ladder_still_gives_a_slope_and_not_a_mean(self):
        """The regression the fractions exist for.

        With the old absolute targets (0.55..0.95) every one of them would land
        above a 0.293 ladder, the nearest-rung picker would hand back the five
        brightest rungs whatever they were, and the lever arm would collapse to
        whatever the top of the ladder happened to span -- here 1.04x.  A slope
        fitted over that is a mean with extra steps.
        """
        alphas = np.linspace(0.15, 0.293, 30)
        csv_path = write_step8_csv(self.dir / "s8.csv", alphas)

        got = OI.read_step8_anchors(csv_path, PAIRS)
        lever = got["v_ref"].max() / got["v_ref"].min()
        self.assertGreater(lever, 1.5)

        old = OI.read_step8_anchors(csv_path, PAIRS,
                                    [a / 0.293 for a in (0.55, 0.65, 0.75,
                                                         0.85, 0.95)])
        old_lever = old["v_ref"].max() / old["v_ref"].min()
        self.assertLess(old_lever, 1.1)
        self.assertGreater(lever, 1.5 * old_lever)

    def test_reference_volts_are_dark_corrected(self):
        csv_path = write_step8_csv(self.dir / "s8.csv", [0.75], dark=-1e-3)
        got = OI.read_step8_anchors(csv_path, PAIRS, (0.75,))
        self.assertAlmostEqual(float(got["v_ref"][0]), 0.4 * 0.75, places=9)

    def test_a_sparse_ladder_gives_distinct_rungs_not_one_counted_five_times(self):
        """Otherwise the slope quietly collapses back into a mean."""
        csv_path = write_step8_csv(self.dir / "s8.csv", [0.60, 0.62, 0.64])
        got = OI.read_step8_anchors(csv_path, PAIRS, (0.55, 0.65, 0.75, 0.85, 0.95))
        self.assertEqual(got["n"], 3)
        self.assertEqual(len(set(got["alpha"].tolist())), 3)

    def test_rejects_a_csv_missing_a_pair_this_run_drives(self):
        csv_path = write_step8_csv(self.dir / "s8.csv", [0.75])
        with self.assertRaises(SystemExit):
            OI.read_step8_anchors(csv_path, [1, 2, 3, 9], (0.75,))

    def test_rejects_a_csv_with_no_alpha_column(self):
        path = self.dir / "plain.csv"
        path.write_text("block,voltage_mean_v\n0,0.1\n", encoding="utf-8")
        with self.assertRaises(SystemExit):
            OI.read_step8_anchors(path, PAIRS)


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
    the machine gets through five reads.  So it steps when a block STARTS, which
    it recognises by the lowest anchor's drive -- ``read_step8_anchors`` returns
    them sorted by alpha, so that frame opens every block and appears nowhere
    else (the inference drives are random).

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

    @classmethod
    def _step8(cls) -> Path:
        """Anchors whose reference volts ARE the model's linear prediction.

        So a correctly measured g is exactly the fake bench's gain, and the test
        can assert on the number rather than only on its presence.
        """
        path = cls.dir / "s8.csv"
        cols = (["kind", "alpha"] + [f"x_{k}" for k in cls.pairs]
                + [f"w_{k}" for k in cls.pairs] + ["voltage_mean_v", "dark_v"])
        with open(path, "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=cols)
            wr.writeheader()
            for a in (0.55, 0.65, 0.75, 0.85, 0.95):
                drive = np.full(len(cls.pairs), a)
                y = float(V.predict(cls.models, cls.phases, cls.pairs, drive, drive))
                row = {"kind": "A", "alpha": f"{a:.6f}",
                       "voltage_mean_v": f"{y:.9g}", "dark_v": "0"}
                for k, d in zip(cls.pairs, drive):
                    row[f"x_{k}"] = row[f"w_{k}"] = f"{d:.6f}"
                wr.writerow(row)
        return path

    @classmethod
    def _run(cls) -> list[dict]:
        slm = _FakeSLM()
        cls.daq = _SteppingDAQ(slm, cls.models, cls.phases,
                               np.full(len(cls.pairs), 0.55))
        layout = SimpleNamespace(n_channels=len(cls.pairs))
        argv = ["onn_infer", "--npz", str(cls._npz()), "--step7", str(cls.step7),
                "--step8-csv", str(cls._step8()), "--t-meas", "0.0",
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
        self.assertGreaterEqual(len(self._of("gain")), 10)   # >= 2 measurements

    def test_anchor_rows_never_collide_with_inference_seq(self):
        gain = [int(r["seq"]) for r in self._of("gain")]
        onn = {int(r["seq"]) for r in self._of("onn")}
        self.assertTrue(all(s < 0 for s in gain))
        self.assertFalse(set(gain) & onn)
        self.assertEqual(len(set(gain)), len(gain))          # all distinct

    def test_g_tracks_the_drifting_bench(self):
        g = [float(r["g"]) for r in self._of("gain")]
        self.assertAlmostEqual(g[0], 1.0, places=6)
        self.assertGreater(g[-1], g[0])

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
