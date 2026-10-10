from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from calibration_module.fit.legacy_intensity import intensity_model
from calibration_module.fit.outliers import (
    BURST_K,
    BURST_K_REPEAT,
    OutlierRemeasurePolicy,
    flag_burst_repeats,
    flag_burst_traces,
    flag_by_residual,
    linear_fit_residuals,
    mad_sigma,
    transfer_fit_residuals,
)


class PolicyTests(unittest.TestCase):
    def test_defaults(self) -> None:
        policy = OutlierRemeasurePolicy()
        self.assertEqual(policy.k_sigma, 4.0)
        self.assertEqual(policy.max_retries, 3)
        self.assertEqual(policy.min_points, 8)

    def test_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "k_sigma"):
            OutlierRemeasurePolicy(k_sigma=0.0)
        with self.assertRaisesRegex(ValueError, "max_retries"):
            OutlierRemeasurePolicy(max_retries=0)
        with self.assertRaisesRegex(ValueError, "min_points"):
            OutlierRemeasurePolicy(min_points=2)


class MadSigmaTests(unittest.TestCase):
    def test_known_spread(self) -> None:
        # residuals -1, 0, 1 -> MAD = 1 -> sigma = 1.4826
        self.assertAlmostEqual(mad_sigma(np.array([-1.0, 0.0, 1.0])), 1.4826)

    def test_ignores_non_finite_and_floors(self) -> None:
        self.assertAlmostEqual(
            mad_sigma(np.array([np.nan, -1.0, 0.0, 1.0, np.inf])), 1.4826
        )
        self.assertGreater(mad_sigma(np.array([np.nan, np.nan])), 0.0)
        self.assertGreater(mad_sigma(np.zeros(5)), 0.0)


class FlagByResidualTests(unittest.TestCase):
    def test_flags_outlier_and_nan_only(self) -> None:
        r = np.array([0.1, -0.2, 0.05, 50.0, np.nan, -0.1])
        flags = flag_by_residual(r, k_sigma=4.0)
        self.assertEqual(flags.tolist(), [False, False, False, True, True, False])

    def test_rel_floor_suppresses_numerical_noise(self) -> None:
        # numerically clean fit: residuals are fit noise around 1e-13 whose MAD
        # collapses -- without the floor everything would be flagged
        rng = np.random.default_rng(7)
        r = rng.normal(0.0, 1e-13, 50)
        r[3] = 5e-13  # >4 sigma of the noise, still physically nothing
        flags = flag_by_residual(r, k_sigma=4.0, rel_floor_scale=770.0)
        self.assertFalse(flags.any())

    def test_explicit_sigma_wins(self) -> None:
        r = np.array([0.0, 0.0, 3.0])
        self.assertTrue(flag_by_residual(r, k_sigma=2.0, sigma=1.0)[2])
        self.assertFalse(flag_by_residual(r, k_sigma=4.0, sigma=1.0)[2])


class LinearFitResidualTests(unittest.TestCase):
    def test_spike_dominates_residuals(self) -> None:
        x = np.arange(20.0)
        y = 3.0 * x + 7.0
        y[11] += 25.0
        r = linear_fit_residuals(x, y)
        flags = flag_by_residual(r, k_sigma=4.0, rel_floor_scale=np.abs(y).max())
        self.assertEqual(np.flatnonzero(flags).tolist(), [11])

    def test_nan_y_gets_nan_residual(self) -> None:
        x = np.arange(10.0)
        y = 2.0 * x
        y[4] = np.nan
        r = linear_fit_residuals(x, y)
        self.assertTrue(np.isnan(r[4]))
        self.assertTrue(np.all(np.isfinite(np.delete(r, 4))))

    def test_too_few_points_all_nan(self) -> None:
        r = linear_fit_residuals(np.array([1.0]), np.array([2.0]))
        self.assertTrue(np.isnan(r).all())


class TransferFitResidualTests(unittest.TestCase):
    def test_sin2_spike_flagged(self) -> None:
        levels = np.linspace(0.0, 1000.0, 11)
        values = intensity_model(levels, 1.0, np.pi / 1000.0, 0.0)
        values[5] += 0.4
        r = transfer_fit_residuals(levels, values)
        flags = flag_by_residual(r, k_sigma=4.0, rel_floor_scale=np.abs(values).max())
        self.assertIn(5, np.flatnonzero(flags).tolist())
        self.assertLessEqual(flags.sum(), 2)   # spike, not the whole curve

    def test_clean_sin2_curve_not_flagged(self) -> None:
        levels = np.linspace(0.0, 1000.0, 11)
        values = intensity_model(levels, 0.8, np.pi / 900.0, 0.3)
        r = transfer_fit_residuals(levels, values)
        flags = flag_by_residual(r, k_sigma=4.0, rel_floor_scale=np.abs(values).max())
        self.assertFalse(flags.any())

    def test_degenerate_data_falls_back_without_raising(self) -> None:
        levels = np.linspace(0.0, 1000.0, 11)
        r = transfer_fit_residuals(levels, np.zeros(11))   # sin^2 fit degenerate
        self.assertTrue(np.all(np.isfinite(r)))

    def test_nan_value_gets_nan_residual(self) -> None:
        levels = np.linspace(0.0, 1000.0, 11)
        values = intensity_model(levels, 1.0, np.pi / 1000.0, 0.0)
        values[2] = np.nan
        r = transfer_fit_residuals(levels, values)
        self.assertTrue(np.isnan(r[2]))


class BurstTraceTests(unittest.TestCase):
    """The sweep-median detector (step 7: one acquisition per point)."""

    def test_flags_the_burst_and_nothing_else(self) -> None:
        # a step-7 ramp's trace stds, with run_0929's tgt-10 burst in it
        s = np.array([1.10, 1.43, 1.67, 363.8, 1.74, 1.43, 1.59, 1.75, 2.39, 1.31])
        mask = flag_burst_traces(s)
        self.assertEqual(list(np.flatnonzero(mask)), [3])

    def test_clean_sweep_flags_nothing(self) -> None:
        # the honest full-scale point runs ~3.7x the sweep median; must survive
        s = np.array([0.28, 0.35, 0.52, 1.022, 0.31, 0.29, 0.33, 0.40])
        self.assertFalse(flag_burst_traces(s).any())

    def test_non_finite_always_flagged(self) -> None:
        s = np.array([1.0, 1.1, np.nan, 0.9, 1.05, np.inf])
        mask = flag_burst_traces(s)
        self.assertTrue(mask[2])
        self.assertTrue(mask[5])

    def test_too_few_points_flags_nothing(self) -> None:
        # under min_points the median is too soft to threshold against
        self.assertFalse(flag_burst_traces(np.array([1.0, 500.0, 1.1])).any())

    def test_all_zero_scale_flags_nothing(self) -> None:
        self.assertFalse(flag_burst_traces(np.zeros(6)).any())

    def test_k_must_exceed_one(self) -> None:
        with self.assertRaisesRegex(ValueError, "k must be"):
            flag_burst_traces(np.ones(6), k=1.0)

    def test_rejects_2d(self) -> None:
        with self.assertRaisesRegex(ValueError, "1-D"):
            flag_burst_traces(np.ones((2, 3)))


class BurstRepeatTests(unittest.TestCase):
    """The within-cell detector (step 6: every level measured several times)."""

    def test_flags_the_burst_repeat_only(self) -> None:
        # run_0929 pair 10, x=1 w=0.9: one repeat at 111 mV against five at ~0.31
        s = np.array([111.02, 0.312, 0.304, 0.313, 0.319, 0.278])
        ids = [("1", "0.9")] * 6
        self.assertEqual(list(np.flatnonzero(flag_burst_repeats(s, ids))), [0])

    def test_catches_a_burst_that_left_the_mean_alone(self) -> None:
        # run_0906 pair 2, x=1 w=0.7: the trace std jumped 5x while all four
        # means agreed to 1%, so nothing reading the mean could have seen it
        s = np.array([0.813, 3.9755, 0.754, 0.720])
        ids = [("1", "0.7")] * 4
        self.assertEqual(list(np.flatnonzero(flag_burst_repeats(s, ids))), [1])

    def test_dynamic_range_across_cells_is_not_flagged(self) -> None:
        # A step-6 grid is mostly dim levels, so its median trace std sits near
        # the dark end and the honest full-scale point runs >10x it.  This is the
        # false positive the sweep ruler actually produced on run_0911 pair 8
        # (0.6031 mV against a 0.05 mV sweep median); the per-cell reference sees
        # the same two rows agree to 1.15x and leaves them alone.
        dim = [0.040, 0.041, 0.039, 0.042, 0.050, 0.051, 0.049, 0.052,
               0.103, 0.121]
        s = np.array(dim + [0.6031, 0.5179])
        ids = [(str(i), "dim") for i in range(len(dim))] + [("1", "1")] * 2
        self.assertTrue(flag_burst_traces(s)[-2:].any())   # the blunter ruler
        self.assertFalse(flag_burst_repeats(s, ids).any())

    def test_two_repeat_cell_flags_the_louder_one(self) -> None:
        # run_0928 pair 3: 66.8 mV against 0.045 mV, only two repeats
        s = np.array([66.777, 0.045])
        ids = [("0.5", "0")] * 2
        self.assertEqual(list(np.flatnonzero(flag_burst_repeats(s, ids))), [0])

    def test_two_repeat_cell_tolerates_honest_scatter(self) -> None:
        # the worst clean 2-repeat cell on disk sits at 1.75x
        s = np.array([0.0371, 0.0649])
        self.assertFalse(flag_burst_repeats(s, [("0", "0.5")] * 2).any())

    def test_single_acquisition_cell_is_never_flagged(self) -> None:
        # it is its own minimum: nothing in the data can say it was disturbed
        s = np.array([500.0, 1.0, 1.1, 1.05])
        ids = [("1", "1"), ("0", "0"), ("0", "0"), ("0", "0")]
        self.assertFalse(flag_burst_repeats(s, ids)[0])

    def test_non_finite_always_flagged(self) -> None:
        s = np.array([1.0, np.nan, 1.1])
        self.assertTrue(flag_burst_repeats(s, [("1", "1")] * 3)[1])

    def test_mismatched_lengths_raise(self) -> None:
        with self.assertRaisesRegex(ValueError, "1-D"):
            flag_burst_repeats(np.ones(3), [("1", "1")] * 2)

    def test_k_must_exceed_one(self) -> None:
        with self.assertRaisesRegex(ValueError, "k must be"):
            flag_burst_repeats(np.ones(3), [("1", "1")] * 3, k=0.5)

    def test_thresholds_are_the_measured_ones(self) -> None:
        # clean populations top out at 3.7x (sweep) and 1.81x (repeats); the
        # mildest real bursts are 80x and 5.1x.  Guard the gap, not the value.
        self.assertGreater(BURST_K, 3.7)
        self.assertLess(BURST_K, 80.0)
        self.assertGreater(BURST_K_REPEAT, 1.81)
        self.assertLess(BURST_K_REPEAT, 5.1)


if __name__ == "__main__":
    unittest.main()
