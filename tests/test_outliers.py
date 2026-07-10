from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slm_module.calibration.calibration import intensity_model
from slm_module.calibration.outliers import (
    OutlierRemeasurePolicy,
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


if __name__ == "__main__":
    unittest.main()
