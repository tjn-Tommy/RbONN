"""Geometry, fractional-guard and learned-encoding tests for ``bit_depth``.

This file also used to hold a ``RefactorParityTest`` comparing the refactored
package against ``fixtures/bit_depth_golden.npz``. That .npz was never
committed, and neither was the ``capture_golden.py`` that wrote it, so the
module failed to import and NOT ONE test here had ever run -- the suite's
standing "1 error, test_bit_depth" was this. The golden cannot be rebuilt: it
captured the pre-refactor code, which is gone, so regenerating it from today's
package would only pin today's behaviour to itself. The parity tests are
therefore dropped rather than skipped; what is left below needs no fixture.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bit_depth as bd  # noqa: E402
from bit_depth import nn_encoder  # noqa: E402

class ActiveModelTest(unittest.TestCase):
    """The one check from the old parity block that needed no golden file."""

    def setUp(self):
        self.c = bd.Cfg()

    def test_validate_active_model(self):
        bd.validate_active_model(self.c)  # raises on failure


class FractionalGuardTest(unittest.TestCase):
    """2.5-px guard (15-px window on 20-px pitch) and integer back-compat."""

    def setUp(self):
        self.c = bd.Cfg()

    def test_geometry_sample_counts(self):
        self.assertEqual(bd.channel_group_px(15, 2.5), 20)
        self.assertEqual(bd.group_samples(self.c, 15, 2.5), 800)
        self.assertEqual(bd.guard_samples(self.c, 2.5), 100)
        self.assertEqual(bd.active_slice(self.c, 15, 2.5), slice(100, 700))

    def test_integer_guard_matches_repeat(self):
        amp = np.array([0.0, 0.4, 1.0])
        for enc in bd.ENCODING_ORDER:
            per_pixel = bd.phase_from_targets(self.c, amp, px_per_ch=5, guard=2, encoding=enc)
            grid = bd.phase_grid_from_targets(self.c, amp, px_per_ch=5, guard=2, encoding=enc)
            np.testing.assert_allclose(grid, np.repeat(per_pixel, self.c.ovs), atol=1e-12)

    def test_fractional_guard_runs(self):
        amp = np.array([0.2, 0.6, 0.9])
        grid = bd.phase_grid_from_targets(self.c, amp, px_per_ch=15, guard=2.5, encoding="flat")
        self.assertEqual(grid.size, 3 * 800)
        # guard bands stay blank (pi), active window carries the command
        self.assertTrue(np.all(grid[:100] == np.pi))
        amp_out = bd.amplitude_from_targets(self.c, amp, px_per_ch=15, guard=2.5, encoding="flat")
        self.assertEqual(amp_out.size, 3 * 800)
        row = bd.monte_carlo_geometry(
            self.c, px_per_ch=15, guard=2.5, n_ch=5, n_trials=2, seed=0,
            correction="lut", encoding="flat",
        )
        self.assertEqual(row["group_px"], 20)
        self.assertTrue(np.isfinite(row["cal_rmse"]))


def _write_synthetic_checkpoint(path, px_per_ch=15):
    """A tiny deterministic MLP checkpoint nn_encoder can load (no torch)."""
    rng = np.random.default_rng(0)
    W0 = rng.normal(0, 0.5, (1, 6))
    b0 = rng.normal(0, 0.1, (6,))
    W1 = rng.normal(0, 0.5, (6, px_per_ch))
    b1 = rng.normal(0, 0.1, (px_per_ch,))
    table_a = np.linspace(0.0, 1.0, 64)

    def forward(a):
        h = np.tanh(a.reshape(-1, 1) @ W0 + b0)
        z = h @ W1 + b1
        return np.clip(1.0 / (1.0 + np.exp(-z)), 0.0, 1.0)

    table_profile = forward(table_a)
    np.savez(
        path, px_per_ch=px_per_ch, symmetric=False, activation="tanh",
        n_layers=2, W0=W0, b0=b0, W1=W1, b1=b1,
        table_a=table_a, table_profile=table_profile,
    )


class NNEncoderTest(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = Path(tempfile.mkdtemp()) / "nn_encoder.npz"
        _write_synthetic_checkpoint(self.tmp)
        self._saved_default = nn_encoder.DEFAULT_CHECKPOINT
        nn_encoder.DEFAULT_CHECKPOINT = self.tmp
        nn_encoder.clear_cache()
        self.c = bd.Cfg()

    def tearDown(self):
        nn_encoder.DEFAULT_CHECKPOINT = self._saved_default
        nn_encoder.clear_cache()

    def test_table_matches_mlp_forward(self):
        a = np.linspace(0, 1, 64)
        np.testing.assert_allclose(
            nn_encoder.nn_profile(a, self.tmp), nn_encoder.mlp_forward(a, self.tmp), atol=1e-9
        )

    def test_profile_shape_and_range(self):
        prof = nn_encoder.nn_profile_single(0.5, 15)
        self.assertEqual(prof.shape, (15,))
        self.assertTrue(np.all((prof >= 0) & (prof <= 1)))
        with self.assertRaises(ValueError):
            nn_encoder.nn_profile_single(0.5, 5)  # wrong window size

    def test_nn_encoding_end_to_end(self):
        row = bd.monte_carlo_geometry(
            self.c, px_per_ch=15, guard=2.5, n_ch=5, n_trials=2, seed=0,
            correction="lut", encoding="nn",
        )
        self.assertEqual(row["encoding"], "nn")
        self.assertTrue(np.isfinite(row["cal_rmse"]))


class CalibrationDataTest(unittest.TestCase):
    def test_the_operating_geometry_is_15_on_20(self):
        geom = bd.operating_geometry()
        self.assertEqual(geom["px_per_ch"], 15)
        self.assertEqual(geom["guard"], 2.5)
        self.assertEqual(geom["group_px"], 20)

    def test_nm_per_px_comes_off_the_measured_calibration(self):
        """Needs ``calibration.csv`` at the repo root, which is not committed."""
        from bit_depth.calibration_data import CALIBRATION_CSV

        if not CALIBRATION_CSV.is_file():
            self.skipTest(f"{CALIBRATION_CSV} not in the tree")
        nm = bd.measured_nm_per_px()
        self.assertTrue(0.001 < nm < 0.02)  # ~0.0057 nm/px
        self.assertEqual(bd.cfg_from_calibration().group, 20)


if __name__ == "__main__":
    unittest.main()
