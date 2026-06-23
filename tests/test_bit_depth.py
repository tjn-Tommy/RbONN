"""Parity, geometry, and learned-encoding tests for the bit_depth package.

The parity tests assert the refactored package reproduces the pre-refactor
behavior captured in ``fixtures/bit_depth_golden.npz`` (see
``scratchpad/capture_golden.py``). The fractional-guard and NN tests cover the
new functionality.
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

GOLDEN = np.load(Path(__file__).resolve().parent / "fixtures" / "bit_depth_golden.npz")
MC_KEYS = [str(k) for k in GOLDEN["mc_keys"]]


class RefactorParityTest(unittest.TestCase):
    """Refactor must not change any numeric behavior."""

    def setUp(self):
        self.c = bd.Cfg()
        self.amp = np.array([0.0, 0.1, 0.37, 0.5, 0.83, 1.0])

    def test_phase_from_targets_parity(self):
        for enc in bd.ENCODING_ORDER:
            phase = bd.phase_from_targets(self.c, self.amp, px_per_ch=5, guard=2, encoding=enc)
            np.testing.assert_allclose(phase, GOLDEN[f"phase_{enc}"], atol=1e-12)

    def test_amplitude_from_targets_parity(self):
        amp = bd.amplitude_from_targets(self.c, self.amp, px_per_ch=5, guard=1, encoding="flat")
        np.testing.assert_allclose(amp, GOLDEN["amp_flat_g1"], atol=1e-12)

    def test_readout_calibration_parity(self):
        blank, response = bd.readout_calibration(self.c, n_ch=7, px_per_ch=5, guard=1)
        np.testing.assert_allclose(blank, GOLDEN["cal_blank"], atol=1e-12)
        np.testing.assert_allclose(response, GOLDEN["cal_response"], atol=1e-12)

    def test_transfer_curves_parity(self):
        t, r = bd.single_channel_transfer_curve(self.c, px_per_ch=5, guard=1, n_ch=7, n_points=21)
        np.testing.assert_allclose(t, GOLDEN["transfer_t"], atol=1e-12)
        np.testing.assert_allclose(r, GOLDEN["transfer_r"], atol=1e-12)
        tt, cmd, rec = bd.lut_corrected_transfer_curve(
            self.c, px_per_ch=5, guard=1, n_ch=7, n_points=21, correction="lut"
        )
        np.testing.assert_allclose(tt, GOLDEN["lutc_targets"], atol=1e-12)
        np.testing.assert_allclose(cmd, GOLDEN["lutc_command"], atol=1e-12)
        np.testing.assert_allclose(rec, GOLDEN["lutc_recovered"], atol=1e-12)

    def test_monte_carlo_geometry_parity(self):
        for enc in bd.ENCODING_ORDER:
            for corr in ("none", "lut"):
                row = bd.monte_carlo_geometry(
                    self.c, px_per_ch=5, guard=1, n_ch=7, n_trials=8, seed=0,
                    correction=corr, encoding=enc,
                )
                got = np.array([row[k] for k in MC_KEYS], dtype=float)
                np.testing.assert_allclose(
                    got, GOLDEN[f"mc_{enc}_{corr}"], atol=1e-10, equal_nan=True,
                    err_msg=f"mismatch for {enc}/{corr}",
                )

    def test_weights_and_power_parity(self):
        for enc in bd.ENCODING_ORDER:
            np.testing.assert_allclose(
                bd.edge_taper_weights(5, enc), GOLDEN[f"weights_{enc}"], atol=1e-12
            )
            self.assertAlmostEqual(
                bd.taper_active_power_factor(5, enc), float(GOLDEN[f"tapf_{enc}"]), places=12
            )

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
    def test_nm_per_px_and_geometry(self):
        nm = bd.measured_nm_per_px()
        self.assertTrue(0.001 < nm < 0.02)  # ~0.0057 nm/px
        cfg = bd.cfg_from_calibration()
        self.assertEqual(cfg.group, 20)
        geom = bd.operating_geometry()
        self.assertEqual(geom["px_per_ch"], 15)
        self.assertEqual(geom["guard"], 2.5)


if __name__ == "__main__":
    unittest.main()
