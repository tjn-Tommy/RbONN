"""The torch ONN layer against ``fit.verify.predict``, the numpy original.

``calibration_module.fit.verify.predict`` is the project's one forward model and
every calibration fit is taken under it.  ``onn_module.layer.ONNLayer`` evaluates
the same equations in torch so they can be differentiated.  That is a real
duplication, and this file is the only thing standing between it and two models
that quietly disagree: if these tests fail, the torch layer is wrong, because
``verify.predict`` is what the bench is calibrated against.

The comparison is made with the single-beam background and the dark switched
OFF -- ``PairModel`` with ``a_x = q_x = a_w = q_w = d = 0`` and
``coherent=True`` -- because those terms sit outside the coherent sum and the
layer deliberately does not model them (they are subtracted on the bench before
Y is read).  What is compared is |E|^2 through the saturation, which is the whole
of what the layer claims to compute.

Runs in the ``rbonn-ml`` env; skipped where torch is absent, which includes the
``rbonn`` hardware env.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from calibration_module.fit.phase import PairModel, phi_half  # noqa: E402
from calibration_module.fit.verify import (  # noqa: E402
    load_forward_model,
    predict,
    saturate,
)

try:
    import torch

    from onn_module.layer import ONNLayer  # noqa: E402
    from onn_module.params import ONNParams  # noqa: E402

    HAVE_TORCH = True
except ImportError:                                   # the rbonn hardware env
    HAVE_TORCH = False

#: A step-7 run with all ten pairs calibrated end to end.
STEP7 = REPO / "src" / "calib_data" / "calib_0917_1617" / "calib.json"

needs_torch = unittest.skipUnless(HAVE_TORCH, "torch not installed (rbonn-ml env only)")


def bare_models(etas) -> dict[int, PairModel]:
    """``PairModel`` per pair with every background term exactly zero.

    The layer models the coherent sum alone, so a comparison against
    ``predict`` has to remove what the layer does not claim: the single-beam
    response and the dark.  Zeroing them is cleaner than passing
    ``coherent=True`` alone, because it also makes ``single_beam`` return 0 and
    so the two code paths are comparable term by term.
    """
    return {i + 1: PairModel(index=i + 1, eta=float(e), a_x=0.0, q_x=0.0,
                             a_w=0.0, q_w=0.0, d=0.0)
            for i, e in enumerate(etas)}


def layer_from(params, w, *, y_sat, n_tiles=1):
    """A minimal ONNLayer holding exactly ``w``: no gain, no readout, no squash."""
    k = params.n_channels
    layer = ONNLayer(k, w.shape[-2], params, n_tiles=n_tiles,
                     weight_parametrization="direct", input_map="clamp",
                     readout="none")
    layer.set_w(torch.as_tensor(w, dtype=torch.float64))
    return layer


@needs_torch
class TestAgainstVerifyPredict(unittest.TestCase):
    """forward() == verify.predict(coherent=True) on random drives."""

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self.rng = np.random.default_rng(20260926)

    def tearDown(self):
        torch.set_default_dtype(torch.float32)

    def _compare(self, k, n_out, n_tiles, y_sat, batch=13):
        etas = self.rng.uniform(0.03, 0.22, k)
        phis = self.rng.uniform(-math.pi, math.pi, k)
        models = bare_models(etas)
        driven = sorted(models)
        phases = {p: float(f) for p, f in zip(driven, phis)}

        params = ONNParams(eta=torch.tensor(etas), phi=torch.tensor(phis),
                           y_sat=y_sat if y_sat else math.inf,
                           channels=driven, source="test")
        # x is per (sample, shot, channel); w is per (shot, output, channel).
        x = self.rng.uniform(0.02, 0.98, (batch, n_tiles, k))
        w = self.rng.uniform(0.02, 0.98, (1, n_tiles, n_out, k))
        layer = layer_from(params, w, y_sat=y_sat, n_tiles=n_tiles)
        got = layer(torch.tensor(x.reshape(batch, -1))).detach().numpy()

        # predict() takes one (x, w) pair per measurement, so broadcast both to
        # (batch, shot, output, channel) and let it sum over the last axis.
        xb = np.broadcast_to(x[:, :, None, :], (batch, n_tiles, n_out, k))
        wb = np.broadcast_to(w[0][None], (batch, n_tiles, n_out, k))
        per_shot = predict(models, phases, driven, xb, wb,
                           coherent=True, y_sat=y_sat)
        want = per_shot.sum(axis=1)               # incoherent sum over the shots

        self.assertEqual(got.shape, want.shape)
        self.assertLess(np.abs(got - want).max(), 1e-12,
                        f"torch layer and verify.predict disagree by "
                        f"{np.abs(got - want).max():.3e}")

    def test_single_shot_no_saturation(self):
        self._compare(k=10, n_out=1, n_tiles=1, y_sat=None)

    def test_single_shot_saturated(self):
        self._compare(k=10, n_out=1, n_tiles=1, y_sat=4.82)

    def test_multi_output(self):
        self._compare(k=10, n_out=10, n_tiles=1, y_sat=4.82)

    def test_tiled_is_an_incoherent_sum_of_shots(self):
        """79 shots of 10 must add as detector readings, not as one wide sum."""
        self._compare(k=10, n_out=10, n_tiles=79, y_sat=4.82, batch=3)

    def test_saturation_matches_verify_saturate(self):
        """The layer's Y/(1+Y/Y_sat) is fit.verify.saturate, not a variant."""
        params = ONNParams.ideal(1, y_sat=0.05)
        layer = layer_from(params, np.full((1, 1, 1, 1), 0.7), y_sat=0.05)
        got = float(layer(torch.tensor([[0.6]])).item())
        # one ideal channel: |E|^2 = x*w before saturation
        self.assertAlmostEqual(got, float(saturate(0.6 * 0.7, 0.05)), places=12)


@needs_torch
class TestRealCalibration(unittest.TestCase):
    """ONNParams.from_step7 must reproduce verify.load_forward_model exactly."""

    @classmethod
    def setUpClass(cls):
        if not STEP7.is_file():
            raise unittest.SkipTest(f"{STEP7.name} not in src/calib_data")

    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def tearDown(self):
        torch.set_default_dtype(torch.float32)

    def test_eta_and_phi_come_through_untouched(self):
        models, phases, ref = load_forward_model(STEP7)
        params = ONNParams.from_step7(STEP7, y_sat=4.82)
        self.assertEqual(params.channels, sorted(models))
        self.assertEqual(params.reference, ref)
        for j, pair in enumerate(params.channels):
            self.assertAlmostEqual(float(params.eta[j]), models[pair].eta, places=15)
            # The sign is NOT adjusted anywhere: a stored dphi_comb_rad IS Phi_k.
            self.assertAlmostEqual(float(params.phi[j]), phases[pair], places=15)
        self.assertEqual(params.phase_sign, 1.0)

    def test_forward_matches_predict_on_the_real_fit(self):
        """The whole chain, on a measured calibration rather than random numbers."""
        models, phases, _ = load_forward_model(STEP7)
        driven = sorted(models)
        params = ONNParams.from_step7(STEP7, y_sat=4.82)
        rng = np.random.default_rng(7)
        k, batch = len(driven), 9
        x = rng.uniform(0.02, 0.98, (batch, k))
        w = rng.uniform(0.02, 0.98, (1, 1, 1, k))

        layer = layer_from(params, w, y_sat=4.82)
        got = layer(torch.tensor(x)).detach().numpy()[:, 0]
        # The background terms are real here, so coherent=True is what drops
        # them -- this is the comparison the bench post-processing implements.
        want = predict(models, phases, driven, x,
                       np.broadcast_to(w[0, 0, 0], (batch, k)),
                       coherent=True, y_sat=4.82)
        self.assertLess(np.abs(got - want).max(), 1e-12)

    def test_phi_half_is_the_layers_half_phase(self):
        """u = asin(sqrt(x)) is the same function on both sides."""
        x = np.linspace(0.0, 1.0, 101)
        want = phi_half(x)
        got = torch.asin(torch.tensor(x).clamp(0, 1).sqrt()).numpy()
        self.assertLess(np.abs(got - want).max(), 1e-14)


@needs_torch
class TestCalibrationIsNotAWeight(unittest.TestCase):
    """eta/Phi must not travel in a checkpoint."""

    def test_absent_from_state_dict(self):
        layer = ONNLayer(10, 10, ONNParams.ideal(10), n_tiles=3)
        keys = set(layer.state_dict())
        self.assertNotIn("eta", keys)
        self.assertNotIn("phi", keys)
        self.assertIn("raw_weight", keys)

    def test_a_reload_does_not_restore_the_old_calibration(self):
        """The bug this guards: --calib being silently overridden by a load."""
        old = ONNParams.ideal(4)
        old.eta = torch.full((4,), 0.10)
        trained = ONNLayer(4, 2, old, n_tiles=1)
        checkpoint = trained.state_dict()

        new = ONNParams.ideal(4)
        new.eta = torch.full((4,), 0.17)          # a fresh step-6 run
        fresh = ONNLayer(4, 2, new, n_tiles=1)
        fresh.load_state_dict(checkpoint)         # strict: keys must still match
        self.assertAlmostEqual(float(fresh.eta[0]), 0.17, places=6)

    def test_trainable_calibration_still_saves_it(self):
        """Opting in makes eta a weight, and then it belongs in the checkpoint."""
        layer = ONNLayer(4, 2, ONNParams.ideal(4), trainable_calibration=True)
        self.assertIn("eta", set(layer.state_dict()))


@needs_torch
class TestDriveQuantization(unittest.TestCase):
    """A 4-bit bench cannot display an arbitrary x in [0, 1]."""

    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def tearDown(self):
        torch.set_default_dtype(torch.float32)

    def test_drive_lands_on_the_grid(self):
        layer = ONNLayer(10, 10, ONNParams.ideal(10), n_tiles=2, drive_bits=4)
        x = torch.rand(32, 20)
        got = layer.drive(x).flatten()
        # the grid is held off 0 and 1 by eps, so compare to that tolerance
        grid = torch.linspace(0.0, 1.0, 16)
        self.assertLessEqual(len(torch.unique(got)), 16)
        self.assertLess((got.unsqueeze(-1) - grid).abs().min(-1).values.max(), 2e-6)

    def test_weights_are_quantized_too(self):
        """w is displayed on the same panel, so it takes the same 16 settings."""
        layer = ONNLayer(10, 4, ONNParams.ideal(10), n_tiles=1, drive_bits=4)
        self.assertLessEqual(len(torch.unique(layer.w())), 16)

    def test_gradient_survives_the_rounding(self):
        """Straight-through, or the input gain would have no gradient at all."""
        layer = ONNLayer(6, 3, ONNParams.ideal(6), n_tiles=2, drive_bits=4,
                         input_gain=True, readout="scale")
        layer(torch.rand(5, 12)).pow(2).sum().backward()
        for name in ("in_scale", "raw_weight"):
            g = getattr(layer, name).grad
            self.assertIsNotNone(g, f"{name} got no gradient")
            self.assertTrue(bool(torch.isfinite(g).all()), f"{name} gradient not finite")
            self.assertGreater(float(g.abs().max()), 0.0, f"{name} gradient is all zero")

    def test_a_custom_grid_overrides_the_uniform_one(self):
        """The real panel is uniform in LEVEL, not in x; the grid is injectable."""
        grid = torch.tensor([0.0, 0.01, 0.04, 0.09, 0.25, 1.0])   # sin^2-ish spacing
        layer = ONNLayer(3, 2, ONNParams.ideal(3), drive_levels=grid)
        got = layer.drive(torch.tensor([[0.5, 0.03, 0.8]])).flatten()
        off = (got.unsqueeze(-1) - grid).abs().min(-1).values
        self.assertLess(float(off.max()), 2e-6, got)
        # nearest-neighbour, not rounding down: 0.5 -> 0.25, 0.8 -> 1.0
        self.assertAlmostEqual(float(got[0]), 0.25, places=6)
        self.assertAlmostEqual(float(got[2]), 1.0, places=5)

    def test_continuous_by_default(self):
        layer = ONNLayer(4, 2, ONNParams.ideal(4))
        self.assertIsNone(layer.drive_levels)
        self.assertGreater(len(torch.unique(layer.drive(torch.rand(8, 4)))), 16)


@needs_torch
class TestSelftest(unittest.TestCase):
    """The layer's own fast-vs-reference check, as part of the suite."""

    def test_fast_path_matches_the_literal_equations(self):
        from onn_module.layer import _selftest
        try:
            _selftest()
        except SystemExit as exc:                  # _selftest raises on failure
            self.fail(f"onn_module.layer --selftest failed: {exc}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
