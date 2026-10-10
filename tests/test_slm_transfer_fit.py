"""The fitted sin^2 transfer curve and the encoding built on it.

Covers :mod:`calibration_module.fit.transfer` and the encoding built on it in
:class:`slm_module.encoding.EncodingChannel` -- inverting the fit is the only
level mapping there is.
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calibration_module.fit.slm_response import (  # noqa: E402
    CalibrationResult,
    load_calibration_result,
    save_calibration_result,
)
from calibration_module.fit.transfer import (  # noqa: E402
    MAX_LEVEL,
    TransferFit,
    TransferFitError,
    fit_transfer_curve,
    fit_transfer_curves,
)
from slm_module.encoding import (  # noqa: E402
    EncodingChannel,
    build_channel_layout,
    build_single_anchor_layout,
    channel_layout_from_calibration,
)

# a realistic sweep: 35 levels over roughly the range Step 3b uses
LEVELS = np.arange(380, 891, 15, dtype=int)
OFF_TRUE, ON_TRUE = 390.0, 860.0


def curve(levels=LEVELS, *, floor=0.0, contrast=1.0,
          off_level=OFF_TRUE, on_level=ON_TRUE) -> np.ndarray:
    """Exact sin^2 curve with retardance 0 at ``off_level`` and pi at ``on_level``."""
    slope = math.pi / (on_level - off_level)
    theta = slope * (np.asarray(levels, dtype=float) - off_level)
    return floor + contrast * np.sin(theta / 2.0) ** 2


class FitTransferCurveTests(unittest.TestCase):
    def test_recovers_known_parameters(self) -> None:
        fit = fit_transfer_curve(LEVELS, curve(contrast=0.97))
        self.assertAlmostEqual(fit.contrast, 0.97, places=4)
        self.assertAlmostEqual(fit.floor, 0.0, places=4)
        self.assertAlmostEqual(fit.off_level_exact, OFF_TRUE, places=2)
        self.assertAlmostEqual(fit.on_level_exact, ON_TRUE, places=2)
        self.assertLess(fit.rms, 1e-5)
        self.assertEqual(fit.n_used, LEVELS.size)
        self.assertEqual((fit.level_min, fit.level_max), (380, 890))

    def test_floor_is_clamped_non_negative(self) -> None:
        """A normalized power cannot be negative, so the floor is bounded at 0.

        Real curves want a slightly negative floor -- the reference
        normalization over-subtracts by ~1% -- and letting them have it would
        break the v -> retardance identity the phase model relies on.
        """
        dipped = curve() - 0.02
        fit = fit_transfer_curve(LEVELS, dipped)
        self.assertGreaterEqual(fit.floor, 0.0)

    def test_slope_normalized_positive_and_offset_on_first_branch(self) -> None:
        """Whatever branch the optimizer lands on, the stored fit is canonical."""
        fit = fit_transfer_curve(LEVELS, curve())
        self.assertGreater(fit.phase_slope, 0.0)
        # retardance ~ 0 at the darkest swept level, pi at full scale
        self.assertAlmostEqual(float(fit.retardance(OFF_TRUE)), 0.0, places=3)
        self.assertAlmostEqual(float(fit.retardance(ON_TRUE)), math.pi, places=3)

    def test_rejects_curves_it_cannot_fit(self) -> None:
        with self.assertRaisesRegex(TransferFitError, "at least 5"):
            fit_transfer_curve(np.array([0, 1, 2]), np.array([0.0, 0.5, 1.0]))
        with self.assertRaisesRegex(TransferFitError, "flat"):
            fit_transfer_curve(LEVELS, np.zeros(LEVELS.size))
        with self.assertRaisesRegex(TransferFitError, "matching 1-D"):
            fit_transfer_curve(LEVELS, np.zeros((2, LEVELS.size)))

    def test_fit_transfer_curves_names_the_bad_row(self) -> None:
        rows = np.stack([curve(), np.zeros(LEVELS.size)])
        with self.assertRaisesRegex(TransferFitError, "channel row 1"):
            fit_transfer_curves(LEVELS, rows)

    def test_extrapolated_flags_a_peak_past_the_sweep(self) -> None:
        inside = fit_transfer_curve(LEVELS, curve())
        self.assertFalse(inside.extrapolated)
        # peak at 1000, well past the 890 the sweep reached
        beyond = fit_transfer_curve(LEVELS, curve(on_level=1000.0))
        self.assertTrue(beyond.extrapolated)
        self.assertFalse(beyond.clipped)          # still inside the panel's range

    def test_dict_round_trip(self) -> None:
        fit = fit_transfer_curve(LEVELS, curve(contrast=0.9))
        back = TransferFit.from_dict(json.loads(json.dumps(fit.to_dict())))
        self.assertEqual(back, fit)
        with self.assertRaises(TransferFitError):
            TransferFit.from_dict({"floor": 0.0})     # missing parameters


class SpikeRejectionTests(unittest.TestCase):
    """The failure this module exists to fix: argmax chasing one noisy sample.

    In the 0903 calibration one channel had a spike at level 770 -- 12% above
    its immediate neighbour, nine swept levels below its real peak.  ``argmax``
    took it for full scale, so ``v = 1`` delivered ~94% of the channel's actual
    maximum and every other ``v`` over-delivered relative to it.
    """

    def setUp(self) -> None:
        self.clean = curve()
        self.spiked = self.clean.copy()
        self.spike_idx = int(np.argmin(np.abs(LEVELS - 770)))
        self.spiked[self.spike_idx] *= 1.12

    def test_argmax_takes_the_spike_but_the_fit_does_not(self) -> None:
        self.assertEqual(int(LEVELS[int(np.argmax(self.spiked))]),
                         int(LEVELS[self.spike_idx]))
        fit = fit_transfer_curve(LEVELS, self.spiked)
        # the fit stays within a few levels of the true peak
        self.assertAlmostEqual(fit.on_level_exact, ON_TRUE, delta=8.0)

    def test_robust_loss_limits_the_spike_pull(self) -> None:
        robust = fit_transfer_curve(LEVELS, self.spiked, robust=True)
        plain = fit_transfer_curve(LEVELS, self.spiked, robust=False)
        self.assertLessEqual(
            abs(robust.on_level_exact - ON_TRUE),
            abs(plain.on_level_exact - ON_TRUE) + 1e-9,
        )

    def test_encoding_channel_full_scale_ignores_the_spike(self) -> None:
        """Full scale is the modelled Delta = pi, not the level of the spike."""
        ch = self._channel(self.spiked)
        swept_argmax = int(LEVELS[self.spike_idx])                       # 770
        self.assertGreater(ch.on_level, swept_argmax + 50)               # ~860
        self.assertAlmostEqual(ch.on_level, round(ON_TRUE), delta=8)

    @staticmethod
    def _channel(values):
        return EncodingChannel(
            index=0, side="x", x_center=100, x_start=93, x_end=108,
            wavelength_nm=780.0, levels=LEVELS.copy(),
            intensity_curve=np.asarray(values, dtype=float),
        )


class FitEncodingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ch = EncodingChannel(
            index=0, side="x", x_center=100, x_start=93, x_end=108,
            wavelength_nm=780.0, levels=LEVELS.copy(),
            intensity_curve=curve(),
        )

    def test_endpoints_are_the_fitted_window(self) -> None:
        self.assertEqual(self.ch.level_for(0.0), self.ch.off_level)
        self.assertEqual(self.ch.level_for(1.0), self.ch.on_level)
        self.assertAlmostEqual(self.ch.off_level, round(OFF_TRUE), delta=1)
        self.assertAlmostEqual(self.ch.on_level, round(ON_TRUE), delta=1)

    def test_out_of_range_values_clip(self) -> None:
        self.assertEqual(self.ch.level_for(-0.5), self.ch.off_level)
        self.assertEqual(self.ch.level_for(1.5), self.ch.on_level)

    def test_monotone_in_v(self) -> None:
        out = [self.ch.level_for(v) for v in np.linspace(0.0, 1.0, 201)]
        self.assertTrue(all(b >= a for a, b in zip(out, out[1:])))

    def test_levels_stay_inside_the_panel_range(self) -> None:
        far = EncodingChannel(
            index=0, side="x", x_center=100, x_start=93, x_end=108,
            wavelength_nm=780.0, levels=LEVELS.copy(),
            intensity_curve=curve(on_level=1400.0),
        )
        for v in (0.0, 0.5, 1.0):
            self.assertTrue(0 <= far.level_for(v) <= MAX_LEVEL)

    def test_delivered_power_matches_the_request(self) -> None:
        """v is the fraction of the fitted contrast above the fitted floor."""
        fit = self.ch.transfer_fit
        for v in (0.1, 0.25, 0.5, 0.75, 0.9):
            got = float(fit.intensity(self.ch.level_for(v)))
            want = fit.floor + fit.contrast * v
            # the residual is the integer-grayscale quantum: rounding the level
            # by up to half a step moves the retardance by slope/2 rad
            self.assertAlmostEqual(got, want, delta=2e-3)

    def test_v_maps_to_the_retardance_the_phase_model_assumes(self) -> None:
        """The identity that interpolation could not promise.

        ``calibration_module.fit.phase.phi_half`` takes phi/2 = asin(sqrt(v)) for a
        channel commanded at v.  Inverting the fit makes that exact, so the
        amplitude the encoder writes and the phase steps 6/7/8 assume come from
        one model.
        """
        from calibration_module.fit.phase import phi_half

        fit = self.ch.transfer_fit
        for v in (0.05, 0.2, 0.5, 0.8, 1.0):
            delivered = float(fit.retardance(fit.level_for(v)))
            self.assertAlmostEqual(delivered, 2.0 * float(phi_half(v)), places=2)

    def test_a_curve_is_fitted_when_none_was_supplied(self) -> None:
        """A hand-built channel arrives with a curve and no fit; it gets one."""
        self.assertIsNotNone(self.ch.transfer_fit)

    def test_neither_a_curve_nor_a_fit_is_refused(self) -> None:
        with self.assertRaisesRegex(TransferFitError, "nothing to encode through"):
            EncodingChannel(
                index=0, side="x", x_center=100, x_start=93, x_end=108,
                wavelength_nm=780.0, levels=LEVELS.copy(),
            )


class CalibrationPlumbingTests(unittest.TestCase):
    """Fits are written with the calibration and picked up by the loader."""

    GRID = np.array([420.0, 440.0, 480.0, 500.0, 520.0, 540.0, 580.0, 600.0])

    def _calib(self) -> CalibrationResult:
        rows = np.stack([curve(contrast=1.0 + 0.01 * i) for i in range(self.GRID.size)])
        return CalibrationResult(
            wavelength=-0.005 * self.GRID + 781.0,
            coordinates=self.GRID.copy(),
            max_level=860,
            min_level=390,
            level_range=LEVELS.copy(),
            intensity_levels=rows,
        )

    def test_save_fits_and_load_restores_them(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = save_calibration_result(self._calib(), Path(tmp) / "step3b.json")
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], "calibration_result_v2")
            self.assertEqual(len(payload["transfer_fits"]), self.GRID.size)
            # the raw sweep stays in the file, so a curve can always be refitted
            self.assertIsNotNone(payload["intensity_levels"])

            back = load_calibration_result(out)
            self.assertEqual(len(back.transfer_fits), self.GRID.size)
            self.assertAlmostEqual(back.transfer_fits[0].on_level_exact, ON_TRUE,
                                   places=2)

    def test_save_without_fitting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = save_calibration_result(self._calib(), Path(tmp) / "raw.json",
                                          fit_transfer=False)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertIsNone(payload["transfer_fits"])
            self.assertIsNone(load_calibration_result(out).transfer_fits)

    def test_unfittable_curves_still_save(self) -> None:
        """A fit failure must never cost the measurement."""
        calib = self._calib()
        calib.intensity_levels = np.zeros_like(calib.intensity_levels)
        with tempfile.TemporaryDirectory() as tmp:
            out = save_calibration_result(calib, Path(tmp) / "flat.json")
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertIsNone(payload["transfer_fits"])
            self.assertIsNotNone(payload["intensity_levels"])

    def test_layout_prefers_stored_fits_over_refitting(self) -> None:
        calib = self._calib()
        stored = fit_transfer_curves(calib.level_range, calib.intensity_levels)
        # a recognisable marker: shift one channel's window by 40 levels
        row = 0
        moved = TransferFit(
            floor=stored[row].floor, contrast=stored[row].contrast,
            phase_slope=stored[row].phase_slope,
            phase_offset=stored[row].phase_offset - 40.0 * stored[row].phase_slope,
            level_min=stored[row].level_min, level_max=stored[row].level_max,
        )
        calib.transfer_fits = [moved] + stored[1:]
        layout = channel_layout_from_calibration(calib, warn=False)

        # row 0 is coordinate 420 px -- the outermost x channel
        ch = next(c for c in layout.all_channels if c.x_center == 420)
        self.assertAlmostEqual(ch.transfer_fit.on_level_exact, ON_TRUE + 40.0, places=2)
        # everything else refits to the true window
        other = next(c for c in layout.all_channels if c.x_center == 500)
        self.assertAlmostEqual(other.transfer_fit.on_level_exact, ON_TRUE, places=2)

    def test_layout_fits_when_the_file_has_none(self) -> None:
        layout = channel_layout_from_calibration(self._calib(), warn=False)
        self.assertTrue(all(c.transfer_fit is not None for c in layout.all_channels))
        # background columns sit at the modelled extinction, not the swept argmin
        self.assertTrue(
            np.allclose(layout.calib_off_levels, round(OFF_TRUE), atol=1)
        )


class AlignmentBuilderTests(unittest.TestCase):
    """The tiling and anchor builders share one fit across several channels.

    They give several channels the SAME measured curve -- one snapped from the
    nearest calibration coordinate, one shared anchor -- so their full scale is
    one number wearing many hats.  That sharing is the approximation, and it is
    what these tests pin; the inversion itself is the fit, as everywhere else.
    """

    def test_build_channel_layout_shares_the_snapped_row_fit(self) -> None:
        coords = np.linspace(100.0, 900.0, 9)
        calib = CalibrationResult(
            wavelength=-0.005 * coords + 781.0,
            coordinates=coords,
            max_level=860,
            min_level=390,
            level_range=LEVELS.copy(),
            intensity_levels=np.stack([curve() for _ in coords]),
        )
        fitted = build_channel_layout(calib, n_channels=4)
        self.assertTrue(all(c.transfer_fit is not None for c in fitted.all_channels))
        for ch in fitted.all_channels:
            self.assertAlmostEqual(ch.on_level, round(ON_TRUE), delta=1)

    def test_single_anchor_layout_shares_the_anchor_fit(self) -> None:
        step2 = CalibrationResult(
            wavelength=np.asarray([780.0, 778.0, 776.0]),
            coordinates=np.asarray([0.0, 100.0, 200.0]),
            max_level=860, min_level=390, level_range=LEVELS.copy(),
        )
        intensity = CalibrationResult(
            wavelength=np.asarray([778.0]),
            coordinates=np.asarray([100.0]),
            max_level=860, min_level=390, level_range=LEVELS.copy(),
            intensity_levels=curve()[None, :],
        )
        fitted, _ = build_single_anchor_layout(step2, intensity,
                                               target_wavelength_nm=778.0)
        # every channel shares the anchor's fit, hence its window
        self.assertEqual({c.on_level for c in fitted.all_channels},
                         {round(ON_TRUE)})


def quad_curve(levels=LEVELS, *, contrast=1.0, curv=-2.0e-6,
               off_level=OFF_TRUE, on_level=ON_TRUE) -> np.ndarray:
    """A sin^2 curve whose retardance is quadratic in level.

    Built from the endpoints so it stays comparable with :func:`curve`:
    ``Delta = 0`` at ``off_level`` and ``pi`` at ``on_level`` either way, with
    ``curv`` bending the path between them.  ``curv = -2e-6`` is the size the
    0907 Step-3c channels actually show.
    """
    lv = np.asarray(levels, dtype=float)
    # Delta = curv*L^2 + a*L + b, pinned to 0 at off and pi at on
    a = (math.pi - curv * (on_level ** 2 - off_level ** 2)) / (on_level - off_level)
    b = -(curv * off_level ** 2 + a * off_level)
    return contrast * np.sin((curv * lv * lv + a * lv + b) / 2.0) ** 2


class CurvatureTests(unittest.TestCase):
    """The quadratic retardance term -- see "Why a curvature term" in transfer.py.

    The 0907 Step-3c sweep resolved a real sublinearity in the panel's phase
    response that the noisier 0903 OSA sweeps could not.  These tests pin the
    two things that matter: it is recovered when present, and it stays exactly
    absent when it is not.
    """

    def test_recovers_a_known_curvature(self) -> None:
        fit = fit_transfer_curve(LEVELS, quad_curve(contrast=0.9, curv=-2.0e-6))
        self.assertAlmostEqual(fit.phase_curv, -2.0e-6, places=9)
        self.assertAlmostEqual(fit.contrast, 0.9, places=4)
        self.assertAlmostEqual(fit.off_level_exact, OFF_TRUE, places=1)
        self.assertAlmostEqual(fit.on_level_exact, ON_TRUE, places=1)
        self.assertLess(fit.rms, 1e-5)

    def test_curvature_beats_the_linear_model_on_a_curved_panel(self) -> None:
        """The whole reason the term exists: the linear fit cannot follow this."""
        data = quad_curve(curv=-2.0e-6)
        straight = fit_transfer_curve(LEVELS, data, curvature=False)
        bent = fit_transfer_curve(LEVELS, data)
        self.assertEqual(straight.phase_curv, 0.0)
        self.assertLess(bent.rms, straight.rms / 5.0)

    def test_linear_curve_keeps_curvature_exactly_zero(self) -> None:
        """A fifth parameter must not appear out of fitting noise.

        The linear fit of an exact sin^2 curve already sits at the optimizer's
        convergence floor, so a quadratic term can always shave the rms a
        little.  It is rejected on whether it moves a grayscale level, not on
        whether it improves the residual.
        """
        fit = fit_transfer_curve(LEVELS, curve())
        self.assertEqual(fit.phase_curv, 0.0)
        self.assertAlmostEqual(fit.on_level_exact, ON_TRUE, places=2)

    def test_curvature_disabled_reproduces_the_linear_fit(self) -> None:
        data = quad_curve()
        off = fit_transfer_curve(LEVELS, data, curvature=False)
        self.assertEqual(off.phase_curv, 0.0)
        # ... and reduces to the plain linear retardance
        np.testing.assert_allclose(
            off.retardance(LEVELS),
            off.phase_slope * LEVELS + off.phase_offset,
        )

    def test_too_few_points_falls_back_to_linear(self) -> None:
        """Five parameters need six points; five still fit the linear model."""
        few = np.array([390, 480, 570, 660, 750], dtype=float)
        fit = fit_transfer_curve(few, quad_curve(few))
        self.assertEqual(fit.phase_curv, 0.0)
        self.assertEqual(fit.n_used, 5)

    def test_branch_convention_holds_with_curvature(self) -> None:
        """Delta still rises from 0 at the darkest level to pi at full scale."""
        fit = fit_transfer_curve(LEVELS, quad_curve())
        self.assertAlmostEqual(float(fit.retardance(fit.off_level_exact)), 0.0, places=6)
        self.assertAlmostEqual(float(fit.retardance(fit.on_level_exact)), math.pi, places=6)
        # rising across the whole sweep, so level_for is single-valued
        self.assertTrue(bool(np.all(fit.phase_rate(LEVELS) > 0.0)))
        self.assertGreater(fit.turning_level, float(LEVELS.max()))

    def test_phase_rate_varies_and_matches_the_derivative(self) -> None:
        fit = fit_transfer_curve(LEVELS, quad_curve(curv=-2.0e-6))
        # np.gradient is one-sided at the ends, so compare the interior only
        numeric = np.gradient(fit.retardance(LEVELS), LEVELS.astype(float))
        np.testing.assert_allclose(fit.phase_rate(LEVELS)[1:-1], numeric[1:-1], rtol=1e-6)
        # sublinear: slower near full scale than near extinction
        self.assertLess(float(fit.phase_rate(ON_TRUE)), float(fit.phase_rate(OFF_TRUE)))

    def test_level_for_inverts_the_curved_model(self) -> None:
        """v -> level -> v is the identity the encoder relies on."""
        fit = fit_transfer_curve(LEVELS, quad_curve())
        for v in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
            delivered = float(np.sin(fit.retardance(fit.level_for(v)) / 2.0) ** 2)
            self.assertAlmostEqual(delivered, v, places=2)   # integer grayscale

    def test_curved_and_linear_models_command_different_levels(self) -> None:
        """If they agreed there would be nothing to fix; on 0907 they differ ~21."""
        data = quad_curve(curv=-2.0e-6)
        straight = fit_transfer_curve(LEVELS, data, curvature=False)
        bent = fit_transfer_curve(LEVELS, data)
        shift = max(abs(bent.level_for(v) - straight.level_for(v))
                    for v in np.linspace(0.0, 1.0, 21))
        self.assertGreater(shift, 5)

    def test_rejects_a_curvature_that_turns_over_inside_the_sweep(self) -> None:
        """A Delta that stops rising mid-sweep is not a transfer curve.

        Data that peaks and comes back down inside the swept range can be
        matched by a parabola turning over; the fitter must decline it rather
        than store a model it cannot invert.
        """
        wide = np.arange(380, 1400, 15, dtype=int)   # sweeps past a full period
        fit = fit_transfer_curve(wide, curve(wide))
        self.assertTrue(
            fit.phase_curv == 0.0
            or not (wide.min() <= fit.turning_level <= wide.max())
        )

    def test_dict_round_trip_carries_curvature(self) -> None:
        fit = fit_transfer_curve(LEVELS, quad_curve())
        self.assertNotEqual(fit.phase_curv, 0.0)
        back = TransferFit.from_dict(json.loads(json.dumps(fit.to_dict())))
        self.assertEqual(back, fit)
        self.assertIn("phase_curv", fit.to_dict())

    def test_legacy_payload_without_curvature_loads_as_linear(self) -> None:
        """Files written before the term existed must not change meaning."""
        legacy = fit_transfer_curve(LEVELS, curve()).to_dict()
        del legacy["phase_curv"]
        back = TransferFit.from_dict(legacy)
        self.assertEqual(back.phase_curv, 0.0)
        np.testing.assert_allclose(
            back.retardance(LEVELS),
            back.phase_slope * LEVELS + back.phase_offset,
        )
        self.assertAlmostEqual(back.on_level_exact, ON_TRUE, places=2)

    def test_curvature_survives_the_calibration_file(self) -> None:
        """End to end: a curved sweep saved by step 3 comes back curved.

        This is the path Step 3c writes and steps 6/7/8 read.
        """
        levels = np.asarray(LEVELS, dtype=float)
        calib = CalibrationResult(
            wavelength=np.array([778.0, 778.2]),
            coordinates=np.array([500.0, 520.0]),
            max_level=int(levels.max()),
            min_level=int(levels.min()),
            level_range=levels,
            intensity_levels=np.stack([quad_curve(), quad_curve(contrast=0.8)]),
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = save_calibration_result(calib, Path(tmp) / "step3c.json")
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertIn("phase_curv", payload["transfer_fits"][0])
            self.assertNotEqual(payload["transfer_fits"][0]["phase_curv"], 0.0)

            fits = load_calibration_result(out).transfer_fits
            self.assertAlmostEqual(fits[0].phase_curv, -2.0e-6, places=9)


# the 0907 channel-11 failure, reproduced: four adjacent cells reading ~25x the
# curve.  Levels chosen inside the sweep so they are ordinary points, not ends.
SPIKE_LEVELS = (845, 860, 875, 890)


def spiked_curve(levels=LEVELS, *, factor=25.0, where=SPIKE_LEVELS, **kw):
    """-> (curve with latched cells, boolean mask of the bad points)."""
    lv = np.asarray(levels, dtype=float)
    y = quad_curve(levels, **kw)
    bad = np.isin(lv, np.asarray(where, dtype=float))
    y = y.copy()
    y[bad] = float(np.max(y)) * factor
    return y, bad


class ClipTests(unittest.TestCase):
    """Dropping bad cells outright -- see "Bad cells" in transfer.py.

    The soft-L1 loss down-weights a disagreeing point; it does not remove it.
    On 0907 channel 11 that was not enough, and because the fit is frozen into
    the JSON at write time, the damage would have reached every later step
    6/7/8 run silently.  These tests pin both directions: the bad cells go, and
    clean curves keep every point they have.
    """

    # --- it removes what it should ---

    def test_drops_exactly_the_latched_cells(self) -> None:
        y, bad = spiked_curve()
        fit = fit_transfer_curve(LEVELS, y)
        self.assertEqual(fit.n_clipped, int(bad.sum()))
        self.assertEqual(fit.n_used, LEVELS.size - int(bad.sum()))

    def test_the_clipped_fit_recovers_the_true_curve(self) -> None:
        y, _ = spiked_curve(contrast=0.9, curv=-2.0e-6)
        fit = fit_transfer_curve(LEVELS, y)
        self.assertAlmostEqual(fit.contrast, 0.9, places=3)
        self.assertAlmostEqual(fit.off_level_exact, OFF_TRUE, places=1)
        self.assertAlmostEqual(fit.on_level_exact, ON_TRUE, places=1)
        self.assertAlmostEqual(fit.phase_curv, -2.0e-6, places=9)

    def test_clipping_beats_the_robust_loss_alone(self) -> None:
        """The whole reason the second stage exists.

        Note what "ruined" looks like: the soft-L1 fit stretches ``contrast``
        to swallow the spikes rather than leaving a large residual, so its
        rms/contrast reads *small*.  The damage is in the parameters, which is
        why that ratio alone is not a health check.
        """
        kept = fit_transfer_curve(LEVELS, spiked_curve()[0], clip=0)
        dropped = fit_transfer_curve(LEVELS, spiked_curve()[0])
        self.assertGreater(kept.contrast, 50.0)                # true value is 1
        self.assertGreater(kept.on_level_exact, 3000.0)        # true value is 860
        self.assertAlmostEqual(dropped.contrast, 1.0, places=3)
        self.assertAlmostEqual(dropped.on_level_exact, ON_TRUE, places=1)

    def test_a_ruined_encoding_window_is_repaired(self) -> None:
        """0907 ch11 in miniature: the window is what reaches the panel."""
        y, _ = spiked_curve()
        kept = fit_transfer_curve(LEVELS, y, clip=0)
        dropped = fit_transfer_curve(LEVELS, y)
        self.assertGreater(abs(kept.on_level - ON_TRUE), 100)
        self.assertLess(abs(dropped.on_level - ON_TRUE), 5)

    def test_clip_zero_keeps_every_point(self) -> None:
        y, _ = spiked_curve()
        fit = fit_transfer_curve(LEVELS, y, clip=0)
        self.assertEqual(fit.n_clipped, 0)
        self.assertEqual(fit.n_used, LEVELS.size)

    # --- it keeps what it should ---

    def test_a_clean_linear_curve_loses_nothing(self) -> None:
        """Guards the sigma floor.

        On an exact curve the residual is the optimizer's convergence floor and
        the MAD collapses with it, so an unfloored ratio test would call every
        point thousands of sigma out and eat the whole sweep.
        """
        fit = fit_transfer_curve(LEVELS, curve(contrast=0.9))
        self.assertEqual(fit.n_clipped, 0)
        self.assertEqual(fit.n_used, LEVELS.size)

    def test_a_clean_curved_curve_loses_nothing(self) -> None:
        """Guards clipping against the final model rather than the linear stage.

        A quadratic sweep judged against the linear fit shows smooth mismatch
        that is largest at the ends -- structure, not noise -- and the endpoints
        would be trimmed off a perfectly good curve.
        """
        fit = fit_transfer_curve(LEVELS, quad_curve(contrast=0.9, curv=-3.0e-6))
        self.assertEqual(fit.n_clipped, 0)
        self.assertEqual(fit.level_min, int(LEVELS.min()))
        self.assertEqual(fit.level_max, int(LEVELS.max()))

    def test_ordinary_noise_survives(self) -> None:
        rng = np.random.default_rng(11)
        y = quad_curve(contrast=0.9) + rng.normal(0.0, 0.01, LEVELS.size)
        fit = fit_transfer_curve(LEVELS, y)
        self.assertEqual(fit.n_clipped, 0)

    def test_a_channel_that_is_not_sin2_is_not_tidied_into_looking_healthy(self):
        """Clipping must not turn a broken channel into a confident wrong fit.

        Half a sweep that stops following sin^2 is not bad cells.  The clip is
        allowed to take a point or two, but what comes out must still be
        obviously unwell -- here the residual stays at 10% of contrast against
        the 0.2% the good 0907 channels sit at.
        """
        y = quad_curve(contrast=0.9).copy()
        y[LEVELS > 700] = 0.45                     # 13 of 35 levels go flat
        fit = fit_transfer_curve(LEVELS, y)
        self.assertLess(fit.n_clipped, 4)
        self.assertGreater(fit.rms / fit.contrast, 0.05)

    def test_outliers_that_capture_the_fit_are_beyond_the_clip(self) -> None:
        """The documented limit of fit-then-clip.

        Clipping can only find points the fit disagrees with.  Four adjacent
        cells at twice full scale pull the curve onto themselves, so their
        residuals are small and no threshold flags them; a lower one deletes
        good points instead and the fit stays just as wrong.  This is a
        boundary, not a regression -- what catches it downstream is the absurd
        contrast and encoding window, which the assertions below pin.
        """
        y, _ = spiked_curve(factor=2.0)
        for clip in (6.0, 3.0):
            fit = fit_transfer_curve(LEVELS, y, clip=clip)
            self.assertGreater(fit.contrast, 5.0)          # true value is 1
            self.assertGreater(fit.on_level_exact, MAX_LEVEL)
            self.assertTrue(fit.clipped)                   # window off the panel

    # --- plumbing ---

    def test_each_channel_is_clipped_on_its_own_residuals(self) -> None:
        y, bad = spiked_curve()
        fits = fit_transfer_curves(LEVELS, np.stack([curve(contrast=0.9), y]))
        self.assertEqual(fits[0].n_clipped, 0)
        self.assertEqual(fits[1].n_clipped, int(bad.sum()))

    def test_n_clipped_round_trips(self) -> None:
        y, bad = spiked_curve()
        fit = fit_transfer_curve(LEVELS, y)
        payload = fit.to_dict()
        self.assertEqual(payload["n_clipped"], int(bad.sum()))
        self.assertEqual(TransferFit.from_dict(payload), fit)

    def test_legacy_payload_reports_no_clipping(self) -> None:
        payload = fit_transfer_curve(LEVELS, curve()).to_dict()
        payload.pop("n_clipped")
        self.assertEqual(TransferFit.from_dict(payload).n_clipped, 0)

    def test_saved_calibration_stores_the_repaired_fit(self) -> None:
        """End to end: the JSON steps 6/7/8 read carries the clipped fit."""
        y, bad = spiked_curve(contrast=0.9)
        calib = CalibrationResult(
            wavelength=np.array([778.0]),
            coordinates=np.array([500.0]),
            max_level=int(LEVELS.max()),
            min_level=int(LEVELS.min()),
            level_range=LEVELS,
            intensity_levels=np.stack([y]),
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = save_calibration_result(calib, Path(tmp) / "step3c.json")
            stored = load_calibration_result(out).transfer_fits[0]
        self.assertEqual(stored.n_clipped, int(bad.sum()))
        self.assertAlmostEqual(stored.on_level_exact, ON_TRUE, places=1)
        self.assertLess(stored.rms / stored.contrast, 0.01)


if __name__ == "__main__":
    unittest.main()
