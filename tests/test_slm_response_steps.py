"""The step 1-3 runners in calibration_module.measure.slm_response.

A runner is only the sequence a step makes -- sweep, grid, clip -- written once
so the GUI and steps/calib_step*.py cannot drift apart.  So each is checked
against making those same calls by hand on identical fake data: same arrays
out, same progress, nothing added.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from osa_module.controller import MeasurementSettings  # noqa: E402
from calibration_module.fit.slm_response import (  # noqa: E402
    CalibrationResult,
    build_channel_calibration_grid,
)
from calibration_module.measure.slm_response import (  # noqa: E402
    batch_intensity_calibration,
    intensity_calibration_daq,
    refine_center_coordinate_with_osa,
    run_step1,
    run_step2,
    run_step3b,
    run_step3c,
)
from test_calibration_new import (  # noqa: E402
    FakeNarrowOSA,
    FakeOSA,
    FakeSLM,
    _BucketDAQ,
    _SimWindowOSA,
    make_trace,
)


def _linear_map(n: int, wl0: float, slope: float) -> CalibrationResult:
    coords = np.arange(n, dtype=float)
    return CalibrationResult(
        wavelength=wl0 + slope * coords,
        coordinates=coords,
        max_level=900,
        min_level=400,
        level_range=np.asarray([], dtype=int),
    )


class _FullScreenSLM(FakeSLM):
    def __init__(self) -> None:
        super().__init__()
        self.grayscales: list[int] = []

    def display_grayscale(self, level: int) -> None:
        self.grayscales.append(int(level))


class RunStep1Tests(unittest.TestCase):
    def test_seed_carries_the_extreme_levels_and_no_map(self) -> None:
        wls = np.asarray([777.0, 778.0, 779.0])
        powers = [0.3, 0.1, 0.9, 0.5]          # levels 0, 300, 600, 900
        osa = FakeOSA([make_trace(wls, [p, p, p]) for p in powers])
        slm = _FullScreenSLM()

        result = run_step1(osa, slm, [0, 300, 600, 900], MeasurementSettings())

        self.assertEqual(slm.grayscales, [0, 300, 600, 900])

        self.assertEqual(result.min_level, 300)
        self.assertEqual(result.max_level, 600)
        np.testing.assert_array_equal(result.level_range, [0, 300, 600, 900])
        self.assertEqual(result.coordinates.size, 0)
        self.assertEqual(result.wavelength.size, 0)


class RunStep2Tests(unittest.TestCase):
    def _run(self, **band):
        slm = FakeSLM(size=(12, 2))
        osa = _SimWindowOSA(slm, bogus_start=-1)     # peak at 710 + 8*start nm
        seed = CalibrationResult(
            wavelength=np.asarray([]), coordinates=np.asarray([]),
            max_level=100, min_level=0, level_range=np.asarray([], dtype=int),
        )
        return run_step2(osa, slm, MeasurementSettings(), seed, window_size=2, **band)

    def test_band_clips_the_saved_map_but_not_the_sweep(self) -> None:
        clipped, swept = self._run(min_wavelength_nm=716.0, max_wavelength_nm=764.0)

        self.assertEqual(swept.coordinates.size, 11)          # 710 .. 790 nm
        np.testing.assert_allclose(clipped.wavelength, 718.0 + 8.0 * np.arange(6))
        # the fit still describes the whole swept region
        np.testing.assert_array_equal(
            clipped.wavelength_fit_coefficients, swept.wavelength_fit_coefficients
        )

    def test_no_band_hands_on_the_whole_sweep(self) -> None:
        clipped, swept = self._run()
        self.assertIs(clipped, swept)


class RunStep3cTests(unittest.TestCase):
    GRID = dict(target_wavelength_nm=778.0, n_channels_per_side=2,
                channel_width_px=8, gap_px=4)

    def test_matches_the_grid_then_daq_sweep_made_by_hand(self) -> None:
        step2 = _linear_map(200, 770.0, 0.08)            # 778 nm at x = 100
        levels = [400, 650, 900]

        slm = FakeSLM(size=(200, 4))
        daq = _BucketDAQ(slm)
        seen = []
        sweep = run_step3c(daq, slm, levels, step2, progress_callback=seen.append,
                           **self.GRID)

        hand_slm = FakeSLM(size=(200, 4))
        seed, center = build_channel_calibration_grid(
            step2, slm_width=200,
            **{k: v for k, v in self.GRID.items()},
        )
        expected = intensity_calibration_daq(
            _BucketDAQ(hand_slm), hand_slm, levels, seed, window_size=8
        )

        self.assertAlmostEqual(sweep.center_coordinate, center)
        self.assertAlmostEqual(sweep.center_coordinate, 100.0)
        np.testing.assert_allclose(sweep.result.coordinates, expected.coordinates)
        np.testing.assert_allclose(
            sweep.result.intensity_levels, expected.intensity_levels
        )
        self.assertEqual((sweep.result.channel_width_px, sweep.result.gap_px), (8, 4))
        self.assertIsNone(sweep.coarse_center)
        self.assertIsNone(sweep.measured_peak_nm)
        self.assertEqual(daq.reads, 1 + 4 * len(levels))
        self.assertEqual(seen[0].phase, "fast_center")
        self.assertEqual({p.phase for p in seen[1:]}, {"intensity"})


class RunStep3bTests(unittest.TestCase):
    # 1 nm per px, 106 nm at x = 6; channels 1 px wide on a 2 px pitch
    GRID = dict(target_wavelength_nm=106.0, n_channels_per_side=2,
                channel_width_px=1, gap_px=1)
    WLS = np.arange(100.0, 112.0)

    def _traces(self, fine_tune: bool) -> list:
        rng = np.random.default_rng(7)
        traces = []
        if fine_tune:
            peak = np.zeros(self.WLS.size)
            peak[5:8] = [0.2, 0.5, 1.0]                # centroid right of 106 nm
            traces += [make_trace(self.WLS, [0.0] * self.WLS.size),
                       make_trace(self.WLS, peak.tolist())]
        traces += [make_trace(self.WLS, [0.0] * self.WLS.size),
                   make_trace(self.WLS, [1.0] * self.WLS.size)]
        traces += [make_trace(self.WLS, rng.uniform(0, 1, self.WLS.size).tolist())
                   for _ in range(4)]                  # 2 groups x 2 levels
        return traces

    def _by_hand(self, fine_tune: bool):
        step2 = _linear_map(12, 100.0, 1.0)
        osa = FakeNarrowOSA(self._traces(fine_tune))
        slm = FakeSLM(size=(12, 2))
        refined = coarse = measured = None
        if fine_tune:
            refined, measured, coarse = refine_center_coordinate_with_osa(
                osa, slm, MeasurementSettings(), step2,
                target_wavelength_nm=106.0, window_size=1, peak_half_window_nm=1.5,
            )
        seed, center = build_channel_calibration_grid(
            step2, center_coordinate=refined, slm_width=12, **self.GRID
        )
        result = batch_intensity_calibration(
            osa, slm, [400, 900], MeasurementSettings(), seed,
            window_size=1, group_skip_channels=1,
        )
        return result, center, coarse, measured

    def _run(self, fine_tune: bool, seen: list):
        return run_step3b(
            FakeNarrowOSA(self._traces(fine_tune)), FakeSLM(size=(12, 2)),
            [400, 900], MeasurementSettings(), _linear_map(12, 100.0, 1.0),
            group_skip_channels=1, fine_tune_center=fine_tune,
            peak_half_window_nm=1.5, progress_callback=seen.append, **self.GRID,
        )

    def test_fine_tuned_run_matches_refine_grid_batch_made_by_hand(self) -> None:
        seen = []
        sweep = self._run(True, seen)
        expected, center, coarse, measured = self._by_hand(True)

        self.assertAlmostEqual(sweep.center_coordinate, center)
        self.assertNotAlmostEqual(center, 6.0)          # the OSA moved it
        self.assertAlmostEqual(sweep.coarse_center, coarse)
        self.assertAlmostEqual(sweep.measured_peak_nm, measured)
        np.testing.assert_allclose(sweep.result.coordinates, expected.coordinates)
        np.testing.assert_allclose(sweep.result.intensity_levels, expected.intensity_levels)
        # refine reports the centre itself; no second grid-centre message
        self.assertEqual([p.phase for p in seen[:2]], ["fast_center", "fast_center"])
        self.assertEqual({p.phase for p in seen[2:]}, {"batch_intensity"})

    def test_untuned_run_tiles_on_the_step2_prediction_and_says_so(self) -> None:
        seen = []
        sweep = self._run(False, seen)
        expected, center, _coarse, _measured = self._by_hand(False)

        self.assertAlmostEqual(sweep.center_coordinate, 6.0)
        self.assertAlmostEqual(center, 6.0)
        self.assertIsNone(sweep.coarse_center)
        np.testing.assert_allclose(sweep.result.intensity_levels, expected.intensity_levels)
        self.assertEqual(seen[0].phase, "fast_center")
        self.assertIn("Step 2 predicts 106.0000 nm", seen[0].message)


if __name__ == "__main__":
    unittest.main()
