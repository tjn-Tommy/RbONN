from __future__ import annotations

import csv
import sys
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from osa_module.controller import MeasurementSettings, TraceData
from slm_module.calibration.calibration import intensity_model, load_calibration_csv
from slm_module.calibration.outliers import OutlierRemeasurePolicy
from slm_module.calibration.calibration_new import (
    CalibrationAborted,
    CalibrationResult,
    batch_intensity_calibration,
    build_channel_calibration_grid,
    intensity_calibration,
    intensity_calibration_daq,
    load_calibration_result,
    load_wavelength_map_csv,
    local_peak_centroid,
    local_peak_centroid_near,
    mean_near_wavelength,
    refine_center_coordinate_with_osa,
    restrict_to_measured_intensity_range,
    save_calibration_result,
    wavelength_calibration,
    write_intensity_calibration_csv,
)


class FakeOSA:
    def __init__(self, traces: list[TraceData]):
        self.traces = list(traces)
        self.measure_calls = 0

    def measure(self, settings: MeasurementSettings) -> TraceData:
        del settings
        if self.measure_calls >= len(self.traces):
            raise AssertionError("No trace left for FakeOSA")
        trace = self.traces[self.measure_calls]
        self.measure_calls += 1
        return trace


class FakeSLM:
    def __init__(self, size: tuple[int, int] = (5, 2)):
        self.size = size
        self.arrays: list[np.ndarray] = []

    def get_slm_info(self) -> tuple[int, int]:
        return self.size

    def display_array(self, arr: np.ndarray, interval: float = 0.2) -> None:
        del interval
        self.arrays.append(np.asarray(arr).copy())


class FakeNarrowOSA:
    """Fake supporting the narrow-sweep path: configure() + bare measure()."""

    def __init__(self, traces: list[TraceData]):
        self.traces = list(traces)
        self.measure_calls = 0
        self.configured: list[MeasurementSettings] = []
        self.measure_settings_seen: list[MeasurementSettings | None] = []

    def configure(self, settings: MeasurementSettings) -> None:
        self.configured.append(settings)

    def measure(self, settings: MeasurementSettings | None = None) -> TraceData:
        self.measure_settings_seen.append(settings)
        if self.measure_calls >= len(self.traces):
            raise AssertionError("No trace left for FakeNarrowOSA")
        trace = self.traces[self.measure_calls]
        self.measure_calls += 1
        return trace


def make_trace(wavelengths_nm: np.ndarray, powers_w: list[float]) -> TraceData:
    return TraceData(
        wavelengths=wavelengths_nm * 1e-9,
        powers=np.asarray(powers_w, dtype=float),
        trace_id="TRA",
        y_unit="LINear",
    )


class CalibrationNewTests(unittest.TestCase):
    def test_mean_near_wavelength_averages_neighbors(self) -> None:
        wavelengths = np.asarray([100.0, 101.0, 102.0, 103.0])
        intensity = np.asarray([1.0, 3.0, 5.0, 7.0])

        value = mean_near_wavelength(wavelengths, intensity, 101.2, half_window_points=1)

        self.assertEqual(value, 3.0)

    def test_local_peak_centroid_near_uses_target_window(self) -> None:
        wavelengths = np.asarray([100.0, 101.0, 102.0, 103.0])
        intensity = np.asarray([10.0, 1.0, 3.0, 1.0])

        center, idx, strength = local_peak_centroid_near(
            wavelengths,
            intensity,
            102.0,
            half_window_nm=0.6,
        )

        self.assertEqual(idx, 2)
        self.assertEqual(strength, 3.0)
        self.assertAlmostEqual(center, 102.0)

    def test_channel_calibration_grid_is_centered_on_target_gap(self) -> None:
        step2 = CalibrationResult(
            wavelength=np.asarray([783.0, 778.0, 773.0]),
            coordinates=np.asarray([0.0, 50.0, 100.0]),
            max_level=900,
            min_level=100,
            level_range=np.asarray([100, 900]),
        )

        grid, center = build_channel_calibration_grid(
            step2,
            target_wavelength_nm=778.0,
            n_channels_per_side=2,
            channel_width_px=15,
            gap_px=5,
            slm_width=120,
        )

        self.assertAlmostEqual(center, 50.0)
        np.testing.assert_allclose(grid.coordinates, np.asarray([20.0, 40.0, 60.0, 80.0]))
        np.testing.assert_allclose(grid.wavelength, np.asarray([781.0, 779.0, 777.0, 775.0]))
        np.testing.assert_allclose(grid.wavelength_fit_coefficients, np.asarray([-0.1, 783.0]))

        refined_grid, refined_center = build_channel_calibration_grid(
            step2,
            target_wavelength_nm=778.0,
            center_coordinate=52.0,
            n_channels_per_side=1,
            channel_width_px=15,
            gap_px=5,
            slm_width=120,
        )
        self.assertAlmostEqual(refined_center, 52.0)
        np.testing.assert_allclose(refined_grid.coordinates, np.asarray([42.0, 62.0]))
        np.testing.assert_allclose(refined_grid.wavelength, np.asarray([779.0, 777.0]))
        np.testing.assert_allclose(
            refined_grid.wavelength_fit_coefficients,
            np.asarray([-0.1, 783.2]),
        )

    def test_channel_calibration_grid_center_gap_matches_geometry(self) -> None:
        from slm_module.encoding import compute_channel_geometry

        step2 = CalibrationResult(
            wavelength=np.asarray([783.0, 778.0, 773.0]),
            coordinates=np.asarray([0.0, 50.0, 100.0]),
            max_level=900,
            min_level=100,
            level_range=np.asarray([100, 900]),
        )

        # first offset ceil((15+10)/2) = 13 -> coordinates center -/+ 13, 33
        grid, center = build_channel_calibration_grid(
            step2,
            target_wavelength_nm=778.0,
            n_channels_per_side=2,
            channel_width_px=15,
            gap_px=5,
            center_gap_px=10,
            slm_width=120,
        )
        self.assertAlmostEqual(center, 50.0)
        np.testing.assert_allclose(
            grid.coordinates, np.asarray([17.0, 37.0, 63.0, 83.0])
        )

        # the measured coordinates must coincide with the encoding layout's
        # channel centers built from the same map + center_gap_px
        geom = compute_channel_geometry(
            step2.coordinates, step2.wavelength,
            n_channels=2, channel_width_px=15, gap_px=5,
            center_gap_px=10, center_wl=778.0, dark_wl_bands=(),
        )
        geo_centers = sorted(
            [c.x_center for c in geom.x] + [c.x_center for c in geom.w]
        )
        np.testing.assert_allclose(grid.coordinates, np.asarray(geo_centers, dtype=float))

        # center_gap_px equal to gap_px reproduces the legacy half-pitch grid
        legacy_grid, _ = build_channel_calibration_grid(
            step2,
            target_wavelength_nm=778.0,
            n_channels_per_side=2,
            channel_width_px=15,
            gap_px=5,
            center_gap_px=5,
            slm_width=120,
        )
        np.testing.assert_allclose(
            legacy_grid.coordinates, np.asarray([20.0, 40.0, 60.0, 80.0])
        )

    def test_channel_calibration_grid_skips_guard_overlap_channels(self) -> None:
        step2 = CalibrationResult(
            wavelength=np.asarray([788.0, 778.0, 766.0]),
            coordinates=np.asarray([0.0, 100.0, 220.0]),
            max_level=900,
            min_level=100,
            level_range=np.asarray([100, 900]),
        )

        grid, center = build_channel_calibration_grid(
            step2,
            target_wavelength_nm=778.0,
            n_channels_per_side=2,
            channel_width_px=15,
            gap_px=5,
            slm_width=240,
            guard_bands_nm=[(779.0, 0.01)],
            # this test exercises the skip mechanics with a single guard, which
            # is intentionally asymmetric about the target
            require_symmetric_guard_bands=False,
        )

        self.assertAlmostEqual(center, 100.0)
        np.testing.assert_allclose(
            grid.coordinates,
            np.asarray([50.0, 70.0, 110.0, 130.0]),
        )
        self.assertNotIn(90.0, set(grid.coordinates.tolist()))

    def test_channel_calibration_grid_rejects_asymmetric_guard_bands(self) -> None:
        step2 = CalibrationResult(
            wavelength=np.asarray([788.0, 778.0, 766.0]),
            coordinates=np.asarray([0.0, 100.0, 220.0]),
            max_level=900,
            min_level=100,
            level_range=np.asarray([100, 900]),
        )

        with self.assertRaisesRegex(ValueError, "symmetric"):
            build_channel_calibration_grid(
                step2,
                target_wavelength_nm=778.0,
                n_channels_per_side=2,
                channel_width_px=15,
                gap_px=5,
                slm_width=240,
                guard_bands_nm=[(780.0, 0.06)],
            )

    def test_channel_calibration_grid_accepts_symmetric_guard_bands(self) -> None:
        step2 = CalibrationResult(
            wavelength=np.asarray([788.0, 778.0, 766.0]),
            coordinates=np.asarray([0.0, 100.0, 220.0]),
            max_level=900,
            min_level=100,
            level_range=np.asarray([100, 900]),
        )

        # 780 and 776 mirror each other about the 778 nm target
        grid, center = build_channel_calibration_grid(
            step2,
            target_wavelength_nm=778.0,
            n_channels_per_side=2,
            channel_width_px=15,
            gap_px=5,
            slm_width=240,
            guard_bands_nm=[(780.0, 0.06), (776.0, 0.06)],
        )

        self.assertAlmostEqual(center, 100.0)
        # left/right channel pairs stay at equal wavelength offset from 778 nm,
        # so the offset set is closed under negation
        offsets = grid.wavelength - 778.0
        np.testing.assert_allclose(np.sort(offsets), np.sort(-offsets))

    def test_refine_center_coordinate_with_osa_uses_linear_slope(self) -> None:
        step2 = CalibrationResult(
            wavelength=np.asarray([783.0, 778.0, 773.0]),
            coordinates=np.asarray([0.0, 50.0, 100.0]),
            max_level=900,
            min_level=100,
            level_range=np.asarray([100, 900]),
        )
        wavelengths = np.asarray([777.8, 778.0, 778.2, 778.4])
        traces = [
            make_trace(wavelengths, [0.0, 0.0, 0.0, 0.0]),
            make_trace(wavelengths, [0.0, 0.2, 1.0, 0.2]),
        ]
        osa = FakeNarrowOSA(traces)

        refined, measured, coarse = refine_center_coordinate_with_osa(
            osa,
            FakeSLM(size=(120, 2)),
            MeasurementSettings(),
            step2,
            target_wavelength_nm=778.0,
            window_size=15,
            peak_half_window_nm=0.4,
        )

        self.assertAlmostEqual(coarse, 50.0)
        self.assertAlmostEqual(measured, 778.2)
        self.assertAlmostEqual(refined, 52.0)
        self.assertEqual(len(osa.configured), 1)

    def test_batch_intensity_calibration_measures_multiple_channels_per_trace(self) -> None:
        wavelengths = np.asarray([100.0, 101.0, 102.0, 103.0, 104.0])
        traces = [
            make_trace(wavelengths, [0, 0, 0, 0, 0]),
            make_trace(wavelengths, [1, 1, 1, 1, 1]),
            make_trace(wavelengths, [0, 0.2, 0, 0.3, 0]),
            make_trace(wavelengths, [0, 0.6, 0, 0.7, 0]),
            make_trace(wavelengths, [0, 0, 0.4, 0, 0.5]),
            make_trace(wavelengths, [0, 0, 0.8, 0, 0.9]),
        ]
        osa = FakeNarrowOSA(traces)
        seed = CalibrationResult(
            wavelength=np.asarray([101.0, 102.0, 103.0, 104.0]),
            coordinates=np.asarray([1.0, 2.0, 3.0, 4.0]),
            max_level=100,
            min_level=0,
            level_range=np.asarray([0, 100]),
            wavelength_fit_coefficients=np.asarray([1.0, 100.0]),
        )

        result = batch_intensity_calibration(
            osa,
            FakeSLM(size=(6, 2)),
            [0, 100],
            MeasurementSettings(),
            seed,
            window_size=1,
            average_half_window=0,
            group_skip_channels=1,
        )

        expected = np.asarray(
            [
                [0.2, 0.6],
                [0.4, 0.8],
                [0.3, 0.7],
                [0.5, 0.9],
            ]
        )
        np.testing.assert_allclose(result.intensity_levels, expected)
        np.testing.assert_allclose(result.raw_intensity_levels, expected)
        self.assertEqual(osa.measure_calls, 6)  # 2 references + 2 groups * 2 levels
        self.assertEqual(len(osa.configured), 1)

    def test_batch_intensity_calibration_guard_bands_force_min_level(self) -> None:
        wavelengths = np.arange(100.0, 110.0)
        traces = [
            make_trace(wavelengths, [0.0] * wavelengths.size),
            make_trace(wavelengths, [1.0] * wavelengths.size),
            make_trace(wavelengths, [0.5] * wavelengths.size),
            make_trace(wavelengths, [0.7] * wavelengths.size),
        ]
        osa = FakeNarrowOSA(traces)
        slm = FakeSLM(size=(10, 2))
        seed = CalibrationResult(
            wavelength=np.asarray([103.0, 107.0]),
            coordinates=np.asarray([3.0, 7.0]),
            max_level=100,
            min_level=0,
            level_range=np.asarray([50]),
        )

        batch_intensity_calibration(
            osa,
            slm,
            [50],
            MeasurementSettings(),
            seed,
            window_size=3,
            average_half_window=0,
            group_skip_channels=1,
            guard_bands_nm=[(103.0, 0.01)],
        )

        bright = slm.arrays[1]
        self.assertTrue(np.all(bright[:, 3] == 0))
        self.assertEqual(int(bright[0, 2]), 100)
        self.assertEqual(int(bright[0, 4]), 100)

        active = slm.arrays[2]
        self.assertEqual(int(active[0, 2]), 50)
        self.assertEqual(int(active[0, 3]), 0)
        self.assertEqual(int(active[0, 4]), 50)

    def test_intensity_calibration_uses_calibrated_wavelength_neighborhood(self) -> None:
        wavelengths = np.asarray([100.0, 101.0, 102.0, 103.0, 104.0])
        traces = [
            make_trace(wavelengths, [0, 0, 0, 0, 0]),
            make_trace(wavelengths, [1, 1, 1, 1, 1]),
            make_trace(wavelengths, [0.1, 0.2, 0.3, 0.9, 0.9]),
            make_trace(wavelengths, [0.4, 0.6, 0.8, 0.9, 0.9]),
            make_trace(wavelengths, [0.9, 0.9, 0.2, 0.4, 0.6]),
            make_trace(wavelengths, [0.9, 0.9, 0.5, 0.7, 0.9]),
        ]
        osa = FakeOSA(traces)
        slm = FakeSLM(size=(5, 2))
        seed = CalibrationResult(
            wavelength=np.asarray([101.0, 103.0]),
            coordinates=np.asarray([1.0, 3.0]),
            max_level=100,
            min_level=0,
            level_range=np.asarray([0, 100]),
        )

        result = intensity_calibration(
            osa,
            slm,
            [0, 100],
            MeasurementSettings(),
            seed,
            window_size=2,
            average_half_window=1,
        )

        np.testing.assert_allclose(
            result.intensity_levels,
            np.asarray([[0.2, 0.6], [0.4, 0.7]]),
        )
        np.testing.assert_array_equal(result.level_range, np.asarray([0, 100]))
        self.assertEqual(osa.measure_calls, 6)

    def test_intensity_calibration_narrow_span_recenters_per_coordinate(self) -> None:
        wide = np.asarray([100.0, 101.0, 102.0, 103.0, 104.0])
        coord0 = np.asarray([100.5, 101.0, 101.5])
        coord1 = np.asarray([102.5, 103.0, 103.5])
        traces = [
            make_trace(wide, [0.1, 0.1, 0.1, 0.1, 0.1]),  # background (wide)
            make_trace(wide, [2.1, 2.1, 2.1, 2.1, 2.1]),  # reference (wide) -> denom 2.0
            make_trace(coord0, [0.5, 0.5, 0.5]),  # coord 0 narrow signal -> raw 0.4
            make_trace(coord1, [0.9, 0.9, 0.9]),  # coord 1 narrow signal -> raw 0.8
        ]
        osa = FakeNarrowOSA(traces)
        slm = FakeSLM(size=(5, 2))
        seed = CalibrationResult(
            wavelength=np.asarray([101.0, 103.0]),
            coordinates=np.asarray([1.0, 3.0]),
            max_level=100,
            min_level=0,
            level_range=np.asarray([200]),
        )

        result = intensity_calibration(
            osa,
            slm,
            [200],
            MeasurementSettings(),
            seed,
            window_size=2,
            average_half_window=1,
            sweep_span_nm=0.5,
        )

        np.testing.assert_allclose(result.raw_intensity_levels, np.asarray([[0.4], [0.8]]))
        np.testing.assert_allclose(result.intensity_levels, np.asarray([[0.2], [0.4]]))
        # one narrow re-center per coordinate, on its Step-2 wavelength
        self.assertEqual(
            [s.center_wl for s in osa.configured], ["101.0000nm", "103.0000nm"]
        )
        self.assertEqual([s.span for s in osa.configured], ["0.5nm", "0.5nm"])
        # references use the wide settings; signal sweeps reuse the per-coordinate
        # config (no settings passed to measure())
        self.assertEqual(
            osa.measure_settings_seen,
            [MeasurementSettings(), MeasurementSettings(), None, None],
        )

    def test_intensity_calibration_stride_skips_coordinates(self) -> None:
        wide = np.asarray([100.0, 101.0, 102.0, 103.0, 104.0])
        traces = [
            make_trace(wide, [0.0, 0.0, 0.0, 0.0, 0.0]),  # background (wide)
            make_trace(wide, [1.0, 1.0, 1.0, 1.0, 1.0]),  # reference (wide)
            make_trace(np.asarray([100.8, 101.0, 101.2]), [0.4, 0.5, 0.4]),  # coord 1
            make_trace(np.asarray([102.8, 103.0, 103.2]), [0.6, 0.7, 0.6]),  # coord 3
        ]
        osa = FakeNarrowOSA(traces)
        seed = CalibrationResult(
            wavelength=np.asarray([101.0, 102.0, 103.0, 104.0]),
            coordinates=np.asarray([1.0, 2.0, 3.0, 4.0]),
            max_level=100,
            min_level=0,
            level_range=np.asarray([200]),
        )

        result = intensity_calibration(
            osa,
            FakeSLM(size=(5, 2)),
            [200],
            MeasurementSettings(),
            seed,
            window_size=1,
            average_half_window=1,
            sweep_span_nm=0.5,
            coordinate_stride=2,
        )

        np.testing.assert_array_equal(result.coordinates, np.asarray([1.0, 3.0]))
        np.testing.assert_array_equal(result.wavelength, np.asarray([101.0, 103.0]))
        self.assertEqual(result.intensity_levels.shape, (2, 1))
        self.assertEqual(
            [s.center_wl for s in osa.configured], ["101.0000nm", "103.0000nm"]
        )
        self.assertEqual(osa.measure_calls, 4)  # 2 references + 2 strided coordinates

    def test_intensity_calibration_refines_wavelength_from_narrow_peak(self) -> None:
        wide = np.asarray([100.0, 101.0, 102.0])
        narrow = np.asarray([100.8, 101.0, 101.2, 101.4, 101.6])
        traces = [
            make_trace(wide, [0.0, 0.0, 0.0]),  # background (wide)
            make_trace(wide, [1.0, 1.0, 1.0]),  # reference (wide) -> denom 1.0
            make_trace(narrow, [0.1, 0.3, 0.6, 1.0, 0.5]),  # narrow peak near 101.4
        ]
        osa = FakeNarrowOSA(traces)
        seed = CalibrationResult(
            wavelength=np.asarray([101.0]),  # Step 2's coarse estimate
            coordinates=np.asarray([1.0]),
            max_level=100,
            min_level=0,
            level_range=np.asarray([200]),
        )

        result = intensity_calibration(
            osa,
            FakeSLM(size=(3, 2)),
            [200],
            MeasurementSettings(),
            seed,
            window_size=1,
            average_half_window=1,
            sweep_span_nm=1.0,
            refine_wavelength=True,
        )

        # centroid over |λ - 101.4| <= 0.5 of [0.3,0.6,1.0,0.5] (min-subtracted)
        np.testing.assert_allclose(result.wavelength, np.asarray([101.383333]), atol=1e-4)
        self.assertIsNotNone(result.wavelength_fit_coefficients)

    def test_intensity_calibration_rejects_non_positive_sweep_span(self) -> None:
        wavelengths = np.asarray([100.0, 101.0, 102.0])
        traces = [make_trace(wavelengths, [0.1, 0.1, 0.1])] * 4
        seed = CalibrationResult(
            wavelength=np.asarray([101.0]),
            coordinates=np.asarray([1.0]),
            max_level=100,
            min_level=0,
            level_range=np.asarray([200]),
        )
        with self.assertRaises(ValueError):
            intensity_calibration(
                FakeNarrowOSA(traces),
                FakeSLM(size=(3, 2)),
                [200],
                MeasurementSettings(),
                seed,
                window_size=1,
                sweep_span_nm=0.0,
            )

    def test_intensity_calibration_keeps_raw_and_normalized_maps(self) -> None:
        wavelengths = np.asarray([100.0, 101.0, 102.0])
        traces = [
            make_trace(wavelengths, [0.1, 0.1, 0.1]),  # background
            make_trace(wavelengths, [2.1, 2.1, 2.1]),  # reference -> denom 2.0
            make_trace(wavelengths, [0.5, 0.5, 0.5]),  # one level measurement
        ]
        osa = FakeOSA(traces)
        slm = FakeSLM(size=(3, 2))
        seed = CalibrationResult(
            wavelength=np.asarray([101.0]),
            coordinates=np.asarray([1.0]),
            max_level=100,
            min_level=0,
            level_range=np.asarray([200]),
        )

        result = intensity_calibration(
            osa,
            slm,
            [200],
            MeasurementSettings(),
            seed,
            window_size=1,
            average_half_window=0,
        )

        # raw = power - background (0.5 - 0.1); normalized = raw / (2.1 - 0.1)
        np.testing.assert_allclose(result.raw_intensity_levels, np.asarray([[0.4]]))
        np.testing.assert_allclose(result.intensity_levels, np.asarray([[0.2]]))

    def test_intensity_calibration_aborts_on_stop_event(self) -> None:
        wavelengths = np.asarray([100.0, 101.0])
        traces = [
            make_trace(wavelengths, [0.0, 0.0]),
            make_trace(wavelengths, [1.0, 1.0]),
            make_trace(wavelengths, [0.5, 0.5]),
        ]
        osa = FakeOSA(traces)
        slm = FakeSLM(size=(2, 2))
        seed = CalibrationResult(
            wavelength=np.asarray([100.0]),
            coordinates=np.asarray([0.0]),
            max_level=100,
            min_level=0,
            level_range=np.asarray([0, 100]),
        )
        stop_event = threading.Event()
        stop_event.set()

        with self.assertRaises(CalibrationAborted):
            intensity_calibration(
                osa,
                slm,
                [0, 100],
                MeasurementSettings(),
                seed,
                window_size=1,
                stop_event=stop_event,
            )

    def test_write_intensity_calibration_csv_includes_raw_column(self) -> None:
        result = CalibrationResult(
            wavelength=np.asarray([101.0]),
            coordinates=np.asarray([1.0]),
            max_level=100,
            min_level=0,
            level_range=np.asarray([0, 100]),
            intensity_levels=np.asarray([[0.2, 0.6]]),
            raw_intensity_levels=np.asarray([[0.4, 1.2]]),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_intensity_calibration_csv(result, Path(temp_dir) / "cal.csv")
            with open(path, encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertIn("raw_intensity_w", rows[0])
        self.assertAlmostEqual(float(rows[0]["raw_intensity_w"]), 0.4)
        self.assertAlmostEqual(float(rows[1]["raw_intensity_w"]), 1.2)

    def test_local_peak_centroid_selects_window_in_nm(self) -> None:
        wavelengths = np.asarray([769.0, 770.0, 771.0, 772.0, 773.0, 774.0])
        intensity = np.asarray([0.0, 0.1, 0.4, 1.0, 0.4, 0.1])

        center, _, _ = local_peak_centroid(wavelengths, intensity, half_window_nm=1.0)

        # the +/- 1 nm window keeps the centroid tight on the true peak (772 nm)
        self.assertAlmostEqual(center, 772.0, places=6)
        with self.assertRaises(ValueError):
            local_peak_centroid(wavelengths, intensity, half_window_nm=0.0)

    def test_wavelength_calibration_runs_from_manual_seed(self) -> None:
        wavelengths = np.asarray([769.0, 770.0, 771.0, 772.0, 773.0])
        # background, reference, then one trace per window position (width 5, win 2 -> 4)
        traces = [
            make_trace(wavelengths, [0, 0, 0, 0, 0]),
            make_trace(wavelengths, [1, 1, 1, 1, 1]),
        ] + [make_trace(wavelengths, [0.1, 0.5, 1.0, 0.5, 0.1]) for _ in range(4)]
        osa = FakeOSA(traces)
        slm = FakeSLM(size=(5, 2))
        # no Step 1: seed only carries min/max levels
        seed = CalibrationResult(
            wavelength=np.asarray([]),
            coordinates=np.asarray([]),
            max_level=100,
            min_level=0,
            level_range=np.asarray([], dtype=int),
        )

        result = wavelength_calibration(
            osa,
            slm,
            [],
            MeasurementSettings(),
            seed,
            window_size=2,
            peak_half_window_nm=1.0,
        )

        self.assertEqual(result.coordinates.size, 4)
        self.assertTrue(np.all(np.isfinite(result.wavelength)))

    def test_wavelength_calibration_region_limits_the_sweep(self) -> None:
        wavelengths = np.asarray([769.0, 770.0, 771.0, 772.0, 773.0])
        peak = [0.1, 0.5, 1.0, 0.5, 0.1]
        # width 20, window 2, region (5, 10) -> window starts 5..9 = 5 positions
        traces = [
            make_trace(wavelengths, [0, 0, 0, 0, 0]),
            make_trace(wavelengths, [1, 1, 1, 1, 1]),
        ] + [make_trace(wavelengths, peak) for _ in range(5)]
        osa = FakeOSA(traces)
        slm = FakeSLM(size=(20, 2))
        seed = CalibrationResult(
            wavelength=np.asarray([]),
            coordinates=np.asarray([]),
            max_level=100,
            min_level=0,
            level_range=np.asarray([], dtype=int),
        )

        result = wavelength_calibration(
            osa, slm, [], MeasurementSettings(), seed, window_size=2, region=(5, 10)
        )

        self.assertEqual(result.coordinates.size, 5)
        self.assertGreaterEqual(result.coordinates.min(), 5)
        self.assertLessEqual(result.coordinates.max(), 10)
        self.assertEqual(osa.measure_calls, 7)  # background + reference + 5 positions

    def test_wavelength_calibration_stride_fills_skipped_columns(self) -> None:
        # linear optical mapping: window start s peaks at (700 + s + 1) nm,
        # i.e. wavelength = 700 + coordinate for window_size 2
        axis = np.arange(700.0, 740.0)
        n_samples = axis.size

        def peak_trace(start: int) -> TraceData:
            powers = np.full(n_samples, 0.1)
            powers[start + 1] = 1.0
            return make_trace(axis, list(powers))

        # width 20, window 2 -> starts 0..18; stride 5 measures 0,5,10,15
        # plus the appended far-edge anchor 18
        measured_starts = [0, 5, 10, 15, 18]
        traces = [
            make_trace(axis, [0.0] * n_samples),
            make_trace(axis, [1.0] * n_samples),
        ] + [peak_trace(start) for start in measured_starts]
        osa = FakeOSA(traces)
        slm = FakeSLM(size=(20, 2))
        seed = CalibrationResult(
            wavelength=np.asarray([]),
            coordinates=np.asarray([]),
            max_level=100,
            min_level=0,
            level_range=np.asarray([], dtype=int),
        )

        result = wavelength_calibration(
            osa,
            slm,
            [],
            MeasurementSettings(),
            seed,
            window_size=2,
            peak_half_window_nm=1.0,
            coordinate_stride=5,
        )

        self.assertEqual(osa.measure_calls, 2 + len(measured_starts))
        # the fit fills every skipped column: same dense grid as stride 1
        np.testing.assert_allclose(result.coordinates, np.arange(19) + 1.0)
        np.testing.assert_allclose(
            result.wavelength, 700.0 + result.coordinates, atol=1e-6
        )

    def test_wavelength_calibration_rejects_bad_stride(self) -> None:
        seed = CalibrationResult(
            wavelength=np.asarray([]),
            coordinates=np.asarray([]),
            max_level=100,
            min_level=0,
            level_range=np.asarray([], dtype=int),
        )
        with self.assertRaisesRegex(ValueError, "coordinate_stride"):
            wavelength_calibration(
                FakeOSA([]),
                FakeSLM(size=(20, 2)),
                [],
                MeasurementSettings(),
                seed,
                window_size=2,
                coordinate_stride=0,
            )

    def test_intensity_calibration_region_filters_loaded_mapping(self) -> None:
        wavelengths = np.asarray([100.0, 101.0, 102.0, 103.0, 104.0])
        # mapping spans the SLM; only coordinates 6 and 8 are inside region (5, 10)
        mapping = CalibrationResult(
            wavelength=np.asarray([100.0, 101.0, 102.0, 103.0]),
            coordinates=np.asarray([2.0, 6.0, 8.0, 14.0]),
            max_level=100,
            min_level=0,
            level_range=np.asarray([0, 100]),
        )
        traces = [
            make_trace(wavelengths, [0, 0, 0, 0, 0]),
            make_trace(wavelengths, [1, 1, 1, 1, 1]),
        ] + [make_trace(wavelengths, [0.2, 0.4, 0.6, 0.4, 0.2]) for _ in range(4)]
        osa = FakeOSA(traces)
        slm = FakeSLM(size=(20, 2))

        result = intensity_calibration(
            osa, slm, [0, 100], MeasurementSettings(), mapping,
            window_size=2, region=(5, 10),
        )

        self.assertEqual(sorted(result.coordinates.tolist()), [6.0, 8.0])
        self.assertEqual(result.intensity_levels.shape, (2, 2))

    def test_load_wavelength_map_csv_drives_intensity_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "map.csv"
            with open(csv_path, "w", encoding="utf-8", newline="") as handle:
                handle.write("coordinate_px,wavelength_nm\n1,101.0\n")
            mapping = load_wavelength_map_csv(
                csv_path, min_level=0, max_level=100, level_range=[0, 100]
            )

        self.assertEqual(mapping.coordinates.size, 1)
        self.assertEqual(mapping.min_level, 0)
        self.assertEqual(mapping.max_level, 100)

        wavelengths = np.asarray([100.0, 101.0, 102.0, 103.0, 104.0])
        traces = [
            make_trace(wavelengths, [0, 0, 0, 0, 0]),
            make_trace(wavelengths, [1, 1, 1, 1, 1]),
            make_trace(wavelengths, [0.1, 0.2, 0.3, 0.9, 0.9]),
            make_trace(wavelengths, [0.4, 0.6, 0.8, 0.9, 0.9]),
        ]
        osa = FakeOSA(traces)
        slm = FakeSLM(size=(5, 2))

        result = intensity_calibration(
            osa,
            slm,
            [0, 100],
            MeasurementSettings(),
            mapping,
            window_size=2,
            average_half_window=1,
        )

        np.testing.assert_allclose(result.intensity_levels, np.asarray([[0.2, 0.6]]))

    def test_save_load_calibration_result_round_trip(self) -> None:
        result = CalibrationResult(
            wavelength=np.asarray([770.0, 772.0]),
            coordinates=np.asarray([1.0, 3.0]),
            max_level=900,
            min_level=20,
            level_range=np.asarray([0, 256, 1023]),
            intensity_levels=np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            raw_intensity_levels=np.asarray([[1e-6, 2e-6, 3e-6], [4e-6, 5e-6, 6e-6]]),
            wavelength_fit_coefficients=np.asarray([1.0, 2.0, 3.0, 4.0]),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = save_calibration_result(result, Path(temp_dir) / "step.json")
            loaded = load_calibration_result(path)

        np.testing.assert_allclose(loaded.wavelength, result.wavelength)
        np.testing.assert_allclose(loaded.coordinates, result.coordinates)
        np.testing.assert_allclose(loaded.raw_intensity_levels, result.raw_intensity_levels)
        np.testing.assert_allclose(
            loaded.wavelength_fit_coefficients, result.wavelength_fit_coefficients
        )
        self.assertEqual(loaded.min_level, 20)
        self.assertEqual(loaded.max_level, 900)

    def test_restricts_quick_calibration_to_measured_min_max_range(self) -> None:
        result = CalibrationResult(
            wavelength=np.asarray([778.0]),
            coordinates=np.asarray([100.0]),
            max_level=950,
            min_level=300,
            level_range=np.asarray([300, 420, 500, 870, 950]),
            intensity_levels=np.asarray([[0.2, 0.1, 0.4, 1.1, 0.8]]),
            raw_intensity_levels=np.asarray([[2.0, 1.0, 4.0, 11.0, 8.0]]),
        )

        restricted = restrict_to_measured_intensity_range(result)

        self.assertEqual(restricted.min_level, 420)
        self.assertEqual(restricted.max_level, 870)
        np.testing.assert_array_equal(restricted.level_range, [420, 500, 870])
        np.testing.assert_allclose(restricted.intensity_levels, [[0.0, 0.3, 1.0]])
        np.testing.assert_allclose(
            restricted.raw_intensity_levels, [[1.0, 4.0, 11.0]]
        )

    def test_rejects_zero_quick_calibration_range(self) -> None:
        result = CalibrationResult(
            wavelength=np.asarray([778.0]),
            coordinates=np.asarray([100.0]),
            max_level=870,
            min_level=420,
            level_range=np.asarray([420, 870]),
            intensity_levels=np.asarray([[0.5, 0.5]]),
        )
        with self.assertRaisesRegex(ValueError, "maximum must occur"):
            restrict_to_measured_intensity_range(result)

    def test_load_calibration_result_preserves_none_fields(self) -> None:
        seed = CalibrationResult(
            wavelength=np.asarray([]),
            coordinates=np.asarray([]),
            max_level=5,
            min_level=0,
            level_range=np.asarray([0, 5]),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = save_calibration_result(seed, Path(temp_dir) / "seed.json")
            loaded = load_calibration_result(path)

        self.assertIsNone(loaded.intensity_levels)
        self.assertIsNone(loaded.raw_intensity_levels)
        self.assertIsNone(loaded.wavelength_fit_coefficients)

    def test_write_intensity_calibration_csv_matches_legacy_loader(self) -> None:
        result = CalibrationResult(
            wavelength=np.asarray([101.0]),
            coordinates=np.asarray([1.0]),
            max_level=100,
            min_level=0,
            level_range=np.asarray([0, 100]),
            intensity_levels=np.asarray([[0.2, 0.6]]),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_intensity_calibration_csv(
                result, Path(temp_dir) / "calibration.csv"
            )
            points = load_calibration_csv(path)

        self.assertEqual(len(points), 2)
        self.assertEqual(points[0].wavelength_nm, 101.0)
        self.assertEqual(points[1].level, 100)
        self.assertEqual(points[1].intensity, 0.6)


class _MonitorSample:
    def __init__(self, value: float):
        self.value = float(value)
        self.std = 0.0
        self.index = 0
        self.timestamp = 0.0


class _BucketDAQ:
    """DAQ-shaped fake: total 'power' summed over the pixels the SLM shows.

    Each pixel contributes a per-pixel sin^2-in-level transfer plus a small dark
    leakage, so the full-panel dark reference and the per-window sweep behave like
    a real bucket detector reading whatever pattern is on the SLM.
    """

    def __init__(self, slm: FakeSLM):
        self.slm = slm
        self.reads = 0

    def _transfer(self, levels: np.ndarray) -> np.ndarray:
        return 0.002 + 0.010 * np.sin(np.pi * (levels - 400) / 1000.0) ** 2

    def monitor_cycle(self, *, index=0, timeout=30.0, stop_event=None):
        if stop_event is not None and stop_event.is_set():
            return None
        self.reads += 1
        row = np.asarray(self.slm.arrays[-1], dtype=float)[0]
        return _MonitorSample(float(np.sum(self._transfer(row))))


class IntensityCalibrationDaqTests(unittest.TestCase):
    def _seed(self) -> CalibrationResult:
        coords = np.asarray([40.0, 100.0, 160.0])
        return CalibrationResult(
            wavelength=np.asarray([775.0, 778.0, 781.0]),
            coordinates=coords,
            max_level=900,
            min_level=400,
            level_range=np.asarray([], dtype=int),
            wavelength_fit_coefficients=np.polyfit(coords, [775.0, 778.0, 781.0], 1),
        )

    def test_shapes_reads_and_dark_frame_subtraction(self) -> None:
        slm = FakeSLM(size=(200, 4))
        daq = _BucketDAQ(slm)
        levels = list(range(400, 901, 50))

        result = intensity_calibration_daq(
            daq, slm, levels, self._seed(), window_size=8,
        )

        self.assertEqual(result.intensity_levels.shape, (3, len(levels)))
        self.assertEqual(result.raw_intensity_levels.shape, (3, len(levels)))
        # one dark ref only (no bright frame) + n_coords * n_levels window reads
        self.assertEqual(daq.reads, 1 + 3 * len(levels))
        # dark-frame subtraction: raw is >= 0 and the darkest level ~ 0 above dark
        self.assertTrue(np.all(result.raw_intensity_levels >= 0.0))
        self.assertAlmostEqual(result.raw_intensity_levels[0, 0], 0.0, places=6)
        # no bright normalization: intensity is exactly the dark-subtracted signal
        np.testing.assert_allclose(
            result.intensity_levels, result.raw_intensity_levels
        )
        # sin^2 shape preserved: the curve rises off the dark end
        row = result.raw_intensity_levels[1]
        self.assertLess(row[0], row[-1])
        # wavelength map is passed through unchanged (no OSA refine)
        np.testing.assert_allclose(result.wavelength, [775.0, 778.0, 781.0])

    def test_region_and_stride_select_coordinates(self) -> None:
        slm = FakeSLM(size=(200, 4))
        daq = _BucketDAQ(slm)
        result = intensity_calibration_daq(
            daq, slm, [400, 600, 900], self._seed(),
            window_size=8, coordinate_stride=2, region=(30, 130),
        )
        # region keeps coords 40 and 100; stride 2 then keeps coord 40 only
        np.testing.assert_allclose(result.coordinates, [40.0])
        self.assertEqual(result.intensity_levels.shape, (1, 3))

    def test_stop_event_aborts(self) -> None:
        slm = FakeSLM(size=(200, 4))
        daq = _BucketDAQ(slm)
        stop = threading.Event()
        stop.set()
        with self.assertRaises(CalibrationAborted):
            intensity_calibration_daq(
                daq, slm, [400, 600, 900], self._seed(), window_size=8,
                stop_event=stop,
            )


class _SimWindowOSA:
    """Deterministic step-2 sim: the trace peak position follows the displayed
    bright window, so re-measurements are consistent no matter the order.

    Window start i (0..10) -> delta peak at trace index 10 + 8*i (wl 710+8i nm,
    linear in coordinate). The FIRST measurement of window start
    ``bogus_start`` instead peaks at index 90 (wl 790) -- the injected outlier.
    """

    WLS = np.arange(700.0, 801.0)  # 101 samples, 1 nm apart

    def __init__(self, slm: FakeSLM, bogus_start: int):
        self.slm = slm
        self.bogus_start = bogus_start
        self.bogus_pending = True
        self.measure_calls = 0

    def measure(self, settings: MeasurementSettings) -> TraceData:
        del settings
        self.measure_calls += 1
        row = np.asarray(self.slm.arrays[-1])[0]
        powers = np.zeros(self.WLS.size)
        if np.all(row == 0):                      # dark background reference
            pass
        elif np.all(row == row.max()) and row.max() > 0:   # bright reference
            powers[:] = 1.0
        else:
            start = int(np.flatnonzero(row == row.max())[0])
            index = 10 + 8 * start
            if start == self.bogus_start and self.bogus_pending:
                self.bogus_pending = False
                index = 90                        # bogus centroid, once
            powers[index] = 1.0
        return make_trace(self.WLS, powers.tolist())


class _SimTransferOSA:
    """Deterministic step-3 sim: the channel value follows the displayed level
    through the sin^2 transfer model. The FIRST measurement at ``bogus_level``
    returns a spiked value instead -- the injected outlier.
    """

    WLS = np.asarray([100.0, 101.0, 102.0, 103.0, 104.0])

    def __init__(self, slm: FakeSLM, column: int, max_level: int,
                 bogus_level: int, bogus_value: float):
        self.slm = slm
        self.column = column
        self.max_level = max_level
        self.bogus_level = bogus_level
        self.bogus_value = bogus_value
        self.bogus_pending = True
        self.measure_calls = 0

    def configure(self, settings: MeasurementSettings) -> None:
        pass

    def measure(self, settings: MeasurementSettings | None = None) -> TraceData:
        del settings
        self.measure_calls += 1
        row = np.asarray(self.slm.arrays[-1])[0]
        powers = np.zeros(self.WLS.size)
        if np.all(row == 0):                      # dark background reference
            pass
        elif np.all(row == self.max_level):       # bright reference
            powers[:] = 1.0
        else:
            level = int(row[self.column])
            value = float(
                intensity_model(level, 1.0, np.pi / self.max_level, 0.0)
            )
            if level == self.bogus_level and self.bogus_pending:
                self.bogus_pending = False
                value = self.bogus_value
            powers[2] = value                     # channel wavelength = 102 nm
        return make_trace(self.WLS, powers.tolist())


class OutlierRemeasureTests(unittest.TestCase):
    POLICY = OutlierRemeasurePolicy(k_sigma=4.0, max_retries=3, min_points=8)

    def _step2_seed(self) -> CalibrationResult:
        return CalibrationResult(
            wavelength=np.asarray([]),
            coordinates=np.asarray([]),
            max_level=100,
            min_level=0,
            level_range=np.asarray([], dtype=int),
        )

    def test_wavelength_outlier_is_remeasured_and_replaced(self) -> None:
        # width 12, window 2 -> 11 window starts; start 5 is bogus once
        slm = FakeSLM(size=(12, 2))
        osa = _SimWindowOSA(slm, bogus_start=5)

        result = wavelength_calibration(
            osa, slm, [], MeasurementSettings(), self._step2_seed(),
            window_size=2, peak_half_window_nm=1.0,
            outlier_policy=self.POLICY,
        )

        # bg + ref + 11 sweep + 1 recheck of the flagged point
        self.assertEqual(osa.measure_calls, 14)
        # measured points are exactly linear after replacement: wl = 8*x + 702
        # (the mapping fit is cubic, so the higher-order terms collapse to ~0)
        coeffs = result.wavelength_fit_coefficients
        self.assertAlmostEqual(float(coeffs[-2]), 8.0, places=6)
        self.assertAlmostEqual(float(coeffs[-1]), 702.0, places=4)
        self.assertAlmostEqual(float(result.wavelength[5]), 750.0, places=4)

    def test_wavelength_no_policy_keeps_outlier_and_call_count(self) -> None:
        slm = FakeSLM(size=(12, 2))
        osa = _SimWindowOSA(slm, bogus_start=5)

        result = wavelength_calibration(
            osa, slm, [], MeasurementSettings(), self._step2_seed(),
            window_size=2, peak_half_window_nm=1.0,
        )

        self.assertEqual(osa.measure_calls, 13)   # no recheck reads
        # the bogus point drags the fitted curve well away from the clean 750
        self.assertGreater(abs(float(result.wavelength[5]) - 750.0), 5.0)

    def _step3_seed(self) -> CalibrationResult:
        return CalibrationResult(
            wavelength=np.asarray([102.0]),
            coordinates=np.asarray([2.0]),
            max_level=1000,
            min_level=0,
            level_range=np.asarray([0, 1000]),
            wavelength_fit_coefficients=np.asarray([1.0, 100.0]),
        )

    def test_batch_intensity_outlier_is_remeasured_and_replaced(self) -> None:
        levels = list(range(0, 1100, 100))        # 11 levels
        slm = FakeSLM(size=(5, 2))
        osa = _SimTransferOSA(slm, column=2, max_level=1000,
                              bogus_level=500, bogus_value=0.9)

        result = batch_intensity_calibration(
            osa, slm, levels, MeasurementSettings(), self._step3_seed(),
            window_size=1, average_half_window=0,
            outlier_policy=self.POLICY,
        )

        expected = intensity_model(
            np.asarray(levels, dtype=float), 1.0, np.pi / 1000.0, 0.0
        )
        np.testing.assert_allclose(result.intensity_levels[0], expected, atol=1e-9)
        # bg + ref + 11 sweep + at least the one recheck read
        self.assertGreaterEqual(osa.measure_calls, 14)
        self.assertFalse(osa.bogus_pending)       # the spike was re-measured

    def test_batch_intensity_no_policy_keeps_outlier(self) -> None:
        levels = list(range(0, 1100, 100))
        slm = FakeSLM(size=(5, 2))
        osa = _SimTransferOSA(slm, column=2, max_level=1000,
                              bogus_level=500, bogus_value=0.9)

        result = batch_intensity_calibration(
            osa, slm, levels, MeasurementSettings(), self._step3_seed(),
            window_size=1, average_half_window=0,
        )

        self.assertEqual(osa.measure_calls, 13)
        self.assertAlmostEqual(float(result.intensity_levels[0, 5]), 0.9)


if __name__ == "__main__":
    unittest.main()
