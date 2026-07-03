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
from slm_module.calibration.calibration import load_calibration_csv
from slm_module.calibration.calibration_new import (
    CalibrationAborted,
    CalibrationResult,
    intensity_calibration,
    load_calibration_result,
    load_wavelength_map_csv,
    local_peak_centroid,
    mean_near_wavelength,
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


if __name__ == "__main__":
    unittest.main()
