import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _make_calibration():
    from slm_module.calibration.calibration_new import CalibrationResult

    coords = np.array([0.0, 50.0, 100.0, 150.0, 200.0], dtype=float)
    wavelengths = np.array([779.0, 778.5, 778.0, 777.5, 777.0], dtype=float)
    levels = np.array([0, 512, 1023], dtype=int)
    curve = np.array([0.0, 0.5, 1.0], dtype=float)
    intensity = np.tile(curve, (coords.size, 1))
    return CalibrationResult(
        wavelength=wavelengths,
        coordinates=coords,
        max_level=1023,
        min_level=0,
        level_range=levels,
        intensity_levels=intensity,
    )


class FitCenterTraceTests(unittest.TestCase):
    def test_recovers_peak_inside_window(self) -> None:
        from slm_module.tpa_center import fit_center_trace

        wl = np.array([777.9, 778.0, 778.1, 778.2, 778.3], dtype=float)
        signal = 0.004 - 0.1 * (wl - 778.1) ** 2
        sem = np.full(wl.shape, 1e-4, dtype=float)
        fit = fit_center_trace(wl, signal, sem)

        self.assertTrue(fit.valid)
        self.assertAlmostEqual(fit.center_wl_nm, 778.1, places=3)
        self.assertAlmostEqual(fit.peak_signal_v, 0.004, places=5)

    def test_marks_peak_outside_window_invalid(self) -> None:
        from slm_module.tpa_center import fit_center_trace

        wl = np.array([777.9, 778.0, 778.1, 778.2, 778.3], dtype=float)
        signal = 0.004 - 0.1 * (wl - 778.45) ** 2
        sem = np.full(wl.shape, 1e-4, dtype=float)
        fit = fit_center_trace(wl, signal, sem)

        self.assertFalse(fit.valid)
        self.assertIn("outside", fit.message)
        self.assertAlmostEqual(fit.best_sample_center_wl_nm, 778.3, places=3)


class _FakeSLM:
    def get_slm_info(self):
        return (256, 4)

    def display_array(self, pattern):
        self.last = np.asarray(pattern)


class _FakeMonitor:
    last_values = None

    def __init__(self, values):
        self._values = list(values)

    def configure_monitor(self, *args, **kwargs) -> None:
        pass

    def monitor_cycle(self, timeout=30.0, **kwargs):
        if not self._values:
            raise AssertionError("monitor read sequence exhausted")

        class _Sample:
            def __init__(self, value):
                self.value = value
                self.std = 0.0

        return _Sample(self._values.pop(0))


class MeasureCenterScanTests(unittest.TestCase):
    def test_scan_forwards_col_ratio_and_fits_peak(self) -> None:
        from slm_module import encoding as encoding_module
        from slm_module import tpa_center as tpa_center_module

        calib = _make_calibration()
        centers = np.array([777.95, 778.00, 778.05, 778.10, 778.15], dtype=float)
        ratio = np.linspace(0.8, 1.0, 15)
        bg = 0.001
        net = 0.004 - 0.1 * (centers - 778.1) ** 2
        reads: list[float] = []
        for value in net:
            reads.extend([bg, bg + float(value)])

        seen = []
        real = encoding_module.encode_to_pattern

        def recorder(*args, **kwargs):
            seen.append(kwargs.get("col_ratio"))
            return real(*args, **kwargs)

        with mock.patch.object(encoding_module, "encode_to_pattern", recorder):
            result = tpa_center_module.measure_center_scan(
                _FakeMonitor(reads),
                _FakeSLM(),
                calib,
                center_wavelengths_nm=centers,
                n_channels=1,
                channel_width_px=15,
                gap_px=5,
                pair_index=0,
                n_trials=1,
                repeats=1,
                settle=0.0,
                col_ratio=ratio,
                subtract_background=True,
            )

        self.assertEqual(result.center_wl_nm.size, centers.size)
        self.assertTrue(result.fit is not None and result.fit.valid)
        self.assertAlmostEqual(result.fit.center_wl_nm, 778.1, places=3)
        self.assertTrue(seen)
        self.assertTrue(all(c is ratio for c in seen))


class SaveLoadJsonTests(unittest.TestCase):
    def _scan_result(self):
        from slm_module import tpa_center as tpa_center_module

        calib = _make_calibration()
        centers = np.array([777.95, 778.00, 778.05, 778.10, 778.15], dtype=float)
        bg = 0.001
        net = 0.004 - 0.1 * (centers - 778.1) ** 2
        reads: list[float] = []
        for value in net:
            reads.extend([bg, bg + float(value)])
        return tpa_center_module.measure_center_scan(
            _FakeMonitor(reads), _FakeSLM(), calib,
            center_wavelengths_nm=centers,
            n_channels=1, channel_width_px=15, gap_px=5,
            pair_index=0, n_trials=1, repeats=1, settle=0.0,
            subtract_background=True,
        )

    def test_round_trip_preserves_rows_and_fit(self) -> None:
        import tempfile

        from slm_module.tpa_center import load_tpa_center_json, save_tpa_center_json

        result = self._scan_result()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "center.json"
            save_tpa_center_json(result, path)
            loaded = load_tpa_center_json(path)

        np.testing.assert_allclose(loaded.center_wl_nm, result.center_wl_nm)
        np.testing.assert_allclose(loaded.net_signal_v, result.net_signal_v)
        np.testing.assert_allclose(loaded.background_v, result.background_v)
        self.assertEqual(loaded.pair_index, result.pair_index)
        self.assertEqual(loaded.subtract_background, result.subtract_background)
        self.assertIsNotNone(loaded.fit)
        self.assertEqual(loaded.fit.valid, result.fit.valid)
        self.assertAlmostEqual(loaded.fit.center_wl_nm, result.fit.center_wl_nm)
        self.assertAlmostEqual(loaded.fit.chi2_red, result.fit.chi2_red)
        np.testing.assert_allclose(loaded.fit.signal_v, result.fit.signal_v)

    def test_load_rejects_wrong_schema(self) -> None:
        import json
        import tempfile

        from slm_module.tpa_center import load_tpa_center_json

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps({"schema": "nope"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema"):
                load_tpa_center_json(path)


if __name__ == "__main__":
    unittest.main()
