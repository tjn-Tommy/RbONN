from __future__ import annotations

import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from calibration_module.fit.legacy_intensity import (
    fit_calibration,
    intensity_model,
    load_calibration_csv,
)


class CalibrationTests(unittest.TestCase):
    def test_load_calibration_csv_long_format(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cal.csv"
            with open(path, "w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["wavelength_nm", "level", "intensity"])
                writer.writerow([800, 0, 0.1])
                writer.writerow([800, 512, 1.2])

            points = load_calibration_csv(path)

        self.assertEqual(len(points), 2)
        self.assertEqual(points[0].wavelength_nm, 800)
        self.assertEqual(points[1].level, 512)

    def test_load_calibration_csv_rejects_missing_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.csv"
            path.write_text("level,intensity\n0,1\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "missing required columns"):
                load_calibration_csv(path)

    def test_fit_calibration_recovers_synthetic_curve(self) -> None:
        levels = np.linspace(0, 1023, 50)
        true_i0 = 2.5
        true_slope = 2.0 * math.pi / 1023.0
        true_offset = 0.35
        intensities = intensity_model(levels, true_i0, true_slope, true_offset)
        points = [
            type("Point", (), {"wavelength_nm": 800.0, "level": float(level), "intensity": float(intensity)})
            for level, intensity in zip(levels, intensities)
        ]

        fit = fit_calibration(points)[800.0]

        self.assertLess(fit.rmse, 1e-6)
        self.assertGreater(fit.r_squared, 0.999999)
        np.testing.assert_allclose(
            intensity_model(levels, fit.i0, fit.phase_slope, fit.phase_offset),
            intensities,
            atol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
