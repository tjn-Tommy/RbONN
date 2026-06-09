from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slm_module.detector import (
    ScanSample,
    SimulatedDetector,
    compute_beam_center,
    write_samples_csv,
)


class SimulatedDetectorTests(unittest.TestCase):
    def test_signal_peaks_at_configured_center(self) -> None:
        detector = SimulatedDetector(center_x=500.0, sigma_px=100.0, noise=0.0)

        detector.on_frame(500.0)
        at_center = detector.read()
        detector.on_frame(900.0)
        off_center = detector.read()

        self.assertGreater(at_center, off_center)
        self.assertAlmostEqual(at_center, 1.05, places=6)

    def test_read_before_frame_returns_baseline(self) -> None:
        detector = SimulatedDetector(center_x=500.0, baseline=0.1, noise=0.0)
        self.assertAlmostEqual(detector.read(), 0.1)

    def test_rejects_invalid_sigma(self) -> None:
        with self.assertRaises(ValueError):
            SimulatedDetector(center_x=0.0, sigma_px=0.0)


class CenterDetectionTests(unittest.TestCase):
    def test_peak_and_centroid_on_clean_gaussian(self) -> None:
        detector = SimulatedDetector(center_x=300.0, sigma_px=80.0, noise=0.0)
        samples = []
        for x in range(0, 601, 20):
            detector.on_frame(float(x))
            samples.append(ScanSample(x_center=float(x), signal=detector.read()))

        result = compute_beam_center(samples)

        self.assertAlmostEqual(result.peak_x, 300.0)
        self.assertAlmostEqual(result.centroid_x, 300.0, delta=1.0)
        self.assertGreater(result.peak_signal, 1.0)

    def test_flat_signal_falls_back_to_peak(self) -> None:
        samples = [ScanSample(x, 0.5) for x in (0.0, 10.0, 20.0)]

        result = compute_beam_center(samples)

        self.assertEqual(result.centroid_x, result.peak_x)

    def test_requires_at_least_two_samples(self) -> None:
        with self.assertRaises(ValueError):
            compute_beam_center([ScanSample(0.0, 1.0)])

    def test_write_samples_csv(self) -> None:
        samples = [ScanSample(0.0, 0.1), ScanSample(5.0, 0.9)]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_samples_csv(samples, Path(temp_dir) / "samples.csv")
            with open(path, encoding="utf-8", newline="") as file:
                rows = list(csv.reader(file))

        self.assertEqual(rows[0], ["x_center", "signal"])
        self.assertEqual(len(rows), 3)
        self.assertEqual(float(rows[2][0]), 5.0)
        self.assertEqual(float(rows[2][1]), 0.9)


if __name__ == "__main__":
    unittest.main()
