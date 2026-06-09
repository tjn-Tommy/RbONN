from __future__ import annotations

import csv
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


class Detector(ABC):
    """Interface for an intensity detector sampled during a center scan.

    Hardware integrations only need to implement read(); on_frame() exists so
    software detectors (e.g. the simulator) can know where the scan window is.
    """

    name: str = "detector"

    def on_frame(self, x_center: float) -> None:
        """Called after each scan frame has settled, before read()."""

    @abstractmethod
    def read(self) -> float:
        """Return the current detector signal."""

    def close(self) -> None:
        """Release hardware resources, if any."""


class SimulatedDetector(Detector):
    """Gaussian beam profile with optional noise, for testing without hardware."""

    name = "simulated"

    def __init__(
        self,
        center_x: float,
        sigma_px: float = 150.0,
        amplitude: float = 1.0,
        baseline: float = 0.05,
        noise: float = 0.01,
        seed: int | None = None,
    ):
        if sigma_px <= 0:
            raise ValueError("sigma_px must be positive")
        self.center_x = float(center_x)
        self.sigma_px = float(sigma_px)
        self.amplitude = float(amplitude)
        self.baseline = float(baseline)
        self.noise = float(noise)
        self._rng = np.random.default_rng(seed)
        self._last_x: float | None = None

    def on_frame(self, x_center: float) -> None:
        self._last_x = float(x_center)

    def read(self) -> float:
        if self._last_x is None:
            return self.baseline
        offset = self._last_x - self.center_x
        signal = self.baseline + self.amplitude * math.exp(
            -(offset * offset) / (2.0 * self.sigma_px * self.sigma_px)
        )
        if self.noise > 0:
            signal += self._rng.normal(0.0, self.noise)
        return max(0.0, signal)


@dataclass(frozen=True)
class ScanSample:
    x_center: float
    signal: float


@dataclass(frozen=True)
class CenterResult:
    peak_x: float
    centroid_x: float
    peak_signal: float


def compute_beam_center(samples: Sequence[ScanSample]) -> CenterResult:
    """Estimate the beam center from (x, signal) samples.

    peak_x is the x with the maximum signal; centroid_x is the mean of x
    weighted by signal above the minimum (falls back to peak_x for a flat
    signal).
    """
    if len(samples) < 2:
        raise ValueError("center detection needs at least 2 samples")

    positions = np.array([sample.x_center for sample in samples], dtype=float)
    signals = np.array([sample.signal for sample in samples], dtype=float)

    peak_index = int(np.argmax(signals))
    peak_x = float(positions[peak_index])
    peak_signal = float(signals[peak_index])

    weights = signals - signals.min()
    total = weights.sum()
    if total > 0:
        centroid_x = float(np.dot(positions, weights) / total)
    else:
        centroid_x = peak_x
    return CenterResult(peak_x=peak_x, centroid_x=centroid_x, peak_signal=peak_signal)


def write_samples_csv(samples: Sequence[ScanSample], csv_path: str | Path) -> Path:
    path = Path(csv_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["x_center", "signal"])
        for sample in samples:
            writer.writerow([sample.x_center, sample.signal])
    return path
