from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import curve_fit


REQUIRED_COLUMNS = ("wavelength_nm", "level", "intensity")


@dataclass(frozen=True)
class CalibrationPoint:
    wavelength_nm: float
    level: float
    intensity: float


@dataclass(frozen=True)
class CalibrationFit:
    wavelength_nm: float
    i0: float
    phase_slope: float
    phase_offset: float
    rmse: float
    r_squared: float
    levels: np.ndarray
    intensities: np.ndarray
    fitted_intensities: np.ndarray

    def to_dict(self) -> dict[str, float]:
        return {
            "wavelength_nm": float(self.wavelength_nm),
            "i0": float(self.i0),
            "phase_slope": float(self.phase_slope),
            "phase_offset": float(self.phase_offset),
            "rmse": float(self.rmse),
            "r_squared": float(self.r_squared),
        }


def intensity_model(
    levels: np.ndarray | float,
    i0: float,
    phase_slope: float,
    phase_offset: float,
) -> np.ndarray:
    """Model intensity = I0 * sin(theta / 2)^2, theta = a * level + b."""
    level_array = np.asarray(levels, dtype=float)
    theta = phase_slope * level_array + phase_offset
    return i0 * np.sin(theta / 2.0) ** 2


def phase_for_level(fit: CalibrationFit, levels: np.ndarray | float) -> np.ndarray:
    return fit.phase_slope * np.asarray(levels, dtype=float) + fit.phase_offset


def predict_intensity(fit: CalibrationFit, levels: np.ndarray | float) -> np.ndarray:
    return intensity_model(levels, fit.i0, fit.phase_slope, fit.phase_offset)


def load_calibration_csv(csv_path: str | Path) -> list[CalibrationPoint]:
    path = Path(csv_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Calibration CSV not found: {path}")

    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError("Calibration CSV is empty")

        normalized = {name.strip(): name for name in reader.fieldnames}
        missing = [column for column in REQUIRED_COLUMNS if column not in normalized]
        if missing:
            raise ValueError(
                "Calibration CSV missing required columns: " + ", ".join(missing)
            )

        points: list[CalibrationPoint] = []
        for row_number, row in enumerate(reader, start=2):
            if not any((value or "").strip() for value in row.values()):
                continue

            wavelength = _parse_float(
                row[normalized["wavelength_nm"]], "wavelength_nm", row_number
            )
            level = _parse_float(row[normalized["level"]], "level", row_number)
            intensity = _parse_float(
                row[normalized["intensity"]], "intensity", row_number
            )

            if wavelength <= 0:
                raise ValueError(f"wavelength_nm must be positive at row {row_number}")
            if not 0 <= level <= 1023:
                raise ValueError(f"level must be in 0..1023 at row {row_number}")
            if intensity < 0:
                raise ValueError(f"intensity must be non-negative at row {row_number}")

            points.append(CalibrationPoint(wavelength, level, intensity))

    if not points:
        raise ValueError("Calibration CSV does not contain any measurements")
    return points


def fit_calibration(
    points: Iterable[CalibrationPoint],
    *,
    maxfev: int = 50000,
) -> dict[float, CalibrationFit]:
    grouped: dict[float, list[CalibrationPoint]] = defaultdict(list)
    for point in points:
        grouped[float(point.wavelength_nm)].append(point)

    if not grouped:
        raise ValueError("No calibration points were provided")

    return {
        wavelength: _fit_single_wavelength(wavelength, wavelength_points, maxfev)
        for wavelength, wavelength_points in sorted(grouped.items())
    }


def _fit_single_wavelength(
    wavelength_nm: float,
    points: list[CalibrationPoint],
    maxfev: int,
) -> CalibrationFit:
    if len(points) < 4:
        raise ValueError(
            f"Need at least 4 points for wavelength {wavelength_nm:g} nm"
        )

    ordered = sorted(points, key=lambda point: point.level)
    levels = np.asarray([point.level for point in ordered], dtype=float)
    intensities = np.asarray([point.intensity for point in ordered], dtype=float)
    level_span = float(np.ptp(levels))
    if level_span <= 0:
        raise ValueError(f"Levels must vary for wavelength {wavelength_nm:g} nm")

    max_intensity = float(np.max(intensities))
    i0_upper = max(max_intensity * 2.5, 1.0)
    slope_limit = max(8.0 * math.pi / level_span, 2.0 * math.pi / 1023.0)
    bounds = ([0.0, -slope_limit, -4.0 * math.pi], [i0_upper, slope_limit, 4.0 * math.pi])

    candidates = []
    i0_guesses = [max(max_intensity, 1.0), max(max_intensity * 1.25, 1.0)]
    slope_guesses = [
        math.pi / 1023.0,
        2.0 * math.pi / 1023.0,
        4.0 * math.pi / 1023.0,
        -math.pi / 1023.0,
        -2.0 * math.pi / 1023.0,
    ]
    offset_guesses = [0.0, 0.5 * math.pi, math.pi, -0.5 * math.pi]

    for i0_guess in i0_guesses:
        for slope_guess in slope_guesses:
            for offset_guess in offset_guesses:
                p0 = [
                    float(np.clip(i0_guess, bounds[0][0], bounds[1][0])),
                    float(np.clip(slope_guess, bounds[0][1], bounds[1][1])),
                    float(np.clip(offset_guess, bounds[0][2], bounds[1][2])),
                ]
                try:
                    params, _ = curve_fit(
                        intensity_model,
                        levels,
                        intensities,
                        p0=p0,
                        bounds=bounds,
                        maxfev=maxfev,
                    )
                except (RuntimeError, ValueError):
                    continue

                fitted = intensity_model(levels, *params)
                sse = float(np.sum((intensities - fitted) ** 2))
                candidates.append((sse, params, fitted))

    if not candidates:
        raise RuntimeError(f"Could not fit calibration for {wavelength_nm:g} nm")

    sse, params, fitted = min(candidates, key=lambda item: item[0])
    rmse = math.sqrt(sse / len(levels))
    centered = intensities - np.mean(intensities)
    sst = float(np.sum(centered**2))
    r_squared = 1.0 if sst == 0.0 and sse == 0.0 else 1.0 - sse / sst if sst else 0.0

    return CalibrationFit(
        wavelength_nm=float(wavelength_nm),
        i0=float(params[0]),
        phase_slope=float(params[1]),
        phase_offset=float(params[2]),
        rmse=float(rmse),
        r_squared=float(r_squared),
        levels=levels,
        intensities=intensities,
        fitted_intensities=fitted,
    )


def _parse_float(value: str | None, column: str, row_number: int) -> float:
    try:
        number = float((value or "").strip())
    except ValueError as exc:
        raise ValueError(f"Invalid {column} at row {row_number}: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{column} must be finite at row {row_number}")
    return number
