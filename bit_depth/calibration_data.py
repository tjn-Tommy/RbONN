"""Operating geometry derived from the measured wavelength calibration.

Per the project decision we take *only* ``nm_per_px`` from the measured
``calibration.csv`` and assume a flat (equal-intensity) input spectrum. The
operating layout is a 15-px window on a 20-px pitch, i.e. a 5-px interval split
as a 2.5-px guard on each side.
"""
import csv
from pathlib import Path

import numpy as np

from .config import Cfg
from .paths import SCRIPT_DIR

CALIBRATION_CSV = SCRIPT_DIR.parent / 'calibration.csv'

# 15-px window, 20-px pitch, 5-px interval = 2.5-px guard each side.
OPERATING_PX_PER_CH = 15
OPERATING_GUARD = 2.5
OPERATING_GROUP_PX = 20


def operating_geometry():
    return dict(px_per_ch=OPERATING_PX_PER_CH, guard=OPERATING_GUARD,
                group_px=OPERATING_GROUP_PX)


def _read_coordinate_wavelength(path):
    """One (coordinate_px, wavelength_nm) pair per coordinate, first row wins."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'calibration CSV not found: {path}')
    mapping = {}
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError('calibration CSV is empty')
        normalized = {name.strip(): name for name in reader.fieldnames}
        for column in ('coordinate_px', 'wavelength_nm'):
            if column not in normalized:
                raise ValueError(f'calibration CSV missing required column: {column}')
        for row in reader:
            if not any((value or '').strip() for value in row.values()):
                continue
            coordinate = float(row[normalized['coordinate_px']])
            wavelength = float(row[normalized['wavelength_nm']])
            mapping.setdefault(coordinate, wavelength)
    if len(mapping) < 2:
        raise ValueError('calibration CSV needs at least two coordinates')
    coordinates = np.fromiter(mapping.keys(), dtype=float)
    wavelengths = np.fromiter(mapping.values(), dtype=float)
    order = np.argsort(coordinates)
    return coordinates[order], wavelengths[order]


def measured_nm_per_px(path=CALIBRATION_CSV):
    """Magnitude of the linear wavelength-vs-coordinate slope [nm/px]."""
    coordinates, wavelengths = _read_coordinate_wavelength(path)
    slope = np.polyfit(coordinates, wavelengths, deg=1)[0]
    return float(abs(slope))


def cfg_from_calibration(path=CALIBRATION_CSV, **overrides) -> Cfg:
    """Operating-geometry ``Cfg`` (20-px group pitch) for the 15-px / 2.5-px layout.

    ``nm_per_px`` is read from the data for reporting/validation; the analytic
    dispersion model and ``cos(phi/2)`` device transfer are otherwise unchanged.
    Pass ``Cfg`` field overrides (e.g. ``w_px=...``) via keyword.
    """
    overrides.setdefault('group', OPERATING_GROUP_PX)
    return Cfg(**overrides)


def summarize_calibration(path=CALIBRATION_CSV):
    nm_per_px = measured_nm_per_px(path)
    geom = operating_geometry()
    return dict(nm_per_px=nm_per_px,
                nm_per_window=nm_per_px*geom['px_per_ch'],
                nm_per_group=nm_per_px*geom['group_px'],
                **geom)
