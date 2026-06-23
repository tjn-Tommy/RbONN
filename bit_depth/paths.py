"""Filesystem paths and lightweight IO helpers for the bit_depth package.

Importing this module configures MPLCONFIGDIR *before* matplotlib is ever
imported (see figures.py), and exposes OUTPUT_DIR / write_csv without pulling in
matplotlib, so the simulation and Monte-Carlo code stays importable in a
headless / torch-free environment.
"""
import csv
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / 'outputs'
MPLCONFIG_DIR = SCRIPT_DIR / '.mplconfig'
MPLCONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(MPLCONFIG_DIR))


def write_csv(filename, rows, fieldnames):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path
