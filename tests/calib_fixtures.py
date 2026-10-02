"""Build a v3 run directory out of the frozen legacy fixtures.

The archived runs under ``src/calib_data/archive/`` are deliberately NOT
migrated -- they are frozen evidence, and the converter skips them -- but they
are also the only committed step-3/step-6/step-7 payloads the suite can fit
against without hardware.  So the tests lift the sections out of those per-step
JSONs and write them into a throwaway ``calib.json``, which is what every step
now reads.

Not a package: ``tests/`` has no ``__init__.py``, so a test module imports this
by putting its own directory on ``sys.path`` first.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _payload(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sibling_meas_csv(result_json: Path, *, required: bool = True) -> Path | None:
    """``calib_step6v2_result_0907_1757.json`` -> its ``..._meas_0907_1757.csv``."""
    csv = result_json.with_name(
        result_json.name.replace("_result_", "_meas_")).with_suffix(".csv")
    if not csv.is_file():
        if not required:
            return None
        raise FileNotFoundError(
            f"{result_json.name} records no sigma floor and no measurement CSV "
            f"sits beside it to measure one from ({csv.name}); pass step6_csv="
        )
    return csv


def _named_csvs(source: Path, section: dict) -> list[Path]:
    """The sweep(s) a v3 section already points at, resolved beside its JSON.

    A v3 ``calib.json`` is a perfectly good source for a seeded one, and it
    names its own CSVs -- including the two a hand-merged cascade carries.
    """
    named = section.get("csv")
    if isinstance(named, str):
        named = [named]
    if not isinstance(named, list):
        return []
    return [Path(source).parent / entry for entry in named]


def seed_run(dst: Path, *, step3: Path | None = None, step6: Path | None = None,
             step7: Path | None = None, copy: tuple[Path, ...] = (),
             step3_csv: Path | None = None,
             step6_csv: Path | None = None,
             step7_csv: Path | None = None) -> Path:
    """Write ``dst/calib.json`` from legacy per-step results; return ``dst``.

    Each argument is a legacy JSON: a bare step-3 ``calibration_result_v2``, or a
    combined step-6/step-7 result whose own section is lifted out.  Sections are
    written in step order, because writing one drops the ones below it.

    ``copy`` names files (CSVs, usually) to copy in beside the calib.json, so a
    section's relative ``csv`` pointer resolves.

    Legacy step-6 results predate the measured sigma floor, and the chain now
    refuses a step 6 without one -- no constant stands in.  So a seeded step 6
    gets the floor that sweep WOULD have measured, from its own dark rows: the
    measurement CSV beside the JSON (same stamp), or ``step6_csv`` when the
    JSON lives elsewhere.

    That CSV is COPIED in and recorded as the section's ``csv``, because that
    pointer is what a v3 section is: the GUI's Load re-fits the sweep a section
    names rather than asking for one, so a seeded file without the pointer is
    not the thing under test.
    """
    from calibration_module.calib_file import write_section
    from calibration_module.fit.pair_v2 import load_meas_csv, sweep_floor

    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for src in copy:
        shutil.copy2(src, dst / Path(src).name)

    def _bring(csv):
        """Copy sweep(s) in beside the calib.json and return the new path(s)."""
        if csv is None:
            return None
        if isinstance(csv, (list, tuple)):
            brought = [_bring(entry) for entry in csv]
            return brought or None
        here = dst / Path(csv).name
        if Path(csv).resolve() != here.resolve():
            shutil.copy2(csv, here)
        return here

    if step3 is not None:
        payload = _payload(step3)
        section = payload.get("step3", payload)
        write_section(dst, 3, {k: v for k, v in section.items() if k != "schema"},
                      csv=_bring(step3_csv), log=None)
    if step6 is not None:
        payload = _payload(step6)
        section = dict(payload.get("step6", payload))
        # Whichever sweep this section belongs to: the one named here, the one
        # a v3 source already points at, or the sibling of a legacy result.  It
        # is REQUIRED only when the floor has to be measured from it.
        csv = (Path(step6_csv) if step6_csv is not None
               else (_named_csvs(Path(step6), section) or [None])[0]
               or _sibling_meas_csv(
                   Path(step6), required="sigma_floor_v" not in section))
        if "sigma_floor_v" not in section:
            floor, source = sweep_floor(load_meas_csv(csv))
            section["sigma_floor_v"], section["sigma_floor_source"] = floor, source
        write_section(dst, 6, section, csv=_bring(csv), log=None)
    if step7 is not None:
        payload = _payload(step7)
        csv = (Path(step7_csv) if step7_csv is not None
               else _named_csvs(Path(step7), payload["step7"])
               or _sibling_meas_csv(Path(step7), required=False))
        write_section(dst, 7, payload["step7"], csv=_bring(csv), log=None)
    return dst
