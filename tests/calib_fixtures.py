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


def seed_run(dst: Path, *, step3: Path | None = None, step6: Path | None = None,
             step7: Path | None = None, copy: tuple[Path, ...] = ()) -> Path:
    """Write ``dst/calib.json`` from legacy per-step results; return ``dst``.

    Each argument is a legacy JSON: a bare step-3 ``calibration_result_v2``, or a
    combined step-6/step-7 result whose own section is lifted out.  Sections are
    written in step order, because writing one drops the ones below it.

    ``copy`` names files (CSVs, usually) to copy in beside the calib.json, so a
    section's relative ``csv`` pointer resolves.
    """
    from calibration_module.calib_file import write_section

    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for src in copy:
        shutil.copy2(src, dst / Path(src).name)

    if step3 is not None:
        payload = _payload(step3)
        section = payload.get("step3", payload)
        write_section(dst, 3, {k: v for k, v in section.items() if k != "schema"},
                      log=None)
    if step6 is not None:
        payload = _payload(step6)
        write_section(dst, 6, payload.get("step6", payload), log=None)
    if step7 is not None:
        payload = _payload(step7)
        write_section(dst, 7, payload["step7"], log=None)
    return dst
