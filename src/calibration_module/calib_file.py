"""The one calibration file every step appends to: ``calib.json``.

In:  a path -- the file, or a directory as shorthand for the ``calib.json``
     inside it, which is all a "run folder" ever was.
Out: a :class:`Calib`: which sections exist, when each ran, where its CSV is,
     and what a new section invalidates.

Each step reads the file, replaces its OWN section and writes it back.  No step
copies another's payload, so the step-3 channel layout exists once instead of
once per downstream result.  The raw readings stay in CSVs beside it, several
per calibration being normal, and the JSON keeps only what was fitted from them
plus a relative pointer back::

    run_0931/
      calib.json            step1 .. step8 sections
      step2_0928_1744.csv   px -> nm
      step3_0931_1802.csv   per-channel transfer sweep
      step6_meas_*.csv      per-pair TPA sweep

THE FILE IS THE CALIBRATION; the folder is only where its CSVs land.

Inheriting an older calibration is a FILESYSTEM operation: copy the
``calib.json`` and the CSVs worth keeping to a new folder, then run the step you
want to redo.  There is no ``--inherit`` flag and no provenance field -- what you
copied is what the file says.  An offline re-fit is the same move, which is why
nothing ever writes a fit back over the file a GUI was pointed at to look at.

WRITING A SECTION DROPS THE SECTIONS BELOW IT.  A fresh step 3 is a new channel
layout, so that file's step-6 etas and step-7 phases were measured through an
aperture that no longer exists; keeping them would leave the file reading as
valid when it is not.  They are dropped, named as they go, and the directory
they were copied from still has them.

Section payloads are not this module's business -- each fit owns its own
(``fit.slm_response``, ``fit.pair_v2``, ``fit.phase``).  What lives here is the
container.

Every section carries ``ran``, a full local ISO timestamp.  Nothing enforces an
order -- a hand-copied file can hold a step 6 older than its step 3 -- but
:meth:`Calib.stale` finds it and :meth:`Calib.chain_lines` says so before an
hour is spent on it.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

SCHEMA = "rbonn_calib_v3"
CALIB_NAME = "calib.json"

#: Steps in dependency order.  4 and 5 never existed; 3b/3c collapsed into 3.
STEP_ORDER: tuple[int, ...] = (1, 2, 3, 6, 7, 8)

#: What each step contributes, for error messages that say what to run.
STEP_WHAT: dict[int, str] = {
    1: "min/max grayscale levels",
    2: "wavelength map (px -> nm)",
    3: "per-channel transfer curves + channel layout",
    6: "per-pair TPA efficiency eta",
    7: "comb phase Phi_k",
    8: "saturation Y_sat",
}


class CalibFileError(Exception):
    """Anything wrong with a calib.json."""


class CalibNotFound(CalibFileError):
    """The named calib.json is not there (nor one in the named directory)."""


class SectionMissing(CalibFileError):
    """The file exists but does not carry the step that was asked for."""


class SchemaMismatch(CalibFileError):
    """Not a v3 calib.json -- probably a pre-v3 per-step result."""


# ----------------------------------------------------------------------
# paths and step keys
# ----------------------------------------------------------------------

def step_key(step: int | str) -> str:
    """``3`` / ``"3"`` / ``"step3"`` -> ``"step3"``, validated against STEP_ORDER."""
    text = str(step).strip().lower()
    text = text[4:] if text.startswith("step") else text
    try:
        number = int(text)
    except ValueError as exc:
        raise CalibFileError(f"not a step: {step!r}") from exc
    if number not in STEP_ORDER:
        raise CalibFileError(
            f"step {number} is not a calibration step "
            f"(have {', '.join(str(s) for s in STEP_ORDER)})"
        )
    return f"step{number}"


def step_number(step: int | str) -> int:
    return int(step_key(step)[4:])


def calib_path(calib: str | Path) -> Path:
    """The named file, or the ``calib.json`` inside a named directory."""
    path = Path(calib).expanduser()
    return path / CALIB_NAME if path.is_dir() or path.suffix == "" else path


def stamp_now() -> str:
    """Local ISO timestamp to the second: ``2026-09-30T18:02:41``.

    Full ISO rather than the filenames' ``MMDD_HHMM``: a section outlives the
    directory it was copied from, and a year-less stamp cannot say when.
    """
    return datetime.now().isoformat(timespec="seconds")


def _pretty_stamp(ran: str | None) -> str:
    """``2026-09-30T18:02:41`` -> ``2026-09-30 18:02``, anything else verbatim."""
    if not ran:
        return "?"
    try:
        return datetime.fromisoformat(str(ran)).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return str(ran)


def _as_csv_list(csv: str | Path | Sequence[str | Path] | None) -> list[Path]:
    if csv is None:
        return []
    if isinstance(csv, (str, Path)):
        return [Path(csv)]
    return [Path(entry) for entry in csv]


def _relative_to(folder: Path, target: Path) -> str:
    """Store CSV pointers relative to the JSON, in posix form.

    Relative so the JSON and its CSVs can be moved or copied together without
    breaking; posix so a file written on Windows still resolves elsewhere.  A
    CSV outside the folder (an offline re-fit of another night's sweep) comes
    out as ``../run_0930/step6_meas_0930_2201.csv``, which is honest about what
    it is.
    """
    target = Path(target).expanduser()
    if not target.is_absolute():
        # A relative pointer is ambiguous: argv and every ``folder / name`` give
        # one relative to the CWD, while a bare filename usually means "beside
        # the JSON".  Whichever names a real file wins, CWD first, because that
        # is what a command line means -- otherwise ``--calib
        # src/calib_data/run_x/calib.json sweep.csv`` from the repo root stores
        # the whole repo-relative path and nothing downstream can resolve it.
        from_cwd = (Path.cwd() / target).resolve()
        from_folder = (folder / target).resolve()
        target = (from_cwd if from_cwd.exists() or not from_folder.exists()
                  else from_folder)
    try:
        rel = os.path.relpath(target, folder)
    except ValueError:                      # different drive on Windows
        return target.as_posix()
    return Path(rel).as_posix()


# ----------------------------------------------------------------------
# the file
# ----------------------------------------------------------------------

@dataclass
class Calib:
    """A loaded ``calib.json``, and the folder it sits in."""

    path: Path
    payload: dict[str, Any] = field(default_factory=dict)

    # -- where it is -------------------------------------------------

    @property
    def folder(self) -> Path:
        """Where this file lives -- and so where its CSVs and plots go."""
        return self.path.parent

    def beside(self, name: str) -> Path:
        """``name`` in this file's folder: the one place a step writes to.

        Every derived artefact of a step -- its meas CSV above all -- is named
        through here, so "beside the calib.json" is one expression rather than a
        ``.parent`` each caller spells for itself.
        """
        return self.folder / name

    # -- what it holds -----------------------------------------------

    def steps(self) -> list[int]:
        """Steps present in the file, in dependency order."""
        return [s for s in STEP_ORDER if isinstance(self.payload.get(f"step{s}"), dict)]

    def __contains__(self, step: int | str) -> bool:
        return isinstance(self.payload.get(step_key(step)), dict)

    def get(self, step: int | str) -> dict | None:
        section = self.payload.get(step_key(step))
        return section if isinstance(section, dict) else None

    def section(self, step: int | str) -> dict:
        """The step's section, or raise saying what to run to get it."""
        section = self.get(step)
        if section is None:
            number = step_number(step)
            raise SectionMissing(
                f"{self.path} has no step{number} section "
                f"({STEP_WHAT.get(number, 'that step')}) -- run step {number} "
                f"against this file, or copy a calib.json that has it. "
                f"Present: {', '.join(f'step{s}' for s in self.steps()) or 'nothing'}"
            )
        return section

    def require(self, *steps: int | str) -> None:
        """Raise on the first missing one; call it before touching hardware."""
        for step in steps:
            self.section(step)

    # -- when it ran -------------------------------------------------

    def ran(self, step: int | str) -> str | None:
        section = self.get(step)
        return None if section is None else section.get("ran")

    def stale(self) -> dict[int, list[int]]:
        """``{step: [upstream steps that ran later]}`` -- empty when consistent.

        Only a hand-assembled file gets here: :func:`write_section` drops
        downstream sections rather than let this happen.
        """
        out: dict[int, list[int]] = {}
        for step in self.steps():
            ran = self.ran(step)
            if not ran:
                continue
            newer = [
                up for up in self.steps()
                if up < step and (self.ran(up) or "") > ran
            ]
            if newer:
                out[step] = newer
        return out

    # -- the CSVs it points at ---------------------------------------

    def csv_paths(self, step: int | str) -> list[Path]:
        """Resolved CSV path(s) the section was fitted from; [] if it names none.

        Existence is not checked here -- a section stays readable when its CSV
        was left behind in the directory it was copied from.
        """
        stored = self.section(step).get("csv")
        return [(self.folder / entry).resolve() for entry in _as_csv_list(stored)]

    def csv_path(self, step: int | str) -> Path:
        """The one CSV the section names; raises if it names none or several."""
        paths = self.csv_paths(step)
        if not paths:
            raise SectionMissing(f"step{step_number(step)} in {self.path} names no csv")
        if len(paths) > 1:
            raise CalibFileError(
                f"step{step_number(step)} in {self.path} names {len(paths)} CSVs "
                f"-- use csv_paths()"
            )
        return paths[0]

    # -- step 3, the one section everything downstream needs ---------

    def step3_result(self):
        """The step-3 section as a :class:`CalibrationResult`.

        Carries the fitted ``transfer_fits`` and no raw curves: the sweep itself
        is in ``step3_*.csv``.  Encoding inverts the fit, which is the only
        method left, so nothing downstream needs the curves back.
        """
        from calibration_module.fit.slm_response import calibration_result_from_dict
        return calibration_result_from_dict(self.section(3))

    def layout(self):
        """The channel layout step 3 defines -- what every level is encoded through."""
        from slm_module.encoding import channel_layout_from_calibration
        return channel_layout_from_calibration(self.step3_result())

    # -- saying all of that out loud ---------------------------------

    def chain_lines(self) -> list[str]:
        """The stamp chain, one line per section, stale ones flagged.

        Printed by every runner before it touches the bench, and beside every
        GUI page's calib file field.  A note, never a refusal: whether a step 6
        older than
        its step 3 matters is the operator's call, but it is not going to be a
        surprise.
        """
        stale = self.stale()
        lines = [str(self.path)]
        for step in self.steps():
            note = ""
            if step in stale:
                older = ", ".join(f"step{s}" for s in stale[step])
                note = f"   <- older than {older}"
            lines.append(f"  step{step}  {_pretty_stamp(self.ran(step))}{note}")
        if len(lines) == 1:
            lines.append("  (empty)")
        return lines

    def print_chain(self, log: Callable[[str], None] = print) -> None:
        for line in self.chain_lines():
            log(line)

    def __str__(self) -> str:
        return "\n".join(self.chain_lines())


# ----------------------------------------------------------------------
# read / write
# ----------------------------------------------------------------------

def load_calib(calib: str | Path, *, missing_ok: bool = False) -> Calib:
    """Load the named ``calib.json`` (or the one in the named directory).

    ``missing_ok`` returns an empty :class:`Calib` instead of raising, which is
    what a step needs when it is the first one to write into a fresh file.
    """
    path = calib_path(calib)
    if not path.is_file():
        if missing_ok:
            return Calib(path=path, payload={})
        raise CalibNotFound(
            f"{path} is not there -- create it with step 1, or copy a "
            f"{CALIB_NAME} into that folder"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CalibFileError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise CalibFileError(f"{path} is not a JSON object")
    schema = payload.get("schema")
    if schema != SCHEMA:
        raise SchemaMismatch(
            f"{path} has schema {schema!r}, expected {SCHEMA!r} -- pre-v3 "
            f"per-step results are converted by tools/migrate_calib_v3.py"
        )
    return Calib(path=path, payload=payload)


def write_section(
    calib_file: str | Path,
    step: int | str,
    payload: dict[str, Any],
    *,
    csv: str | Path | Sequence[str | Path] | None = None,
    ran: str | None = None,
    log: Callable[[str], None] | None = print,
) -> Calib:
    """Stamp ``payload``, put it in the calib.json, drop what it invalidates.

    ``csv`` is stored relative to the JSON.  ``ran`` defaults to now -- pass it
    only to preserve an original timestamp, which is what the converter does.
    Returns the file as it now stands.
    """
    key = step_key(step)
    number = step_number(step)
    path = calib_path(calib_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    calib = load_calib(path, missing_ok=True)

    section: dict[str, Any] = {"ran": ran or stamp_now()}
    stored = [_relative_to(path.parent, entry) for entry in _as_csv_list(csv)]
    if len(stored) == 1:
        section["csv"] = stored[0]
    elif stored:
        section["csv"] = stored
    section.update({k: v for k, v in payload.items() if k not in ("ran", "csv")})

    dropped = _drop_downstream(calib.payload, number)
    calib.payload[key] = section
    calib.payload["schema"] = SCHEMA
    _write_atomic(path, _ordered(calib.payload))

    if log is not None:
        if dropped:
            log(f"[calib] step{number} written; dropped "
                + ", ".join(f"step{n} (ran {_pretty_stamp(r)})" for n, r in dropped)
                + " -- fitted against the previous "
                + f"step{number}")
        else:
            log(f"[calib] step{number} written to {path}")
    return calib


def _drop_downstream(payload: dict[str, Any], number: int) -> list[tuple[int, str]]:
    """Remove every section after ``number``; return ``[(step, ran)]`` removed."""
    dropped: list[tuple[int, str]] = []
    for other in STEP_ORDER:
        if other <= number:
            continue
        section = payload.get(f"step{other}")
        if isinstance(section, dict):
            dropped.append((other, section.get("ran", "")))
            payload.pop(f"step{other}")
    return dropped


def _ordered(payload: dict[str, Any]) -> dict[str, Any]:
    """``schema`` first, then the steps in order, then anything else."""
    out: dict[str, Any] = {"schema": payload.get("schema", SCHEMA)}
    for step in STEP_ORDER:
        key = f"step{step}"
        if key in payload:
            out[key] = payload[key]
    for key, value in payload.items():
        if key != "schema" and not key.startswith("step"):
            out[key] = value
    return out


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write through a temp file in the same directory, then replace.

    A half-written calib.json is the one failure this design cannot absorb:
    it is the only copy of every fit in the directory.
    """
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def sections_of(payload: dict[str, Any]) -> Iterable[tuple[int, dict]]:
    """``(step, section)`` for a raw payload dict -- for the converter's checks."""
    for step in STEP_ORDER:
        section = payload.get(f"step{step}")
        if isinstance(section, dict):
            yield step, section
