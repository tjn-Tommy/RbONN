"""One-shot: per-step calibration JSONs -> one ``calib.json`` per run directory.

    python tools/migrate_calib_v3.py                  # plan only, touches nothing
    python tools/migrate_calib_v3.py --apply          # write calib.json, stash the old
    python tools/migrate_calib_v3.py --verify         # prove the conversion lossless
    python tools/migrate_calib_v3.py --prune          # delete the stashed legacy JSONs

What it does per ``src/calib_data/run_*`` directory:

* classifies every JSON by CONTENT, not filename -- the names are a zoo
  (``calib_step6v2_*``, ``calib_step6v2_result_*``, ``calib_step7v2_*``,
  ``calib_step7_meas_*.json``, ``calib_step7_result_0916_merged_ref2.json``,
  ``system_accuracy_*``, ``calib_fullscale_*``, ``calib_step8_result_*``);
* keeps the NEWEST candidate per step and reports the ones it passes over;
* writes step 2's wavelength map out as ``step2_MMDD_HHMM.csv`` -- it is the one
  section whose raw arrays have no CSV today, and they leave the JSON here;
* drops ``intensity_levels`` / ``raw_intensity_levels`` from step 3 (verbatim in
  its sibling CSV) and the ``encoding`` marker (every run on disk is ``fit``);
* drops the absolute ``step6_json`` / ``step7_json`` pointers and rewrites
  ``csv`` / ``png`` as paths relative to the run directory;
* preserves each section's original timestamp as ``ran``.

Existing CSVs keep their filenames: the section points at whatever is there, so
converting renames nothing you might be holding open.  Only newly written files
(step 2's map) use the ``step{N}_MMDD_HHMM`` convention.

``--apply`` moves the superseded per-step JSONs into ``<run>/legacy_v2/`` rather
than deleting them -- ``run_0927_onn`` onward are not in git, so there is no
second copy.  Run ``--verify``, then ``--prune`` when satisfied.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from calibration_module.calib_file import (  # noqa: E402
    CALIB_NAME,
    STEP_ORDER,
    load_calib,
    write_section,
)

CALIB_DATA = REPO_ROOT / "src" / "calib_data"
LEGACY_DIR = "legacy_v2"
STAMP_RE = re.compile(r"(\d{2})(\d{2})_(\d{2})(\d{2})")

#: Dropped from the step-3 section: both are the sibling CSV, column for column.
STEP3_DROP = ("schema", "intensity_levels", "raw_intensity_levels")

#: Temp copies of the stashed originals that --verify hands to the old loaders.
_TEMP_FILES: list[Path] = []


# ----------------------------------------------------------------------
# classify
# ----------------------------------------------------------------------

def classify(payload: Any) -> int | None:
    """Which step a legacy JSON carries, by content.  None = not a section.

    Ordered most-specific first: a combined step-7 result embeds step 3 and
    step 6 too, so the presence of ``step7`` has to win over both.
    """
    if not isinstance(payload, dict):
        return None
    if "y_sat_v" in payload:                       # step 8 / system accuracy
        return 8
    if isinstance(payload.get("step7"), dict):
        return 7
    if isinstance(payload.get("step6"), dict):
        return 6
    if payload.get("schema") == "calibration_result_v2":
        if payload.get("intensity_levels") or payload.get("transfer_fits"):
            return 3                               # transfer curves per channel
        if len(payload.get("wavelength") or ()) > 1:
            return 2                               # the px -> nm map
        return 1                                   # min/max only, empty arrays
    return None                                    # full-scale / accuracy report


def stamp_of(path: Path) -> tuple[str, str]:
    """``(ran_iso, note)`` for a legacy file: MMDD_HHMM from the name, year from mtime.

    The filename stamp is authoritative for month/day/time -- it is what the run
    was labelled with -- and the only thing it lacks is the year, which the
    mtime supplies.  A disagreement between the two is reported rather than
    silently resolved.
    """
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    match = STAMP_RE.search(path.name)
    if match is None:
        return mtime.isoformat(timespec="seconds"), "stamp from mtime (none in name)"
    month, day, hour, minute = (int(g) for g in match.groups())
    try:
        ran = datetime(mtime.year, month, day, hour, minute)
    except ValueError:
        return mtime.isoformat(timespec="seconds"), "bad stamp in name, used mtime"
    note = ""
    if (mtime.month, mtime.day) != (month, day):
        note = f"name says {month:02d}-{day:02d}, mtime says {mtime:%m-%d}"
    return ran.isoformat(timespec="seconds"), note


# ----------------------------------------------------------------------
# section payloads
# ----------------------------------------------------------------------

def step1_section(payload: dict) -> dict:
    return {k: payload[k] for k in ("min_level", "max_level", "level_range")
            if payload.get(k) is not None}


def step2_section(payload: dict) -> dict:
    return {k: payload[k] for k in
            ("min_level", "max_level", "wavelength_fit_coefficients")
            if payload.get(k) is not None}


def step2_csv(payload: dict, out: Path) -> int:
    """Write the px -> nm map as the CSV the section will point at.

    Header matches ``fit.slm_response.load_wavelength_map_csv``, so the file is
    readable by the same loader ``--step2`` already accepted for a hand CSV.
    """
    coordinates = payload.get("coordinates") or []
    wavelengths = payload.get("wavelength") or []
    lines = ["coordinate_px,wavelength_nm"]
    lines += [f"{c!r},{w!r}" for c, w in zip(coordinates, wavelengths)]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines) - 1


def step3_section(payload: dict) -> dict:
    return {k: v for k, v in payload.items() if k not in STEP3_DROP}


def step7_section(payload: dict) -> dict:
    section = dict(payload["step7"])
    section.pop("step6_json", None)                # absolute, and now implicit
    section.pop("csv", None)                       # re-stored relative by the caller
    return section


def step8_section(payload: dict) -> dict:
    section = {k: v for k, v in payload.items()
               if k not in ("step7_json", "csv", "png")}
    if isinstance(section.get("gain_reference"), dict):
        section["gain_reference"] = {
            k: (Path(v).name if k == "csv" else v)
            for k, v in section["gain_reference"].items()
        }
    if payload.get("png"):
        section["png"] = Path(payload["png"]).name
    return section


def recorded_csvs(payload: dict, step: int, run: Path) -> list[Path]:
    """The CSV(s) a legacy section names, kept only where they are in the dir.

    Step 7 records its own (absolutely, sometimes as a stringified list).  Step 8
    records one.  Steps 3 and 6 record none, so they are matched by stamp.
    """
    if step == 7:
        raw = payload.get("step7", {}).get("csv")
    elif step == 8:
        raw = payload.get("csv")
    else:
        raw = None
    if raw is None:
        return []
    if isinstance(raw, str) and raw.startswith("["):     # run_0916's merged fit
        raw = [part.strip(" '\"") for part in raw.strip("[]").split(",")]
    names = [Path(str(entry)).name for entry in
             (raw if isinstance(raw, (list, tuple)) else [raw])]
    return [run / name for name in names if (run / name).is_file()]


def matched_csv(run: Path, step: int, ran: str) -> tuple[list[Path], str]:
    """Best CSV for a section that never recorded one -- steps 3 and 6.

    Prefers the same MMDD_HHMM stamp as the fit; failing that, the newest
    measurement CSV at or before it.  A guess is labelled as one, because an
    offline re-fit of an earlier sweep is indistinguishable after the fact.
    """
    patterns = {3: ("calib_step3*.csv", "step3_*.csv"),
                6: ("calib_step6v2_meas_*.csv", "step6_meas_*.csv")}.get(step)
    if not patterns:
        return [], ""
    candidates = sorted({p for pattern in patterns for p in run.glob(pattern)})
    if not candidates:
        return [], "no CSV in the directory"
    exact = [p for p in candidates if STAMP_RE.search(p.name)
             and _name_stamp(p) == _iso_stamp(ran)]
    if exact:
        return exact[:1], ""
    earlier = [p for p in candidates if _name_stamp(p) and _name_stamp(p) <= _iso_stamp(ran)]
    pick = (earlier or candidates)[-1]
    return [pick], f"CSV inferred: {pick.name}"


def _name_stamp(path: Path) -> str:
    match = STAMP_RE.search(path.name)
    return "".join(match.groups()) if match else ""


def _iso_stamp(ran: str) -> str:
    try:
        return datetime.fromisoformat(ran).strftime("%m%d%H%M")
    except ValueError:
        return ""


# ----------------------------------------------------------------------
# plan one directory
# ----------------------------------------------------------------------

@dataclass
class Candidate:
    path: Path
    step: int
    ran: str
    note: str
    payload: dict


@dataclass
class DirPlan:
    run: Path
    chosen: dict[int, Candidate] = field(default_factory=dict)
    passed_over: list[Candidate] = field(default_factory=list)
    untouched: list[Path] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.chosen


def plan_dir(run: Path) -> DirPlan:
    plan = DirPlan(run=run)
    by_step: dict[int, list[Candidate]] = {}
    for path in sorted(run.glob("*.json")):
        if path.name == CALIB_NAME:
            plan.notes.append(f"{CALIB_NAME} already here -- skipping this directory")
            return plan
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            plan.untouched.append(path)
            plan.notes.append(f"{path.name}: not valid JSON, left alone")
            continue
        step = classify(payload)
        if step is None:
            plan.untouched.append(path)
            continue
        ran, note = stamp_of(path)
        by_step.setdefault(step, []).append(
            Candidate(path=path, step=step, ran=ran, note=note, payload=payload))

    for step, candidates in by_step.items():
        candidates.sort(key=lambda c: c.ran)
        plan.chosen[step] = candidates[-1]
        plan.passed_over += candidates[:-1]
    return plan


def describe(plan: DirPlan) -> list[str]:
    lines = [f"{plan.run.name}/"]
    if plan.empty:
        lines += [f"    {note}" for note in plan.notes] or ["    nothing to convert"]
        return lines
    for step in STEP_ORDER:
        chosen = plan.chosen.get(step)
        if chosen is None:
            continue
        csvs, note = _csvs_for(plan.run, chosen)
        detail = ", ".join(p.name for p in csvs) or "(no csv)"
        lines.append(f"    step{step}  {chosen.ran[:16].replace('T', ' ')}  "
                     f"<- {chosen.path.name}   csv: {detail}")
        for extra in (chosen.note, note):
            if extra:
                lines.append(f"              ! {extra}")
    for other in plan.passed_over:
        lines.append(f"    passed over: {other.path.name} "
                     f"(older step{other.step}, {other.ran[:16].replace('T', ' ')})")
    for note in plan.notes:
        lines.append(f"    ! {note}")
    if plan.untouched:
        lines.append("    left as-is: "
                     + ", ".join(sorted(p.name for p in plan.untouched)))
    return lines


def _csvs_for(run: Path, chosen: Candidate) -> tuple[list[Path], str]:
    csvs = recorded_csvs(chosen.payload, chosen.step, run)
    if csvs:
        return csvs, ""
    if chosen.step == 2:
        return [run / f"step2_{_stamp_name(chosen.ran)}.csv"], "written from the JSON map"
    return matched_csv(run, chosen.step, chosen.ran)


def _stamp_name(ran: str) -> str:
    try:
        return datetime.fromisoformat(ran).strftime("%m%d_%H%M")
    except ValueError:
        return "unknown"


# ----------------------------------------------------------------------
# apply
# ----------------------------------------------------------------------

def apply_dir(plan: DirPlan, log) -> None:
    for step in STEP_ORDER:                       # in order: a write drops downstream
        chosen = plan.chosen.get(step)
        if chosen is None:
            continue
        csvs, _ = _csvs_for(plan.run, chosen)
        if step == 1:
            section = step1_section(chosen.payload)
        elif step == 2:
            section = step2_section(chosen.payload)
            rows = step2_csv(chosen.payload, csvs[0])
            log(f"    wrote {csvs[0].name} ({rows} rows)")
        elif step == 3:
            section = step3_section(chosen.payload)
        elif step == 6:
            section = dict(chosen.payload["step6"])
        elif step == 7:
            section = step7_section(chosen.payload)
        else:
            section = step8_section(chosen.payload)
        write_section(plan.run, step, section, csv=csvs or None,
                      ran=chosen.ran, log=None)

    stash = plan.run / LEGACY_DIR
    stash.mkdir(exist_ok=True)
    for chosen in list(plan.chosen.values()) + plan.passed_over:
        shutil.move(str(chosen.path), str(stash / chosen.path.name))
    log(f"    stashed {len(plan.chosen) + len(plan.passed_over)} legacy JSON(s) "
        f"in {LEGACY_DIR}/")
    calib = load_calib(plan.run)
    for line in calib.chain_lines()[1:]:
        log(f"    {line.strip()}")
    for step, newer in calib.stale().items():
        log(f"    ! step{step} predates "
            + ", ".join(f"step{n}" for n in newer)
            + " -- kept as measured, flagged on every load")


# ----------------------------------------------------------------------
# verify
# ----------------------------------------------------------------------

def verify_dir(run: Path, log) -> bool:
    """Compare the converted file against the stashed originals, section by section.

    The check that matters is the layout: step 3 lost its raw curves, so every
    channel's ``level_for`` is re-derived from the stored fit alone and must
    return the same grayscale level it did before, for every value the encoder
    can ask for.
    """
    from calibration_module.fit.phase import load_comb_phase_json, load_pair_models
    from calibration_module.fit.slm_response import calibration_result_from_dict
    from slm_module.encoding import channel_layout_from_calibration

    stash = run / LEGACY_DIR
    if not stash.is_dir():
        log(f"{run.name}: no {LEGACY_DIR}/ -- nothing to compare against")
        return True
    calib = load_calib(run)
    # Newest per step, by the same stamp rule apply_dir chose with -- comparing
    # against a different original than the one converted would prove nothing.
    newest: dict[int, tuple[str, dict]] = {}
    for path in sorted(stash.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        step = classify(payload)
        if step is None:
            continue
        ran, _ = stamp_of(path)
        if step not in newest or ran > newest[step][0]:
            newest[step] = (ran, payload)
    old = {step: payload for step, (_, payload) in newest.items()}
    ok = True

    if 3 in calib and 3 in old:
        before = channel_layout_from_calibration(
            calibration_result_from_dict(old[3]))
        after = channel_layout_from_calibration(calib.step3_result())
        ok &= _same_layout(before, after, run, log)

    if 6 in calib and (6 in old or 7 in old):
        source = old.get(7) or old[6]
        before = load_pair_models([_write_tmp(source, run)])
        after = load_pair_models([calib.path])
        ok &= _same_models(before, after, run, log)

    if 7 in calib and 7 in old:
        ref_before, before = load_comb_phase_json(_write_tmp(old[7], run))
        ref_after, after = load_comb_phase_json(calib.path)
        if ref_before != ref_after or sorted(before) != sorted(after):
            log(f"{run.name}: step7 reference or pair set differs")
            ok = False
        else:
            for k in before:
                a = before[k]["fit"].get("dphi_comb_rad")
                b = after[k]["fit"].get("dphi_comb_rad")
                if a != b:
                    log(f"{run.name}: pair {k} dphi_comb {a} -> {b}")
                    ok = False

    if 8 in calib and 8 in old:
        for key in ("y_sat_v", "y_sat_sigma_v", "gain_lever", "rmse_v"):
            if key in old[8] and old[8][key] != calib.section(8).get(key):
                log(f"{run.name}: step8 {key} {old[8][key]} -> "
                    f"{calib.section(8).get(key)}")
                ok = False

    log(f"{run.name}: {'ok' if ok else 'MISMATCH'} "
        f"({', '.join(f'step{s}' for s in calib.steps())})")
    return ok


def _write_tmp(payload: dict, run: Path) -> Path:
    """The old loaders take a path, so hand them one -- outside the run directory.

    Never inside it: a second JSON there, even for a moment, is exactly what this
    migration exists to get rid of, and an interrupted verify would leave one.
    """
    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".json", prefix=f"{run.name}_", delete=False, encoding="utf-8")
    with handle:
        json.dump(payload, handle)
    path = Path(handle.name)
    _TEMP_FILES.append(path)
    return path


def _same_layout(before, after, run: Path, log) -> bool:
    grid = np.linspace(0.0, 1.0, 101)
    pairs = list(zip(before.all_channels, after.all_channels))
    if len(before.all_channels) != len(after.all_channels):
        log(f"{run.name}: step3 channel count "
            f"{len(before.all_channels)} -> {len(after.all_channels)}")
        return False
    for index, (b, a) in enumerate(pairs):
        for value in grid:
            if b.level_for(float(value)) != a.level_for(float(value)):
                log(f"{run.name}: step3 channel {index} level_for({value:.2f}) "
                    f"{b.level_for(float(value))} -> {a.level_for(float(value))}")
                return False
    return True


def _same_models(before: dict, after: dict, run: Path, log) -> bool:
    if sorted(before) != sorted(after):
        log(f"{run.name}: step6 pairs {sorted(before)} -> {sorted(after)}")
        return False
    for k in sorted(before):
        if (before[k].eta, before[k].eta_err) != (after[k].eta, after[k].eta_err):
            log(f"{run.name}: pair {k} eta {before[k].eta} -> {after[k].eta}")
            return False
    return True


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def run_dirs(root: Path) -> list[Path]:
    """Every run directory under calib_data, archive excluded -- it is frozen."""
    return sorted(p for p in root.iterdir()
                  if p.is_dir() and p.name != "archive"
                  and any(p.glob("*.json")))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="migrate_calib_v3.py",
        description="Per-step calibration JSONs -> one calib.json per run directory.")
    parser.add_argument("--apply", action="store_true",
                        help="write calib.json and stash the legacy JSONs")
    parser.add_argument("--verify", action="store_true",
                        help="compare converted files against the stashed originals")
    parser.add_argument("--prune", action="store_true",
                        help=f"delete every {LEGACY_DIR}/ (after --verify passes)")
    parser.add_argument("--root", type=Path, default=CALIB_DATA,
                        help=f"calibration data directory (default {CALIB_DATA})")
    parser.add_argument("--only", nargs="+", metavar="NAME",
                        help="limit to these run directory names")
    args = parser.parse_args(argv)

    targets = run_dirs(args.root)
    if args.only:
        wanted = set(args.only)
        targets = [p for p in targets if p.name in wanted]
    if not targets:
        print(f"no run directories with JSONs under {args.root}")
        return 1

    if args.verify:
        failed = [p.name for p in targets if not verify_dir(p, print)]
        print()
        print("verify: " + (f"MISMATCH in {', '.join(failed)}" if failed
                            else f"all {len(targets)} directories match"))
        for path in _TEMP_FILES:
            path.unlink(missing_ok=True)
        return 1 if failed else 0

    if args.prune:
        for run in targets:
            stash = run / LEGACY_DIR
            if stash.is_dir():
                count = len(list(stash.glob("*")))
                shutil.rmtree(stash)
                print(f"{run.name}: removed {LEGACY_DIR}/ ({count} files)")
        return 0

    for run in targets:
        plan = plan_dir(run)
        for line in describe(plan):
            print(line)
        if args.apply and not plan.empty:
            apply_dir(plan, print)
        print()
    if not args.apply:
        print("plan only -- nothing written.  Re-run with --apply, then --verify.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
