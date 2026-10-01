"""Draft: run calibration steps 6 -> 7 -> 8 back to back into ONE run folder.

    python src/drafts/calib_step6-8_v2.py                  # full 6 -> 7 -> 8 run
    python src/drafts/calib_step6-8_v2.py --dry-run        # resolve inputs, touch no hardware
    python src/drafts/calib_step6-8_v2.py --pairs 1,3,5    # override the default pair list
    python src/drafts/calib_step6-8_v2.py --pairs 1,3,5 --ref 1
    python src/drafts/calib_step6-8_v2.py --verify-n 1,2,3    # step 8 at n = 1, 2 and 3
    python src/drafts/calib_step6-8_v2.py --stop-at 7         # steps 6 and 7 only
    python src/drafts/calib_step6-8_v2.py --start-at 7 --run-dir src/calib_data/calib_run_0904_1530

Every run gets a fresh directory ``src/calib_data/calib_run_<MMDD_HHMM>/`` and
*all* of the three steps' outputs -- raw CSVs, result JSONs, PNGs -- land in it.
Nothing goes to the flat ``src/calib_data`` any more, so one run is one
self-contained folder that can be zipped, moved or thrown away whole.

Wiring
------
Each step script is imported by path.  Its output directory is a module-level
constant this runner rebinds, and every step takes the same ``--run <folder>``;
steps 6 and 7 additionally take the pair list (``--pairs`` / ``--targets``) and
step 7 its ``--ref``, passed through ``main(argv)``:

===========  =========================  ==================
step         reads from calib.json      output dir rebound
===========  =========================  ==================
6 (v2)       ``step3``                  ``CALIB_PATH``
7 (v2)       ``step3`` + ``step6``      ``CALIB_PATH``
8 (verify)   ``step3`` + ``step6``      ``OUT_DIR``
             + ``step7``, once per
             ``--verify-n``
===========  =========================  ==================

Nothing is chained by this script any more.  Each step writes its section into
the run folder's one ``calib.json`` and the next step reads it from there, so
there is no result file to glob for and no path to carry -- the run folder has
to arrive with a ``step3`` section in it, which is what ``--run-dir`` names.

``--flip`` is never passed to any step: the DAQ sign on this rig is positive.

Failure
-------
A step that returns non-zero or raises stops the sequence -- the run folder
keeps whatever it produced, ``sequence.json`` records how far it got, and the
exact ``--start-at`` command to resume is printed.  Console output is teed to
``run.log`` in the run folder, so the diagnostics the steps print survive.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STEPS_DIR = REPO_ROOT / "src/calibration_module/steps"
CALIB_PATH = REPO_ROOT / "src/calib_data"       # where run folders are created

# ---- Edit these to match your setup ----
RUN_PREFIX = "run"                   # run folder is <RUN_PREFIX>_<MMDD_HHMM>

# One pair list for all three steps (step 6 PAIR_INDICES, step 7 TGT_INDICES,
# step 8 PAIRS), so the three cannot disagree -- a pair measured in 6 but missing
# from 8 is a silent hole.  --pairs overrides it; None falls back to each step's
# own hard-coded list.
#
# REF_INDEX must be IN the pair list: step 7 fits every target against it and
# needs its step-6 eta, and step 8 predicts from phases defined relative to it.
# Leaving it in TGT_INDICES too is deliberate -- step 7 then measures the
# reference against itself, which is not fitted (fit_csv drops k == REF_INDEX)
# but gives a free reference-only baseline to check step 6's eta against.
PAIRS = [2, 3, 4, 5, 6]               # pairs calibrated end to end
REF_INDEX = 2                   # step 7's reference pair (Phi = 0); must be in PAIRS

# Step 8 runs once per entry.  n = 1 checks step 6 alone (one pair has no phase
# to get wrong); n = 2 is the first check of step 7's phases.  --verify-n
# overrides it.
VERIFY_N = (1, 2)

# Per step: module stem, the constants to rebind, and its pair flags.  Every
# step now takes ``--run <run folder>`` and reads its own input out of that
# folder's one calib.json, so there is nothing for this script to carry from one
# step to the next and no result file to go looking for -- which is the whole
# point of the layout.  A step with an ``n_flag`` is run once per entry of
# VERIFY_N.
STEPS = {
    6: {"module": "calib_step6_v2", "out_attr": "CALIB_PATH",
        "needs": (3,), "pairs_attr": "PAIR_INDICES", "pairs_flag": "--pairs"},
    7: {"module": "calib_step7_v2", "out_attr": "CALIB_PATH",
        "needs": (3, 6), "pairs_attr": "TGT_INDICES", "pairs_flag": "--targets",
        "ref_flag": "--ref"},
    8: {"module": "calib_step8", "out_attr": "OUT_DIR",
        "needs": (3, 6, 7), "pairs_attr": "PAIRS"},
}


# ======================================================================
# plumbing
# ======================================================================

class _Tee:
    """Mirror a text stream into a file, so run.log holds the full console."""

    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh

    def write(self, s):
        self._stream.write(s)
        self._stream.flush()        # unbuffered: a crash must not eat the tail
        self._fh.write(s)
        self._fh.flush()
        return len(s)

    def flush(self):
        self._stream.flush()
        self._fh.flush()

    def __getattr__(self, name):    # isatty, encoding, ... -> the real stream
        return getattr(object.__getattribute__(self, "_stream"), name)


def _load_step(stem: str):
    """Import a step script by path (they live outside any importable package)."""
    if stem in sys.modules:
        return sys.modules[stem]
    path = STEPS_DIR / f"{stem}.py"
    if not path.is_file():
        raise FileNotFoundError(f"step script not found: {path}")
    spec = importlib.util.spec_from_file_location(stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[stem] = mod         # register before exec, as import does
    spec.loader.exec_module(mod)
    return mod


def make_run_dir(explicit: str | None, *, create: bool = True) -> Path:
    """Create (or reuse, with --run-dir) the folder every output goes into.

    ``create=False`` only resolves the name -- a --dry-run must not litter
    calib_data with empty run folders.
    """
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path
    stamp = time.strftime("%m%d_%H%M")
    path = CALIB_PATH / f"{RUN_PREFIX}_{stamp}"
    suffix = 0
    while path.exists():            # two runs inside the same minute
        suffix += 1
        path = CALIB_PATH / f"{RUN_PREFIX}_{stamp}_{suffix}"
    if create:
        path.mkdir(parents=True)
    return path


def _rel(path: Path) -> str:
    """Repo-relative display path, falling back to absolute for off-tree files."""
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


# ======================================================================
# the sequence
# ======================================================================

def _set_ref_index(mod, ref_index) -> int | None:
    """Bind the reference pair wherever this step keeps it; return what it is.

    Step 7 moved its drive settings into a frozen ``CONFIG`` dataclass, so its
    reference is ``CONFIG.ref_index``; the older steps keep a bare
    ``REF_INDEX`` global.  Returns None for a step that has no reference.
    """
    cfg = getattr(mod, "CONFIG", None)
    if cfg is not None and hasattr(cfg, "ref_index"):
        if ref_index is not None:
            mod.CONFIG = replace(cfg, ref_index=int(ref_index))
        return int(mod.CONFIG.ref_index)
    if hasattr(mod, "REF_INDEX"):
        if ref_index is not None:
            mod.REF_INDEX = int(ref_index)
        return int(mod.REF_INDEX)
    return None


def run_step(n: int, run_dir: Path, *, pairs=None,
             ref_index=None, dry_run: bool = False,
             verify_n=VERIFY_N) -> Path | None:
    """Point step ``n`` at this run folder and run it.

    Nothing is returned and nothing is carried: the step writes its section into
    ``run_dir/calib.json``, which is where the next step looks for it.
    """
    spec = STEPS[n]
    mod = _load_step(spec["module"])
    setattr(mod, spec["out_attr"], run_dir)     # all outputs -> the run folder
    argv: list[str] = ["--run", str(run_dir)]
    runs = ([[spec["n_flag"], str(k)] for k in verify_n]
            if spec.get("n_flag") else [[]])
    if pairs is not None:
        setattr(mod, spec["pairs_attr"], list(pairs))
    ref = _set_ref_index(mod, ref_index)
    # Steps 6 and 7 take the pair list (and 7 the reference) as required flags;
    # the step's own main() rebinds its constants from them.
    if spec.get("pairs_flag"):
        argv += [spec["pairs_flag"],
                 ",".join(str(k) for k in getattr(mod, spec["pairs_attr"]))]
    if spec.get("ref_flag") and ref is not None:
        argv += [spec["ref_flag"], str(ref)]

    head = (f"\n{'=' * 70}\n== STEP {n}  ({spec['module']})\n"
            f"==   run  : {_rel(run_dir)}\n"
            f"==   reads: {', '.join(f'step{k}' for k in spec['needs'])}\n"
            f"==   pairs: {getattr(mod, spec['pairs_attr'])}")
    if ref is not None:
        head += f"   ref: {ref}"
    if spec.get("n_flag"):
        head += f"   n: {','.join(str(k) for k in verify_n)}"
    print(f"{head}\n{'=' * 70}", flush=True)
    if dry_run:
        print("   (dry run -- not executed)")
        return None

    t0 = time.time()
    for extra in runs:
        rc = mod.main(argv + extra)  # never None: the step must not see OUR argv
        if rc:
            raise RuntimeError(
                f"step {n} ({spec['module']} {' '.join(extra)}) returned {rc}")
    print(f"\n-- step {n} finished in {(time.time() - t0) / 60:.1f} min")

    # The section it was supposed to write, or the next step has nothing to
    # read: a sweep can succeed while its fit fails, and that used to show up
    # only as a confusing error one step later.
    from calibration_module.calib_file import load_calib

    calib = load_calib(run_dir)
    if n not in calib:
        raise RuntimeError(
            f"step {n} wrote no step{n} section into {calib.path} -- the sweep "
            f"may have run while the fit failed, so there is nothing for the "
            f"next step to read"
        )
    result = calib.path
    print(f"-- step {n} wrote step{n} ({calib.ran(n)}) into {_rel(result)}")
    return result


def _write_manifest(run_dir: Path, manifest: dict) -> None:
    """Rewrite sequence.txt after each step, so a crash still leaves a record.

    Text, not JSON: a run folder holds exactly one JSON and it is the
    calibration.  Nothing reads this file -- it is for the operator reading back
    what a night did -- so a flat ``key: value`` dump is all it ever needed.
    """
    manifest["outputs"] = sorted(
        q.name for q in run_dir.iterdir()
        if q.is_file() and q.name not in ("sequence.txt", "run.log"))
    lines = []
    for key, value in manifest.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            lines += [f"  {k}: {v}" for k, v in value.items()]
        elif isinstance(value, list):
            lines.append(f"{key}:")
            lines += [f"  {v}" for v in value]
        else:
            lines.append(f"{key}: {value}")
    (run_dir / "sequence.txt").write_text("\n".join(lines) + "\n",
                                          encoding="utf-8")


def run_sequence(*, run_dir: Path, start_at: int = 6,
                 stop_at: int = 8, pairs=None, ref_index=None,
                 dry_run: bool = False, verify_n=VERIFY_N) -> int:
    order = [n for n in (6, 7, 8) if start_at <= n <= stop_at]
    manifest = {
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir": _rel(run_dir),
        "steps_run": order,
        "pairs": list(pairs) if pairs else None,
        "ref_index": ref_index,
        "verify_n": list(verify_n),
        "status": "running",
        "outputs": [],
    }

    # Every step reads its input out of the run folder's calib.json, so
    # resuming is just a question of whether the sections it needs are there.
    from calibration_module.calib_file import CalibFileError, load_calib

    if order and not dry_run:
        try:
            calib = load_calib(run_dir)
        except CalibFileError as exc:
            print(f"ERROR: {exc}")
            return 2
        missing = [k for k in STEPS[order[0]]["needs"] if k not in calib]
        if missing:
            print(f"ERROR: --start-at {order[0]} needs "
                  + ", ".join(f"step{k}" for k in missing)
                  + f" in {_rel(calib.path)}, which has "
                  + (", ".join(f"step{k}" for k in calib.steps()) or "nothing"))
            return 2
        for line in calib.chain_lines():
            print(f"  {line}")

    if not dry_run:
        _write_manifest(run_dir, manifest)

    for n in order:
        try:
            result = run_step(n, run_dir, pairs=pairs,
                              ref_index=ref_index, dry_run=dry_run,
                              verify_n=verify_n)
        except Exception as exc:
            print(f"\n!! STEP {n} FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            manifest["status"] = f"failed at step {n}: {type(exc).__name__}: {exc}"
            manifest["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
            if not dry_run:
                _write_manifest(run_dir, manifest)
            print(f"\nRun folder kept: {_rel(run_dir)}")
            print(f"Resume with:  python {_rel(Path(__file__))} "
                  f"--start-at {n} --run-dir {_rel(run_dir)}")
            return 1
        if result is not None:
            manifest[f"step{n}"] = _rel(result)
        if not dry_run:
            _write_manifest(run_dir, manifest)

    manifest["status"] = "ok"
    manifest["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if dry_run:
        print(f"\nDry run complete -- would write into {_rel(run_dir)}")
        return 0
    _write_manifest(run_dir, manifest)
    print(f"\n{'=' * 70}\nSequence complete -- everything in {_rel(run_dir)}")
    for name in manifest["outputs"]:
        print(f"   {name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Run calibration steps 6 -> 7 -> 8 into one run folder.")
    ap.add_argument("--run-dir", metavar="PATH", default=None,
                    help="reuse this folder instead of creating a new one "
                         "(required with --start-at 7/8)")
    ap.add_argument("--start-at", type=int, choices=(6, 7, 8), default=6,
                    help="skip earlier steps, reusing their results in --run-dir")
    ap.add_argument("--stop-at", type=int, choices=(6, 7, 8), default=8,
                    help="last step to run (default 8; 7 skips the n-pair verify)")
    ap.add_argument("--pairs", metavar="1,3,5", default=None,
                    help="pair list for all three steps (default: each step's own)")
    ap.add_argument("--ref", type=int, default=None,
                    help="step-7 reference pair (default: the step's own)")
    ap.add_argument("--verify-n", metavar="1,2", default=None,
                    help="pair counts step 8 verifies, one run each "
                         f"(default: {','.join(str(k) for k in VERIFY_N)})")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve and print the wiring; touch no hardware")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)

    if args.start_at > 6 and not args.run_dir:
        ap.error("--start-at 7/8 needs --run-dir (the earlier steps' outputs)")
    if args.stop_at < args.start_at:
        ap.error(f"--stop-at {args.stop_at} is before --start-at {args.start_at}")

    pairs = PAIRS
    if args.pairs:
        pairs = [int(tok) for tok in args.pairs.replace(" ", "").split(",") if tok]
    ref_index = REF_INDEX if args.ref is None else args.ref
    verify_n = VERIFY_N
    if args.verify_n:
        verify_n = tuple(int(tok) for tok in args.verify_n.replace(" ", "").split(",") if tok)
    # Caught here rather than an hour into step 7: the reference has to be one
    # of the pairs step 6 measured, or load_models() has no eta for it.
    if pairs and ref_index is not None and ref_index not in pairs:
        ap.error(f"--ref {ref_index} is not in --pairs {pairs}: step 7 fits every "
                 f"target against the reference and needs its step-6 eta")

    run_dir = make_run_dir(args.run_dir, create=not args.dry_run)

    if args.dry_run:
        print(f"Run folder : {_rel(run_dir)}")
        return run_sequence(run_dir=run_dir, start_at=args.start_at,
                            stop_at=args.stop_at,
                            pairs=pairs, ref_index=ref_index, dry_run=True,
                            verify_n=verify_n)

    log_path = run_dir / "run.log"
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} "
                 f"start-at {args.start_at} stop-at {args.stop_at} ====\n")
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout = _Tee(stdout, fh)
        sys.stderr = _Tee(stderr, fh)
        try:
            print(f"Run folder : {_rel(run_dir)}")
            print(f"Log        : {_rel(log_path)}")
            return run_sequence(run_dir=run_dir,
                                start_at=args.start_at, stop_at=args.stop_at,
                                pairs=pairs,
                                ref_index=ref_index, verify_n=verify_n)
        finally:
            sys.stdout, sys.stderr = stdout, stderr


if __name__ == "__main__":
    sys.exit(main())
