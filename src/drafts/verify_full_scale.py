"""Draft -- check step 8's claimed full scale Y_max on the bench.

    python src/drafts/verify_full_scale.py --step7 RESULT.json              # measure
    python src/drafts/verify_full_scale.py --step7 RESULT.json --dry-run    # JSON + table only
    python src/drafts/verify_full_scale.py --step7 RESULT.json --p 2 3 4    # a subset of pairs

Step 8 quotes every error against FS = Y_max, the largest |E|^2 the pairs can
be driven to with their measured Phi_k (verify.reachable_max).  Nothing on the
bench has ever been driven there -- step 8's random patterns stay far below it.
This drives every selected pair at once with a handful of patterns and reads
each one:

    ymax        the drive reachable_max says attains Y_max
    all_on      x = w = 1 on every pair                (typically ~1/3 of Y_max)
    psi+DEG     the support-optimal drive for axis psi* + DEG
    psi-DEG     ... and psi* - DEG                     (both must read LOWER)
    ymax_again  the ymax drive once more               (drift over the run)

If ``ymax`` reads close to its prediction AND both psi-rotated patterns fall off
on either side, the Phi_k and the full scale are confirmed.  ``all_on`` matching
while ``ymax`` reads low means the phases are off: all_on barely depends on them.

The patterns are computed from --step7 at run time and written to
``calib_fullscale_<MMDD_HHMM>.json`` BEFORE the hardware is touched; the reads
go to a step-8-format CSV beside it (partial if the run is stopped).  The
acquisition -- dark, per-read range, near-rail escalation, sign check -- is step
8's own ``run_patterns``, unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from calibration_module.fit.verify import (  # noqa: E402
    full_scale,
    load_forward_model,
    pairs_tag,
    predict,
    support_h,
    write_verify_csv,
)
from calibration_module.steps import calib_step8 as S8  # noqa: E402

PSI_OFFSET_DEG = 30.0       # the two rotated controls sit at psi* +/- this


def drive_at_psi(fs, psi: float) -> np.ndarray:
    """``x = w`` per pair maximising the projection of E onto axis ``psi``.

    At ``psi = fs.psi`` this is exactly ``fs.drive``; away from it the phasors
    line up on the wrong axis, so ``|E|^2`` must come out below ``Y_max``.
    """
    t = np.array([support_h(np.array([psi]), eta, phi)[1][0]
                  for eta, phi in zip(fs.amplitudes, fs.phases)])
    return (1.0 - np.cos(t)) / 2.0


def coherent(models, phases, pairs, x, w) -> float:
    """``|E|^2`` alone -- what ``Y_max`` is quoted as (no single-beam background)."""
    zero = {k: type(models[k])(index=models[k].index, eta=models[k].eta,
                               a_x=0.0, q_x=0.0, a_w=0.0, q_w=0.0, d=0.0)
            for k in pairs}
    return float(predict(zero, phases, pairs, x, w))


def build_patterns(models, phases, pairs, offset_deg: float):
    """``(fs, [(name, drive), ...])`` -- every pattern drives x = w."""
    fs = full_scale(models, phases, pairs)
    off = np.radians(offset_deg)
    tag = f"{offset_deg:g}"
    patterns = [
        ("ymax", np.asarray(fs.drive)),
        ("all_on", np.ones(len(pairs))),
        (f"psi+{tag}", drive_at_psi(fs, fs.psi + off)),
        (f"psi-{tag}", drive_at_psi(fs, fs.psi - off)),
        ("ymax_again", np.asarray(fs.drive)),
    ]
    # 6 decimals is what the CSV stores, so a prediction recomputed from the
    # file is the one made here.
    return fs, [(name, np.round(np.clip(d, 0.0, 1.0), 6)) for name, d in patterns]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="verify_full_scale.py",
        description="Drive the claimed Y_max pattern (plus controls) and read it.")
    parser.add_argument("--step7", required=True, type=Path, metavar="JSON",
                        help="combined step-7 result JSON (required)")
    parser.add_argument("--p", nargs="+", type=int, default=None, metavar="PAIR",
                        help="only these pairs (default: every calibrated pair)")
    parser.add_argument("--psi-offset", type=float, default=PSI_OFFSET_DEG,
                        metavar="DEG", help=f"rotation of the two controls "
                                            f"(default {PSI_OFFSET_DEG:g})")
    parser.add_argument("--method", default=S8.PHASE_METHOD,
                        help="stored step-7 fit to predict from")
    parser.add_argument("--out", type=Path, default=None, metavar="DIR",
                        help="directory for the JSON and CSV (default: the "
                             "step-7 file's folder)")
    parser.add_argument("--dry-run", action="store_true",
                        help="write the JSON and print the table; touch no hardware")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    step7 = args.step7.resolve()
    out = (args.out or step7.parent).resolve()
    out.mkdir(parents=True, exist_ok=True)
    models, phases, ref = load_forward_model(step7, pairs=S8.select_pairs(args.p),
                                             method=args.method)
    pairs = sorted(models)
    fs, patterns = build_patterns(models, phases, pairs, args.psi_offset)

    preds = [float(predict(models, phases, pairs, d, d)) for _, d in patterns]
    cohs = [coherent(models, phases, pairs, d, d) for _, d in patterns]
    print(f"Step 7 in : {step7}")
    print(f"Pairs     : {pairs}  (reference {ref})")
    print(f"Y_max     : {fs.fs * 1e3:.4f} mV coherent   S^2 {fs.bound * 1e3:.4f} mV "
          f"({100 * fs.headroom:.1f} %)   psi* {np.degrees(fs.psi):.2f} deg")
    print(f"\n  {'pattern':<11s}  {'|E|^2 (mV)':>10s}  {'pred (mV)':>9s}   x = w per pair")
    for (name, d), c, p in zip(patterns, cohs, preds):
        print(f"  {name:<11s}  {c * 1e3:10.4f}  {p * 1e3:9.4f}   "
              + " ".join(f"{v:.3f}" for v in d))

    stamp = time.strftime("%m%d_%H%M")
    json_path = out / f"calib_fullscale_{stamp}.json"
    json_path.write_text(json.dumps({
        "step7_json": str(step7),
        "pairs": pairs,
        "reference": ref,
        "method": args.method,
        "Y_max_v": fs.fs,
        "S2_v": fs.bound,
        "psi_rad": fs.psi,
        "psi_offset_deg": args.psi_offset,
        "eta": dict(zip(map(str, pairs), fs.amplitudes)),
        "phi_rad": dict(zip(map(str, pairs), fs.phases)),
        "patterns": [
            {"name": name, "coherent_v": c, "pred_v": p,
             "x": dict(zip(map(str, pairs), map(float, d))),
             "w": dict(zip(map(str, pairs), map(float, d)))}
            for (name, d), c, p in zip(patterns, cohs, preds)
        ],
    }, indent=2), encoding="utf-8")
    print(f"\nPatterns written to {json_path}")
    if args.dry_run:
        return 0

    drives = np.stack([d for _, d in patterns])[None]      # (1 block, P, n)
    block = [tuple(pairs)]
    slm = S8.connect_slm(S8.SLM_DISPLAY_NO, S8.USB_SLM_NO)
    daq = S8.connect_daq(device=S8.DAQ_DEVICE, channel=S8.DAQ_CHANNEL,
                         t_both=S8.ACQ.t_meas_s, t_single=S8.ACQ.t_meas_s,
                         min_val=-S8.ACQ.range_v, max_val=S8.ACQ.range_v)
    layout = S8.load_layout(step7, pairs)
    rows: list[dict] = []
    status = "complete"
    try:
        S8.run_patterns(daq, slm, layout, block, drives, drives, models, phases,
                        rows=rows)
    except BaseException as exc:
        status = f"partial ({type(exc).__name__} after {len(rows)} of {len(patterns)} reads)"
        raise
    finally:
        try:
            if rows:
                csv_path = write_verify_csv(
                    out / f"calib_fullscale_{stamp}.csv", pairs, rows,
                    meta={"step7_json": step7, "n_pairs": len(pairs),
                          "patterns": " ".join(n for n, _ in patterns),
                          "patterns_json": json_path, "status": status})
                print(f"\nCSV ({len(rows)} reads, {status}) written to {csv_path}")
        finally:
            slm.close_slm()
            daq.disconnect()

    meas = {name: r["mean_v"] - r["dark_v"] for (name, _), r in zip(patterns, rows)}
    print(f"\n=== full scale check, pairs {pairs_tag(pairs)} ===")
    print(f"  {'pattern':<11s}  {'meas (mV)':>9s}  {'pred (mV)':>9s}  {'diff (mV)':>9s}"
          f"  {'meas/pred':>9s}  {'std (mV)':>8s}  range")
    for (name, _), p, r in zip(patterns, preds, rows):
        m = meas[name]
        print(f"  {name:<11s}  {m * 1e3:9.4f}  {p * 1e3:9.4f}  {(m - p) * 1e3:+9.4f}"
              f"  {m / p if p else float('nan'):9.3f}  {r['std_v'] * 1e3:8.4f}"
              f"  +/-{r['range_v']:g} V")
    top = (meas["ymax"] + meas["ymax_again"]) / 2.0
    rot = [n for n, _ in patterns if n.startswith("psi")]
    print(f"\n  ymax / all_on  : meas {top / meas['all_on']:.2f}   "
          f"pred {preds[0] / preds[1]:.2f}")
    print(f"  ymax drift     : {(meas['ymax_again'] - meas['ymax']) * 1e3:+.4f} mV over the run")
    print(f"  peak check     : ymax (mean of both) {top * 1e3:.4f} mV vs "
          + ", ".join(f"{n} {meas[n] * 1e3:.4f}" for n in rot)
          + ("  -> ymax is the brightest" if all(top > meas[n] for n in rot)
             else "  -> a rotated pattern reads BRIGHTER: the phases are off"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
