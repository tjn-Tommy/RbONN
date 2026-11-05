"""Score an ONN inference CSV: three confusion matrices and three accuracies.

    python report.py onn_infer_1027_2145_clean.csv bench_reference_200img.npz

The matrices print to the terminal and are also drawn side by side into
``<csv stem>_confusion.png`` beside the CSV (``--png``, ``--no-png``).

The other end of :mod:`onn_module.onn_infer`, which measures and writes a CSV
and scores nothing.  Everything needed to recover the classes is in that CSV
and the npz it was run from, so this replays the decision layer offline and can
be re-run on a filtered CSV without touching the bench.

The decision layer, per image
-----------------------------
A row of the CSV is one ``(bank, image, shot, output)`` read, carrying the Y
the bench recovered::

    y_sum[i, o] = sum over shots and banks of  sign(bank) * y_meas_v
    logits      = y_sum[i] * out_scale          (out_scale from the npz)
    class       = argmax(logits)

``sign`` is +1 for bank 0 and -1 for bank 1 -- the differential split that
gives signed weights a non-negative optical channel.  ``out_scale`` is a
PER-OUTPUT rescale, so it is not argmax-invariant and has to be applied.  The
simulation's own ``logits_ref`` is ``out_scale`` times the model's ``y_sum``,
so both sides pass through the identical readout and the difference between
them is the optics alone.

Only images read ALL the way through are scored: an interrupted run leaves a
partial image whose missing shots would read as zero and drag its class.  All
three matrices are built on that same subset, which is the only way the third
one means anything.

This script filters nothing else.  To exclude images, hand it a CSV with those
rows already removed.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


# ======================================================================
# the inputs
# ======================================================================

def load_npz(path: Path) -> dict:
    """The reference dump, only the fields the decision layer needs."""
    d = np.load(path, allow_pickle=False)
    return {
        "idx": np.asarray(d["idx"], dtype=int),
        "labels": np.asarray(d["labels"], dtype=int),
        "logits_ref": np.asarray(d["logits_ref"], dtype=float),
        "out_scale": (np.asarray(d["out_scale"], dtype=float)
                      if "out_scale" in d else np.ones(int(d["n_out"]))),
        "n_out": int(d["n_out"]),
        "n_tiles": int(d["n_tiles"]),
        "n_banks": int(d["n_banks"]) if "n_banks" in d else 1,
    }


def recovered_y(row: dict) -> float:
    """One read's Y: the column if it is filled, else re-derived from volts.

        y = (voltage_mean_v - dark_v) / g - sb_pred_v

    ``onn_infer`` writes ``y_meas_v`` for every inference read and rewrites
    them all with a time-interpolated ``g`` when the run ends.  A run killed
    before that leaves the last flush blank, so those rows are recomputed here
    at whatever ``g`` was current -- a step instead of a ramp, but a number
    rather than a hole.
    """
    y = row.get("y_meas_v")
    if y not in ("", None):
        return float(y)
    g = row.get("g")
    if g in ("", None):
        raise SystemExit(f"read seq {row.get('seq')} has neither y_meas_v nor g; "
                         f"nothing in the row recovers its Y")
    return ((float(row["voltage_mean_v"]) - float(row["dark_v"])) / float(g)
            - float(row["sb_pred_v"]))


def read_csv(path: Path, n_out: int):
    """``(y_sum, counts)`` per image, accumulated over shots and banks.

    ``kind == "onn"`` keeps the inference reads and drops both the gain anchors
    and the trailing ``# key,value`` metadata lines in one test.
    """
    tot: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(n_out))
    cnt: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(n_out, dtype=int))
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("kind") != "onn":
                continue
            i = int(row["image"])
            sign = -1.0 if int(row["bank"] or 0) else 1.0
            tot[i][int(row["out"])] += sign * recovered_y(row)
            cnt[i][int(row["out"])] += 1
    if not tot:
        raise SystemExit(f"{path}: no kind=onn rows -- is this an inference CSV?")
    return dict(tot), dict(cnt)


# ======================================================================
# the classes
# ======================================================================

def classify(tot, cnt, ref: dict):
    """``(images, true, sim, bench, n_partial)`` over the complete images only.

    An image is complete when every output got ``n_tiles * n_banks`` reads.
    The CSV's ``image`` column is the DATASET index, not a position, so it is
    looked up through the npz's ``idx``.
    """
    want = ref["n_tiles"] * ref["n_banks"]
    whole = sorted(i for i in tot if cnt[i].min() == want)
    pos = {int(v): j for j, v in enumerate(ref["idx"])}
    missing = [i for i in whole if i not in pos]
    if missing:
        raise SystemExit(f"image(s) {missing[:5]} are in the CSV but not in the "
                         f"npz idx -- the CSV and the npz are from different runs")
    rows = [pos[i] for i in whole]
    true = ref["labels"][rows]
    sim = np.argmax(ref["logits_ref"][rows], axis=1)
    bench = np.array([int(np.argmax(tot[i] * ref["out_scale"])) for i in whole])
    return np.array(whole), true, sim, bench, len(tot) - len(whole)


def confuse(rows: np.ndarray, cols: np.ndarray, n: int) -> np.ndarray:
    """``n x n`` counts; ``[r, c]`` is how many of class ``r`` came out ``c``."""
    m = np.zeros((n, n), dtype=int)
    for r, c in zip(rows, cols):
        m[int(r), int(c)] += 1
    return m


# ======================================================================
# the report
# ======================================================================


def show(title: str, m, row_name: str, col_name: str, rate: str) -> None:
    """One matrix as text: counts, then the per-row rate in the last column.

    A zero prints as "." so the diagonal stands out of a 10x10 of mostly
    zeros, and the rate column is the recall (or the match rate for C).
    """
    n = m.shape[0]
    hit = int(np.trace(m))
    print()
    print(f"{title}   {hit}/{m.sum()} = {100 * hit / m.sum():.1f}%")
    print(f"{row_name + ' v ' + col_name:>30}  "
          + "  ".join(f"{c:>4}" for c in range(n)) + f"  {rate:>7}")
    for r in range(n):
        tot = m[r].sum()
        pct = f"{100 * m[r, r] / tot:.0f}%" if tot else "-"
        print(f"{r:>30}  "
              + "  ".join(f"{m[r, c]:>4d}" if m[r, c] else "   ."
                          for c in range(n))
              + f"  {pct:>7}")


def plot(panels, out: Path, subtitle: str) -> Path:
    """The three matrices side by side, one PNG.

    Colour is the ROW fraction, the annotation the raw count: a row of 8 with
    one error and a row of 19 with one error are the same count and very
    different recalls, and it is the recall the colour should show.  Side by
    side because C is only meaningful read against A and B.
    """
    import matplotlib
    matplotlib.use("Agg")                     # no window, no display needed
    import matplotlib.pyplot as plt

    # constrained, not tight: tight_layout reserves no room for a suptitle and
    # lets it land on the middle panel's title.
    fig, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 5.6),
                             layout="constrained")
    for ax, (title, m, row_name, col_name, _rate) in zip(axes, panels):
        n = m.shape[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.nan_to_num(m / m.sum(axis=1, keepdims=True))
        ax.imshow(frac, cmap="Blues", vmin=0.0, vmax=1.0)
        for r in range(n):
            for c in range(n):
                if m[r, c]:
                    ax.text(c, r, str(m[r, c]), ha="center", va="center",
                            fontsize=8,
                            color="white" if frac[r, c] > 0.5 else "0.2")
        hit = int(np.trace(m))
        ax.set_title(f"{title}\n{hit}/{m.sum()} = {100 * hit / m.sum():.1f}%",
                     fontsize=11)
        ax.set_xlabel(col_name)
        ax.set_ylabel(row_name)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.tick_params(labelsize=8)
    fig.suptitle(subtitle, fontsize=10)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("csv", type=Path, help="an onn_infer CSV")
    p.add_argument("npz", type=Path, help="the bench_reference npz it was run from")
    p.add_argument("--png", type=Path, default=None, metavar="PATH",
                   help="where to write the figure (default: beside the CSV, "
                        "<csv stem>_confusion.png)")
    p.add_argument("--no-png", action="store_true", help="terminal only")
    args = p.parse_args(argv)

    ref = load_npz(args.npz)
    tot, cnt = read_csv(args.csv, ref["n_out"])
    images, true, sim, bench, n_partial = classify(tot, cnt, ref)
    n = len(images)
    if not n:
        raise SystemExit("no image was read all the way through; nothing to score")

    print(f"csv       : {args.csv}")
    print(f"npz       : {args.npz}   {len(ref['idx'])} images, "
          f"{ref['n_tiles']} shots, {ref['n_out']} outputs, "
          f"{ref['n_banks']} bank(s)")
    print(f"scored    : {n} complete image(s)"
          + (f", {n_partial} dropped as partial" if n_partial else ""))

    k = ref["n_out"]
    panels = [
        ("A. SIMULATION", confuse(true, sim, k),
         "true label", "simulation class", "recall"),
        ("B. BENCH", confuse(true, bench, k),
         "true label", "bench class", "recall"),
        ("C. ALIGNMENT", confuse(sim, bench, k),
         "simulation class", "bench class", "match"),
    ]
    for panel in panels:
        show(*panel)

    png = None
    if not args.no_png:
        png = args.png or args.csv.with_name(args.csv.stem + "_confusion.png")

    all_sim = int((np.argmax(ref["logits_ref"], axis=1) == ref["labels"]).sum())
    n_all = len(ref["idx"])
    print()
    for name, hit, note in (("simulation", sim == true, ""),
                            ("bench", bench == true, ""),
                            ("agreement", bench == sim, "bench vs simulation")):
        print(f"{name:<12} {int(hit.sum()):>4}/{n}  {100 * hit.mean():>5.1f}%"
              + (f"   {note}" if note else ""))
    print(f"\nthe same simulation over all {n_all} npz images: "
          f"{all_sim}/{n_all}  {100 * all_sim / n_all:.1f}%"
          f"   -- whether the scored subset is representative")

    if png is not None:
        sub = (f"{args.csv.name}   {n} image(s)"
               + (f", {n_partial} partial dropped" if n_partial else "")
               + f"   |   {args.npz.name}")
        print()
        print(f"wrote {plot(panels, png, sub)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
