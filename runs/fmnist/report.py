"""Score a bench run of fMNIST same/different pairs: three confusion matrices.

    python runs\\fmnist\\report.py --npz runs\\fmnist\\bench_reference_200pairs.npz
                                 --csv runs\\fmnist\\onn_infer_1029_2034.csv

The other end of :mod:`onn_module.onn_infer`, which measures and writes a CSV
and scores nothing.  Everything needed to recover a verdict is in that CSV and
the npz it was run from, so this replays the decision layer offline and can be
re-run on a filtered CSV without touching the bench.

The decision layer, per pair
----------------------------
Not the ten-class argmax MNIST uses -- this task is binary, "are these two
images the same class", so there is one output and a sign::

    logit = sum_{o,t} v[o,t] * y_meas[n,o,t] + c      (> 0 means "same")

``y_meas_v`` is what ``onn_infer`` recovered (dark, gain and the single-beam
background already off).  Under the 'nobias' rule c = 0, which makes the
verdict GAIN-INVARIANT: scaling every read by a common factor cannot move a
logit across zero.  That is the property the bench is being tested against, so
a disagreement with the simulation means per-read error, never gain drift.

A pair missing any read -- a run stopped early -- is left out and counted.

Out: three 2x2 matrices in ``<csv stem>_confusion.png`` beside the CSV
(``--png``, ``--no-png``), the same three in the terminal, and the Y residual.
No CSV is written; the npz and the measurement CSV are the only records.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

CLASSES = ("different", "same")          # index == the label, 0 and 1


# ======================================================================
# the inputs
# ======================================================================

def read_reads(path: Path) -> list[dict]:
    """The onn_infer CSV's rows, whatever meta lines sit above its header."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for k, row in enumerate(csv.reader(lines)):
        if {"kind", "image", "shot", "out", "y_meas_v"} <= set(row):
            return list(csv.DictReader(lines[k:]))
    raise SystemExit(f"{path}: no header with kind, image, shot, out and y_meas_v -- "
                     f"is this an onn_infer.py CSV?")


def score(npz: Path, csv_path: Path) -> dict:
    """Everything the report needs: the two logits, the truth, the residual."""
    d = np.load(npz, allow_pickle=False)
    if "v" not in d:
        raise SystemExit(f"{npz} has no decision layer (v): it is not an fmnist pairs npz")
    idx = d["idx"]
    row_of = {int(p): n for n, p in enumerate(idx)}
    n, o, t = len(idx), int(d["n_out"]), int(d["n_tiles"])
    y_pred = np.asarray(d["y_block"], dtype=float).reshape(n, o, t)
    y_meas = np.full((n, o, t), np.nan)
    stray = 0
    for r in read_reads(csv_path):
        if r["kind"] != "onn" or r.get("y_meas_v") in ("", None):
            continue
        if int(r.get("bank") or 0) != 0:
            continue
        pair = int(r["image"])
        if pair not in row_of:
            stray += 1
            continue
        y_meas[row_of[pair], int(r["out"]), int(r["shot"])] = float(r["y_meas_v"])
    if stray:
        raise SystemExit(f"{stray} reads in {csv_path.name} are for pairs this npz does not "
                         f"have: the CSV was taken from a different npz")

    v, c = np.asarray(d["v"], dtype=float), float(d["c"])
    labels = np.asarray(d["labels"]).astype(int)
    done = ~np.isnan(y_meas).any(axis=(1, 2))
    logit_ref = np.asarray(d["logits_ref"], dtype=float)
    logit_meas = np.where(done, np.nansum(v[None] * y_meas, axis=(1, 2)) + c, np.nan)
    return dict(d=d, n=n, done=done, labels=labels, logit_ref=logit_ref,
                logit_meas=logit_meas, y_meas=y_meas, y_pred=y_pred,
                names=[str(s) for s in d["class_names"]])


# ======================================================================
# the matrices
# ======================================================================

def confuse(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """``2 x 2`` counts; ``[r, c]`` is how many of class ``r`` came out ``c``."""
    m = np.zeros((2, 2), dtype=int)
    for r, c in zip(rows, cols):
        m[int(r), int(c)] += 1
    return m


def verdicts(s: dict):
    """``(truth, simulation, bench)`` as 0/1 arrays over the complete pairs."""
    k = s["done"]
    return (s["labels"][k],
            (s["logit_ref"][k] > 0).astype(int),
            (s["logit_meas"][k] > 0).astype(int))


def panels_of(s: dict) -> list:
    """The three comparisons, as ``(title, matrix, row name, col name)``.

    C is the one that matters here: A and B can both be 94.5% and still be
    94.5% on different pairs.  Read against each other, C says whether the
    optics changed any decision at all.
    """
    true, sim, bench = verdicts(s)
    return [("A. SIMULATION", confuse(true, sim), "truth", "simulation"),
            ("B. BENCH", confuse(true, bench), "truth", "bench"),
            ("C. ALIGNMENT", confuse(sim, bench), "simulation", "bench")]


def show(title: str, m: np.ndarray, row_name: str, col_name: str) -> None:
    """One matrix as text, counts only -- it is 2x2, it fits."""
    hit = int(np.trace(m))
    print(f"\n{title}   {hit}/{m.sum()} = {100 * hit / m.sum():.1f}%")
    print(f"{row_name + ' v ' + col_name:>24}  "
          + "  ".join(f"{c:>9}" for c in CLASSES))
    for r, name in enumerate(CLASSES):
        print(f"{name:>24}  " + "  ".join(f"{m[r, c]:>9d}" for c in range(2)))


def plot(panels, out: Path, subtitle: str) -> Path:
    """The three matrices side by side, one PNG.

    Colour is the ROW fraction, the annotation the raw count: an unbalanced
    set makes two equal counts two very different recalls, and it is the
    recall the colour should show.
    """
    import matplotlib
    matplotlib.use("Agg")                     # no window, no display needed
    import matplotlib.pyplot as plt

    # constrained, not tight: tight_layout reserves no room for a suptitle and
    # lets it land on the middle panel's title.
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 5.0),
                             layout="constrained")
    for ax, (title, m, row_name, col_name) in zip(axes, panels):
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.nan_to_num(m / m.sum(axis=1, keepdims=True))
        ax.imshow(frac, cmap="Blues", vmin=0.0, vmax=1.0)
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(m[r, c]), ha="center", va="center",
                        fontsize=15,
                        color="white" if frac[r, c] > 0.5 else "0.2")
        hit = int(np.trace(m))
        ax.set_title(f"{title}\n{hit}/{m.sum()} = {100 * hit / m.sum():.1f}%",
                     fontsize=11)
        ax.set_xlabel(col_name)
        ax.set_ylabel(row_name)
        ax.set_xticks(range(2), CLASSES)
        ax.set_yticks(range(2), CLASSES, rotation=90, va="center")
        ax.tick_params(labelsize=9)
    fig.suptitle(subtitle, fontsize=10)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ======================================================================
# the report
# ======================================================================

def disagreements(s: dict) -> None:
    """Only the pairs where the bench and the simulation part company.

    The per-pair table used to print all 200 rows.  Agreement is the normal
    case and says nothing read one line at a time; a disagreement is the only
    row worth a name, a truth and two logits.
    """
    d, k = s["d"], s["done"]
    sim, bench = s["logit_ref"] > 0, s["logit_meas"] > 0
    rows = [i for i in range(s["n"]) if k[i] and sim[i] != bench[i]]
    if not rows:
        print("\nno pair where the bench and the simulation disagree")
        return
    print(f"\n{len(rows)} pair(s) where the bench and the simulation disagree:")
    print(f"{'pair':>6}  {'class a':>12} {'class b':<12} {'truth':>9} "
          f"{'model':>10} {'bench':>10}")
    for i in rows:
        print(f"{int(d['idx'][i]):6d}  "
              f"{s['names'][int(d['class_a'][i])]:>12} "
              f"{s['names'][int(d['class_b'][i])]:<12} "
              f"{CLASSES[s['labels'][i]]:>9} "
              f"{s['logit_ref'][i]:+10.3f} {s['logit_meas'][i]:+10.3f}")


def residual(s: dict) -> float:
    """Print the measured-minus-predicted Y, and return its rms in mV.

    The rms is the number to carry into ``acc_vs_read_noise_mV``: that table
    is the simulation re-run with gaussian read noise of a stated size, so the
    row nearest this rms is what the accuracy was expected to be.
    """
    k = s["done"]
    yp = s["y_pred"][k].ravel() * 1e3
    ym = s["y_meas"][k].ravel() * 1e3
    r = ym - yp
    rms = float(np.sqrt((r ** 2).mean()))
    slope = float((yp * ym).sum() / (yp * yp).sum())
    print(f"\nY measured - predicted   mean {r.mean():+.3f} mV   rms {rms:.3f} mV"
          f"   (predicted Y: mean {yp.mean():.3f} mV)"
          f"   best-fit measured/predicted {slope:.3f}")
    return rms


def noise_table(s: dict, rms: float) -> None:
    """The npz's simulated accuracy-vs-read-noise curve, with ours marked."""
    meta = json.loads(str(s["d"]["meta"]))
    tol = meta.get("acc_vs_read_noise_mV")
    if not tol:
        return
    pairs = sorted((float(mv), float(acc)) for mv, acc in tol.items())
    near = min(pairs, key=lambda p: abs(p[0] - rms))
    print("simulated accuracy vs read noise:  "
          + "  ".join(f"{'>' if (mv, acc) == near else ''}{mv:g} mV {100 * acc:.1f}%"
                      for mv, acc in pairs))
    print(f"   '>' marks the row nearest this run's {rms:.3f} mV rms "
          f"-- the accuracy the noise alone predicts")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz", required=True, type=Path, help="bench_reference_<N>pairs.npz")
    p.add_argument("--csv", required=True, type=Path, help="the onn_infer.py CSV taken from it")
    p.add_argument("--png", type=Path, default=None, metavar="PATH",
                   help="where to write the figure (default: beside the CSV, "
                        "<csv stem>_confusion.png)")
    p.add_argument("--no-png", action="store_true", help="terminal only")
    a = p.parse_args(argv)

    s = score(a.npz, a.csv)
    m = int(s["done"].sum())
    if not m:
        raise SystemExit("no pair was read all the way through; nothing to score")

    true, sim, bench = verdicts(s)
    print(f"csv       : {a.csv}")
    print(f"npz       : {a.npz}   {s['n']} pairs, "
          f"{int(s['d']['n_tiles'])} shots, {int(s['d']['n_out'])} output")
    print(f"scored    : {m} complete pair(s)"
          + (f", {s['n'] - m} dropped as partial" if m != s["n"] else ""))
    print(f"balance   : {int((true == 1).sum())} same, {int((true == 0).sum())} different")

    panels = panels_of(s)
    for panel in panels:
        show(*panel)

    print()
    for name, hit, note in (("simulation", sim == true, ""),
                            ("bench", bench == true, ""),
                            ("agreement", bench == sim, "bench vs simulation")):
        print(f"{name:<12} {int(hit.sum()):>4}/{m}  {100 * hit.mean():>5.1f}%"
              + (f"   {note}" if note else ""))

    disagreements(s)
    noise_table(s, residual(s))

    if not a.no_png:
        out = a.png or a.csv.with_name(a.csv.stem + "_confusion.png")
        sub = (f"{a.csv.name}   {m} pair(s)"
               + (f", {s['n'] - m} partial dropped" if m != s["n"] else "")
               + f"   |   {a.npz.name}   |   "
                 f"logit = v . y + c > 0, c = 0 (gain-invariant)")
        print(f"\nwrote {plot(panels, out, sub)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
