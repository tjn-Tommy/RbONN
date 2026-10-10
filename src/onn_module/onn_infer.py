"""Run one inference on the bench: display what the npz says, read what it predicts.

The other end of :mod:`onn_module.bench_reference`: that file writes down the
drives and the ``Y`` each should produce, this one displays them, reads the
detector, undoes the instrument response and reports whether the optics picked
the right digit.

In:  a ``bench_reference.npz`` and the run's ``calib.json`` (``--calib``),
     whose forward model the gain is measured against.  ``--t-meas`` is the
     window for every read --
     inference, anchor and the opening dark alike -- and defaults to the bench
     window :data:`~calibration_module.measure.pair_v2.T_MEAS_S`, the one steps
     6-8 read at.
Out: a CSV, one row per read: the volts, the recovered ``Y``, the ``g`` used,
     and ``kind=gain`` rows for the anchors it was built from.

How the instrument response is undone.  The detector sees, for gain ``g`` and
the step-6 single-beam response ``sb``,

    V = g * [ Y + sb(x, w) ] + dark

so ``Y = (V - dark) / g - sb``, in that order -- ``sb`` is light that went
through the same gain, and step 6 fits it where ``g`` is 1 by definition.  ``g``
is measured on :data:`GAIN_RUNGS` step-8-style ladder rungs spread across
:data:`GAIN_BAND`.  The run's first block fits ``g0``, the origin-slope of
``(V - dark)`` against the model's linear prediction of those rungs; every
later block replays the same drives verbatim -- so eta, Phi and Y_sat cancel
-- and fits the ratio ``s`` of its volts to the first block's, ``g = g0 * s``.
``g`` is interpolated in time between blocks (:func:`apply_gain`).  The dark is
read once and held.

Runs in ``rbonn``, not ``rbonn-ml``: this imports the SLM and DAQ drivers and no
torch.  The npz is the seam, read with numpy alone.

With ``--no-gain`` nothing measures the gain and ``g = 1`` means "whatever it
was when the step-7 fit behind this npz was taken".  The CSV meta then records
``g_reference = 1 (assumed, ...)``.

    python src/onn_module/onn_infer.py --npz runs/onn784_ctrl/bench_reference.npz
        --calib src/calib_data/<run>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from calibration_module.calib_file import calib_path  # noqa: E402
from calibration_module.fit.verify import (  # noqa: E402
    full_scale,
    load_forward_model,
    write_verify_csv,
)
from calibration_module.measure.bench import connect_daq, connect_slm  # noqa: E402
from calibration_module.measure.patterns import (  # noqa: E402
    DEFAULT_ACQ,
    load_layout,
    run_patterns,
    select_pairs,
)
from calibration_module.measure.pmt_temp import open_pmt  # noqa: E402
from calibration_module.steps.calib_step8 import (  # noqa: E402
    LADDER_CEILING_V,
    alpha_for_volts,
    build as build_ladder,
)

# ---- Edit these to match your setup ----
SLM_DISPLAY_NO = None       # None -> auto-detect the LCOS-SLM display
USB_SLM_NO = 1              # SLM_Ctrl_* device index for the DVI-mode switch

DAQ_DEVICE = "Dev1"
DAQ_CHANNEL = "ai0"

T_MEAS_S = 4.8        # DAQ window per read, seconds; --t-meas overrides

ACQ = replace(DEFAULT_ACQ, t_meas_s=T_MEAS_S)   # the bench acquisition

GAIN_EVERY_S = 1200.0       # re-measure g (and the PMT temperature) this often (20 min)

#: The gain anchors: they measure the system's sensitivity drift as ``g``, which
#: an inference read then divides out -- ``Y = (V - dark) / g - sb``.
#: How: replay ``GAIN_RUNGS`` of step 8's own ladder rungs, spread over
#: ``GAIN_BAND`` as fractions of ``Y_max``, and fit the slope through the origin
#: -- the first block's against the model (``g0``), every later one's against
#: the first block's volts.
GAIN_BAND = (0.4, 0.8)
GAIN_RUNGS = 10

#: A rung more than this (relative) off its block's fitted slope is dropped from
#: that block's fit.  Inference reads are never checked.
GAIN_TOL = 0.04

OUT_DIR = REPO_ROOT / "runs"

#: Extra CSV columns, in order.  The first group is fixed before the run
#: (``extra``); the second can only be filled after a read (``row_hook``).
BEFORE = ["kind", "seq", "image", "label", "shot", "out", "bank",
          "alpha", "g", "g_rms_v", "sb_pred_v", "y_pred_v"]

#: Every key ``BEFORE`` names, blank.  Both row kinds fill the same columns, so
#: a reader never has to know which keys a given ``kind`` happens to carry.
BLANK = {k: "" for k in BEFORE}
AFTER = ["t_s", "y_meas_v", "resid_v"]


# ======================================================================
# the npz
# ======================================================================

def load_npz(path: Path) -> dict:
    """Unpack the reference dump into plain arrays, banks normalised away.

    ``W`` and ``y_block`` lose their bank axis when there is only one, so both
    are reshaped back to a leading bank dimension here.
    """
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    n_banks = int(d["n_banks"]) if "n_banks" in d else 1
    x = np.asarray(d["x"], dtype=float)                      # (N, T*K)
    w = np.asarray(d["W"], dtype=float)
    y = np.asarray(d["y_block"], dtype=float)
    if n_banks == 1:
        w, y = w[None], y[None]                              # (1, O, T*K), (1, N, O, T)
    n, t, k = x.shape[0], int(d["n_tiles"]), int(d["n_channels"])
    o = int(d["n_out"])
    return {
        "x": x.reshape(n, t, k),
        "w": w.reshape(n_banks, o, t, k),
        "y_pred": y.reshape(n_banks, n, o, t),
        "idx": np.asarray(d["idx"], dtype=int),
        "labels": np.asarray(d["labels"], dtype=int),
        "channels": [int(c) for c in d["channels"]],
        "y_sat": float(d["y_sat"]),
        "out_scale": (np.asarray(d["out_scale"], dtype=float)
                      if "out_scale" in d else np.ones(o)),
        "logits_ref": np.asarray(d["logits_ref"], dtype=float),
        "n": n, "n_tiles": t, "n_channels": k, "n_out": o, "n_banks": n_banks,
        "step7_json": str(d["step7_json"]) if "step7_json" in d else meta.get("calib"),
        "meta": meta, "path": str(path),
    }


# ======================================================================
# the gain
# ======================================================================

def build_anchors(models, phases, pairs: list[int], *, seed: int,
                  n: int = GAIN_RUNGS, band=GAIN_BAND) -> dict:
    """The gain ladder to replay: ``n`` solved rungs over ``band``, in alpha order.

    In:  the forward model and a seed.  ``band`` is in fractions of ``Y_max``,
         its top capped at :data:`LADDER_CEILING_V` as step 8 caps its own.
    Out: per rung, its drive, its achieved alpha, its linear prediction in
         volts, ``v_ref = None`` and ``g0 = None`` (the first block's volts and
         its gain against the model, which :func:`measure_gain` fills) and
         ``keep = True``.

    How: solved with step 8's :func:`build` -- a random drive moved by
    least-move projection onto its level set -- so a rung here is one of its
    rungs.  Only the drive matters for the ratio.
    """
    lo, hi = float(band[0]), float(band[1])
    cap = alpha_for_volts(full_scale(models, phases, pairs).fs)
    scale, patterns, preds = build_ladder(models, phases, pairs, seed=int(seed),
                                          n=int(n), alpha_lo=lo,
                                          alpha_hi=min(hi, cap))
    order = np.argsort([p.alpha for p in patterns])
    patterns = [patterns[i] for i in order]
    preds = [preds[i] for i in order]
    return {
        "x": np.array([p.x for p in patterns], dtype=float),
        "w": np.array([p.w for p in patterns], dtype=float),
        "alpha": np.array([p.alpha for p in patterns], dtype=float),
        "pred_v": np.array(preds, dtype=float),
        "labels": [p.label for p in patterns],
        "exact": [bool(p.exact) for p in patterns],
        "v_ref": None,
        "g0": None,
        "keep": np.ones(len(patterns), dtype=bool),
        "band": (lo, float(min(hi, cap))),
        "capped": bool(cap < hi),
        "y_max_v": float(scale.fs),
        "n": len(patterns),
        "seed": int(seed),
    }


def fit_gain(v_now: np.ndarray, v_ref: np.ndarray) -> tuple[float, float]:
    """``(g, rms)`` -- least squares slope through the origin, and its residual.

    Through the origin: the dark is already off both sides.  ``rms`` is the
    residual in volts, and says whether ``g`` alone explains the block -- a pure
    gain change scales every rung by one factor, so anything else lands here.
    """
    v_now = np.asarray(v_now, dtype=float)
    v_ref = np.asarray(v_ref, dtype=float)
    g = float((v_ref * v_now).sum() / (v_ref ** 2).sum())
    rms = float(np.sqrt(np.mean((v_now - g * v_ref) ** 2)))
    return g, rms


def flag_off_rungs(v, ref, *, tol: float = GAIN_TOL) -> np.ndarray:
    """Mask of rungs more than ``tol`` (relative) off the block's fitted slope.

    How: the slope is :func:`fit_gain`'s ``v = g * ref``, the block's common
    factor, so a rung whose ``v / (g * ref)`` is off by more than ``tol`` is
    dropped.  Relative, so the test is the same at 200 mV as at 450 mV.  A
    deviation that is not a number (a reference at zero) counts as off.
    """
    v = np.asarray(v, dtype=float)
    ref = np.asarray(ref, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        g, _ = fit_gain(v, ref)
        dev = np.abs(v / (g * ref) - 1.0)
    return ~(dev <= float(tol))


def _read_rungs(bench, anchors: dict, idx, *, seq0: int, t0: float,
                dark: float | None, pmt, log) -> list[dict]:
    """Display the rungs ``idx`` once each; their rows, in read order."""
    idx = np.asarray(idx, dtype=int)
    extra = [[dict(BLANK, kind="gain", seq=seq0 - k,
                   alpha=f"{anchors['alpha'][i]:.6f}") for k, i in enumerate(idx)]]
    got: list[dict] = []
    run_patterns(bench["daq"], bench["slm"], bench["layout"], [tuple(bench["pairs"])],
                 anchors["x"][idx][None], anchors["w"][idx][None],
                 bench["models"], bench["phases"], rows=got, extra=extra,
                 acq=bench["acq"], y_sat=bench["y_sat"], dark=dark, log=log,
                 row_hook=_hook(t0, pmt, live=False))
    return got


def measure_gain(bench, anchors: dict, *, seq0: int, t0: float, rows: list,
                 dark: float | None, pmt=None, log=print) -> tuple[float, float, float]:
    """Replay the ladder once; return ``(g, rms, dark)`` against the first block.

    In:  ``dark``, ``None`` on the first call (which measures one) and that
         value on every call after.
    Out: ``(g, rms, dark)``, and every read appended to ``rows`` in read order
         -- ``kind=gain`` for the ones the fit used, ``kind=gain_drop`` for the
         rest.

    How: the first call fills ``anchors["v_ref"]`` with what the rungs read and
    ``anchors["g0"]`` with their origin-slope against the model's prediction,
    and returns ``g = g0``; every later call fits the ratio ``s`` of its volts
    to the first block's and returns ``g = g0 * s`` -- ``g`` stays in the
    model's units, and no model enters the ratio.  Each
    block is checked on its own with :func:`flag_off_rungs` at
    :data:`GAIN_TOL`.  The first block judges against the model's prediction and
    re-reads what it drops once -- off again, the rung leaves the ladder for the
    run (``anchors["keep"]``).  Later blocks read only the rungs still in it.

    The PMT is read once per block, so the gain rows and every inference row
    until the next block carry that temperature.
    """
    n = anchors["n"]
    alpha = anchors["alpha"]
    if pmt is not None:
        pmt.read()
    first = anchors["v_ref"] is None
    idx = np.arange(n) if first else np.flatnonzero(anchors["keep"])
    reads = _read_rungs(bench, anchors, idx, seq0=seq0, t0=t0, dark=dark,
                        pmt=pmt, log=log)
    dark = float(reads[0]["dark_v"])
    last: dict[int, dict] = dict(zip(idx.tolist(), reads))   # the read each rung is judged by
    v = np.full(n, np.nan)
    for i, r in last.items():
        v[i] = r["mean_v"] - r["dark_v"]

    def alphas(mask) -> str:
        return " ".join(f"{a:.4f}" for a in alpha[mask])

    if first:
        off = flag_off_rungs(v, anchors["pred_v"])
        if off.any():
            log(f"  rung(s) at alpha {alphas(off)} more than {GAIN_TOL:.0%} off "
                f"the block's slope to the model: re-reading")
            again = _read_rungs(bench, anchors, np.flatnonzero(off),
                                seq0=seq0 - len(reads), t0=t0, dark=dark,
                                pmt=pmt, log=log)
            for i, r in zip(np.flatnonzero(off).tolist(), again):
                last[i] = r
                v[i] = r["mean_v"] - r["dark_v"]
            reads = reads + again
            off = flag_off_rungs(v, anchors["pred_v"])
            if off.any():
                log(f"  still off after the re-read, out of the ladder for "
                    f"this run: alpha {alphas(off)}")
        keep = ~off
        if not keep.any():
            raise SystemExit(f"no anchor rung within {GAIN_TOL:.0%} of the block's "
                             f"own slope to the model, read twice; the bench is "
                             f"not holding a level")
        if not np.all(v[keep] > 0):
            raise SystemExit(f"anchor rung(s) at alpha {alphas(keep & ~(v > 0))} "
                             f"read at or below the dark; there is nothing to "
                             f"take a ratio against")
        g, rms = fit_gain(v[keep], anchors["pred_v"][keep])
        anchors["v_ref"], anchors["keep"], anchors["g0"] = v, keep, g
        use = keep
    else:
        keep = anchors["keep"]
        off = np.zeros(n, dtype=bool)
        off[keep] = flag_off_rungs(v[keep], anchors["v_ref"][keep])
        use = keep & ~off
        if not use.any():
            log("  every rung is off the block's own slope: nothing to drop "
                "against, the block is fitted whole")
            use = keep
        s, rms = fit_gain(v[use], anchors["v_ref"][use])
        g = anchors["g0"] * s

    judged = {id(r): i for i, r in last.items()}
    for r in reads:
        i = judged.get(id(r))
        # a read that was re-read is not judged by anything and is dropped
        r["kind"] = "gain" if i is not None and use[i] else "gain_drop"
        r["g"], r["g_rms_v"] = f"{g:.9g}", f"{rms:.9g}"
    rows.extend(reads)

    if first:
        log(f"  g0 = {g:.5f} fitted to the model   fit rms {rms * 1e3:.4f} mV "
            f"over {int(use.sum())} of {n} rungs, "
            f"{v[use].min() * 1e3:.3f}..{v[use].max() * 1e3:.3f} mV above dark"
            + (f"   out of the ladder {int(off.sum())}: alpha {alphas(off)}"
               if off.any() else ""))
        for i in range(n):
            pred = anchors["pred_v"][i]
            log(f"    alpha {alpha[i]:.4f}  {v[i] * 1e3:9.4f} mV read vs "
                f"{pred * 1e3:9.4f} predicted   resid {(v[i] - g * pred) * 1e3:+.4f} mV"
                + ("   DROPPED" if off[i] else ""))
        return g, rms, dark
    log(f"  g = {g:.5f} = {s:.5f} x g0 {anchors['g0']:.5f}   fit rms "
        f"{rms * 1e3:.4f} mV over {int(use.sum())} rungs"
        + (f"   dropped {int(off.sum())}: alpha {alphas(off)}" if off.any() else ""))
    for i in np.flatnonzero(keep):
        ref, now = anchors["v_ref"][i], v[i]
        log(f"    alpha {alpha[i]:.4f}  {now * 1e3:9.4f} mV now vs {ref * 1e3:9.4f} "
            f"at start   resid {(now - s * ref) * 1e3:+.4f} mV"
            + ("   DROPPED" if off[i] else ""))
    return g, rms, dark


# ======================================================================
# the run
# ======================================================================

def _hook(t0: float, pmt, live: bool):
    """Row hook: stamp the clock, the PMT, and a provisional Y.

    ``live`` is ``False`` for the anchor reads, which have no predicted Y.  An
    inference row's Y uses the ``g`` measured before the chunk;
    :func:`apply_gain` rewrites every one at the end.
    """
    def hook(row: dict) -> None:
        row["t_s"] = f"{time.monotonic() - t0:.1f}"
        if pmt is not None:
            pmt(row)
        if not live or row.get("g") in ("", None):
            row["y_meas_v"], row["resid_v"] = "", ""
            return
        _set_y(row, float(row["g"]))
    return hook


def _set_y(row: dict, g: float) -> None:
    """Write ``y_meas_v`` / ``resid_v`` for one inference row at gain ``g``.

        y = (mean_v - dark_v) / g - sb_pred_v

    Divide first, subtract second: ``sb`` went through ``g`` too, and step 6
    fits it where ``g`` is 1.
    """
    y = (row["mean_v"] - row["dark_v"]) / g - float(row["sb_pred_v"])
    row["g"] = f"{g:.9g}"
    row["y_meas_v"] = f"{y:.9g}"
    row["resid_v"] = f"{y - float(row['y_pred_v']):.9g}"


def apply_gain(rows: list, log=print) -> int:
    """Re-derive every inference Y with ``g`` interpolated in time.  Returns n.

    How: ``np.interp`` on each row's ``t_s`` between the blocks that bracket it,
    clamping outside them.  Idempotent, and safe on a partial run -- one block
    reduces to a constant, none leaves the rows alone.
    """
    onn = [r for r in rows if r.get("kind") == "onn" and r.get("y_pred_v") not in ("", None)]
    # One mark per MEASUREMENT, not per read: the rungs of one anchor share a
    # fit, so their timestamps average to the middle of the block.  Grouped by
    # position, because ``rows`` is in read order and a block is contiguous --
    # grouping by the value of g instead would silently merge two measurements
    # that happened to land on the same number, which is exactly what a bench
    # that is NOT drifting does.
    tg: list[tuple[float, float]] = []
    run: list[float] = []
    for r in rows + [{}]:
        if r.get("kind") == "gain" and r.get("g") not in ("", None):
            run.append(float(r["t_s"]))
            g_here = float(r["g"])
        elif run:
            tg.append((float(np.mean(run)), g_here))
            run = []
    if not tg or not onn:
        return 0
    t_at = np.array([t for t, _ in tg])
    g_at = np.array([g for _, g in tg])
    for r in onn:
        _set_y(r, float(np.interp(float(r["t_s"]), t_at, g_at)))
    log(f"  g interpolated over {len(tg)} measurement(s), "
        f"{g_at.min():.5f}..{g_at.max():.5f}, onto {len(onn)} inference reads")
    return len(onn)


def build_reads(ref: dict, pairs: list[int], models) -> tuple[np.ndarray, np.ndarray, list]:
    """Every (bank, image, shot, output) read, flattened into one block.

    Read order is image-major and then shot, so a partial run covers whole
    images.
    """
    slots = [ref["channels"].index(k) for k in pairs]
    xs, ws, extra = [], [], []
    seq = 0
    for b in range(ref["n_banks"]):
        for i in range(ref["n"]):
            for t in range(ref["n_tiles"]):
                for o in range(ref["n_out"]):
                    xv = ref["x"][i, t][slots]
                    wv = ref["w"][b, o, t][slots]
                    sb = float(sum(models[k].single_beam(xk, wk)
                                   for k, xk, wk in zip(pairs, xv, wv)))
                    xs.append(xv)
                    ws.append(wv)
                    extra.append(dict(BLANK, kind="onn", seq=seq,
                                      image=int(ref["idx"][i]),
                                      label=int(ref["labels"][i]),
                                      shot=t, out=o, bank=b,
                                      sb_pred_v=f"{sb:.9g}",
                                      y_pred_v=f"{float(ref['y_pred'][b, i, o, t]):.9g}"))
                    seq += 1
    return (np.asarray(xs)[None], np.asarray(ws)[None], extra)


def summarise(rows: list, ref: dict, log=print) -> None:
    """Residual against y_block, and whether the optics picked the right digit.

    The accuracy is computed from the measured ``y_sum`` through the same
    readout gain the simulation used.
    """
    onn = [r for r in rows if r.get("kind") == "onn" and r.get("y_meas_v") != ""]
    if not onn:
        log("\nno completed inference reads to summarise")
        return
    res = np.array([float(r["resid_v"]) for r in onn])
    pred = np.array([float(r["y_pred_v"]) for r in onn])
    span = float(pred.max() - pred.min()) or 1.0
    log(f"\n{len(onn)} of {ref['n'] * ref['n_tiles'] * ref['n_out'] * ref['n_banks']} "
        f"inference reads")
    log(f"  residual vs y_block   rms {res.std() * 1e3:.4f} mV  "
        f"bias {res.mean() * 1e3:+.4f} mV  max |{np.abs(res).max() * 1e3:.4f}| mV"
        f"   ({100 * res.std() / span:.2f}% of the predicted span)")

    # y_sum per (image, output), but only for images read all the way through.
    got: dict[int, np.ndarray] = {}
    counts: dict[int, np.ndarray] = {}
    for r in onn:
        i = int(r["image"])
        sign = 1.0 if int(r["bank"] or 0) == 0 else -1.0
        got.setdefault(i, np.zeros(ref["n_out"]))
        counts.setdefault(i, np.zeros(ref["n_out"]))
        got[i][int(r["out"])] += sign * float(r["y_meas_v"])
        counts[i][int(r["out"])] += 1
    want = ref["n_tiles"] * ref["n_banks"]
    whole = [i for i in got if counts[i].min() == want]
    if not whole:
        log("  no image was read all the way through, so there is no class to compare")
        return
    pos = {int(v): j for j, v in enumerate(ref["idx"])}
    hit = agree = 0
    for i in sorted(whole):
        j = pos[i]
        bench_class = int(np.argmax(got[i] * ref["out_scale"]))
        hit += bench_class == int(ref["labels"][j])
        agree += bench_class == int(np.argmax(ref["logits_ref"][j]))
    log(f"  {len(whole)} complete image(s): bench accuracy {hit}/{len(whole)}, "
        f"agrees with the simulation on {agree}/{len(whole)}")


def print_anchor(anchors: dict, log=print) -> None:
    """The gain ladder: the band, the seed, and what each rung should read.

    Listed rung by rung, not summarised: the first block's precision is a
    common-mode systematic on every ``y_meas_v`` of the run.
    """
    lo, hi = anchors["band"]
    log(f"g anchor  : {anchors['n']} rungs over alpha {lo:.3f}..{hi:.3f} of "
        f"Y_max = {anchors['y_max_v'] * 1e3:.1f} mV   seed {anchors['seed']}   "
        f"g0 fitted to the model at the first block of this run")
    if anchors["capped"]:
        log(f"            top capped at {LADDER_CEILING_V * 1e3:.0f} mV, "
            f"step 8's ladder ceiling")
    for a, v, lab, ok in zip(anchors["alpha"], anchors["pred_v"],
                             anchors["labels"], anchors["exact"]):
        log(f"            alpha {a:.4f}   {v * 1e3:9.4f} mV predicted   {lab}"
            + ("" if ok else "   [solver fell back: deterministic drive]"))
    lo_v, hi_v = anchors["pred_v"].min(), anchors["pred_v"].max()
    if lo_v > 0:
        log(f"            lever arm {hi_v / lo_v:.2f}x "
            f"({lo_v * 1e3:.1f}..{hi_v * 1e3:.1f} mV)"
            + ("   -- too short to be a slope, widen the band"
               if hi_v / lo_v < 1.3 else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True, type=Path,
                    help="bench_reference.npz: the drives to display and the Y "
                         "each should produce")
    ap.add_argument("--calib", required=True, type=Path, metavar="FILE|DIR",
                    help="the run's calib.json (or the directory holding one): "
                         "every step the forward model needs")
    ap.add_argument("--t-meas", type=float, default=ACQ.t_meas_s, metavar="S",
                    help=f"DAQ averaging window, in seconds, for EVERY read of "
                         f"the run -- inference, anchor and the opening dark "
                         f"alike (default {ACQ.t_meas_s:g}, the same window "
                         f"steps 6-8 read at). It is the whole cost of the run "
                         f"and also what decides whether the read noise fits "
                         f"the budget bench_reference.py prints")
    ap.add_argument("--p", "--pairs", dest="pairs", nargs="+", type=int, default=None,
                    metavar="PAIR", help="drive only these pairs (default: all)")
    ap.add_argument("--gain-every", type=float, default=GAIN_EVERY_S, metavar="S",
                    help=f"replay the gain anchors, and read the PMT thermistor, "
                         f"this often (default {GAIN_EVERY_S:g})")
    ap.add_argument("--gain-band", nargs=2, type=float, default=GAIN_BAND,
                    metavar=("LO", "HI"),
                    help=f"the anchor ladder, as fractions of Y_max (default "
                         f"{GAIN_BAND[0]:g} {GAIN_BAND[1]:g}); the top is capped "
                         f"at {LADDER_CEILING_V * 1e3:.0f} mV like step 8's")
    ap.add_argument("--gain-n", type=int, default=GAIN_RUNGS, metavar="N",
                    help=f"rungs in the anchor ladder (default {GAIN_RUNGS})")
    ap.add_argument("--gain-seed", type=int, default=None, metavar="SEED",
                    help="seed for the anchor drives (default: fresh, printed "
                         "and written to the CSV meta)")
    ap.add_argument("--no-gain", action="store_true",
                    help="measure no gain at all and assume g = 1 throughout; "
                         "nothing then tracks a drift")
    ap.add_argument("--out", type=Path, default=None, metavar="DIR")
    ap.add_argument("--pmt", default=None, metavar="RESOURCE",
                    help="VISA address of the Keithley 2400 reading the PMT "
                         "thermistor (default: daq_pmt_monitor.K2400_RESOURCE)")
    ap.add_argument("--no-pmt", action="store_true",
                    help="do not log the PMT thermistor")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and the cost; touch no instrument")
    args = ap.parse_args()

    ref = load_npz(args.npz)
    calib = calib_path(args.calib)
    if not calib.is_file():
        raise SystemExit(f"no calib.json at {calib}")
    want = select_pairs(args.pairs) or list(ref["channels"])
    models, phases, reference = load_forward_model(calib, pairs=want)
    pairs = sorted(models)
    anchors = None
    if not args.no_gain:
        seed = (int(np.random.SeedSequence().entropy % 2**32)
                if args.gain_seed is None else int(args.gain_seed))
        anchors = build_anchors(models, phases, pairs, seed=seed,
                                n=args.gain_n, band=args.gain_band)

    x, w, extra = build_reads(ref, pairs, models)
    n_reads = x.shape[1]
    # Nominal only -- the real cadence is set from measured elapsed time once
    # the first chunk has run.  See the loop.
    per_read = args.t_meas + ACQ.settle_s
    chunk = max(1, int(args.gain_every // per_read))
    n_gain = (-(-n_reads // chunk) + 1) if anchors else 0
    secs = (n_reads + n_gain * (anchors["n"] if anchors else 0)) * per_read

    print(f"npz       : {args.npz}  ({ref['n']} images, {ref['n_tiles']} shots, "
          f"{ref['n_out']} outputs, {ref['n_banks']} bank(s))")
    print(f"calib     : {calib}   pairs {pairs}   ref {reference}")
    print(f"y_sat     : {ref['y_sat']}")
    if anchors is None:
        print("g         : 1.0 assumed (--no-gain) -- nothing here measures a drift")
    else:
        print_anchor(anchors)
    print(f"Read      : {n_reads} inference reads at {args.t_meas:g} s "
          f"+ {ACQ.settle_s:g} s settle")
    if anchors is not None:
        print(f"Gain      : {anchors['n']} rungs every {args.gain_every / 60:g} min, "
              f"~{n_gain} measurements ({n_gain * anchors['n']} reads); "
              f"the first fits g0 to the model, the rest a ratio to it; a rung "
              f"more than {GAIN_TOL:.0%} off its block's slope is dropped")
    print("Dark      : one read, at the start, held for the run")
    print(f"PMT       : {'off (--no-pmt)' if args.no_pmt else 'read with every gain block'}")
    print(f"Estimate  : {secs / 60:.1f} min ({secs / 3600:.2f} h)   "
          f"[nominal; the 0929 run cost ~2 s/read more than this]")
    if args.dry_run:
        return 0

    pmt = open_pmt(args.pmt, period_s=args.gain_every, enabled=not args.no_pmt)

    out = Path(args.out or OUT_DIR)
    stamp = time.strftime("%m%d_%H%M")
    csv_path = out / f"onn_infer_{stamp}.csv"
    columns = BEFORE + AFTER + (list(pmt.COLUMNS) if pmt else [])

    slm = connect_slm(SLM_DISPLAY_NO, USB_SLM_NO)
    daq = connect_daq(device=DAQ_DEVICE, channel=DAQ_CHANNEL,
                      t_both=args.t_meas, t_single=args.t_meas,
                      min_val=-ACQ.range_v, max_val=ACQ.range_v)
    bench = {"slm": slm, "daq": daq, "pairs": pairs, "models": models,
             "phases": phases, "y_sat": ref["y_sat"],
             "layout": load_layout(calib, pairs),
             "acq": replace(ACQ, t_meas_s=args.t_meas)}

    rows: list[dict] = []
    status, t0 = "complete", time.monotonic()
    dark: float | None = None       # measured by the first anchor block, then held

    def flush():
        """The CSV on disk, whole rows only, after every chunk and at the end."""
        if rows:
            if anchors is None:
                gain_meta = {"g_reference": "1 (assumed, --no-gain)", "g0": "",
                             "g_reference_v": "", "gain_alphas": "",
                             "gain_seed": "", "gain_band": "", "gain_tol": "",
                             "gain_rungs_out": "", "y_max_v": ""}
            else:
                gain_meta = {
                    "g_reference": "first gain block of this run, fitted to the model (g0)",
                    "g0": ("" if anchors["g0"] is None else f"{anchors['g0']:.9g}"),
                    "g_reference_v": (
                        "" if anchors["v_ref"] is None else
                        " ".join(f"{v:.6g}" for v in anchors["v_ref"])),
                    "gain_alphas": " ".join(f"{a:.4f}" for a in anchors["alpha"]),
                    "gain_seed": anchors["seed"],
                    "gain_band": f"{anchors['band'][0]:.4f} {anchors['band'][1]:.4f}",
                    "gain_tol": f"{GAIN_TOL:g}",
                    "gain_rungs_out": " ".join(
                        f"{a:.4f}" for a in anchors["alpha"][~anchors["keep"]]),
                    "y_max_v": f"{anchors['y_max_v']:.9g}"}
            write_verify_csv(csv_path, pairs, rows,
                             meta={"npz": ref["path"], "calib": calib.resolve(),
                                   "npz_calib": ref["step7_json"] or "",
                                   **gain_meta,
                                   "gain_every_s": f"{args.gain_every:g}",
                                   "t_meas_s": f"{args.t_meas:g}",
                                   "dark_v": "" if dark is None else f"{dark:.9g}",
                                   "y_sat_v": f"{ref['y_sat']:g}",
                                   "status": status,
                                   "checkpoint": ref["meta"].get("checkpoint", "")},
                             extra=columns)

    def anchor(seq0: int) -> float:
        """One gain measurement; keeps the run's single dark and returns g."""
        nonlocal dark
        g, _, dark = measure_gain(bench, anchors, seq0=seq0, t0=t0, rows=rows,
                                  dark=dark, pmt=pmt)
        return g

    def taken() -> int:
        """Anchor reads so far, re-reads included: the next block's seq offset."""
        return sum(1 for r in rows if r.get("kind") in ("gain", "gain_drop"))

    try:
        g = 1.0
        if anchors is not None:
            g = anchor(-1)
            flush()
        lo = 0
        while lo < n_reads:
            # Size the chunk from what a read has ACTUALLY been costing, not
            # from t_meas + settle: the 0929 run spent ~2 s/read more than
            # nominal on SLM frames and range changes, which would turn a
            # 15-minute cadence into an 18-minute one.  The first chunk uses
            # the nominal estimate; every one after it uses the measured rate.
            hi = min(lo + max(1, int(args.gain_every / max(per_read, 1e-6))), n_reads)
            if lo and anchors is not None:
                g = anchor(-1 - taken())
            print(f"\n--- reads {lo}..{hi - 1} of {n_reads}  (g = {g:.5f}) ---")
            part = [dict(e, g=f"{g:.9g}") for e in extra[lo:hi]]
            t_chunk = time.monotonic()
            run_patterns(daq, slm, bench["layout"], [tuple(pairs)],
                         x[:, lo:hi], w[:, lo:hi], models, phases, rows=rows,
                         extra=[part], acq=bench["acq"], y_sat=ref["y_sat"],
                         dark=dark, row_hook=_hook(t0, pmt, live=True))
            per_read = (time.monotonic() - t_chunk) / (hi - lo)
            lo = hi
            flush()
        if anchors is not None:
            anchor(-1 - taken())
    except BaseException as exc:
        done = sum(1 for r in rows if r.get("kind") == "onn")
        status = f"partial ({type(exc).__name__} after {done} of {n_reads} reads)"
        raise
    finally:
        if anchors is not None:
            apply_gain(rows)
        flush()
        print(f"\n{csv_path}   [{status}]")
        summarise(rows, ref)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
