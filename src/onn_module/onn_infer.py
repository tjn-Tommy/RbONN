"""Run one inference on the bench: display what the npz says, read what it predicts.

This is the other end of :mod:`onn_module.bench_reference`.  That file simulates
the layer and writes down, for every measurement, both the drive to display and
the ``Y`` it should produce; this one displays those drives, reads the detector,
undoes the instrument response, and asks whether the optics agree -- and, at the
end, whether they classified the digit.

    python src/onn_module/onn_infer.py --npz runs/onn784_ctrl/bench_reference.npz
        --step8-csv src/calib_data/<run>/calib_step8_<stamp>.csv

``--t-meas`` is the window for EVERY read of the run -- inference, anchor and
the opening dark alike -- and defaults to the bench window
(:data:`~calibration_module.measure.pair_v2.T_MEAS_S`), the one steps 6, 7 and 8
also read at, so an inference read and the step-8 rung it is scaled against
average the same noise unless you say otherwise.  It is still worth deciding:
the window is the whole cost of the run (``N * n_tiles * n_out`` reads, 7,900
for ten onn784 images) and also what sets whether the read noise fits inside the
budget ``bench_reference.py`` prints.  Those pull in opposite directions.

Which env
---------
``rbonn``, not ``rbonn-ml``: this module imports the SLM and DAQ drivers and no
torch, which is the opposite of the rest of the package.  The npz is the seam --
it is read with numpy alone.

The post-processing
-------------------
The detector sees, for gain ``g`` and the step-6 single-beam response,

    V = g * [ Y + SUM_k (a_x x_k + q_x x_k^2 + a_w w_k + q_w w_k^2) ] + dark

so recovering the ``Y`` the layer models means

    Y = (V - dark) / g  -  SUM_k (...)

in that order: the single-beam term is light on the detector, so it went through
the same gain, and step 6 fits ``a_x``/``a_w`` from reads where ``g`` is 1 by
definition.  Subtracting before dividing would be right only if those
coefficients were in raw ADC volts, and they are not.

The gain
--------
``g`` is measured, not assumed, and relative to the step-8 sweep that DEFINES
``g = 1`` (see ``gain_reference`` in a step-8 result JSON).  Five of that
sweep's own ladder rungs -- :data:`ANCHOR_FRACTIONS` of that ladder's top rung,
so the bright end of it, where the SNR is -- are replayed here, and ``g`` is the
slope of

    (V_now - dark)  =  g * (V_step8 - dark_step8)

through the origin.  Identical SLM frames on both sides, so ``eta``, ``Phi`` and
``Y_sat`` all cancel and ``g`` needs no model at all.  The drives are taken
VERBATIM from the step-8 CSV rather than recomputed, so a later change to the
level solver cannot silently make the two runs different measurements.

Five DIFFERENT rungs rather than five repeats of one: the same read budget buys
a slope instead of a mean, and a slope can be wrong in a way a mean cannot show.
A pure gain change scales every level by the same factor, so a bad fit means
something other than gain moved -- the residuals are printed per rung for that
reason.

The fit is through the origin, not slope-plus-intercept.  All five anchors sit in
the top half of the ladder, so extrapolating a line back to zero inflates the
intercept's noise 2.2x over a single read -- millivolts of noise on a quantity
whose true value is -0.08 mV.  An intercept there costs more than it measures.

``--gain-every`` re-measures during a long run, and every inference read is
corrected with ``g`` INTERPOLATED IN TIME between the two measurements that
bracket it (:func:`apply_gain`), not with the one that preceded it.  The drift is
continuous; a staircase would leave half of it in the residual.  The ``g`` column
holds the interpolated value actually used, and the ``kind=gain`` rows hold the
measurements it was built from.

The dark
--------
Read ONCE, at the start, and held for the run -- the way a single-sweep step
gets one for free.  Over the 14 h 0929 run the dark moved 0.034 mV with a
0.026 mV warm-up transient, against a 4.7 mV residual, and it is a common offset
across the ten outputs so it cancels exactly in a per-image argmax.  Tracking it
at the default :data:`~calibration_module.measure.patterns.DARK_EVERY_S` cost
that run 182 reads and half an hour.  Step 6 must keep re-reading it -- there the
all-off level is a fitted point of the background block, not an offset -- and
does; this is not step 6.

Without a step-8 CSV
--------------------
``--step8-csv`` may be omitted, and then ``g = 1`` -- which is not "the gain is
one" but "the gain is what it was when the step-7 fit behind this npz was taken",
since ``y_block`` is in that fit's volts.  Nothing in the run measures it, so a
gain error since then shows up as a multiplicative tilt on the residuals and
cannot be told apart from a model error.  Measured on the 0929 run, which had no
anchor: the best-fit scale wandered 0.939..1.064 over 14 h, and removing it per
2 h bin took the residual from 4.71 to ~3.9 mV.  A joint (g, Y_sat) fit on the
inference reads themselves is degenerate -- Y_sat wandered 1.0..5.0 V per bin
with no gain in rms -- so the run cannot anchor itself and the replay is the only
way.  For a quick check on a bench that has not moved since its calibration
omitting it is a fair trade; for anything quantitative it is not, and the CSV
meta records ``g_reference_v = 1 (assumed, ...)`` so a later reader cannot
mistake one run for the other.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from calibration_module.fit.verify import (  # noqa: E402
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
from calibration_module.measure.pmt_temp import (  # noqa: E402
    PMT_PERIOD_S,
    PmtTemperature,
)

# ---- Edit these to match your setup ----
SLM_DISPLAY_NO = None       # None -> auto-detect the LCOS-SLM display
USB_SLM_NO = 1              # SLM_Ctrl_* device index for the DVI-mode switch

DAQ_DEVICE = "Dev1"
DAQ_CHANNEL = "ai0"

ACQ = DEFAULT_ACQ           # the bench acquisition; --t-both may override the window

GAIN_EVERY_S = 900.0        # re-measure g at most this often (15 min)

#: The step-8 ladder rungs replayed to measure ``g``, as fractions of THAT
#: LADDER'S TOP RUNG -- not of Y_max.
#:
#: All in the bright end of the ladder on purpose: a gain is a scale factor, so
#: it can be measured at any level, and the bright end is where the SNR is.  Five
#: distinct rungs rather than five repeats of one buys a slope for the same read
#: budget; see "The gain" above for why the fit runs through the origin.
#:
#: Relative to the ladder rather than to Y_max because step 8 no longer drives
#: full scale.  Its ceiling is :data:`~calibration_module.steps.calib_step8.LADDER_CEILING_V`
#: in VOLTS, so the top alpha depends on the PMT gain -- presently ~0.29.  Absolute
#: targets of 0.55..0.95 would all land above it, the nearest-rung picker would
#: hand back the five brightest rungs whatever they were, and a slope fitted over
#: rungs 10 % apart is a mean with extra steps.  As fractions, the 1.8x lever arm
#: survives any gain.
ANCHOR_FRACTIONS = (0.55, 0.66, 0.77, 0.88, 1.00)

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
    are reshaped back to a leading bank dimension here and the rest of the file
    never has to care whether the model was differential.
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

def read_step8_anchors(path: Path, pairs: list[int],
                       fractions=ANCHOR_FRACTIONS) -> dict:
    """Step-8 ladder rungs to replay: their drives, and their g = 1 volts.

    ``fractions`` are of the ladder's OWN top rung, which this reads off the CSV
    -- step 8 caps itself in volts, so its top alpha is whatever the PMT gain
    made it.  See :data:`ANCHOR_FRACTIONS`.

    Each rung is ONE row of the CSV -- one drive, read once -- so this returns
    the drive exactly as it was displayed and the dark-corrected voltage it
    produced.  Replaying the drive and dividing gives ``g`` with no model in it.

    Rungs are picked nearest-first and never twice, so a sparse ladder degrades
    to fewer distinct anchors rather than to the same one counted five times --
    which would quietly turn the slope back into a mean.

    The ``F`` rows (the rung step 8 brackets its run with) are deliberately
    eligible: they are ordinary reads at the top of the ladder, and the
    brightest anchors available.
    """
    text = [ln for ln in Path(path).read_text(encoding="utf-8").splitlines()
            if not ln.startswith("#")]
    rows = list(csv.DictReader(text))
    if not rows:
        raise SystemExit(f"{Path(path).name} has no data rows")
    if "alpha" not in rows[0]:
        raise SystemExit(f"{Path(path).name} has no 'alpha' column -- point "
                         f"--step8-csv at a step-8 / system-accuracy CSV")
    missing = [k for k in pairs if f"x_{k}" not in rows[0]]
    if missing:
        raise SystemExit(f"{Path(path).name} has no x_/w_ columns for pair(s) "
                         f"{missing}: it did not drive the pairs this run does")
    usable = [r for r in rows if r.get("alpha") and r.get(f"x_{pairs[0]}")]
    if not usable:
        raise SystemExit(f"{Path(path).name} has no rows with both an alpha and "
                         f"a drive to replay")

    have = np.array([float(r["alpha"]) for r in usable])
    # The ladder's own ceiling, so the anchors span its bright end whatever alpha
    # that turned out to be.  max() rather than the config, because a run stopped
    # early has a CSV that stops early too and the rows are the only truth here.
    top = float(have.max())
    alphas = [float(f) * top for f in fractions]
    picked: list[int] = []
    for target in alphas:
        order = np.argsort(np.abs(have - float(target)))
        for j in order:
            if int(j) not in picked:
                picked.append(int(j))
                break
    picked.sort(key=lambda j: have[j])
    got = [usable[j] for j in picked]

    x = np.array([[float(r[f"x_{k}"]) for k in pairs] for r in got])
    w = np.array([[float(r[f"w_{k}"]) for k in pairs] for r in got])
    v = np.array([float(r["voltage_mean_v"]) - float(r["dark_v"]) for r in got])
    if not np.all(v > 0):
        raise SystemExit(f"{Path(path).name}: anchor rung(s) at alpha "
                         f"{have[picked][v <= 0]} read at or below their dark; "
                         f"there is nothing to take a ratio against")
    return {
        "x": x, "w": w, "v_ref": v,
        "alpha": have[picked],
        "want": np.asarray(alphas, dtype=float),
        "top": top,
        "n": len(got),
        "csv": str(path),
    }


def fit_gain(v_now: np.ndarray, v_ref: np.ndarray) -> tuple[float, float]:
    """``(g, rms)`` -- least squares slope through the origin, and its residual.

    Through the origin because the dark is already off both sides and because
    every anchor is bright (see "The gain"): a free intercept here is fitted
    almost entirely to noise.  ``rms`` is the residual in volts, and it is the
    number that says whether ``g`` explains what happened -- a pure gain change
    scales all five rungs by one factor, so anything else shows up here.
    """
    v_now = np.asarray(v_now, dtype=float)
    v_ref = np.asarray(v_ref, dtype=float)
    g = float((v_ref * v_now).sum() / (v_ref ** 2).sum())
    rms = float(np.sqrt(np.mean((v_now - g * v_ref) ** 2)))
    return g, rms


def measure_gain(bench, anchors: dict, *, seq0: int, t0: float, rows: list,
                 dark: float | None, pmt=None, log=print) -> tuple[float, float, float]:
    """Replay every anchor once; return ``(g, rms, dark)`` against step 8.

    The reads are appended to ``rows`` as ``kind=gain``, in read order, so the
    CSV carries its own evidence for every number in the ``g`` column.

    ``dark`` is passed straight through to :func:`run_patterns`: ``None`` on the
    first call, which measures one, and that value on every call after it.  The
    dark it used comes back so the caller can hold it.
    """
    pairs = bench["pairs"]
    n = anchors["n"]
    extra = [[dict(BLANK, kind="gain", seq=seq0 - i,
                   alpha=f"{anchors['alpha'][i]:.6f}") for i in range(n)]]
    got: list[dict] = []
    run_patterns(bench["daq"], bench["slm"], bench["layout"], [tuple(pairs)],
                 anchors["x"][None], anchors["w"][None],
                 bench["models"], bench["phases"], rows=got, extra=extra,
                 acq=bench["acq"], y_sat=bench["y_sat"], dark=dark, log=log,
                 row_hook=_hook(t0, pmt, live=False))
    v = np.array([r["mean_v"] - r["dark_v"] for r in got])
    g, rms = fit_gain(v, anchors["v_ref"])
    for r in got:
        r["g"], r["g_rms_v"] = f"{g:.9g}", f"{rms:.9g}"
    rows.extend(got)
    log(f"  g = {g:.5f}   fit rms {rms * 1e3:.4f} mV over {n} rungs")
    for a, ref, now in zip(anchors["alpha"], anchors["v_ref"], v):
        log(f"    alpha {a:.4f}  {now * 1e3:9.4f} mV now vs {ref * 1e3:9.4f} "
            f"at step 8   resid {(now - g * ref) * 1e3:+.4f} mV")
    return g, rms, float(got[0]["dark_v"])


# ======================================================================
# the run
# ======================================================================

def _hook(t0: float, pmt, live: bool):
    """Row hook: stamp the clock, the PMT, and a provisional Y.

    ``live`` is ``False`` for the anchor reads, which have no predicted Y to be
    compared against -- they are the measurement the comparison depends on, not
    part of it.

    For inference reads the Y written here uses the ``g`` measured BEFORE the
    chunk, so an interrupted run still lands with usable numbers.  It is
    provisional: :func:`apply_gain` rewrites every one of them at the end with
    ``g`` interpolated between the anchors that bracket the read.
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

    The order is fixed by where the gain sits in the chain: the single-beam
    background is light on the detector, so it went through ``g`` too, while
    step 6 fits ``a_x``/``a_w`` from reads where ``g`` is 1 by definition.  So
    divide first, subtract second.
    """
    y = (row["mean_v"] - row["dark_v"]) / g - float(row["sb_pred_v"])
    row["g"] = f"{g:.9g}"
    row["y_meas_v"] = f"{y:.9g}"
    row["resid_v"] = f"{y - float(row['y_pred_v']):.9g}"


def apply_gain(rows: list, log=print) -> int:
    """Re-derive every inference Y with ``g`` interpolated in time.  Returns n.

    The anchors are measurements of a continuous quantity at discrete times, so
    the read halfway between two of them is best served by the value halfway
    between: holding the earlier one leaves a sawtooth in the residual whose
    amplitude is the full drift across the interval.  Outside the bracketing
    anchors -- there are none before the first or after the last -- ``np.interp``
    clamps, which is the right answer: the nearest measurement is all there is.

    Idempotent, and safe on a partial run: with a single anchor it reduces to
    that one constant, and with none it leaves the rows alone.
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
    images: an interrupted 10-image run leaves 6 complete digits rather than 10
    unusable fragments.
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

    The accuracy is the point of the run.  It is computed from the measured
    y_sum through the SAME readout gain the simulation used, so a disagreement
    is optical rather than a difference in how the logits were formed.
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
    """The g = 1 reference: which CSV, which rungs, and what they read.

    The anchors' own precision propagates straight into every ``y_meas_v``, and
    it is a COMMON-MODE systematic on the whole run -- a slope fitted against
    five single step-8 reads cannot be averaged down by reading more here.  That
    is why the rungs are listed rather than summarised: a step-8 sweep that read
    each of them once is the floor, and reading them 3x at each end of step 8 is
    what moves it.
    """
    log(f"g anchor  : {Path(anchors['csv']).name}   {anchors['n']} rungs, "
        f"g = 1 by definition   (ladder top alpha {anchors['top']:.4f})")
    for want, got, v in zip(anchors["want"], anchors["alpha"], anchors["v_ref"]):
        near = "" if abs(want - got) < 5e-3 else f"  (wanted {want:.4f})"
        log(f"            alpha {got:.4f}   {v * 1e3:9.4f} mV{near}")
    lo, hi = anchors["v_ref"].min(), anchors["v_ref"].max()
    if lo > 0:
        log(f"            lever arm {hi / lo:.2f}x "
            f"({lo * 1e3:.1f}..{hi * 1e3:.1f} mV)"
            + ("   -- too short to be a slope, check the ladder"
               if hi / lo < 1.3 else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True, type=Path,
                    help="bench_reference.npz: the drives to display and the Y "
                         "each should produce")
    ap.add_argument("--t-meas", type=float, default=ACQ.t_meas_s, metavar="S",
                    help=f"DAQ averaging window, in seconds, for EVERY read of "
                         f"the run -- inference, anchor and the opening dark "
                         f"alike (default {ACQ.t_meas_s:g}, the same window "
                         f"steps 6-8 read at). It is the whole cost of the run "
                         f"and also what decides whether the read noise fits "
                         f"the budget bench_reference.py prints")
    ap.add_argument("--step8-csv", type=Path, default=None, metavar="CSV",
                    help="the step-8 CSV that DEFINES g = 1; the ladder rungs "
                         "nearest --gain-alphas give both the drives to replay "
                         "and the reference voltages. "
                         "Omit it to assume g = 1 -- i.e. to assert the detector "
                         "gain is unchanged since the step-7 fit the npz was built "
                         "against.  Nothing then measures a drift, and any gain "
                         "error lands in the residual as a multiplicative tilt")
    ap.add_argument("--step7", type=Path, default=None,
                    help="calibration JSON (default: the npz's step7_json)")
    ap.add_argument("--p", "--pairs", dest="pairs", nargs="+", type=int, default=None,
                    metavar="PAIR", help="drive only these pairs (default: all)")
    ap.add_argument("--gain-every", type=float, default=GAIN_EVERY_S, metavar="S",
                    help=f"re-measure g at most this often (default {GAIN_EVERY_S:g})")
    ap.add_argument("--gain-alphas", nargs="+", type=float,
                    default=ANCHOR_FRACTIONS, metavar="F",
                    help=f"step-8 ladder rungs to replay as gain anchors, as "
                         f"fractions of that ladder's TOP rung (default "
                         f"{' '.join(f'{a:g}' for a in ANCHOR_FRACTIONS)})")
    ap.add_argument("--out", type=Path, default=None, metavar="DIR")
    ap.add_argument("--pmt", nargs="?", const="", default=None, metavar="RESOURCE",
                    help="log the PMT thermistor with each read")
    ap.add_argument("--pmt-period", type=float, default=PMT_PERIOD_S, metavar="S")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and the cost; touch no instrument")
    args = ap.parse_args()

    ref = load_npz(args.npz)
    step7 = Path(args.step7 or ref["step7_json"] or "")
    if not step7.is_file():
        raise SystemExit(f"step-7 calibration not found at {step7} -- pass --step7 "
                         f"(the npz records {ref['step7_json']!r})")
    want = select_pairs(args.pairs) or list(ref["channels"])
    models, phases, reference = load_forward_model(step7, pairs=want)
    pairs = sorted(models)
    anchors = (read_step8_anchors(args.step8_csv, pairs, args.gain_alphas)
               if args.step8_csv else None)

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
    print(f"step7     : {step7.name}   pairs {pairs}   ref {reference}")
    print(f"y_sat     : {ref['y_sat']}")
    if anchors is None:
        print("g         : 1.0 assumed (no --step8-csv) -- the detector gain is "
              "taken to be unchanged since the step-7 fit; nothing here measures it")
    else:
        print_anchor(anchors)
    print(f"Read      : {n_reads} inference reads at {args.t_meas:g} s "
          f"+ {ACQ.settle_s:g} s settle")
    if anchors is not None:
        print(f"Gain      : {anchors['n']} rungs every {args.gain_every / 60:g} min, "
              f"~{n_gain} measurements ({n_gain * anchors['n']} reads)")
    print("Dark      : one read, at the start, held for the run")
    print(f"Estimate  : {secs / 60:.1f} min ({secs / 3600:.2f} h)   "
          f"[nominal; the 0929 run cost ~2 s/read more than this]")
    if args.dry_run:
        return 0

    pmt = None
    if args.pmt is not None:
        try:
            pmt = PmtTemperature(args.pmt or None, period_s=args.pmt_period)
            pmt.read()
        except Exception as exc:                       # noqa: BLE001 -- never fatal
            print(f"--pmt ignored ({type(exc).__name__}: {exc})")

    out = Path(args.out or OUT_DIR)
    stamp = time.strftime("%m%d_%H%M")
    csv_path = out / f"onn_infer_{stamp}.csv"
    columns = BEFORE + AFTER + (list(PmtTemperature.COLUMNS) if pmt else [])

    slm = connect_slm(SLM_DISPLAY_NO, USB_SLM_NO)
    daq = connect_daq(device=DAQ_DEVICE, channel=DAQ_CHANNEL,
                      t_both=args.t_meas, t_single=args.t_meas,
                      min_val=-ACQ.range_v, max_val=ACQ.range_v)
    bench = {"slm": slm, "daq": daq, "pairs": pairs, "models": models,
             "phases": phases, "y_sat": ref["y_sat"],
             "layout": load_layout(step7, pairs),
             "acq": replace(ACQ, t_meas_s=args.t_meas)}

    rows: list[dict] = []
    status, t0 = "complete", time.monotonic()
    dark: float | None = None       # measured by the first anchor block, then held

    def flush():
        """The CSV on disk, whole rows only, after every chunk and at the end."""
        if rows:
            write_verify_csv(csv_path, pairs, rows,
                             meta={"npz": ref["path"], "step7_json": step7.resolve(),
                                   "step8_csv": anchors["csv"] if anchors else "",
                                   "g_reference_v": (
                                       " ".join(f"{v:.6g}" for v in anchors["v_ref"])
                                       if anchors else "1 (assumed, no step-8 anchor)"),
                                   "gain_alphas": (
                                       " ".join(f"{a:.4f}" for a in anchors["alpha"])
                                       if anchors else ""),
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

    try:
        g, taken = 1.0, 0
        if anchors is not None:
            g = anchor(-1)
            taken = anchors["n"]
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
                g = anchor(-1 - taken)
                taken += anchors["n"]
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
            anchor(-1 - taken)
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
