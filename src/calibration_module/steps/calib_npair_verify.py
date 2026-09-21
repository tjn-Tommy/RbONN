"""Step 8 -- verify the calibration on n pairs at once: random patterns, measured vs predicted.

    python src/calibration_module/steps/calib_npair_verify.py --step7 RESULT.json --n 2           # COLLECT
    python src/calibration_module/steps/calib_npair_verify.py --step7 RESULT.json some.csv        # COMPARE offline
    python src/calibration_module/steps/calib_npair_verify.py --step7 RESULT.json some.csv --n 1  # one n only
    python src/calibration_module/steps/calib_npair_verify.py --step7 RESULT.json --n 2 --out DIR # CSV + PNG into DIR
    python src/calibration_module/steps/calib_npair_verify.py --step7 RESULT.json --n 2 --p 2 3 4  # only these pairs

Every n-subset of the N pairs the step-7 result calibrates is a block, C(N, n)
of them -- or of the pairs --p names, when a run should skip some -- and each
block is driven with PATTERNS random patterns: every driven
pair gets its own x_k and w_k, independent and uniform in [DRIVE_MIN, DRIVE_MAX],
and every other pair is off.  Nothing is fitted.  Step 6's eta and single-beam
response and step 7's Phi_k predict each pattern, and each block's error is::

    RMSE  = rms(meas - pred)              (mV)
    NRMSE = rms(meas - pred) / Y_max      (% of full scale)

Y_max is the block's reachable ceiling -- the largest |E|^2 its pairs can be
driven to, which is what a computation has to divide into levels.  It is not
(sum_k eta_k)^2: a pair only reaches |z_k| = eta_k with its phase pinned, so
that free-phase bound over-predicts by 54..67 % on these runs.  NRMSE is the
number to read across blocks, since RMSE in mV cannot be compared between
blocks whose full scales differ by 3x.

Output: calib_verify_<n>pair_<MMDD_HHMM>.csv, one row per read with its drive,
dark, mean, std, input range and prediction -- written even when a run is
stopped part-way -- and <csv stem>_compare_<method>.png, step 8's report:
measured vs predicted, the pulls, the RMSE per block, and the NRMSE per
block against that block's full scale.

n = 1 checks step 6 alone: a single pair has no relative phase, so Phi drops out.
Step 6 measured its cross line at x = 1 plus one interior point, so random (x, w)
tests its model almost entirely away from where it was fitted.  n = 2 is the
first check of step 7.

Independent x_k, w_k move each pair's phase separately.

Input range.  Each read uses the smallest range the board offers that keeps the
predicted signal under RANGE_FILL of it, so a dim pattern keeps the sensitive
range and a bright one does not clip.  Step 6's near-rail escalation, reused,
remeasures any read the model under-predicted.

Sign.  There is no --flip, and there must not be: the TIA's negative-for-light
output is inverted once, at the read (ACQ.invert).  A COLLECT aborts if its
first bright pattern reads below the dark; a COMPARE refuses a CSV whose signal
is negative.

Old step-8 CSVs (calib_step8_simple_*.csv, x = w = v) load unchanged and are
compared per n.

Not a unittest (it needs real hardware) -- run it directly.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from calibration_module.fit.verify import (  # noqa: E402
    BlockError,
    blocks_for,
    drive_bounds,
    evaluate,
    load_forward_model,
    load_verify_csv,
    pairs_tag,
    predict,
    random_patterns,
    write_verify_csv,
)
from calibration_module.fit.sigma import floor_std  # noqa: E402
from calibration_module.measure.bench import connect_daq, connect_slm  # noqa: E402
from calibration_module.measure.pair_v2 import (  # noqa: E402
    PairV2Acq,
    can_autorange,
    read_point_autorange,
    set_range,
)
from calibration_module.fit.slm_response import calibration_result_from_dict  # noqa: E402
from slm_module.encoding import channel_layout_from_calibration  # noqa: E402

# ---- Edit these to match your setup ----
CALIB_PATH = REPO_ROOT / "src/calib_data"   # data directory
OUT_DIR = CALIB_PATH                        # CSVs and PNGs land here

# The step-7 result is NOT here -- it is the required --step7 argument, as
# step 6's --step3 and step 7's --step6 are.

PAIRS = None                # None -> every pair the step-7 result calibrates; --p overrides
PAIR_INDEX_BASE = 1         # pairs are numbered 1..N; keep in step with steps 6 and 7
PHASE_METHOD = None         # stored step-7 fit; None -> the single one a v2 JSON has

# ---- The drive ----
PATTERNS = 8                        # random patterns per block
DRIVE_MIN, DRIVE_MAX = 0.1, 1     # x_k, w_k uniform in this box
SEED = None                         # None -> a fresh seed, printed and written into the CSV

# ---- Acquisition ----
ACQ = PairV2Acq(
    t_single_s=10.0,         # the all-off dark
    t_both_s=5.0,           # every pattern
    settle_s=0.25,          # wait after each SLM pattern change, before reading
    range_v=0.1,            # the dark's range; each pattern's comes from range_for()
    range_wide_v=0.2,
    autorange=True,
    invert=True,
)
BOARD_RANGES_V = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)   # the board rounds a request UP to these
RANGE_FILL = 0.7            # a read's range is the smallest with |pred| <= RANGE_FILL * range
DARK_EVERY_S = 300.0        # re-read the all-off dark this often; every row keeps the latest
SIGN_CHECK_V = 1e-3         # the sign check waits for a pattern predicted at least this bright

SLM_DISPLAY_NO = None       # None -> auto-detect the LCOS-SLM display
USB_SLM_NO = 1              # SLM_Ctrl_* device index for the DVI-mode switch

DAQ_DEVICE = "Dev1"
DAQ_CHANNEL = "ai0"


# ======================================================================
# inputs
# ======================================================================

def select_pairs(pairs) -> list[int] | None:
    """Normalize a ``--p`` / ``PAIRS`` selection; ``None`` means every pair.

    The selection is the POOL the blocks are drawn from, not a block: with
    ``--n 2 --p 2 3 4`` the run drives C(3, 2) = 3 blocks, (2,3), (2,4), (3,4).
    Duplicates collapse and the order does not matter -- blocks come out in
    lexical order either way.
    """
    if pairs is None:
        return None
    out = sorted({int(k) for k in pairs})
    if not out:
        raise ValueError("pair selection is empty; drop it to use every pair")
    return out


def _slot(pair: int) -> int:
    """Pair label -> its 0-based slot in the Step-3 layout / SLM drive arrays."""
    return int(pair) - PAIR_INDEX_BASE


def load_layout(step7: Path, pairs):
    """Channel layout from the Step-3 calibration embedded in the step-7 JSON.

    The embedded payload and the ``encoding`` marker were carried forward from
    step 6, so this drives the same aperture and the same level mapping the etas
    and phases were measured under.
    """
    payload = json.loads(Path(step7).read_text(encoding="utf-8"))
    step3 = payload.get("step3")
    if step3 is None:
        raise ValueError(f"{Path(step7).name} has no embedded 'step3' calibration; "
                         f"point --step7 at a combined step-7 result")
    enc_method = (payload.get("encoding") or {}).get("method", "interp")
    layout = channel_layout_from_calibration(
        calibration_result_from_dict(step3), method=enc_method)
    bad = [k for k in pairs if not 0 <= _slot(k) < layout.n_channels]
    if bad:
        raise ValueError(f"pair(s) {bad} out of range: the layout has "
                         f"{layout.n_channels} pairs, numbered from {PAIR_INDEX_BASE}")
    return layout


# ======================================================================
# collect
# ======================================================================

def range_for(pred_v: float) -> tuple[float, float]:
    """``(range, wide)`` for a read predicted at ``pred_v``.

    ``range`` is the smallest board range with ``|pred_v| <= RANGE_FILL * range``;
    ``wide`` is the next one up, where the near-rail escalation remeasures.
    """
    need = abs(float(pred_v)) / RANGE_FILL
    last = len(BOARD_RANGES_V) - 1
    i = next((i for i, r in enumerate(BOARD_RANGES_V) if need <= r), last)
    return BOARD_RANGES_V[i], BOARD_RANGES_V[min(i + 1, last)]


def run_patterns(daq, slm, layout, blocks, x, w, models, phases, *, rows: list,
                 acq: PairV2Acq = ACQ, encode=None, log=print,
                 extra=None, y_sat: float | None = None, t_both_of=None,
                 row_hook=None) -> None:
    """Drive every block's patterns; append one row per read to ``rows``.

    ``x`` and ``w`` are ``(len(blocks), patterns, n)``.  ``rows`` is the caller's
    list, filled as the run goes, so an interrupted run keeps what it read.
    ``encode`` defaults to ``encode_to_pattern``.

    ``extra``, if given, is ``(len(blocks), patterns)`` of mappings merged into
    each row, so a caller can label its reads without this function knowing what
    the labels mean.  It defaults to ``None``, which is exactly the behaviour
    every existing caller gets.

    ``row_hook(row)``, if given, is called with each finished row just before it
    is appended, so a caller can add columns it can only measure *during* the
    run -- unlike ``extra``, which is fixed before the first read.  It runs
    between the DAQ read and the next SLM frame, so a slow instrument there
    stretches the run but never overlaps an acquisition.  It must not raise.

    ``y_sat`` is passed to :func:`predict`, so a pattern's range, logged diff and
    ``pred_v`` all use the saturated model.  ``t_both_of(pred_v)``, if given,
    sets each pattern read's window in seconds; the dark keeps ``acq.t_single_s``.
    Both default to ``None``: linear model, fixed ``acq.t_both_s``.
    """
    if encode is None:
        from slm_module.encoding import encode_to_pattern as encode
    zeros = np.zeros(layout.n_channels)
    slm_width, slm_height = slm.get_slm_info()
    total = len(blocks) * x.shape[1]

    def display(driven, xs, ws) -> None:
        x_vals, w_vals = zeros.copy(), zeros.copy()
        for k, xk, wk in zip(driven, xs, ws):
            x_vals[_slot(k)], w_vals[_slot(k)] = xk, wk
        slm.display_array(encode(x_vals, w_vals, layout, slm_width, slm_height))
        if acq.settle_s:
            time.sleep(acq.settle_s)

    def read(single: bool, pred: float) -> tuple[float, float, float]:
        rng, wide = range_for(pred)
        if can_autorange(daq):
            set_range(daq, rng)
            if t_both_of is not None and not single:
                daq.configure_monitor(replace(daq._settings,  # noqa: SLF001
                                              duration=float(t_both_of(pred))))
        return read_point_autorange(daq, single=single,
                                    acq=replace(acq, range_v=rng, range_wide_v=wide),
                                    log=log)

    def read_dark() -> tuple[float, float]:
        display((), (), ())
        dark, _, _ = read(True, 0.0)
        log(f"dark (all off, {acq.t_single_s:g}s) = {dark * 1e3:.4f} mV")
        return dark, time.monotonic()

    dark, dark_t = read_dark()
    checked = False
    i = 0
    for b, driven in enumerate(blocks):
        log(f"\n--- block {b + 1}/{len(blocks)}: pairs {pairs_tag(driven)} ---")
        for j in range(x.shape[1]):
            if time.monotonic() - dark_t > DARK_EVERY_S:
                dark, dark_t = read_dark()
            xs, ws = x[b, j], w[b, j]
            pred = float(predict(models, phases, driven, xs, ws, y_sat=y_sat))
            display(driven, xs, ws)
            mean, std, used = read(False, pred)
            i += 1
            if not checked and pred >= SIGN_CHECK_V:
                checked = True
                if mean - dark < 0.0:
                    # The read already un-inverts the TIA, so a bright pattern
                    # has to come in ABOVE the dark.  Stop on the first one
                    # rather than collect a run nothing can be compared against.
                    raise RuntimeError(
                        f"pattern predicted at {pred * 1e3:.3f} mV read BELOW the "
                        f"dark ({(mean - dark) * 1e3:+.4f} mV): the light signal is "
                        f"arriving negative.  The read already inverts the TIA "
                        f"(ACQ.invert) -- check the amplifier polarity and that the "
                        f"beam is on; do not negate it a second time.")
            row = {"block": b, "driven": driven, "x": xs, "w": ws,
                   "dark_v": dark, "mean_v": mean, "std_v": std,
                   "range_v": used, "pred_v": pred}
            if extra is not None:
                row.update(extra[b][j])
            if row_hook is not None:
                row_hook(row)
            rows.append(row)
            log(f"[{i}/{total}] {pairs_tag(driven)} #{j + 1}: "
                f"{(mean - dark) * 1e3:9.4f} mV  pred {pred * 1e3:9.4f}  "
                f"diff {(mean - dark - pred) * 1e3:+.4f}  [+/-{used:g} V]")


def collect(step7: Path, n: int, *, pairs=None, method=None, out_dir=None) -> str:
    """Drive every n-pair block, write the CSV (partial if interrupted), compare.

    ``pairs`` (``--p``) restricts the pool the blocks are drawn from -- useful
    when one channel is dead, or when a full C(N, n) would run too long; ``None``
    falls back to ``PAIRS``, and ``PAIRS = None`` to every calibrated pair.  A
    pair named here that the step-7 file does not calibrate end to end is an
    error, not a silent drop.

    The CSV and the PNG go to ``out_dir``, created if missing (default OUT_DIR).
    """
    out = Path(OUT_DIR if out_dir is None else out_dir)
    out.mkdir(parents=True, exist_ok=True)
    wanted = select_pairs(PAIRS if pairs is None else pairs)
    models, phases, ref = load_forward_model(step7, pairs=wanted, method=method)
    pairs = sorted(models)
    layout = load_layout(step7, pairs)
    blocks = blocks_for(pairs, n)
    seed = SEED if SEED is not None else int(np.random.SeedSequence().entropy % 2**32)
    x, w = random_patterns(len(blocks), n, PATTERNS, DRIVE_MIN, DRIVE_MAX, seed=seed)
    reads = len(blocks) * PATTERNS
    secs = reads * (ACQ.t_both_s + ACQ.settle_s)
    secs += ACQ.t_single_s * (1 + secs // DARK_EVERY_S)
    print(f"Step 7 in : {step7}")
    print(f"Pairs     : {pairs}  (N = {len(pairs)}, reference {ref})"
          + ("" if wanted is None else "  [selected with --p]"))
    print(f"Blocks    : C({len(pairs)}, {n}) = {len(blocks)} x {PATTERNS} patterns "
          f"= {reads} reads, ~{secs / 60:.0f} min")
    print(f"Drive     : x_k, w_k uniform in [{DRIVE_MIN:g}, {DRIVE_MAX:g}], seed {seed}")

    slm = connect_slm(SLM_DISPLAY_NO, USB_SLM_NO)
    daq = connect_daq(device=DAQ_DEVICE, channel=DAQ_CHANNEL,
                      t_both=ACQ.t_both_s, t_single=ACQ.t_single_s,
                      min_val=-ACQ.range_v, max_val=ACQ.range_v)
    rows: list[dict] = []
    status = "complete"
    csv_path = None
    try:
        run_patterns(daq, slm, layout, blocks, x, w, models, phases, rows=rows)
    except BaseException as exc:
        status = f"partial ({type(exc).__name__} after {len(rows)} of {reads} reads)"
        raise
    finally:
        try:
            if rows:    # on disk before anything else can fail
                csv_path = write_verify_csv(
                    out / f"calib_verify_{n}pair_{time.strftime('%m%d_%H%M')}.csv",
                    pairs, rows,
                    meta={"step7_json": Path(step7).resolve(), "n_pairs": n,
                          "drive": f"min={DRIVE_MIN},max={DRIVE_MAX},"
                                   f"patterns={PATTERNS},seed={seed}",
                          "status": status})
                print(f"\nCSV ({len(rows)} reads, {status}) written to {csv_path}")
        finally:
            slm.close_slm()
            daq.disconnect()
    print(f"Re-compare with:  python {Path(__file__).name} --step7 {step7} {csv_path}")
    compare_csv(csv_path, step7, method=method, out_dir=out)
    return csv_path


# ======================================================================
# compare  (no fit -- measured vs predicted, block by block)
# ======================================================================

def report(errors: list[BlockError], *, n: int, lo: float, hi: float) -> None:
    """Print each block's error in mV and against its full scale, then the totals.

    ``FS`` is the block's reachable ``Y_max``, so ``NRMSE`` is the error as a
    fraction of the span an n-pair computation actually has to divide into
    levels.  The free-phase ``S^2`` is quoted only as ``FS/S^2``, because it is
    unreachable whenever the ``Phi_k`` disagree.
    """
    tag_w = max(12, max(len(pairs_tag(e.driven)) for e in errors))
    print(f"\n=== {n}-pair blocks: {len(errors)} block(s), drive [{lo:g}, {hi:g}] ===")
    print(f"  {'block':<{tag_w}s}  RMSE (mV)  max|err| (mV)   FS (mV)  FS/S^2"
          f"  NRMSE (%FS)  max (%FS)")
    for e in errors:
        print(f"  {pairs_tag(e.driven):<{tag_w}s}  {e.rms_v * 1e3:9.4f}  "
              f"{float(np.max(np.abs(e.resid))) * 1e3:13.4f}  "
              f"{e.scale.fs * 1e3:8.3f}  {100 * e.scale.headroom:5.1f}%"
              f"  {e.nrmse_pct:11.3f}  {e.max_pct:9.3f}")
    rmse = np.array([e.rms_v for e in errors]) * 1e3
    nrmse = np.array([e.nrmse_pct for e in errors])
    head = np.array([e.scale.headroom for e in errors]) * 100.0
    worst = errors[int(np.argmax(rmse))]
    worst_n = errors[int(np.argmax(nrmse))]
    resid = np.concatenate([e.resid for e in errors])
    print(f"  RMSE over all {resid.size} reads = {np.sqrt(np.mean(resid ** 2)) * 1e3:.4f} mV   "
          f"per block: mean {rmse.mean():.4f}, worst {rmse.max():.4f} mV "
          f"({pairs_tag(worst.driven)})")
    print(f"  NRMSE per block: mean {nrmse.mean():.3f} %FS, "
          f"worst {nrmse.max():.3f} %FS ({pairs_tag(worst_n.driven)})")
    print(f"  FS = Y_max is {head.min():.1f}..{head.max():.1f} % of the free-phase S^2")


def make_plot(errors: list[BlockError], stds: list[np.ndarray], *, n: int,
              lo: float, hi: float, tag: str, path) -> None:
    """Step 8's four panels: measured against predicted, the pulls, the RMSE per
    block with its noise floor, and that same error against each block's full
    scale.

    ``stds`` are the per-read sigmas, the trace std with the systematic floor in
    quadrature -- the same sigma every upstream fit weights by.

    The NRMSE panel is the one to read across blocks and across runs: an RMSE in
    mV cannot be compared between blocks whose full scales differ by 3x, and
    ``RMSE / Y_max`` can.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless: write a PNG rather than open a window
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(22.5, 4.8))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    labelled = len(errors) <= 15
    solo = n == 1
    marker = "o" if solo else "s"

    for i, (e, std) in enumerate(zip(errors, stds)):
        col = colors[i % len(colors)]
        label = pairs_tag(e.driven) if labelled else None
        ax1.errorbar(e.pred * 1e3, e.y * 1e3, yerr=std * 1e3, fmt=marker, ms=4.5,
                     color=col, mfc="none" if solo else col, ecolor="lightgray",
                     elinewidth=0.8, capsize=1.5, ls="none", zorder=3, label=label)
        ax2.plot(e.pred * 1e3, e.resid / std, marker, color=col, ms=5,
                 mfc="none" if solo else col, label=label)

    top = 1e3 * max(max(float(np.max(e.pred)), float(np.max(e.y))) for e in errors)
    ax1.plot([0, top], [0, top], "--", color="gray", lw=1, zorder=1)
    ax1.set_xlabel("predicted Y  (mV)")
    ax1.set_ylabel("measured Y, dark-subtracted  (mV)")
    ax1.set_title(f"Predicted vs measured  [{tag}]\n"
                  f"{n}-pair blocks, {errors[0].y.size} patterns each")
    if labelled:
        ax1.legend(fontsize=7, loc="upper left", ncol=2 if len(errors) > 8 else 1)

    ax2.axhspan(-1, 1, color="tab:blue", alpha=0.12, label=r"$\pm1\sigma$")
    ax2.axhline(0, color="gray", ls="--", lw=1)
    ax2.set_xlabel("predicted Y  (mV)")
    ax2.set_ylabel("Pull = (meas - pred) / std")
    ax2.set_title("Pulls")
    ax2.legend(fontsize=7, ncol=2)

    rmse = np.array([e.rms_v for e in errors]) * 1e3
    floor = np.array([np.sqrt(np.mean(std ** 2)) for std in stds]) * 1e3
    xs = np.arange(len(errors))
    for i in xs:
        col = colors[i % len(colors)]
        ax3.bar(i, rmse[i], width=0.62, color=col, alpha=0.45 if solo else 0.85,
                edgecolor=col, lw=1.4, hatch="//" if solo else "")
        ax3.plot([i - 0.31, i + 0.31], [floor[i]] * 2, "-", color="0.35", lw=1.6, zorder=4)
        if labelled:
            ax3.text(i, rmse[i], f"{rmse[i]:.3f}", ha="center", va="bottom", fontsize=7)
    ax3.axhline(rmse.mean(), color="crimson", ls="--", lw=1.5,
                label=f"mean RMSE = {rmse.mean():.3f} mV")
    ax3.plot([], [], "-", color="0.35", lw=1.6, label="rms(std), noise floor")
    if len(errors) <= 30:
        ax3.set_xticks(xs)
        ax3.set_xticklabels([pairs_tag(e.driven) for e in errors],
                            rotation=0 if solo else 45,
                            ha="center" if solo else "right", fontsize=7)
        ax3.set_xlabel("driven block")
    else:
        ax3.set_xlabel(f"driven block (1 .. {len(errors)})")
    ax3.set_ylabel("RMSE = rms(meas - pred)  (mV)")
    ax3.set_title(f"Model error per block  [{tag}]\n"
                  f"x, w uniform in [{lo:g}, {hi:g}]")
    ax3.legend(fontsize=7, loc="upper left")

    # --- 4. the same error, but against each block's own full scale
    nrmse = np.array([e.nrmse_pct for e in errors])
    floor_pct = np.array([100.0 * np.sqrt(np.mean(std ** 2)) / e.scale.fs
                          for e, std in zip(errors, stds)])
    for i in xs:
        col = colors[i % len(colors)]
        ax4.bar(i, nrmse[i], width=0.62, color=col, alpha=0.45 if solo else 0.85,
                edgecolor=col, lw=1.4, hatch="//" if solo else "")
        ax4.plot([i - 0.31, i + 0.31], [floor_pct[i]] * 2, "-", color="0.35",
                 lw=1.6, zorder=4)
        if labelled:
            ax4.text(i, nrmse[i], f"{nrmse[i]:.2f}", ha="center", va="bottom",
                     fontsize=7)
    ax4.axhline(nrmse.mean(), color="crimson", ls="--", lw=1.5,
                label=f"mean NRMSE = {nrmse.mean():.2f} %FS")
    ax4.plot([], [], "-", color="0.35", lw=1.6, label="rms(std) / FS, noise floor")
    if len(errors) <= 30:
        ax4.set_xticks(xs)
        ax4.set_xticklabels([pairs_tag(e.driven) for e in errors],
                            rotation=0 if solo else 45,
                            ha="center" if solo else "right", fontsize=7)
        ax4.set_xlabel("driven block")
    else:
        ax4.set_xlabel(f"driven block (1 .. {len(errors)})")
    ax4.set_ylabel("NRMSE = rms(meas - pred) / FS  (%)")
    ax4.set_title(f"Model error vs full scale  [{tag}]")
    ax4.legend(fontsize=7, loc="upper left")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def compare_csv(path, step7: Path, *, pairs=None, method=None, n: int | None = None,
                out_dir=None) -> dict[int, list[BlockError]]:
    """Recompute every prediction for a recorded CSV and quote the error (no hardware).

    ``pairs`` (``--p``) keeps only the blocks driven entirely from that pool, so
    one CSV can be re-read for a subset of the pairs it recorded; ``n`` then
    selects a block size within what is left.

    The PNGs go to ``out_dir``, created if missing (default OUT_DIR).
    """
    path = Path(path)
    out_dir = Path(OUT_DIR if out_dir is None else out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _, blocks, meta = load_verify_csv(path)
    recorded = meta.get("step7_json")
    if recorded and Path(recorded).name != Path(step7).name:
        print(f"NOTE: {path.name} was collected against {Path(recorded).name}; "
              f"comparing against {Path(step7).name}")
    lo, hi = drive_bounds(meta, (DRIVE_MIN, DRIVE_MAX))
    wanted = select_pairs(PAIRS if pairs is None else pairs)
    if wanted is not None:
        # A block is kept only if EVERY pair it drove is in the pool: a block
        # that also drove an excluded pair measured a different interference,
        # so it cannot stand in for one of the selected blocks.
        keep = [b for b in blocks if set(b.driven) <= set(wanted)]
        if not keep:
            recorded = sorted({k for b in blocks for k in b.driven})
            raise ValueError(f"{path.name} has no block driven only from pairs "
                             f"{wanted}; it recorded {recorded}")
        if len(keep) != len(blocks):
            print(f"Selected pairs {wanted}: keeping {len(keep)} of "
                  f"{len(blocks)} recorded blocks")
        blocks = keep
    driven = sorted({k for b in blocks for k in b.driven})
    models, phases, _ = load_forward_model(step7, pairs=driven, method=method)
    in_file = sorted({b.n for b in blocks})
    ns = in_file
    if n is not None:
        if n not in in_file:
            raise ValueError(f"{path.name} has no {n}-pair blocks; it has n = {in_file}")
        ns = [n]
    print(f"Loaded {sum(b.y.size for b in blocks)} reads in {len(blocks)} blocks "
          f"(n = {in_file}) from {path}")
    if meta.get("status", "complete") != "complete":
        print(f"NOTE: this run is {meta['status']}")

    tag = method or "auto"
    out: dict[int, list[BlockError]] = {}
    for k in ns:
        mine = [b for b in blocks if b.n == k]
        errors = [evaluate(b, models, phases) for b in mine]
        report(errors, n=k, lo=lo, hi=hi)
        # step 8's report name: <csv stem>_compare_<method>.png
        suffix = "" if len(in_file) == 1 else f"_{k}pair"
        png = out_dir / f"{path.stem}{suffix}_compare_{tag}.png"
        make_plot(errors, [floor_std(b.std_v) for b in mine], n=k, lo=lo, hi=hi,
                  tag=tag, path=png)
        print(f"  Plot saved to {png}")
        out[k] = errors
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="calib_npair_verify.py",
        description="Step 8 -- n-pair verification: random patterns, measured vs "
                    "predicted, error as RMSE in mV.",
        epilog="--step7 has no default: that one file is the whole forward model "
               "(its embedded step 3 is the layout, step 6 the etas, step 7 the "
               "phases), so comparing against the wrong one reports a model error "
               "that is really a file mix-up.",
    )
    parser.add_argument("--step7", required=True, type=Path, metavar="JSON",
                        help="combined step-7 result JSON (required)")
    parser.add_argument("--n", type=int, default=None,
                        help="pairs driven at once; required to collect, "
                             "selects one n when comparing")
    parser.add_argument("--p", nargs="+", type=int, default=None, metavar="PAIR",
                        help="only use these pairs, e.g. --p 2 3 4: the blocks "
                             "are the n-subsets of THIS pool, not of every "
                             "calibrated pair (when comparing, keeps only the "
                             "recorded blocks driven entirely from it)")
    parser.add_argument("csv", nargs="?", type=Path,
                        help="compare this recorded CSV offline instead of collecting")
    parser.add_argument("--method", default=PHASE_METHOD,
                        help="stored step-7 fit to predict from (a v1 JSON stores several)")
    parser.add_argument("--out", type=Path, default=None, metavar="DIR",
                        help=f"directory for the CSV and PNGs, created if missing "
                             f"(default {OUT_DIR})")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    try:
        wanted = select_pairs(args.p)
    except ValueError as exc:
        parser.error(str(exc))
    if wanted is not None and args.n is not None and args.n > len(wanted):
        parser.error(f"--n {args.n} needs at least {args.n} pairs, but --p "
                     f"selects {len(wanted)}: {wanted}")

    if args.csv is not None:            # offline comparison, no hardware
        compare_csv(args.csv, args.step7, pairs=wanted, method=args.method,
                    n=args.n, out_dir=args.out)
        return 0
    if args.n is None:
        parser.error("--n is required to collect (how many pairs to drive at once)")
    collect(args.step7, args.n, pairs=wanted, method=args.method, out_dir=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
