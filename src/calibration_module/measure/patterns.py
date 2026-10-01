"""Drive a list of multi-pair patterns and read each one -- the measurement loop
steps that verify a forward model share.

Sibling to :mod:`~calibration_module.measure.pair_v2` (step 6) and
:mod:`~calibration_module.measure.phase_v2` (step 7), and the same split: the
*step* file owns its run design, its output format and its fit, while the drive
loop, the input-range policy and the layout lookup live here, where more than
one step can reach them.

This used to live inside ``steps/calib_npair_verify.py``, which made the step
file that hosted it a library every later step had to import as ``S8``.  That
file is gone: its n-pair block verification was superseded by the level sweep
in :mod:`~calibration_module.steps.calib_step8`, and only this plumbing was
worth keeping.

Sign.  There is no ``flip`` here, and there must not be: the TIA's
negative-for-light output is inverted once, at the read (``acq.invert``).
:func:`run_patterns` aborts on the first bright pattern that reads below the
dark rather than hand back a run nothing can be compared against.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np

from calibration_module.fit.slm_response import calibration_result_from_dict
from calibration_module.fit.verify import pairs_tag, predict
from calibration_module.measure.pair_v2 import (
    DEFAULT_ACQ as BENCH_ACQ,
    PairV2Acq,
    read_point,
)
from slm_module.encoding import channel_layout_from_calibration

#: Pairs are numbered 1..N on the bench; keep in step with steps 6 and 7.
PAIR_INDEX_BASE = 1

#: The board rounds a requested range UP to one of these.
BOARD_RANGES_V = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)
#: A read's range is the smallest with ``|pred| <= RANGE_FILL * range``.
RANGE_FILL = 0.7
#: Re-read the all-off dark this often; every row keeps the latest.
DARK_EVERY_S = 300.0
#: The sign check waits for a pattern predicted at least this bright.
SIGN_CHECK_V = 1e-3


class PatternsAborted(Exception):
    """``stop_event`` was set between reads.

    Raised rather than returned so a caller cannot mistake a stopped run for a
    finished one: every writer of these CSVs marks the file ``partial`` off the
    back of the exception, and a silent early return would write ``complete``.
    """


@dataclass
class PatternsProgress:
    """One read of a sweep -- announced when it starts, again when it lands.

    Announced first because a read IS an averaging window: ten seconds at the
    bottom of a step-8 ladder, and the opening dark is ten more.  A UI told only
    about finished reads sits blank through the first fifteen seconds of every
    run and cannot tell that from a hang, which is the whole reason this exists.
    ``mean_v`` is ``None`` on the announcement and carries the read's inverted
    mean once it lands.

    ``step`` counts pattern reads FINISHED -- 0 through the opening dark,
    ``total`` once the last pattern lands -- so a bar can be driven off it
    without knowing that darks are not part of the count.  ``dark_v`` is the
    dark those reads are measured against, and is 0 on a dark's own events, so
    :attr:`net_v` is the light signal for a pattern and the dark itself for a
    dark.  ``row`` is the row just appended, extras included, so a caller can
    read back labels this function never interpreted.
    """

    step: int
    total: int
    block: int
    driven: tuple[int, ...]
    single: bool
    t_read_s: float
    pred_v: float = 0.0
    dark_v: float = 0.0
    mean_v: float | None = None
    std_v: float | None = None
    range_v: float | None = None
    row: dict | None = None

    @property
    def net_v(self) -> float | None:
        """The read's mean over its dark; ``None`` while the read is in flight."""
        return None if self.mean_v is None else self.mean_v - self.dark_v

    @property
    def std_ratio(self) -> float:
        net = self.net_v
        if self.std_v is None or not net:
            return float("inf")
        return abs(self.std_v / net)


ProgressCallback = Callable[[PatternsProgress], None]


#: The bench acquisition, re-exported.  A pattern sweep used to declare its own
#: (10 s dark, 5 s patterns, +/-0.1 V, per-read autoranging); it now takes step
#: 6's and step 7's, unchanged, because comparing their numbers to this step's
#: only means anything if they were measured the same way.
DEFAULT_ACQ = BENCH_ACQ


# ======================================================================
# inputs
# ======================================================================

def select_pairs(pairs) -> list[int] | None:
    """Normalize a ``--p`` / ``PAIRS`` selection; ``None`` means every pair.

    Duplicates collapse and the order does not matter -- the caller sorts the
    pool it ends up driving either way.
    """
    if pairs is None:
        return None
    out = sorted({int(k) for k in pairs})
    if not out:
        raise ValueError("pair selection is empty; drop it to use every pair")
    return out


def slot(pair: int) -> int:
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
    layout = channel_layout_from_calibration(calibration_result_from_dict(step3))
    bad = [k for k in pairs if not 0 <= slot(k) < layout.n_channels]
    if bad:
        raise ValueError(f"pair(s) {bad} out of range: the layout has "
                         f"{layout.n_channels} pairs, numbered from {PAIR_INDEX_BASE}")
    return layout


def peak_v(daq) -> float:
    """The RAW trace peak of the last read, in volts, or NaN if unavailable.

    The raw trace is ``monitor.last_values``, cached BEFORE the low-pass runs,
    so this is the largest of the window's unfiltered samples -- 8000 of them at
    8 s and 1 kS/s.  ``abs`` so the amplifier's sign convention cannot hide a
    rail.

    It is worth a column because the mean structurally cannot show clipping: the
    20 Hz low-pass averages a flat-topped plateau together with the samples that
    were not clipped and hands back a plausible number.  The peak is sampled
    upstream of the filter, so it is the only value in the chain that sees the
    rail.

    Nothing acts on it.  It is a record to look at if a run turns out wrong, and
    the one input to
    :func:`~calibration_module.steps.calib_step8.probe_verdict`'s clipping
    verdict.  Note it is a MAXIMUM and not a count: a single spike out of 8000
    samples gives the same value as a sustained flat-top while moving the mean by
    ~0.01 %, so a large ``peak_v`` means look, not discard.
    """
    raw = getattr(daq, "last_values", None)
    if raw is None or not np.size(raw):
        return float("nan")
    return float(np.max(np.abs(raw)))


def range_for(pred_v: float) -> tuple[float, float]:
    """``(range, wide)`` for a read predicted at ``pred_v``.

    ``range`` is the smallest board range with ``|pred_v| <= RANGE_FILL * range``;
    ``wide`` is the next one up, where the near-rail escalation remeasures.

    ADVISORY.  Nothing in a run calls this any more: every read is taken at the
    one fixed :data:`~calibration_module.measure.pair_v2.RANGE_V`.  It survives
    because "what range would this read need?" is still the right question for a
    dry-run table, which is where step 8 asks it.
    """
    need = abs(float(pred_v)) / RANGE_FILL
    last = len(BOARD_RANGES_V) - 1
    i = next((i for i, r in enumerate(BOARD_RANGES_V) if need <= r), last)
    return BOARD_RANGES_V[i], BOARD_RANGES_V[min(i + 1, last)]


# ======================================================================
# the loop
# ======================================================================

def run_patterns(daq, slm, layout, blocks, x, w, models, phases, *, rows: list,
                 acq: PairV2Acq = DEFAULT_ACQ, encode=None, log=print,
                 extra=None, y_sat: float | None = None, dark: float | None = None,
                 row_hook=None, stop_event=None,
                 progress_callback: ProgressCallback | None = None) -> None:
    """Drive every block's patterns; append one row per read to ``rows``.

    ``x`` and ``w`` are ``(len(blocks), patterns, n)``.  ``rows`` is the caller's
    list, filled as the run goes, so an interrupted run keeps what it read.
    ``encode`` defaults to ``encode_to_pattern``.

    ``extra``, if given, is ``(len(blocks), patterns)`` of mappings merged into
    each row, so a caller can label its reads without this function knowing what
    the labels mean.

    ``row_hook(row)``, if given, is called with each finished row just before it
    is appended, so a caller can add columns it can only measure *during* the
    run -- unlike ``extra``, which is fixed before the first read.  It runs
    between the DAQ read and the next SLM frame, so a slow instrument there
    stretches the run but never overlaps an acquisition.  It must not raise.

    ``progress_callback(p)``, if given, is called TWICE per read -- once when it
    starts, with ``p.mean_v`` None, and once when it lands -- so a UI can say
    what it is waiting for instead of going quiet for a whole averaging window.
    The all-off darks are announced the same way and carry ``single``.  Like
    ``row_hook`` it must not raise, and it is called on this thread.

    ``stop_event``, if given, is checked between reads: set it and the loop
    raises :class:`PatternsAborted` with ``rows`` holding everything read so
    far.  Between reads, not during one -- a half-finished acquisition is not a
    reading, and the caller keeps a CSV of whole rows either way.

    Every row also carries ``peak_v``, the raw trace peak of its read -- see
    :func:`peak_v`.  It reaches a CSV only if the caller names it in ``extra``.

    ``y_sat`` is passed to :func:`predict`, so a pattern's logged diff and
    ``pred_v`` use the saturated model; ``None`` (the default) keeps it linear.

    Every read takes the SAME window, the one the caller configured the monitor
    with.  There used to be a ``t_both_of(pred_v)`` hook that re-configured the
    DAQ per read so step 8 could run a short window at the bright end of its
    ladder; it is gone, along with the per-read range change, because a sweep
    whose points were each measured differently cannot be compared point to
    point -- which is the whole job of a ladder.

    ``dark``, if given, is USED as the all-off level and none is ever read --
    not the opening one and not the :data:`DARK_EVERY_S` refresh.  It is for a
    caller that spans many calls to this function and wants one dark for the
    lot, the way a single-sweep step gets one for free.  :mod:`onn_module.onn_infer`
    is that caller: it drives its pattern list in ~60 chunks, so the default
    costs it ~130 dark reads to track a level that moved 0.03 mV over a 14 h run
    and cancels exactly in a per-image argmax.

    Steps 6, 7 and 8 leave it ``None``.  Step 6 in particular must: there the
    all-off read is not an offset to subtract but a fitted point of the
    background block, the same magnitude as the ``a_x``/``a_w`` it anchors.
    """
    if encode is None:
        from slm_module.encoding import encode_to_pattern as encode
    zeros = np.zeros(layout.n_channels)
    slm_width, slm_height = slm.get_slm_info()
    total = len(blocks) * x.shape[1]

    def display(driven, xs, ws) -> None:
        x_vals, w_vals = zeros.copy(), zeros.copy()
        for k, xk, wk in zip(driven, xs, ws):
            x_vals[slot(k)], w_vals[slot(k)] = xk, wk
        slm.display_array(encode(x_vals, w_vals, layout, slm_width, slm_height))
        if acq.settle_s:
            time.sleep(acq.settle_s)

    def read(single: bool) -> tuple[float, float, float]:
        """One read, at the one range, for the one window.

        This used to pick a range from the prediction and set it before every
        read -- so a 90-rung ladder spanned three or four of them and took its
        dark on a fifth, which put the board's per-range offset straight into
        the dark subtraction.  Nothing switches now.
        """
        return read_point(daq, single=single, acq=acq, log=log)

    def report(**fields) -> None:
        if progress_callback is not None:
            progress_callback(PatternsProgress(step=len(rows), total=total,
                                               **fields))

    def read_dark() -> tuple[float, float]:
        # Announced BEFORE the frame goes up, not after: the settle and the
        # ten-second window are exactly the silence this is here to fill.
        dark_fields = dict(block=0, driven=(), single=True,
                           t_read_s=acq.t_meas_s)
        report(**dark_fields)
        display((), (), ())
        dark, _, _ = read(True)
        log(f"dark (all off, {acq.t_meas_s:g}s) = {dark * 1e3:.4f} mV")
        report(**dark_fields, mean_v=dark)
        return dark, time.monotonic()

    given_dark = dark is not None
    dark, dark_t = (float(dark), time.monotonic()) if given_dark else read_dark()
    checked = False
    i = 0
    for b, driven in enumerate(blocks):
        if len(blocks) > 1:
            log(f"\n--- block {b + 1}/{len(blocks)}: pairs {pairs_tag(driven)} ---")
        for j in range(x.shape[1]):
            if stop_event is not None and stop_event.is_set():
                raise PatternsAborted(f"stopped after {len(rows)} of {total} reads")
            if not given_dark and time.monotonic() - dark_t > DARK_EVERY_S:
                dark, dark_t = read_dark()
            xs, ws = x[b, j], w[b, j]
            pred = float(predict(models, phases, driven, xs, ws, y_sat=y_sat))
            fields = dict(block=b, driven=tuple(driven), single=False,
                          t_read_s=acq.t_meas_s, pred_v=pred, dark_v=dark)
            report(**fields)
            display(driven, xs, ws)
            mean, std, used = read(False)
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
                        f"(acq.invert) -- check the amplifier polarity and that the "
                        f"beam is on; do not negate it a second time.")
            row = {"block": b, "driven": driven, "x": xs, "w": ws,
                   "dark_v": dark, "mean_v": mean, "std_v": std,
                   "range_v": used, "pred_v": pred, "peak_v": peak_v(daq)}
            if extra is not None:
                row.update(extra[b][j])
            if row_hook is not None:
                row_hook(row)
            rows.append(row)
            report(**fields, mean_v=mean, std_v=std, range_v=used, row=row)
            log(f"[{i}/{total}] {pairs_tag(driven)} #{j + 1}: "
                f"{(mean - dark) * 1e3:9.4f} mV  pred {pred * 1e3:9.4f}  "
                f"diff {(mean - dark - pred) * 1e3:+.4f}  [+/-{used:g} V]")
