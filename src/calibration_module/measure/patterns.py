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
from dataclasses import replace
from pathlib import Path

import numpy as np

from calibration_module.fit.slm_response import calibration_result_from_dict
from calibration_module.fit.verify import pairs_tag, predict
from calibration_module.measure.pair_v2 import (
    PairV2Acq,
    can_autorange,
    read_point_autorange,
    set_range,
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


#: The acquisition a pattern sweep defaults to.  A step overrides what it needs.
DEFAULT_ACQ = PairV2Acq(
    t_single_s=10.0,        # the all-off dark
    t_both_s=5.0,           # every pattern, unless t_both_of() overrides per read
    settle_s=0.25,          # wait after each SLM pattern change, before reading
    range_v=0.1,            # the dark's range; each pattern's comes from range_for()
    range_wide_v=0.2,
    autorange=True,
    invert=True,
)


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
    enc_method = (payload.get("encoding") or {}).get("method", "interp")
    layout = channel_layout_from_calibration(
        calibration_result_from_dict(step3), method=enc_method)
    bad = [k for k in pairs if not 0 <= slot(k) < layout.n_channels]
    if bad:
        raise ValueError(f"pair(s) {bad} out of range: the layout has "
                         f"{layout.n_channels} pairs, numbered from {PAIR_INDEX_BASE}")
    return layout


def range_for(pred_v: float) -> tuple[float, float]:
    """``(range, wide)`` for a read predicted at ``pred_v``.

    ``range`` is the smallest board range with ``|pred_v| <= RANGE_FILL * range``;
    ``wide`` is the next one up, where the near-rail escalation remeasures.
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
                 extra=None, y_sat: float | None = None, t_both_of=None,
                 row_hook=None, stop_event=None) -> None:
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

    ``stop_event``, if given, is checked between reads: set it and the loop
    raises :class:`PatternsAborted` with ``rows`` holding everything read so
    far.  Between reads, not during one -- a half-finished acquisition is not a
    reading, and the caller keeps a CSV of whole rows either way.

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
            x_vals[slot(k)], w_vals[slot(k)] = xk, wk
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
        if len(blocks) > 1:
            log(f"\n--- block {b + 1}/{len(blocks)}: pairs {pairs_tag(driven)} ---")
        for j in range(x.shape[1]):
            if stop_event is not None and stop_event.is_set():
                raise PatternsAborted(f"stopped after {len(rows)} of {total} reads")
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
                        f"(acq.invert) -- check the amplifier polarity and that the "
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
