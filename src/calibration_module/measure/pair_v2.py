"""Step 6 v2 acquisition -- drive one pair's schedule, return rows.

This half touches the instruments and does no fitting.  It hands back the raw
``(repeat, x, w, mean_v, std_v)`` rows that
:func:`calibration_module.fit.pair_v2.average_levels` turns into levels, so the
offline script and the GUI collect data through exactly one implementation and
differ only in what they do with the rows afterwards.

``monitor`` is anything with the ``monitor_cycle(timeout=..., single=...)``
protocol -- a :class:`~daq_module.DAQController` in every current caller.  The
near-rail escalation additionally wants ``last_values`` and the ``_settings``
the controller was configured with; a monitor without them still measures, with
autoranging skipped and said so once, rather than failing.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np

from ..fit.pair_v2 import DEFAULT_CONFIG, PairV2Config

__all__ = [
    "PairV2Acq",
    "DEFAULT_ACQ",
    "PairV2Aborted",
    "PairV2Progress",
    "build_schedule",
    "read_point_autorange",
    "set_range",
    "can_autorange",
    "measure_pair",
]


@dataclass(frozen=True)
class PairV2Acq:
    """The acquisition half of a step-6 run's setup -- timing and input range.

    Split from :class:`~calibration_module.fit.pair_v2.PairV2Config` because the
    two have different lifetimes: re-fitting a saved CSV needs the fit config and
    must not care what windows the data was taken with, while the GUI's DAQ panel
    tunes these without touching the estimator.
    """

    #: DAQ averaging window when at most one beam is on (``x == 0 or w == 0``,
    #: including the all-off dark).  A weak signal needs the extra averaging.
    t_single_s: float = 10.0
    #: Averaging window when both beams of the pair are on.
    t_both_s: float = 8.0
    #: Wait after each SLM pattern change, before reading.
    settle_s: float = 0.25
    #: Default +/- input range.  The board quantizes it (+/-0.1, 0.2, 0.5, 1, 2,
    #: 5, 10 V) and rounds a request UP, so ask for one of those.  +/-0.1 V is
    #: the most sensitive and right for almost every level.
    range_v: float = 0.1
    #: Remeasure range for a near-rail read; the next quantized step up, which
    #: costs one bit of resolution.
    range_wide_v: float = 0.2
    #: "Near the rail": raw ``|peak| >=`` this fraction of the range.
    near_rail_frac: float = 0.95
    #: Escalate a near-rail read to :attr:`range_wide_v`.  A clipped read is a
    #: silently wrong mean rather than an error, so this defaults on.
    autorange: bool = True
    #: Negate the reading.  The transimpedance amplifier outputs a NEGATIVE
    #: voltage for positive light, so recording a positive light signal means
    #: inverting.  Getting this wrong is not a cosmetic error: the estimator
    #: fits ``b = eta^2`` and takes its square root, so a sign-flipped run
    #: yields ``b < 0`` and ``eta = nan`` rather than a plausible wrong answer.
    invert: bool = True


#: The acquisition every validated run to date used.
DEFAULT_ACQ = PairV2Acq()


class PairV2Aborted(Exception):
    """Raised when a stop_event interrupts a sweep."""


@dataclass
class PairV2Progress:
    """One acquisition's worth of progress, for a GUI bar or a print line."""

    step: int                 # 1-based, across the whole run
    total: int
    pair_index: int
    repeat: int
    x: float
    w: float
    mean_v: float
    std_v: float
    range_v: float
    single: bool
    duration_s: float

    @property
    def std_ratio(self) -> float:
        return abs(self.std_v / self.mean_v) if self.mean_v else float("inf")

    def line(self) -> str:
        """The offline script's progress line."""
        wide = f"  [+/-{self.range_v:g} V]" if self.range_v != DEFAULT_ACQ.range_v else ""
        return (f"[{self.step}/{self.total}] pair {self.pair_index} "
                f"rep {self.repeat} x={self.x:.3f} w={self.w:.3f} "
                f"({self.duration_s:.0f}s) -> {self.mean_v*1000:.4f} mV  "
                f"std ratio {self.std_ratio*100:.2f}%{wide}")


ProgressCallback = Callable[[PairV2Progress], None]


def build_schedule(cfg: PairV2Config = DEFAULT_CONFIG, grid=None) -> list[tuple[int, float, float]]:
    """Interleaved acquisition order -> ``(repeat, x, w)`` per acquisition.

    Round-robin passes rather than n back-to-back reads of one level, so a slow
    drift shows up as scatter across a level's repeats instead of masquerading
    as a slope.  Within each pass, levels run brightest first, so the very first
    acquisition of a pair is its brightest point and a dead or blocked beam is
    visible immediately.

    Defaults to ``cfg.full_grid()`` -- the estimator's levels plus the
    verification block.
    """
    grid = cfg.full_grid() if grid is None else grid
    order = sorted(
        range(len(grid)),
        key=lambda i: (-(grid[i][0] * grid[i][1]), -max(grid[i][0], grid[i][1]),
                       -grid[i][0]),
    )
    sched: list[tuple[int, float, float]] = []
    for rep in range(max(g[2] for g in grid)):
        for i in order:
            x, w, n = grid[i]
            if n > rep:
                sched.append((rep, float(x), float(w)))
    return sched


def run_seconds(schedule: Sequence[tuple[int, float, float]],
                acq: PairV2Acq = DEFAULT_ACQ) -> float:
    """How long one pair's schedule will take, so a caller can say so first.

    Ten pairs of the default grid is most of an hour; a user deserves that
    number before pressing Run, not after.
    """
    n_single = sum(1 for _, x, w in schedule if x == 0.0 or w == 0.0)
    n_both = len(schedule) - n_single
    return n_single * acq.t_single_s + n_both * acq.t_both_s + len(schedule) * acq.settle_s


def set_range(daq, rng: float) -> None:
    """Reconfigure only the input range, keeping every other setting.

    ``configure_monitor`` just stores the settings object and every acquisition
    hands them to the driver afresh, so swapping the range between reads is safe
    and instant.  ``_settings`` is the object the caller installed; rebuilding it
    with ``replace`` keeps the channel and windows in lockstep by construction
    rather than by copy.
    """
    from daq_module import DAQMonitorSettings

    base = daq._settings or DAQMonitorSettings()  # noqa: SLF001 -- see docstring
    daq.configure_monitor(replace(base, min_val=-rng, max_val=rng))


def can_autorange(monitor) -> bool:
    return hasattr(monitor, "_settings") and hasattr(monitor, "last_values")


def read_point_autorange(
    monitor, *, single: bool, acq: PairV2Acq = DEFAULT_ACQ, timeout: float = 30.0,
    log: Callable[[str], None] | None = None,
) -> tuple[float, float, float]:
    """One read, with a single near-rail escalation to ``acq.range_wide_v``.

    The clip test runs on the RAW trace peak (``monitor.last_values``), not on
    the reported mean: the mean comes off the low-passed trace, which pulls a
    clipped flat-top back below the rail, so a mean comfortably under 0.1 V can
    still hide railed samples.  A read whose raw peak is within
    ``acq.near_rail_frac`` of the range is remeasured once at the wide range and
    the wide reading replaces it; the sensitive range is restored either way.

    Returns ``(mean_v, std_v, range_v)`` where ``range_v`` is the +/- range the
    kept reading was taken at.  Inversion is applied here, so the mean handed
    back is already a positive light signal (see :attr:`PairV2Acq.invert`).
    """
    def _read() -> tuple[float, float]:
        sample = monitor.monitor_cycle(timeout=timeout, single=single)
        if sample is None:
            raise PairV2Aborted("monitor read aborted")
        mean = float(sample.value)
        std = getattr(sample, "std", None)
        if std is None or not np.isfinite(std):
            raw = getattr(monitor, "last_values", None)
            std = float(np.std(raw)) if raw is not None and np.size(raw) > 1 else 0.0
        # std is a spread and stays non-negative -- negating a trace leaves it
        # unchanged -- so only the mean carries the sign convention.
        return (-mean if acq.invert else mean), float(std)

    def _peak(fallback: float) -> float:
        raw = getattr(monitor, "last_values", None)
        if raw is None or not np.size(raw):
            return abs(fallback)
        return float(np.max(np.abs(raw)))

    mean_v, std_v = _read()
    if not (acq.autorange and can_autorange(monitor)):
        return mean_v, std_v, acq.range_v

    peak = _peak(mean_v)
    if peak < acq.near_rail_frac * acq.range_v:
        return mean_v, std_v, acq.range_v

    if log is not None:
        log(f"    near the +/-{acq.range_v:g} V rail (raw peak {peak:.4f} V, "
            f"mean {mean_v*1e3:.2f} mV) -> remeasuring at +/-{acq.range_wide_v:g} V")
    set_range(monitor, acq.range_wide_v)
    try:
        mean_v, std_v = _read()
        if log is not None and _peak(mean_v) >= acq.near_rail_frac * acq.range_wide_v:
            log(f"    ** WARNING: still near the rail at +/-{acq.range_wide_v:g} V "
                f"(raw peak {_peak(mean_v):.4f} V) -- this reading is suspect **")
    finally:
        set_range(monitor, acq.range_v)
    return mean_v, std_v, acq.range_wide_v


def measure_pair(
    monitor,
    slm,
    layout,
    index: int,
    schedule: Sequence[tuple[int, float, float]],
    *,
    cfg: PairV2Config = DEFAULT_CONFIG,
    acq: PairV2Acq = DEFAULT_ACQ,
    col_ratio: np.ndarray | None = None,
    read_timeout: float = 30.0,
    step0: int = 0,
    total: int | None = None,
    progress_callback: ProgressCallback | None = None,
    stop_event: threading.Event | None = None,
    log: Callable[[str], None] | None = None,
    row_hook: Callable[[], Mapping[str, str]] | None = None,
) -> list[tuple]:
    """Run one pair's interleaved schedule -> raw ``(rep, x, w, mean, std)`` rows.

    ``index`` is the pair LABEL; ``cfg.slot`` maps it onto the 0-based drive
    arrays.  Only that pair is on, every other channel off.  The SLM is
    rewritten for **every** acquisition, repeats included -- that rewrite is what
    makes a level's repeat scatter measure encoding repeatability rather than
    detector jitter, so it must not be optimised away for repeated levels.

    ``step0``/``total`` let a multi-pair run report one continuous progress count
    across pairs; they affect nothing but the numbers in
    :class:`PairV2Progress`.  Raises :class:`PairV2Aborted` if ``stop_event``
    is set between acquisitions.

    ``row_hook`` is called once per acquisition for two extra cells appended to
    every row -- in practice
    :meth:`~calibration_module.measure.pmt_temp.PmtTemperature.sample`, whose
    cached reading costs a GPIB round trip only once a minute.  Without it the
    rows are the plain 5-tuples and the CSV's two thermistor columns are blank.
    """
    from slm_module.encoding import encode_to_pattern

    zeros = np.zeros(layout.n_channels)
    slm_width, slm_height = slm.get_slm_info()
    total = len(schedule) if total is None else total
    slot = cfg.slot(index)

    rows: list[tuple] = []
    for offset, (rep, x_val, w_val) in enumerate(schedule):
        if stop_event is not None and stop_event.is_set():
            raise PairV2Aborted("step-6 sweep stopped by request")
        x_vals = zeros.copy()
        w_vals = zeros.copy()
        x_vals[slot] = x_val
        w_vals[slot] = w_val
        slm.display_array(
            encode_to_pattern(x_vals, w_vals, layout, slm_width, slm_height,
                              col_ratio=col_ratio)
        )
        if acq.settle_s:
            time.sleep(acq.settle_s)
        single = x_val == 0.0 or w_val == 0.0
        mean_v, std_v, range_v = read_point_autorange(
            monitor, single=single, acq=acq, timeout=read_timeout, log=log
        )
        row = (rep, float(x_val), float(w_val), mean_v, std_v)
        if row_hook is not None:
            cells = row_hook()
            row += (cells.get("pmt_ohm_Ohm", ""), cells.get("pmt_c", ""))
        rows.append(row)
        if progress_callback is not None:
            progress_callback(PairV2Progress(
                step=step0 + offset + 1, total=total, pair_index=index,
                repeat=rep, x=float(x_val), w=float(w_val),
                mean_v=mean_v, std_v=std_v, range_v=range_v, single=single,
                duration_s=acq.t_single_s if single else acq.t_both_s,
            ))
    return rows
