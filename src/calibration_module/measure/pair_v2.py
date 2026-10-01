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
    "T_MEAS_S",
    "RANGE_V",
    "PairV2Aborted",
    "PairV2Progress",
    "build_schedule",
    "read_point",
    "set_range",
    "measure_pair",
]


#: The ONE measurement window, for every read of every step.
#:
#: Steps 6, 7 and 8 used to run 8, 10 and a 10 -> 5 s ramp, and the all-off dark
#: ran longer still.  Nothing was gained by that: a read's noise is the trace
#: spread, the fit floors it at ``STD_FLOOR_V`` anyway, and every difference in
#: window was one more reason two steps' numbers could not be compared directly.
#: One window, dark included, so a step-7 point and the step-6 eta it is pinned
#: against were measured the same way.
T_MEAS_S = 8.0

#: The ONE input range.  Deliberately the same number as
#: :data:`~calibration_module.steps.calib_step8.AMP_LIMIT_V`: the transimpedance
#: amplifier saturates at 1 V, so a read the board could only hold at +/-2 V is
#: already an amplifier flat-top and measuring it more precisely measures
#: nothing.  Board rail and amplifier rail coincide, which makes the near-rail
#: peak test an amplifier-clip test for free.
#:
#: Widening costs nothing.  The 0928 gain sweep switches +/-0.1 -> +/-0.2 V mid
#: run at fixed PMT gain and the all-off trace std moves 33.78 -> 33.75 uV; the
#: board is dither-dominated even at its most sensitive range.  On a 16-bit
#: USB-6251 the +/-1 V LSB is 30.5 uV, which the 20 Hz low-pass takes to 1.8 uV
#: in the reported std and 0.09 uV on the reported mean -- 2000x under
#: ``STD_FLOOR_V``.  So the range is chosen for headroom, never for resolution.
RANGE_V = 1.0


@dataclass(frozen=True)
class PairV2Acq:
    """The acquisition half of a run's setup -- timing and input range.

    Split from :class:`~calibration_module.fit.pair_v2.PairV2Config` because the
    two have different lifetimes: re-fitting a saved CSV needs the fit config and
    must not care what windows the data was taken with, while the GUI's DAQ panel
    tunes these without touching the estimator.

    Named for step 6, but it is now every step's: steps 6, 7 and 8 and the ONN
    inference loop all take :data:`DEFAULT_ACQ` unchanged.  The per-step
    variants are gone -- see :data:`T_MEAS_S`.
    """

    #: Averaging window.  ONE field: every read of every step takes the same
    #: one, the all-off dark included.  There used to be a ``t_single_s`` for
    #: the dark and a ``t_both_s`` for everything else, on the argument that a
    #: near-zero read needs more averaging -- but the dark is subtracted from
    #: every point of a sweep, so a dark averaged differently puts its own noise
    #: into all of them.
    t_meas_s: float = T_MEAS_S
    #: Wait after each SLM pattern change, before reading.
    settle_s: float = 0.25
    #: The +/- input range, for EVERY read including the dark.  The board
    #: quantizes it (+/-0.1, 0.2, 0.5, 1, 2, 5, 10 V) and rounds a request UP,
    #: so ask for one of those.  See :data:`RANGE_V` for why it is 1 V and why
    #: nothing is gained by going narrower.
    range_v: float = RANGE_V
    #: "Near the rail": raw ``|peak| >=`` this fraction of the range.  At
    #: :data:`RANGE_V` that is the AMPLIFIER's rail too, so tripping this means
    #: the read is a flat-top and no range setting can rescue it.  Nothing is
    #: remeasured; :func:`read_point` logs it and ``peak_v`` records it.
    #: Negate the reading.  The transimpedance amplifier outputs a NEGATIVE
    #: voltage for positive light, so recording a positive light signal means
    #: inverting.  Getting this wrong is not a cosmetic error: the estimator
    #: fits ``b = eta^2`` and takes its square root, so a sign-flipped run
    #: yields ``b < 0`` and ``eta = nan`` rather than a plausible wrong answer.
    invert: bool = True


#: The bench acquisition: one window, one range, one settle, for every read of
#: every step.  Steps 6/7/8 and the ONN loop all import THIS rather than
#: declaring their own, so "same conditions" is a property of the code and not
#: of whoever set the spinboxes.
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
    return len(schedule) * (acq.t_meas_s + acq.settle_s)


def set_range(daq, rng: float) -> None:
    """Reconfigure only the input range, keeping every other setting.

    No step calls this any more -- they all run at the one fixed
    :data:`RANGE_V` -- but a bench script that deliberately walks the signal
    over decades (``drafts/pmt_gain_sweep.py``) still wants it.

    ``configure_monitor`` just stores the settings object and every acquisition
    hands them to the driver afresh, so swapping the range between reads is safe
    and instant.  ``_settings`` is the object the caller installed; rebuilding it
    with ``replace`` keeps the channel and windows in lockstep by construction
    rather than by copy.
    """
    from daq_module import DAQMonitorSettings

    base = daq._settings or DAQMonitorSettings()  # noqa: SLF001 -- see docstring
    daq.configure_monitor(replace(base, min_val=-rng, max_val=rng))


def read_point(
    monitor, *, single: bool, acq: PairV2Acq = DEFAULT_ACQ, timeout: float = 30.0,
    log: Callable[[str], None] | None = None,
) -> tuple[float, float, float]:
    """One read at ``acq.range_v``.  One read: no escalation, and no warning.

    This used to be ``read_point_autorange``, which remeasured a near-rail read
    one range up; then it kept the near-rail test as a log line.  Both are gone.

    Why the WARNING went too, which is less obvious than why the re-read did.
    The test fired when the raw trace peak passed 0.95 of the range, i.e. 950 mV.
    On this bench the peak runs ~1.22x the mean (p90 1.31), so real signal
    cannot reach that threshold until the MEAN is around 724..780 mV -- and above
    950 mV the mean is at the rail and reports it by itself.  So the test only
    ever said something the mean could not in a narrow window, and
    :data:`~calibration_module.steps.calib_step8.LADDER_CEILING_V` now stops
    every sweep at 450 mV, a factor of 1.6 below it.  What CAN still reach
    950 mV from a 450 mV read is a spike: the observed peak/mean tail is 1.90 at
    p99 and 4.04 at worst.  Every firing would therefore have been a false
    alarm -- the same mechanism that used to trip the escalation, minus the
    re-read.

    The peak itself is not gone and costs nothing: ``run_patterns`` still puts
    :func:`~calibration_module.measure.patterns.peak_v` on every row as the
    durable record, and
    :func:`~calibration_module.steps.calib_step8.probe_verdict` still judges
    clipping on it.  That is the one place it earns its keep, because the probe
    is the one thing that deliberately drives up into 724..950 mV looking for
    where the amplifier starts to compress.

    ``single`` no longer picks a window (there is one) but is still passed to
    the monitor and still marks the read as a dark / one-beam point.

    Returns ``(mean_v, std_v, range_v)``.  Inversion is applied here, so the
    mean handed back is already a positive light signal (see
    :attr:`PairV2Acq.invert`).
    """
    sample = monitor.monitor_cycle(timeout=timeout, single=single)
    if sample is None:
        raise PairV2Aborted("monitor read aborted")
    mean_v = float(sample.value)
    std_v = getattr(sample, "std", None)
    if std_v is None or not np.isfinite(std_v):
        raw = getattr(monitor, "last_values", None)
        std_v = float(np.std(raw)) if raw is not None and np.size(raw) > 1 else 0.0
    # std is a spread and stays non-negative -- negating a trace leaves it
    # unchanged -- so only the mean carries the sign convention.
    mean_v = -mean_v if acq.invert else mean_v
    std_v = float(std_v)

    return mean_v, std_v, acq.range_v


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
        mean_v, std_v, range_v = read_point(
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
                duration_s=acq.t_meas_s,
            ))
    return rows
