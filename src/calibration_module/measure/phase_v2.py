"""Step 7 v2 acquisition -- sweep one target pair against the reference.

This half touches the instruments and does no fitting.  It hands back a raw
:class:`~calibration_module.fit.phase.PhaseResult` (no ``fit``) that
:func:`~calibration_module.fit.phase.fit_result` turns into a ``dPhi_comb``, so
the offline script and the GUI collect data through exactly one implementation
and differ only in what they do with the rows afterwards.  Same split, same
reasons, as :mod:`calibration_module.measure.pair_v2` -- and the near-rail
escalation is literally that module's, imported rather than retyped.

``monitor`` is anything with the ``monitor_cycle(timeout=..., single=...)``
protocol -- a :class:`~daq_module.DAQController` in every current caller.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ..fit.phase import PhaseResult
from .pair_v2 import PairV2Acq, PairV2Aborted, read_point_autorange

__all__ = [
    "PhaseV2Config",
    "DEFAULT_CONFIG",
    "PhaseV2Acq",
    "DEFAULT_ACQ",
    "PhaseV2Aborted",
    "PhaseV2Progress",
    "build_xw_sweep",
    "run_seconds",
    "measure_target",
]

#: A ``stop_event`` interrupted the sweep.  Step 6's exception reused rather
#: than twinned: the shared reader raises it, and a caller running both steps
#: back to back wants one ``except``.
PhaseV2Aborted = PairV2Aborted


@dataclass(frozen=True)
class PhaseV2Config:
    """The drive half of a step-7 run: which pairs, and the target ramp.

    The reference pair is held at :attr:`ref_level`; the target pair's TWO
    channels are swept together (``x_t = w_t``) over the
    :attr:`sweep_min`..:attr:`sweep_max` ramp.

    Both arms deliberately stop below 1.0.  Everything the fit pins comes from
    step 6, which fits its etas over ``fit_w_range = [0.2, 0.9]`` and EXCLUDES
    the measured ``(1, 1)`` point, so driving either arm fully on leans the
    whole fringe on an extrapolation.  1.0 is also the worst point to lean on:
    ``d(dPhi_SLM)/dv = 1/sqrt(v(1-v))`` diverges there while the trace std is
    the smallest of the sweep, so ``1/std^2`` weighting hands it most of the
    fit -- on the 0903 pair-3 fringe ~70% of the Fisher information, and
    dropping it alone moved ``dPhi_comb`` by 17 deg against a quoted +/-1.4 deg.
    """

    #: Pairs are LABELLED from here; the SLM drive arrays are always 0-based.
    #: Keep in step with the step-6 run whose JSON is being read back.
    pair_index_base: int = 1
    #: The common reference pair.  It defines ``Phi = 0``.
    ref_index: int = 1
    #: The reference is held at ``x_r = w_r = ref_level`` for every point.
    ref_level: float = 0.9
    #: Lowest per-side target intensity in the ramp.
    sweep_min: float = 0.1
    #: Highest per-side target intensity in the ramp.
    sweep_max: float = 0.9
    #: Points in the ramp, evenly spaced from :attr:`sweep_min` to
    #: :attr:`sweep_max`.
    n_points: int = 10

    def __post_init__(self) -> None:
        if self.n_points < 2:
            raise ValueError(f"need >= 2 ramp points, got {self.n_points}")
        if not (0.0 < self.sweep_min <= self.sweep_max <= 1.0):
            raise ValueError(
                "ramp must satisfy 0 < min <= max <= 1, got "
                f"{self.sweep_min} .. {self.sweep_max}"
            )
        if not (0.0 < self.ref_level <= 1.0):
            raise ValueError(f"ref_level must be in (0, 1], got {self.ref_level}")

    def slot(self, pair: int) -> int:
        """Pair label -> its 0-based slot in the SLM layout / drive arrays."""
        return int(pair) - self.pair_index_base

    def ramp(self) -> np.ndarray:
        """The commanded target intensities, in sweep order."""
        return np.round(
            np.linspace(self.sweep_min, self.sweep_max, self.n_points), 6
        )


#: The drive every validated v2 run from 0904 on used.
DEFAULT_CONFIG = PhaseV2Config()


@dataclass(frozen=True)
class PhaseV2Acq(PairV2Acq):
    """Step 7's acquisition settings -- step 6's, with step-7 defaults.

    A subclass rather than a twin so the shared
    :func:`~calibration_module.measure.pair_v2.read_point_autorange` takes it as
    the type it documents.  Only the defaults differ, and they differ for one
    reason: step 7 is the brightest of the calibration steps.  The reference
    stays on for every point while a second pair ramps up on top of it, so the
    bright end runs past the +/-0.1 V step 6 lives at -- where the board clips
    and returns a wrong mean rather than an error.
    """

    #: All-off dark only; at zero signal it needs the averaging.
    t_single_s: float = 10.0
    #: Every sweep point: the reference is on, so they are all bright.
    t_both_s: float = 10.0
    settle_s: float = 0.25
    #: One quantized step above step 6's, costing one bit of resolution.
    range_v: float = 0.2
    range_wide_v: float = 0.5


#: The acquisition every validated v2 run to date used.
DEFAULT_ACQ = PhaseV2Acq()


@dataclass
class PhaseV2Progress:
    """One acquisition's worth of progress, for a GUI bar or a print line."""

    step: int                 # 1-based, across the whole run
    total: int
    tgt_index: int
    ref_index: int
    x_t: float
    w_t: float
    x_r: float
    w_r: float
    mean_v: float
    std_v: float
    range_v: float
    single: bool              # the all-off dark read
    duration_s: float

    @property
    def std_ratio(self) -> float:
        return abs(self.std_v / self.mean_v) if self.mean_v else float("inf")

    def line(self) -> str:
        """The offline script's progress line."""
        wide = (f"  [+/-{self.range_v:g} V]"
                if self.range_v != DEFAULT_ACQ.range_v else "")
        head = f"[{self.step}/{self.total}] pair {self.tgt_index} "
        if self.single:
            return (f"{head}dark (all off, {self.duration_s:.0f}s) "
                    f"= {self.mean_v*1000:.4f} mV{wide}")
        return (f"{head}x=w={self.x_t:.3f} ({self.duration_s:.0f}s) -> "
                f"{self.mean_v*1000:.4f} mV  "
                f"std ratio {self.std_ratio*100:.2f}%{wide}")


ProgressCallback = Callable[[PhaseV2Progress], None]

Drive = tuple[float, float, float, float]


def build_xw_sweep(cfg: PhaseV2Config = DEFAULT_CONFIG) -> list[Drive]:
    """Drive tuples ``(x_t, w_t, x_r, w_r)``, target first.

    ``x_r = w_r = cfg.ref_level`` on every row and ``x_t = w_t`` steps over the
    ramp.  The fit follows:
    :func:`~calibration_module.fit.phase.fit_phase_fixed` takes
    ``g_ref = sqrt(x_r w_r)`` so ``a = eta_ref g_ref``, and ``dPhi_SLM`` already
    carried the reference's ``-phi_half(x_r) - phi_half(w_r)`` -- so a reference
    held below 1.0 needs no other change anywhere.
    """
    r = float(cfg.ref_level)
    return [(float(v), float(v), r, r) for v in cfg.ramp()]


def run_seconds(drive: Sequence[Drive], acq: PhaseV2Acq = DEFAULT_ACQ) -> float:
    """How long one target's sweep will take, so a caller can say so first."""
    n = len(drive)
    return acq.t_single_s + n * acq.t_both_s + (n + 1) * acq.settle_s


def measure_target(
    monitor,
    slm,
    layout,
    tgt_index: int,
    drive: Sequence[Drive],
    *,
    cfg: PhaseV2Config = DEFAULT_CONFIG,
    acq: PhaseV2Acq = DEFAULT_ACQ,
    col_ratio: np.ndarray | None = None,
    read_timeout: float = 30.0,
    step0: int = 0,
    total: int | None = None,
    progress_callback: ProgressCallback | None = None,
    stop_event: threading.Event | None = None,
    log: Callable[[str], None] | None = None,
    row_hook: Callable[[], Mapping[str, str]] | None = None,
) -> PhaseResult:
    """Drive one target pair against the reference over ``drive`` -> raw rows.

    ``tgt_index`` and ``cfg.ref_index`` are pair LABELS; ``cfg.slot`` maps them
    onto the 0-based drive arrays.  Only those two channels are driven, every
    other one held off.  An all-off dark is read once at the start (the longer
    ``t_single`` window) and stored per row, so the fit subtracts it row by row.

    Needs no step-6 model -- raw data only.  The returned
    :class:`~calibration_module.fit.phase.PhaseResult` carries no ``fit``;
    hand it to :func:`~calibration_module.fit.phase.fit_result` for that.

    ``step0``/``total`` let a multi-target run report one continuous progress
    count; they affect nothing but the numbers in :class:`PhaseV2Progress`.
    Raises :data:`PhaseV2Aborted` if ``stop_event`` is set between acquisitions.

    ``row_hook`` is called once per acquisition for the two PMT-thermistor cells
    stored beside every row -- in practice
    :meth:`~calibration_module.measure.pmt_temp.PmtTemperature.sample`, whose
    cached reading costs a GPIB round trip only once a minute.  The dark is read
    before the first hook call, so it is not stamped.
    """
    from slm_module.encoding import encode_to_pattern

    zeros = np.zeros(layout.n_channels)
    slm_width, slm_height = slm.get_slm_info()
    total = len(drive) + 1 if total is None else total
    tgt_slot = cfg.slot(tgt_index)
    ref_slot = cfg.slot(cfg.ref_index)

    def display(x_t: float, w_t: float, x_r: float, w_r: float) -> None:
        x_vals = zeros.copy()
        w_vals = zeros.copy()
        x_vals[tgt_slot], w_vals[tgt_slot] = x_t, w_t
        x_vals[ref_slot], w_vals[ref_slot] = x_r, w_r
        slm.display_array(
            encode_to_pattern(x_vals, w_vals, layout, slm_width, slm_height,
                              col_ratio=col_ratio)
        )
        if acq.settle_s:
            time.sleep(acq.settle_s)

    def check_stop() -> None:
        if stop_event is not None and stop_event.is_set():
            raise PhaseV2Aborted("step-7 sweep stopped by request")

    def emit(offset: int, d: Drive, mean_v: float, std_v: float,
             range_v: float, single: bool) -> None:
        if progress_callback is None:
            return
        progress_callback(PhaseV2Progress(
            step=step0 + offset + 1, total=total,
            tgt_index=tgt_index, ref_index=cfg.ref_index,
            x_t=d[0], w_t=d[1], x_r=d[2], w_r=d[3],
            mean_v=mean_v, std_v=std_v, range_v=range_v, single=single,
            duration_s=acq.t_single_s if single else acq.t_both_s,
        ))

    check_stop()
    display(0.0, 0.0, 0.0, 0.0)                      # all-off dark, once
    dark_v, dark_std, dark_range = read_point_autorange(
        monitor, single=True, acq=acq, timeout=read_timeout, log=log
    )
    emit(0, (0.0, 0.0, 0.0, 0.0), dark_v, dark_std, dark_range, True)

    rows: list[tuple[float, float, float, float, float, float]] = []
    cells: list[tuple[str, str]] = []
    for offset, (x_t, w_t, x_r, w_r) in enumerate(drive, start=1):
        check_stop()
        display(x_t, w_t, x_r, w_r)
        mean_v, std_v, range_v = read_point_autorange(
            monitor, single=False, acq=acq, timeout=read_timeout, log=log
        )
        rows.append((x_t, w_t, x_r, w_r, mean_v, std_v))
        if row_hook is not None:
            s = row_hook()
            cells.append((s.get("pmt_ohm_Ohm", ""), s.get("pmt_c", "")))
        emit(offset, (x_t, w_t, x_r, w_r), mean_v, std_v, range_v, False)

    def col(i: int) -> np.ndarray:
        return np.array([r[i] for r in rows], dtype=float)

    return PhaseResult(
        tgt_index=int(tgt_index), ref_index=int(cfg.ref_index),
        trial=np.zeros(len(rows), dtype=int),
        x_t=col(0), w_t=col(1), x_r=col(2), w_r=col(3),
        voltage_mean_v=col(4), voltage_std_v=col(5),
        dark_v=np.full(len(rows), float(dark_v)),
        n_trials=1,
        pmt_ohm=np.array([c[0] for c in cells], dtype=object) if cells else None,
        pmt_c=np.array([c[1] for c in cells], dtype=object) if cells else None,
    )
