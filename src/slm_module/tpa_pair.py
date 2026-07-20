"""Per-channel-pair TPA efficiency (eta) calibration by a 2-D level grid.

This supersedes the diagonal-only sweep in :mod:`scope_tpa`. Instead of driving
a pair along ``x = w = sqrt(u)`` and fitting ``a*u^2`` against a *separately
measured* background, this sweeps the two sides of a pair **independently** over
a grid (with the zero axes included) and fits the full response

    Y = eta^2*(x*w) + a_x*x + q_x*x^2 + a_w*w + q_w*w^2 + d
      └ 2-photon CROSS term ┘└ x single-beam ┘└ w single-beam ┘ └ dark ┘

directly.  x, w are per-channel commanded INTENSITIES in [0, 1]; eta multiplies
the field amplitude, so the cross term is ``eta^2 * (x*w)``.  The fit is LINEAR
in ``b := eta^2, a_x, q_x, a_w, q_w, d`` and solved by weighted least squares;
``eta = sqrt(b)`` is recovered afterwards.  Because the swept grid includes the
``x=0`` and ``w=0`` axes (which carry ``x*w = 0``), the single-channel terms are
pinned without eta contamination and eta is cleanly identifiable -- no separate
background measurement is needed (the dark offset ``d`` and the single-beam
slopes are fit in-model).

The measurement is instrument-agnostic: it drives an SLM (``get_slm_info`` +
``display_array``) and reads whatever *monitor* object exposes the
``ScopeController`` / ``DAQController`` shape (``monitor_cycle`` returning a
``MonitorSample`` and caching the raw waveform on ``last_values``).  Each grid
point is read ONCE over a long fixed window -- T_single (``x == 0 or w == 0``,
the weak single-beam and dark points) or T_both (both beams on) on the DAQ --
and weighted by the instrument-reported SEM; the per-parameter errors are
Birge-scaled when chi2/dof > 1 so the reported eta uncertainty is honest.

Raw rows are persisted as a CSV (one row per grid point; the ``trial`` column
is kept so multi-trial CSVs from older runs still load) matching the
``tests/tpa_pair_calibration_test.py`` layout, so a run can be reloaded and
re-fit offline.
"""
from __future__ import annotations

import csv
import json
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Column order of the design matrix / fitted-parameter vector.
PARAMS: tuple[str, ...] = ("b", "a_x", "q_x", "a_w", "q_w", "d")


class TPAPairAborted(Exception):
    """Raised when a stop_event interrupts a pair-grid sweep."""


@dataclass
class TPAPairProgress:
    step: int
    total: int
    message: str
    pair_index: int | None = None
    eta: float | None = None          # filled in once a pair's grid is fit


ProgressCallback = Callable[["TPAPairProgress"], None]


@dataclass
class PairFit:
    """Weighted-least-squares fit of one pair's grid to the TPA model."""

    eta: float
    eta_err: float
    params: dict[str, tuple[float, float]]   # name -> (value, Birge-scaled err)
    chi2_red: float
    dof: int
    birge: float
    r2: float
    # averaged-cell arrays the fit ran on (kept for plotting)
    x: np.ndarray = field(repr=False)
    w: np.ndarray = field(repr=False)
    y: np.ndarray = field(repr=False)
    sem: np.ndarray = field(repr=False)
    y_pred: np.ndarray = field(repr=False)
    residuals: np.ndarray = field(repr=False)


@dataclass
class ChannelPairGrid:
    """One channel pair's raw grid rows (all trials) plus its fit."""

    index: int
    wl_x_nm: float
    wl_w_nm: float
    nominal_wl_nm: float
    x_center_x: int
    x_center_w: int
    # raw rows, one entry per (trial, grid point); kept for save + re-fit
    trial: np.ndarray = field(repr=False)
    x: np.ndarray = field(repr=False)
    w: np.ndarray = field(repr=False)
    voltage_mean_v: np.ndarray = field(repr=False)
    voltage_std_v: np.ndarray = field(repr=False)   # raw low-passed trace std (diagnostic)
    voltage_sem_v: np.ndarray = field(repr=False)    # SEM of the mean -> the fit weight
    fit: PairFit | None = None


@dataclass
class TPAPairResult:
    sweep: np.ndarray                # per-side commanded levels swept (incl. 0)
    n_trials: int
    channels: list[ChannelPairGrid]
    center_wl: float = 0.0
    csv_path: str | None = None

    def pair_by_index(self) -> dict[int, ChannelPairGrid]:
        return {c.index: c for c in self.channels}

    def eta_by_index(self) -> dict[int, float]:
        return {c.index: (c.fit.eta if c.fit else float("nan")) for c in self.channels}


# ======================================================================
# fit  (linear least squares in b = eta^2, a_x, q_x, a_w, q_w, d)
# ======================================================================

def design_matrix(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Columns match PARAMS: [x*w, x, x^2, w, w^2, 1]."""
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    return np.column_stack([x * w, x, x**2, w, w**2, np.ones_like(x)])


def average_cells(
    trial: np.ndarray,
    x: np.ndarray,
    w: np.ndarray,
    y: np.ndarray,
    sem: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Average repeated trials per (x, w) cell -> x, w, y, sem arrays.

    ``sem`` (returned) is the standard error of the mean across trials for a cell
    (std/sqrt(n)) when it was measured more than once.  A cell measured only
    once has no across-trial spread, so it falls back to the recorded per-point
    SEM ``sem`` (the DAQ's reported standard error of the mean stored alongside
    each row) when available -- that is the real per-point uncertainty and keeps
    the weighted fit meaningful even with ``n_trials == 1`` (otherwise every cell
    would be floored to a bogus 1.0 V, flattening the fit).  A cell with neither
    repeats nor a recorded SEM is floored to the median positive sem so the
    weighted fit never divides by zero/NaN.
    """
    ycells: dict[tuple[float, float], list[float]] = defaultdict(list)
    scells: dict[tuple[float, float], list[float]] = defaultdict(list)
    sem_arr = np.asarray(sem, dtype=float) if sem is not None else None
    for idx, (cx, cw, cy) in enumerate(zip(np.asarray(x), np.asarray(w), np.asarray(y))):
        key = (float(cx), float(cw))
        ycells[key].append(float(cy))
        if sem_arr is not None:
            scells[key].append(float(sem_arr[idx]))

    cx_out, cw_out, cy_out, csem_out = [], [], [], []
    for (cx, cw), vals in sorted(ycells.items()):
        arr = np.asarray(vals, dtype=float)
        cx_out.append(cx)
        cw_out.append(cw)
        cy_out.append(arr.mean())
        if arr.size > 1:
            csem_out.append(arr.std(ddof=1) / np.sqrt(arr.size))   # across-trial spread
        else:
            rec = np.asarray(scells[(cx, cw)], dtype=float)         # recorded per-point SEM
            rec = rec[np.isfinite(rec) & (rec > 0)]
            csem_out.append(float(rec.mean()) if rec.size else np.nan)

    xs = np.asarray(cx_out)
    ws = np.asarray(cw_out)
    ys = np.asarray(cy_out)
    sem = np.asarray(csem_out)

    # Floor missing/degenerate errors so weighting never divides by zero/NaN.
    finite = sem[np.isfinite(sem) & (sem > 0)]
    floor = float(np.median(finite)) if finite.size else 1.0
    sem = np.where(np.isfinite(sem) & (sem > 0), sem, floor)
    return xs, ws, ys, sem


def fit_cells(
    x: np.ndarray, w: np.ndarray, y: np.ndarray, sem: np.ndarray,
    *, drop_q: bool = False,
) -> PairFit:
    """Weighted least-squares fit of averaged cells to the TPA model.

    Errors are Birge-scaled by ``sqrt(chi2/dof)`` when chi2/dof > 1 so unmodeled
    reproducibility scatter inflates the reported uncertainties.  ``eta`` is
    recovered as ``sqrt(b)`` with propagated error ``b_err/(2*sqrt(b))``.

    ``drop_q=True`` drops the ``q_x``/``q_w`` saturation columns and fits the
    purely linear background ``Y = b*(x*w) + a_x*x + a_w*w + d`` (the a's then
    carry the full single-beam slopes, with no a<->q split).  The q entries stay
    in ``params`` pinned to ``(0.0, 0.0)`` so downstream report/plot/JSON code
    sees the usual keys.
    """
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    y = np.asarray(y, dtype=float)
    sem = np.asarray(sem, dtype=float)

    A = design_matrix(x, w)
    names: tuple[str, ...] = PARAMS
    if drop_q:
        keep = [i for i, n in enumerate(PARAMS) if n not in ("q_x", "q_w")]
        A = A[:, keep]
        names = tuple(PARAMS[i] for i in keep)
    Aw = A / sem[:, None]
    coeffs, *_ = np.linalg.lstsq(Aw, y / sem, rcond=None)
    cov = np.linalg.inv(Aw.T @ Aw)

    y_pred = A @ coeffs
    residuals = y - y_pred
    dof = max(len(y) - len(coeffs), 1)
    chi2_red = float(np.sum((residuals / sem) ** 2) / dof)
    birge = max(1.0, np.sqrt(chi2_red))
    errs = np.sqrt(np.diag(cov)) * birge

    params = {name: (float(v), float(e)) for name, v, e in zip(names, coeffs, errs)}
    if drop_q:
        params["q_x"] = (0.0, 0.0)
        params["q_w"] = (0.0, 0.0)

    b, b_err = params["b"]
    if b > 0:
        eta, eta_err = float(np.sqrt(b)), float(b_err / (2.0 * np.sqrt(b)))
    else:
        eta, eta_err = float("nan"), float("nan")

    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return PairFit(
        eta=eta, eta_err=eta_err, params=params,
        chi2_red=chi2_red, dof=dof, birge=birge, r2=r2,
        x=x, w=w, y=y, sem=sem, y_pred=y_pred, residuals=residuals,
    )


def fit_grid(grid: ChannelPairGrid, *, drop_q: bool = False) -> PairFit:
    """Average a pair's raw trials into cells, fit them, and store the fit."""
    xs, ws, ys, sem = average_cells(
        grid.trial, grid.x, grid.w, grid.voltage_mean_v, grid.voltage_sem_v
    )
    grid.fit = fit_cells(xs, ws, ys, sem, drop_q=drop_q)
    return grid.fit


def recompute_fits(result: TPAPairResult) -> TPAPairResult:
    for grid in result.channels:
        fit_grid(grid)
    return result


# ======================================================================
# measurement  (instrument-agnostic grid sweep)
# ======================================================================

def _read_mean_std(
    monitor, timeout: float = 30.0, single: bool = False
) -> tuple[float, float, float]:
    """One averaged reading, its raw trace std, and the per-point SEM of the mean.

    Returns ``(mean, std, sem)``.  ``std`` is the raw (low-passed) trace spread,
    kept for diagnostics; ``sem`` is the standard error of the mean and is what
    the fit weights by.  The DAQ path reports ``sample.sem`` directly (spread
    divided by the effective independent-sample count ``2 * duration * f_cut``);
    monitors that don't report one (e.g. the scope) fall back to the raw-waveform
    std on ``last_values`` for both.  ``single`` marks a weak point -- at most
    one beam on (``x == 0 or w == 0``, including the all-off dark): the DAQ
    reads it over its longer T_single window (``single_duration``); the scope
    ignores the flag.
    """
    sample = monitor.monitor_cycle(timeout=timeout, single=single)
    if sample is None:
        raise TPAPairAborted("monitor read aborted")
    mean_v = float(sample.value)
    std = getattr(sample, "std", None)
    waveform = getattr(monitor, "last_values", None)
    std_v = (
        float(std) if std is not None and np.isfinite(std)
        else (float(np.std(waveform)) if waveform is not None and np.size(waveform) > 1
              else 0.0)
    )
    sem = getattr(sample, "sem", None)
    if sem is not None and np.isfinite(sem):
        sem_v = float(sem)                                     # SEM of the mean (DAQ)
    else:
        sem_v = std_v                                          # scope: no effective-N -> raw std
    return mean_v, std_v, sem_v


def build_sweep(sweep_min: float, sweep_max: float, n_points: int) -> np.ndarray:
    """Per-side commanded levels: the zero axis prepended to a linear ramp.

    The leading 0 gives the ``x=0`` / ``w=0`` axis points that pin the
    single-channel terms (see module docstring).
    """
    ramp = np.linspace(float(sweep_min), float(sweep_max), int(n_points))
    return np.concatenate(([0.0], ramp))


def build_pair_points(
    sweep_min: float, sweep_max: float, n_points: int
) -> list[tuple[float, float]]:
    """Reduced 1-D calibration curves for a pair (not the full 2-D grid).

    Rather than the ``(n+1) x (n+1)`` outer-product grid, this measures only the
    lines that each fit term needs, ``n_points`` per line plus one shared dark
    point::

        dark    (0, 0)   -- anchors the offset d
        x-only  (r, 0)   -- only x on -> pins a_x, q_x
        w-only  (0, r)   -- only w on -> pins a_w, q_w
        cross   (1, r)   -- x pinned at 1, w swept -> the ONLY points with
                            x*w != 0, so they pin eta once the single-beam
                            terms above are known

    ``r`` runs over ``linspace(sweep_min, sweep_max, n_points)``.  The full
    TPA model stays identifiable because the w-only line sees ``a_w``/``q_w``
    but carries ``x*w = 0``, so it separates the single-beam ``w`` response
    from the ``eta^2*(x*w)`` cross term measured on the ``x=1`` line.  Points
    are de-duplicated so a level shared across lines is measured once.
    """
    ramp = np.linspace(float(sweep_min), float(sweep_max), int(n_points))
    pts: list[tuple[float, float]] = [(0.0, 0.0)]
    pts += [(float(r), 0.0) for r in ramp]   # x-only  -> a_x, q_x
    pts += [(0.0, float(r)) for r in ramp]   # w-only  -> a_w, q_w
    pts += [(1.0, float(r)) for r in ramp]   # cross (x=1) -> eta
    seen: set[tuple[float, float]] = set()
    unique: list[tuple[float, float]] = []
    for p in pts:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def measure_pair_grids(
    monitor,
    slm,
    layout,
    *,
    pair_indices: Sequence[int],
    sweep: Sequence[float],
    points: Sequence[tuple[float, float]] | None = None,
    settle: float = 0.15,
    read_timeout: float = 30.0,
    col_ratio: np.ndarray | None = None,
    stop_event: threading.Event | None = None,
    progress_callback: ProgressCallback | None = None,
) -> TPAPairResult:
    """Sweep each requested pair's (x, w) points once and fit eta for each.

    ``monitor`` must already be configured (the caller runs
    ``configure_monitor``); this only calls ``monitor_cycle``.  Each ``(x, w)``
    point is measured once (all other channels held off) -- the DAQ's long
    fixed windows already average enough per point -- then the pair's data is
    fit to the TPA model.  Points with at most one beam on (``x == 0 or
    w == 0``, including the all-off dark) are read over the DAQ's longer
    T_single window; both-beams points use T_both.  By default the points
    are the full outer-product grid of ``sweep`` x ``sweep``; pass ``points`` to
    measure an explicit list instead (e.g. the reduced 1-D curves from
    :func:`build_pair_points`), in which case ``sweep`` is only recorded as the
    ramp on the result.  ``settle`` seconds are waited after every pattern change
    before reading.  ``col_ratio`` is the per-column encoding shape forwarded to
    :func:`encode_to_pattern` so the calibration is measured with the same
    channel taper that will be deployed (``None`` = flat band).  Raises
    :class:`TPAPairAborted` if ``stop_event`` is set.
    """
    sweep_arr = np.asarray(list(sweep), dtype=float)
    if points is not None:
        grid_pts = [(float(x), float(w)) for x, w in points]
    else:
        grid_pts = [(float(x), float(w)) for x in sweep_arr for w in sweep_arr]
    indices = list(pair_indices)
    n = layout.n_channels
    zeros = np.zeros(n)

    slm_width, slm_height = slm.get_slm_info()

    from .encoding import encode_to_pattern

    def _check_stop() -> None:
        if stop_event is not None and stop_event.is_set():
            raise TPAPairAborted("pair-grid sweep stopped by request")

    # accumulate raw rows per pair across all trials
    rows: dict[int, list[tuple[int, float, float, float, float, float]]] = {
        i: [] for i in indices
    }

    total = max(len(indices) * len(grid_pts), 1)
    step = 0
    for i in indices:
        _check_stop()
        x_ch = layout.x_channels[i]
        w_ch = layout.w_channels[i]
        wl_pair = 0.5 * (x_ch.wavelength_nm + w_ch.wavelength_nm)
        for x_val, w_val in grid_pts:
            _check_stop()
            x_vals = zeros.copy()
            w_vals = zeros.copy()
            x_vals[i] = x_val
            w_vals[i] = w_val
            pattern = encode_to_pattern(x_vals, w_vals, layout, slm_width,
                                        slm_height, col_ratio=col_ratio)
            slm.display_array(pattern)
            if settle:
                time.sleep(settle)
            single = x_val == 0.0 or w_val == 0.0       # at most one beam on
            mean_v, std_v, sem_v = _read_mean_std(
                monitor, read_timeout, single=single
            )
            rows[i].append((0, x_val, w_val, mean_v, std_v, sem_v))
            step += 1
            if progress_callback is not None:
                progress_callback(
                    TPAPairProgress(
                        step=step, total=total,
                        message=(
                            f"pair[{i}] @ {wl_pair:.2f} nm "
                            f"x={x_val:.2f} w={w_val:.2f} "
                            f"-> {mean_v*1000:.4f} mV"
                        ),
                        pair_index=i,
                    )
                )

    channels: list[ChannelPairGrid] = []
    for i in indices:
        x_ch = layout.x_channels[i]
        w_ch = layout.w_channels[i]
        data = rows[i]
        grid = ChannelPairGrid(
            index=i,
            wl_x_nm=float(x_ch.wavelength_nm),
            wl_w_nm=float(w_ch.wavelength_nm),
            nominal_wl_nm=0.5 * (x_ch.wavelength_nm + w_ch.wavelength_nm),
            x_center_x=int(x_ch.x_center),
            x_center_w=int(w_ch.x_center),
            trial=np.array([r[0] for r in data], dtype=int),
            x=np.array([r[1] for r in data], dtype=float),
            w=np.array([r[2] for r in data], dtype=float),
            voltage_mean_v=np.array([r[3] for r in data], dtype=float),
            voltage_std_v=np.array([r[4] for r in data], dtype=float),
            voltage_sem_v=np.array([r[5] for r in data], dtype=float),
        )
        fit_grid(grid)
        channels.append(grid)
        if progress_callback is not None and grid.fit is not None:
            progress_callback(
                TPAPairProgress(
                    step=total, total=total,
                    message=f"pair[{i}] fit: eta = {grid.fit.eta:.4g}",
                    pair_index=i, eta=grid.fit.eta,
                )
            )

    return TPAPairResult(
        sweep=sweep_arr, n_trials=1, channels=channels,
        center_wl=float(getattr(layout, "center_wl", 0.0)),
    )


# ======================================================================
# persistence
# ======================================================================

_CSV_HEADER = [
    "trial", "pair_index", "x", "w", "product",
    "voltage_mean_v", "voltage_std_v", "voltage_sem_v", "sem_ratio",
]


def write_tpa_pair_csv(result: TPAPairResult, path: str | Path) -> str:
    """Raw rows: one line per (trial, pair, grid point).  Round-trips via load.

    ``voltage_std_v`` is the raw low-passed trace spread (diagnostic);
    ``voltage_sem_v`` is the standard error of the mean the fit weights by; and
    ``sem_ratio`` = sem/|mean| is derived per row for at-a-glance measurement
    quality (recomputed on load, so :func:`load_tpa_pair_csv` ignores it).  With
    per-point durations the std and sem are not related by a single fixed
    ``n_eff``, so both are recorded explicitly.
    """
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_CSV_HEADER)
        for grid in result.channels:
            for t, x, w, mean_v, std_v, sem_v in zip(
                grid.trial, grid.x, grid.w, grid.voltage_mean_v,
                grid.voltage_std_v, grid.voltage_sem_v,
            ):
                ratio = abs(sem_v / mean_v) if mean_v else float("inf")
                writer.writerow(
                    [int(t), grid.index, f"{x:.6g}", f"{w:.6g}", f"{x*w:.6g}",
                     f"{mean_v:.9g}", f"{std_v:.9g}", f"{sem_v:.9g}", f"{ratio:.6g}"]
                )
    result.csv_path = str(out)
    return str(out)


def load_tpa_pair_csv(
    path: str | Path,
    *,
    layout=None,
) -> TPAPairResult:
    """Load a raw pair-grid CSV back into a result and re-fit every pair.

    Wavelengths are recovered from ``layout`` when supplied (the CSV carries only
    x/w/voltage), otherwise left as NaN.
    """
    grouped: dict[int, list[tuple[int, float, float, float, float, float]]] = defaultdict(list)
    with open(Path(path), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            idx = int(float(row["pair_index"]))
            std_v = float(row.get("voltage_std_v", "nan") or "nan")
            sem_v = float(row["voltage_sem_v"])  # the fit weight; every CSV records it
            grouped[idx].append(
                (
                    int(float(row.get("trial", 0))),
                    float(row["x"]),
                    float(row["w"]),
                    float(row["voltage_mean_v"]),
                    std_v,
                    sem_v,
                )
            )

    channels: list[ChannelPairGrid] = []
    n_trials = 1
    sweep_vals: set[float] = set()
    for idx in sorted(grouped):
        data = grouped[idx]
        trials = np.array([r[0] for r in data], dtype=int)
        xs = np.array([r[1] for r in data], dtype=float)
        ws = np.array([r[2] for r in data], dtype=float)
        n_trials = max(n_trials, int(trials.max()) + 1 if trials.size else 1)
        sweep_vals.update(xs.tolist())
        if layout is not None and idx < layout.n_channels:
            x_ch = layout.x_channels[idx]
            w_ch = layout.w_channels[idx]
            wl_x, wl_w = float(x_ch.wavelength_nm), float(w_ch.wavelength_nm)
            xc_x, xc_w = int(x_ch.x_center), int(w_ch.x_center)
        else:
            wl_x = wl_w = float("nan")
            xc_x = xc_w = 0
        grid = ChannelPairGrid(
            index=idx, wl_x_nm=wl_x, wl_w_nm=wl_w,
            nominal_wl_nm=0.5 * (wl_x + wl_w),
            x_center_x=xc_x, x_center_w=xc_w,
            trial=trials, x=xs, w=ws,
            voltage_mean_v=np.array([r[3] for r in data], dtype=float),
            voltage_std_v=np.array([r[4] for r in data], dtype=float),
            voltage_sem_v=np.array([r[5] for r in data], dtype=float),
        )
        fit_grid(grid)
        channels.append(grid)

    result = TPAPairResult(
        sweep=np.array(sorted(sweep_vals)), n_trials=n_trials,
        channels=channels,
        center_wl=float(getattr(layout, "center_wl", 0.0)) if layout is not None else 0.0,
        csv_path=str(Path(path).resolve()),
    )
    return result


def save_tpa_pair_json(result: TPAPairResult, path: str | Path) -> str:
    """Human-readable per-pair eta + fitted-parameter summary."""
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    def _fit_dict(fit: PairFit | None) -> dict | None:
        if fit is None:
            return None
        return {
            "eta": fit.eta,
            "eta_err": fit.eta_err,
            "params": {k: {"value": v[0], "err": v[1]} for k, v in fit.params.items()},
            "chi2_red": fit.chi2_red,
            "dof": fit.dof,
            "birge": fit.birge,
            "r2": fit.r2,
        }

    payload = {
        "sweep": result.sweep.tolist(),
        "n_trials": result.n_trials,
        "center_wl": result.center_wl,
        "channels": [
            {
                "index": c.index,
                "wl_x_nm": c.wl_x_nm,
                "wl_w_nm": c.wl_w_nm,
                "nominal_wl_nm": c.nominal_wl_nm,
                "fit": _fit_dict(c.fit),
            }
            for c in result.channels
        ],
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(out)


__all__ = [
    "PARAMS",
    "TPAPairAborted",
    "TPAPairProgress",
    "PairFit",
    "ChannelPairGrid",
    "TPAPairResult",
    "design_matrix",
    "average_cells",
    "fit_cells",
    "fit_grid",
    "recompute_fits",
    "build_sweep",
    "build_pair_points",
    "measure_pair_grids",
    "write_tpa_pair_csv",
    "load_tpa_pair_csv",
    "save_tpa_pair_json",
]
