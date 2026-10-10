"""Step 6 v1 -- per-pair TPA efficiency (eta) from a 2-D level grid.

The joint fit, kept as the cross-check on :mod:`pair_v2`'s difference estimator.

In:  commanded levels and measured volts, live or from a CSV through
     :func:`load_tpa_pair_csv`.
Out: a :class:`PairFit` per pair.

The two sides of a pair are swept INDEPENDENTLY over a grid with the zero axes
included, and the full response is fitted directly::

    Y = eta^2*(x*w) + a_x*x + q_x*x^2 + a_w*w + q_w*w^2 + d
      |-- 2-photon --||-- x single-beam --||-- w single --| |dark|

``x``, ``w`` are per-channel commanded INTENSITIES in [0, 1]; eta multiplies the
field amplitude, hence the square on the cross term.  The fit is LINEAR in
``b := eta^2, a_x, q_x, a_w, q_w, d``, solved by weighted least squares, and
``eta = sqrt(b)`` is recovered afterwards.  Because the grid includes the
``x = 0`` and ``w = 0`` axes, which carry ``x*w = 0``, the single-channel terms
are pinned without eta contamination and no separate background measurement is
needed.

Instrument-agnostic: it drives an SLM (``get_slm_info`` + ``display_array``) and
reads anything with the ``ScopeController`` / ``DAQController`` shape.  Each
point is read ONCE over a long fixed window -- ``T_single`` when either arm is
off, ``T_both`` otherwise -- and weighted by the instrument-reported trace std,
used AS MEASURED and undivided by any effective-N: the low-passed samples are
correlated, so a standard error derived from them would claim an uncertainty far
below the scatter seen between repeats.

Raw rows persist as a CSV, one row per grid point, so a run can be re-fit
offline.  The ``trial`` column is kept so multi-trial CSVs from older runs still
load.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Column order of the design matrix / fitted-parameter vector.
PARAMS: tuple[str, ...] = ("b", "a_x", "q_x", "a_w", "q_w", "d")


@dataclass
class PairFit:
    """Weighted-least-squares fit of one pair's grid to the TPA model."""

    eta: float
    eta_err: float
    params: dict[str, tuple[float, float]]   # name -> (value, err)
    r2: float
    # averaged-cell arrays the fit ran on (kept for plotting)
    x: np.ndarray = field(repr=False)
    w: np.ndarray = field(repr=False)
    y: np.ndarray = field(repr=False)
    std: np.ndarray = field(repr=False)
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
    voltage_std_v: np.ndarray = field(repr=False)   # low-passed trace std -> the fit weight
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
    std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Average repeated trials per (x, w) cell -> x, w, y, std arrays.

    The returned ``std`` is the across-trial standard deviation -- as measured,
    NOT divided by sqrt(n).  A cell measured ONCE has none and falls back to its
    recorded per-point trace spread, which keeps a weighted fit meaningful at
    ``n_trials == 1``; a cell with neither is floored to the median positive std,
    so the fit never divides by zero.
    """
    ycells: dict[tuple[float, float], list[float]] = defaultdict(list)
    scells: dict[tuple[float, float], list[float]] = defaultdict(list)
    std_arr = np.asarray(std, dtype=float) if std is not None else None
    for idx, (cx, cw, cy) in enumerate(zip(np.asarray(x), np.asarray(w), np.asarray(y))):
        key = (float(cx), float(cw))
        ycells[key].append(float(cy))
        if std_arr is not None:
            scells[key].append(float(std_arr[idx]))

    cx_out, cw_out, cy_out, cstd_out = [], [], [], []
    for (cx, cw), vals in sorted(ycells.items()):
        arr = np.asarray(vals, dtype=float)
        cx_out.append(cx)
        cw_out.append(cw)
        cy_out.append(arr.mean())
        if arr.size > 1:
            cstd_out.append(arr.std(ddof=1))                        # across-trial spread
        else:
            rec = np.asarray(scells[(cx, cw)], dtype=float)         # recorded per-point std
            rec = rec[np.isfinite(rec) & (rec > 0)]
            cstd_out.append(float(rec.mean()) if rec.size else np.nan)

    xs = np.asarray(cx_out)
    ws = np.asarray(cw_out)
    ys = np.asarray(cy_out)
    std = np.asarray(cstd_out)

    # Floor missing/degenerate spreads so weighting never divides by zero/NaN.
    finite = std[np.isfinite(std) & (std > 0)]
    floor = float(np.median(finite)) if finite.size else 1.0
    std = np.where(np.isfinite(std) & (std > 0), std, floor)
    return xs, ws, ys, std


def fit_cells(
    x: np.ndarray, w: np.ndarray, y: np.ndarray, std: np.ndarray,
    *, drop_q: bool = False,
) -> PairFit:
    """Weighted least-squares fit of averaged cells to the TPA model.

    Weighted by ``1/std``, errors straight from the weighted covariance -- no
    chi2/dof, no Birge rescaling.  ``eta = sqrt(b)``, with propagated error
    ``b_err/(2 sqrt(b))``.

    ``drop_q=True`` drops the saturation columns and fits
    ``Y = b*(x*w) + a_x*x + a_w*w + d``, the a's then carrying the full
    single-beam slopes with no a<->q split.  The q entries stay in ``params``
    pinned at ``(0.0, 0.0)``, so the report, plot and JSON see the usual keys.
    """
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    y = np.asarray(y, dtype=float)
    std = np.asarray(std, dtype=float)

    A = design_matrix(x, w)
    names: tuple[str, ...] = PARAMS
    if drop_q:
        keep = [i for i, n in enumerate(PARAMS) if n not in ("q_x", "q_w")]
        A = A[:, keep]
        names = tuple(PARAMS[i] for i in keep)
    Aw = A / std[:, None]
    coeffs, *_ = np.linalg.lstsq(Aw, y / std, rcond=None)
    cov = np.linalg.inv(Aw.T @ Aw)

    y_pred = A @ coeffs
    residuals = y - y_pred
    errs = np.sqrt(np.diag(cov))

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
        eta=eta, eta_err=eta_err, params=params, r2=r2,
        x=x, w=w, y=y, std=std, y_pred=y_pred, residuals=residuals,
    )


def fit_grid(grid: ChannelPairGrid, *, drop_q: bool = False) -> PairFit:
    """Average a pair's raw trials into cells, fit them, and store the fit."""
    xs, ws, ys, std = average_cells(
        grid.trial, grid.x, grid.w, grid.voltage_mean_v, grid.voltage_std_v
    )
    grid.fit = fit_cells(xs, ws, ys, std, drop_q=drop_q)
    return grid.fit


def recompute_fits(result: TPAPairResult) -> TPAPairResult:
    for grid in result.channels:
        fit_grid(grid)
    return result


# ======================================================================
# drive builders  (which (x, w) points a sweep visits)
# ======================================================================


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

    Only the lines each fit term needs, ``n_points`` per line plus one shared
    dark point, with ``r`` over ``linspace(sweep_min, sweep_max, n_points)``::

        dark    (0, 0)   anchors the offset d
        x-only  (r, 0)   pins a_x, q_x
        w-only  (0, r)   pins a_w, q_w
        cross   (1, r)   the ONLY points with x*w != 0, so they pin eta once
                         the single-beam terms above are known

    The model stays identifiable because the w-only line sees ``a_w``/``q_w``
    while carrying ``x*w = 0``, which separates the single-beam w response from
    the cross term measured on the ``x = 1`` line.  Points are de-duplicated, so
    a level shared across lines is measured once.
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


# ======================================================================
# persistence
# ======================================================================

_CSV_HEADER = [
    "trial", "pair_index", "x", "w", "product",
    "voltage_mean_v", "voltage_std_v", "std_ratio",
]


def write_tpa_pair_csv(result: TPAPairResult, path: str | Path) -> str:
    """Raw rows: one line per (trial, pair, grid point).  Round-trips via load.

    ``voltage_std_v`` is the low-passed trace spread the fit weights by;
    ``std_ratio`` is derived per row for at-a-glance quality and recomputed on
    load, so :func:`load_tpa_pair_csv` ignores it.
    """
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_CSV_HEADER)
        for grid in result.channels:
            for t, x, w, mean_v, std_v in zip(
                grid.trial, grid.x, grid.w, grid.voltage_mean_v, grid.voltage_std_v,
            ):
                ratio = abs(std_v / mean_v) if mean_v else float("inf")
                writer.writerow(
                    [int(t), grid.index, f"{x:.6g}", f"{w:.6g}", f"{x*w:.6g}",
                     f"{mean_v:.9g}", f"{std_v:.9g}", f"{ratio:.6g}"]
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
    grouped: dict[int, list[tuple[int, float, float, float, float]]] = defaultdict(list)
    with open(Path(path), newline="", encoding="utf-8") as f:
        # v2 CSVs may end in trailing "# key,value" meta lines (the PMT
        # thermistor brackets the sweep there); skipping them is what keeps the
        # promise that either version can re-fit either file.
        for row in csv.DictReader(l for l in f if not l.startswith("#")):
            idx = int(float(row["pair_index"]))
            std_v = float(row["voltage_std_v"])  # the fit weight; every CSV records it
            grouped[idx].append(
                (
                    int(float(row.get("trial", 0))),
                    float(row["x"]),
                    float(row["w"]),
                    float(row["voltage_mean_v"]),
                    std_v,
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
    "write_tpa_pair_csv",
    "load_tpa_pair_csv",
    "save_tpa_pair_json",
]
