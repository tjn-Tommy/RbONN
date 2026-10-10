"""Step 6 v2 -- the per-pair TPA efficiency estimator, driver-free.

In:  arrays of commanded levels and measured volts -- live from
     :mod:`calibration_module.measure.pair_v2`, or from a saved measurement CSV
     through :func:`load_meas_csv`, which re-runs the whole estimator with the
     bench powered down.
Out: a :class:`PairV2Fit` per pair (``eta`` and its error, the background
     parameters, the product check), written as the run's ``step6`` section by
     :func:`save_step6_section`.

No SLM, no DAQ, no paths but this step's own CSV/JSON formats.  The runnable
script is ``steps/calib_step6_v2.py``; what the grid measures, level by level,
is on :class:`PairV2Config`.

Estimator
---------
Along the cross line x = 1, subtract the w-only background::

    D(w) = Y(1, w) - Bhat(w) = eta^2 w + (a_x + q_x)

``a_w``, ``q_w`` and ``d`` cancel identically, and ``a_x``/``q_x`` survive only
as their sum, so eta is the slope of a two-parameter straight line and needs no
assumption about the shape of the single-beam background.

``Bhat(w) = a_w w + d`` is FITTED, not measured: the w-only block sits at
w in {0.5, 1} and the cross block at w in {0.2, 0.45, 0.7, 0.9}, so no
point-by-point partner exists.  Every D therefore shares the background
parameters and the D points are CORRELATED, so the background covariance is
propagated into a full covariance for D and the slope is a GLS solution against
it.

The q columns are off by default (:attr:`PairV2Config.fit_q`): with two drive
levels per arm plus dark they saturate the background block.  Dropped,
``a_x x + a_w w + d`` has 2 dof and the residual is printed with every fit.

There is no separate fit window -- the window IS the ramp the sweep drove, so a
re-fit needs nothing but the CSV.  The fitted w range is reported, the cross
line not being exactly straight.

Verification
------------
One check, printed with every fit and feeding nothing.  The model says ``Y``
depends on the drives only through ``x*w``, so two levels sharing a product but
splitting the drive differently must leave the same TPA residue.  The groups are
found in the data (:func:`product_groups`), so a recorded CSV carries its own
check.  A split means the pair has NO single eta, which would invalidate the
one-number-per-pair model steps 7 and 8 are built on; the slope fit cannot catch
it, a drive-split tilt being nearly linear in w along x = 1.

``verify_enabled=False`` gives the bare grid, and the check then reports itself
as not measured rather than silently passing.

Uncertainties
-------------
Every level is repeated with the SLM rewritten in between, so the repeat scatter
measures ENCODING repeatability and not just detector jitter::

    sigma = hypot(max(rep_std, trace_std) / sqrt(n_repeats), floor)

``trace_std`` is a floor only.  ``floor`` is the sweep's own systematic floor --
the median trace std of its dark acquisitions, pooled over every pair
(:func:`sweep_floor`) -- so it is a voltage and follows the detection gain, and
it goes on AFTER the ``/sqrt(n)``.  A sweep with no dark level cannot be fitted:
there is nothing to measure the floor from and no constant stands in.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .outliers import (
    BURST_K_REPEAT,
    BURST_MAX_V,
    flag_burst_repeats,
    flag_over_range,
)
from .sigma import FloorSource, floor_std, measured_floor

__all__ = [
    "PairV2Config",
    "DEFAULT_CONFIG",
    "VERIFY_X",
    "VERIFY_W",
    "Level",
    "PairV2Fit",
    "average_levels",
    "sweep_floor",
    "step6_floor",
    "burst_rows",
    "block_of",
    "design_bg",
    "grad_bg_w",
    "fit_background",
    "fit_difference",
    "fit_pair",
    "product_groups",
    "verify_product",
    "report",
    "report_verification",
    "make_plot",
    "save_plot",
    "write_meas_csv",
    "meas_csv_meta",
    "load_meas_csv",
    "save_step6_section",
]


# ======================================================================
# configuration
# ======================================================================

#: The estimator's levels: ``(x, w, n_repeats)``.  28 acquisitions.  Every
#: level is repeated (n >= 2) with an SLM rewrite between repeats, so the
#: repeat scatter measures encoding repeatability rather than detector jitter.
DEFAULT_GRID: tuple[tuple[float, float, int], ...] = (
    (0.0, 0.00, 2),   # dark    -> d ; also the sigma floor (sweep_floor)
    (0.5, 0.00, 2),   # x-only  -> a_x, q_x
    (1.0, 0.00, 2),   # x-only  -> a_x, q_x
    (0.0, 0.50, 2),   # w-only  -> a_w, q_w
    (0.0, 1.00, 2),   # w-only  -> a_w, q_w
    (1.0, 0.20, 4),   # cross   -> eta^2
    (1.0, 0.45, 4),   # cross
    (1.0, 0.70, 4),   # cross
    (1.0, 0.90, 6),   # cross   (highest leverage -> most repeats)
)

#: Where the product-check level sits: OFF the ``x = 1`` line, so no ramp point
#: can ever land on it, at the same ``x*w = 0.25`` as ``(0.5, 0.5)``.  ``w`` is
#: rounded to what the CSV writes (``%.6g``) so a reloaded row matches exactly.
VERIFY_X = 0.95
VERIFY_W = round(0.25 / VERIFY_X, 6)            # 0.263158

#: Extra levels measured purely to test the model, never to fit eta; 6 more
#: acquisitions, ~+1 min/pair.  Three repeats each, below which there is no
#: usable repeat scatter to quote the check's pull against.
DEFAULT_VERIFY_GRID: tuple[tuple[float, float, int], ...] = (
    (VERIFY_X, VERIFY_W, 3),
    (0.5, 0.50, 3),
)


@dataclass(frozen=True)
class PairV2Config:
    """The sweep's specification -- what to drive, and the two fit switches.

    The grid is what a SWEEP drives; a re-fit reads the levels off the rows
    instead.  Carried on every :class:`PairV2Fit` for ``params_bg`` and the
    pair-label convention.

    Only the FIT's parameters live here: acquisition timing is on
    :class:`~calibration_module.measure.pair_v2.PairV2Acq` and the sigma floor
    in :mod:`calibration_module.fit.sigma`.
    """

    #: Levels the sweep drives for the estimator.  See :data:`DEFAULT_GRID`.
    grid: tuple[tuple[float, float, int], ...] = DEFAULT_GRID
    #: Model-check levels.  See :data:`DEFAULT_VERIFY_GRID`.
    verify_grid: tuple[tuple[float, float, int], ...] = DEFAULT_VERIFY_GRID
    #: False -> the bare 28-acquisition grid, no product check.
    verify_enabled: bool = True
    #: Keep the quadratic saturation columns ``q_x x^2`` and ``q_w w^2``.  Off
    #: by default: two drive levels per arm plus dark saturates the background
    #: block, so one measured curve splits between ``a`` and ``q`` any way at
    #: all and noise routinely lands on a negative one of them, which a
    #: single-beam response cannot be.  Dropped, the block has 2 dof and its
    #: residual becomes visible instead of being absorbed into a q.  If those
    #: pulls go large the w-only arm is genuinely curved: add w levels, so a and
    #: q are separately MEASURED, rather than returning to the arbitrary split.
    fit_q: bool = False
    #: Pairs are labelled from here and that label is what every CSV / JSON
    #: records; the Step-3 layout arrays are always 0-based, so :meth:`slot`
    #: converts.  0 restores the old 0-based convention.
    pair_index_base: int = 1

    @classmethod
    def from_ramp(cls, sweep_min: float, sweep_max: float, points: int,
                  **kw) -> "PairV2Config":
        """Rebuild the grid with an evenly spaced cross-line ramp.

        Only the cross line is replaced, the part eta is the slope of; the
        single-beam block stays the five levels :func:`fit_background` needs and
        the verification block is defined by its product.  Four repeats per
        level, six on the last.  The fit then uses exactly these levels.

        A DIFFERENT grid from :data:`DEFAULT_GRID`, whose cross levels are not
        evenly spaced.  A re-fit of an existing CSV sees only the CSV's levels.
        """
        lo, hi, n = float(sweep_min), float(sweep_max), int(points)
        if n < 2:
            raise ValueError(f"need >= 2 ramp points, got {n}")
        if not (0.0 < lo < hi <= 1.0):
            raise ValueError(
                f"ramp must satisfy 0 < min < max <= 1, got {lo} .. {hi}"
            )
        ramp = np.round(np.linspace(lo, hi, n), 6)
        cross = tuple((1.0, float(w), 6 if i == n - 1 else 4)
                      for i, w in enumerate(ramp))
        background = tuple(lv for lv in DEFAULT_GRID if block_of(lv[0], lv[1]) != "cross")
        return cls(grid=background + cross, **kw)

    # -- derived ---------------------------------------------------------
    @property
    def params_bg(self) -> tuple[str, ...]:
        """Background parameters, in the order the design columns are built."""
        return ("a_x", "q_x", "a_w", "q_w", "d") if self.fit_q else ("a_x", "a_w", "d")

    @property
    def x_side(self) -> tuple[str, ...]:
        """The x-side terms -- what the difference's intercept measures.

        The cross line sits at ``x = 1``, so ``D(w) = eta^2 w + (a_x + q_x)``,
        or ``+ a_x`` alone without the q columns.
        """
        return tuple(n for n in self.params_bg if n in ("a_x", "q_x"))

    @property
    def x_side_label(self) -> str:
        return " + ".join(self.x_side)

    @property
    def bhat_label(self) -> str:
        return "a_w w + q_w w^2 + d" if self.fit_q else "a_w w + d"

    def slot(self, pair: int) -> int:
        """Pair label -> its 0-based slot in the Step-3 layout / drive arrays."""
        return pair - self.pair_index_base

    def full_grid(self) -> tuple[tuple[float, float, int], ...]:
        """Everything the sweep drives: the estimator's levels + the check."""
        return self.grid + self.verify_grid if self.verify_enabled else self.grid


#: The configuration every validated run to date used.
DEFAULT_CONFIG = PairV2Config()


_CSV_HEADER = [
    "trial", "pair_index", "x", "w", "product",
    "voltage_mean_v", "voltage_std_v", "std_ratio",
    # The PMT thermistor, blank when the sweep ran without it.  Trailing and
    # read back by name, so v1's loader -- which the v2 CSV promises to stay
    # compatible with -- is unaffected by their presence or absence.
    "pmt_ohm_Ohm", "pmt_c",
]

_BLOCK_ORDER = {"dark": 0, "x-only": 1, "w-only": 2, "cross": 3}


# ======================================================================
# levels  (repeat averaging)
# ======================================================================
def block_of(x: float, w: float) -> str:
    """Which block of the grid a level belongs to."""
    if x == 0.0 and w == 0.0:
        return "dark"
    if w == 0.0:
        return "x-only"
    if x == 0.0:
        return "w-only"
    return "cross"



@dataclass
class Level:
    """One commanded ``(x, w)`` setting, averaged over its repeats.

    ``rep_std`` is the scatter of the repeat means (encoding repeatability plus
    drift plus detector noise); ``trace_std`` is the mean within-acquisition
    spread (detector noise alone).  Comparing the two is the point of repeating
    every level.  ``sigma`` is what the fits weight by.
    """

    x: float
    w: float
    means: np.ndarray          # one entry per repeat, volts
    trace_stds: np.ndarray     # one entry per repeat, volts
    #: This sweep's sigma floor (:func:`sweep_floor`), carried on the level so a
    #: fit, its report and its JSON all read the floor they were actually
    #: weighted by.  No default: a level without a measured floor is not a level.
    floor: float
    #: How that floor was arrived at -- a :class:`FloorSource` value.
    floor_source: str

    @property
    def n(self) -> int:
        return int(self.means.size)

    @property
    def block(self) -> str:
        return block_of(self.x, self.w)

    @property
    def mean(self) -> float:
        return float(np.mean(self.means))

    @property
    def rep_std(self) -> float:
        """Sample std across repeats (nan for a single acquisition)."""
        if self.n < 2:
            return float("nan")
        return float(np.std(self.means, ddof=1))

    @property
    def trace_std(self) -> float:
        return float(np.mean(self.trace_stds))

    @property
    def sigma(self) -> float:
        """Uncertainty of this level's mean: trace spread, then the systematic.

        The ``/sqrt(n)`` is legitimate -- independent acquisitions with a panel
        rewrite between them -- but applies only to what averaging reduces, so
        :attr:`floor` goes on AFTER it: no level reads as better known than the
        floor however many repeats it got.
        """
        rep = self.rep_std
        spread = max(rep if np.isfinite(rep) else 0.0, self.trace_std)
        sigma = spread / np.sqrt(self.n)
        sigma = sigma if sigma > 0.0 else 1e-12   # never an infinite weight
        return float(floor_std(sigma, self.floor))

    def is_verification(self) -> bool:
        """A cross-block level OFF the ``x = 1`` line: measured for the product
        check only, never on the slope (see :func:`product_groups`)."""
        return self.block == "cross" and self.x != 1.0

    def on_cross_line(self) -> bool:
        """On the ``x = 1`` line the difference estimator is derived for.

        ``D(w) = Y(1,w) - Bhat(w)`` assumes ``x = 1``, so a cross level at any
        other x must never reach the slope fit.  Every level that IS on it is
        fitted: the window is whatever was swept.
        """
        return self.block == "cross" and self.x == 1.0


def burst_rows(rows, *, k: float = BURST_K_REPEAT,
               rail_v: float = BURST_MAX_V) -> np.ndarray:
    """Boolean mask of ``rows`` that are bursts, not measurements.

    How: each row is judged against the other repeats of its OWN ``(x, w)`` cell
    (:func:`~calibration_module.fit.outliers.flag_burst_repeats`), not against
    the sweep, whose trace stds vary ~100x from dark to full scale.

    Never empties a cell: if every repeat of one is flagged, all are kept.  The
    absolute ceiling is the exception
    (:func:`~calibration_module.fit.outliers.flag_over_range`) -- past
    :data:`~calibration_module.fit.outliers.BURST_MAX_V` the detector is
    railed, so the cell is dropped and ``fit_pair`` says so.
    """
    rows = list(rows)
    if not rows:
        return np.zeros(0, dtype=bool)
    cell_of = [(round(float(r[1]), 6), round(float(r[2]), 6)) for r in rows]
    mask = flag_burst_repeats(
        np.array([float(r[4]) for r in rows], dtype=float), cell_of, k=k)
    cells: dict[tuple[float, float], list[int]] = {}
    for i, c in enumerate(cell_of):
        cells.setdefault(c, []).append(i)
    for idx in cells.values():
        if all(mask[i] for i in idx):
            for i in idx:
                mask[i] = False
    railed = flag_over_range([float(r[3]) for r in rows], limit=rail_v)
    return mask | railed


def sweep_floor(rows_by_pair) -> tuple[float, str]:
    """This step-6 sweep's sigma floor, from its own all-off acquisitions.

    Out: ``(volts, source)`` -- the median trace std of the ``(0, 0)`` rows,
    pooled across every pair, the floor being a property of the detector.

    Rows are selected by COMMANDED drive, never by voltage, so a sweep sitting
    on a stray-light pedestal still reports the noise its points carry.  A file
    with no ``(0, 0)`` rows raises: no constant stands in.
    """
    means: list[float] = []
    stds: list[float] = []
    for rows in (rows_by_pair.values() if hasattr(rows_by_pair, "values")
                 else rows_by_pair):
        for row in rows:
            _rep, x, w, mean_v, std_v = (r for r in row[:5])
            if float(x) == 0.0 and float(w) == 0.0:
                means.append(float(mean_v))
                stds.append(float(std_v))
    if not means:
        raise ValueError(
            "no dark level (x = w = 0) in this sweep -- the sigma floor cannot "
            "be measured, and there is no constant to fall back on"
        )
    # quiet_frac=1.0: the dark level IS the pre-selected quiet set; min_points=1
    # because it was selected by commanded drive, not by voltage, so even one
    # pair's two repeats are the detector's own all-off noise.
    floor, source = measured_floor(means, stds, quiet_frac=1.0, min_points=1)
    if source is not FloorSource.MEASURED:
        raise ValueError(
            f"dark level has no usable trace std ({len(means)} acquisitions) -- "
            "the sigma floor cannot be measured"
        )
    return float(floor), source.value


def step6_floor(run) -> tuple[float, str]:
    """A run's step-6 sigma floor, for the steps that inherit it (step 7).

    Out: ``(volts, INHERITED)`` -- the section's own ``sigma_floor_v``, or, for
    an older step 6 that recorded none, measured from the CSV it names the way
    :func:`sweep_floor` would have.  A run with neither raises.
    """
    from calibration_module.calib_file import CalibFileError, load_calib

    try:
        calib = load_calib(run)
        section = calib.section(6)
    except CalibFileError as exc:
        raise ValueError(f"needs the run's step 6 for its sigma floor: {exc}") from exc
    value = section.get("sigma_floor_v")
    if value is None:
        try:
            csv = calib.csv_path(6)
        except CalibFileError as exc:
            raise ValueError(
                f"{Path(run).name}: its step 6 records no sigma floor and names no "
                f"CSV to measure one from ({exc}) -- re-fit step 6 first"
            ) from exc
        if not csv.is_file():
            raise ValueError(
                f"{Path(run).name}: its step 6 records no sigma floor and its CSV "
                f"{csv.name} is not beside it -- re-fit step 6 first"
            )
        value, _measured = sweep_floor(load_meas_csv(csv))
    return float(value), FloorSource.INHERITED.value


def average_levels(rows, *, drop_bursts: bool = True,
                   floor: tuple[float, str] | None = None,
                   rail_v: float = BURST_MAX_V) -> list[Level]:
    """Group raw ``(rep, x, w, mean_v, std_v)`` rows into repeat-averaged levels.

    Rows may carry the two PMT-thermistor cells after ``std_v``; only the first
    five fields are read.  Bursts (:func:`burst_rows`) go first,
    :attr:`Level.mean` being a plain mean over repeats.

    ``floor`` is :func:`sweep_floor`'s ``(volts, source)``, measured over the
    WHOLE sweep and so passed by the caller.  Omitted, it is measured from this
    pair's own dark repeats, and raises for rows with no dark level.
    """
    rows = list(rows)
    if drop_bursts and rows:
        mask = burst_rows(rows, rail_v=rail_v)
        rows = [r for r, bad in zip(rows, mask) if not bad]
    if not rows:
        return []
    grouped: dict[tuple[float, float], list[tuple[float, float]]] = {}
    for _rep, x, w, mean_v, std_v in (r[:5] for r in rows):
        grouped.setdefault((round(float(x), 6), round(float(w), 6)), []).append(
            (float(mean_v), float(std_v))
        )
    f_v, f_src = floor if floor is not None else sweep_floor({0: rows})
    levels = [
        Level(
            x=x, w=w,
            means=np.array([m for m, _ in vals], dtype=float),
            trace_stds=np.array([s for _, s in vals], dtype=float),
            floor=float(f_v), floor_source=str(f_src),
        )
        for (x, w), vals in grouped.items()
    ]
    levels.sort(key=lambda L: (_BLOCK_ORDER[L.block], L.x, L.w))
    return levels


# ======================================================================
# the fit
# ======================================================================

def design_bg(x, w, cfg: "PairV2Config") -> np.ndarray:
    """Background design matrix, one column per :attr:`PairV2Config.params_bg` entry.

    ``[x, x^2, w, w^2, 1]`` with :attr:`PairV2Config.fit_q`, ``[x, w, 1]`` without.  The
    order always follows ``cfg.params_bg`` so the covariance rows line up with the
    parameter names.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    w = np.atleast_1d(np.asarray(w, dtype=float))
    cols = {"a_x": x, "q_x": x**2, "a_w": w, "q_w": w**2, "d": np.ones_like(x)}
    return np.column_stack([cols[n] for n in cfg.params_bg])


def grad_bg_w(w, cfg: "PairV2Config") -> np.ndarray:
    """d/dparams of the w-only background ``Bhat(w)`` (:attr:`PairV2Config.bhat_label`).

    Same parameter order as :attr:`PairV2Config.params_bg`; the x-side columns are zero --
    that is exactly why ``a_x``/``q_x`` cancel out of the difference.
    """
    w = np.atleast_1d(np.asarray(w, dtype=float))
    z = np.zeros_like(w)
    cols = {"a_x": z, "q_x": z, "a_w": w, "q_w": w**2, "d": np.ones_like(w)}
    return np.column_stack([cols[n] for n in cfg.params_bg])


@dataclass
class PairV2Fit:
    """Everything the v2 estimator produces for one pair.

    Carries its own :class:`PairV2Config` and levels, so the report, the plot
    and the JSON all read what was fitted off the fit itself.
    """

    index: int
    cfg: PairV2Config
    levels: list[Level]
    # -- background block
    bg: dict[str, tuple[float, float]]      # name -> (value, err)
    bg_cov: np.ndarray
    # -- difference fit (the eta estimator)
    fit_w: np.ndarray                       # cross levels used
    fit_d: np.ndarray                       # measured D(w)
    fit_sigma: np.ndarray                   # sqrt of the diagonal of cov_d
    fit_pred: np.ndarray
    b: float                                # eta^2, the slope
    b_err: float
    beta0: float                            # intercept = a_x + q_x
    beta0_err: float
    eta: float
    eta_err: float
    r2: float
    # -- the check
    checks: dict = field(default_factory=dict)   # see verify_product
    wl_x_nm: float = float("nan")
    wl_w_nm: float = float("nan")
    nominal_wl_nm: float = float("nan")

    @property
    def fit_w_range(self) -> tuple[float, float]:
        """The w range the slope was fitted over: the cross line as measured."""
        return float(np.min(self.fit_w)), float(np.max(self.fit_w))

    @property
    def residuals(self) -> np.ndarray:
        return self.fit_d - self.fit_pred

    @property
    def pulls(self) -> np.ndarray:
        return self.residuals / self.fit_sigma

    def bg_residuals(self) -> tuple[list["Level"], np.ndarray, np.ndarray]:
        """Background-block levels with their fit residuals and pulls.

        Empty arrays when the block is saturated (:attr:`PairV2Config.fit_q`):
        a zero-dof fit passes through its points, so the residual says nothing.
        With the q columns dropped there are 2 dof, and these pulls are the
        evidence that a straight single-beam background is good enough.
        """
        cfg = self.cfg
        used = [L for L in self.levels if L.block != "cross"]
        if len(used) <= len(cfg.params_bg):
            return used, np.zeros(0), np.zeros(0)
        p = np.array([self.bg[n][0] for n in cfg.params_bg])
        resid = (np.array([L.mean for L in used])
                 - design_bg([L.x for L in used], [L.w for L in used], cfg) @ p)
        return used, resid, resid / np.array([L.sigma for L in used])

    def background(self, x, w) -> np.ndarray:
        """Full single-beam background at ``(x, w)`` -- for plots and step 7/8."""
        p = {k: v[0] for k, v in self.bg.items()}
        x = np.asarray(x, dtype=float)
        w = np.asarray(w, dtype=float)
        return (p["a_x"] * x + p.get("q_x", 0.0) * x**2
                + p["a_w"] * w + p.get("q_w", 0.0) * w**2 + p["d"])

    def background_w(self, w) -> np.ndarray:
        """The w-only background ``Bhat(w)`` that the difference subtracts."""
        p = {k: v[0] for k, v in self.bg.items()}
        w = np.asarray(w, dtype=float)
        return p["a_w"] * w + p.get("q_w", 0.0) * w**2 + p["d"]


def fit_background(
    levels: list[Level], cfg: "PairV2Config",
) -> tuple[dict[str, tuple[float, float]], np.ndarray]:
    """WLS the single-beam block -> ``(params, covariance)``.

    Uses the dark, x-only and w-only levels: five settings for the
    :attr:`PairV2Config.params_bg` parameters.  With
    :attr:`PairV2Config.fit_q` that is five for five, saturated, so the a/q
    split is interpolation; without it, three parameters and 2 dof, whose
    residual is a check (:meth:`PairV2Fit.bg_residuals`).  The covariance comes
    entirely from the level sigmas.
    """
    used = [L for L in levels if L.block != "cross"]
    if len(used) < len(cfg.params_bg):
        raise ValueError(
            f"background block has {len(used)} levels, need >= {len(cfg.params_bg)}"
        )
    a = design_bg([L.x for L in used], [L.w for L in used], cfg)
    y = np.array([L.mean for L in used], dtype=float)
    sigma = np.array([L.sigma for L in used], dtype=float)

    aw = a / sigma[:, None]
    yw = y / sigma
    params, *_ = np.linalg.lstsq(aw, yw, rcond=None)
    cov = np.linalg.pinv(aw.T @ aw)
    errs = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    return {n: (float(params[i]), float(errs[i])) for i, n in enumerate(cfg.params_bg)}, cov


def fit_difference(
    levels: list[Level],
    bg: dict[str, tuple[float, float]],
    bg_cov: np.ndarray,
    cfg: "PairV2Config",
) -> dict:
    """GLS ``D(w) = eta^2*w + beta0`` over every cross level at ``x = 1``.

    ``D`` is formed against the FITTED background, so its points share those
    parameters and are correlated::

        Cov(D) = diag(sigma_Y^2) + G Cov(bg) G^T

    with ``G`` the gradient of ``Bhat(w)``.  The slope is solved against that
    full matrix.
    """
    used = [L for L in levels if L.on_cross_line()]
    if len(used) < 2:
        raise ValueError(f"need >= 2 cross levels at x = 1, got {len(used)}")

    w = np.array([L.w for L in used], dtype=float)
    y = np.array([L.mean for L in used], dtype=float)
    y_sig = np.array([L.sigma for L in used], dtype=float)

    g = grad_bg_w(w, cfg)
    d = y - (g @ np.array([bg[n][0] for n in cfg.params_bg]))
    cov_d = np.diag(y_sig**2) + g @ bg_cov @ g.T

    a = np.column_stack([w, np.ones_like(w)])
    cov_inv = np.linalg.pinv(cov_d)
    m = np.linalg.pinv(a.T @ cov_inv @ a)
    beta = m @ (a.T @ cov_inv @ d)
    pred = a @ beta

    resid = d - pred
    chi2 = float(resid @ cov_inv @ resid)
    one = np.ones_like(d)
    d_bar = float((one @ cov_inv @ d) / (one @ cov_inv @ one))
    total = float((d - d_bar) @ cov_inv @ (d - d_bar))
    r2 = 1.0 - chi2 / total if total > 0 else float("nan")

    return {
        "w": w, "d": d, "sigma": np.sqrt(np.clip(np.diag(cov_d), 0.0, None)),
        "pred": pred, "beta": beta, "cov": m, "r2": r2, "used": used,
    }


def _level_at(levels: list[Level], x: float, w: float) -> Level | None:
    for L in levels:
        if abs(L.x - x) < 1e-9 and abs(L.w - w) < 1e-9:
            return L
    return None


def fit_pair(
    index: int, levels: list[Level], cfg: "PairV2Config" | None = None,
) -> PairV2Fit:
    """Background block, then the difference slope; plus the product check.

    ``cfg`` supplies only ``params_bg`` (the q switch) and the label convention.
    Which levels are fitted is read off ``levels`` -- a recorded CSV is fitted
    exactly as it was swept, whatever the current sweep settings are.
    """
    cfg = DEFAULT_CONFIG if cfg is None else cfg
    bg, bg_cov = fit_background(levels, cfg)
    df = fit_difference(levels, bg, bg_cov, cfg)

    b, beta0 = float(df["beta"][0]), float(df["beta"][1])
    b_err = float(np.sqrt(max(df["cov"][0, 0], 0.0)))
    beta0_err = float(np.sqrt(max(df["cov"][1, 1], 0.0)))
    if b > 0:
        eta = float(np.sqrt(b))
        eta_err = b_err / (2.0 * eta)        # sigma_eta = sigma_b / 2 eta
    else:
        eta, eta_err = float("nan"), float("nan")

    fit = PairV2Fit(
        index=index, cfg=cfg, levels=levels, bg=bg, bg_cov=bg_cov,
        fit_w=df["w"], fit_d=df["d"], fit_sigma=df["sigma"], fit_pred=df["pred"],
        b=b, b_err=b_err, beta0=beta0, beta0_err=beta0_err,
        eta=eta, eta_err=eta_err, r2=float(df["r2"]),
    )
    fit.checks = {"product": verify_product(fit)}
    return fit


# ======================================================================
# verification  (a check on the model, not an input to it)
# ======================================================================

def product_groups(levels: list[Level], *, decimals: int = 3) -> list[list[Level]]:
    """Cross-block levels sharing a product ``x*w`` at different ``(x, w)``.

    Found in the data rather than declared, so a recorded CSV carries its own
    product check: ``(VERIFY_X, VERIFY_W)`` and ``(0.5, 0.5)`` share 0.25.  Ramp
    levels all sit at ``x = 1`` with distinct ``w``, so they never form a group.
    Each group is ordered hardest x side first.
    """
    by_product: dict[float, list[Level]] = {}
    for L in levels:
        if L.block == "cross":
            by_product.setdefault(round(L.x * L.w, decimals), []).append(L)
    return [sorted(g, key=lambda L: -L.x)
            for g in by_product.values() if len(g) >= 2]


def verify_product(fit: "PairV2Fit") -> list[dict]:
    """``Y`` must depend on the drives only through the product ``x*w``.

    How: within a :func:`product_groups` group every member shares ``x*w`` but
    splits the drive differently (``(0.95, 0.263)`` hard on x, ``(0.5, 0.5)``
    even).  Subtract the full single-beam background and the TPA residue left is
    ``eta^2*(x*w)``, identical across the group.  Both residues come from the
    same fitted background, so the difference propagates
    ``(g_a - g_b) C (g_a - g_b)^T`` rather than adding in quadrature.

    A split means the pair has no single eta at all.
    """
    cfg = fit.cfg
    out: list[dict] = []
    p = np.array([fit.bg[n][0] for n in cfg.params_bg])
    for group in product_groups(fit.levels):
        pts = []
        for level in group:
            g = design_bg([level.x], [level.w], cfg)   # full background, cfg.params_bg order
            tpa = level.mean - float((g @ p)[0])
            pts.append({
                "x": level.x, "w": level.w, "n": level.n, "tpa": tpa,
                "tpa_err": float(np.sqrt(level.sigma**2
                                         + float((g @ fit.bg_cov @ g.T)[0, 0]))),
                "_g": g[0], "_sigma": level.sigma,
            })
        product = float(np.mean([L.x * L.w for L in group]))
        rec = {"product": product, "expected": fit.b * product,
               "points": pts, "pairs": []}
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                a, b = pts[i], pts[j]
                dg = a["_g"] - b["_g"]
                var = (a["_sigma"] ** 2 + b["_sigma"] ** 2
                       + float(dg @ fit.bg_cov @ dg))
                err = float(np.sqrt(max(var, 0.0)))
                diff = a["tpa"] - b["tpa"]
                mid = 0.5 * (a["tpa"] + b["tpa"])
                rec["pairs"].append({
                    "a": (a["x"], a["w"]), "b": (b["x"], b["w"]),
                    "diff": diff, "err": err,
                    "pull": diff / err if err else float("nan"),
                    "frac": diff / mid if mid else float("nan"),
                })
        out.append(rec)
    return out


# ======================================================================
# reporting
# ======================================================================

def _sigma_ratio(value: float, err: float) -> float:
    return abs(value) / err if err else float("nan")


def report(fit: PairV2Fit) -> None:
    """Print the level table, both fit stages and the product check."""
    cfg = fit.cfg
    lo, hi = fit.fit_w_range
    print("Estimator: difference   "
          f"D(w) = Y(1,w) - [{cfg.bhat_label}] = eta^2 w + ({cfg.x_side_label})")
    print(f"Fit window: w in [{lo:.2f}, {hi:.2f}]  ({fit.fit_w.size} levels, "
          "the cross line as swept)")

    floor_v = fit.levels[0].floor
    floor_src = fit.levels[0].floor_source
    print(f"\n  Level means  (sigma = hypot(max(rep_std, trace_std)/sqrt(n), "
          f"{floor_v*1e3:.3f} mV floor [{floor_src}])):")
    print("    block     x     w   n   mean(mV)  rep_std(mV)  trace_std(mV)  sigma(mV)")
    for L in fit.levels:
        tag = "verify" if L.is_verification() else L.block
        rep = f"{L.rep_std*1e3:11.4f}" if np.isfinite(L.rep_std) else f"{'--':>11}"
        print(f"    {tag:<7} {L.x:5.2f} {L.w:5.3f} {L.n:3d} {L.mean*1e3:10.4f} "
              f"{rep}  {L.trace_std*1e3:13.4f} {L.sigma*1e3:10.4f}")
    print("    (verify: off the x = 1 line -- product check only, not on the slope)")

    repeated = [L for L in fit.levels if np.isfinite(L.rep_std)]
    if not repeated:
        print("    no level was repeated -- sigma is the trace spread alone, so these "
              "error bars carry no encoding repeatability")
    else:
        dominated = [L for L in repeated if L.rep_std > L.trace_std]
        print(f"    repeat scatter exceeds trace noise on {len(dominated)}/{len(repeated)} "
              f"repeated levels -- encoding repeatability "
              f"{'dominates' if len(dominated) > len(repeated) / 2 else 'is not dominant'}")

    n_bg = sum(1 for L in fit.levels if L.block != "cross")
    shape = "saturated" if n_bg == len(cfg.params_bg) else f"{n_bg - len(cfg.params_bg)} dof"
    print(f"\n  Background block ({n_bg} levels, {len(cfg.params_bg)} parameters -> {shape}; "
          "used by steps 7/8, not by eta):")
    for name in cfg.params_bg:
        v, e = fit.bg[name]
        scale, unit = (1e3, "mV") if name == "d" else (1.0, "")
        print(f"    {name:<3} = {v*scale:.4e} +/- {e*scale:.3e} {unit}".rstrip())
    if not cfg.fit_q:
        print("    (q_x/q_w dropped -- cfg.fit_q = False; with two drive levels per arm")
        print("     they are not separable from a_x/a_w, which is what produced "
              "negative background terms)")

    # With the q columns gone the block has dof, so its residual is a real
    # statement about whether the straight single-beam background holds.  A
    # saturated block returns nothing here: it fits its own points exactly.
    used_bg, bg_resid, bg_pull = fit.bg_residuals()
    if bg_resid.size:
        print("      block     x     w   resid(mV)   pull")
        for L, r, pl in zip(used_bg, bg_resid, bg_pull):
            print(f"      {L.block:<7} {L.x:5.2f} {L.w:5.2f} "
                  f"{r*1e3:10.4f} {pl:6.2f}")
        rms_bg = float(np.sqrt(np.mean(bg_resid**2)))
        mx = float(np.max(np.abs(bg_pull)))
        verdict = "OK" if mx < 3.0 else "** CHECK: arm is curved, add w levels **"
        print(f"      residual RMS = {rms_bg*1e3:.4f} mV   "
              f"max |pull| = {mx:.2f}   {verdict}")

    print("\n  Difference fit (GLS, background covariance propagated):")
    print(f"    eta^2 = {fit.b:.4e} +/- {fit.b_err:.3e}")
    print(f"    eta   = {fit.eta:.4e} +/- {fit.eta_err:.3e}   "
          f"({_sigma_ratio(fit.eta, fit.eta_err):.1f} sigma)   over w in [{lo:.2f}, {hi:.2f}]")
    print(f"    beta0 = {fit.beta0*1e3:.4f} +/- {fit.beta0_err*1e3:.4f} mV   "
          f"[= {cfg.x_side_label}]")
    print(f"    R^2   = {fit.r2:.4f}")

    print("      w    D_meas(mV)   D_fit(mV)  resid(mV)   pull")
    for i in np.argsort(fit.fit_w):
        print(f"    {fit.fit_w[i]:5.2f} {fit.fit_d[i]*1e3:11.4f} "
              f"{fit.fit_pred[i]*1e3:11.4f} {fit.residuals[i]*1e3:10.4f} "
              f"{fit.pulls[i]:6.2f}")
    rms = float(np.sqrt(np.mean(fit.residuals**2)))
    print(f"    residual RMS = {rms*1e3:.4f} mV")

    report_verification(fit)


def report_verification(fit: PairV2Fit) -> None:
    """The product-only dependence check.

    It does not feed the fit -- it says whether the model eta is defined within
    still holds on this pair.
    """
    prods = (fit.checks or {}).get("product") or []
    print("\n  Verification (a check on the model, not an input to it):")
    if not prods:
        print("    product-only dependence: NOT MEASURED (no two cross levels share "
              "an x*w -- set cfg.verify_enabled = True and re-run)")
        return
    print("    product-only dependence   Y depends on (x, w) only via x*w")
    for rec in prods:
        print(f"       x*w = {rec['product']:.3f}   "
              f"model expects eta^2*(x*w) = {rec['expected']*1e3:.4f} mV")
        for pt in rec["points"]:
            print(f"         ({pt['x']:.2f}, {pt['w']:.3f}) n={pt['n']}  "
                  f"TPA residue = {pt['tpa']*1e3:9.4f} +/- {pt['tpa_err']*1e3:.4f} mV")
        for pr in rec["pairs"]:
            verdict = "OK" if abs(pr["pull"]) < 3.0 else "** CHECK **"
            print(f"         {pr['a']} - {pr['b']} = {pr['diff']*1e3:+.4f} +/- "
                  f"{pr['err']*1e3:.4f} mV  ({pr['frac']*100:+.2f}%)  "
                  f"pull = {pr['pull']:+.2f}   {verdict}")


def make_plot(fig, fit: PairV2Fit) -> None:
    """Six panels drawn into ``fig`` -- a Matplotlib Figure the caller owns, so
    the GUI canvas and the headless PNG (:func:`save_plot`) share one
    implementation.

    Top-left, the single-beam block with its background curves and the shared
    dark point.  Top-right, THE ESTIMATOR: ``D(w)`` with the GLS line and the
    fitted range shaded.  Bottom-left, pulls against the +/-1 sigma band.
    Bottom-right, repeatability -- each level's repeat scatter beside its trace
    noise, a bar above the diagonal meaning the SLM rewrite moves the reading
    more than the detector does.  The other two are the product check and its
    summary text.
    """
    cfg = fit.cfg
    lo, hi = fit.fit_w_range
    fig.clear()
    axes = fig.subplots(2, 3)
    (ax1, ax2, ax5), (ax3, ax4, ax6) = axes

    # --- background block ---------------------------------------------------
    for label, sel, level_of, curve, color in (
        ("x-only (w=0)", lambda L: L.block in ("dark", "x-only"), lambda L: L.x,
         lambda r: fit.background(r, 0.0), "tab:blue"),
        ("w-only (x=0)", lambda L: L.block in ("dark", "w-only"), lambda L: L.w,
         lambda r: fit.background(0.0, r), "tab:orange"),
    ):
        pts = [L for L in fit.levels if sel(L)]
        r = np.array([level_of(L) for L in pts])
        y = np.array([L.mean for L in pts])
        e = np.array([L.sigma for L in pts])
        r_fine = np.linspace(0.0, float(r.max()) if r.size else 1.0, 200)
        ax1.plot(r_fine, curve(r_fine) * 1e3, "-", color=color, lw=1.4, zorder=2)
        order = np.argsort(r)
        ax1.errorbar(r[order], y[order] * 1e3, yerr=e[order] * 1e3, fmt="o",
                     color=color, ms=6, capsize=2, lw=0.8, mec="k", mew=0.3,
                     zorder=3, label=label)
    p = {k: v[0] for k, v in fit.bg.items()}
    txt = "\n".join(f"{n} = {p[n]*(1e3 if n == 'd' else 1.0):.3g}"
                     f"{' mV' if n == 'd' else ''}" for n in cfg.params_bg)
    _, bg_resid, bg_pull = fit.bg_residuals()
    if bg_resid.size:
        txt += (f"\nresid RMS = {float(np.sqrt(np.mean(bg_resid**2)))*1e3:.4f} mV"
                f"\nmax |pull| = {float(np.max(np.abs(bg_pull))):.2f}")
    ax1.text(0.03, 0.97, txt,
             transform=ax1.transAxes, va="top", fontsize=8,
             bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    ax1.set_xlabel("per-side level r")
    ax1.set_ylabel("Voltage (mV)")
    n_bg = sum(1 for L in fit.levels if L.block != "cross")
    dof = n_bg - len(cfg.params_bg)
    ax1.set_title(f"Single-beam block ({'saturated' if dof == 0 else f'{dof} dof'} fit)"
                  " -- for steps 7/8, not for $\\eta$")
    ax1.legend(loc="lower right", fontsize=8)

    # --- the difference line: this is the estimator --------------------------
    ax2.axvspan(lo, hi, color="tab:green", alpha=0.10, zorder=0,
                label=f"fitted range [{lo:g}, {hi:g}]")
    w_fine = np.linspace(0.0, 1.05, 200)
    ax2.plot(w_fine, (fit.b * w_fine + fit.beta0) * 1e3, "-", color="tab:green",
             lw=1.5, zorder=2, label="GLS  $D = \\eta^2 w + \\beta_0$")
    order = np.argsort(fit.fit_w)
    ax2.errorbar(fit.fit_w[order], fit.fit_d[order] * 1e3,
                 yerr=fit.fit_sigma[order] * 1e3, fmt="o", color="tab:green",
                 ms=6, capsize=2, lw=0.8, mec="k", mew=0.3, zorder=4,
                 label="$D(w)$ fitted")
    ax2.text(0.03, 0.97,
             f"eta = {fit.eta:.4g} $\\pm$ {fit.eta_err:.2g}  "
             f"({_sigma_ratio(fit.eta, fit.eta_err):.0f}$\\sigma$)\n"
             f"$\\eta^2$ = {fit.b:.4g}\n"
             f"$\\beta_0$ = {fit.beta0*1e3:.3f} mV\n"
             f"R$^2$ = {fit.r2:.4f}",
             transform=ax2.transAxes, va="top", fontsize=8,
             bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    ax2.set_xlabel("w   (x = 1 on this line)")
    ax2.set_ylabel("$D(w) = Y(1,w) - \\hat{B}(w)$   (mV)")
    ax2.set_title("Difference estimator -- $\\eta$ is this slope")
    ax2.legend(loc="lower right", fontsize=8)

    # --- pulls ---------------------------------------------------------------
    ax3.axhspan(-1, 1, color="tab:blue", alpha=0.12, label="$\\pm1\\sigma$")
    ax3.axhline(0, color="gray", ls="--", lw=1, zorder=1)
    ax3.scatter(fit.fit_w, fit.pulls, c="tab:green", s=55, edgecolor="k", lw=0.4,
                zorder=3, label="fitted $D(w)$")
    rms = float(np.sqrt(np.mean(fit.residuals**2)))
    ax3.text(0.03, 0.97, f"RMS = {rms*1e3:.4f} mV", transform=ax3.transAxes,
             va="top", fontsize=8,
             bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    span = float(np.max(np.abs(fit.pulls))) * 1.25 if fit.pulls.size else 1.5
    ax3.set_ylim(-max(1.5, span), max(1.5, span))    # never clip a pull
    ax3.set_xlabel("w")
    ax3.set_ylabel("Pull = residual / $\\sigma$")
    ax3.set_title("Difference-fit residuals")
    ax3.legend(loc="lower right", fontsize=8)

    # --- repeatability -------------------------------------------------------
    # Ratio rather than a scatter: the only question is whether rewriting the
    # panel moves a level more than the detector noise does, and a bar chart
    # answers that at a glance without overlapping point labels.
    repeated = [L for L in fit.levels if np.isfinite(L.rep_std) and L.trace_std > 0]
    if repeated:
        ratios = [L.rep_std / L.trace_std for L in repeated]
        labels = [f"({L.x:g}, {L.w:g})  n={L.n}" for L in repeated]
        colors = [{"dark": "k", "x-only": "tab:blue", "w-only": "tab:orange",
                   "cross": "tab:green"}[L.block] for L in repeated]
        pos = np.arange(len(repeated))
        ax4.barh(pos, ratios, color=colors, alpha=0.75, edgecolor="k", lw=0.4)
        ax4.axvline(1.0, color="0.4", ls="--", lw=1.2,
                    label="rewrite = detector noise")
        ax4.set_yticks(pos)
        ax4.set_yticklabels(labels, fontsize=7)
        ax4.invert_yaxis()
        ax4.set_xlabel("repeat std / trace std")
        ax4.set_title("Repeatability: past the line, encoding dominates")
        ax4.legend(loc="lower right", fontsize=8)
    else:
        ax4.text(0.5, 0.5, "no level was repeated\n(sigma is the trace spread alone)",
                 transform=ax4.transAxes, ha="center", va="center", fontsize=10,
                 color="0.4")
        ax4.set_xticks([])
        ax4.set_yticks([])
        ax4.set_title("Repeatability")

    # --- product check: equal x*w must give equal TPA residue ---------------
    prods = (fit.checks or {}).get("product") or []
    if prods:
        for k, rec in enumerate(prods):
            labels = [f"({p['x']:g}, {p['w']:g})" for p in rec["points"]]
            vals = np.array([p["tpa"] for p in rec["points"]]) * 1e3
            errs = np.array([p["tpa_err"] for p in rec["points"]]) * 1e3
            pos = np.arange(len(vals)) + k * (len(vals) + 1)
            ax5.errorbar(pos, vals, yerr=errs, fmt="o", ms=8, capsize=4, lw=1.2,
                         color="tab:purple", mec="k", mew=0.4, zorder=3)
            ax5.axhline(rec["expected"] * 1e3, color="tab:green", ls="--", lw=1.2,
                        zorder=1,
                        label="$\\eta^2(x\\,w)$ from the slope fit" if k == 0 else None)
            ax5.set_xticks(pos)
            ax5.set_xticklabels([f"{lb}\n$x\\,w$={rec['product']:g}" for lb in labels],
                                fontsize=8)
            for n_pr, pr in enumerate(rec["pairs"]):
                ax5.text(0.03, 0.97 - 0.07 * n_pr,
                         f"split = {pr['diff']*1e3:+.4f} $\\pm$ {pr['err']*1e3:.4f} mV "
                         f"({pr['frac']*100:+.2f}%, pull {pr['pull']:+.2f})",
                         transform=ax5.transAxes, ha="left", va="top", fontsize=8,
                         bbox=dict(boxstyle="round", fc="white", alpha=0.85))
        n_slots = sum(len(r["points"]) + 1 for r in prods) - 1
        ax5.set_xlim(-0.6, n_slots - 0.4)      # keep ticks off the axis edges
        ax5.set_ylabel("TPA residue $Y - \\hat{B}(x,w)$   (mV)")
        ax5.set_title("Product check -- same $x\\,w$, different split")
        ax5.legend(loc="upper right", fontsize=8)
    else:
        ax5.text(0.5, 0.5,
                 "product check not measured\n\nneeds the verification levels\n"
                 "(cfg.verify_enabled = True)",
                 transform=ax5.transAxes, ha="center", va="center", fontsize=10,
                 color="0.4")
        ax5.set_xticks([])
        ax5.set_yticks([])
        ax5.set_title("Product check")

    # --- verification summary text ------------------------------------------
    ax6.axis("off")
    lines = ["Verification", "", "product-only dependence"]
    if prods:
        for rec in prods:
            lines.append(f"     $x\\,w$ = {rec['product']:.3g}")
            for pr in rec["pairs"]:
                lines.append(
                    f"     {pr['a']} vs {pr['b']}: {pr['frac']*100:+.2f}%   "
                    f"pull = {pr['pull']:+.2f}   "
                    f"{'OK' if abs(pr['pull']) < 3 else 'CHECK'}")
    else:
        lines.append("     not measured")
    lines += ["", "The check does not feed the fit -- it says whether",
              "the model $\\eta$ is defined within still holds."]
    ax6.text(0.02, 0.98, "\n".join(lines), transform=ax6.transAxes, va="top",
             ha="left", fontsize=9, family="monospace",
             bbox=dict(boxstyle="round", fc="0.97", ec="0.7"))

    fig.suptitle(f"Step 6 v2 -- pair {fit.index}   "
                 f"$\\eta$ = {fit.eta:.4g} $\\pm$ {fit.eta_err:.2g}", fontsize=12)
    fig.tight_layout()


def save_plot(fit: PairV2Fit, path: str | Path) -> Path:
    """Render :func:`make_plot` to a PNG with no display attached.

    Forces the Agg backend before the pyplot import, so this stays callable
    from a headless run; the GUI never comes through here -- it owns its own
    canvas figure and calls :func:`make_plot` directly.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(19.5, 9.5))
    try:
        make_plot(fig, fit)
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
    finally:
        plt.close(fig)
    return out


# ======================================================================
# persistence
# ======================================================================
def write_meas_csv(rows_by_pair: dict[int, list], path: str | Path,
                   *, meta: Mapping[str, object] | None = None) -> Path:
    """Raw rows, v1's column layout (``trial`` carries the repeat index), so the
    same file can be re-fit with v1's joint 6-parameter estimator.

    A row is ``(rep, x, w, mean_v, std_v)``, optionally plus the two
    PMT-thermistor cells; without them those columns are blank, so one sweep's
    file has one shape either way.  ``meta`` is appended as trailing
    ``# key,value`` lines.
    """
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_CSV_HEADER)
        for index in sorted(rows_by_pair):
            for row in rows_by_pair[index]:
                rep, x, w, mean_v, std_v = row[:5]
                ohm, celsius = (row[5], row[6]) if len(row) >= 7 else ("", "")
                ratio = abs(std_v / mean_v) if mean_v else float("inf")
                writer.writerow([int(rep), index, f"{x:.6g}", f"{w:.6g}",
                                 f"{x*w:.6g}", f"{mean_v:.9g}", f"{std_v:.9g}",
                                 f"{ratio:.6g}", ohm, celsius])
        for key, value in (meta or {}).items():
            writer.writerow([f"# {key}", value])
    return out


def load_meas_csv(path: str | Path) -> dict[int, list]:
    """Read a v2 measurement CSV back into ``{pair_index: rows}``.

    Rows come back as 7-tuples carrying the PMT-thermistor cells as written --
    strings, blank where the sweep had no 2400 -- so a load/save round trip does
    not quietly drop the temperature.  The fit reads only the first five
    (:func:`average_levels`).
    """
    rows_by_pair: dict[int, list] = {}
    with open(Path(path), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(l for l in f if not l.startswith("#")):
            index = int(float(row["pair_index"]))
            rows_by_pair.setdefault(index, []).append((
                int(float(row.get("trial", 0))),
                float(row["x"]), float(row["w"]),
                float(row["voltage_mean_v"]), float(row["voltage_std_v"]),
                row.get("pmt_ohm_Ohm") or "", row.get("pmt_c") or "",
            ))
    return rows_by_pair


def meas_csv_meta(path: str | Path) -> dict[str, str]:
    """The trailing ``# key,value`` lines of a measurement CSV, as a dict."""
    meta: dict[str, str] = {}
    with open(Path(path), newline="", encoding="utf-8") as f:
        for raw in f:
            if raw.startswith("#"):
                parts = raw.lstrip("#").strip().split(",")
                if len(parts) >= 2:
                    meta[parts[0].strip()] = ",".join(parts[1:]).strip()
    return meta


def _checks_dict(fit: PairV2Fit) -> dict:
    """JSON-safe view of the verification results (drops the working arrays)."""
    checks = fit.checks or {}
    return {
        "product": [
            {
                "product": rec["product"],
                "expected": rec["expected"],
                "points": [{k: v for k, v in pt.items() if not k.startswith("_")}
                           for pt in rec["points"]],
                "pairs": rec["pairs"],
            }
            for rec in (checks.get("product") or [])
        ],
    }


def _background_dict(fit: PairV2Fit) -> dict:
    """Background-block shape and residual -- the evidence for the cfg.fit_q choice."""
    cfg = fit.cfg
    used, resid, pull = fit.bg_residuals()
    return {
        "fit_q": cfg.fit_q,
        "params": list(cfg.params_bg),
        "n_levels": len(used),
        "dof": len(used) - len(cfg.params_bg),
        "resid_rms_v": float(np.sqrt(np.mean(resid**2))) if resid.size else None,
        "max_abs_pull": float(np.max(np.abs(pull))) if pull.size else None,
        "levels": [{"x": L.x, "w": L.w, "resid_v": float(r), "pull": float(pl)}
                   for L, r, pl in zip(used, resid, pull)],
    }


def save_step6_section(fits: list[PairV2Fit], run: str | Path,
                       *, center_wl: float = 0.0,
                       dropped_bursts: dict[int, list[dict]] | None = None,
                       csv: str | Path | None = None,
                       log=print):
    """Write every v2 pair fit as the run's ``step6`` section.

    ``channels[].fit.{eta, eta_err, params.{a_x,q_x,a_w,q_w,d}.value}`` is the
    contract :meth:`calibration_module.fit.phase.PairModel.from_json_channel`
    parses, so step 7 consumes this unchanged; the v2-specific numbers ride
    alongside.  ``grid`` is the levels AS MEASURED, what the CSV holds.

    ``csv`` is the sweep this came from, stored relative to the run directory.
    ``dropped_bursts`` maps a pair index to the burst acquisitions left out of
    its fit.  Step 3 is not copied in, being a section of the same file.
    """
    from calibration_module.calib_file import load_calib, write_section

    if not fits:
        raise ValueError("no pair fits to save")
    # A step 6 with no layout above it is not interpretable; say so before the
    # file is touched rather than leaving a section nothing can read.
    load_calib(run, missing_ok=True).require(3)
    cfg = fits[0].cfg
    first = fits[0]

    def _fit_dict(fit: PairV2Fit) -> dict:
        params = {n: {"value": v, "err": e} for n, (v, e) in fit.bg.items()}
        # PairModel reads a_x/q_x/a_w/q_w/d unconditionally, so a dropped q is
        # written as an exact zero rather than left out of the file.
        for n in ("q_x", "q_w"):
            params.setdefault(n, {"value": 0.0, "err": 0.0})
        params["b"] = {"value": fit.b, "err": fit.b_err}
        params["beta0"] = {"value": fit.beta0, "err": fit.beta0_err}
        return {
            "eta": fit.eta,
            "eta_err": fit.eta_err,
            "params": params,
            "r2": fit.r2,
            "estimator": "difference",
            "fit_w_range": list(fit.fit_w_range),
            "checks": _checks_dict(fit),
            "background": _background_dict(fit),
            "levels": [
                {"x": L.x, "w": L.w, "n": L.n, "mean_v": L.mean,
                 "rep_std_v": None if not np.isfinite(L.rep_std) else L.rep_std,
                 "trace_std_v": L.trace_std, "sigma_v": L.sigma}
                for L in fit.levels
            ],
        }

    step6 = {
        "version": 2,
        "estimator": "difference",
        "fit_q": cfg.fit_q,
        "params_bg": list(cfg.params_bg),
        "fit_w_range": list(first.fit_w_range),
        # As measured: every distinct level in the file with its repeat count.
        "grid": [{"x": L.x, "w": L.w, "n": L.n} for L in first.levels],
        "center_wl": center_wl,
        # The floor these etas were weighted by -- this sweep's own dark level.
        # Recorded because it is a voltage: an eta_err is only comparable with
        # another night's once you know both runs' floors (see fit.sigma).
        "sigma_floor_v": float(first.levels[0].floor),
        "sigma_floor_source": str(first.levels[0].floor_source),
        "dropped_bursts": {str(k): v for k, v in (dropped_bursts or {}).items()},
        "channels": [
            {
                "index": fit.index,
                "wl_x_nm": fit.wl_x_nm,
                "wl_w_nm": fit.wl_w_nm,
                "nominal_wl_nm": fit.nominal_wl_nm,
                "fit": _fit_dict(fit),
            }
            for fit in fits
        ],
    }
    return write_section(run, 6, step6, csv=csv, log=log)

