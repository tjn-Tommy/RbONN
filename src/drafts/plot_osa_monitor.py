"""Analyse a ``daq_osa_monitor`` run: spectrum, temperature and laser power.

Reads the master CSV written by ``daq_osa_monitor.py`` and every spectrum in
its sibling ``daq_osa_monitor_<stamp>/`` folder, then draws a 3x2 figure:

* spectra overlaid, coloured by elapsed time      * wavelength x time heat map
* wavelength + width vs time                      * OSA power vs time
* temperature / resistance + PD power vs time     * all trends z-scored together

and prints a full statistical analysis: per-channel linear drift with its
significance, detrended scatter, whether the spectrum translated or changed
shape, and the lagged cross-correlations between the channels.

The figure is saved next to the master CSV as ``<stem>_analysis.png``.
Defaults to the newest monitor CSV under ``src/calib_data``::

    python src/drafts/plot_osa_monitor.py
    python src/drafts/plot_osa_monitor.py src/calib_data/daq_osa_monitor_0826_2328.csv --no-show

Two things to keep in mind when reading the output:

* **Temperature comes from the Thorlabs TH10K curve** (10 kOhm NTC, spec
  sheet 4813-S01), so it is good to a few tens of mK *relative*; the absolute
  value still carries the sensor's +-1 C accuracy at 25 C and whatever offset
  exists between the bead and the thing you care about.  Read the changes,
  not the absolute number.
* **The PD power is not the OSA power.**  The photodiode sits on a different
  light path, so its coupling and alignment losses are unknown and its scale
  cannot be compared against the OSA's.  Only the *relative* change over time
  is meaningful, and even that is confounded by any mechanical drift in its
  own path.

Cycles whose OSA sweep failed (blank ``spectrum_csv``) are skipped.
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

sys.path.insert(0, str(Path(__file__).resolve().parent))

from th10k import th10k_celsius  # noqa: E402,F401  (re-exported for callers)

TRAPZ = getattr(np, "trapezoid", None) or np.trapz  # renamed in numpy 2.0

OUT_DIR = Path(__file__).resolve().parents[1] / "calib_data"
MAX_OVERLAY = 48        # overlay at most this many spectra (evenly spaced in time)
SMOOTH_PTS = 25         # moving-average width (points) for peak / width metrics
OVERLAY_CMAP = "viridis"
HEATMAP_CMAP = "magma"

MAX_LAG = 8             # cross-correlation lag range, in cycles (+-2 h at 15 min)


@dataclass
class Run:
    name: str
    elapsed_h: np.ndarray      # (n,)
    timestamps: list[str]      # (n,) ISO strings from the master CSV
    peak_wl_nm: np.ndarray     # (n,) raw argmax as logged by the monitor
    peak_power: np.ndarray     # (n,) uW as logged
    daq_mv: np.ndarray         # (n,) photodiode reading (mV), separate light path
    ohm: np.ndarray            # (n,) Keithley 2400 resistance
    wl_nm: np.ndarray          # (m,) common wavelength grid
    power: np.ndarray          # (n, m) spectra in ``unit``
    unit: str                  # "uW" (linear traces) or "dBm" (log traces)


def _float(text: str) -> float:
    return float(text) if text.strip() else float("nan")


def load_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    """One ``TraceData.to_csv`` file -> (wavelength nm, power, unit)."""
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        data = np.array([[float(a), float(b)] for a, b in reader])
    wl_nm = data[:, 0] * 1e9
    if "dbm" in header[1].lower():
        return wl_nm, data[:, 1], "dBm"
    return wl_nm, data[:, 1] * 1e6, "uW"          # W -> uW


def load_run(master_csv: Path) -> Run:
    spectra_dir = master_csv.with_suffix("")
    with master_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"{master_csv} has no data rows")

    elapsed, stamps, peak_wl, peak_pw, daq, ohm, spectra = [], [], [], [], [], [], []
    grid: np.ndarray | None = None
    unit = ""
    for row in rows:
        name = row["spectrum_csv"].strip()
        if not name:
            print(f"skip {row['timestamp']}: no spectrum (OSA failed that cycle)")
            continue
        path = spectra_dir / name
        if not path.exists():
            print(f"skip {row['timestamp']}: {path} missing")
            continue
        wl, power, unit = load_spectrum(path)
        if grid is None:
            grid = wl
        elif wl.shape != grid.shape or not np.allclose(wl, grid):
            power = np.interp(grid, wl, power)     # different sweep grid -> resample
        elapsed.append(_float(row["elapsed_min"]) / 60.0)
        stamps.append(row["timestamp"])
        peak_wl.append(_float(row["osa_peak_wl_nm"]))
        peak_pw.append(_float(row["osa_peak_uW"]))
        daq.append(_float(row["daq_mean_mV"]))
        ohm.append(_float(row.get("ohm_Ohm", "")))
        spectra.append(power)

    if grid is None:
        sys.exit(f"no spectra found for {master_csv}")
    return Run(
        name=master_csv.stem,
        elapsed_h=np.asarray(elapsed),
        timestamps=stamps,
        peak_wl_nm=np.asarray(peak_wl),
        peak_power=np.asarray(peak_pw),
        daq_mv=np.asarray(daq),
        ohm=np.asarray(ohm),
        wl_nm=grid,
        power=np.vstack(spectra),
        unit=unit,
    )


def linear_power(run: Run) -> np.ndarray:
    """Spectra in linear units (uW) regardless of how they were saved."""
    if run.unit == "dBm":
        return 10.0 ** (run.power / 10.0) * 1e3      # dBm -> mW -> uW
    return run.power


def smooth_rows(lin: np.ndarray, width: int = SMOOTH_PTS) -> np.ndarray:
    width = min(width, lin.shape[1])
    if width < 3:
        return lin
    kernel = np.ones(width) / width
    return np.vstack([np.convolve(row, kernel, mode="same") for row in lin])


def half_max_metrics(wl: np.ndarray, lin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-spectrum (half-max midpoint, FWHM) in nm, from the smoothed trace.

    On a broad envelope this is far steadier than argmax: it uses the two
    steep flanks, where the lineshape actually carries wavelength information,
    instead of the flat noisy top.
    """
    smooth = smooth_rows(lin)
    edge = min(SMOOTH_PTS, lin.shape[1]) // 2       # convolution taper
    mids, widths = [], []
    for row in smooth:
        core = row[edge:len(row) - edge] if edge else row
        wlc = wl[edge:len(row) - edge] if edge else wl
        half = core.max() / 2.0
        above = np.flatnonzero(core >= half)
        if above.size == 0 or above[0] == 0 or above[-1] == core.size - 1:
            mids.append(np.nan)                     # flank falls outside the scan
            widths.append(np.nan)
            continue
        i, j = above[0], above[-1]
        left = np.interp(half, [core[i - 1], core[i]], [wlc[i - 1], wlc[i]])
        right = np.interp(half, [core[j + 1], core[j]], [wlc[j + 1], wlc[j]])
        mids.append((left + right) / 2.0)
        widths.append(right - left)
    return np.asarray(mids), np.asarray(widths)


def linfit(t: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Least-squares line -> (slope, slope stderr, residual std, r)."""
    ok = np.isfinite(t) & np.isfinite(y)
    if ok.sum() < 3:
        return (np.nan,) * 4
    tt, yy = t[ok], y[ok]
    A = np.vstack([tt, np.ones_like(tt)]).T
    coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
    resid = yy - A @ coef
    dof = len(tt) - 2
    s = float(np.sqrt((resid**2).sum() / dof))
    cov = s**2 * np.linalg.inv(A.T @ A)
    r = float(np.corrcoef(tt, yy)[0, 1])
    return float(coef[0]), float(np.sqrt(cov[0, 0])), s, r


def best_lag(a: np.ndarray, b: np.ndarray, dt_h: float,
             max_lag: int = MAX_LAG) -> tuple[float, float, float]:
    """Cross-correlate two series -> (r at zero lag, best r, lag in hours).

    A positive lag means ``b`` follows ``a``.
    """
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok] - np.nanmean(a[ok]), b[ok] - np.nanmean(b[ok])
    if a.size < 5:
        return (np.nan,) * 3
    r0 = float(np.corrcoef(a, b)[0, 1])
    best_r, best_k = r0, 0
    for k in range(-max_lag, max_lag + 1):
        x, y = (a[:a.size - k], b[k:]) if k >= 0 else (a[-k:], b[:b.size + k])
        if x.size < 5:
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        if abs(r) > abs(best_r):
            best_r, best_k = r, k
    return r0, best_r, best_k * dt_h


def detrend(t: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Remove the best-fit line in time, leaving the wiggles."""
    ok = np.isfinite(t) & np.isfinite(v)
    out = np.full(v.shape, np.nan)
    if ok.sum() < 3:
        return out
    A = np.vstack([t[ok], np.ones(ok.sum())]).T
    coef, *_ = np.linalg.lstsq(A, v[ok], rcond=None)
    out[ok] = v[ok] - A @ coef
    return out


def zscore(v: np.ndarray) -> np.ndarray:
    ok = np.isfinite(v)
    if ok.sum() < 2 or np.nanstd(v) == 0:
        return np.full_like(v, np.nan, dtype=float)
    return (v - np.nanmean(v)) / np.nanstd(v)


def analyse(run: Run) -> dict:
    lin = linear_power(run)
    wl, t = run.wl_nm, run.elapsed_h
    centroid = (lin * wl).sum(1) / lin.sum(1)
    mid_nm, fwhm_nm = half_max_metrics(wl, lin)
    integrated = TRAPZ(lin, wl, axis=1)
    temp_c = th10k_celsius(run.ohm)
    return dict(
        t=t, lin=lin, centroid=centroid, mid_nm=mid_nm, fwhm_nm=fwhm_nm,
        integrated=integrated, temp_c=temp_c,
    )


def report(run: Run, a: dict) -> None:
    t = a["t"]
    n = len(t)
    dt_h = float(np.median(np.diff(t))) if n > 1 else 0.0
    pm = 1000.0                                   # nm -> pm
    spacing_pm = float(np.diff(run.wl_nm).mean()) * pm

    print("=" * 78)
    print(f"{run.name}   {run.timestamps[0]} -> {run.timestamps[-1]}")
    print(f"{n} cycles over {t[-1]:.2f} h ({dt_h * 60:.0f} min apart)   "
          f"grid {run.wl_nm[0]:.2f}-{run.wl_nm[-1]:.2f} nm, {run.wl_nm.size} pts "
          f"({spacing_pm:.1f} pm/pt)")
    print("=" * 78)

    channels = [
        ("centroid wl",     a["centroid"],   "nm",   1e3, "pm"),
        ("half-max mid wl", a["mid_nm"],     "nm",   1e3, "pm"),
        ("raw argmax wl",   run.peak_wl_nm,  "nm",   1e3, "pm"),
        ("FWHM",            a["fwhm_nm"],    "nm",   1e3, "pm"),
        ("OSA integrated",  a["integrated"], "uW*nm", 1.0, "uW*nm"),
        ("OSA peak",        run.peak_power,  "uW",   1.0, "uW"),
        ("PD power (DAQ)",  run.daq_mv,      "mV",   1.0, "mV"),
        ("2400 resistance", run.ohm,         "Ohm",  1.0, "Ohm"),
        ("temperature",     a["temp_c"],     "C",    1e3, "mC"),
    ]

    print("\n--- drift: least-squares line over the whole run ------------------")
    print(f"{'channel':<16}{'slope/h':>14}{'total':>12}{'scatter':>11}"
          f"{'sigma':>8}{'drift/scatter':>15}")
    stats = {}
    for name, v, unit, scale, sunit in channels:
        slope, serr, s, r = linfit(t, v)
        if not np.isfinite(slope):
            print(f"{name:<16}{'n/a':>14}")
            continue
        total = slope * (t[-1] - t[0])
        sig = abs(slope / serr) if serr else float("inf")
        stats[name] = dict(slope=slope, err=serr, total=total, scatter=s, sigma=sig, r=r)
        print(f"{name:<16}{slope * scale:>10.3f} {sunit:<3}{total * scale:>9.2f} {sunit:<3}"
              f"{s * scale:>8.2f} {sunit:<3}{sig:>7.0f}{abs(total) / s if s else 0:>14.1f}x")
    print("  scatter = std of the residual about the line (cycle-to-cycle repeatability)")
    print("  sigma   = slope / its standard error; >3 means the trend is not noise")
    print("  temperature: Thorlabs TH10K curve (4813-S01); relative changes are solid,")
    print("    the absolute value carries the sensor's +-1 C spec.")

    # --- turning points -----------------------------------------------------
    print("\n--- shape of each trend ------------------------------------------")
    for name, v, unit, *_ in channels:
        if not np.isfinite(v).any():
            continue
        i_min, i_max = int(np.nanargmin(v)), int(np.nanargmax(v))
        mono = "monotone" if abs(stats.get(name, {}).get("r", 0)) > 0.95 else "turns over"
        print(f"{name:<16} min {v[i_min]:>12.4g} {unit:<6} @ {t[i_min]:>5.2f} h   "
              f"max {v[i_max]:>12.4g} {unit:<6} @ {t[i_max]:>5.2f} h   {mono}")

    # --- translation vs shape change ---------------------------------------
    print("\n--- did the spectrum move, or change shape? ----------------------")
    lin = a["lin"]
    k = max(3, n // 8)
    first, last = lin[:k].mean(0), lin[-k:].mean(0)
    fn = first / TRAPZ(first, run.wl_nm)          # area-normalised: shape only
    ln = last / TRAPZ(last, run.wl_nm)
    guard = max(5, run.wl_nm.size // 20)
    sl = slice(guard, -guard)
    shifts = np.arange(-40, 41)
    resid = [np.sum((np.roll(fn, int(s)) - ln)[sl] ** 2) for s in shifts]
    b = int(shifts[int(np.argmin(resid))])
    r_none, r_shift = float(np.sum((fn - ln)[sl] ** 2)), float(min(resid))
    print(f"first {k} vs last {k} spectra, area-normalised:")
    print(f"  best rigid shift  {b * spacing_pm:+.0f} pm  "
          f"(explains {(1 - r_shift / r_none) * 100:.0f}% of the difference)")
    print(f"  amplitude change  {(TRAPZ(last, run.wl_nm) / TRAPZ(first, run.wl_nm) - 1) * 100:+.1f}%")
    if (1 - r_shift / r_none) < 0.5:
        print("  -> the change is mostly NOT a rigid shift; the lineshape itself changed")
    else:
        print("  -> the envelope essentially translated")

    # --- cross-correlations -------------------------------------------------
    print("\n--- cross-correlation (positive lag = second channel follows first)")
    pairs = [
        ("temperature",   a["temp_c"],     "centroid wl",    a["centroid"]),
        ("temperature",   a["temp_c"],     "OSA integrated", a["integrated"]),
        ("temperature",   a["temp_c"],     "PD power",       run.daq_mv),
        ("temperature",   a["temp_c"],     "FWHM",           a["fwhm_nm"]),
        ("PD power",      run.daq_mv,      "OSA integrated", a["integrated"]),
        ("PD power",      run.daq_mv,      "centroid wl",    a["centroid"]),
        ("centroid wl",   a["centroid"],   "OSA integrated", a["integrated"]),
    ]
    print(f"{'pair':<34}{'r (0 lag)':>11}{'best r':>10}{'at lag':>10}"
          f"{'r detrended':>13}{'best':>8}{'at lag':>9}")
    edge = MAX_LAG * dt_h
    pinned = False
    for na, va, nb, vb in pairs:
        r0, rb, lag = best_lag(va, vb, dt_h)
        d0, db, dlag = best_lag(detrend(t, va), detrend(t, vb), dt_h)
        if not np.isfinite(r0):
            continue
        mark = "!" if abs(lag) >= edge - 1e-9 else " "
        pinned |= mark == "!"
        print(f"{na + ' vs ' + nb:<34}{r0:>+11.3f}{rb:>+10.3f}{lag:>+8.2f} h{mark}"
              f"{d0:>+13.3f}{db:>+8.3f}{dlag:>+8.2f} h")
    print(f"  lag searched +-{edge:.1f} h; with n={n} points |r|>~0.29 is p<0.05,")
    print("  but neighbouring cycles are correlated, so treat these as indicative.")
    print("  'r detrended' removes each channel's own linear drift first: it asks whether")
    print("  the wiggles track, not whether both happen to trend over the same night.")
    if pinned:
        print("  ! = best lag sits on the edge of the search window -- that is a shared")
        print("      trend dominating the correlation, NOT a measured lag.")

    # --- is the wavelength drift actually driven by temperature? -----------
    print("\n--- is the wavelength drift temperature-driven? -------------------")
    temp = a["temp_c"]
    if np.isfinite(temp).sum() > 6:
        i_turn = int(np.nanargmin(temp))
        branches = [("cooling", slice(0, i_turn + 1)), ("warming", slice(i_turn, None))]
        print(f"temperature turns at {t[i_turn]:.2f} h ({run.timestamps[i_turn][11:]}), "
              f"{temp[i_turn]:.2f} C")
        slopes = []
        for label, sl in branches:
            tt, cc = temp[sl], a["centroid"][sl]
            if np.isfinite(tt).sum() < 4:
                continue
            k, kerr, _, r = linfit(tt, cc)
            slopes.append(k)
            print(f"  {label:<8} n={np.isfinite(tt).sum():<3} "
                  f"dlambda/dT = {k * 1000:+8.1f} +- {abs(kerr) * 1000:.1f} pm/C   "
                  f"(r={r:+.3f}, dT={tt[-1] - tt[0]:+.2f} C)")
        if len(slopes) == 2:
            if slopes[0] * slopes[1] < 0 or min(abs(np.array(slopes))) < 0.2 * max(abs(np.array(slopes))):
                print("  -> the two branches DISAGREE: the wavelength does not retrace when the")
                print("     room warms back up, so room temperature is not what is driving it.")
            else:
                print("  -> both branches give a consistent coefficient: consistent with a")
                print("     genuine temperature dependence.")


def plot_run(run: Run, a: dict) -> plt.Figure:
    t, lin = a["t"], a["lin"]
    n = len(t)
    power_label = f"power ({run.unit})"
    has_ohm = np.isfinite(run.ohm).any()

    fig, axes = plt.subplots(3, 2, figsize=(15, 13))
    (ax_over, ax_map), (ax_wl, ax_pw), (ax_env, ax_z) = axes
    fig.suptitle(
        f"{run.name}   {run.timestamps[0]} -> {run.timestamps[-1]}   "
        f"({n} cycles, {t[-1]:.1f} h)", fontsize=13)

    # --- overlay, coloured by elapsed time ---------------------------------
    cmap = plt.get_cmap(OVERLAY_CMAP)
    norm = Normalize(vmin=t[0], vmax=t[-1] if t[-1] > t[0] else t[0] + 1.0)
    idx = np.unique(np.linspace(0, n - 1, min(n, MAX_OVERLAY)).round().astype(int))
    for i in idx:
        ax_over.plot(run.wl_nm, run.power[i], color=cmap(norm(t[i])), linewidth=0.8)
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax_over, label="elapsed (h)")
    ax_over.set_title(f"spectra ({len(idx)} of {n} shown)")
    ax_over.set_xlabel("wavelength (nm)")
    ax_over.set_ylabel(power_label)
    ax_over.grid(True, alpha=0.3)

    # --- heat map: wavelength x time ---------------------------------------
    mesh = ax_map.pcolormesh(run.wl_nm, t, run.power, shading="nearest", cmap=HEATMAP_CMAP)
    fig.colorbar(mesh, ax=ax_map, label=power_label)
    ax_map.set_title("power vs wavelength and time")
    ax_map.set_xlabel("wavelength (nm)")
    ax_map.set_ylabel("elapsed (h)")

    # --- wavelength + width -------------------------------------------------
    ax_wl.plot(t, run.peak_wl_nm, "o-", color="C0", markersize=2.5, linewidth=0.5,
               alpha=0.3, label="peak (raw argmax)")
    ax_wl.plot(t, a["mid_nm"], "^-", color="C2", markersize=3, linewidth=0.9,
               label="half-max midpoint")
    ax_wl.plot(t, a["centroid"], "s-", color="C1", markersize=3, linewidth=1.1, label="centroid")
    ax_wl.set_title("wavelength vs time")
    ax_wl.set_xlabel("elapsed (h)")
    ax_wl.set_ylabel("wavelength (nm)")
    ax_wl.grid(True, alpha=0.3)
    ax_fw = ax_wl.twinx()
    ax_fw.plot(t, a["fwhm_nm"], ":", color="C7", linewidth=1.2, label="FWHM")
    ax_fw.set_ylabel("FWHM (nm)", color="C7")
    ax_fw.tick_params(axis="y", labelcolor="C7")
    h = ax_wl.get_legend_handles_labels()[0] + ax_fw.get_legend_handles_labels()[0]
    ax_wl.legend(handles=h, fontsize=8, loc="best")

    # --- OSA power ----------------------------------------------------------
    ax_pw.plot(t, run.peak_power, "o-", color="C0", markersize=3, linewidth=0.8,
               alpha=0.6, label="peak (uW)")
    ax_pw.set_ylabel("peak power (uW)", color="C0")
    ax_pw.tick_params(axis="y", labelcolor="C0")
    ax_int = ax_pw.twinx()
    ax_int.plot(t, a["integrated"], "s-", color="C3", markersize=3, linewidth=1.1,
                label="integrated")
    ax_int.set_ylabel("integrated power (uW*nm)", color="C3")
    ax_int.tick_params(axis="y", labelcolor="C3")
    ax_pw.set_title("OSA power vs time")
    ax_pw.set_xlabel("elapsed (h)")
    ax_pw.grid(True, alpha=0.3)
    h = ax_pw.get_legend_handles_labels()[0] + ax_int.get_legend_handles_labels()[0]
    ax_pw.legend(handles=h, fontsize=8, loc="best")

    # --- environment: temperature + PD power --------------------------------
    if has_ohm:
        ax_env.plot(t, a["temp_c"], "o-", color="C4", markersize=3, linewidth=1.1,
                    label="T (TH10K)")
        ax_env.set_ylabel("temperature (C, TH10K)", color="C4")
        ax_env.tick_params(axis="y", labelcolor="C4")
        ax_r = ax_env.secondary_yaxis(
            "right",
            functions=(lambda c: np.interp(c, a["temp_c"][::-1], run.ohm[::-1]),
                       lambda o: np.interp(o, run.ohm, a["temp_c"])),
        )
        ax_r.set_ylabel("2400 resistance (Ohm)")
    else:
        ax_env.text(0.5, 0.5, "no 2400 data in this run", ha="center", va="center",
                    transform=ax_env.transAxes)
    ax_pd = ax_env.twinx()
    ax_pd.spines["right"].set_position(("axes", 1.18))
    ax_pd.plot(t, run.daq_mv, "s-", color="C5", markersize=3, linewidth=1.1,
               label="PD power (separate path)")
    ax_pd.set_ylabel("PD (mV)", color="C5")
    ax_pd.tick_params(axis="y", labelcolor="C5")
    ax_env.set_title("temperature and photodiode power vs time")
    ax_env.set_xlabel("elapsed (h)")
    ax_env.grid(True, alpha=0.3)
    h = ax_env.get_legend_handles_labels()[0] + ax_pd.get_legend_handles_labels()[0]
    ax_env.legend(handles=h, fontsize=8, loc="best")

    # --- everything z-scored on one axis ------------------------------------
    for label, v, style in (
        ("centroid wl", a["centroid"], "-"),
        ("OSA integrated", a["integrated"], "-"),
        ("PD power", run.daq_mv, "-"),
        ("temperature (TH10K)", a["temp_c"], "--"),
        ("FWHM", a["fwhm_nm"], ":"),
    ):
        z = zscore(v)
        if np.isfinite(z).any():
            ax_z.plot(t, z, style, linewidth=1.2, marker="o", markersize=2.5, label=label)
    ax_z.axhline(0, color="k", linewidth=0.6, alpha=0.4)
    ax_z.set_title("all channels, z-scored (shape comparison only)")
    ax_z.set_xlabel("elapsed (h)")
    ax_z.set_ylabel("z-score")
    ax_z.grid(True, alpha=0.3)
    ax_z.legend(fontsize=8, ncol=2, loc="best")

    fig.tight_layout()
    return fig


def newest_master_csv() -> Path:
    files = sorted(OUT_DIR.glob("daq_osa_monitor_*.csv"), key=lambda p: p.stat().st_mtime)
    if not files:
        sys.exit(f"no daq_osa_monitor_*.csv under {OUT_DIR}")
    return files[-1]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Analyse a daq_osa_monitor run.")
    p.add_argument("csv", nargs="?", type=Path,
                   help="master daq_osa_monitor_<stamp>.csv (default: newest in src/calib_data)")
    p.add_argument("--out", type=Path, help="PNG path (default: <csv stem>_analysis.png)")
    p.add_argument("--no-show", action="store_true", help="save the figure without opening a window")
    args = p.parse_args(argv)

    master_csv = (args.csv or newest_master_csv()).resolve()
    print(f"reading {master_csv}")
    run = load_run(master_csv)
    a = analyse(run)
    report(run, a)
    fig = plot_run(run, a)

    out = args.out or master_csv.with_name(f"{master_csv.stem}_analysis.png")
    fig.savefig(out, dpi=150)
    print(f"\nsaved -> {out}")
    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
