"""Monte-Carlo evaluation of encoding geometries and the top-level run driver."""
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from .config import Cfg
from .paths import OUTPUT_DIR, write_csv
from .encoding import (
    ENCODING_ORDER, ordered_encodings, resolve_edge_taper_px,
    amplitude_from_targets, taper_active_power_factor,
)
from .geometry import channel_group_px
from .readout import (
    CORRECTION_ORDER, channel_readout, guard_readout,
    group_power_readout, guard_power_readout, readout_calibration,
    calibrated_channel_readout, build_lut_for_correction, apply_lut,
)
from .metrics import error_metrics, crosstalk_summary, validate_active_model


def monte_carlo_geometry(c: Cfg, px_per_ch=5, guard=0, n_ch=31, n_trials=300,
                         seed=0, edge_crop=0, window='group', correction='none',
                         encoding='flat', edge_taper_px=None):
    rng = np.random.default_rng(seed)
    edge_taper_px = resolve_edge_taper_px(encoding, edge_taper_px)
    blank, response = readout_calibration(
        c, n_ch, px_per_ch, guard, window, encoding, edge_taper_px
    )
    eval_idx = np.arange(edge_crop, n_ch - edge_crop if edge_crop else n_ch)
    if eval_idx.size == 0:
        raise ValueError('edge_crop removes all channels')
    if correction not in CORRECTION_ORDER:
        raise ValueError(f'correction must be one of {CORRECTION_ORDER}')
    lut = None
    if correction != 'none':
        lut = build_lut_for_correction(
            c, correction, px_per_ch, guard, n_ch, window, encoding, edge_taper_px
        )

    raw_all, cal_all, target_all, command_all = [], [], [], []
    guard_all, group_power_all, guard_power_all = [], [], []
    for _ in range(n_trials):
        target = rng.uniform(0.0, 1.0, n_ch)
        command = apply_lut(target, lut) if lut is not None else target
        amplitude = amplitude_from_targets(
            c, command, px_per_ch, guard, quantize=True, flicker=True, rng=rng,
            encoding=encoding, edge_taper_px=edge_taper_px
        )
        raw = channel_readout(c, amplitude, n_ch, px_per_ch, guard, window)
        guard_amplitude = guard_readout(c, amplitude, n_ch, px_per_ch, guard)
        group_power = group_power_readout(c, amplitude, n_ch, px_per_ch, guard)
        guard_power = guard_power_readout(c, amplitude, n_ch, px_per_ch, guard)
        cal = calibrated_channel_readout(
            c, amplitude, px_per_ch, guard, blank=blank, response=response, window=window,
            encoding=encoding, edge_taper_px=edge_taper_px
        )
        raw_all.append(raw[eval_idx])
        cal_all.append(cal[eval_idx])
        target_all.append(target[eval_idx])
        command_all.append(command[eval_idx])
        guard_all.append(guard_amplitude[eval_idx])
        group_power_all.append(group_power[eval_idx])
        guard_power_all.append(guard_power[eval_idx])

    raw_all = np.concatenate(raw_all)
    cal_all = np.concatenate(cal_all)
    target_all = np.concatenate(target_all)
    command_all = np.concatenate(command_all)
    guard_all = np.concatenate(guard_all)
    group_power_all = np.concatenate(group_power_all)
    guard_power_all = np.concatenate(guard_power_all)
    raw_metrics = error_metrics(raw_all, target_all)
    cal_metrics = error_metrics(cal_all, target_all)
    xtalk = crosstalk_summary(response, eval_idx)
    group_px = channel_group_px(px_per_ch, guard)
    throughput = px_per_ch/group_px
    active_power_factor = taper_active_power_factor(px_per_ch, encoding, edge_taper_px)
    return dict(
        px_per_ch=int(px_per_ch),
        guard=guard,
        correction=correction,
        encoding=encoding,
        edge_taper_px=edge_taper_px,
        window=window,
        edge_crop=int(edge_crop),
        group_px=group_px,
        aperture_channels=1920//group_px,
        throughput=throughput,
        guard_geometric_loss=1 - throughput,
        taper_active_power_factor=active_power_factor,
        encoded_group_power_fraction=throughput*active_power_factor,
        raw_enob=raw_metrics['enob'],
        raw_rmse=raw_metrics['rmse'],
        cal_enob=cal_metrics['enob'],
        cal_rmse=cal_metrics['rmse'],
        cal_mae=cal_metrics['mae'],
        cal_bias=cal_metrics['bias'],
        cal_p95_abs=cal_metrics['p95_abs'],
        cal_max_abs=cal_metrics['max_abs'],
        cal_mean_rel_signal=cal_metrics['mean_rel_signal'],
        guard_mean_amplitude=np.mean(guard_all),
        guard_p95_amplitude=np.percentile(guard_all, 95),
        guard_max_amplitude=np.max(guard_all),
        mean_group_power=np.mean(group_power_all),
        mean_guard_power=np.mean(guard_power_all),
        guard_p95_power=np.percentile(guard_power_all, 95),
        guard_max_power=np.max(guard_power_all),
        power_loss_vs_guard0=np.nan,
        nearest_xtalk=xtalk['nearest_xtalk'],
        mean_row_xtalk=xtalk['mean_row_xtalk'],
        max_offdiag=xtalk['max_offdiag'],
        target_samples=target_all,
        command_samples=command_all,
        recovered_samples=cal_all,
    )


def _monte_carlo_task(args):
    (c, px_per_ch, guard, n_ch, n_trials, seed, edge_crop,
     window, correction, encoding, edge_taper_px) = args
    return monte_carlo_geometry(
        c, px_per_ch, guard, n_ch, n_trials,
        seed, edge_crop, window, correction, encoding, edge_taper_px
    )


def resolve_worker_count(n_tasks: int, n_workers=None):
    if n_workers is None:
        env_workers = os.environ.get('BIT_DEPTH_WORKERS')
        if env_workers:
            try:
                n_workers = int(env_workers)
            except ValueError as exc:
                raise ValueError('BIT_DEPTH_WORKERS must be an integer') from exc
        else:
            n_workers = min(20, os.cpu_count() or 1)
    return max(1, min(int(n_workers), n_tasks))


def monte_carlo_sweep(c: Cfg, sizes=(5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19),
                      guards=(0, 1, 2, 3, 4), n_ch=31, n_trials=300,
                      seed=0, edge_crop=0, window='group', corrections=CORRECTION_ORDER,
                      encodings=ENCODING_ORDER, n_workers=None):
    tasks = []
    for px_per_ch in sizes:
        for guard in guards:
            for encoding_idx, encoding in enumerate(encodings):
                edge_taper_px = resolve_edge_taper_px(encoding)
                for correction in corrections:
                    tasks.append((
                        c, px_per_ch, guard, n_ch, n_trials,
                        seed + 1000*px_per_ch + 10*guard + 100000*encoding_idx,
                        edge_crop, window, correction, encoding, edge_taper_px
                    ))

    workers = resolve_worker_count(len(tasks), n_workers)
    print(f'Monte Carlo sweep: {len(tasks)} jobs, {workers} worker(s), {n_trials} trials/job')
    if workers == 1:
        rows = [_monte_carlo_task(task) for task in tasks]
        add_power_loss_references(rows)
        return rows

    rows = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_monte_carlo_task, task): task for task in tasks}
        for done, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            print(f"  finished {done:3d}/{len(tasks)}: "
                  f"px={row['px_per_ch']}, guard={row['guard']}, "
                  f"{row['encoding']}, {row['correction']}")

    order = {(task[1], task[2], task[9], task[8]): i for i, task in enumerate(tasks)}
    rows.sort(key=lambda r: order[(r['px_per_ch'], r['guard'], r['encoding'], r['correction'])])
    add_power_loss_references(rows)
    return rows


def add_power_loss_references(rows):
    refs = {}
    for row in rows:
        if row['guard'] == 0:
            key = (row['px_per_ch'], row['encoding'], row['edge_taper_px'], row['correction'])
            refs[key] = row['mean_group_power']
    for row in rows:
        key = (row['px_per_ch'], row['encoding'], row['edge_taper_px'], row['correction'])
        ref = refs.get(key, np.nan)
        if np.isfinite(ref) and ref > 0:
            row['power_loss_vs_guard0'] = 1 - row['mean_group_power']/ref
        else:
            row['power_loss_vs_guard0'] = np.nan
    return rows


def csv_ready_rows(rows):
    skip = {'target_samples', 'command_samples', 'recovered_samples'}
    return [{k: v for k, v in row.items() if k not in skip} for row in rows]


def save_monte_carlo_rows(rows):
    fieldnames = [
        'px_per_ch', 'guard', 'encoding', 'edge_taper_px', 'correction',
        'window', 'edge_crop', 'group_px', 'aperture_channels', 'throughput',
        'guard_geometric_loss', 'taper_active_power_factor',
        'encoded_group_power_fraction', 'mean_group_power',
        'power_loss_vs_guard0',
        'raw_enob', 'raw_rmse', 'cal_enob', 'cal_rmse', 'cal_mae',
        'cal_bias', 'cal_p95_abs', 'cal_max_abs', 'cal_mean_rel_signal',
        'guard_mean_amplitude', 'guard_p95_amplitude', 'guard_max_amplitude',
        'mean_guard_power', 'guard_p95_power', 'guard_max_power',
        'nearest_xtalk', 'mean_row_xtalk', 'max_offdiag',
    ]
    return write_csv('monte_carlo_encoding_results.csv', csv_ready_rows(rows), fieldnames)


def print_monte_carlo_rows(rows):
    print('\nMonte Carlo amplitude estimate (group readout, guard + edge channels included)')
    print('  px guard encoding        correction group thru% cal_ENOB cal_RMSE bias    guardA95 pwrLoss% guardP95%')
    for r in rows:
        print(f"  {r['px_per_ch']:2d} {r['guard']:5g} {r['encoding']:<15s}"
              f" {r['correction']:>10s} {r['group_px']:5g}"
              f" {100*r['throughput']:5.1f}"
              f" {r['cal_enob']:8.2f} {r['cal_rmse']:8.4f}"
              f" {r['cal_bias']:7.4f} {r['guard_p95_amplitude']:10.4f}"
              f" {100*r['power_loss_vs_guard0']:8.2f}"
              f" {100*r['guard_p95_power']:9.3f}")
    for encoding in ordered_encodings(rows):
        rr = [r for r in rows if r['encoding'] == encoding]
        if not rr:
            continue
        best = max(rr, key=lambda r: r['cal_enob'])
        print(f"  best {encoding}: correction={best['correction']}, "
              f"px={best['px_per_ch']}, guard={best['guard']}, "
              f"ENOB={best['cal_enob']:.2f}, RMSE={best['cal_rmse']:.4f}, "
              f"bias={best['cal_bias']:.4f}, "
              f"power_loss={100*best['power_loss_vs_guard0']:.2f}%")


def simulation_grid(c: Cfg, n_trials=300, n_workers=None):
    from . import figures
    validate_active_model(c)
    rows = monte_carlo_sweep(c, n_trials=n_trials, n_workers=n_workers)
    csv_path = save_monte_carlo_rows(rows)
    figures.fig_mc_summary(rows)
    figures.fig_mc_heatmaps(rows)
    figures.fig_guard_leakage_summary(rows)
    figures.fig_guard_power_loss(rows)
    figures.fig_mc_scatter(rows)
    figures.fig_target_actual_profile(c)
    figures.fig_guard_transfer_diagnostic(c, rows)
    print_monte_carlo_rows(rows)
    print(f'\nMonte Carlo CSV: {csv_path}')
    print(f'Figures saved to: {OUTPUT_DIR}')
