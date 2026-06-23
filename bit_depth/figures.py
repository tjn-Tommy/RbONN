"""All Monte-Carlo / diagnostic figures.

Importing this module pulls in matplotlib (Agg backend). MPLCONFIGDIR is already
set by ``paths`` on import, so this stays headless-safe.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .paths import OUTPUT_DIR
from .encoding import (
    ENCODING_LABELS, ENCODING_STYLES, ENCODING_MARKERS, ordered_encodings,
)
from .readout import (
    CORRECTION_ORDER, CORRECTION_LABELS, CORRECTION_STYLES, CORRECTION_MARKERS,
    ordered_corrections, preferred_correction,
    single_channel_transfer_curve, lut_corrected_transfer_curve, recovered_for_target,
)

plt.rcParams.update({'font.size': 10.5, 'figure.dpi': 130, 'axes.grid': False,
                     'font.family': 'DejaVu Sans', 'axes.unicode_minus': False})


def save_figure(fig, filename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path


def fig_mc_summary(rows):
    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.2))
    guards = sorted({r['guard'] for r in rows})
    encodings = ordered_encodings(rows)
    correction = preferred_correction(rows)
    for encoding in encodings:
        for guard in guards:
            rr = sorted(
                [r for r in rows
                 if r['guard'] == guard and r['correction'] == correction and r['encoding'] == encoding],
                key=lambda r: r['px_per_ch']
            )
            if not rr:
                continue
            px = [r['px_per_ch'] for r in rr]
            label = f'{ENCODING_LABELS.get(encoding, encoding)}, guard={guard}'
            fmt = ENCODING_MARKERS.get(encoding, 'o') + ENCODING_STYLES.get(encoding, '-')
            ax[0].plot(px, [r['cal_enob'] for r in rr], fmt, label=label)
            ax[1].plot(px, [r['cal_rmse'] for r in rr], fmt, label=label)
            ax[2].plot(px, [r['cal_p95_abs'] for r in rr], fmt, label=label)

    ax[0].set_xlabel('Active encoding size [px]')
    ax[0].set_ylabel('Monte Carlo ENOB')
    correction_label = CORRECTION_LABELS.get(correction, correction)
    ax[0].set_title(f'(a) Calibrated amplitude ENOB\n{correction_label}')
    ax[1].set_xlabel('Active encoding size [px]')
    ax[1].set_ylabel('Amplitude RMSE')
    ax[1].set_title('(b) Calibrated amplitude RMSE')
    ax[2].set_xlabel('Active encoding size [px]')
    ax[2].set_ylabel('95th percentile |amplitude error|')
    ax[2].set_title('(c) Amplitude error tail')
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8.5)
    return save_figure(fig, 'fig_mc_encoding_summary.png')


def fig_mc_heatmaps(rows):
    sizes = sorted({r['px_per_ch'] for r in rows})
    guards = sorted({r['guard'] for r in rows})
    encodings = ordered_encodings(rows)
    preferred = preferred_correction(rows)
    row_keys = [(encoding, guard) for encoding in encodings for guard in guards]
    preferred_enob = np.full((len(row_keys), len(sizes)), np.nan)
    preferred_gain = np.full_like(preferred_enob, np.nan)
    preferred_vs_zero = np.full_like(preferred_enob, np.nan)
    lookup = {(r['correction'], r['encoding'], r['guard'], r['px_per_ch']): r for r in rows}
    for i, (encoding, guard) in enumerate(row_keys):
        for j, size in enumerate(sizes):
            lut = lookup.get((preferred, encoding, guard, size))
            base = lookup.get(('none', encoding, guard, size))
            zero_lut = lookup.get(('lut', encoding, guard, size))
            if lut is None:
                continue
            preferred_enob[i, j] = lut['cal_enob']
            if base is not None:
                preferred_gain[i, j] = lut['cal_enob'] - base['cal_enob']
            if zero_lut is not None:
                preferred_vs_zero[i, j] = lut['cal_enob'] - zero_lut['cal_enob']

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    label = CORRECTION_LABELS.get(preferred, preferred)
    specs = [(preferred_enob, f'{label} MC ENOB'),
             (preferred_gain, f'{label} ENOB gain vs no LUT'),
             (preferred_vs_zero, f'{label} ENOB gain vs zero-bg LUT')]
    for a, (mat, title) in zip(ax, specs):
        im = a.imshow(mat, aspect='auto', origin='lower', cmap='viridis')
        a.set_xticks(range(len(sizes))); a.set_xticklabels(sizes)
        a.set_yticks(range(len(row_keys)))
        a.set_yticklabels([f'{ENCODING_LABELS.get(e, e)}, g={g}' for e, g in row_keys], fontsize=7)
        a.set_xlabel('Active encoding size [px]')
        a.set_ylabel('Encoding, guard')
        a.set_title(title)
        plt.colorbar(im, ax=a, fraction=0.046)
    return save_figure(fig, 'fig_mc_encoding_heatmaps.png')


def fig_guard_leakage_summary(rows):
    fig, ax = plt.subplots(1, 2, figsize=(12.2, 4.2))
    guards = [g for g in sorted({r['guard'] for r in rows}) if g > 0]
    encodings = ordered_encodings(rows)
    correction = preferred_correction(rows)
    for encoding in encodings:
        for guard in guards:
            rr = sorted(
                [r for r in rows
                 if r['guard'] == guard and r['correction'] == correction and r['encoding'] == encoding],
                key=lambda r: r['px_per_ch']
            )
            if not rr:
                continue
            px = [r['px_per_ch'] for r in rr]
            fmt = ENCODING_MARKERS.get(encoding, 'o') + ENCODING_STYLES.get(encoding, '-')
            label = f'{ENCODING_LABELS.get(encoding, encoding)}, guard={guard}'
            ax[0].plot(px, [r['guard_p95_amplitude'] for r in rr], fmt, label=label)
            ax[1].plot(px, [r['guard_max_amplitude'] for r in rr], fmt, label=label)

    ax[0].set_xlabel('Active encoding size [px]')
    ax[0].set_ylabel('Guard amplitude, 95th percentile')
    ax[0].set_title('(a) Guard amplitude leakage')
    ax[1].set_xlabel('Active encoding size [px]')
    ax[1].set_ylabel('Guard amplitude, max')
    ax[1].set_title('(b) Worst guard amplitude leakage')
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8.5)
    return save_figure(fig, 'fig_guard_leakage_summary.png')


def fig_guard_power_loss(rows):
    sizes = sorted({r['px_per_ch'] for r in rows})
    guards = sorted({r['guard'] for r in rows})
    encodings = ordered_encodings(rows)
    row_keys = [(encoding, guard) for encoding in encodings for guard in guards]
    correction = preferred_correction(rows)
    lookup = {(r['correction'], r['encoding'], r['guard'], r['px_per_ch']): r for r in rows}

    geometric_loss = np.full((len(row_keys), len(sizes)), np.nan)
    propagated_loss = np.full_like(geometric_loss, np.nan)
    guard_power = np.full_like(geometric_loss, np.nan)
    for i, (encoding, guard) in enumerate(row_keys):
        for j, size in enumerate(sizes):
            row = lookup.get((correction, encoding, guard, size))
            if row is None:
                continue
            geometric_loss[i, j] = row['guard_geometric_loss']
            propagated_loss[i, j] = row['power_loss_vs_guard0']
            guard_power[i, j] = row['guard_p95_power']

    fig, ax = plt.subplots(1, 3, figsize=(16, 5.4))
    specs = [
        (geometric_loss, 'Geometric guard loss'),
        (propagated_loss, f'Propagated power loss vs guard=0\n{CORRECTION_LABELS.get(correction, correction)}'),
        (guard_power, 'Guard power leakage, 95th percentile'),
    ]
    ylabels = [f'{ENCODING_LABELS.get(e, e)}, g={g}' for e, g in row_keys]
    for a, (mat, title) in zip(ax, specs):
        im = a.imshow(100*mat, aspect='auto', origin='lower', cmap='magma')
        a.set_xticks(range(len(sizes))); a.set_xticklabels(sizes)
        a.set_yticks(range(len(row_keys))); a.set_yticklabels(ylabels, fontsize=7)
        a.set_xlabel('Active encoding size [px]')
        a.set_ylabel('Encoding, guard')
        a.set_title(title)
        plt.colorbar(im, ax=a, fraction=0.046, label='%')
    return save_figure(fig, 'fig_guard_power_loss.png')


def fig_mc_scatter(rows):
    chosen = []
    correction = preferred_correction(rows)
    for encoding in ordered_encodings(rows):
        rr = [r for r in rows if r['correction'] == correction and r['encoding'] == encoding]
        if not rr:
            continue
        chosen.append(min(rr, key=lambda r: (abs(r['px_per_ch'] - 5), abs(r['guard'] - 0))))
    corrected = [r for r in rows if r['correction'] == correction]
    if corrected:
        best_corrected = max(corrected, key=lambda r: r['cal_enob'])
        if best_corrected not in chosen:
            chosen.append(best_corrected)
    fig, ax = plt.subplots(1, len(chosen), figsize=(5.5*len(chosen), 4.7), squeeze=False)
    for a, r in zip(ax[0], chosen):
        target = r['target_samples']
        recovered = r['recovered_samples']
        if len(target) > 5000:
            idx = np.linspace(0, len(target) - 1, 5000).astype(int)
            target = target[idx]; recovered = recovered[idx]
        a.plot([0, 1], [0, 1], 'k--', lw=1)
        a.scatter(target, recovered, s=5, alpha=0.25)
        a.set_xlim(-0.05, 1.05); a.set_ylim(-0.05, 1.05)
        a.set_xlabel('Target amplitude')
        a.set_ylabel('Recovered amplitude')
        label = f"{ENCODING_LABELS.get(r['encoding'], r['encoding'])}, {CORRECTION_LABELS.get(r['correction'], r['correction'])}"
        a.set_title(f"{label}: px={r['px_per_ch']}, guard={r['guard']}\nENOB={r['cal_enob']:.2f}, RMSE={r['cal_rmse']:.4f}")
        a.grid(alpha=0.25)
    return save_figure(fig, 'fig_mc_target_vs_recovered.png')


def fig_target_actual_profile(c, px_per_ch=5, guards=(0, 1, 2), n_ch=31, seed=11,
                              encoding='flat'):
    rng = np.random.default_rng(seed)
    target = rng.uniform(0.0, 1.0, n_ch)
    fig, ax = plt.subplots(len(guards), 2, figsize=(12.5, 3.2*len(guards)), squeeze=False)
    ch = np.arange(n_ch)
    corrections = CORRECTION_ORDER
    error_offsets = np.linspace(-0.24, 0.24, len(corrections))
    error_width = 0.48/len(corrections)
    for row, guard in enumerate(guards):
        recovered_by_correction = {}
        command_by_correction = {}
        error_by_correction = {}
        for correction in corrections:
            recovered, command, _ = recovered_for_target(
                c, target, px_per_ch, guard, seed + guard, n_ch,
                correction=correction, encoding=encoding
            )
            recovered_by_correction[correction] = recovered
            command_by_correction[correction] = command
            error_by_correction[correction] = recovered - target

        ax[row, 0].plot(ch, target, 'o-', lw=1.5, ms=3.5, label='Target amplitude')
        for correction in corrections:
            ax[row, 0].plot(
                ch, recovered_by_correction[correction],
                CORRECTION_MARKERS.get(correction, 'o') + CORRECTION_STYLES.get(correction, '-'),
                lw=1.1, ms=3.0, label=CORRECTION_LABELS.get(correction, correction)
            )
        ax[row, 0].set_ylim(-0.1, 1.1)
        ax[row, 0].set_ylabel('Amplitude')
        ax[row, 0].set_title(
            f'px={px_per_ch}, guard={guard}, {ENCODING_LABELS.get(encoding, encoding)}: target vs group amplitude'
        )
        ax[row, 0].grid(alpha=0.25)
        ax[row, 0].legend(fontsize=8.5)

        ax[row, 1].axhline(0, color='k', lw=1)
        for offset, correction in zip(error_offsets, corrections):
            ax[row, 1].bar(
                ch + offset, error_by_correction[correction],
                width=error_width, label=CORRECTION_LABELS.get(correction, correction)
            )
        ax[row, 1].set_ylim(-0.35, 0.35)
        ax[row, 1].set_ylabel('Actual amplitude - target')
        rmse_parts = [
            f"{correction}={np.sqrt(np.mean(error_by_correction[correction]**2)):.4f}"
            for correction in corrections
        ]
        command_lut_bg = command_by_correction.get('lut_bg05', target)
        ax[row, 1].set_title('; '.join(rmse_parts) +
                             f"; bg05 command={command_lut_bg.min():.2f}-{command_lut_bg.max():.2f}")
        ax[row, 1].grid(alpha=0.25)
        ax[row, 1].legend(fontsize=8.5)
    ax[-1, 0].set_xlabel('Channel')
    ax[-1, 1].set_xlabel('Channel')
    return save_figure(fig, 'fig_target_vs_actual_profile.png')


def fig_guard_transfer_diagnostic(c, rows=None, px_per_ch=5, guards=(0, 1, 2),
                                  encoding='flat'):
    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.2))
    for guard in guards:
        target, recovered = single_channel_transfer_curve(c, px_per_ch, guard, encoding=encoding)
        ax[0].plot(target, recovered, '--', label=f'no LUT, guard={guard}')
        for correction in ('lut', 'lut_bg05'):
            _, _, recovered_lut = lut_corrected_transfer_curve(
                c, px_per_ch, guard, correction=correction, encoding=encoding
            )
            ax[0].plot(
                target, recovered_lut,
                CORRECTION_STYLES.get(correction, '-'),
                label=f'{CORRECTION_LABELS.get(correction, correction)}, guard={guard}'
            )

    ax[0].plot([0, 1], [0, 1], 'k--', lw=1, label='ideal')
    ax[0].set_xlabel('Target amplitude')
    ax[0].set_ylabel('Group-calibrated actual amplitude')
    ax[0].set_title('(a) Isolated transfer curve\nwhole group includes guard light')

    if rows is not None:
        corrections = ordered_corrections(rows)
        lookup = {
            (r['correction'], r['guard']): r
            for r in rows if r['px_per_ch'] == px_per_ch and r['encoding'] == encoding
        }
        x = np.arange(len(guards))
        width = 0.72/len(corrections)
        ax[1].axhline(0, color='k', lw=1)
        offsets = (np.arange(len(corrections)) - (len(corrections) - 1)/2)*width
        for offset, correction in zip(offsets, corrections):
            values = [lookup.get((correction, guard), {}).get('cal_bias', np.nan)
                      for guard in guards]
            ax[1].bar(x + offset, values, width,
                      label=f'{CORRECTION_LABELS.get(correction, correction)} bias')
        ax[1].set_xticks(x); ax[1].set_xticklabels([str(g) for g in guards])
        ax[1].set_xlabel('Guard [px]')
        ax[1].set_ylabel('Mean amplitude error')
        ax[1].set_title(f'(b) Bias by correction, px={px_per_ch}, {ENCODING_LABELS.get(encoding, encoding)}')
        ax[1].legend(fontsize=8.5)

        for offset, correction in zip(offsets, corrections):
            values = [lookup.get((correction, guard), {}).get('cal_rmse', np.nan)
                      for guard in guards]
            ax[2].bar(x + offset, values, width,
                      label=f'{CORRECTION_LABELS.get(correction, correction)} RMSE')
        ax[2].set_xticks(x); ax[2].set_xticklabels([str(g) for g in guards])
        ax[2].set_xlabel('Guard [px]')
        ax[2].set_ylabel('Amplitude RMSE')
        ax[2].set_title('(c) Corrected random-vector error')
        ax[2].legend(fontsize=8.5)
    else:
        for guard in guards:
            target, recovered = single_channel_transfer_curve(c, px_per_ch, guard, encoding=encoding)
            err = recovered - target
            ax[1].plot(target, err, label=f'guard={guard}')
            ax[2].plot(target, np.abs(err), label=f'guard={guard}')
        ax[1].axhline(0, color='k', lw=1)
        ax[1].set_xlabel('Target amplitude')
        ax[1].set_ylabel('Actual amplitude - target')
        ax[1].set_title('(b) Transfer nonlinearity')
        ax[2].set_xlabel('Target amplitude')
        ax[2].set_ylabel('|Actual amplitude - target|')
        ax[2].set_title('(c) Absolute transfer error')

    for a in ax:
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=8.5)
    return save_figure(fig, 'fig_guard_transfer_diagnostic.png')


def fig_nn_comparison(c, px_per_ch=15, guard=2.5, n_ch=21, n_trials=200, seed=1,
                      encodings=('flat', 'edge_taper_1px', 'edge_taper_2px', 'nn'),
                      correction='lut', frontier=None):
    """Compare the learned ``nn`` encoding against flat/taper at the operating geometry.

    Three panels: (a) the accuracy/intensity trade-off (ENOB vs max-intensity
    efficiency), (b) the per-pixel profile each encoding applies at full drive,
    and (c) calibrated amplitude RMSE. Requires a trained ``nn_encoder.npz``.

    ``frontier`` is an optional list of dicts with ``eff`` and ``enob`` keys
    (the NN swept over its intensity-penalty weight) drawn as a curve in (a).
    """
    from .montecarlo import monte_carlo_geometry
    from .encoding import edge_taper_weights
    from . import nn_encoder

    def profile_at_max(enc):
        if enc == 'nn':
            return nn_encoder.nn_profile_single(1.0, px_per_ch)
        return edge_taper_weights(px_per_ch, enc)

    rows = {enc: monte_carlo_geometry(
        c, px_per_ch=px_per_ch, guard=guard, n_ch=n_ch, n_trials=n_trials, seed=seed,
        correction=correction, encoding=enc) for enc in encodings}
    profiles = {enc: profile_at_max(enc) for enc in encodings}
    eff = {enc: float(np.mean(profiles[enc] ** 2)) for enc in encodings}

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))
    if frontier:
        fr = sorted(frontier, key=lambda p: p['eff'])
        ax[0].plot([p['eff'] for p in fr], [p['enob'] for p in fr], '-', color='#c44e52',
                   alpha=0.6, zorder=2, label='NN frontier (intensity-penalty sweep)')
    for enc in encodings:
        label = ENCODING_LABELS.get(enc, enc)
        marker = ENCODING_MARKERS.get(enc, 'o')
        ax[0].scatter(eff[enc], rows[enc]['cal_enob'], s=90, marker=marker, label=label, zorder=3)
        ax[0].annotate(label, (eff[enc], rows[enc]['cal_enob']),
                       textcoords='offset points', xytext=(6, 4), fontsize=8.5)
    ax[0].set_xlabel('Max-intensity efficiency  mean(p(a=1)^2)')
    ax[0].set_ylabel('Calibrated ENOB (LUT)')
    ax[0].set_title('(a) Accuracy vs intensity trade-off\nupper-right is better')
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)

    px_axis = np.arange(px_per_ch)
    for enc in encodings:
        ax[1].plot(px_axis, profiles[enc],
                   ENCODING_MARKERS.get(enc, 'o') + ENCODING_STYLES.get(enc, '-'),
                   ms=4, label=f'{ENCODING_LABELS.get(enc, enc)} (eff={eff[enc]:.2f})')
    ax[1].set_xlabel('Pixel within 15-px window')
    ax[1].set_ylabel('Encoded amplitude p(a=1)')
    ax[1].set_ylim(-0.05, 1.1)
    ax[1].set_title('(b) Learned vs hand-designed profiles')
    ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8.5)

    names = [ENCODING_LABELS.get(e, e) for e in encodings]
    ax[2].bar(names, [rows[e]['cal_rmse'] for e in encodings],
              color=['#4c72b0', '#dd8452', '#55a868', '#c44e52'][:len(encodings)])
    ax[2].set_ylabel('Calibrated amplitude RMSE (LUT)')
    ax[2].set_title('(c) Random-vector error\nlower is better')
    ax[2].tick_params(axis='x', labelrotation=20)
    ax[2].grid(alpha=0.3, axis='y')
    return save_figure(fig, 'fig_nn_encoding_comparison.png')
