"""Channel readout, blank/response calibration, and inverse LUT corrections."""
import numpy as np

from .config import Cfg
from .geometry import group_samples, guard_samples, active_slice
from .encoding import amplitude_from_targets, resolve_edge_taper_px

CORRECTION_ORDER = ('none', 'lut', 'lut_bg05')
CORRECTION_LABELS = {
    'none': 'no LUT',
    'lut': 'LUT, zero bg',
    'lut_bg05': 'LUT, bg=0.5',
}
CORRECTION_STYLES = {'none': '--', 'lut': '-', 'lut_bg05': '-.'}
CORRECTION_MARKERS = {'none': 'o', 'lut': 's', 'lut_bg05': '^'}


def ordered_corrections(rows):
    present = {r['correction'] for r in rows}
    ordered = [c for c in CORRECTION_ORDER if c in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def preferred_correction(rows):
    corrections = {r['correction'] for r in rows}
    if 'lut_bg05' in corrections:
        return 'lut_bg05'
    if 'lut' in corrections:
        return 'lut'
    return 'none'


def channel_readout(c: Cfg, amplitude: np.ndarray, n_ch: int, px_per_ch=5, guard=0,
                    window='group'):
    """Average the propagated amplitude over each channel readout window.

    The default TPA-oriented window is the whole channel group, so residual light
    in the guard pixels is included in the measured amplitude.
    """
    gsamp = group_samples(c, px_per_ch, guard)
    if len(amplitude) != n_ch*gsamp:
        raise ValueError('amplitude length does not match n_ch, px_per_ch, guard, and ovs')
    blocks = amplitude.reshape(n_ch, gsamp)
    if window == 'group':
        return blocks.mean(axis=1)
    if window == 'active':
        return blocks[:, active_slice(c, px_per_ch, guard)].mean(axis=1)
    raise ValueError("window must be 'group' or 'active'")


def guard_readout(c: Cfg, amplitude: np.ndarray, n_ch: int, px_per_ch=5, guard=0):
    gsamp = group_samples(c, px_per_ch, guard)
    if len(amplitude) != n_ch*gsamp:
        raise ValueError('amplitude length does not match n_ch, px_per_ch, guard, and ovs')
    gsamples = guard_samples(c, guard)
    if gsamples == 0:
        return np.zeros(n_ch)
    blocks = amplitude.reshape(n_ch, gsamp)
    guard_blocks = np.concatenate([blocks[:, :gsamples],
                                   blocks[:, -gsamples:]], axis=1)
    return guard_blocks.mean(axis=1)


def group_power_readout(c: Cfg, amplitude: np.ndarray, n_ch: int, px_per_ch=5, guard=0):
    gsamp = group_samples(c, px_per_ch, guard)
    if len(amplitude) != n_ch*gsamp:
        raise ValueError('amplitude length does not match n_ch, px_per_ch, guard, and ovs')
    blocks = amplitude.reshape(n_ch, gsamp)
    return np.mean(blocks**2, axis=1)


def guard_power_readout(c: Cfg, amplitude: np.ndarray, n_ch: int, px_per_ch=5, guard=0):
    gsamp = group_samples(c, px_per_ch, guard)
    if len(amplitude) != n_ch*gsamp:
        raise ValueError('amplitude length does not match n_ch, px_per_ch, guard, and ovs')
    gsamples = guard_samples(c, guard)
    if gsamples == 0:
        return np.zeros(n_ch)
    blocks = amplitude.reshape(n_ch, gsamp)
    guard_blocks = np.concatenate([blocks[:, :gsamples],
                                   blocks[:, -gsamples:]], axis=1)
    return np.mean(guard_blocks**2, axis=1)


def readout_calibration(c: Cfg, n_ch: int, px_per_ch=5, guard=0, window='group',
                        encoding='flat', edge_taper_px=None):
    blank = channel_readout(
        c, amplitude_from_targets(
            c, np.zeros(n_ch), px_per_ch, guard,
            encoding=encoding, edge_taper_px=edge_taper_px
        ),
        n_ch, px_per_ch, guard, window
    )
    response = np.zeros((n_ch, n_ch))
    for i in range(n_ch):
        target = np.zeros(n_ch)
        target[i] = 1.0
        response[:, i] = channel_readout(
            c, amplitude_from_targets(
                c, target, px_per_ch, guard,
                encoding=encoding, edge_taper_px=edge_taper_px
            ),
            n_ch, px_per_ch, guard, window
        ) - blank
    return blank, response


def calibrated_channel_readout(c: Cfg, amplitude: np.ndarray, px_per_ch=5, guard=0,
                               blank=None, response=None, window='group',
                               encoding='flat', edge_taper_px=None):
    n_ch = len(amplitude)//group_samples(c, px_per_ch, guard)
    raw = channel_readout(c, amplitude, n_ch, px_per_ch, guard, window)
    if blank is None or response is None:
        blank, response = readout_calibration(
            c, n_ch, px_per_ch, guard, window, encoding, edge_taper_px
        )
    diag = np.diag(response)
    if np.any(diag == 0):
        raise ValueError('zero diagonal response in readout calibration')
    return (raw - blank)/diag


def _monotone_lut(command_grid, actual_grid):
    actual_mono = np.maximum.accumulate(actual_grid)
    actual_unique, unique_idx = np.unique(actual_mono, return_index=True)
    command_unique = command_grid[unique_idx]
    if actual_unique[0] > 0:
        actual_unique = np.r_[0.0, actual_unique]
        command_unique = np.r_[0.0, command_unique]
    if actual_unique[-1] < 1:
        actual_unique = np.r_[actual_unique, 1.0]
        command_unique = np.r_[command_unique, 1.0]
    return actual_unique, command_unique


def build_single_channel_lut(c: Cfg, px_per_ch=5, guard=0, n_ch=31,
                             n_lut=501, window='group', quantize=True,
                             encoding='flat', edge_taper_px=None):
    """Build an inverse LUT target -> command from the isolated channel response."""
    blank, response = readout_calibration(
        c, n_ch, px_per_ch, guard, window, encoding, edge_taper_px
    )
    mid = n_ch//2
    command_grid = np.linspace(0, 1, n_lut)
    actual_grid = np.zeros_like(command_grid)
    for i, command in enumerate(command_grid):
        target = np.zeros(n_ch)
        target[mid] = command
        amplitude = amplitude_from_targets(
            c, target, px_per_ch, guard, quantize=quantize, flicker=False,
            encoding=encoding, edge_taper_px=edge_taper_px
        )
        actual_grid[i] = calibrated_channel_readout(
            c, amplitude, px_per_ch, guard, blank=blank, response=response, window=window
        )[mid]

    actual_unique, command_unique = _monotone_lut(command_grid, actual_grid)
    return dict(command_grid=command_grid,
                actual_grid=actual_grid,
                actual_unique=actual_unique,
                command_unique=command_unique)


def build_background_lut(c: Cfg, px_per_ch=5, guard=0, n_ch=31,
                         n_lut=501, window='group', quantize=True, background=0.5,
                         encoding='flat', edge_taper_px=None):
    """Build an inverse LUT with neighboring channels held at a fixed background."""
    blank, response = readout_calibration(
        c, n_ch, px_per_ch, guard, window, encoding, edge_taper_px
    )
    mid = n_ch//2
    command_grid = np.linspace(0, 1, n_lut)
    actual_grid = np.zeros_like(command_grid)
    background_target = np.full(n_ch, background, dtype=float)
    for i, command in enumerate(command_grid):
        target = background_target.copy()
        target[mid] = command
        amplitude = amplitude_from_targets(
            c, target, px_per_ch, guard, quantize=quantize, flicker=False,
            encoding=encoding, edge_taper_px=edge_taper_px
        )
        actual_grid[i] = calibrated_channel_readout(
            c, amplitude, px_per_ch, guard, blank=blank, response=response, window=window
        )[mid]

    actual_unique, command_unique = _monotone_lut(command_grid, actual_grid)
    return dict(command_grid=command_grid,
                actual_grid=actual_grid,
                actual_unique=actual_unique,
                command_unique=command_unique,
                background=background)


def apply_lut(target: np.ndarray, lut):
    target = np.clip(np.asarray(target, dtype=float), 0, 1)
    command = np.interp(target, lut['actual_unique'], lut['command_unique'])
    return np.clip(command, 0, 1)


def build_lut_for_correction(c: Cfg, correction: str, px_per_ch=5, guard=0, n_ch=31,
                             window='group', encoding='flat', edge_taper_px=None):
    if correction == 'lut':
        return build_single_channel_lut(
            c, px_per_ch, guard, n_ch, window=window,
            encoding=encoding, edge_taper_px=edge_taper_px
        )
    if correction == 'lut_bg05':
        return build_background_lut(
            c, px_per_ch, guard, n_ch, window=window, background=0.5,
            encoding=encoding, edge_taper_px=edge_taper_px
        )
    raise ValueError(f'correction {correction!r} does not use a LUT')


def recovered_for_target(c: Cfg, target: np.ndarray, px_per_ch=5, guard=0, seed=0,
                         n_ch=None, window='group', correction='none',
                         encoding='flat', edge_taper_px=None):
    rng = np.random.default_rng(seed)
    target = np.asarray(target, dtype=float)
    n_ch = len(target) if n_ch is None else n_ch
    edge_taper_px = resolve_edge_taper_px(encoding, edge_taper_px)
    blank, response = readout_calibration(
        c, n_ch, px_per_ch, guard, window, encoding, edge_taper_px
    )
    if correction != 'none':
        lut = build_lut_for_correction(
            c, correction, px_per_ch, guard, n_ch, window, encoding, edge_taper_px
        )
        command = apply_lut(target, lut)
    elif correction == 'none':
        command = target
    else:
        raise ValueError(f'correction must be one of {CORRECTION_ORDER}')
    amplitude = amplitude_from_targets(
        c, command, px_per_ch, guard, quantize=True, flicker=True, rng=rng,
        encoding=encoding, edge_taper_px=edge_taper_px
    )
    recovered = calibrated_channel_readout(
        c, amplitude, px_per_ch, guard, blank=blank, response=response, window=window,
        encoding=encoding, edge_taper_px=edge_taper_px
    )
    return recovered, command, amplitude


def single_channel_transfer_curve(c: Cfg, px_per_ch=5, guard=0, n_ch=31,
                                  n_points=101, window='group', encoding='flat',
                                  edge_taper_px=None):
    edge_taper_px = resolve_edge_taper_px(encoding, edge_taper_px)
    blank, response = readout_calibration(
        c, n_ch, px_per_ch, guard, window, encoding, edge_taper_px
    )
    mid = n_ch//2
    targets = np.linspace(0, 1, n_points)
    recovered = np.zeros_like(targets)
    for i, T in enumerate(targets):
        target = np.zeros(n_ch)
        target[mid] = T
        amplitude = amplitude_from_targets(
            c, target, px_per_ch, guard, encoding=encoding, edge_taper_px=edge_taper_px
        )
        recovered[i] = calibrated_channel_readout(
            c, amplitude, px_per_ch, guard, blank=blank, response=response, window=window,
            encoding=encoding, edge_taper_px=edge_taper_px
        )[mid]
    return targets, recovered


def lut_corrected_transfer_curve(c: Cfg, px_per_ch=5, guard=0, n_ch=31,
                                 n_points=101, window='group', correction='lut',
                                 encoding='flat', edge_taper_px=None):
    edge_taper_px = resolve_edge_taper_px(encoding, edge_taper_px)
    lut = build_lut_for_correction(
        c, correction, px_per_ch, guard, n_ch, window, encoding, edge_taper_px
    )
    targets = np.linspace(0, 1, n_points)
    commands = apply_lut(targets, lut)
    blank, response = readout_calibration(
        c, n_ch, px_per_ch, guard, window, encoding, edge_taper_px
    )
    mid = n_ch//2
    recovered = np.zeros_like(targets)
    for i, command in enumerate(commands):
        target = np.zeros(n_ch)
        target[mid] = command
        amplitude = amplitude_from_targets(
            c, target, px_per_ch, guard, quantize=True,
            encoding=encoding, edge_taper_px=edge_taper_px
        )
        recovered[i] = calibrated_channel_readout(
            c, amplitude, px_per_ch, guard, blank=blank, response=response, window=window,
            encoding=encoding, edge_taper_px=edge_taper_px
        )[mid]
    return targets, commands, recovered
