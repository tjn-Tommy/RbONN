"""Channel encodings: how a per-channel target amplitude becomes commanded phase.

Built-in encodings:
  * ``flat``            - uniform amplitude across the active window.
  * ``edge_taper_1px``  - raised-cosine taper over the outer 1 px of each edge.
  * ``edge_taper_2px``  - raised-cosine taper over the outer 2 px of each edge.
  * ``nn``              - a universal MLP profile learned by ``train_nn.py`` and
                          loaded (torch-free) via ``nn_encoder``.

The ``nn`` encoding is intentionally absent from the default ``ENCODING_ORDER``
sweep so the standard run never depends on a trained checkpoint; pass it
explicitly (e.g. ``encodings=('flat', 'nn')``) once ``nn_encoder.npz`` exists.
"""
import numpy as np

from .config import Cfg
from .optics import phi_for_amplitude, amplitude_from_phase
from .geometry import (
    channel_group_px, group_samples, guard_samples, active_samples,
    quantize_channel_phases, phase_from_channel_phases,
)

ENCODING_ORDER = ('flat', 'edge_taper_1px', 'edge_taper_2px')
ENCODING_LABELS = {
    'flat': 'flat',
    'edge_taper_1px': 'taper 1 px',
    'edge_taper_2px': 'taper 2 px',
    'nn': 'NN learned',
}
ENCODING_STYLES = {'flat': '-', 'edge_taper_1px': '--', 'edge_taper_2px': '-.', 'nn': '-'}
ENCODING_MARKERS = {'flat': 'o', 'edge_taper_1px': 's', 'edge_taper_2px': '^', 'nn': 'D'}
ENCODING_TAPER_PX = {'flat': 0.0, 'edge_taper_1px': 1.0, 'edge_taper_2px': 2.0, 'nn': 0.0}


def ordered_encodings(rows):
    present = {r['encoding'] for r in rows}
    ordered = [e for e in ENCODING_ORDER if e in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def resolve_edge_taper_px(encoding='flat', edge_taper_px=None):
    if encoding not in ENCODING_TAPER_PX:
        raise ValueError(f'encoding must be one of {tuple(ENCODING_TAPER_PX)}')
    if edge_taper_px is None:
        edge_taper_px = ENCODING_TAPER_PX[encoding]
    edge_taper_px = float(edge_taper_px)
    if edge_taper_px < 0:
        raise ValueError('edge_taper_px must be non-negative')
    return edge_taper_px


def edge_taper_weights(px_per_ch=5, encoding='flat', edge_taper_px=None):
    px_per_ch = int(px_per_ch)
    if px_per_ch <= 0:
        raise ValueError('px_per_ch must be positive')
    edge_taper_px = resolve_edge_taper_px(encoding, edge_taper_px)
    weights = np.ones(px_per_ch, dtype=float)
    if edge_taper_px <= 0:
        return weights
    j = np.arange(px_per_ch, dtype=float)
    d = np.minimum(j + 0.5, px_per_ch - j - 0.5)
    taper = d < edge_taper_px
    weights[taper] = 0.5 - 0.5*np.cos(np.pi*d[taper]/edge_taper_px)
    return weights


def taper_active_power_factor(px_per_ch=5, encoding='flat', edge_taper_px=None):
    weights = edge_taper_weights(px_per_ch, encoding, edge_taper_px)
    return float(np.mean(weights**2))


def _active_amplitude(encoding, amplitude, weights, px_per_ch):
    """Per-pixel active amplitude profile for one channel at target ``amplitude``."""
    if encoding == 'nn':
        from .nn_encoder import nn_profile_single
        return np.clip(nn_profile_single(float(amplitude), int(px_per_ch)), 0.0, 1.0)
    return np.clip(amplitude*weights, 0.0, 1.0)


def phase_from_targets(c: Cfg, amplitude_ch: np.ndarray, px_per_ch=5, guard=0,
                       quantize=False, flicker=False, rng=None,
                       encoding='flat', edge_taper_px=None):
    """Per-pixel commanded phase for a vector of channel target amplitudes.

    Requires integer guard. The eval path uses :func:`phase_grid_from_targets`,
    which also covers fractional guard; this per-pixel form is kept for the flat
    fast path, model validation, and callers that want the pixel grid directly.
    """
    amplitude_ch = np.asarray(amplitude_ch, dtype=float)
    edge_taper_px = resolve_edge_taper_px(encoding, edge_taper_px)
    if not float(guard).is_integer():
        raise ValueError('phase_from_targets requires integer guard; '
                         'use phase_grid_from_targets for fractional guard')
    if encoding == 'flat' and edge_taper_px == 0:
        phi_ch = phi_for_amplitude(amplitude_ch)
        if quantize:
            phi_ch = quantize_channel_phases(c, phi_ch)
        if flicker:
            rng = rng or np.random.default_rng()
            phi_ch = phi_ch + rng.normal(0, c.sig_flicker, len(phi_ch))
        return phase_from_channel_phases(phi_ch, px_per_ch, guard)

    weights = edge_taper_weights(px_per_ch, encoding, edge_taper_px)
    group_px = int(channel_group_px(px_per_ch, guard))
    guard = int(guard)
    phase = np.full(len(amplitude_ch)*group_px, np.pi, dtype=float)
    rng = rng or np.random.default_rng()
    for i, amplitude in enumerate(amplitude_ch):
        active_amp = _active_amplitude(encoding, amplitude, weights, px_per_ch)
        phi_px = phi_for_amplitude(active_amp)
        if quantize:
            phi_px = quantize_channel_phases(c, phi_px)
        if flicker:
            phi_px = phi_px + rng.normal(0, c.sig_flicker, len(phi_px))
        start = i*group_px + guard
        phase[start:start+px_per_ch] = phi_px
    return phase


def phase_grid_from_targets(c: Cfg, amplitude_ch: np.ndarray, px_per_ch=5, guard=0,
                            quantize=False, flicker=False, rng=None,
                            encoding='flat', edge_taper_px=None):
    """Commanded phase on the oversampled grid; supports fractional guard.

    For integer guard this equals ``np.repeat(phase_from_targets(...), ovs)``
    exactly. For fractional guard the active window is placed on the grid in
    sample units (``guard*ovs`` rounded), which the per-pixel representation
    cannot express.
    """
    guard = float(guard)
    if guard.is_integer():
        phase = phase_from_targets(c, amplitude_ch, px_per_ch, int(guard),
                                   quantize, flicker, rng, encoding, edge_taper_px)
        return np.repeat(phase, c.ovs)

    amplitude_ch = np.asarray(amplitude_ch, dtype=float)
    edge_taper_px = resolve_edge_taper_px(encoding, edge_taper_px)
    gs = group_samples(c, px_per_ch, guard)
    off = guard_samples(c, guard)
    asamp = active_samples(c, px_per_ch)
    grid = np.full(len(amplitude_ch)*gs, np.pi, dtype=float)
    rng = rng or np.random.default_rng()
    flat = (encoding == 'flat' and edge_taper_px == 0)
    weights = None if flat else edge_taper_weights(px_per_ch, encoding, edge_taper_px)
    for i, amplitude in enumerate(amplitude_ch):
        if flat:
            phi = float(phi_for_amplitude(np.array([amplitude]))[0])
            if quantize:
                phi = float(quantize_channel_phases(c, np.array([phi]))[0])
            if flicker:
                phi = phi + rng.normal(0, c.sig_flicker)
            block = np.full(asamp, phi)
        else:
            active_amp = _active_amplitude(encoding, amplitude, weights, px_per_ch)
            phi_px = phi_for_amplitude(active_amp)
            if quantize:
                phi_px = quantize_channel_phases(c, phi_px)
            if flicker:
                phi_px = phi_px + rng.normal(0, c.sig_flicker, len(phi_px))
            block = np.repeat(phi_px, c.ovs)
        start = i*gs + off
        grid[start:start+asamp] = block
    return grid


def encoding_to_phase(c: Cfg, x_ch: np.ndarray, y_ch: np.ndarray, px_per_ch=5, guard=0,
                      quantize=False, flicker=False, rng=None,
                      encoding='flat', edge_taper_px=None):
    return phase_from_targets(
        c, np.concatenate([x_ch, y_ch]), px_per_ch, guard, quantize, flicker, rng,
        encoding, edge_taper_px
    )


def amplitude_from_targets(c: Cfg, amplitude_ch: np.ndarray, px_per_ch=5, guard=0,
                           quantize=False, flicker=False, rng=None,
                           encoding='flat', edge_taper_px=None):
    phase_grid = phase_grid_from_targets(
        c, amplitude_ch, px_per_ch, guard, quantize, flicker, rng,
        encoding, edge_taper_px
    )
    return amplitude_from_phase(c, phase_grid)


intensity_from_targets = amplitude_from_targets


def single_simulation(c: Cfg, x_ch: np.ndarray, y_ch: np.ndarray, px_per_ch=5, guard=0,
                      quantize=False, flicker=False, rng=None,
                      encoding='flat', edge_taper_px=None):
    return amplitude_from_targets(
        c, np.concatenate([x_ch, y_ch]), px_per_ch, guard, quantize, flicker, rng,
        encoding, edge_taper_px
    )
