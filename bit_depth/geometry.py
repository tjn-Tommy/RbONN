"""Channel geometry: how active pixels and blank guards are laid out per group.

A channel occupies ``px_per_ch`` active pixels plus ``guard`` blank pixels on
each side. ``guard`` may be fractional (e.g. 2.5 px for a 15-px window on a
20-px pitch); the sample-level helpers below handle that by working on the
oversampled grid (``ovs`` samples per pixel). For integer guard the layout is
identical to the historical ``np.repeat(per_pixel_phase, ovs)`` construction.
"""
import numpy as np

from .config import Cfg
from .optics import phi_for_amplitude


def channel_group_px(px_per_ch=5, guard=0):
    """Total pixels per encoded channel: active pixels plus blank guards.

    Returns an int when the total is whole (the common case, including the
    2.5-px guard / 15-px window which totals 20 px), else a float.
    """
    px_per_ch = int(px_per_ch)
    guard = float(guard)
    if px_per_ch <= 0:
        raise ValueError('px_per_ch must be positive')
    if guard < 0:
        raise ValueError('guard must be non-negative')
    total = px_per_ch + 2*guard
    return int(total) if float(total).is_integer() else total


def group_samples(c: Cfg, px_per_ch=5, guard=0):
    """Oversampled-grid samples spanning one channel group."""
    return int(round(channel_group_px(px_per_ch, guard)*c.ovs))


def guard_samples(c: Cfg, guard=0):
    """Oversampled-grid samples in one edge guard band."""
    return int(round(float(guard)*c.ovs))


def active_samples(c: Cfg, px_per_ch=5):
    """Oversampled-grid samples spanning the active window of one channel."""
    return int(px_per_ch)*c.ovs


def active_slice(c: Cfg, px_per_ch=5, guard=0):
    start = guard_samples(c, guard)
    stop = start + active_samples(c, px_per_ch)
    return slice(start, stop)


def quantize_channel_phases(c: Cfg, phi_ch: np.ndarray):
    dphi = 2*np.pi/c.n_levels
    return np.round(phi_ch/dphi)*dphi


def phase_from_channel_phases(phi_ch: np.ndarray, px_per_ch=5, guard=0):
    """Per-pixel commanded phase from one phase value per channel.

    Requires integer guard (the per-pixel representation cannot place a
    fractional number of blank pixels); fractional guard is handled on the grid
    by :func:`bit_depth.encoding.phase_grid_from_targets`.
    """
    phi_ch = np.asarray(phi_ch, dtype=float)
    if not float(guard).is_integer():
        raise ValueError('phase_from_channel_phases requires integer guard; '
                         'use phase_grid_from_targets for fractional guard')
    guard = int(guard)
    group_px = int(channel_group_px(px_per_ch, guard))
    phase = np.full(len(phi_ch)*group_px, np.pi, dtype=float)
    for i, phi in enumerate(phi_ch):
        start = i*group_px + guard
        phase[start:start+px_per_ch] = phi
    return phase
