"""Hardware interface: encode 20 weight phases onto the SLM.

Scope readout is NOT implemented here — that branch is under development.
Fill in ``_read_scope`` once it is merged.

Spatial encoding layout
-----------------------
The SLM is partitioned into N_IN vertical strips (one per PCA channel).
Strip i spans columns [i * strip_w, (i+1) * strip_w) across the full SLM
height.  Each strip is set to the grayscale level corresponding to φ_w[i].

This layout is a placeholder: adjust ``ChannelLayout.pixel_region`` to match
the actual optical alignment once characterised.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


# ── SLM spatial layout ──────────────────────────────────────────────────────

@dataclass
class ChannelLayout:
    """Maps PCA channel index to a rectangular SLM pixel region.

    Parameters
    ----------
    slm_height, slm_width : SLM pixel dimensions
    n_channels            : number of input channels (= 20)
    """
    slm_height: int
    slm_width: int
    n_channels: int = 20

    def pixel_region(self, channel: int) -> tuple[slice, slice]:
        """(row_slice, col_slice) for channel i — full-height vertical strip."""
        strip_w = self.slm_width // self.n_channels
        c0 = channel * strip_w
        c1 = c0 + strip_w
        return slice(0, self.slm_height), slice(c0, c1)


def phases_to_grayscale(phi: np.ndarray, n_bits: int = 10) -> np.ndarray:
    """Wrap φ to [0, 2π] then quantise to SLM grayscale levels [0, 2^n_bits - 1]."""
    phi_w = phi % (2.0 * math.pi)
    max_level = (2 ** n_bits) - 1
    return np.round(phi_w / (2.0 * math.pi) * max_level).astype(np.uint16)


def build_slm_pattern(
    phi_w: np.ndarray,
    layout: ChannelLayout,
    n_bits: int = 10,
) -> np.ndarray:
    """Build the SLM grayscale pattern from 20 weight phases.

    Parameters
    ----------
    phi_w   : (n_channels,) weight phases in radians
    layout  : channel → pixel-region mapping
    n_bits  : SLM bit depth (Santec: 10 bits)

    Returns
    -------
    (slm_height, slm_width) uint16 array ready for SLMController.display_array()
    """
    pattern = np.zeros((layout.slm_height, layout.slm_width), dtype=np.uint16)
    gray = phases_to_grayscale(phi_w, n_bits=n_bits)
    for i, g in enumerate(gray):
        row_sl, col_sl = layout.pixel_region(i)
        pattern[row_sl, col_sl] = int(g)
    return pattern


# ── Scope stub ───────────────────────────────────────────────────────────────

def _read_scope(scope: Any) -> float:
    """Read 420 nm intensity as a scalar voltage from the scope.

    TODO: implement when scope interface branch is merged.
          Expected return: float in [0, V_max] representing the 420 nm signal.
    """
    raise NotImplementedError(
        "Scope readout not yet implemented. "
        "Wire the scope interface here once that branch is ready."
    )


# ── Single-shot measurement ───────────────────────────────────────────────────

def run_inference_single(
    phi_w: np.ndarray,
    slm_controller: Any,
    scope: Any,
    layout: ChannelLayout,
    settle_s: float = 0.05,
) -> float:
    """Set SLM weight pattern, wait for settle, return scope reading.

    Parameters
    ----------
    phi_w          : (n_channels,) weight phases in radians
    slm_controller : SLMController from src/slm_module/controller.py
    scope          : scope interface (TODO)
    layout         : SLM channel layout
    settle_s       : seconds to wait after SLM update before reading

    Returns
    -------
    float — 420 nm intensity (scope voltage)
    """
    pattern = build_slm_pattern(phi_w, layout)
    slm_controller.display_array(pattern)
    time.sleep(settle_s)
    return _read_scope(scope)
