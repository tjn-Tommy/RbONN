from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .calibration.calibration_new import CalibrationResult


class EncodingStrategy(Protocol):
    name: str

    def encode(self, values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        ...


class TPAEncodingStub:
    name = "TPA Multiplication"

    def encode(self, values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        raise NotImplementedError("TPA multiplication encoding is not implemented yet")


@dataclass
class EncodingChannel:
    index: int
    side: str           # 'x' or 'w'
    x_center: int
    x_start: int        # inclusive
    x_end: int          # exclusive
    wavelength_nm: float
    # measured transfer curve from the nearest calibration coordinate
    levels: np.ndarray = field(repr=False)           # SLM levels swept (ascending)
    intensity_curve: np.ndarray = field(repr=False)  # measured normalised power

    # derived in __post_init__
    on_level: int = field(init=False)    # level of maximum measured output
    off_level: int = field(init=False)   # level of minimum measured output
    _seg_levels: np.ndarray = field(init=False, repr=False)
    _seg_curve: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        on_idx = int(np.argmax(self.intensity_curve))
        off_idx = int(np.argmin(self.intensity_curve))
        self.on_level = int(self.levels[on_idx])
        self.off_level = int(self.levels[off_idx])

        # rising segment between off (min) and on (max), made monotonic
        # non-decreasing with a cumulative-max envelope so measurement noise
        # near the flat top cannot map a higher target onto a lower level
        lo, hi = sorted((off_idx, on_idx))
        self._seg_levels = self.levels[lo : hi + 1]
        self._seg_curve = np.maximum.accumulate(self.intensity_curve[lo : hi + 1])

    def level_for(self, val: float) -> int:
        """Map a normalised output power val in [0, 1] to an SLM level.

        Nearest-neighbour lookup on the *measured* transfer curve. The target
        output is  val * (max - min)  above the channel's minimum, and we pick
        the swept level whose measured output is closest to that target. The
        curve is taken over the off->on segment with a monotonic envelope, so
        the mapping is non-decreasing: val = 0 -> off_level, val = 1 -> on_level.
        """
        val = float(np.clip(val, 0.0, 1.0))
        off_p = float(self._seg_curve[0])
        on_p = float(self._seg_curve[-1])
        target = off_p + val * (on_p - off_p)
        idx = int(np.argmin(np.abs(self._seg_curve - target)))
        return int(self._seg_levels[idx])


@dataclass
class ChannelLayout:
    x_channels: list[EncodingChannel]  # wavelength > center_wl, x[0] nearest centre
    w_channels: list[EncodingChannel]  # wavelength < center_wl, w[0] nearest centre
    center_wl: float
    center_x: float
    channel_width_px: int
    pitch_px: int
    nm_per_px: float                    # |wavelength slope| of the x->lambda fit
    # sorted calibration coordinates + their off levels, used to build a
    # per-column background so padding columns sit at their local off level
    calib_coords: np.ndarray = field(repr=False)
    calib_off_levels: np.ndarray = field(repr=False)

    @property
    def all_channels(self) -> list[EncodingChannel]:
        return self.x_channels + self.w_channels

    @property
    def n_channels(self) -> int:
        return len(self.x_channels)

    def background_for_columns(self, slm_width: int) -> np.ndarray:
        """Per-column off level: each column takes the nearest cali coord's off.

        Columns outside the calibrated x-range clamp to the nearest edge value.
        """
        cols = np.arange(slm_width)
        nearest = _nearest_index_sorted(self.calib_coords, cols)
        return self.calib_off_levels[nearest].astype(np.uint16)


def _nearest_index_sorted(sorted_values: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Index of the nearest entry in sorted_values for each query (vectorised)."""
    idx = np.searchsorted(sorted_values, queries)
    idx = np.clip(idx, 1, len(sorted_values) - 1)
    left = sorted_values[idx - 1]
    right = sorted_values[idx]
    choose_left = (queries - left) <= (right - queries)
    return np.where(choose_left, idx - 1, idx)


def build_channel_layout(
    calib: CalibrationResult,
    *,
    n_channels: int = 20,
    channel_width_px: int = 15,
    gap_px: int = 5,
    center_wl: float = 778.0,
) -> ChannelLayout:
    """Build a 2*n_channels encoding layout centred at center_wl.

    x-channels: wavelength > center_wl (lower SLM x values).
    w-channels: wavelength < center_wl (higher SLM x values).
    x[0] and w[0] are the pair closest to the centre wavelength.

    Each encoding channel is snapped to the nearest calibration coordinate and
    carries that coordinate's full measured transfer curve, which the encoder
    inverts directly (nearest-neighbour on measured power). gap_px is the
    padding *between* adjacent channels, so pitch = channel_width_px + gap_px.
    """
    if calib.intensity_levels is None:
        raise ValueError("CalibrationResult has no intensity data (Step 3 not run)")

    coords  = np.asarray(calib.coordinates,      dtype=float)
    wls     = np.asarray(calib.wavelength,       dtype=float)
    intens  = np.asarray(calib.intensity_levels, dtype=float)  # (n_calib, n_levels)
    levels  = np.asarray(calib.level_range,      dtype=int)

    # sort calibration by coordinate so nearest-column lookup can use searchsorted
    order   = np.argsort(coords)
    coords  = coords[order]
    intens  = intens[order]

    off_per_coord = levels[np.argmin(intens, axis=1)]   # (n_calib,) for background

    # wl = a*x + b  (a < 0: higher x -> lower wavelength)
    a, b = np.polyfit(np.asarray(calib.coordinates, dtype=float),
                      np.asarray(calib.wavelength, dtype=float), 1)
    center_x = (center_wl - b) / a

    pitch_px = channel_width_px + gap_px
    half_w   = channel_width_px // 2

    def _make(index: int, side: str, x_c: int) -> EncodingChannel:
        nearest = int(np.argmin(np.abs(coords - x_c)))
        return EncodingChannel(
            index=index,
            side=side,
            x_center=x_c,
            x_start=x_c - half_w,
            x_end=x_c - half_w + channel_width_px,
            wavelength_nm=float(a * x_c + b),
            levels=levels.copy(),
            intensity_curve=intens[nearest].copy(),
        )

    x_channels = [
        _make(i, "x", int(round(center_x - (i + 0.5) * pitch_px)))
        for i in range(n_channels)
    ]
    w_channels = [
        _make(i, "w", int(round(center_x + (i + 0.5) * pitch_px)))
        for i in range(n_channels)
    ]

    return ChannelLayout(
        x_channels=x_channels,
        w_channels=w_channels,
        center_wl=center_wl,
        center_x=center_x,
        channel_width_px=channel_width_px,
        pitch_px=pitch_px,
        nm_per_px=abs(float(a)),
        calib_coords=coords,
        calib_off_levels=off_per_coord,
    )


def encode_to_pattern(
    x_vals: np.ndarray,
    w_vals: np.ndarray,
    layout: ChannelLayout,
    slm_width: int,
    slm_height: int,
) -> np.ndarray:
    """Map value arrays onto an SLM grayscale pattern.

    x_vals / w_vals: each layout.n_channels floats in [0, 1].
    Background/padding columns are set to the off level of their nearest
    calibration coordinate; each channel band is set to level_for(val).
    """
    x_vals = np.asarray(x_vals, dtype=float)
    w_vals = np.asarray(w_vals, dtype=float)
    n = layout.n_channels
    if x_vals.shape != (n,) or w_vals.shape != (n,):
        raise ValueError(f"x_vals and w_vals must each have {n} elements")

    # per-column off-level background, broadcast across all rows
    bg_row = layout.background_for_columns(slm_width)
    pattern = np.broadcast_to(bg_row, (slm_height, slm_width)).copy()

    for ch, val in list(zip(layout.x_channels, x_vals)) + list(zip(layout.w_channels, w_vals)):
        level = ch.level_for(val)
        x0 = max(0, ch.x_start)
        x1 = min(slm_width, ch.x_end)
        if x0 < x1:
            pattern[:, x0:x1] = level

    return pattern
