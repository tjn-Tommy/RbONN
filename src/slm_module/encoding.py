from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

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
    *,
    col_ratio: np.ndarray | None = None,
    level_trim: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Map value arrays onto an SLM grayscale pattern.

    x_vals / w_vals: each layout.n_channels floats in [0, 1].
    Background/padding columns are set to the off level of their nearest
    calibration coordinate.

    col_ratio: optional per-column ratio profile of length
        ``layout.channel_width_px`` (values in [0, 1]), applied *multiplicatively*
        to the channel value so column j of every channel encodes
        ``level_for(val * col_ratio[j])``. Because ``level_for`` maps through the
        measured transfer curve (target = off + v*(on-off)), this realises
        ``edge = ratio * (max - min) + min`` with the floor being the channel's
        *measured* off level (ratio -> 0 sits the column at the measured
        background, not literal zero). ``None`` -> uniform 1.0, i.e. the flat band
        used before this profile existed (byte-identical output). Ratios are
        normalised intensity ratios, not field-amplitude ratios.

    level_trim: optional callable applied to each channel's per-column level
        vector after ``level_for``, before it is written into the pattern;
        ``None`` -> identity. It is independent of the OSA intensity-profile
        optimiser.
    """
    x_vals = np.asarray(x_vals, dtype=float)
    w_vals = np.asarray(w_vals, dtype=float)
    n = layout.n_channels
    if x_vals.shape != (n,) or w_vals.shape != (n,):
        raise ValueError(f"x_vals and w_vals must each have {n} elements")

    width = int(layout.channel_width_px)
    if col_ratio is None:
        ratios = np.ones(width, dtype=float)
    else:
        ratios = np.asarray(col_ratio, dtype=float)
        if ratios.shape != (width,):
            raise ValueError(
                f"col_ratio must have {width} elements (channel_width_px)"
            )
        ratios = np.clip(ratios, 0.0, 1.0)

    # per-column off-level background, broadcast across all rows
    bg_row = layout.background_for_columns(slm_width)
    pattern = np.broadcast_to(bg_row, (slm_height, slm_width)).copy()

    for ch, val in list(zip(layout.x_channels, x_vals)) + list(zip(layout.w_channels, w_vals)):
        col_levels = np.array(
            [ch.level_for(float(val) * float(r)) for r in ratios], dtype=np.uint16
        )
        if level_trim is not None:
            col_levels = np.clip(level_trim(col_levels), 0, 1023).astype(np.uint16)
        # profile index -> absolute column, honouring x-range clipping
        x0 = max(0, ch.x_start)
        x1 = min(slm_width, ch.x_end)
        if x0 < x1:
            off = x0 - ch.x_start
            pattern[:, x0:x1] = col_levels[off:off + (x1 - x0)]

    return pattern


def optimize_from_osa(
    layout: ChannelLayout,
    trace=None,
    *,
    col_ratio: np.ndarray | None = None,
    level_trim: Callable[[np.ndarray], np.ndarray] | None = None,
    osa=None,
    slm=None,
    initial_l: np.ndarray | None = None,
    config=None,
    stop_event=None,
    progress_callback=None,
    **kwargs,
):
    """Run the live two-stage OSA optimisation.

    ``initial_l`` is the independent half-profile and always contains
    normalised *intensity* ratios (eight values for the required 15-pixel
    channel).  For compatibility with the Edge Ratio UI, a symmetric full
    ``col_ratio`` may be supplied instead.  The optimiser does not load an
    initial profile from a model or file.

    A downloaded ``trace`` is insufficient because each COBYQA evaluation
    requires a new SLM pattern and OSA sweep; callers must provide live ``osa``
    and ``slm`` controllers.
    """
    if kwargs:
        names = ", ".join(sorted(kwargs))
        raise TypeError(f"unexpected optimisation arguments: {names}")
    if trace is not None:
        raise ValueError("live optimisation does not accept a pre-recorded trace")
    if level_trim is not None:
        raise ValueError("per-level trim is not part of the intensity-profile plan")
    if osa is None or slm is None:
        raise ValueError("live OSA and SLM controllers are required")

    from .optimization import (
        independent_intensity_profile,
        run_osa_optimization,
    )

    if initial_l is None:
        if col_ratio is None:
            raise ValueError("an eight-value initial intensity profile is required")
        initial_l = independent_intensity_profile(col_ratio)
    return run_osa_optimization(
        osa,
        slm,
        layout,
        initial_l,
        config=config,
        stop_event=stop_event,
        progress_callback=progress_callback,
    )
