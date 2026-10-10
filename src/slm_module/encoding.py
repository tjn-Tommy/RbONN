from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from calibration_module.fit.slm_response import CalibrationResult
from calibration_module.fit.transfer import TransferFit, TransferFitError, fit_transfer_curve


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
    # Measured normalised power.  Empty is allowed when a ``transfer_fit`` is
    # supplied: the raw sweep lives in the step-3 CSV and no longer in
    # calib.json, and inverting the fit does not need it.  Kept for plots and
    # for fitting a hand-built channel that arrives without one.
    intensity_curve: np.ndarray = field(
        repr=False, default_factory=lambda: np.asarray([], dtype=float))

    # The fitted sin^2 model level_for() inverts
    # (calibration_module.fit.transfer).  Full scale is the fitted Delta = pi
    # rather than the swept argmax, every level is informed by all the swept
    # points rather than the two bracketing it, and `v` means sin^2(Delta/2)
    # exactly -- the same relation calibration_module.fit.phase.phi_half assumes
    # downstream.  Fitted here when absent, which is what a hand-built channel
    # and the alignment builders rely on; a step-3 section supplies it already
    # fitted, so the encoding is frozen with the calibration.
    transfer_fit: TransferFit | None = None

    # derived in __post_init__
    on_level: int = field(init=False)    # level of full scale (v = 1)
    off_level: int = field(init=False)   # level of extinction (v = 0)

    def __post_init__(self) -> None:
        curve = np.asarray(self.intensity_curve, dtype=float).reshape(-1)
        if self.transfer_fit is None:
            if curve.size == 0:
                raise TransferFitError(
                    f"channel {self.side}{self.index} has neither a measured "
                    f"curve nor a transfer fit -- nothing to encode through"
                )
            self.transfer_fit = fit_transfer_curve(self.levels, self.intensity_curve)

        # Full scale is the model's Delta = pi, extinction its Delta = 0 -- NOT
        # the swept argmax/argmin, which is the whole point (one noisy sample
        # moved the 0903 argmax of one channel by 120 levels).
        self.on_level = self.transfer_fit.on_level
        self.off_level = self.transfer_fit.off_level

    def level_for(self, val: float) -> int:
        """Invert the fitted sin^2 transfer model -- see :mod:`calibration_module.fit.transfer`.

        ``sin^2(Delta/2) = val`` gives ``Delta = 2 asin(sqrt(val))``, exactly
        the retardance ``calibration_module.fit.phase.phi_half`` assumes for a
        channel commanded at ``val``; the linear ``Delta(L)`` then gives the
        level.  So the amplitude this writes and the phase the step-6/7/8 model
        assumes come from one fit, which interpolation cannot promise.
        """
        if self.transfer_fit is None:
            raise TransferFitError(
                f"channel {self.side}{self.index} has no transfer fit to invert"
            )
        return self.transfer_fit.level_for(val)


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
    # inclusive px ranges forced dark (never covered by a channel): the two
    # wavelength guard bands plus the centre divider column. Kept for
    # inspection / preview; the renderer leaves them at their off level anyway.
    dark_px_ranges: list[tuple[int, int]] = field(default_factory=list, repr=False)
    # requested centre dark-pad width; None -> legacy gap_px-wide pad
    center_gap_px: int | None = None

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


@dataclass
class ChannelGeometry:
    """Pixel span + wavelength of one channel, without any measured curve."""
    index: int
    side: str           # 'x' (wl > center) or 'w' (wl < center)
    x_center: int
    x_start: int        # inclusive
    x_end: int          # exclusive
    wavelength_nm: float


@dataclass
class LayoutGeometry:
    """Pure geometry of a channel layout (no transfer curves attached).

    Shared by build_channel_layout -- which snaps each channel to a calibration
    coordinate and attaches its measured transfer curve -- and by UI previews
    that only have the Step-2 wavelength map and want to show the pixel/channel
    layout before any Step-3 intensity data exists.
    """
    x: list[ChannelGeometry]            # x[0] nearest the centre
    w: list[ChannelGeometry]            # mirror of x[i] about the centre column
    center_wl: float
    center_x: float                     # exact fractional centre pixel
    c0: int                             # rounded centre pixel (pad midpoint)
    channel_width_px: int
    pitch_px: int
    nm_per_px: float                    # |slope| of the x->wavelength fit
    dark_px_ranges: list[tuple[int, int]]  # inclusive guard px ranges, always dark
    # requested centre dark-pad width; None -> legacy gap_px-wide pad
    center_gap_px: int | None = None

    @property
    def n_channels(self) -> int:
        return len(self.x)


def compute_channel_geometry(
    coordinates: np.ndarray,
    wavelengths: np.ndarray,
    *,
    n_channels: int = 20,
    channel_width_px: int = 15,
    gap_px: int = 5,
    center_gap_px: int | None = None,
    center_wl: float = 778.0,
    dark_wl_bands: tuple[tuple[float, float], ...] = (
        (779.9, 780.1),
        (775.9, 776.1),
    ),
) -> LayoutGeometry:
    """Tile symmetric channel pairs around center_wl from a Step-2 wl<->px map.

    This is the geometry half of build_channel_layout (steps 1-5 of its
    docstring). It needs only the coordinate -> wavelength mapping, so a UI can
    preview the layout from a Step-2 result before any Step-3 sweep has run.

    ``center_gap_px`` widens the central dark pad (between the innermost x/w
    pair) to at least that many pixels, reducing crosstalk between the two
    innermost spectral bands. ``None`` keeps the legacy behaviour where the pad
    is exactly ``gap_px`` wide (tiling starts at half a pitch). The pitch
    between neighbouring channels on the same side is unaffected.
    """
    coords = np.asarray(coordinates, dtype=float).reshape(-1)
    wl = np.asarray(wavelengths, dtype=float).reshape(-1)
    if coords.size < 2 or coords.size != wl.size:
        raise ValueError("need at least two matching coordinate/wavelength points")
    if center_gap_px is not None and int(center_gap_px) < 0:
        raise ValueError("center_gap_px must be non-negative")

    # 1. wl = a*x + b  (a < 0: higher x -> lower wavelength)
    a, b = np.polyfit(coords, wl, 1)
    if a == 0.0 or not np.isfinite(a) or not np.isfinite(b):
        raise ValueError("degenerate wavelength fit (need a sloped x->wl map)")

    # 2. centre anchor. Round to an integer pixel so x/w pairs placed at
    # c0 -/+ m are exactly mirror-symmetric about the centre column (and thus,
    # under the linear fit, symmetric in wavelength about center_wl). The centre
    # sits in the middle of a pad, so no channel covers it.
    center_x = (center_wl - b) / a
    c0 = int(round(center_x))

    pitch_px = int(channel_width_px) + int(gap_px)
    half_w   = int(channel_width_px) // 2
    coord_lo = int(np.ceil(coords.min()))
    coord_hi = int(np.floor(coords.max()))

    # 3. wavelength guard bands -> inclusive px ranges that must stay dark
    def _wl_to_px(w: float) -> float:
        return (w - b) / a

    guard_ranges: list[tuple[int, int]] = []
    for lo_wl, hi_wl in dark_wl_bands:
        p1, p2 = _wl_to_px(lo_wl), _wl_to_px(hi_wl)
        guard_ranges.append((int(np.floor(min(p1, p2))), int(np.ceil(max(p1, p2)))))

    def _clear_offset(m: int) -> int:
        """Smallest offset >= m whose x-channel (c0-m) AND w-channel (c0+m)
        windows both clear every guard band.

        A single shared offset keeps the pair mirror-symmetric: whenever either
        side's window would cover a guard, m is pushed outward past it (using the
        larger requirement of the two sides). Each push strictly increases m past
        a guard, so this terminates.
        """
        while True:
            need = m
            for lo, hi in guard_ranges:
                # w side, centre c0 + m: clear when x_start >= hi + 1
                if (c0 + m - half_w) <= hi and (c0 + m + half_w) >= lo:
                    need = max(need, hi + 1 + half_w - c0)
                # x side, centre c0 - m: clear when x_end - 1 <= lo - 1
                if (c0 - m - half_w) <= hi and (c0 - m + half_w) >= lo:
                    need = max(need, c0 - lo + half_w + 1)
            if need == m:
                return m
            m = need

    def _geo(index: int, side: str, x_c: int) -> ChannelGeometry:
        x_start = x_c - half_w
        return ChannelGeometry(
            index=index,
            side=side,
            x_center=x_c,
            x_start=x_start,
            x_end=x_start + int(channel_width_px),
            wavelength_nm=float(a * x_c + b),
        )

    # 4/5. Tile a single shared offset outward from the centre pad (half-pitch
    # start). Each step places a mirror pair c0-m (x, wl > center) and c0+m (w,
    # wl < center). If either window would cover a guard band, m jumps past it
    # (both sides move together, so the pair stays symmetric); tiling then
    # resumes from the jumped offset. Stops when either side leaves the
    # calibrated range, keeping the two sides equal length.
    x_geo: list[ChannelGeometry] = []
    w_geo: list[ChannelGeometry] = []
    if center_gap_px is None:
        offset = 0.5 * pitch_px          # legacy: centre pad exactly gap_px wide
    else:
        # First offset m0 so the centre dark pad (2*m0 - width) is at least
        # center_gap_px wide. Integer ceil avoids the truncation that
        # ``center_gap_px // 2 + half_w`` would suffer for odd widths.
        offset = float((int(channel_width_px) + int(center_gap_px) + 1) // 2)
    while len(x_geo) < n_channels:
        m = int(round(offset))
        cleared = _clear_offset(m)
        if cleared != m:
            m = cleared
            offset = float(m)
        if (c0 - m - half_w) < coord_lo or (c0 + m + half_w) > coord_hi:
            break
        idx = len(x_geo)
        x_geo.append(_geo(idx, "x", c0 - m))
        w_geo.append(_geo(idx, "w", c0 + m))
        offset += pitch_px

    return LayoutGeometry(
        x=x_geo,
        w=w_geo,
        center_wl=center_wl,
        center_x=center_x,
        c0=c0,
        channel_width_px=int(channel_width_px),
        pitch_px=pitch_px,
        nm_per_px=abs(float(a)),
        dark_px_ranges=list(guard_ranges),
        center_gap_px=None if center_gap_px is None else int(center_gap_px),
    )


def build_channel_layout(
    calib: CalibrationResult,
    *,
    n_channels: int = 20,
    channel_width_px: int = 15,
    gap_px: int = 5,
    center_gap_px: int | None = None,
    center_wl: float = 778.0,
    dark_wl_bands: tuple[tuple[float, float], ...] = (
        (779.9, 780.1),
        (775.9, 776.1),
    ),
) -> ChannelLayout:
    """Build an encoding layout centred at center_wl with dark guard bands.

    This builder TILES a grid and gives each channel the curve of its NEAREST
    calibration coordinate, so one measured curve can back several channels.
    That sharing is the approximation here, and it is unaffected by how the
    curve is inverted: channels snapped to the same row get the same mapping
    either way.  Inversion is the fitted sin^2 model, the only one there is --
    see :meth:`EncodingChannel.level_for`.

    .. deprecated::
        Do NOT use this to consume a Step-3b/3c channel calibration -- those
        rows already ARE the channels; load them verbatim with
        :func:`channel_layout_from_calibration` (what the GUI, the pipeline
        stages and the draft scripts do).  Re-tiling here only remains for
        cases where no measured channel grid exists yet: the coarse Step-1+2
        preview and the tpa_center scan, which must move the layout centre.

    Geometry (see the numbered plan this implements):
      1. Fit wl = a*x + b over the Step-2 mapping (a < 0: higher x -> lower wl).
      2. Anchor the centre pixel c0 = round((center_wl - b) / a). The centre sits
         in the middle of a pad (a ``gap_px``-wide dark gap straddling c0), not on
         a channel edge, so no channel covers it.
      3. Convert each ``dark_wl_bands`` wavelength range to an inclusive px range;
         those columns must stay dark (they are the Rb guard lines, ~780 & ~776).
      4. Tile a single shared offset m outward from c0 with a half-pitch start
         (pitch = width + gap), placing at each step a mirror pair:
             x-channel (wl > center_wl) centred at c0 - m,
             w-channel (wl < center_wl) centred at c0 + m.
         Because both use the same m, x[i] and w[i] are exactly symmetric about
         the centre column -- and, under the linear fit, about center_wl.
      5. If either window of a pair would cover a guard band, m jumps outward
         past it (both sides move together, so the pair stays symmetric) and
         channels end up on both sides of the Rb line; tiling resumes from the
         jumped offset. Tiling stops when either side leaves the calibrated
         range, so the two sides are always equal length (the encoder's x/w
         pairing contract).

    Each kept channel snaps to the nearest calibration coordinate and carries
    that coordinate's measured transfer curve, which the encoder inverts.
    ``n_channels`` is the maximum per side; fewer are returned when the guard
    bands or the calibrated range cut the tiling short.
    """
    if calib.intensity_levels is None:
        raise ValueError("CalibrationResult has no intensity data (Step 3 not run)")

    coords  = np.asarray(calib.coordinates,      dtype=float)
    intens  = np.asarray(calib.intensity_levels, dtype=float)  # (n_calib, n_levels)
    levels  = np.asarray(calib.level_range,      dtype=int)

    # sort calibration by coordinate so nearest-column lookup can use searchsorted
    order   = np.argsort(coords)
    coords  = coords[order]
    intens  = intens[order]

    off_per_coord = levels[np.argmin(intens, axis=1)]   # (n_calib,) for background

    geom = compute_channel_geometry(
        np.asarray(calib.coordinates, dtype=float),
        np.asarray(calib.wavelength, dtype=float),
        n_channels=n_channels,
        channel_width_px=channel_width_px,
        gap_px=gap_px,
        center_gap_px=center_gap_px,
        center_wl=center_wl,
        dark_wl_bands=dark_wl_bands,
    )

    # One fit per calibration row, shared by every channel that snaps to it.
    row_fits = [fit_transfer_curve(levels, intens[row])
                for row in range(intens.shape[0])]

    def _make(g: ChannelGeometry) -> EncodingChannel:
        nearest = int(np.argmin(np.abs(coords - g.x_center)))
        return EncodingChannel(
            index=g.index,
            side=g.side,
            x_center=g.x_center,
            x_start=g.x_start,
            x_end=g.x_end,
            wavelength_nm=g.wavelength_nm,
            levels=levels.copy(),
            intensity_curve=intens[nearest].copy(),
            transfer_fit=row_fits[nearest],
        )

    return ChannelLayout(
        x_channels=[_make(g) for g in geom.x],
        w_channels=[_make(g) for g in geom.w],
        center_wl=center_wl,
        center_x=geom.center_x,
        channel_width_px=geom.channel_width_px,
        pitch_px=geom.pitch_px,
        nm_per_px=geom.nm_per_px,
        calib_coords=coords,
        calib_off_levels=off_per_coord,
        dark_px_ranges=list(geom.dark_px_ranges),
        center_gap_px=geom.center_gap_px,
    )


def channel_layout_from_calibration(
    calib: CalibrationResult,
    *,
    channel_width_px: int | None = None,
    assumed_gap_px: int = 5,
    warn: bool = True,
) -> ChannelLayout:
    """Rebuild a ChannelLayout verbatim from a channels-only calibration.

    A Step-3b/3c result already *is* the channel structure: each row was
    measured at one encoding-channel centre of a mirror-symmetric grid, so the
    target centre, pitch and guard skips are all baked into ``coordinates``.
    This loader takes those rows as the channels directly -- no re-tiling, no
    guard-band rebuild, no nearest-coordinate snapping -- so the encoder always
    agrees with what was actually measured.

    How the pitch splits into window + pad comes from the calibration itself:
    Step 3 stores the geometry it swept with in ``channel_width_px`` /
    ``gap_px``, and this loader drives that, so the encoded aperture is the one
    the transfer curves were measured through.  ``channel_width_px`` here
    overrides the stored value; a file written before the split was recorded has
    neither, and only then does the width fall back to the old
    ``pitch - assumed_gap_px`` guess -- announced under ``warn``, because a
    guessed aperture is not the calibrated one.

    ``level_for`` inverts the fitted sin^2 model of
    :mod:`calibration_module.fit.transfer` -- the only inversion there is.  This
    is the one loader every offline calibration step goes through (steps 6, 7
    and 8 all call it), so it is where the pipeline's encoding convention is set.

    Fitted parameters come from ``calib.transfer_fits`` when the calibration
    file carries them, which is what step 3 writes: the encoding is then frozen
    with the calibration, so steps 6, 7 and 8 -- run at different times against
    the same file -- cannot drift apart because an optimizer took a different
    path.  A calib.json step-3 section carries ONLY the fits, its raw sweep
    having moved to the CSV.  An older file with curves and no fits is fitted
    here instead, which keeps every existing calibration usable.

    ``warn`` prints a one-line notice per channel whose fitted encoding window
    runs past the levels actually swept.  Not an error (a peak just past the top
    of the sweep is still encodable), but its ``v = 1`` is an extrapolation and
    that is worth seeing before a run rather than after.
    """
    coords = np.asarray(calib.coordinates, dtype=float)
    wls = np.asarray(calib.wavelength, dtype=float)
    levels = np.asarray(calib.level_range, dtype=int)
    n = coords.size

    # A calib.json step-3 section stores the fits and leaves the raw sweep in
    # its CSV, so the curves can legitimately be absent.  Then the fits ARE the
    # calibration and there has to be one per channel; without either, there is
    # no transfer curve at all and step 3 was not run.
    stored_fits = getattr(calib, "transfer_fits", None)
    curves_only = calib.intensity_levels is None
    if curves_only and (stored_fits is None or len(stored_fits) != n):
        raise ValueError(
            "CalibrationResult has neither intensity curves nor one transfer "
            f"fit per channel ({n} coordinates, "
            f"{0 if stored_fits is None else len(stored_fits)} fits) "
            "-- step 3 not run"
        )
    intens = (np.zeros((n, levels.size), dtype=float) if curves_only
              else np.asarray(calib.intensity_levels, dtype=float))

    if n < 4 or n % 2:
        raise ValueError(
            f"channel calibration needs an even number of coordinates (>= 4), got {n}"
        )
    if wls.size != n or intens.ndim != 2 or intens.shape[0] != n:
        raise ValueError("wavelength/intensity rows do not match the coordinates")

    order = np.argsort(coords)
    coords, wls, intens = coords[order], wls[order], intens[order]

    # mirror pairs (outermost-in) share one midpoint; a spread means the file
    # is not a symmetric channel grid
    midpoints = 0.5 * (coords + coords[::-1])
    center_x = float(np.mean(midpoints))
    if float(np.max(np.abs(midpoints - center_x))) > 1.0:
        raise ValueError(
            "coordinates are not mirror-symmetric channel pairs (midpoint "
            f"spread {float(np.ptp(midpoints)):.2f} px) -- is this a Step 3b/3c "
            "channel result?"
        )

    diffs = np.diff(coords)
    pitch = int(round(float(np.min(diffs))))
    if pitch < 1:
        raise ValueError("duplicate channel coordinates in the calibration")

    stored_width = getattr(calib, "channel_width_px", None)
    stored_gap = getattr(calib, "gap_px", None)
    if channel_width_px is not None:
        width = int(channel_width_px)
        if width < 1:
            raise ValueError("channel_width_px must be positive")
    elif stored_width is not None:
        width = int(stored_width)
        if width < 1:
            raise ValueError(
                f"calibration records channel_width_px = {stored_width!r}, "
                "which is not a usable window"
            )
        # width + pad IS the pitch by construction; a mismatch means the file
        # was hand-edited or stitched, and the recorded window still wins --
        # it is the aperture the curves were measured through.
        if warn and stored_gap is not None and width + int(stored_gap) != pitch:
            print(
                f"  [encoding] calibration says {width} px window + "
                f"{int(stored_gap)} px pad, but its channels are pitched "
                f"{pitch} px apart -- driving the recorded {width} px window"
            )
    else:
        width = pitch - int(assumed_gap_px)
        if width < 1:
            raise ValueError(
                f"grid pitch {pitch} px is too fine to infer a window width -- "
                "this looks like a dense Step-3 scan, not a channel grid"
            )
        if warn:
            print(
                f"  [encoding] calibration records no window/pad split; "
                f"guessing {width} px window + {int(assumed_gap_px)} px pad "
                f"from the {pitch} px pitch -- re-run or re-save Step 3 to "
                f"record the geometry it actually swept"
            )
    if width > pitch:
        raise ValueError(
            f"channel width {width} px exceeds the {pitch} px grid pitch"
        )

    a, b = np.polyfit(coords, wls, 1)
    if a == 0.0 or not np.isfinite(a) or not np.isfinite(b):
        raise ValueError("degenerate wavelength fit (need a sloped x->wl map)")
    center_wl = float(a * center_x + b)

    # x side = the half with wavelengths above the centre; both sides ordered
    # innermost-first so x[i]/w[i] stay mirror pairs (the encoder contract)
    half = n // 2
    if a < 0:
        x_rows = range(half - 1, -1, -1)
        w_rows = range(half, n)
    else:
        x_rows = range(half, n)
        w_rows = range(half - 1, -1, -1)

    half_w = width // 2

    # One transfer fit per row, in the SAME sorted-by-coordinate order as
    # `intens`.  Stored fits are reordered with the rows; absent ones are
    # fitted here.
    if stored_fits is not None and len(stored_fits) == n:
        fits = [stored_fits[int(i)] for i in order]
    else:
        fits = [fit_transfer_curve(levels, intens[row]) for row in range(n)]
    if warn:
        for row, fit in enumerate(fits):
            if fit.extrapolated:
                print(
                    f"  [encoding] channel at {wls[row]:.3f} nm: fitted "
                    f"encoding window {fit.off_level_exact:.0f}.."
                    f"{fit.on_level_exact:.0f} runs outside the swept "
                    f"{fit.level_min}..{fit.level_max} -- v = 1 is an "
                    f"extrapolation (widen level_range in step 3)"
                )

    def _make(index: int, side: str, row: int) -> EncodingChannel:
        x_c = int(round(coords[row]))
        x_start = x_c - half_w
        return EncodingChannel(
            index=index,
            side=side,
            x_center=x_c,
            x_start=x_start,
            x_end=x_start + width,
            wavelength_nm=float(wls[row]),
            levels=levels.copy(),
            # Empty, not a row of zeros, when the file carries fits only: a zero
            # curve would read as a measured one in any plot that shows it.
            intensity_curve=(np.asarray([], dtype=float) if curves_only
                             else intens[row].copy()),
            transfer_fit=fits[row],
        )

    x_channels = [_make(i, "x", row) for i, row in enumerate(x_rows)]
    w_channels = [_make(i, "w", row) for i, row in enumerate(w_rows)]

    # guard skips show up as larger-than-pitch jumps in the grid; keep the
    # pixels between the flanking windows as dark ranges for inspection (the
    # renderer leaves unlit columns at their off level anyway)
    dark_ranges: list[tuple[int, int]] = []
    centers = np.round(coords).astype(int)
    for left, right, diff in zip(centers[:-1], centers[1:], diffs):
        if int(round(float(diff))) > pitch:
            lo = int(left - half_w + width)
            hi = int(right - half_w - 1)
            if lo <= hi:
                dark_ranges.append((lo, hi))

    return ChannelLayout(
        x_channels=x_channels,
        w_channels=w_channels,
        center_wl=center_wl,
        center_x=center_x,
        channel_width_px=width,
        pitch_px=pitch,
        nm_per_px=abs(float(a)),
        calib_coords=coords,
        # Under "fit" the background sits at each channel's MODELLED
        # extinction (Delta = 0), not at the darkest swept sample, so the dark
        # columns and the encoded v = 0 agree.
        calib_off_levels=(
            np.array([f.off_level for f in fits], dtype=int)
            if fits is not None
            else levels[np.argmin(intens, axis=1)]
        ),
        dark_px_ranges=dark_ranges,
        center_gap_px=None,
    )


def interpolate_coordinate_for_wavelength(
    calibration: CalibrationResult,
    wavelength_nm: float,
) -> float:
    """Invert a monotonic Step-2 wavelength map by linear interpolation."""
    coordinates = np.asarray(calibration.coordinates, dtype=float)
    wavelengths = np.asarray(calibration.wavelength, dtype=float)
    if coordinates.ndim != 1 or wavelengths.ndim != 1:
        raise ValueError("Step 2 coordinates and wavelengths must be 1-D arrays")
    if coordinates.size < 2 or coordinates.size != wavelengths.size:
        raise ValueError("at least two matching Step 2 calibration points are required")
    if not np.all(np.isfinite(coordinates)) or not np.all(np.isfinite(wavelengths)):
        raise ValueError("Step 2 calibration contains NaN or infinity")

    order = np.argsort(coordinates)
    coordinates = coordinates[order]
    wavelengths = wavelengths[order]
    if np.any(np.diff(coordinates) <= 0.0):
        raise ValueError("Step 2 coordinates must be unique")
    wavelength_steps = np.diff(wavelengths)
    increasing = bool(np.all(wavelength_steps > 0.0))
    decreasing = bool(np.all(wavelength_steps < 0.0))
    if not (increasing or decreasing):
        raise ValueError(
            "Step 2 wavelength map must be strictly monotonic for interpolation"
        )

    target = float(wavelength_nm)
    if not np.isfinite(target):
        raise ValueError("target wavelength must be finite")
    lower = float(np.min(wavelengths))
    upper = float(np.max(wavelengths))
    if target < lower or target > upper:
        raise ValueError(
            f"target wavelength {target:g} nm is outside the Step 2 range "
            f"{lower:g}..{upper:g} nm"
        )
    if decreasing:
        wavelengths = wavelengths[::-1]
        coordinates = coordinates[::-1]
    return float(np.interp(target, wavelengths, coordinates))


def build_single_anchor_layout(
    wavelength_calibration: CalibrationResult,
    intensity_calibration: CalibrationResult,
    *,
    target_wavelength_nm: float = 778.0,
    channel_width_px: int = 15,
    gap_px: int = 5,
) -> tuple[ChannelLayout, float]:
    """Build a layout whose offset-0 channel is the interpolated target pixel.

    The intensity calibration is intentionally allowed to contain only the
    target coordinate.  Its measured transfer curve is reused by the nearby
    channels needed to form fixed OSA bins; only the target channel is used as
    an optimisation anchor.

    Every channel shares the anchor's fit, so its full scale is one number
    wearing many hats -- the sharing is the approximation, not the inversion.
    This is an alignment layout, not a calibrated one.
    """
    if channel_width_px < 1:
        raise ValueError("channel_width_px must be positive")
    if gap_px < 0:
        raise ValueError("gap_px must be non-negative")
    levels = np.asarray(intensity_calibration.level_range, dtype=int)
    intensity = np.asarray(intensity_calibration.intensity_levels, dtype=float)
    intensity_coordinates = np.asarray(intensity_calibration.coordinates, dtype=float)
    if levels.ndim != 1 or levels.size < 2:
        raise ValueError("quick intensity calibration requires at least two levels")
    if intensity.ndim != 2 or intensity.shape[1] != levels.size:
        raise ValueError("quick intensity calibration has an invalid intensity map")
    if intensity.shape[0] < 1 or intensity.shape[0] != intensity_coordinates.size:
        raise ValueError("quick intensity calibration has no calibrated coordinate")
    if not np.all(np.isfinite(intensity)):
        raise ValueError("quick intensity calibration contains NaN or infinity")

    interpolated_x = interpolate_coordinate_for_wavelength(
        wavelength_calibration, target_wavelength_nm
    )
    center_x = int(round(interpolated_x))
    source_row = int(np.argmin(np.abs(intensity_coordinates - center_x)))
    curve = intensity[source_row].copy()

    map_coordinates = np.asarray(wavelength_calibration.coordinates, dtype=float)
    map_wavelengths = np.asarray(wavelength_calibration.wavelength, dtype=float)
    order = np.argsort(map_coordinates)
    map_coordinates = map_coordinates[order]
    map_wavelengths = map_wavelengths[order]
    pitch_px = int(channel_width_px) + int(gap_px)
    wavelength_slope = float(
        np.median(np.diff(map_wavelengths) / np.diff(map_coordinates))
    )
    if not np.isfinite(wavelength_slope) or wavelength_slope == 0.0:
        raise ValueError("Step 2 wavelength slope is zero or invalid")

    # x channels run toward higher wavelengths, with x[0] exactly at the
    # requested wavelength. w channels run toward lower wavelengths. Keeping
    # both lists the same length preserves the encoder's x/w array contract.
    high_direction = 1 if wavelength_slope > 0.0 else -1
    low_direction = -high_direction

    def available_steps(direction: int) -> int:
        boundary = map_coordinates[-1] if direction > 0 else map_coordinates[0]
        return int(np.floor(abs(float(boundary) - center_x) / pitch_px))

    high_steps = available_steps(high_direction)
    low_steps = available_steps(low_direction)
    n_channels = min(high_steps + 1, low_steps)
    if n_channels < 3:
        raise ValueError(
            "Step 2 range must fit at least two neighbours on each side of "
            "the target channel"
        )

    half_width = int(channel_width_px) // 2

    def wavelength_at(coordinate: int) -> float:
        return float(np.interp(coordinate, map_coordinates, map_wavelengths))

    # one fit for the anchor curve, shared by every channel that reuses it
    anchor_fit = fit_transfer_curve(levels, curve)

    def make_channel(index: int, side: str, coordinate: int) -> EncodingChannel:
        wavelength = (
            float(target_wavelength_nm)
            if side == "x" and index == 0
            else wavelength_at(coordinate)
        )
        return EncodingChannel(
            index=index,
            side=side,
            x_center=coordinate,
            x_start=coordinate - half_width,
            x_end=coordinate - half_width + int(channel_width_px),
            wavelength_nm=wavelength,
            levels=levels.copy(),
            intensity_curve=curve.copy(),
            transfer_fit=anchor_fit,   # every channel reuses the anchor curve
        )

    x_channels = [
        make_channel(i, "x", center_x + high_direction * i * pitch_px)
        for i in range(n_channels)
    ]
    w_channels = [
        make_channel(i, "w", center_x + low_direction * (i + 1) * pitch_px)
        for i in range(n_channels)
    ]
    off_level = (
        anchor_fit.off_level if anchor_fit is not None
        else int(levels[int(np.argmin(curve))])
    )
    return (
        ChannelLayout(
            x_channels=x_channels,
            w_channels=w_channels,
            center_wl=float(target_wavelength_nm),
            center_x=float(center_x),
            channel_width_px=int(channel_width_px),
            pitch_px=pitch_px,
            nm_per_px=abs(wavelength_slope),
            calib_coords=map_coordinates.copy(),
            calib_off_levels=np.full(map_coordinates.size, off_level, dtype=int),
        ),
        interpolated_x,
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
