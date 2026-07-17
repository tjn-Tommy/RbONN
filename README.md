# RbONN

## Alignment

Set the polarization chain before calibrating, so the SLM modulates at full
contrast. Work through the steps in order; they iterate.

1. **TM into the grating.** The PBS reflection port outputs TE, so the first
   HWP must rotate it to TM. Put a power meter at the grating output and rotate
   the first HWP until the diffracted power is at its **minimum** — the grating
   is least efficient for TM light, so minimum diffracted power means the beam
   is fully TM.
2. **TE out of the SLM.** Set the SLM to its bright level and put the power
   meter at the PBS output. Rotate the second HWP for **maximum** transmission,
   then fine-tune the SLM bright level and rotate again, iterating until the
   transmission peaks.
3. **Maximize extinction.** With the power meter still at the PBS output, set
   the SLM to its dark level and fine-tune the dark level for **minimum**
   transmitted power. If the extinction ratio (bright / dark) is below
   **25 dB**, repeat from step 1.

## Calibration

### Step 2 — wavelength map (px → nm)

Maps each SLM column to the wavelength it controls. An all-dark and an
all-bright frame are measured once as background / reference; then a
`window_size`-wide bright window walks across the scan region, and each
position's normalized spectrum is reduced to one `(coordinate, wavelength)`
point by a weighted centroid around its peak (`peak ± window` nm). A polynomial
fit (degree ≤ 3) over the points gives the map; the saved result always carries
the **dense per-column grid** plus `wavelength_fit_coefficients`, regardless of
how few positions were actually measured.

Acquisition knobs (GUI: Step 2 tab / pipeline wl_map stage):

| Knob | Effect |
|------|--------|
| `coordinate_stride` | measure every Nth column; the near-linear fit fills the skipped ones |
| `sweep_span_nm` | fast mode: measure the two region-edge positions with the wide OSA span first (**anchors**), draw a line through them, then re-center this narrow span (~1 nm) on the predicted wavelength at every other position — ~8× fewer samples per sweep with AUTO sampling |
| `min_peak_wavelength_nm` | ignore peak-search samples below this wavelength; masks artifacts below the source band (GUI default 775) |
| `max_peak_wavelength_nm` | ignore peak-search samples above this wavelength; masks a fixed leakage artifact the SLM never modulates (light landing outside the active area, ~781.7 nm on this setup → set ≈781.5) |
| `outlier_policy` | post-sweep auto-remeasure of points that sit off the linear map |

Normalized traces are only trusted where the bright reference carries at least
5% of its peak power — outside the source spectrum the reference − background
denominator is ≈0 and would inflate drift residue into spurious peaks (the
historical cause of stride points landing off the line).

### Step 3 — grayscale transfer curve per coordinate

For every coordinate mapped in Step 2, the panel lights one `window_size`-wide
window at that coordinate (rest of the panel at `min_level`) and sweeps the
window's grayscale level across `level_range`. The measured output at each level
is that channel's **transfer curve** — how commanded grayscale maps to optical
power at its wavelength.

| Field | Meaning |
|-------|---------|
| `level_range` | SLM grayscale levels swept (ascending) |
| `intensity_levels` | normalized output power, shape `(n_coordinates, n_levels)` |
| `raw_intensity_levels` | background-subtracted output (watts / volts) |
| `min_level` / `max_level` | off / on grayscale levels carried from Step 1 |

Two acquisition backends produce the same `CalibrationResult`:

- **OSA** (`intensity_calibration`) — reduces the spectrum around each
  coordinate's calibrated wavelength, and refines the wl→px map from this
  narrower sweep.
- **DAQ bucket detector** (`intensity_calibration_daq`) — no spectral
  resolution, so intensity is a plain dark-frame subtraction: an all-`min_level`
  frame is read once as the DC background and subtracted from every window
  reading (clamped at 0). No all-bright reference is taken — the downstream
  $I_0\,\sin^2(\theta/2)$ model fits $I_0$ as a free amplitude, so absolute scale
  is irrelevant, and a full-bright panel could saturate the photodiode.

**Channels only** (DAQ): instead of walking every calibrated coordinate, scan
only where an encoding channel lands. The panel builds the same channel-map
geometry as the preview (Window px = channel width, Pad px = gap), then lights
one channel window at a time — hopping from channel centre to channel centre and
skipping the dark pads and Rb guard bands. Each row of the result is a channel,
so the sweep is roughly `n_coordinates / n_channels` times shorter (≈20× on a
typical map) while producing a `CalibrationResult` the encoder reads unchanged.
Stride is ignored in this mode.

**Step 3c — channel grid + DAQ**: the same DAQ sweep, but the scan coordinates
come from Step 3b's channel structuring: mirror-symmetric channel pairs are
tiled around a configurable **target center** wavelength (default 778 nm), and
any channel whose window overlaps a **guard band** (default 780 / 776 ± 0.06 nm)
is skipped, with the next pitch outward tried instead. Guard bands must be
symmetric about the target so the x/w pairs keep equal wavelength offsets.

At encode time the transfer curve is **inverted**: `EncodingChannel.level_for(val)`
maps a target normalized value $val \in [0, 1]$ to a grayscale level by linear
interpolation between the two swept points whose measured outputs bracket the
target, taken over the off→on rising segment (made monotonic with a
cumulative-max envelope so noise near the flat top can't invert the mapping).
The returned level is rounded to the nearest integer grayscale, so even a
coarse level sweep can command grayscales between the swept points.
`val = 0` → `off_level`, `val = 1` → `on_level`.

### Step 6 — TPA efficiency ($\eta$) per pair

For a channel pair with per-side commanded intensities $x, w \in [0, 1]$, the 420 intensity $Y$ can be written as:

$$Y = \eta^2 (x \cdot w) + a_x\, x + q_x\, x^2 + a_w\, w + q_w\, w^2 + d$$

| Param | Physical meaning |
|-------|------------------|
| $\eta$ | two-photon efficiency of the pair (fit is linear in $b = \eta^2$; $\eta = \sqrt{b}$) |
| $a_x,\ a_w$ | single-beam linear response of each sideband |
| $q_x,\ q_w$ | single-beam quadratic (saturation) response of each sideband |
| $d$ | dark offset (readout with both sides off) |

Single beam — one sideband on, amplitude swept (pins $a$, $q$):

![Single sideband swept](docs/images/step6_single.png)

Cross (pair) — one sideband pinned at $x = 1$, the other swept; the only points with $x \cdot w \neq 0$, so they pin $\eta$:

![Both sidebands, one swept](docs/images/step6_pair.png)

## Encoding

### Channel layout

The channel geometry is decided **once**, when the Step-3b/3c measurement grid
is designed (`build_channel_calibration_grid`), tiling the panel into
symmetric channel **pairs** around the 778 nm centre:

1. Fit `wl = a·x + b` over the Step-2 map (`a < 0`: higher pixel → lower λ).
2. Anchor the centre pixel `c0 = round((778 − b) / a)`. It sits in the middle of
   a `gap_px` pad, so no channel covers it.
3. Convert the Rb guard bands (default 779.9–780.1 and 775.9–776.1 nm) to
   inclusive pixel ranges that must stay dark.
4. Tile a shared offset `m` outward from `c0` (half-pitch start,
   `pitch = width + gap`), placing a mirror pair each step: an **x**-channel at
   `c0 − m` (λ > 778) and a **w**-channel at `c0 + m` (λ < 778). One shared `m`
   keeps each pair exactly symmetric about the centre column — and, under the
   linear fit, symmetric in wavelength about 778 nm.
5. If either window would cover a guard band, `m` jumps past it (both sides move
   together, staying symmetric), so channels land on both sides of the Rb line.
   Tiling stops when either side leaves the calibrated range, so the two sides
   are always equal length (the encoder's x/w pairing contract).

Defaults: `channel_width_px = 15`, `gap_px = 5`, `n_channels = 20` per side.
Padding, guard-band, and centre columns render at their local off level, so
they stay dark with no extra masking.

Every consumer of a Step-3b/3c result — the TPA Encoding tab, the pipeline's
Step-6/7 stages, and the draft scripts — loads it **verbatim**
(`channel_layout_from_calibration`): the calibration already *is* the channel
structure (one row per channel centre, with the target centre, pitch and guard
skips baked into the coordinates), so centre, pitch, x/w pairing and guard
gaps are all derived from the mirror-symmetric grid. No re-tiling and no
nearest-coordinate snapping — the encoder always drives exactly the channels
that were measured, and pair indices mean the same thing everywhere. The one
number the file does not record is the window/gap split; the window width is
taken as `pitch − gap_px` (default gap 5 px). To move the channels, re-run
Step 3c with a new target centre.

The legacy re-tiling consumer (`build_channel_layout`) is deprecated for
Step-3 results; it remains only where no measured channel grid exists yet —
the coarse Step-1+2 quick-test preview and the TPA centre scan, which must
move the layout centre to sweep it.
