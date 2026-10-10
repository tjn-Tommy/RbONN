# RbONN bench protocol

The contract between the ONN layer as simulated in
[`src/onn_module/layer.py`](../src/onn_module/layer.py) and as measured on the
bench. Both produce `Y` in detector volts.

`Y` is the same quantity
[`fit/verify.py`](../src/calibration_module/fit/verify.py) `predict(...,
coherent=True)` returns — that function is the project's one forward model, and
`tests/test_onn_layer.py` holds the torch layer to it at 1e-12. Where this
document and that test disagree, the test is right.

Numbers below are measured from `runs/onn784_ctrl/model.pt` (onn784, 97.66%
clean / 97.58% at input noise 0.1) over 512 MNIST test images, under the
calibration that checkpoint was trained with. They move with the calibration —
re-run `bench_reference.py` after a recalibration rather than trusting them.

## Symbols

| symbol | shape | range | what |
|---|---|---|---|
| `x_k` | per shot, K=10 | **[0,1]** | input-arm transmission displayed |
| `w_k` | per shot, per output, K=10 | **[0,1]** | weight-arm transmission |
| `eta_k` (η) | 10 | 0.040 – 0.179 | per-channel amplitude, calibrated |
| `Phi_k` (Φ) | 10 | rad | per-channel static phase, calibrated |
| `Y_sat` | scalar | volts | atom-saturation scale |
| `Y` | per (shot, output) | volts | **the comparable quantity** |

Feature `j` of a layer maps to shot `t = j // 10`, channel `k = j % 10`. For
onn784: `F = 790`, `T = 79` shots, `O = 10` outputs. One detector, so the O
weight vectors share the K channels and are read one at a time —
`T * O = 790` measurements per image.

## The layer computes

```
u_k = asin(sqrt(x_k))                 v_k = asin(sqrt(w_k))

E   = SUM_k  eta_k * sqrt(x_k * w_k) * exp( i * [ u_k + v_k + Phi_k ] )

Y   = |E|^2 / ( 1 + |E|^2 / Y_sat )
```

Coherent **within** a shot; the `T` shots sum **incoherently**.

`u` and `v` are not free: `u = asin(sqrt(x))` means amplitude `sqrt(x)` and
phase `u` are the same variable, so a channel cannot be brightened without
being rotated.

## The bench reads

```
V = g * [ Y + SUM_k ( a_x,k * x_k  +  a_w,k * w_k ) ] + dark
```

## The bench post-processes

```
Y = (V - dark) / g  -  SUM_k ( a_x,k * x_k  +  a_w,k * w_k )
```

`g`, `dark`, `a_x`, `a_w` are instrument response, calibrated and removed on the
bench side. The simulation carries none of them — it produces the `Y` that
comes out of this step.

`g` is **measured during the run**, anchored to the run's own first minute.
`onn_infer` solves ten ladder rungs over `GAIN_BAND` of `Y_max` with step 8's
own builder (top capped at step 8's 450 mV ceiling) and reads them right after
the dark. That first block fits `g0`, the slope of `(V - dark)` against the
model's linear prediction through the origin, so `g` is in the model's units
from the start (run_1001 defined `g = 1` there instead and carried its +2.3 %
gap to the model through the whole run). The same drives are replayed every 20
minutes and `g = g0 * s`, with `s` the slope of `(V_now - dark)` against
`(V_first - dark)` through the origin: identical SLM frames on both sides, so
`eta`, `Phi` and `Y_sat` all cancel and no model enters the ratio. Each
inference read is then corrected with `g` interpolated in time between the two
blocks that bracket it.

Each anchor block is checked on its own: a rung more than `GAIN_TOL` (4 %) off
the block's fitted slope is dropped from that fit and written as
`kind=gain_drop`. The first block judges against the model's prediction, re-reads
what it drops once, and a rung off twice leaves the ladder for the run; later
blocks judge against the first block's volts. This exists because run_1001 showed single
anchor reads 4–6 % high with an ordinary trace std — a level shift lasting the
whole window, invisible to the step-6/7 trace guards. Inference reads are never
checked.

It matters: on the 0929 run, which had
no anchor, the best-fit scale wandered 0.939..1.064 over 14 h and removing it
per 2 h bin took the residual from 4.71 mV to ~3.9 mV. A common scale does not
change a per-image argmax, so this buys the residual and absolute `Y`, not the
accuracy number. What it does *not* measure is the gain change between the
step-7 fit and the start of the run; without a step 8 that stays in the residual
as a multiplicative tilt. The PMT thermistor is read with every gain block and
stamped on every row until the next (`--no-pmt` to skip).

`dark` is read once, at the start, and held. Over that same 14 h run it moved
0.034 mV with a 0.026 mV warm-up transient, against a 4.7 mV residual, and it is
common to all ten outputs. Step 6 re-reads it throughout because there the
all-off level is a fitted point of the background block at the same magnitude as
the `a_x`/`a_w` it anchors; nothing downstream of step 6 is in that regime.

## Expected magnitudes

Per-measurement `Y`, over 512 images x 79 shots x 10 outputs:

```
p1       p25      p50      p75      p99      max
0.00009  0.00231  0.00557  0.01146  0.05302  0.11836   volts
```

Log-normal-ish, centred at 5.6 mV, three decades wide; 11% of measurements
read below 1 mV. Summed over the 79 shots:

```
pedestal (mean summed Y)        0.7252 V
class-discriminative part, std  0.0795 V   = 11.0% of the pedestal
```

Contrast across the 10 outputs of one shot is 48%; summing 79 shots dilutes it
to 11%. Errors accumulate as `sqrt(79)`, so per-measurement noise below
**0.89 mV** (0.76% of full scale) keeps logit SNR above 10.

## The two parameters that were assumed

Both are now read from the calibration rather than assumed, by
[`src/onn_module/params.py`](../src/onn_module/params.py).

**`phase_sign`.** Settled: the project has one convention, and a stored
`dphi_comb_rad` **is** `Phi_k` in `exp(i[u + v + Phi])`, so `phase_sign = +1`
always. `ONNParams` gets `Phi_k` from `verify.load_forward_model`, the one
reader that defines it, and never walks a step-7 JSON itself.

The caveat is the file, not the code: step-7 JSONs written before 2026-09-25
carry a `"convention": "comb-slm"` tag and store **−Φₖ**. They are not migrated.
`calib_step7_result_0917_1831.json` is one of them, so anything trained against
it — including the `runs/onn784_ctrl` checkpoint — has the sign inverted and its
`y_block` is not valid ground truth. Re-run step 7.

**`Y_sat`.** No longer assumed: step 8 fits it and writes `y_sat_v` beside the
`step7_json` it resolved its forward model from, so `ONNParams.from_step8` — the
preferred loader — takes `y_sat` from the step-8 result and `eta`/`phi_rad` from
the step-7 file it names. One calibration, reached through the artifact that
measured the saturation against it. A step-7 file alone carries no saturation
level and the layer then runs linear. That
costs little here: at the brightest measurement the saturation denominator is
1.023, i.e. **2.3% peak compression**, so the layer is a near-pure coherent
quadratic sum either way. `Y_sat` near 0.1 V, or ~50x more drive, would be needed
for saturation to do real work.

**The drive is quantized, and that is not yet measured.** The bench displays
integer grayscale levels, 16 of them on the 4-bit 10-pair layout, on *both* arms.
`ONNLayer(drive_bits=4)` models that, but on a grid uniform in `x` — the panel's
is uniform in *level*, and `x = sin²(retardance/2)` is not linear in level. Pass
`drive_levels` built from the step-3c `TransferFit`s to remove the
approximation. Every accuracy quoted below was measured with a continuous drive
and is therefore optimistic.

## The zero-padding is not dark

784 features over 79 shots of 10 leaves 6 padding channels in the last shot.
They are padded with zero *before* the digital input gain, so they end up
driven at `tanh(in_shift[k])`, constant across images:

```
[ 0.0994  0.1953  0.2031  0.0745  0.0  0.7576 ]
```

One sits at 0.758 — lit, carrying no data, interfering coherently with the 4
real channels of that shot. **Shot 78 must display all 10 channels**, and a
reconstruction from only the 784 real features will not reproduce it. Hence
`x` ships with 790 columns.

## Reference dump

Written by
[`src/onn_module/bench_reference.py`](../src/onn_module/bench_reference.py) to
`runs/onn784_ctrl/bench_reference.npz`. It prints the magnitudes in this document
back out, recomputed from the dump, so a stale number here is visible rather than
inherited:

| key | shape | units | |
|---|---|---|---|
| `idx`, `labels` | (N,) | — | test-set positions, true labels |
| `x` | (N, 790) | [0,1] | drive to display; `reshape(N,79,10)` for per-shot |
| `W` | (10, 790) | [0,1] | transmissions; cols 784:790 are the constant pad |
| `y_block` | (N, 10, 79) | volts | simulated `Y` — **compare against this** |
| `y_sum` | (N, 10) | volts | `y_block.sum(-1)`, pre-gain |
| `logits_ref` | (N, 10) | — | the model's literal output |
| `eta`, `phi` | (10,) | —, rad | calibration |
| `channels` | (10,) | — | calibration index to bench channel |
| `y_sat`, `phase_sign`, `n_tiles`, `n_channels`, `n_out` | scalars | | step 8's `y_sat_v` (2.41 V on 0929, `inf` if none was passed), +1, 79, 10, 10 |
| `out_scale` | (10,) | — | the digital readout gain; `logit = y_sum * out_scale` |
| `n_banks` | scalar | — | 2 for a `--onn-differential` model, which the bench reads as two shots |
| `meta` | JSON | | checkpoint, calib file, units, reads per image |

The training env writes it; the bench env reads it with numpy alone and needs no
torch. `x` and `W` are commanded intensities in [0, 1], ready for
`slm_module.encoding.channel_layout_from_calibration` to turn into grayscale
levels.

The digital chain producing `x`:

```
conv trunk -> flatten -> reshape (B,79,10) + zero-pad
           -> x * in_scale + in_shift      (10 learned constants each)
           -> tanh                          (the [0,1] bound)
           -> clamp(1e-6, 1-1e-6)           == the npz `x`
```

All digital and **fixed at inference**: the constants live in the checkpoint,
nothing is recalibrated, and the drive is batch-independent (verified identical
for the same image in batches of 1, 16 and 512).
