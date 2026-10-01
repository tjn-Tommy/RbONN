"""
Differentiable optical-neural-network layer (RbONN forward model).

Physics
-------
Per channel k the two SLM arms encode x_k and w_k as half-phases

    u_k = asin(sqrt(x_k))        input arm      x_k in [0, 1]
    v_k = asin(sqrt(w_k))        weight arm     w_k in [0, 1]

the K channels interfere coherently on one detector,

    E   = sum_k  eta_k sqrt(x_k w_k) exp( i [ s*(u_k + v_k) + Phi_k ] )
    I   = |E|^2
    Y   = I / (1 + I / Y_sat)

with a saturating atomic response.  The single-beam backgrounds and the dark
level are not modelled: on the bench they are subtracted away before Y is
read, so the layer works in the same already-subtracted units.
`s` is `phase_sign`: +1 reproduces the convention above, -1 reproduces the
`exp(j(Phi - u - v))` convention used by rbonn_3pair_sweep.py.  The fitted
Phi_k are only meaningful together with the sign they were fitted under, so it
is an explicit knob and never guessed.

Closed form used here
---------------------
Because u = asin(sqrt(x)) on [0, pi/2],

    sin u = sqrt(x),   cos u = sqrt(1 - x)
    sqrt(x) e^{iu} = sqrt(x(1-x)) + i x

and likewise for w.  The field therefore factorises,

    E_o = sum_k [eta_k e^{i Phi_k}] * p_k * q_ok ,
    p_k  = sqrt(x_k(1-x_k)) + i s x_k ,   q_ok = sqrt(w(1-w)) + i s w

which is a complex matrix product: no asin, no per-(batch, out, channel)
tensor, and no 1/sqrt(x(1-x)) blow-up from differentiating asin.
`forward_reference()` evaluates the equations literally instead, and
`--selftest` checks the two agree.

A layer maps K channels -> O outputs.  The hardware has one detector, so O
outputs means the same K channels are re-used with O different weight vectors
(time multiplexing); the calibration eta/Phi is a property of the
channel, so it is shared across outputs.

A bench with K channels cannot take more than K inputs in one shot.  Wider
layers are run as `n_tiles` successive shots of K inputs each, and the
detector readings are added:

    Y_o = sum_t  Y_o^(t) ,    Y^(t) = the K-channel model above

Interference is coherent *within* a shot and incoherent *between* shots, so a
tiled layer is genuinely a different (weaker) operator than one wide coherent
sum -- it is what the hardware can actually do, not an approximation of it.

Parameters come from `onn_module.params.ONNParams`, which loads them through
`calibration_module.fit.verify` -- this file parses no calibration and holds no
sign convention of its own.  `tests/test_onn_layer.py` pins `forward()` against
`verify.predict`, the numpy original, to 1e-12: that check is what makes this a
port of the project's one forward model rather than a second one.

The calibration is registered NON-persistent, so it never travels in a
checkpoint.  eta and Phi are measurements, not weights: a checkpoint carrying
them would silently restore the calibration it was trained under and make
`--calib` a no-op on every load.

CLI
    python layer.py --template onn_params.json
    python layer.py --calib src/calib_data/.../calib_step7_result_0917_1831.json
    python layer.py --selftest
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor, nn

if __package__ in (None, ""):      # run as a script: put src/ on the path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from onn_module.params import ONNParams, write_template  # noqa: E402

HALF_PI = math.pi / 2.0


# -------------------------------------------------------------- layer ----
class ONNLayer(nn.Module):
    """K optical channels -> O outputs, trained by ordinary autograd.

    Args:
        in_features:  K, number of coherent channels.
        out_features: O, weight vectors multiplexed onto the same channels.
        params:       ONNParams; defaults to an ideal device of K channels.
        weight_parametrization:
            "phase"  (default) the trained variable is the SLM half-phase,
                     v = (pi/2) * sigmoid(theta), w = sin^2 v.  w stays in
                     (0, 1) by construction and nothing is ever clamped.
            "direct" the trained variable is w itself, clamped to [eps, 1-eps].
                     Gradients die at the bounds; kept for interpretability.
        n_tiles:      run `n_tiles` successive K-channel shots and sum the
                      detector readings; the layer then accepts
                      n_tiles * K features.  Each tile gets its own weights.
        input_map:    how an arbitrary feature vector is squashed into [0,1]:
                      "clamp"   already in range (e.g. ToTensor pixels)
                      "sigmoid" smooth, but maps 0 -> 0.5, i.e. "half on"
                      "tanh"    maps 0 -> 0 and negatives -> off, which is what
                                you want for post-ReLU conv features.
        input_gain:   learnable per-channel scale+shift before input_map --
                      the digital stand-in for laser power / SLM drive trim.
        differential: two weight banks, Y = Y_plus - Y_minus.  Y is otherwise
                      non-negative, so this is how a signed weight is realised
                      on the bench (two shots, subtracted).
        readout:      digital post-detector processing of Y:
                      "none"   Y is the output
                      "scale"  learnable per-output gain, no bias -- detector
                               gain, needed because Y is of order eta^2 and a
                               softmax over a ~0.3 V range cannot train
                      "affine" gain and bias.
        trainable_calibration: let eta/phi drift during training.
                      Off by default -- they are measurements, not weights.
        drive_bits:   quantize BOTH arms to 2**bits settings per channel, which
                      is what an N-bit SLM drive can actually display.  None
                      leaves the drive continuous, i.e. optimistic.  See
                      _quantize for what the uniform grid does and does not
                      model.
        drive_levels: the exact x values each channel can produce, shape (K, L),
                      overriding the uniform grid drive_bits would build.  Fill
                      it from the step-3c transfer fits to model the real,
                      non-uniform level spacing.
    """

    def __init__(self, in_features: int, out_features: int,
                 params: Optional[ONNParams] = None, *,
                 n_tiles: int = 1,
                 weight_parametrization: str = "phase",
                 input_map: str = "clamp",
                 input_gain: bool = False,
                 differential: bool = False,
                 readout: str = "none",
                 trainable_calibration: bool = False,
                 drive_bits: Optional[int] = None,
                 drive_levels: Optional[Tensor] = None,
                 eps: float = 1e-6):
        super().__init__()
        if weight_parametrization not in ("phase", "direct"):
            raise ValueError(f"weight_parametrization: {weight_parametrization!r}")
        if input_map not in ("clamp", "sigmoid", "tanh"):
            raise ValueError(f"input_map: {input_map!r}")
        if readout not in ("none", "scale", "affine"):
            raise ValueError(f"readout: {readout!r}")

        params = params or ONNParams.ideal(in_features)
        if params.n_channels != in_features:
            raise ValueError(
                f"calibration has {params.n_channels} channels but layer needs "
                f"{in_features}; use params.select(...) or params.tile_to({in_features})")

        self.in_features, self.out_features = in_features, out_features
        self.n_tiles = int(n_tiles)
        self.features = self.n_tiles * in_features   # what forward() accepts
        self.weight_parametrization = weight_parametrization
        self.input_map, self.differential, self.readout = input_map, differential, readout
        self.eps, self.y_sat = eps, params.y_sat
        self.phase_sign = params.phase_sign
        self.channels = params.channels

        # persistent=False: the calibration must NOT travel in a checkpoint.
        # A buffer is in state_dict() by default, so load_state_dict would put
        # the eta/Phi the run was trained under back over the ones just loaded
        # from --calib -- silently, and strict=True would not complain because
        # the keys match.  Every recalibration would then be a no-op on any
        # model that was ever saved.  eta and Phi are measurements; the
        # calibration file is their only home.
        kind = nn.Parameter if trainable_calibration else (lambda t: t)
        for name in ("eta", "phi"):
            t = getattr(params, name).clone()
            if trainable_calibration:
                self.register_parameter(name, kind(t))
            else:
                self.register_buffer(name, t, persistent=False)

        # The drive grid is a property of the panel, not of the trained model,
        # so it is non-persistent for the same reason the calibration is.
        if drive_levels is not None:
            grid = torch.as_tensor(drive_levels, dtype=torch.get_default_dtype())
            if grid.ndim == 1:
                grid = grid.expand(in_features, -1)
            if grid.shape[0] != in_features:
                raise ValueError(f"drive_levels has {grid.shape[0]} channels, "
                                 f"layer needs {in_features}")
            self.drive_bits = drive_bits
        elif drive_bits is not None:
            if drive_bits < 1:
                raise ValueError(f"drive_bits: {drive_bits!r}, expected >= 1")
            grid = torch.linspace(0.0, 1.0, 2 ** int(drive_bits))
            grid = grid.expand(in_features, -1)
            self.drive_bits = int(drive_bits)
        else:
            grid, self.drive_bits = None, None
        if grid is not None:
            # Held off the hard ends for the same reason _input_arm clamps: the
            # arms are sqrt(x) and sqrt(1-x), whose derivative is infinite at 0.
            # A grid built on [0, 1] therefore has a level that produces a NaN
            # gradient as soon as any channel snaps to it -- and the off level is
            # the one a dark channel snaps to constantly.  eps is 1e-6 against a
            # 4-bit step of 0.067, so nothing physical moves.
            grid = grid.clamp(eps, 1.0 - eps)
        self.register_buffer("drive_levels", None if grid is None else grid.contiguous(),
                             persistent=False)

        banks = 2 if differential else 1
        self.raw_weight = nn.Parameter(
            torch.empty(banks, self.n_tiles, out_features, in_features))
        self.reset_parameters()

        if input_gain:
            self.in_scale = nn.Parameter(torch.ones(in_features))
            self.in_shift = nn.Parameter(torch.zeros(in_features))
        else:
            self.in_scale = self.in_shift = None
        if readout in ("scale", "affine"):
            self.out_scale = nn.Parameter(torch.ones(out_features))
        if readout == "affine":
            self.out_bias = nn.Parameter(torch.zeros(out_features))

    # -- weights ---------------------------------------------------------
    def reset_parameters(self) -> None:
        """Start with w spread over the responsive middle of the range."""
        with torch.no_grad():
            self.set_w(torch.empty_like(self.raw_weight).uniform_(0.15, 0.85))

    @torch.no_grad()
    def set_w(self, w: Tensor) -> None:
        """Set the physical transmissions w (broadcastable to raw_weight)."""
        w = w.expand_as(self.raw_weight).clamp(self.eps, 1 - self.eps)
        if self.weight_parametrization == "phase":
            v = torch.asin(w.sqrt())                      # in (0, pi/2)
            self.raw_weight.copy_(torch.logit(v / HALF_PI))
        else:
            self.raw_weight.copy_(w)

    @torch.no_grad()
    def clamp_(self, limit: float = 4.0) -> None:
        """Keep the raw weights inside the range where gradients survive.

        Call after every optimizer step.  With w = sin^2((pi/2) sigmoid(theta)),
        a theta that runs off to -30 puts sigmoid deep in saturation: the
        channel is dark, every output collapses to the same value, softmax
        goes uniform, and sigmoid' ~ 0 means there is no gradient left to
        climb back out.  Cross-entropy reaches that plateau the moment the
        quickest way down is to stop predicting at all, so it is not a rare
        accident -- a 79-tile head fell into it on roughly half of otherwise
        identical runs.

        +/-4 still spans w in [8e-4, 0.9992], wider than any real SLM's
        contrast, so nothing physical is given up.
        """
        if self.weight_parametrization == "phase":
            self.raw_weight.clamp_(-limit, limit)
        else:
            self.raw_weight.clamp_(self.eps, 1 - self.eps)

    def w(self) -> Tensor:
        """Physical weights, shape (banks, out_features, in_features)."""
        sw, cw = self._weight_arm()
        return sw * sw

    def _weight_arm(self):
        """(sin v, cos v) = (sqrt(w), sqrt(1-w)).

        The weight arm is displayed on the same panel as the input arm, so it is
        quantized on the same grid -- a 4-bit bench gives 16 settings for w as
        well as for x, and w is the arm a training run is free to place.
        """
        if self.weight_parametrization == "phase":
            v = HALF_PI * torch.sigmoid(self.raw_weight)
            w = torch.sin(v) ** 2
        else:
            w = self.raw_weight.clamp(self.eps, 1 - self.eps)
        if self.drive_levels is not None:
            w = self._quantize(w)
        return w.clamp(0.0, 1.0).sqrt(), (1 - w).clamp(0.0, 1.0).sqrt()

    def _drive_raw(self, x: Tensor) -> Tensor:
        """Scale, shift and squash -- every digital step but the final clamp.

        Kept separate because the clamp is a hard gradient wall: once a
        channel's tanh has gone negative the clamp pins it at eps and
        d(output)/d(in_scale) is exactly 0, so no loss term can pull it back.
        Before the clamp that same channel reads negative, which is where its
        gradient still lives.  See drive_headroom.
        """
        if self.in_scale is not None:
            x = x * self.in_scale + self.in_shift
        if self.input_map == "sigmoid":
            return torch.sigmoid(x)
        if self.input_map == "tanh":
            return torch.tanh(x)         # 0 -> 0, negatives clamp to "off"
        return x

    def _quantize(self, x: Tensor) -> Tensor:
        """Snap a drive to the values the panel can actually display.

        The SLM takes an integer grayscale level, not a real number, so the
        bench cannot display an arbitrary x in [0, 1].  Leaving that out makes
        every simulated accuracy optimistic: a 4-bit drive is 16 settings per
        channel, and the interference is a coherent sum, so the error does not
        average away over the 79 shots the way an incoherent one would.

        Straight-through: the rounding is subtracted back out of the graph, so
        the forward pass sees the quantized drive and the backward pass sees the
        gradient of the continuous one.  Without that, d(round)/dx = 0 and the
        input gain would have no gradient at all.

        ``drive_levels`` is the grid of x values one channel can produce,
        shape (K, L).  Uniform in x when the layer was built from ``drive_bits``
        alone -- which the hardware is NOT: it is uniform in grayscale LEVEL,
        and x = sin^2(retardance/2) is not linear in level, so a uniform grid
        understates the error where the transfer curve is flat and overstates it
        where it is steep.  Pass the real per-channel grid (from a step-3c
        ``TransferFit.intensity`` over its encoding window) to remove that
        approximation.
        """
        grid = self.drive_levels                      # (K, L)
        # |x - level| over the grid, nearest wins.  x is (..., K); unsqueeze a
        # level axis so the comparison broadcasts against (K, L).
        j = (x.unsqueeze(-1) - grid).abs().argmin(dim=-1)
        snapped = torch.gather(grid.expand(*x.shape, -1), -1, j.unsqueeze(-1)).squeeze(-1)
        # `snapped + (x - x.detach())`, not `x + (snapped - x).detach()`: the
        # second is only snapped to within a rounding error, so the drive would
        # take a few more distinct values than the panel has levels.  Here
        # x - x.detach() is exactly 0.0, so the forward value is bit-for-bit a
        # grid point while the gradient is still 1.
        return snapped.detach() + (x - x.detach())

    def _input_arm(self, x: Tensor):
        """x -> (x in [0,1], sin u, cos u) = (x, sqrt(x), sqrt(1-x))."""
        x = self._drive_raw(x).clamp(self.eps, 1 - self.eps)
        if self.drive_levels is not None:
            x = self._quantize(x)
        return x, x.sqrt(), (1 - x).sqrt()

    def _tiles(self, x: Tensor) -> Tensor:
        """(batch, features) -> (batch, n_tiles, in_features), zero-padded."""
        x = x.reshape(x.shape[0], -1)
        if x.shape[-1] > self.features:
            raise ValueError(f"expected at most {self.features} features "
                             f"({self.n_tiles} tiles x {self.in_features}), "
                             f"got {x.shape[-1]}")
        if x.shape[-1] < self.features:
            # Padded with 0, but NOT dark: in_scale/in_shift are applied after
            # this, so a pad channel is driven at tanh(in_shift[k]) -- measured
            # up to 0.758 on a trained onn784.  They are image-independent, so
            # they act as a fixed reference beam interfering with the real
            # channels of the last tile.  Any bench reconstruction needs their
            # drive values, not just the real features.
            x = torch.nn.functional.pad(x, (0, self.features - x.shape[-1]))
        return x.view(-1, self.n_tiles, self.in_features)

    # -- drive -----------------------------------------------------------
    def drive(self, x: Tensor) -> Tensor:
        """The SLM transmissions this input asks for: (batch, n_tiles, K).

        Everything in front of this -- in_scale, in_shift, input_map, the
        clamp -- is digital preprocessing that happens in the computer before
        any light is modulated.  `drive` is the handover point: these numbers,
        in [0, 1], are what the modulator is told to display.
        """
        return self._input_arm(self._tiles(x))[0]

    def drive_headroom(self, x: Tensor, top_frac: float = 0.01) -> Tensor:
        """Per channel, how far the bright tail falls short of full scale.

        Cross-entropy has no SNR term, so nothing in the loss stops the
        learned input gain from parking a channel in the dark: a negative
        in_shift sends tanh below zero, the clamp pins it at eps, and that
        channel then contributes nothing but is still shot at n_tiles times.
        Adding this to the loss pays for drive.

        Two-sided, and it has to be: the target is the *top* of each
        channel's distribution pulled to 1 AND its *bottom* held at 0.
        Rewarding only the top does not work.  tanh(x * in_scale + in_shift)
        can raise its top two ways -- by in_scale, which lifts the bright
        features and leaves a zero feature at tanh(in_shift); or by in_shift,
        which lifts everything.  The second is the cheaper gradient, so a
        one-sided reward takes it: measured, every in_shift went to +0.65..
        +1.48, a zero feature displayed 0.57..0.90, contrast was gone and
        onn784 fell from 97.7% to 61.7%.  Holding the bottom at 0 closes that
        route, and costs nothing at initialisation, where in_shift = 0.

        Measured before the clamp, so a channel that has already gone dark
        still has a gradient (see _drive_raw).  That also means the value can
        exceed 1, and the reported drive = 1 - sqrt(headroom) can go negative,
        when a channel is being driven below zero.
        """
        d = self._drive_raw(self._tiles(x))                  # (B, T, K)
        per_chan = d.permute(2, 0, 1).reshape(self.in_features, -1)
        k = max(1, int(round(top_frac * per_chan.shape[1])))
        # topk, not quantile: quantile is unimplemented on MPS.
        top = per_chan.topk(k, dim=1).values.mean(1)         # (K,)
        bot = per_chan.topk(k, dim=1, largest=False).values.mean(1)
        # bot is squared, not hinged: pre-clamp it can go negative, and a
        # channel driven below zero is dead, so 0 is the target from both
        # sides.
        return (((1.0 - top) ** 2).mean() + (bot ** 2).mean()) / 2

    # -- forward ---------------------------------------------------------
    def _detector(self, x: Tensor) -> Tensor:
        """Y per bank and per shot: (banks, batch, n_tiles, out_features)."""
        s = self.phase_sign
        xv, sx, cx = self._input_arm(self._tiles(x))
        sw, cw = self._weight_arm()
        ww = sw * sw

        p_r, p_i = sx * cx, s * xv                 # sqrt(x) e^{i s u}
        q_r, q_i = sw * cw, s * ww                 # sqrt(w) e^{i s v}
        g, h = self.eta * torch.cos(self.phi), self.eta * torch.sin(self.phi)
        a_r, a_i = g * p_r - h * p_i, g * p_i + h * p_r   # fold in eta e^{i Phi}

        # b batch, t tile/shot, k channel, o output, g weight bank
        mm = lambda a, q: torch.einsum("btk,gtok->gbto", a, q)
        e_r = mm(a_r, q_r) - mm(a_i, q_i)
        e_i = mm(a_r, q_i) + mm(a_i, q_r)
        intensity = e_r * e_r + e_i * e_i

        return intensity / (1.0 + intensity / self.y_sat)

    def _readout(self, y: Tensor) -> Tensor:
        """Sum the shots, subtract the banks, apply the digital gain."""
        y = y.sum(-2)                              # incoherent sum over tiles
        y = y[0] - y[1] if self.differential else y[0]
        if self.readout in ("scale", "affine"):
            y = y * self.out_scale
        if self.readout == "affine":
            y = y + self.out_bias
        return y

    def forward(self, x: Tensor) -> Tensor:
        return self._readout(self._detector(x))

    @torch.no_grad()
    def calibrate_readout(self, x: Tensor, target_std: float = 1.0,
                          only_if_below: Optional[float] = None) -> float:
        """Set the detector gain from one batch.  Returns the spread it found.

        Y is an intensity of order eta^2, and tiling stacks n_tiles of them on
        top of each other, so every output sits on a large common mode with
        only a small variation between outputs.  Softmax ignores the common
        mode, so what matters is the spread *across* outputs -- typically
        ~0.03 V.  A softmax over 0.03 V cannot put more than 1/n_out + a few
        percent on any class, cross-entropy sits at log(n_out), and the
        gradients never grow enough to escape.

        One rescaling of the (bias-free) gain fixes it.  This is detector gain,
        the same knob as the transimpedance amplifier on the bench; it adds no
        offset, so a logit is still nothing but the coherent quadratic sum.

        With `only_if_below` this becomes a one-sided automatic gain control:
        the gain is raised only when the logit spread has fallen below that
        floor, which is the one situation the head cannot recover from by
        itself.  Called once per epoch it costs one batch and makes "turn
        every channel off" a dead end rather than an absorbing state, without
        otherwise constraining how confident the model chooses to be.
        """
        if self.readout == "none":
            return 0.0
        y = self._readout(self._detector(x))
        spread = (y - y.mean(-1, keepdim=True)).std().clamp_min(1e-12)
        if only_if_below is None or spread < only_if_below:
            self.out_scale.mul_(target_std / spread)
        return float(spread)

    def forward_reference(self, x: Tensor) -> Tensor:
        """The equations written out literally.  Slow; used to check forward()."""
        s = self.phase_sign
        xv, _, _ = self._input_arm(self._tiles(x))
        w = self.w()
        u = torch.asin(xv.clamp(0, 1).sqrt())              # (B, T, K)
        v = torch.asin(w.clamp(0, 1).sqrt())               # (banks, T, O, K)
        amp = self.eta * (xv[None, :, :, None, :] * w[:, None]).sqrt()
        phase = s * (u[None, :, :, None, :] + v[:, None]) + self.phi
        e = (amp * torch.exp(1j * phase)).sum(-1)          # (banks, B, T, O)
        intensity = e.abs() ** 2
        return self._readout(intensity / (1.0 + intensity / self.y_sat))

    def extra_repr(self) -> str:
        if self.drive_levels is None:
            drive = "continuous"
        else:
            bits = f" ({self.drive_bits}-bit)" if self.drive_bits else ""
            drive = f"{self.drive_levels.shape[-1]} levels{bits}"
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"n_tiles={self.n_tiles} (accepts {self.features} features), "
                f"banks={self.raw_weight.shape[0]}, param={self.weight_parametrization}, "
                f"input_map={self.input_map}, readout={self.readout}, "
                f"y_sat={self.y_sat}, phase_sign={self.phase_sign:+.0f}, drive={drive}")


def onn_linear(in_features: int, out_features: int,
               params: Optional[ONNParams] = None, **kw) -> ONNLayer:
    """nn.Linear(in_features, out_features) -> ONNLayer, tiled to fit the bench.

    `in_features` is how many features the layer must swallow; the number of
    optical channels comes from the calibration, and the layer is tiled into
    however many shots that needs.  `out_features` must fit within the K
    channels: the O weight vectors are multiplexed onto the same K channels and
    read one at a time off the single detector, so a sample costs n_tiles * O
    measurements.

    Y is a non-negative intensity of order eta^2, so a bare ONNLayer makes a
    poor logit head: readout="scale" gives it a trainable detector gain (no
    bias -- the logit stays a pure coherent quadratic sum).
    """
    k = (params or ONNParams.ideal(in_features)).n_channels
    if out_features > k:
        raise ValueError(f"{out_features} outputs needs {out_features} channels, "
                         f"calibration has {k}")
    kw.setdefault("n_tiles", -(-in_features // k))
    kw.setdefault("readout", "scale")
    return ONNLayer(k, out_features, params, **kw)


# --------------------------------------------------------------- main ----
def _selftest() -> None:
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    ok = True

    for phase_sign in (1.0, -1.0):
        for diff in (False, True):
            for par in ("phase", "direct"):
                p = ONNParams.ideal(6)
                p.eta = torch.rand(6) * 0.2 + 0.05
                p.phi = torch.randn(6) * 0.5
                p.y_sat, p.phase_sign = 0.5, phase_sign
                layer = ONNLayer(6, 4, p, n_tiles=3, weight_parametrization=par,
                                 differential=diff, readout="affine")
                x = torch.rand(7, 18)
                d = (layer(x) - layer.forward_reference(x)).abs().max().item()
                tag = f"sign{phase_sign:+.0f} diff={diff:<5} {par:<6}"
                print(f"  fast vs reference  {tag}  max|diff| = {d:.3e}")
                ok &= d < 1e-9

    # gradients reach the weights and the calibration is left alone
    layer = ONNLayer(6, 3, ONNParams.ideal(6), n_tiles=2,
                     differential=True, readout="scale")
    y = layer(torch.rand(5, 12))
    y.pow(2).sum().backward()
    gw = layer.raw_weight.grad
    print(f"  weight grad        finite={torch.isfinite(gw).all().item()}  "
          f"max|g|={gw.abs().max():.3e}")
    ok &= bool(torch.isfinite(gw).all()) and gw.abs().max() > 0
    ok &= not isinstance(layer.eta, nn.Parameter)

    # x at the ends of the range must not produce nan
    layer = ONNLayer(6, 3)
    x = torch.zeros(2, 6)
    x[1] = 1.0
    x.requires_grad_(True)
    layer(x).sum().backward()
    fin = bool(torch.isfinite(x.grad).all())
    print(f"  x=0 and x=1        grad finite={fin}")
    ok &= fin

    # set_w round-trips through both parametrisations
    for par in ("phase", "direct"):
        layer = ONNLayer(6, 3, weight_parametrization=par, n_tiles=2)
        want = torch.rand(1, 2, 3, 6) * 0.98 + 0.01
        layer.set_w(want)
        d = (layer.w() - want).abs().max().item()
        print(f"  set_w round-trip   {par:<6}  max|diff| = {d:.3e}")
        ok &= d < 1e-6

    # ideal single channel: Y = |1 * sqrt(x w) e^{i(u+v)}|^2 = x w
    layer = ONNLayer(1, 1, ONNParams.ideal(1), weight_parametrization="direct")
    layer.set_w(torch.full((1, 1, 1, 1), 0.3))
    got = layer(torch.full((1, 1), 0.4)).item()
    print(f"  ideal 1-channel    Y = {got:.6f}  (expect 0.120000)")
    ok &= abs(got - 0.12) < 1e-6

    # a short layer padded up to a whole number of tiles ignores the padding
    layer = ONNLayer(4, 2, ONNParams.ideal(4), n_tiles=3)
    x = torch.rand(3, 10)
    d = (layer(x) - layer(torch.cat([x, torch.zeros(3, 2)], 1))).abs().max().item()
    print(f"  zero-padding       max|diff| = {d:.3e}")
    ok &= d < 1e-12

    print("SELFTEST", "OK" if ok else "FAILED")
    if not ok:
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", metavar="OUT.json", help="write the flat JSON schema")
    ap.add_argument("--n-channels", type=int, default=8, help="channels for --template")
    ap.add_argument("--calib", metavar="IN.json", help="load params and print a summary")
    ap.add_argument("--channels", type=int, nargs="+", help="subset of channels to use")
    ap.add_argument("--y-sat", type=float, default=None,
                    help="saturation level in detector volts; overrides the file")
    ap.add_argument("--selftest", action="store_true", help="check forward() and gradients")
    args = ap.parse_args()

    if args.template:
        write_template(args.template, args.n_channels)
        print(f"wrote template for {args.n_channels} channels to {args.template}")
    if args.calib:
        p = ONNParams.from_json(args.calib, args.channels, args.y_sat)
        print(p.summary())
        layer = onn_linear(p.n_channels, 4, p)
        print(layer)
        print(f"  forward: {tuple(layer(torch.rand(3, p.n_channels)).shape)}")
    if args.selftest:
        _selftest()
    if not (args.template or args.calib or args.selftest):
        ap.print_help()


if __name__ == "__main__":
    main()
