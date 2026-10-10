"""
Differentiable optical-neural-network layer (RbONN forward model).

In:  a feature vector, and `ONNParams` (eta, Phi, Y_sat) from
     `onn_module.params`.  This file parses no calibration.
Out: O outputs.  The hardware has one detector, so O outputs means re-using the
     same K channels with O weight vectors, eta/Phi being a channel property.
     More than K inputs runs as `n_tiles` shots of K whose readings are ADDED:
     coherent within a shot, incoherent between.

The model.  Per channel k the two SLM arms encode x_k, w_k in [0, 1] as
half-phases u_k = asin(sqrt(x_k)) and v_k = asin(sqrt(w_k)), and the K channels
interfere coherently on one detector:

    E = sum_k  eta_k sqrt(x_k w_k) exp( i [ s*(u_k + v_k) + Phi_k ] )
    Y = |E|^2 / (1 + |E|^2 / Y_sat)

`s` is `phase_sign`: +1 is the project convention, -1 the `exp(j(Phi - u - v))`
of rbonn_3pair_sweep.py.  The single-beam backgrounds and the dark are not
modelled; the bench subtracts them before Y is read.

How `forward()` evaluates it: the closed form
sqrt(x) e^{iu} = sqrt(x(1-x)) + i x factorises E into a complex matrix product,
with no asin.  `forward_reference()` evaluates the equations literally,
`--selftest` checks the two agree, and `tests/ml/test_onn_layer.py` pins
`forward()` against `calibration_module.fit.verify.predict` to 1e-12.

The calibration is registered non-persistent, so it never travels in a
checkpoint and `--calib` sets it on every load.

CLI
    python layer.py --template onn_params.json
    python layer.py --calib src/calib_data/<run>/calib.json
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
        weight_parametrization: "phase" (default) trains the SLM half-phase,
                      w = sin^2((pi/2) sigmoid(theta)), so w stays in (0, 1) and
                      nothing is clamped; "direct" trains w itself, clamped to
                      [eps, 1-eps].
        n_tiles:      this many successive K-channel shots, readings summed; the
                      layer then accepts n_tiles * K features, each tile with its
                      own weights.
        input_map:    squash features into [0, 1] -- "clamp" (already in range),
                      "sigmoid" (smooth, but 0 -> 0.5, i.e. half on), "tanh"
                      (0 -> 0, negatives off, for post-ReLU conv features).
        input_gain:   learnable per-channel scale+shift before input_map.
        differential: two weight banks, Y = Y_plus - Y_minus, for signed weights
                      on a detector where Y is otherwise non-negative.
        readout:      "none", "scale" (per-output gain, no bias; see
                      :meth:`calibrate_readout`), "affine" (gain and bias).
        trainable_calibration: let eta/Phi drift during training.  Off by
                      default.
        drive_bits:   quantize BOTH arms to 2**bits settings per channel, what an
                      N-bit SLM drive can display.  None is continuous.  See
                      :meth:`_quantize`.
        drive_levels: the x values each channel can actually produce, (K, L),
                      overriding drive_bits' uniform grid.  Fill it from the
                      step-3c transfer fits.
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

        # persistent=False keeps the calibration out of state_dict(), so
        # load_state_dict cannot put the eta/Phi a run was trained under back
        # over the ones --calib just loaded.
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
            # Off the hard ends, as _input_arm clamps: the arms are sqrt(x) and
            # sqrt(1-x), whose derivative is infinite at 0, so a grid level at
            # 0 or 1 gives a NaN gradient to any channel that snaps to it.  eps
            # is 1e-6 against a 4-bit step of 0.067.
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
        """Clamp the raw weights to +/-4.  Call after every optimizer step.

        That spans w in [8e-4, 0.9992], wider than any real SLM's contrast, and
        keeps theta out of sigmoid saturation, where the channel is dark and
        sigmoid' ~ 0 leaves no gradient.
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

        Quantized on the same grid as the input arm, both being displayed on the
        same panel.
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

        Separate from :meth:`drive` so :meth:`drive_headroom` can read a channel
        before the clamp, which is a hard gradient wall: past it a negative tanh
        is pinned at eps and d(output)/d(in_scale) is exactly 0.
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

        In:  ``drive_levels``, the x values one channel can produce, (K, L).
        Out: the drive snapped onto that grid, straight-through -- the rounding
             is subtracted back out of the graph, so the forward pass sees the
             quantized drive and the backward pass the gradient of the
             continuous one.

        A grid from ``drive_bits`` alone is uniform in x; the hardware is uniform
        in grayscale LEVEL, and x = sin^2(retardance/2) is not linear in level.
        Pass a step-3c ``TransferFit.intensity`` over its encoding window for the
        real per-channel grid.
        """
        grid = self.drive_levels                      # (K, L)
        # |x - level| over the grid, nearest wins.  x is (..., K); unsqueeze a
        # level axis so the comparison broadcasts against (K, L).
        j = (x.unsqueeze(-1) - grid).abs().argmin(dim=-1)
        snapped = torch.gather(grid.expand(*x.shape, -1), -1, j.unsqueeze(-1)).squeeze(-1)
        # `snapped + (x - x.detach())`, not `x + (snapped - x).detach()`:
        # x - x.detach() is exactly 0.0, so the forward value is bit-for-bit a
        # grid point.  The second form snaps only to within a rounding error.
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
            # Padded with 0, but NOT dark: in_scale/in_shift run after this, so
            # a pad channel is displayed at tanh(in_shift[k]) and interferes
            # with the last tile's real channels.  A bench reconstruction needs
            # these drive values, not just the real features.
            x = torch.nn.functional.pad(x, (0, self.features - x.shape[-1]))
        return x.view(-1, self.n_tiles, self.in_features)

    # -- drive -----------------------------------------------------------
    def drive(self, x: Tensor) -> Tensor:
        """The SLM transmissions this input asks for: (batch, n_tiles, K).

        The handover point: these numbers, in [0, 1], are what the modulator is
        told to display.  in_scale, in_shift, input_map and the clamp all run in
        front of it, in the computer.
        """
        return self._input_arm(self._tiles(x))[0]

    def drive_headroom(self, x: Tensor, top_frac: float = 0.01) -> Tensor:
        """Per channel, how far the bright tail falls short of full scale.

        Out: a penalty for the loss, which pays for drive.  Two-sided: each
        channel's top pulled to 1 AND its bottom held at 0.

        Measured on ``_drive_raw``, before the clamp, so a channel that has gone
        dark still has a gradient.  The value can therefore exceed 1, and the
        reported ``drive = 1 - sqrt(headroom)`` go negative.
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

        How: rescale the bias-free readout gain so the spread ACROSS outputs
        reaches ``target``.  That spread, not the common mode softmax ignores, is
        what a logit needs, and it runs ~0.03 V untouched -- under a softmax
        that is a flat 1/n_out and cross-entropy sits at log(n_out).

        ``only_if_below`` rescales only once the spread has fallen under that
        floor, making this a one-sided automatic gain control.
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

    In:  ``in_features``, how many features the layer must swallow.  The channel
         count K comes from the calibration.  ``out_features`` must fit within K.
    Out: the layer, tiled into as many shots as ``in_features`` needs.  A sample
         then costs ``n_tiles * O`` measurements.

    ``readout="scale"`` gives it a trainable detector gain, no bias.
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
