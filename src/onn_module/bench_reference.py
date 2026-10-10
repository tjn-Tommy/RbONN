"""The .npz the bench side consumes: what to display, and what Y to expect.

In:  a trained checkpoint, and ``--images``, which sets the bench cost at
     ``N * 79 * 10`` reads (10 images is 7,900; 512 is 404,480).  Default 10.
Out: the npz, in the units the bench works in.  Drives are commanded
     intensities in [0, 1], ready for ``slm_module.encoding``; ``y_block`` is
     dark-free, background-free detector volts -- what
     ``fit.verify.predict(..., coherent=True)`` returns.

How the bench replays it.  For image ``n``, shot ``t``, output ``o``: display
``x[n].reshape(79, 10)[t]`` on the input arm and ``W[o].reshape(79, 10)[t]`` on
the weight arm, read, subtract dark and the single-beam background, compare
against ``y_block[n, o, t]``.  Summing over ``t`` gives ``y_sum[n, o]``, times
the readout gain in ``meta`` for the logit.

``x`` ships 790 columns, not 784: 79 shots of 10 leaves 6 channels of the last
shot with no data, and they are padded with zero BEFORE the learned input gain,
so they are displayed at ``tanh(in_shift[k])``, not dark.  Shot 78 must display
all ten.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):      # run as a script: put src/ on the path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from onn_module.layer import ONNLayer  # noqa: E402
from onn_module.params import ONNParams  # noqa: E402
from onn_module.finetune import (  # noqa: E402
    DEFAULT_CALIB,
    build_model,
    load_test_set,
)

__all__ = ["dump", "describe"]


def _head(model) -> ONNLayer:
    head = model[-1]
    if not isinstance(head, ONNLayer):
        raise SystemExit(f"{type(head).__name__} is not an ONN head -- "
                         f"--model bench has nothing to hand to the bench")
    return head


@torch.no_grad()
def dump(model, dataset, idx, *, calib_source: str, checkpoint: str) -> dict:
    """Run `idx` of `dataset` and collect everything the bench needs.

    Out: the dict saved as the npz.

    How: the drives are captured at the layer's own handover point
    (``ONNLayer.drive``), after in_scale, in_shift, tanh, the clamp and any
    drive quantization, rather than recomputed here.
    """
    head = _head(model)
    model.eval()
    device = next(model.parameters()).device

    caught: list[torch.Tensor] = []
    handle = head.register_forward_pre_hook(
        lambda _m, inp: caught.append(inp[0].detach()))

    images = torch.stack([dataset[i][0] for i in idx]).to(device)
    labels = np.asarray([int(dataset[i][1]) for i in idx])
    logits = model(images)
    handle.remove()

    features = caught[0]
    # (banks, batch, tile, out) -> (batch, out, tile), bank 0.  Differential
    # models have two banks and the bench reads them as two separate shots, so
    # they are dumped separately rather than pre-subtracted.
    per_shot = head._detector(features)
    y_block = per_shot.permute(0, 1, 3, 2).cpu().numpy()      # (banks,N,out,tile)

    drive = head.drive(features)                               # (N, tile, K)
    w = head.w()                                               # (banks, tile, out, K)
    # (tile, out, K) -> (out, tile*K), so column j = tile*K + k matches x's
    # feature order.  This is the layout bench_protocol.md documents as (10, 790).
    weights = w.permute(0, 2, 1, 3).reshape(w.shape[0], w.shape[2], -1).cpu().numpy()

    out = {
        "idx": np.asarray(idx, dtype=np.int64),
        "labels": labels,
        "x": drive.reshape(len(idx), -1).cpu().numpy(),
        "W": weights[0] if weights.shape[0] == 1 else weights,
        "y_block": y_block[0] if y_block.shape[0] == 1 else y_block,
        "y_sum": y_block.sum(-1)[0] if y_block.shape[0] == 1 else y_block.sum(-1),
        "logits_ref": logits.cpu().numpy(),
        "eta": head.eta.cpu().numpy(),
        "phi": head.phi.cpu().numpy(),
        "channels": np.asarray(head.channels, dtype=np.int64),
        "y_sat": np.float64(head.y_sat),
        "phase_sign": np.float64(head.phase_sign),
        "n_tiles": np.int64(head.n_tiles),
        "n_channels": np.int64(head.in_features),
        "n_out": np.int64(head.out_features),
        "n_banks": np.int64(w.shape[0]),
        # A named key, not a line in `meta`: the bench side cannot build the
        # channel layout, and cannot subtract the single-beam background, without
        # this file -- the npz carries eta and Phi but NOT a_x/a_w.  A required
        # input belongs in a field of its own rather than inside a JSON string.
        "step7_json": str(calib_source),
    }
    gain = getattr(head, "out_scale", None)
    if gain is not None:
        out["out_scale"] = gain.detach().cpu().numpy()
    out["meta"] = json.dumps({
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": str(checkpoint),
        "calib": str(calib_source),
        "readout": head.readout,
        "differential": bool(head.differential),
        "drive_bits": head.drive_bits,
        "drive_levels": None if head.drive_levels is None
                        else int(head.drive_levels.shape[-1]),
        "n_images": len(idx),
        "reads_per_image": int(head.n_tiles * head.out_features * w.shape[0]),
        "units": "y_block/y_sum in detector volts, dark-free and background-free "
                 "(fit.verify.predict(..., coherent=True)); x/W are commanded "
                 "intensities in [0, 1]",
    })
    return out


def describe(d: dict) -> str:
    """The magnitudes bench_protocol.md quotes, recomputed from this dump.

    Out: the per-measurement percentiles (the input range to read on), the
    pedestal-to-contrast ratio (how much of the signal is class-discriminative)
    and the resulting noise budget, the shots adding as sqrt(n_tiles).
    """
    y = np.asarray(d["y_block"], dtype=float)
    if y.ndim == 4:                      # differential: report both banks together
        y = y.reshape(-1, *y.shape[2:])
    flat = y.ravel()
    ps = np.percentile(flat, [1, 25, 50, 75, 99])
    y_sum = y.sum(-1)                    # (N, out)
    pedestal = float(y_sum.mean())
    disc = float((y_sum - y_sum.mean(axis=1, keepdims=True)).std())
    n_tiles = int(np.asarray(d["n_tiles"]))
    # Contrast across the outputs of ONE shot, before the shots dilute it.
    one = y[:, :, 0]
    contrast = float((one.max(1) - one.min(1)).mean() / max(one.mean(), 1e-30))
    # Per-measurement noise that keeps the logit SNR at 10 after sqrt(n_tiles).
    budget = disc / 10.0 / np.sqrt(n_tiles) if n_tiles else float("nan")
    reads = int(json.loads(str(d["meta"]))["reads_per_image"]) * len(d["idx"])
    return (
        f"  per-measurement Y [V]   p1 {ps[0]:.5f}  p25 {ps[1]:.5f}  "
        f"p50 {ps[2]:.5f}  p75 {ps[3]:.5f}  p99 {ps[4]:.5f}  max {flat.max():.5f}\n"
        f"  below 1 mV              {100 * (flat < 1e-3).mean():.1f}% of measurements\n"
        f"  summed over {n_tiles:3d} shots  pedestal {pedestal:.4f} V, "
        f"class-discriminative std {disc:.4f} V ({100 * disc / pedestal:.1f}%)\n"
        f"  contrast in one shot    {100 * contrast:.0f}%\n"
        f"  noise budget            < {budget * 1e3:.2f} mV per read keeps the "
        f"logit SNR above 10\n"
        f"  bench cost              {reads:,} reads"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="runs/onn784_ctrl/model.pt",
                    help="trained model.pt (weights only; the calibration comes "
                         "from --calib)")
    ap.add_argument("--model", default="onn784", choices=("onn784", "onn49"))
    ap.add_argument("--calib", default=DEFAULT_CALIB,
                    help="step-7 or step-8 calibration JSON")
    ap.add_argument("--y-sat", type=float, default=None,
                    help="override the calibration's saturation level, in volts")
    ap.add_argument("--drive-bits", type=int, default=None,
                    help="quantize both arms to 2**BITS settings, as the bench does")
    ap.add_argument("--images", type=int, default=10,
                    help="how many test images to dump (default: 10).  This is "
                         "what sets the bench cost: N * n_tiles * n_out reads, "
                         "i.e. 7,900 for 10 images on onn784 and 404,480 for 512")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="input noise std; 0 is the clean test set")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for which test images are picked")
    ap.add_argument("--out", default=None,
                    help="output .npz (default: beside the checkpoint)")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        raise SystemExit(f"{ckpt} not found -- train it first with finetune.py")
    params = ONNParams.from_json(args.calib, y_sat=args.y_sat)
    print(params.summary())

    model = build_model(args.model, p_conv=0.0, p_fc=0.0, p_head=0.0,
                        onn_params=params, input_map="tanh", input_gain=True,
                        readout="scale", drive_bits=args.drive_bits)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    for k in [k for k in state if k.endswith((".eta", ".phi"))]:
        state.pop(k)                      # the calibration comes from --calib
    model.load_state_dict(state)

    dataset = load_test_set(noise=args.noise)
    rng = np.random.default_rng(args.seed)
    n = min(args.images, len(dataset))
    idx = sorted(rng.choice(len(dataset), size=n, replace=False).tolist())

    d = dump(model, dataset, idx, calib_source=args.calib, checkpoint=str(ckpt))
    out = Path(args.out) if args.out else ckpt.with_name("bench_reference.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **d)

    acc = float((d["logits_ref"].argmax(1) == d["labels"]).mean())
    print(f"\n{out}  ({out.stat().st_size / 1e6:.2f} MB)")
    print(f"  {n} images, accuracy on them {acc:.2%}")
    print(describe(d))


if __name__ == "__main__":
    main()
