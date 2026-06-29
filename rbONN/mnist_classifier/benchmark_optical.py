"""Benchmark + tuning harness: single optical NL layer (784 -> 10) on MNIST.

Counterpart of benchmark_linear.py.  The layer is:

    optical :  logit_k = | sum_i  W~[k,i] * x~_i |^2   (coherent sum + |.|^2)

  x~_i = amplitudes_to_efield(x_i)   (pixel -> E-field, |x~| = pixel)
  W~[k,i] = E-field of the learnable phase phi_w[k,i]

Goal: optimise this fixed 784x10 architecture to its best MNIST accuracy.

Tuning knobs (all CLI flags)
----------------------------
  --encoding flat|patch  : input pixel order.  *Provably identical* for a dense
                           784x10 layer (a permutation) -- 'patch' is provided
                           only to verify that empirically.
  --bias                 : add a learnable real per-class offset to |S|^2.
                           Gives "negative evidence" the bare square-law lacks.
  --logit-scale          : learnable scalar temperature on the logits.  Fixes
                           the huge-loss (|S|^2 ~ 100s) optimisation problem.
  --lr --epochs --batch-size --weight-decay --init-scale --seed

7,840 phase params (+10 if --bias, +1 if --logit-scale).

Usage
-----
  python -m rbONN.mnist.benchmark_optical --name opt_base
  python -m rbONN.mnist.benchmark_optical --bias --logit-scale --epochs 200 --name opt_best
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import trackio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..common import confusion_matrix, launch_trackio, per_class_acc, print_model_summary
from .data import load_mnist
from .twin import amplitudes_to_efield, phases_to_efield

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "output"
N_CLASSES = 10
TRACKIO_PROJECT = "rbONN_mnist_opt"

# NOTE: saturation / saturable-absorption modelling removed for now -- the
# previous measured-sigmoid model was based on an incorrect math model.
# Sticking with the pure square-law layer:  logit_k = | sum_i W~[k,i] x~_i |^2.


def _patch_perm() -> np.ndarray:
    """Index permutation reordering flat-784 pixels into 4x4-block raster order."""
    idx = np.arange(784).reshape(28, 28)
    blocks = [idx[by * 4:(by + 1) * 4, bx * 4:(bx + 1) * 4].reshape(-1)
              for by in range(7) for bx in range(7)]
    return np.concatenate(blocks)


class OpticalLayer(nn.Module):
    """One optical neuron per class: global coherent sum over all 784 pixels.

    Same fan-in as nn.Linear(784, 10, bias=False) but with complex E-field
    weights and square-law (|.|^2) detection.  Optional learnable bias and
    logit temperature are training aids, not extra boundary-shape power.
    """

    def __init__(self, n_in: int = 784, n_out: int = N_CLASSES,
                 bias: bool = False, logit_scale: bool = False, init_scale: float = 1.0,
                 phase_insensitive: bool = False):
        super().__init__()
        self.phi_w = nn.Parameter(torch.rand(n_out, n_in) * 2.0 * math.pi * init_scale)
        self.bias = nn.Parameter(torch.zeros(n_out)) if bias else None
        self.log_s = nn.Parameter(torch.zeros(())) if logit_scale else None
        self.phase_insensitive = phase_insensitive

    def raw_intensity(self, a: torch.Tensor) -> torch.Tensor:
        x = amplitudes_to_efield(a)        # (batch, 784) complex
        W = phases_to_efield(self.phi_w)   # (n_out, 784) complex
        if self.phase_insensitive:
            # phase-INSENSITIVE quadratic sum: drop cross-terms -> sum of intensities
            return x.abs().pow(2) @ W.abs().pow(2).T
        return (x @ W.T).abs().pow(2)      # phase-SENSITIVE |S|^2 (interference)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        out = self.raw_intensity(a)
        if self.log_s is not None:
            out = out * self.log_s.exp()
        if self.bias is not None:
            out = out + self.bias
        return out


def train_optical(
    epochs: int = 80,
    batch_size: int = 256,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    encoding: str = "flat",
    bias: bool = False,
    logit_scale: bool = False,
    init_scale: float = 1.0,
    optimizer_name: str = "adam",
    warmup: int = 0,
    label_smoothing: float = 0.0,
    phase_insensitive: bool = False,
    seed: int = 0,
    name: str = "optical",
) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)

    launch_trackio(TRACKIO_PROJECT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else ""
    print(f"Device: {device}" + (f" ({gpu})" if gpu else "") + f"  |  run: {name}")

    X_tr, y_tr, X_te, y_te = load_mnist()
    if encoding == "patch":
        perm = _patch_perm()
        X_tr, X_te = X_tr[:, perm], X_te[:, perm]

    A_tr = torch.from_numpy(X_tr).to(device)
    A_te = torch.from_numpy(X_te).to(device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long, device=device)
    y_te_t = torch.tensor(y_te, dtype=torch.long, device=device)

    model = OpticalLayer(784, N_CLASSES, bias=bias, logit_scale=logit_scale,
                         init_scale=init_scale, phase_insensitive=phase_insensitive).to(device)

    quad = "phase-insensitive" if phase_insensitive else "phase-sensitive"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: optical 784->{N_CLASSES} {quad} quad-sum  encoding={encoding}  "
          f"bias={bias}  logit_scale={logit_scale}  |  {n_params} params")
    print_model_summary(model, (1, 784), device)

    trackio.init(project=TRACKIO_PROJECT, name=name, config={
        "model": "optical_784x10", "encoding": encoding, "phase_insensitive": phase_insensitive,
        "bias": bias, "logit_scale": logit_scale, "init_scale": init_scale,
        "optimizer": optimizer_name, "warmup": warmup, "label_smoothing": label_smoothing,
        "n_params": n_params, "epochs": epochs, "lr": lr, "batch_size": batch_size,
        "weight_decay": weight_decay,
    })

    loader = DataLoader(TensorDataset(A_tr, y_tr_t), batch_size=batch_size, shuffle=True)
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup), eta_min=1e-4)
    if warmup > 0:
        warm = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warm, cosine], milestones=[warmup])
    else:
        scheduler = cosine
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    best_acc, best_state, history = 0.0, None, []
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running += loss.item() * len(xb)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            acc = (model(A_te).argmax(1) == y_te_t).float().mean().item()
        avg_loss = running / len(A_tr)
        if acc > best_acc:
            best_acc, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}

        trackio.log({"train_loss": avg_loss, "test_acc": acc,
                     "best_acc": best_acc, "lr": optimizer.param_groups[0]["lr"],
                     "epoch": epoch + 1})

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
            history.append({"epoch": epoch + 1, "train_loss": round(avg_loss, 6),
                            "test_acc": round(acc, 6), "best_acc": round(best_acc, 6),
                            "lr": round(optimizer.param_groups[0]["lr"], 8)})
            print(f"  epoch {epoch+1:4d}/{epochs}  loss={avg_loss:.4f}  "
                  f"acc={acc:.2%}  best={best_acc:.2%}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(A_te).argmax(1)
    cm = confusion_matrix(pred, y_te_t, N_CLASSES)

    per_class = per_class_acc(cm)
    trackio.log({**{f"class_{k}_acc": per_class[str(k)] for k in range(N_CLASSES)},
                 "epoch": epochs})
    trackio.finish()

    metrics_path = OUTPUT_DIR / f"metrics_{name}.json"
    with open(metrics_path, "w") as f:
        json.dump({"run_name": name, "model": "optical_784x10", "encoding": encoding,
                   "phase_insensitive": phase_insensitive,
                   "bias": bias, "logit_scale": logit_scale,
                   "optimizer": optimizer_name, "warmup": warmup,
                   "label_smoothing": label_smoothing,
                   "n_params": n_params, "best_acc": best_acc, "epochs": epochs, "lr": lr,
                   "batch_size": batch_size, "weight_decay": weight_decay,
                   "per_class_acc": per_class, "history": history}, f, indent=2)

    print(f"\nBest test accuracy: {best_acc:.2%}  ({n_params} params)")
    print(f"Metrics -> {metrics_path}")
    return {"best_acc": best_acc, "cm": cm}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--encoding", choices=["flat", "patch"], default="flat")
    p.add_argument("--bias", action="store_true", help="learnable per-class offset")
    p.add_argument("--logit-scale", action="store_true", help="learnable logit temperature")
    p.add_argument("--phase-insensitive", action="store_true",
                   help="phase-insensitive quadratic sum (drop cross-terms): Sum |w~|^2 |x~|^2")
    p.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    p.add_argument("--warmup", type=int, default=0, help="linear warmup epochs before cosine")
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--init-scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", type=str, default="optical")
    args = p.parse_args()
    train_optical(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                  weight_decay=args.weight_decay, encoding=args.encoding,
                  bias=args.bias, logit_scale=args.logit_scale,
                  init_scale=args.init_scale, optimizer_name=args.optimizer,
                  warmup=args.warmup, label_smoothing=args.label_smoothing,
                  phase_insensitive=args.phase_insensitive,
                  seed=args.seed, name=args.name)


if __name__ == "__main__":
    main()
