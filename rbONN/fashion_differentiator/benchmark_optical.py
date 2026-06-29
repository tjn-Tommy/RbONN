"""Benchmark 2: optical NL layer (no saturation) on Same/Different Fashion-MNIST.

    logit_k = | sum_i W~[k,i] * x~_i |^2     x = [img1 ; img2]  (1568-vector)
    pred    = argmax_k logit_k               k in {different=0, same=1}

Same single-layer structure as the linear benchmark (1568 -> 2), but with complex
E-field weights and square-law detection.  Expanding the square,

    |S_k|^2 = sum_(i,j) W~_ki W~_kj* x~_i x~_j*

contains pairwise products x~_i x~_j -- including cross terms between img1 and
img2 -- which are the correlations the "same vs different" decision needs and the
linear model cannot represent.  No saturation term.

Usage
-----
  python -m rbONN.fashion_differentiater.benchmark_optical
  python -m rbONN.fashion_differentiater.benchmark_optical --bias --logit-scale --epochs 120
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import trackio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..common import confusion_matrix, launch_trackio, per_class_acc, print_model_summary
from ..mnist_classifier.twin import amplitudes_to_efield, phases_to_efield
from .datasets import make_pairs

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "output"
N_CLASSES = 2
N_IMG = 784       # pixels per single image
N_IN = 1568       # two stacked images
TRACKIO_PROJECT = "rbONN_fashion_diff"


class OpticalLayer(nn.Module):
    """Single optical layer: one |S|^2 per output class (rank-1 per class)."""

    def __init__(self, n_in=N_IN, n_out=N_CLASSES, bias=False, logit_scale=False,
                 phase_insensitive=False):
        super().__init__()
        self.phi_w = nn.Parameter(torch.rand(n_out, n_in) * 2.0 * math.pi)
        self.bias = nn.Parameter(torch.zeros(n_out)) if bias else None
        self.log_s = nn.Parameter(torch.zeros(())) if logit_scale else None
        self.phase_insensitive = phase_insensitive

    def forward(self, a):
        x = amplitudes_to_efield(a)        # (batch, 1568) complex
        W = phases_to_efield(self.phi_w)   # (n_out, 1568) complex
        if self.phase_insensitive:
            out = x.abs().pow(2) @ W.abs().pow(2).T   # phase-insensitive: no cross-terms
        else:
            out = (x @ W.T).abs().pow(2)              # phase-sensitive |S|^2
        if self.log_s is not None:
            out = out * self.log_s.exp()
        if self.bias is not None:
            out = out + self.bias
        return out


class OpticalMLP(nn.Module):
    """Widened: hidden layer of `hidden` optical |S|^2 units + linear readout.

        x (1568) --|S|^2--> h (hidden real features) --Linear--> logits (2) --argmax--> {0,1}

    K hidden units -> rank-K quadratic decision discriminant, breaking the
    single-layer rank-1 ceiling.  The linear readout is the decision layer.
    """

    def __init__(self, n_in=N_IN, hidden=20, n_out=N_CLASSES, bias=True, logit_scale=True,
                 phase_insensitive=False):
        super().__init__()
        self.phi_w = nn.Parameter(torch.rand(hidden, n_in) * 2.0 * math.pi)  # hidden optical units
        self.log_s = nn.Parameter(torch.zeros(())) if logit_scale else None
        self.readout = nn.Linear(hidden, n_out, bias=bias)                   # decision layer
        self.phase_insensitive = phase_insensitive

    def forward(self, a):
        x = amplitudes_to_efield(a)            # (batch, 1568) complex
        W = phases_to_efield(self.phi_w)       # (hidden, 1568) complex
        if self.phase_insensitive:
            h = x.abs().pow(2) @ W.abs().pow(2).T   # phase-insensitive: no cross-terms
        else:
            h = (x @ W.T).abs().pow(2)              # phase-sensitive |S|^2 (interference)
        if self.log_s is not None:
            h = h * self.log_s.exp()
        return self.readout(h)                 # (batch, n_out) class logits


class OpticalCrossMLP(nn.Module):
    """Cross-term-only variant: keep ONLY the img1-img2 interference term.

    Split the pair into u = img1 (first 784) and v = img2 (last 784).  Per hidden
    unit j with separate weight rows a_j (img1) and b_j (img2):

        A_j = a_j . u~        (complex scalar)
        B_j = b_j . v~        (complex scalar)
        h_j = Re( A_j * conj(B_j) )      <- the cross term only

    This is exactly the  2*Re[(a.u)(b.v)*]  piece of  |a.u + b.v|^2, with the two
    within-image self-terms |A_j|^2 and |B_j|^2 thrown away.  h_j is REAL and
    signed (built-in negative evidence), unlike |S|^2 >= 0.  Same parameter count
    as OpticalMLP (2 * hidden * 784 phases).  Linear readout -> 2 classes.
    """

    def __init__(self, n_img=N_IMG, hidden=20, n_out=N_CLASSES, bias=True, logit_scale=True):
        super().__init__()
        self.n_img = n_img
        self.phi_a = nn.Parameter(torch.rand(hidden, n_img) * 2.0 * math.pi)  # img1 weights
        self.phi_b = nn.Parameter(torch.rand(hidden, n_img) * 2.0 * math.pi)  # img2 weights
        self.log_s = nn.Parameter(torch.zeros(())) if logit_scale else None
        self.readout = nn.Linear(hidden, n_out, bias=bias)

    def forward(self, a):
        u = a[:, :self.n_img]                  # img1  (batch, 784)
        v = a[:, self.n_img:]                  # img2  (batch, 784)
        xu = amplitudes_to_efield(u)
        xv = amplitudes_to_efield(v)
        A = xu @ phases_to_efield(self.phi_a).T   # (batch, hidden) complex  a_j . u
        B = xv @ phases_to_efield(self.phi_b).T   # (batch, hidden) complex  b_j . v
        h = (A * B.conj()).real                   # (batch, hidden) cross term Re(A B*)
        if self.log_s is not None:
            h = h * self.log_s.exp()
        return self.readout(h)                    # (batch, n_out) class logits


def train_optical(epochs=80, batch_size=256, lr=1e-2, bias=False, logit_scale=False,
                  hidden=0, cross=False, phase_insensitive=False,
                  n_train=60000, n_test=10000, seed=0,
                  name="opt_fashion_diff") -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)

    launch_trackio(TRACKIO_PROJECT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else ""
    print(f"Device: {device}" + (f" ({gpu})" if gpu else "") + f"  |  run: {name}")

    X_tr, y_tr = make_pairs(n_train, train=True, seed=0)
    X_te, y_te = make_pairs(n_test, train=False, seed=1)
    A_tr = torch.from_numpy(X_tr).to(device)
    A_te = torch.from_numpy(X_te).to(device)
    y_tr_t = torch.from_numpy(y_tr).to(device)
    y_te_t = torch.from_numpy(y_te).to(device)

    quad = "phase-insensitive" if phase_insensitive else "phase-sensitive"
    if hidden > 0 and cross:
        model = OpticalCrossMLP(N_IMG, hidden=hidden, n_out=N_CLASSES,
                                bias=True, logit_scale=True).to(device)
        arch = f"optical_CrossMLP_h{hidden}"
        print(f"Model: optical CROSS MLP  Re(A_j B_j*) x{hidden}->Linear->{N_CLASSES}", end="")
    elif hidden > 0:
        model = OpticalMLP(N_IN, hidden=hidden, n_out=N_CLASSES,
                           bias=bias or True, logit_scale=logit_scale or True,
                           phase_insensitive=phase_insensitive).to(device)
        arch = f"optical_MLP_h{hidden}" + ("_phaseins" if phase_insensitive else "")
        print(f"Model: optical MLP {N_IN}->{quad} quad x{hidden}->Linear->{N_CLASSES}", end="")
    else:
        model = OpticalLayer(N_IN, N_CLASSES, bias=bias, logit_scale=logit_scale,
                             phase_insensitive=phase_insensitive).to(device)
        arch = "optical_NL" + ("_phaseins" if phase_insensitive else "")
        print(f"Model: optical {N_IN}->{N_CLASSES} {quad} quad  bias={bias}  logit_scale={logit_scale}", end="")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  |  {n_params} params")
    print_model_summary(model, (1, N_IN), device)

    trackio.init(project=TRACKIO_PROJECT, name=name, config={
        "model": arch, "n_in": N_IN, "hidden": hidden, "cross": cross,
        "phase_insensitive": phase_insensitive,
        "bias": bias, "logit_scale": logit_scale,
        "n_params": n_params, "epochs": epochs, "lr": lr, "batch_size": batch_size,
        "n_train": n_train, "n_test": n_test,
    })

    loader = DataLoader(TensorDataset(A_tr, y_tr_t), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
    criterion = nn.CrossEntropyLoss()

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

        trackio.log({"train_loss": avg_loss, "test_acc": acc, "best_acc": best_acc,
                     "lr": optimizer.param_groups[0]["lr"], "epoch": epoch + 1})
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
            history.append({"epoch": epoch + 1, "train_loss": round(avg_loss, 6),
                            "test_acc": round(acc, 6), "best_acc": round(best_acc, 6)})
            print(f"  epoch {epoch+1:4d}/{epochs}  loss={avg_loss:.4f}  acc={acc:.2%}  best={best_acc:.2%}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(A_te).argmax(1)
    cm = confusion_matrix(pred, y_te_t, N_CLASSES)
    per_class = per_class_acc(cm)
    trackio.log({**{f"class_{k}_acc": per_class[str(k)] for k in range(N_CLASSES)}, "epoch": epochs})
    trackio.finish()

    with open(OUTPUT_DIR / f"metrics_{name}.json", "w") as f:
        json.dump({"run_name": name, "task": "fashion_same_different", "model": arch,
                   "hidden": hidden, "cross": cross, "phase_insensitive": phase_insensitive,
                   "bias": bias, "logit_scale": logit_scale,
                   "n_params": n_params, "best_acc": best_acc, "epochs": epochs,
                   "per_class_acc": per_class, "history": history}, f, indent=2)

    print(f"\nBest test accuracy: {best_acc:.2%}  ({n_params} params)  [chance = 50%]")
    print(f"Metrics -> {OUTPUT_DIR / f'metrics_{name}.json'}")
    return {"best_acc": best_acc, "cm": cm}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--bias", action="store_true", help="learnable per-class offset (single layer)")
    p.add_argument("--logit-scale", action="store_true", help="learnable logit temperature (single layer)")
    p.add_argument("--hidden", type=int, default=0,
                   help="K optical |S|^2 hidden units + linear readout (0 = single layer)")
    p.add_argument("--cross", action="store_true",
                   help="cross-term-only model: keep Re(A_j B_j*), drop |A_j|^2+|B_j|^2 (needs --hidden)")
    p.add_argument("--phase-insensitive", action="store_true",
                   help="phase-insensitive quadratic sum (drop cross-terms): Sum |w~|^2 |x~|^2")
    p.add_argument("--n-train", type=int, default=60000)
    p.add_argument("--n-test", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", type=str, default="opt_fashion_diff")
    args = p.parse_args()
    train_optical(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                  bias=args.bias, logit_scale=args.logit_scale, hidden=args.hidden,
                  cross=args.cross, phase_insensitive=args.phase_insensitive,
                  n_train=args.n_train, n_test=args.n_test,
                  seed=args.seed, name=args.name)


if __name__ == "__main__":
    main()
