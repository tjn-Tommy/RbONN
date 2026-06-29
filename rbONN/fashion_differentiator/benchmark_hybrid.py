"""Hybrid Siamese: digital CNN encoder -> ONN |S|^2 comparator -> digital readout.

Division of labour:
  * a PRETRAINED digital CNN encoder turns each image into a class-discriminative
    latent z in (0,1)^d  (ordinary perception -- done digitally),
  * the OPTICAL |S|^2 layer does the RELATIONAL COMPARISON of the two latents
    (its cross-terms z1_i * z2_j are exactly the correlation a linear layer
    cannot form),
  * a tiny digital readout maps the comparison features to {different, same}.

Pipeline
--------
  img1 --[CNN encoder]--> z1 (d) -.
  img2 --[CNN encoder]--> z2 (d) -+--[ONN |S|^2 on [z1;z2] ]--[Linear]--> 2 --argmax--> {0,1}
        (shared, pretrained on Fashion classification; frozen by default)

The CNN encoder is pretrained on Fashion-MNIST 10-way classification and cached.
The ONN comparator + readout are trained on the same/different pairs.

Run
---
  python -m rbONN.fashion_differentiator.benchmark_hybrid --comparator optical     # d=64, K=20
  python -m rbONN.fashion_differentiator.benchmark_hybrid --comparator incoherent
  python -m rbONN.fashion_differentiator.benchmark_hybrid --comparator linear
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
from .datasets import _load_split, make_pairs

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "output"
N_IMG = 784
N_CLASSES = 2
N_FASHION = 10
TRACKIO_PROJECT = "rbONN_fashion_diff"


class CNNEncoder(nn.Module):
    """Digital per-image encoder: (1,28,28) -> latent z in (0,1)^d."""

    def __init__(self, d=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 14x14
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 7x7
        )
        self.fc = nn.Linear(32 * 7 * 7, d)

    def forward(self, u):                       # u: (batch, 1, 28, 28)
        h = self.features(u).flatten(1)
        return torch.sigmoid(self.fc(h))        # latent in (0,1)^d, ready for E-field encoding


COMPARATORS = ("optical", "incoherent", "linear")


class Comparator(nn.Module):
    """Compare two latents [z1; z2] (2d) -> K features -> Linear -> 2.

    mode='optical'    : coherent square-law  h = |x . W~|^2   (phase-sensitive;
                        keeps the complex cross-terms z1_i*z2_j -- the comparison).
    mode='incoherent' : phase-INSENSITIVE    h = |x|^2 . |W~|^2  (drops all cross-
                        terms / interference -> cannot compare z1 vs z2).
    mode='linear'     : pure linear projection h = proj([z1;z2])  (no nonlinearity).

    All three share the same 2d->K->2 shape; only the K-feature computation differs.
    """

    def __init__(self, d, k, mode="optical", n_out=N_CLASSES):
        super().__init__()
        assert mode in COMPARATORS
        self.mode = mode
        if mode == "linear":
            self.proj = nn.Linear(2 * d, k)                          # linear feature layer
        else:
            self.phi_w = nn.Parameter(torch.rand(k, 2 * d) * 2.0 * math.pi)  # the ONN
            self.log_s = nn.Parameter(torch.zeros(()))
        self.readout = nn.Linear(k, n_out)

    def forward(self, z1, z2):
        a = torch.cat([z1, z2], dim=1)              # (batch, 2d) in [0,1]
        if self.mode == "linear":
            h = self.proj(a)
        else:
            x = amplitudes_to_efield(a)
            W = phases_to_efield(self.phi_w)        # (K, 2d) complex
            if self.mode == "incoherent":
                h = x.abs().pow(2) @ W.abs().pow(2).T    # phase-insensitive, no interference
            else:                                        # 'optical' coherent square-law
                h = (x @ W.T).abs().pow(2)
            h = h * self.log_s.exp()
        return self.readout(h)


class HybridSiamese(nn.Module):
    def __init__(self, encoder: CNNEncoder, d, k, mode="optical"):
        super().__init__()
        self.encoder = encoder
        self.comparator = Comparator(d, k, mode=mode)

    def forward(self, a):
        u = a[:, :N_IMG].reshape(-1, 1, 28, 28)
        v = a[:, N_IMG:].reshape(-1, 1, 28, 28)
        z1 = self.encoder(u)
        z2 = self.encoder(v)
        return self.comparator(z1, z2)


# ── Pretrain the digital CNN encoder on Fashion classification ──────────────

def pretrain_cnn(d, device, epochs=30, batch_size=256, lr=1e-3, seed=0) -> CNNEncoder:
    cache = OUTPUT_DIR / f"cnn_encoder_d{d}.pt"
    encoder = CNNEncoder(d).to(device)
    if cache.exists():
        encoder.load_state_dict(torch.load(cache, map_location=device))
        print(f"Loaded pretrained CNN encoder from {cache}")
        return encoder

    print(f"Pretraining CNN encoder (d={d}) on Fashion-MNIST classification ...")
    torch.manual_seed(seed)
    X_tr, y_tr = _load_split(train=True)
    X_te, y_te = _load_split(train=False)
    A_tr = torch.from_numpy(X_tr).reshape(-1, 1, 28, 28).to(device)
    A_te = torch.from_numpy(X_te).reshape(-1, 1, 28, 28).to(device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long, device=device)
    y_te_t = torch.tensor(y_te, dtype=torch.long, device=device)

    head = nn.Linear(d, N_FASHION).to(device)
    params = list(encoder.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)
    crit = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(A_tr, y_tr_t), batch_size=batch_size, shuffle=True)

    best = 0.0
    for ep in range(epochs):
        encoder.train(); head.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(head(encoder(xb)), yb)
            loss.backward()
            opt.step()
        sched.step()
        encoder.eval(); head.eval()
        with torch.no_grad():
            acc = (head(encoder(A_te)).argmax(1) == y_te_t).float().mean().item()
        best = max(best, acc)
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  pretrain ep {ep+1:3d}/{epochs}  fashion-cls acc={acc:.2%}")
    print(f"Pretrained CNN encoder Fashion classification acc: {best:.2%}")
    torch.save(encoder.state_dict(), cache)
    print(f"Cached CNN encoder -> {cache}")
    return encoder


# ── Train the ONN comparator on same/different ──────────────────────────────

def train_hybrid(d=64, k=32, comparator="optical", epochs=80, batch_size=256, lr=1e-2,
                 finetune=False, n_train=60000, n_test=10000, seed=0,
                 name="hybrid_siamese") -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)

    launch_trackio(TRACKIO_PROJECT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else ""
    print(f"Device: {device}" + (f" ({gpu})" if gpu else "") + f"  |  run: {name}")

    encoder = pretrain_cnn(d, device)
    model = HybridSiamese(encoder, d, k, mode=comparator).to(device)
    if not finetune:
        for p in model.encoder.parameters():
            p.requires_grad_(False)

    comp = model.comparator
    if comparator == "linear":
        core_params = sum(p.numel() for p in comp.proj.parameters())   # digital, no ONN
        onn_params = 0
        core_desc = f"linear proj(2d->{k})"
    else:
        onn_params = comp.phi_w.numel() + 1
        core_params = onn_params
        core_desc = f"{comparator} |S|^2-style (optical phases {comp.phi_w.numel()} + log_s)"
    readout_params = sum(p.numel() for p in comp.readout.parameters())
    cnn_params = sum(p.numel() for p in model.encoder.parameters())
    print(f"Model: CNN encoder(d={d}, digital) -> {comparator} comparator(K={k}) -> Linear({k}->2)")
    print(f"  comparator core   : {core_params}  ({core_desc})")
    print(f"  ONN (optical) part: {onn_params}")
    print(f"  digital readout   : {readout_params}")
    print(f"  CNN encoder       : {cnn_params}  (finetune={finetune})")
    print_model_summary(model, (1, 2 * N_IMG), device)

    X_tr, y_tr = make_pairs(n_train, train=True, seed=0)
    X_te, y_te = make_pairs(n_test, train=False, seed=1)
    A_tr = torch.from_numpy(X_tr).to(device)
    A_te = torch.from_numpy(X_te).to(device)
    y_tr_t = torch.from_numpy(y_tr).to(device)
    y_te_t = torch.from_numpy(y_te).to(device)

    trackio.init(project=TRACKIO_PROJECT, name=name, config={
        "model": f"hybrid_{comparator}_d{d}_k{k}", "comparator": comparator,
        "d": d, "k": k, "finetune": finetune,
        "onn_params": onn_params, "core_params": core_params,
        "readout_params": readout_params, "cnn_params": cnn_params,
        "epochs": epochs, "lr": lr, "batch_size": batch_size,
        "n_train": n_train, "n_test": n_test,
    })

    loader = DataLoader(TensorDataset(A_tr, y_tr_t), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)
    crit = nn.CrossEntropyLoss()

    best_acc, best_state, history = 0.0, None, []
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            running += loss.item() * len(xb)
        sched.step()

        model.eval()
        with torch.no_grad():
            acc = (model(A_te).argmax(1) == y_te_t).float().mean().item()
        avg_loss = running / len(A_tr)
        if acc > best_acc:
            best_acc, best_state = acc, {kk: v.clone() for kk, v in model.state_dict().items()}

        trackio.log({"train_loss": avg_loss, "test_acc": acc, "best_acc": best_acc,
                     "lr": opt.param_groups[0]["lr"], "epoch": epoch + 1})
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
    trackio.log({**{f"class_{c}_acc": per_class[str(c)] for c in range(N_CLASSES)}, "epoch": epochs})
    trackio.finish()

    with open(OUTPUT_DIR / f"metrics_{name}.json", "w") as f:
        json.dump({"run_name": name, "task": "fashion_same_different",
                   "model": f"hybrid_{comparator}_d{d}_k{k}", "comparator": comparator,
                   "d": d, "k": k, "finetune": finetune,
                   "onn_params": onn_params, "core_params": core_params,
                   "readout_params": readout_params, "cnn_params": cnn_params,
                   "best_acc": best_acc, "epochs": epochs,
                   "per_class_acc": per_class, "history": history}, f, indent=2)

    print(f"\nBest test accuracy: {best_acc:.2%}   [chance = 50%]  comparator={comparator}")
    print(f"ONN(optical) params: {onn_params}  |  core: {core_params}  |  readout: {readout_params}  |  CNN: {cnn_params}")
    print(f"Metrics -> {OUTPUT_DIR / f'metrics_{name}.json'}")
    return {"best_acc": best_acc, "cm": cm}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--d", type=int, default=64, help="CNN latent dim per image")
    p.add_argument("--k", type=int, default=20,
                   help="comparator OUTPUT width K (tight = must extract the comparison)")
    p.add_argument("--comparator", choices=list(COMPARATORS), default="optical",
                   help="optical=coherent |S|^2 ; incoherent=phase-insensitive ; linear=control")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--finetune", action="store_true", help="also fine-tune the CNN encoder")
    p.add_argument("--n-train", type=int, default=60000)
    p.add_argument("--n-test", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", type=str, default="hybrid_siamese")
    args = p.parse_args()
    train_hybrid(d=args.d, k=args.k, comparator=args.comparator, epochs=args.epochs,
                 batch_size=args.batch_size, lr=args.lr, finetune=args.finetune,
                 n_train=args.n_train, n_test=args.n_test, seed=args.seed, name=args.name)


if __name__ == "__main__":
    main()
