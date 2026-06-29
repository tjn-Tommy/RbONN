"""Benchmark 1: pure linear one-layer classifier on Same/Different Fashion-MNIST.

    logit_k = sum_i W[k,i] * x_i + b_k      x = [img1 ; img2]  (1568-vector)
    pred    = argmax_k logit_k              k in {different=0, same=1}

A linear layer is w1.img1 + w2.img2 + b -- a sum of two independent per-image
functions with no img1*img2 product, so it has no access to the correlation
between the two images and should sit near 50% (chance).

Usage
-----
  python -m rbONN.fashion_differentiater.benchmark_linear
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import trackio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..common import confusion_matrix, launch_trackio, per_class_acc, print_model_summary
from .datasets import make_pairs

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "output"
N_CLASSES = 2
N_IN = 1568
TRACKIO_PROJECT = "rbONN_fashion_diff"


def train_linear(epochs=80, batch_size=256, lr=1e-2, bias=True,
                 n_train=60000, n_test=10000, name="lin_fashion_diff") -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    launch_trackio(TRACKIO_PROJECT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  run: {name}")

    X_tr, y_tr = make_pairs(n_train, train=True, seed=0)
    X_te, y_te = make_pairs(n_test, train=False, seed=1)
    A_tr = torch.from_numpy(X_tr).to(device)
    A_te = torch.from_numpy(X_te).to(device)
    y_tr_t = torch.from_numpy(y_tr).to(device)
    y_te_t = torch.from_numpy(y_te).to(device)

    model = nn.Linear(N_IN, N_CLASSES, bias=bias).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: pure linear {N_IN} -> {N_CLASSES}  (bias={bias})  |  {n_params} params")
    print_model_summary(model, (1, N_IN), device)

    trackio.init(project=TRACKIO_PROJECT, name=name, config={
        "model": "linear", "n_in": N_IN, "bias": bias, "n_params": n_params,
        "epochs": epochs, "lr": lr, "batch_size": batch_size,
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
                     "epoch": epoch + 1})
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
        json.dump({"run_name": name, "task": "fashion_same_different", "model": "linear",
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
    p.add_argument("--no-bias", action="store_true")
    p.add_argument("--n-train", type=int, default=60000)
    p.add_argument("--n-test", type=int, default=10000)
    p.add_argument("--name", type=str, default="lin_fashion_diff")
    args = p.parse_args()
    train_linear(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                 bias=not args.no_bias, n_train=args.n_train, n_test=args.n_test, name=args.name)


if __name__ == "__main__":
    main()
