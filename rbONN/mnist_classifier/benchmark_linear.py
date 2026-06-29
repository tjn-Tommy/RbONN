"""Benchmark: pure linear classifier on raw 28x28 MNIST pixels.

This is the reference upper bound for a *single* weight layer with no optical
physics and no nonlinearity:

    logits_k = sum_i  W[k,i] * x_i        (k = class, i = pixel)
    pred     = argmax_k logits_k

10 classes x 784 pixels = 7,840 real weights (+ 10 bias).  Plain multinomial
logistic regression.  Trained identically to the RbONN twin (Adam + cosine LR,
CrossEntropyLoss) so the numbers are directly comparable, and the metrics JSON
is written in the same schema as train_sim.py.

Usage
-----
  python -m rbONN.benchmark_linear
  python -m rbONN.benchmark_linear --epochs 80 --no-bias --name linear_nobias
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..common import confusion_matrix, per_class_acc, print_model_summary
from .data import load_mnist

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "output"
N_CLASSES = 10


def train_linear(
    epochs: int = 80,
    batch_size: int = 256,
    lr: float = 1e-2,
    bias: bool = True,
    name: str = "linear_baseline",
) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else ""
    print(f"Device: {device}" + (f" ({gpu})" if gpu else ""))

    X_tr, y_tr, X_te, y_te = load_mnist()
    A_tr = torch.from_numpy(X_tr).to(device)
    A_te = torch.from_numpy(X_te).to(device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long, device=device)
    y_te_t = torch.tensor(y_te, dtype=torch.long, device=device)

    model = nn.Linear(784, N_CLASSES, bias=bias).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: pure linear 784 -> {N_CLASSES}  (bias={bias})  |  {n_params} parameters")
    print_model_summary(model, (1, 784), device)

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

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
            history.append({
                "epoch": epoch + 1,
                "train_loss": round(avg_loss, 6),
                "test_acc": round(acc, 6),
                "best_acc": round(best_acc, 6),
                "lr": round(optimizer.param_groups[0]["lr"], 8),
            })
            print(f"  epoch {epoch+1:4d}/{epochs}  loss={avg_loss:.4f}  "
                  f"acc={acc:.2%}  best={best_acc:.2%}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(A_te).argmax(1)
    cm = confusion_matrix(pred, y_te_t, N_CLASSES)

    metrics_path = OUTPUT_DIR / f"metrics_{name}.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "run_name": name,
            "detection_mode": "linear",
            "encoding": "raw_784px_pure_linear",
            "n_params": n_params,
            "best_acc": best_acc,
            "epochs": epochs,
            "per_class_acc": per_class_acc(cm),
            "history": history,
        }, f, indent=2)

    print(f"\nBest test accuracy: {best_acc:.2%}  ({n_params} params)")
    print(f"Metrics -> {metrics_path}")
    return {"best_acc": best_acc, "cm": cm}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--no-bias", action="store_true", help="pure weighted sum, no bias term")
    p.add_argument("--name", type=str, default="linear_baseline")
    args = p.parse_args()
    train_linear(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        bias=not args.no_bias, name=args.name,
    )


if __name__ == "__main__":
    main()
