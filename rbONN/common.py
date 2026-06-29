"""Shared utilities for the RbONN benchmark scripts (mnist_classifier, fashion_differentiator).

Keeps the per-benchmark files focused on the *model* by centralising the
boilerplate that was previously copy-pasted into each one:
  * confusion_matrix  -- per-class accuracy bookkeeping
  * launch_trackio    -- start the local trackio dashboard
  * print_model_summary -- torchinfo summary with a Windows-console UTF-8 fix
"""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import torch


def confusion_matrix(pred: torch.Tensor, truth: torch.Tensor, n_classes: int) -> np.ndarray:
    """Integer confusion matrix; rows = truth, cols = prediction."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(truth.cpu().numpy(), pred.cpu().numpy()):
        cm[t, p] += 1
    return cm


def per_class_acc(cm: np.ndarray) -> dict[str, float]:
    """Diagonal / row-sum per class, as a {str(class): acc} dict."""
    return {str(k): float(cm[k, k] / cm[k].sum()) for k in range(cm.shape[0])}


def launch_trackio(project: str, port: int = 7860) -> None:
    """Start the local trackio dashboard in the background for `project`."""
    subprocess.Popen(
        [sys.executable, "-m", "trackio", "show", "--project", project],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    print(f"Trackio dashboard: http://localhost:{port}  (project: {project})")


def print_model_summary(model, input_size, device) -> None:
    """Print a torchinfo summary; never fatal (handles missing pkg / console encoding)."""
    try:
        from torchinfo import summary
    except Exception as e:                       # torchinfo not installed
        print(f"  [torchinfo summary skipped: {e}]")
        return
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # torchinfo draws box chars
    except Exception:
        pass
    try:
        print(summary(model, input_size=input_size, device=device, verbose=0,
                      col_names=("input_size", "output_size", "num_params", "trainable")))
    except Exception as e:
        print(f"  [torchinfo summary skipped: {e}]")
