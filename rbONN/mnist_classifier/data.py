"""MNIST loading shared by the mnist_classifier benchmarks."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from torchvision import datasets, transforms

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"


def load_mnist(root: Path = DATA_DIR):
    """Return X_tr, y_tr, X_te, y_te as flat float32 [0,1] arrays / int labels."""
    t = transforms.ToTensor()
    tr = datasets.MNIST(root, train=True, download=True, transform=t)
    te = datasets.MNIST(root, train=False, download=True, transform=t)
    X_tr = tr.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    X_te = te.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    return X_tr, tr.targets.numpy(), X_te, te.targets.numpy()
