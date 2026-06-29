"""Same/Different Fashion-MNIST pair dataset.

Identical task to the MNIST differentiater, but on Fashion-MNIST (28x28 grayscale
clothing images, 10 classes).  Each example is TWO images stacked into one
1568-vector; the label is whether they are the SAME class (1) or DIFFERENT (0),
balanced 50/50.

Same rationale: "same vs different" is a relational task driven by the
correlation between the two images (pairwise products img1_p * img2_p), which a
linear layer cannot form but the optical |S|^2 cross-terms can.

Train pairs are drawn from Fashion-MNIST-train, test pairs from the test split.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from torchvision import datasets, transforms

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"


def _load_split(train: bool):
    ds = datasets.FashionMNIST(DATA_DIR, train=train, download=True, transform=transforms.ToTensor())
    X = ds.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y = ds.targets.numpy()
    return X, y


def make_pairs(n: int, train: bool, seed: int = 0):
    """Return X (n, 1568) float32 and y (n,) in {0,1}, balanced same/different."""
    X, y = _load_split(train)
    rng = np.random.default_rng(seed)
    by_class = {c: np.where(y == c)[0] for c in range(10)}

    Xa = np.empty((n, 784), np.float32)
    Xb = np.empty((n, 784), np.float32)
    lab = np.empty(n, np.int64)
    for i in range(n):
        if i % 2 == 0:                                   # same class
            c = int(rng.integers(10))
            ia, ib = rng.choice(by_class[c], size=2, replace=False)
            lab[i] = 1
        else:                                            # different classes
            ca, cb = rng.choice(10, size=2, replace=False)
            ia = int(rng.choice(by_class[ca]))
            ib = int(rng.choice(by_class[cb]))
            lab[i] = 0
        Xa[i], Xb[i] = X[ia], X[ib]

    Xpair = np.concatenate([Xa, Xb], axis=1)             # (n, 1568)
    p = rng.permutation(n)
    return Xpair[p], lab[p]
