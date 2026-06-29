"""Fixed, label-blind PCA-20 input encoder.

Fits PCA on MNIST pixel data (normalized to [0,1]), then maps 784-D inputs
to 20-D features in [0,1] via per-component min-max scaling, ready for SLM
amplitude encoding: phi = 2*arcsin(a).

The scaler is fit on the PCA-projected training set so that each component's
full dynamic range maps to [0,1]. This is label-blind: no digit labels are
used in fitting.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import joblib


class PCAEncoder:
    def __init__(self, n_components: int = 20):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        self._fitted = False

    def fit(self, X: np.ndarray) -> "PCAEncoder":
        """X: (N, 784) pixels in [0, 1]."""
        pca_out = self.pca.fit_transform(X)
        self.scaler.fit(pca_out)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Returns (N, n_components) amplitudes in [0, 1]."""
        self._check_fitted()
        return self.scaler.transform(self.pca.transform(X)).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        pca_out = self.pca.fit_transform(X)
        out = self.scaler.fit_transform(pca_out)
        self._fitted = True
        return out.astype(np.float32)

    def explained_variance_ratio(self) -> np.ndarray:
        self._check_fitted()
        return self.pca.explained_variance_ratio_

    def save(self, path: str | Path) -> None:
        self._check_fitted()
        joblib.dump({"pca": self.pca, "scaler": self.scaler}, path)

    @classmethod
    def load(cls, path: str | Path) -> "PCAEncoder":
        data = joblib.load(path)
        obj = cls.__new__(cls)
        obj.pca = data["pca"]
        obj.scaler = data["scaler"]
        obj.n_components = obj.pca.n_components_
        obj._fitted = True
        return obj

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("PCAEncoder has not been fitted. Call fit() first.")
