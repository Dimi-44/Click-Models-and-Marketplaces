"""BTS-only rate baselines and ridge logistic regression."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression


EPS = 1e-12


class ConstantCTR:
    def fit(self, train: pd.DataFrame) -> "ConstantCTR":
        self.rate_ = float(np.clip(train["click"].mean(), EPS, 1 - EPS))
        return self

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        return np.full(len(data), self.rate_, dtype=np.float64)


class SmoothedItemPositionCTR:
    def __init__(self, prior_strength: float = 200.0):
        if prior_strength <= 0:
            raise ValueError("Prior strength must be positive")
        self.prior_strength = prior_strength

    def fit(self, train: pd.DataFrame) -> "SmoothedItemPositionCTR":
        self.base_rate_ = float(train["click"].mean())
        grouped = train.groupby(["item_id", "position"])["click"].agg(["sum", "count"])
        smooth = (grouped["sum"] + self.prior_strength * self.base_rate_) / (
            grouped["count"] + self.prior_strength
        )
        self.rates_ = {tuple(map(int, key)): float(value) for key, value in smooth.items()}
        return self

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        result = np.array(
            [self.rates_.get((int(i), int(p)), self.base_rate_)
             for i, p in zip(data["item_id"], data["position"])], dtype=np.float64
        )
        return np.clip(result, EPS, 1 - EPS)


def fit_ridge_logistic(x_train: sparse.csr_matrix, y_train: np.ndarray, seed: int):
    """Fixed pilot settings only; no tuning or full-scale training."""
    model = LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=300, tol=1e-6,
        random_state=seed, fit_intercept=True,
    )
    model.fit(x_train, y_train)
    return model
