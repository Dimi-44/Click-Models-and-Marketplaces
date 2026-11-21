"""Probability scoring and calibration diagnostics for rare binary outcomes."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, log_loss, roc_auc_score,
)


def probability_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=np.int8)
    p = np.asarray(probability, dtype=np.float64)
    if len(y) != len(p) or len(y) == 0 or not np.isfinite(p).all():
        raise ValueError("Nonfinite, missing, or mismatched probabilities")
    if not ((0 <= p) & (p <= 1)).all():
        raise ValueError("Probabilities outside [0,1]")
    return {
        "n": int(len(y)),
        "clicks": int(y.sum()),
        "ctr": float(y.mean()),
        "mean_prediction": float(p.mean()),
        "min_prediction": float(p.min()),
        "max_prediction": float(p.max()),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "average_precision": float(average_precision_score(y, p)) if y.any() else float("nan"),
    }


def reliability_bins(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> pd.DataFrame:
    data = pd.DataFrame({"actual": y_true, "predicted": probability})
    # Rank-based equal-count bins are robust when many scores have equal values.
    data["bin"] = np.minimum(
        (data["predicted"].rank(method="first").to_numpy() - 1) * bins // len(data),
        bins - 1,
    ).astype(int)
    return data.groupby("bin", as_index=False).agg(
        n=("actual", "size"), observed_ctr=("actual", "mean"),
        mean_prediction=("predicted", "mean"),
    )
