"""Training-only categorical codebook with explicit unknown columns."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse


ITEM_POSITION = ("item_id", "position")
CONTEXTUAL = ITEM_POSITION + tuple(f"user_feature_{i}" for i in range(4))
UNKNOWN_LABEL = "__UNKNOWN__"


class SparseCategoricalEncoder:
    def __init__(self, columns: tuple[str, ...]):
        self.columns = tuple(columns)
        self.levels_: dict[str, list[str]] = {}
        self.lookup_: dict[str, dict[str, int]] = {}
        self.offset_: dict[str, int] = {}
        self.n_features_: int | None = None

    @staticmethod
    def _strings(series: pd.Series) -> pd.Series:
        return series.astype("string").fillna("__MISSING__")

    def fit(self, train: pd.DataFrame) -> "SparseCategoricalEncoder":
        offset = 0
        for col in self.columns:
            if col not in train.columns:
                raise KeyError(f"Missing training predictor: {col}")
            levels = sorted(self._strings(train[col]).unique().tolist())
            if UNKNOWN_LABEL in levels:
                raise ValueError("Reserved unknown label occurs in training source data")
            self.levels_[col] = levels
            # Each feature block starts with an explicit unknown indicator column.
            self.lookup_[col] = {level: offset + j + 1 for j, level in enumerate(levels)}
            self.offset_[col] = offset
            offset += 1 + len(levels)
        self.n_features_ = offset
        return self

    def transform(self, records: pd.DataFrame) -> tuple[sparse.csr_matrix, dict[str, int]]:
        if self.n_features_ is None:
            raise RuntimeError("Encoder must be fitted on BTS training records first")
        n = len(records)
        row_index = np.arange(n, dtype=np.int32)
        all_columns = []
        unknown_counts: dict[str, int] = {}
        for col in self.columns:
            code = self._strings(records[col]).map(self.lookup_[col])
            is_unknown = code.isna().to_numpy()
            unknown_counts[col] = int(is_unknown.sum())
            indices = code.fillna(self.offset_[col]).to_numpy(dtype=np.int32)
            all_columns.append(indices)
        columns = np.concatenate(all_columns)
        rows = np.tile(row_index, len(self.columns))
        values = np.ones(len(columns), dtype=np.float32)
        matrix = sparse.csr_matrix(
            (values, (rows, columns)), shape=(n, self.n_features_), dtype=np.float32
        )
        if not np.all(np.diff(matrix.indptr) == len(self.columns)):
            raise AssertionError("Each record must activate exactly one indicator per predictor")
        return matrix, unknown_counts

    def description(self) -> dict:
        if self.n_features_ is None:
            raise RuntimeError("Encoder has not been fitted")
        return {"columns": list(self.columns), "levels": self.levels_,
                "unknown_policy": "explicit indicator; zero training coefficient unless fitted",
                "n_encoded_features": self.n_features_}
