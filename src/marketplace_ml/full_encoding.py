"""Memory-efficient training-only one-hot encoder for compact logged records."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from .data import USER_COLUMNS


ITEM_POSITION = ("item_id", "position")
CONTEXTUAL = ITEM_POSITION + USER_COLUMNS


class CompactSparseEncoder:
    def __init__(self, fields: tuple[str, ...]):
        self.fields = fields
        self.levels: dict[str, np.ndarray] = {}
        self.offsets: dict[str, int] = {}
        self.width = 0

    def fit(self, rows: np.ndarray) -> "CompactSparseEncoder":
        offset = 0
        for feature in self.fields:
            values = np.unique(rows[feature])
            self.levels[feature] = values
            self.offsets[feature] = offset
            offset += 1 + len(values)  # Dedicated UNKNOWN indicator for each feature
        self.width = offset
        return self

    def transform(self, rows: np.ndarray) -> tuple[sparse.csr_matrix, dict[str, int]]:
        if not self.levels:
            raise RuntimeError("Encoder must be fitted on BTS training subset")
        n, k = len(rows), len(self.fields)
        indices = np.empty(n * k, dtype=np.int32)
        unknown_counts: dict[str, int] = {}
        for j, feature in enumerate(self.fields):
            values = rows[feature]
            levels = self.levels[feature]
            found = np.searchsorted(levels, values)
            matched = found < len(levels)
            matched[matched] = levels[found[matched]] == values[matched]
            unknown_counts[feature] = int(np.count_nonzero(~matched))
            indices[j::k] = self.offsets[feature] + np.where(matched, found + 1, 0)
        indptr = np.arange(0, (n + 1) * k, k, dtype=np.int32)
        data = np.ones(n * k, dtype=np.float32)
        matrix = sparse.csr_matrix((data, indices, indptr), shape=(n, self.width))
        if matrix.nnz != n * k:
            raise AssertionError("Unexpected one-hot matrix density")
        return matrix, unknown_counts

    def metadata(self) -> dict:
        return {"fields": list(self.fields), "training_levels": {
            name: [int(x) for x in values] for name, values in self.levels.items()
        }, "n_features": self.width,
                "unseen_rule": "Per-feature dedicated UNKNOWN indicator; no cross-policy code mapping"}
