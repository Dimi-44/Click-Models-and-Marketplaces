"""Compact, lossless-for-selected-features representation of logged impressions.

The categorical dictionaries here are storage identifiers, not trained encoders.
Only the training subset determines the one-hot feature vocabulary later.
"""

from __future__ import annotations

import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from .data import READ_COLUMNS, USER_COLUMNS


DTYPE = np.dtype([
    ("minute_id", "<u2"), ("item_id", "u1"), ("position", "u1"),
    ("click", "u1"), *((field, "<u2") for field in USER_COLUMNS),
])
UNKNOWN_CODE = np.iinfo(np.uint16).max
ZIP_MEMBERS = {"BTS": "all/All bts.csv", "Random": "all/All random.csv"}


def manifest_minutes(path: Path) -> pd.DataFrame:
    minutes = pd.read_csv(path)
    needed = {"minute_utc", "date_utc", "n_bts", "split"}
    if not needed.issubset(minutes.columns) or minutes["minute_utc"].duplicated().any():
        raise ValueError("Invalid frozen minute-split manifest")
    if len(minutes) != 10080 or set(minutes["split"]) != {"train", "validation", "internal_test"}:
        raise ValueError("Frozen split differs from approved 10,080-minute protocol")
    return minutes


def _column_codes(values: pd.Series, codebook: dict[str, int], allow_new: bool) -> np.ndarray:
    if values.isna().any():
        raise ValueError("Unexpected null user category")
    if allow_new:
        for code in pd.unique(values):
            if code not in codebook:
                if len(codebook) >= UNKNOWN_CODE:
                    raise ValueError("Too many categorical levels for compact representation")
                codebook[code] = len(codebook)
    mapped = values.map(codebook)
    return mapped.fillna(UNKNOWN_CODE).to_numpy(dtype=np.uint16)


def pack_policy(
    archive: Path, policy: str, split_manifest: pd.DataFrame, packed_path: Path,
    *, codebooks: dict[str, dict[str, int]], n_expected: int,
    chunk_size: int = 200_000, allow_new_codes: bool = False,
) -> dict:
    """Write every original row once; never remove identical observations.

    Random is read only when invoked explicitly, after tuning lock is complete.
    """
    if policy not in ZIP_MEMBERS:
        raise ValueError(f"Unknown policy: {policy}")
    packed_path.parent.mkdir(parents=True, exist_ok=True)
    array = np.lib.format.open_memmap(packed_path, mode="w+", dtype=DTYPE, shape=(n_expected,))
    mapping = {minute: j for j, minute in enumerate(split_manifest["minute_utc"])}
    counts = np.zeros(len(split_manifest), dtype=np.int64)
    observed = 0
    dtype = {"Unnamed: 0": "int64", "item_id": "int16", "position": "int8", "click": "int8"}
    with zipfile.ZipFile(archive) as archive_zip:
        with archive_zip.open(ZIP_MEMBERS[policy]) as handle:
            reader = pd.read_csv(handle, usecols=list(READ_COLUMNS), dtype=dtype,
                                 chunksize=chunk_size)
            for chunk in reader:
                n = len(chunk)
                if observed + n > n_expected:
                    raise ValueError("Actual CSV exceeds expected row count")
                if not np.array_equal(chunk["Unnamed: 0"].to_numpy(),
                                      np.arange(observed, observed + n)):
                    raise ValueError("Original row index is not sequential")
                if chunk["click"].isna().any() or not chunk["click"].isin([0, 1]).all():
                    raise ValueError("Missing or invalid binary outcomes")
                if not chunk["item_id"].between(0, 79).all() or not chunk["position"].isin((1, 2, 3)).all():
                    raise ValueError("Item or position outside observed source support")
                minutes = chunk["timestamp"].str[:16].map(mapping)
                if minutes.isna().any():
                    raise ValueError("Timestamp minute missing from frozen BTS calendar")
                minute_ids = minutes.to_numpy(dtype=np.uint16)
                block = array[observed:observed + n]
                block["minute_id"] = minute_ids
                block["item_id"] = chunk["item_id"].to_numpy(dtype=np.uint8)
                block["position"] = chunk["position"].to_numpy(dtype=np.uint8)
                block["click"] = chunk["click"].to_numpy(dtype=np.uint8)
                for feature in USER_COLUMNS:
                    block[feature] = _column_codes(chunk[feature], codebooks[feature], allow_new_codes)
                counts += np.bincount(minute_ids, minlength=len(split_manifest))
                observed += n
    array.flush()
    del array
    if observed != n_expected:
        raise ValueError(f"Row count mismatch in {policy}: {observed} vs {n_expected}")
    if policy == "BTS" and not np.array_equal(counts, split_manifest["n_bts"].to_numpy()):
        raise AssertionError("Frozen split minute counts changed from approved pilot")
    return {"policy": policy, "rows": observed, "minutes": int(np.count_nonzero(counts)),
            "clicks": int(np.load(packed_path, mmap_mode="r")["click"].sum()),
            "unknown_feature_counts": {feature: int(np.sum(
                np.load(packed_path, mmap_mode="r")[feature] == UNKNOWN_CODE
            )) for feature in USER_COLUMNS}}


def save_codebooks(path: Path, dictionaries: dict[str, dict[str, int]]) -> None:
    path.write_text(json.dumps(dictionaries, sort_keys=True, indent=2), encoding="utf-8")


def load_codebooks(path: Path) -> dict[str, dict[str, int]]:
    return json.loads(path.read_text(encoding="utf-8"))


def split_indices(packed: np.ndarray, minutes: pd.DataFrame, split: str) -> np.ndarray:
    ids = np.flatnonzero(minutes["split"].to_numpy() == split)
    return np.flatnonzero(np.isin(packed["minute_id"], ids)).astype(np.int32)
