"""Date-balanced randomized minute-block split; never group on user hashes."""

from __future__ import annotations

from collections import defaultdict
import hashlib

import numpy as np
import pandas as pd


SPLITS = ("train", "validation", "internal_test")


def split_minute_counts(
    minute_counts: dict[str, int], *, seed: int, train_frac: float = 0.70,
    val_frac: float = 0.15
) -> dict[str, str]:
    if not 0 < train_frac < 1 or not 0 < val_frac < 1 - train_frac:
        raise ValueError("Invalid split fractions")
    daily: dict[str, list[str]] = defaultdict(list)
    for minute, count in minute_counts.items():
        if count <= 0 or len(minute) != 16:
            raise ValueError("Invalid minute key or count")
        daily[minute[:10]].append(minute)
    assignment: dict[str, str] = {}
    for date in sorted(daily):
        minutes = sorted(daily[date])
        payload = f"split|{seed}|{date}".encode("utf-8")
        date_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        rng = np.random.default_rng(date_seed)
        rng.shuffle(minutes)
        total = sum(minute_counts[minute] for minute in minutes)
        train_target = total * train_frac
        validation_target = total * (train_frac + val_frac)
        assigned_rows = 0
        for minute in minutes:
            if assigned_rows < train_target:
                group = "train"
            elif assigned_rows < validation_target:
                group = "validation"
            else:
                group = "internal_test"
            assignment[minute] = group
            assigned_rows += minute_counts[minute]
    if set(assignment) != set(minute_counts):
        raise AssertionError("Missing minute assignments")
    return assignment


def partition_pilot(
    pilot: pd.DataFrame, assignment: dict[str, str]
) -> dict[str, pd.DataFrame]:
    labels = pilot["minute_utc"].map(assignment)
    if labels.isna().any():
        raise AssertionError("Pilot contains unassigned minute blocks")
    out = {split: pilot.loc[labels.eq(split)].reset_index(drop=True) for split in SPLITS}
    if any(len(df) == 0 for df in out.values()):
        raise AssertionError("One pilot split is empty; increase pilot minute fraction")
    minutes = [set(df["minute_utc"]) for df in out.values()]
    if any(minutes[i] & minutes[j] for i in range(3) for j in range(i + 1, 3)):
        raise AssertionError("Minute block overlap across partitions")
    indices = [set(df["Unnamed: 0"]) for df in out.values()]
    if any(indices[i] & indices[j] for i in range(3) for j in range(i + 1, 3)):
        raise AssertionError("Record index overlap across partitions")
    return out
