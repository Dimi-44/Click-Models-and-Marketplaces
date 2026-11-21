"""Stream only the BTS source; draw a deterministic minute-based pilot subset."""

from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
import zipfile

import pandas as pd


BTS_MEMBER = "all/All bts.csv"
USER_COLUMNS = tuple(f"user_feature_{i}" for i in range(4))
READ_COLUMNS = (
    "Unnamed: 0", "timestamp", "item_id", "position", "click", *USER_COLUMNS
)


def source_checksum(archive: Path) -> str:
    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_pilot_minute(minute: str, fraction: float, seed: int) -> bool:
    """Stable across machines and Python processes; independent of outcomes."""
    if not 0 < fraction <= 1:
        raise ValueError("pilot fraction must be in (0, 1]")
    payload = f"pilot|{seed}|{minute}".encode("utf-8")
    number = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return number / (1 << 64) < fraction


def stream_bts_pilot(
    archive: Path, *, minute_fraction: float, seed: int, chunk_size: int = 200_000
) -> tuple[pd.DataFrame, dict[str, int], int]:
    """One complete read-only BTS pass, retaining only selected minute blocks.

    Full-dataset minute counts are calculated without keeping all rows in RAM.
    The Random CSV is never opened by this function.
    """
    chunks: list[pd.DataFrame] = []
    minute_counts: Counter[str] = Counter()
    minute_selected: dict[str, bool] = {}
    scanned = 0
    dtype = {"Unnamed: 0": "int64", "item_id": "int16", "position": "int8", "click": "int8"}

    with zipfile.ZipFile(archive) as z:
        if BTS_MEMBER not in z.namelist():
            raise ValueError(f"BTS member {BTS_MEMBER!r} is missing")
        with z.open(BTS_MEMBER) as source:
            reader = pd.read_csv(source, usecols=list(READ_COLUMNS), dtype=dtype, chunksize=chunk_size)
            for chunk in reader:
                if list(chunk.columns) != list(READ_COLUMNS):
                    raise ValueError("Unexpected BTS schema or column ordering")
                if chunk[list(READ_COLUMNS)].isna().any().any():
                    raise ValueError("Unexpected null in required BTS columns")
                if not chunk["click"].isin([0, 1]).all():
                    raise ValueError("Click outcome is not binary")
                if not chunk["position"].isin([1, 2, 3]).all():
                    raise ValueError("Position outside {1,2,3}")
                if not chunk["item_id"].between(0, 79).all():
                    raise ValueError("Item outside 0..79")
                n = len(chunk)
                expected = range(scanned, scanned + n)
                if not (chunk["Unnamed: 0"].to_numpy() == expected).all():
                    raise ValueError("Sequential CSV record index has unexpected discontinuity")
                scanned += n
                minute = chunk["timestamp"].str.slice(0, 16)
                if minute.str.len().ne(16).any():
                    raise ValueError("Invalid timestamp format")
                minute_counts.update(minute.value_counts().to_dict())
                unique_minutes = minute.unique()
                for key in unique_minutes:
                    if key not in minute_selected:
                        minute_selected[key] = choose_pilot_minute(key, minute_fraction, seed)
                mask = minute.map(minute_selected).to_numpy(dtype=bool)
                if mask.any():
                    selected = chunk.loc[mask].copy()
                    selected["minute_utc"] = minute.loc[mask].to_numpy()
                    chunks.append(selected)
    if not chunks:
        raise ValueError("Pilot minute selection is empty; increase --pilot-minute-fraction")
    pilot = pd.concat(chunks, ignore_index=True)
    if not pilot["Unnamed: 0"].is_unique:
        raise AssertionError("Pilot rows must have unique original record indices")
    return pilot, dict(minute_counts), scanned
