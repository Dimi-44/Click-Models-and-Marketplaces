"""Freeze compact BTS rows and the approved training permutations before fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from .data import USER_COLUMNS, source_checksum
from .full_data import load_codebooks, manifest_minutes, pack_policy, save_codebooks, split_indices


SIZES = (100_000, 300_000, 1_000_000, 3_000_000)
SEED = 20261114


def minute_nested_order(packed: np.ndarray, train_indices: np.ndarray, seed: int) -> np.ndarray:
    """Randomly sequence train-only minute blocks; shuffle rows *within* blocks.

    Every prefix is nested. An exact requested row count may end part-way
    through one minute; no minute can leak into validation/internal test.
    """
    minute_id = packed["minute_id"][train_indices]
    sorted_indices = train_indices[np.argsort(minute_id, kind="stable")]
    counts = np.bincount(minute_id, minlength=10080)
    ends = np.cumsum(counts)
    starts = ends - counts
    minutes = np.flatnonzero(counts)
    rng = np.random.default_rng(seed)
    rng.shuffle(minutes)
    order = np.empty(len(train_indices), dtype=np.int32)
    cursor = 0
    for minute in minutes:
        block = sorted_indices[starts[minute]:ends[minute]].copy()
        rng.shuffle(block)
        order[cursor:cursor + len(block)] = block
        cursor += len(block)
    if cursor != len(train_indices):
        raise AssertionError("Nested minute sampling lost BTS rows")
    return order


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(archive: Path, pilot_dir: Path, output: Path) -> None:
    started = time.monotonic()
    archive = archive.resolve(strict=True)
    pilot_dir = pilot_dir.resolve(strict=True)
    output.mkdir(parents=True, exist_ok=True)
    pilot_metadata = json.loads((pilot_dir / "manifest.json").read_text())
    archive_hash = source_checksum(archive)
    if archive_hash != pilot_metadata["archive_sha256"]:
        raise ValueError("Source dataset differs from approved pilot SHA-256")
    original_split = pilot_dir / "minute_split_manifest.csv"
    frozen_copy = output / "minute_split_manifest.csv"
    if frozen_copy.exists() and sha256(frozen_copy) != sha256(original_split):
        raise ValueError("Existing frozen split does not match the pilot")
    shutil.copy2(original_split, frozen_copy)
    split = manifest_minutes(frozen_copy)
    if int(split["n_bts"].sum()) != pilot_metadata["bts_total_scanned_rows"]:
        raise ValueError("Inconsistent BTS row totals")

    packed_path = output / "bts_compact.npy"
    codebook_path = output / "bts_storage_codebooks.json"
    if packed_path.exists() or codebook_path.exists():
        if not (packed_path.exists() and codebook_path.exists()):
            raise ValueError("Partial compact BTS dataset; investigate before overwriting")
        dictionaries = load_codebooks(codebook_path)
        packed = np.load(packed_path, mmap_mode="r")
        if len(packed) != int(split["n_bts"].sum()):
            raise ValueError("Existing compact BTS shape mismatch")
        print("Reusing existing compact BTS data", flush=True)
        ingest = {"policy": "BTS", "rows": int(len(packed)),
                  "clicks": int(packed["click"].sum()), "resumed": True}
    else:
        dictionaries = {field: {} for field in USER_COLUMNS}
        print("Reading and packing all BTS rows; Random remains unopened", flush=True)
        ingest = pack_policy(
            archive, "BTS", split, packed_path, codebooks=dictionaries,
            n_expected=int(split["n_bts"].sum()), allow_new_codes=True,
        )
        save_codebooks(codebook_path, dictionaries)
        packed = np.load(packed_path, mmap_mode="r")
    print("BTS compact storage complete", flush=True)

    membership: dict[str, int] = {}
    for part in ("train", "validation", "internal_test"):
        path = output / f"indices_{part}.npy"
        if path.exists():
            indices = np.load(path)
        else:
            indices = split_indices(packed, split, part)
            np.save(path, indices)
        membership[part] = len(indices)
        expected = int(split.loc[split["split"].eq(part), "n_bts"].sum())
        if len(indices) != expected:
            raise ValueError(f"{part} frozen row membership changed")
    if sum(membership.values()) != len(packed):
        raise AssertionError("Missing or duplicated row membership")

    train = np.load(output / "indices_train.npy", mmap_mode="r")
    for rep in range(3):
        order_path = output / f"train_order_minute_rep{rep}.npy"
        if order_path.exists():
            order = np.load(order_path, mmap_mode="r")
        else:
            order = minute_nested_order(packed, np.asarray(train), SEED + 10_000 + rep)
            np.save(order_path, order)
        if len(order) != len(train) or not np.array_equal(np.sort(order), train):
            raise AssertionError(f"Training repetition {rep} is not a permutation of frozen BTS train")
        print(f"Training permutation {rep} verified ({len(order):,} rows)", flush=True)
    records = {"source_archive_sha256": archive_hash,
               "frozen_pilot_minute_manifest_sha256": sha256(original_split),
               "frozen_copy_minute_manifest_sha256": sha256(frozen_copy),
               "split_seed": SEED, "subset_seed_formula": "20261114 + 10000 + repetition",
               "train_sizes": list(SIZES) + [len(train)],
               "three_repetitions_at": [100_000, 300_000, 1_000_000],
               "single_repetition_at": [3_000_000, len(train)],
               "subset_definition": "Exact prefix of independently seeded randomized calendar-minute-block orders; within-minute row shuffle; at most one partial terminal minute per size; nested within repetition; no outcome stratification",
               "complete_train_pool": int(len(train)), "membership_counts": membership,
               "data_ingestion": ingest,
               "bts_storage_level_counts": {x: len(dictionaries[x]) for x in USER_COLUMNS},
               "seconds": round(time.monotonic() - started, 2),
               "random_opened": False}
    (output / "prepare_manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps({"split": membership, "storage_levels": records["bts_storage_level_counts"],
                      "elapsed_seconds": records["seconds"]}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--pilot-dir", type=Path, default=Path("outputs/pilot"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/full"))
    args = parser.parse_args()
    prepare(args.archive, args.pilot_dir, args.output_dir)


if __name__ == "__main__":
    main()
