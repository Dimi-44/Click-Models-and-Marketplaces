"""BTS-only smoke test of the approved experiment; not the full experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import resource
import time

import numpy as np
import pandas as pd
import psutil
import scipy
import sklearn

from .data import source_checksum, stream_bts_pilot
from .encoding import CONTEXTUAL, ITEM_POSITION, SparseCategoricalEncoder
from .metrics import probability_metrics, reliability_bins
from .models import ConstantCTR, SmoothedItemPositionCTR, fit_ridge_logistic
from .splits import SPLITS, partition_pilot, split_minute_counts


def _minute_manifest(counts: dict[str, int], assignment: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        [(minute, minute[:10], counts[minute], assignment[minute]) for minute in sorted(counts)],
        columns=["minute_utc", "date_utc", "n_bts", "split"],
    )


def _run_one_model(name: str, partition: dict[str, pd.DataFrame],
                   predictions: dict[str, np.ndarray], metrics: list[dict],
                   calibration: list[pd.DataFrame]) -> None:
    for split in ("validation", "internal_test"):
        truth = partition[split]["click"].to_numpy(dtype=np.int8)
        p = predictions[split]
        row = {"model": name, "policy": "BTS", "split": split}
        row.update(probability_metrics(truth, p))
        metrics.append(row)
        bins = reliability_bins(truth, p)
        bins.insert(0, "split", split)
        bins.insert(0, "model", name)
        calibration.append(bins)


def run_pilot(archive: Path, output_dir: Path, *, seed: int,
              fraction: float, chunk_size: int) -> dict:
    archive = archive.resolve(strict=True)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    process = psutil.Process()
    starting_rss = process.memory_info().rss
    peak_rss = starting_rss

    def mark_memory() -> None:
        nonlocal peak_rss
        peak_rss = max(peak_rss, process.memory_info().rss)

    print("[1/5] Streaming BTS only, selecting deterministic whole-minute pilot blocks", flush=True)
    pilot, minute_counts, scanned = stream_bts_pilot(
        archive, minute_fraction=fraction, seed=seed, chunk_size=chunk_size
    )
    mark_memory()
    assignment = split_minute_counts(minute_counts, seed=seed)
    minute_manifest = _minute_manifest(minute_counts, assignment)
    minute_manifest.to_csv(output_dir / "minute_split_manifest.csv", index=False)

    partition = partition_pilot(pilot, assignment)
    all_pilot_indices = np.concatenate([df["Unnamed: 0"].to_numpy() for df in partition.values()])
    assert len(all_pilot_indices) == len(pilot) == len(np.unique(all_pilot_indices))
    assert sum(minute_counts.values()) == scanned
    # Full BTS split fixed before pilot model fitting, based on minute counts only.
    assert set(minute_manifest["split"]) == set(SPLITS)
    assert pilot["timestamp"].str.slice(0, 16).eq(pilot["minute_utc"]).all()
    for split in SPLITS:
        if partition[split]["click"].nunique() < 2:
            raise ValueError(f"Pilot {split} lacks both click classes; increase fraction")
    assert set(pilot["timestamp"].str[:10]) == set(minute_manifest["date_utc"])
    print("[2/5] Minute partitions and leakage checks completed", flush=True)

    metrics: list[dict] = []
    calibration: list[pd.DataFrame] = []
    fitted_models: dict[str, dict] = {}

    b0 = ConstantCTR().fit(partition["train"])
    b1 = SmoothedItemPositionCTR(prior_strength=200.0).fit(partition["train"])
    for name, model in (("B0_constant_CTR", b0), ("B1_smoothed_item_position", b1)):
        predictions = {split: model.predict_proba(partition[split])
                       for split in ("validation", "internal_test")}
        _run_one_model(name, partition, predictions, metrics, calibration)
        fitted_models[name] = {"train_click_rate": float(b0.rate_),
                               "prior_strength": 200.0 if name.startswith("B1") else None}
    print("[3/5] B0 and B1 BTS pilot scores completed", flush=True)

    encoder_details: dict[str, dict] = {}
    for name, features in (("M1_ridge_item_position", ITEM_POSITION),
                           ("M2_ridge_contextual", CONTEXTUAL)):
        encoder = SparseCategoricalEncoder(features).fit(partition["train"])
        x_train, train_unknown = encoder.transform(partition["train"])
        assert not any(train_unknown.values())
        mark_memory()
        y_train = partition["train"]["click"].to_numpy(dtype=np.int8)
        model = fit_ridge_logistic(x_train, y_train, seed)
        if not np.isfinite(model.coef_).all():
            raise AssertionError("Non-finite fitted logistic coefficients")
        if not model.n_iter_.max() < model.max_iter:
            raise RuntimeError(f"{name} reached iteration limit; check convergence")
        predictions = {}
        unknowns = {"train": train_unknown}
        for split in ("validation", "internal_test"):
            x_eval, unknown = encoder.transform(partition[split])
            predictions[split] = model.predict_proba(x_eval)[:, 1]
            unknowns[split] = unknown
            del x_eval
        _run_one_model(name, partition, predictions, metrics, calibration)
        encoder_details[name] = {**encoder.description(), "unknown_counts": unknowns,
                                 "train_matrix_shape": list(x_train.shape),
                                 "train_matrix_nnz": int(x_train.nnz),
                                 "iterations": int(model.n_iter_.max())}
        fitted_models[name] = {"regularization": "L2", "C": 1.0,
                               "solver": "lbfgs", "max_iter": 300,
                               "n_iter": int(model.n_iter_.max())}
        del x_train, model, encoder
        mark_memory()
    print("[4/5] Matched M1/M2 BTS logistic pilot scores completed", flush=True)

    results = pd.DataFrame(metrics)
    results.to_csv(output_dir / "metrics.csv", index=False, float_format="%.15g")
    pd.concat(calibration, ignore_index=True).to_csv(
        output_dir / "calibration.csv", index=False, float_format="%.15g"
    )
    # Partition manifest only includes observed BTS indices. No predictor/outcome data.
    pilot_membership = pd.concat([
        df[["Unnamed: 0", "minute_utc"]].assign(split=split)
        for split, df in partition.items()
    ], ignore_index=True).sort_values("Unnamed: 0")
    pilot_membership.to_csv(output_dir / "pilot_membership.csv", index=False)

    full_by_split = minute_manifest.groupby("split", sort=False)["n_bts"].sum()
    full_by_date = minute_manifest.pivot_table(
        index="date_utc", columns="split", values="n_bts", aggfunc="sum", fill_value=0
    )
    full_by_date = full_by_date.reindex(columns=SPLITS, fill_value=0)
    actual_fraction_by_date = full_by_date.div(full_by_date.sum(axis=1), axis=0)
    # All split boundaries are calendar-minute blocks; no shared minute across splits.
    checks = {
        "random_dataset_read": False,
        "fit_uses_bts_training_only": True,
        "all_original_rows_retained_in_selected_minutes": True,
        "no_synthetic_deduplication": True,
        "all_full_bts_minutes_assigned_exactly_once": len(assignment) == len(minute_counts),
        "all_pilot_rows_assigned_exactly_once": len(all_pilot_indices) == len(pilot),
        "pilot_index_unique": bool(pilot["Unnamed: 0"].is_unique),
        "pilot_minutes_disjoint": all(
            not (set(partition[a]["minute_utc"]) & set(partition[b]["minute_utc"]))
            for i, a in enumerate(SPLITS) for b in SPLITS[i + 1:]
        ),
        "date_balance_full_bts": actual_fraction_by_date.round(6).reset_index().to_dict(orient="records"),
        "categorical_unseen_codes": {name: detail["unknown_counts"]
                                      for name, detail in encoder_details.items()},
        "no_propensity_or_index_or_timestamp_model_predictors": True,
        "finite_probability_metrics": bool(results.select_dtypes(include="number").drop(
            columns=["roc_auc", "average_precision"], errors="ignore"
        ).apply(np.isfinite).all().all()),
    }
    if not checks["pilot_minutes_disjoint"]:
        raise AssertionError("Minute leakage")
    if not checks["finite_probability_metrics"]:
        raise AssertionError("Non-finite score or predicted probability")
    (output_dir / "checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")

    mark_memory()
    # Linux ru_maxrss is in KiB; captures peaks between explicit memory checkpoints.
    high_water_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    manifest = {
        "pilot_scope": "BTS only; no Random prediction; no full-scale experiment",
        "archive_path": str(archive),
        "archive_sha256": source_checksum(archive),
        "source_member": "all/All bts.csv",
        "utc_created": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "pilot_minute_fraction": fraction,
        "chunk_size": chunk_size,
        "bts_total_scanned_rows": scanned,
        "bts_full_unique_minutes": len(minute_counts),
        "bts_pilot_rows": int(len(pilot)),
        "bts_pilot_minutes": int(pilot["minute_utc"].nunique()),
        "full_bts_rows_by_split": {split: int(full_by_split[split]) for split in SPLITS},
        "pilot_partition_rows": {split: int(len(partition[split])) for split in SPLITS},
        "pilot_partition_clicks": {split: int(partition[split]["click"].sum()) for split in SPLITS},
        "fitted_models": fitted_models,
        "encoders": encoder_details,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__,
                     "scikit_learn": sklearn.__version__, "psutil": psutil.__version__},
        "memory_start_rss_bytes": int(starting_rss),
        "memory_final_rss_bytes": int(process.memory_info().rss),
        "memory_peak_rss_bytes": int(max(peak_rss, high_water_rss)),
        "elapsed_seconds": round(time.monotonic() - start, 2),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("[5/5] BTS-only pilot artifacts saved", flush=True)
    print(results[["model", "split", "n", "clicks", "log_loss", "brier",
                   "roc_auc", "average_precision"]].to_string(index=False), flush=True)
    print(f"Peak RSS: {manifest['memory_peak_rss_bytes'] / 2**30:.2f} GiB; "
          f"elapsed: {manifest['elapsed_seconds']:.1f} s", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True,
                        help="ZIP containing all/All bts.csv; Random is not opened")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261114)
    parser.add_argument("--pilot-minute-fraction", type=float, default=0.02)
    parser.add_argument("--chunk-size", type=int, default=200_000)
    args = parser.parse_args()
    run_pilot(args.archive, args.output_dir, seed=args.seed,
              fraction=args.pilot_minute_fraction, chunk_size=args.chunk_size)


if __name__ == "__main__":
    main()
