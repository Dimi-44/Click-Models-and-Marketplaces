"""Tune approved models exclusively on frozen BTS validation log loss."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.linear_model import LogisticRegression

from .full_encoding import CONTEXTUAL, ITEM_POSITION, CompactSparseEncoder


SEED = 20261114
PRIOR_GRID = (50.0, 200.0, 1000.0)
RIDGE_C_GRID = (0.1, 1.0, 10.0)
MODELS = ("B0", "B1", "M1", "M2")


def log_loss_binary(y: np.ndarray, probability: np.ndarray) -> float:
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-12, 1 - 1e-12)
    outcome = np.asarray(y, dtype=np.float64)
    return float(np.mean(-outcome * np.log(p) - (1 - outcome) * np.log1p(-p)))


def rates_item_position(rows: np.ndarray, prior: float) -> tuple[np.ndarray, float]:
    keys = rows["item_id"].astype(np.int32) * 3 + rows["position"].astype(np.int32) - 1
    prevalence = float(rows["click"].mean())
    counts = np.bincount(keys, minlength=240)
    clicks = np.bincount(keys, weights=rows["click"], minlength=240)
    return (clicks + prior * prevalence) / (counts + prior), prevalence


def score_item_position(rows: np.ndarray, rates: np.ndarray) -> np.ndarray:
    return rates[rows["item_id"].astype(np.int32) * 3 + rows["position"].astype(np.int32) - 1]


def plan_file(output: Path, n_full: int) -> dict:
    plan = {
        "plan_version": "approved_phase3_v1", "seed": SEED,
        "sizes": [100_000, 300_000, 1_000_000, 3_000_000, n_full],
        "repetitions": {"100000": 3, "300000": 3, "1000000": 3,
                        "3000000": 1, str(n_full): 1},
        "model_grid": {"B0": [None], "B1_prior": list(PRIOR_GRID),
                       "M1_C": list(RIDGE_C_GRID), "M2_C": list(RIDGE_C_GRID)},
        "validation_selection": "Minimum BTS validation log loss; stable ascending grid tie-break",
        "unknown_categories": "One indicator for every feature, fitted on current BTS training subset only",
        "logistic": {"solver": "lbfgs", "tol": 1e-6, "max_iter": 300,
                     "penalty": "L2 ridge", "fit_intercept": True},
        "b1_definition": "(observed clicks + prior*train_click_rate)/(exposures + prior)",
        "random_may_be_read": False,
    }
    path = output / "frozen_tuning_plan.json"
    serialized = json.dumps(plan, sort_keys=True, indent=2)
    if path.exists() and path.read_text(encoding="utf-8") != serialized:
        raise ValueError("Refusing to change existing frozen hyperparameter grid")
    path.write_text(serialized, encoding="utf-8")
    return plan


def selected_config(output: Path, key: str, model: str) -> Path:
    return output / "selection" / f"{key}_{model}.json"


def _fit_run_model(model: str, train_rows: np.ndarray, validation: np.ndarray,
                   y_val: np.ndarray) -> tuple[dict, list[dict]]:
    start = time.monotonic()
    y_train = train_rows["click"].astype(np.int8)
    if model == "B0":
        prevalence = float(y_train.mean())
        metric = log_loss_binary(y_val, np.full(len(y_val), prevalence))
        payload = {"kind": "B0", "ctr": prevalence}
        grid = [{"parameter": "none", "value": None, "validation_log_loss": metric}]
    elif model == "B1":
        all_trials = []
        for prior in PRIOR_GRID:
            rates, prevalence = rates_item_position(train_rows, prior)
            score = log_loss_binary(y_val, score_item_position(validation, rates))
            all_trials.append((score, prior, rates, prevalence))
        all_trials.sort(key=lambda row: (row[0], row[1]))
        score, best_prior, best_rates, prevalence = all_trials[0]
        payload = {"kind": "B1", "ctr": prevalence,
                   "prior": best_prior, "rates": best_rates}
        grid = [{"parameter": "prior", "value": p, "validation_log_loss": s}
                for s, p, _rates, _ctr in sorted(all_trials, key=lambda x: x[1])]
    else:
        fields = ITEM_POSITION if model == "M1" else CONTEXTUAL
        encoder = CompactSparseEncoder(fields).fit(train_rows)
        x_train, unknown_train = encoder.transform(train_rows)
        x_val, unknown_val = encoder.transform(validation)
        if any(unknown_train.values()):
            raise AssertionError("Training encoder reported an unseen training code")
        all_trials = []
        for c in RIDGE_C_GRID:
            t = time.monotonic()
            fitted = LogisticRegression(C=c, solver="lbfgs", tol=1e-6, max_iter=300,
                                        random_state=SEED, fit_intercept=True)
            fitted.fit(x_train, y_train)
            score = log_loss_binary(y_val, fitted.predict_proba(x_val)[:, 1])
            all_trials.append((score, c, fitted, int(fitted.n_iter_[0]), time.monotonic() - t))
            print(f"    {model} C={c:g}, BTS val logloss={score:.8f}, "
                  f"iterations={fitted.n_iter_[0]}, seconds={all_trials[-1][4]:.1f}", flush=True)
        all_trials.sort(key=lambda row: (row[0], row[1]))
        score, best_c, best_model, iteration, _duration = all_trials[0]
        payload = {"kind": model, "C": best_c, "model": best_model,
                   "encoder": encoder, "train_unknown_counts": unknown_train,
                   "validation_unknown_counts": unknown_val}
        grid = [{"parameter": "C", "value": c, "validation_log_loss": s,
                 "iterations": it, "training_seconds": duration}
                for s, c, _m, it, duration in sorted(all_trials, key=lambda x: x[1])]
        del x_train, x_val
    return payload, grid


def tune(output: Path, *, max_size: int | None) -> None:
    output = output.resolve(strict=True)
    packed = np.load(output / "bts_compact.npy", mmap_mode="r")
    validation_idx = np.load(output / "indices_validation.npy", mmap_mode="r")
    train_indices = np.load(output / "indices_train.npy", mmap_mode="r")
    sizes = [100_000, 300_000, 1_000_000, 3_000_000, len(train_indices)]
    plan = plan_file(output, len(train_indices))
    plan_digest = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    (output / "frozen_tuning_plan.sha256").write_text(plan_digest + "\n")
    (output / "selection").mkdir(exist_ok=True)
    (output / "models").mkdir(exist_ok=True)
    validation = packed[validation_idx]
    y_val = validation["click"].astype(np.int8)
    print(f"Validation-only tuning: {len(validation):,} BTS observations", flush=True)
    process = psutil.Process()

    for size in sizes:
        if max_size is not None and size > max_size:
            break
        reps = range(3) if size <= 1_000_000 else range(1)
        for rep in reps:
            key = f"n{size}_rep{rep}"
            if all(selected_config(output, key, model).exists() for model in MODELS):
                print(f"Skipping completed run {key}", flush=True)
                continue
            print(f"Training {key} (all hyperparameters selected on BTS validation)", flush=True)
            order = np.load(output / f"train_order_minute_rep{rep}.npy", mmap_mode="r")
            training_idx = order[:size]
            train_rows = packed[training_idx]
            if len(np.unique(training_idx)) != size:
                raise AssertionError("Repeated training indices in one nested prefix")
            for model in MODELS:
                selection_path = selected_config(output, key, model)
                if selection_path.exists():
                    continue
                t = time.monotonic()
                fitted, grid = _fit_run_model(model, train_rows, validation, y_val)
                model_path = output / "models" / f"{key}_{model}.joblib"
                joblib.dump(fitted, model_path, compress=3)
                best = min(grid, key=lambda r: (r["validation_log_loss"],
                                               -1 if r["value"] is None else r["value"]))
                info = {"run": key, "size": size, "repetition": rep, "model": model,
                        "selection": best, "grid": grid,
                        "training_minutes": float((time.monotonic() - t) / 60),
                        "rss_bytes": int(process.memory_info().rss),
                        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
                        "model_file": str(model_path.relative_to(output)),
                        "frozen_plan_sha256": plan_digest}
                selection_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
                print(f"  Selected {model} {best['parameter']}={best['value']} "
                      f"(logloss={best['validation_log_loss']:.8f})", flush=True)
            del train_rows

    records = []
    choices = []
    complete = True
    for size in sizes:
        for rep in (range(3) if size <= 1_000_000 else range(1)):
            key = f"n{size}_rep{rep}"
            for model in MODELS:
                path = selected_config(output, key, model)
                if not path.exists():
                    complete = False
                    continue
                s = json.loads(path.read_text())
                if s["frozen_plan_sha256"] != plan_digest:
                    raise ValueError("Tuning plan changed during execution")
                choices.append({"size": size, "repetition": rep, "model": model,
                                "selected_parameter": s["selection"]["parameter"],
                                "selected_value": s["selection"]["value"],
                                "validation_log_loss": s["selection"]["validation_log_loss"]})
                for row in s["grid"]:
                    records.append({"size": size, "repetition": rep, "model": model, **row})
    pd.DataFrame(records).to_csv(output / "validation_grid.csv", index=False)
    pd.DataFrame(choices).to_csv(output / "chosen_hyperparameters.csv", index=False)
    if complete:
        lock = {"frozen_plan_sha256": plan_digest,
                "choices_sha256": hashlib.sha256(
                    (output / "chosen_hyperparameters.csv").read_bytes()).hexdigest(),
                "validation_grid_sha256": hashlib.sha256(
                    (output / "validation_grid.csv").read_bytes()).hexdigest(),
                "all_44_model_size_repetition_selections_complete": len(choices) == 44,
                "random_opened_during_training": False}
        (output / "TUNING_LOCK.json").write_text(json.dumps(lock, indent=2))
        print("All choices frozen. Random may now be opened for external scoring.", flush=True)
    else:
        print(f"Tuning incomplete ({len(choices)} / 44 models); Random remains sealed.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/full"))
    parser.add_argument("--max-size", type=int, default=None,
                        help="Stop after an approved size for controlled scaling")
    args = parser.parse_args()
    tune(args.output_dir, max_size=args.max_size)


if __name__ == "__main__":
    main()
