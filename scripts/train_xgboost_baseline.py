"""Train and evaluate an order-agnostic XGBoost cost-model baseline.

Features are handcrafted aggregates of program context and transformation types,
loop levels, numeric parameters, and co-occurrence. Transformation order is
intentionally discarded so this is a conventional feature-based baseline.
"""
import argparse
import gzip
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score

TRANSFORM_TYPES = (
    "interchange", "skewing", "tiling", "reversal",
    "parallelization", "unrolling", "fusion", "shifting",
)
MAX_SLOTS = 3


def _signed_log(value):
    value = float(value)
    return math.copysign(math.log1p(abs(value)), value)


def feature_names():
    names = [
        "context_iterators", "context_computations",
        "context_buffers", "context_memory_size",
        "transformation_count", "unique_transformation_types",
    ]
    for kind in TRANSFORM_TYPES:
        names.append(f"type_{kind}_count")
        for slot in range(MAX_SLOTS):
            for stat in ("mean", "max", "min", "sum"):
                names.append(f"{kind}_parameter_{slot}_{stat}")
            for stat in ("mean", "max", "min"):
                names.append(f"{kind}_loop_{slot}_{stat}")
        for stat in ("mean", "max", "sum"):
            names.append(f"{kind}_computation_count_{stat}")
    for i, left in enumerate(TRANSFORM_TYPES):
        for right in TRANSFORM_TYPES[i:]:
            names.append(f"cooccur_{left}_{right}")
    names.extend([
        "all_parameter_count", "all_parameter_mean", "all_parameter_std",
        "all_parameter_min", "all_parameter_max",
        "all_loop_count", "all_loop_mean", "all_loop_std",
        "all_loop_min", "all_loop_max",
    ])
    return names


FEATURE_NAMES = feature_names()


def extract_features(program, schedule):
    annotation = program["program_annotation"]
    context = [
        len(annotation["iterators"]),
        len(annotation["computations"]),
        len(annotation["buffers"]["buffers_info"]),
        float(annotation["memory_size"]),
    ]
    values = [math.log1p(max(float(v), 0.0)) for v in context]
    transformations = schedule.get("transformations_list", [])
    counts = Counter()
    by_type = {kind: [] for kind in TRANSFORM_TYPES}
    all_parameters, all_loops = [], []

    for transformation in transformations:
        kind = transformation["type"]
        if kind not in by_type:
            raise ValueError(f"Unsupported transformation type: {kind}")
        counts[kind] += 1
        params = [_signed_log(v) for v in transformation.get("parameters", [])]
        loops = [float(v) for v in transformation.get("loop_levels", [])]
        computations = math.log1p(len(transformation.get("computations", [])))
        by_type[kind].append((params, loops, computations))
        all_parameters.extend(params)
        all_loops.extend(loops)

    values.extend([len(transformations), len(counts)])
    for kind in TRANSFORM_TYPES:
        items = by_type[kind]
        values.append(counts[kind])
        for slot in range(MAX_SLOTS):
            params = [item[0][slot] for item in items if len(item[0]) > slot]
            values.extend([
                float(np.mean(params)) if params else 0.0,
                float(np.max(params)) if params else 0.0,
                float(np.min(params)) if params else 0.0,
                float(np.sum(params)) if params else 0.0,
            ])
            loops = [item[1][slot] for item in items if len(item[1]) > slot]
            values.extend([
                float(np.mean(loops)) if loops else 0.0,
                float(np.max(loops)) if loops else 0.0,
                float(np.min(loops)) if loops else 0.0,
            ])
        comp = [item[2] for item in items]
        values.extend([
            float(np.mean(comp)) if comp else 0.0,
            float(np.max(comp)) if comp else 0.0,
            float(np.sum(comp)) if comp else 0.0,
        ])

    for i, left in enumerate(TRANSFORM_TYPES):
        for right in TRANSFORM_TYPES[i:]:
            if left == right:
                pair_count = counts[left] * (counts[left] - 1) / 2
            else:
                pair_count = counts[left] * counts[right]
            values.append(float(pair_count))

    values.extend([
        len(all_parameters),
        float(np.mean(all_parameters)) if all_parameters else 0.0,
        float(np.std(all_parameters)) if all_parameters else 0.0,
        float(np.min(all_parameters)) if all_parameters else 0.0,
        float(np.max(all_parameters)) if all_parameters else 0.0,
        len(all_loops),
        float(np.mean(all_loops)) if all_loops else 0.0,
        float(np.std(all_loops)) if all_loops else 0.0,
        float(np.min(all_loops)) if all_loops else 0.0,
        float(np.max(all_loops)) if all_loops else 0.0,
    ])
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (len(FEATURE_NAMES),) or not np.all(np.isfinite(result)):
        raise ValueError("Invalid feature vector")
    return result


def load_split(path):
    X, targets, execution_times, program_names = [], [], [], []
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            program = json.loads(line)
            initial_time = float(program["initial_execution_time"])
            if not math.isfinite(initial_time) or initial_time <= 0:
                raise ValueError(f"Invalid initial time: {program.get('program_name')}")
            for schedule in program["schedules_list"]:
                times = schedule["execution_times"]
                if len(times) != 1:
                    raise ValueError("Expected one pooled execution time per schedule")
                execution_time = float(times[0])
                if not math.isfinite(execution_time) or execution_time <= 0:
                    raise ValueError("Invalid execution time")
                speedup = max(initial_time / execution_time, 0.01)
                X.append(extract_features(program, schedule))
                targets.append(speedup)
                execution_times.append(execution_time)
                program_names.append(program["program_name"])
    if not X:
        raise ValueError(f"No samples found in {path}")
    frame = pd.DataFrame({
        "program_name": program_names,
        "speedup": np.asarray(targets, dtype=np.float32),
        "execution_time": np.asarray(execution_times, dtype=np.float32),
    })
    return np.stack(X), frame


def program_weights(frame):
    counts = frame.groupby("program_name")["program_name"].transform("count").to_numpy()
    return (1.0 / counts).astype(np.float32)


def program_weighted_mape(predicted_log, matrix):
    truth = np.exp(np.clip(matrix.get_label(), -20, 20))
    prediction = np.exp(np.clip(predicted_log, -20, 20))
    weights = matrix.get_weight()
    ape = np.abs(truth - prediction) / (truth + 1e-5) * 100.0
    return "program_mape", float(np.average(ape, weights=weights))


def calculate_metrics(frame):
    records = []
    for program_name, group in frame.groupby("program_name", sort=True):
        truth = group["speedup"].to_numpy(dtype=float)
        prediction = group["prediction"].to_numpy(dtype=float)
        times = group["execution_time"].to_numpy(dtype=float)
        if len(group) < 2:
            continue
        correlation = spearmanr(truth, prediction).statistic
        if not np.isfinite(correlation):
            correlation = np.nan
        relevance = np.maximum(truth, 0).reshape(1, -1)
        scores = prediction.reshape(1, -1)
        selected = int(np.argmax(prediction))
        optimal = int(np.argmin(times))
        records.append({
            "program_name": program_name,
            "mape": float(group["ape"].mean()),
            "spearman": correlation,
            "ndcg": float(ndcg_score(relevance, scores)),
            "ndcg_at_5": float(ndcg_score(relevance, scores, k=min(5, len(group)))),
            "ndcg_at_1": float(ndcg_score(relevance, scores, k=1)),
            "top1_optimal": float(np.isclose(times[selected], times[optimal], rtol=1e-5, atol=1e-8)),
            "regret_percent": float((times[selected] / times[optimal] - 1.0) * 100.0),
        })
    metrics = pd.DataFrame(records)
    if metrics.empty:
        raise ValueError("No programs with at least two schedules")
    summary = {
        "programs": int(len(metrics)),
        "MAPE": float(frame.groupby("program_name")["ape"].mean().mean()),
        "Spearman": float(metrics["spearman"].mean()),
        "Spearman_valid_programs": int(metrics["spearman"].count()),
        "nDCG": float(metrics["ndcg"].mean()),
        "nDCG@5": float(metrics["ndcg_at_5"].mean()),
        "nDCG@1": float(metrics["ndcg_at_1"].mean()),
        "Top-1 optimal (%)": float(metrics["top1_optimal"].mean() * 100),
        "Mean regret (%)": float(metrics["regret_percent"].mean()),
        "Median regret (%)": float(metrics["regret_percent"].median()),
        "P90 regret (%)": float(metrics["regret_percent"].quantile(0.90)),
    }
    return summary, metrics


def train_one_seed(X_train, train_frame, X_tune, tune_frame, X_test, test_frame, seed, output_dir, rounds, patience):
    output_dir = Path(output_dir) / f"seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    dtrain = xgb.DMatrix(
        X_train, label=np.log(train_frame["speedup"].to_numpy()),
        weight=program_weights(train_frame), feature_names=FEATURE_NAMES,
    )
    dtune = xgb.DMatrix(
        X_tune, label=np.log(tune_frame["speedup"].to_numpy()),
        weight=program_weights(tune_frame), feature_names=FEATURE_NAMES,
    )
    dtest = xgb.DMatrix(X_test, feature_names=FEATURE_NAMES)
    params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "max_depth": 8,
        "eta": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "lambda": 1.0,
        "alpha": 0.0,
        "seed": int(seed),
        "nthread": 4,
        "disable_default_eval_metric": 1,
    }
    booster = xgb.train(
        params, dtrain, num_boost_round=rounds,
        evals=[(dtune, "tuning")],
        custom_metric=program_weighted_mape,
        maximize=False, early_stopping_rounds=patience,
        verbose_eval=25,
    )
    best_iteration = int(booster.best_iteration)
    prediction_log = booster.predict(dtest, iteration_range=(0, best_iteration + 1))
    predictions = np.exp(np.clip(prediction_log, -20, 20))
    result = test_frame.copy()
    result["prediction"] = predictions
    result["ape"] = (
        np.abs(result["speedup"].to_numpy() - predictions)
        / (result["speedup"].to_numpy() + 1e-5) * 100.0
    )
    summary, metrics = calculate_metrics(result)
    result.to_csv(output_dir / "xgboost_predictions.csv", index=False)
    metrics.to_csv(output_dir / "xgboost_program_metrics.csv", index=False)
    booster.save_model(output_dir / "xgboost_model.json")
    report = {
        "seed": int(seed),
        "best_iteration": best_iteration,
        "best_tuning_program_weighted_mape": float(booster.best_score),
        "feature_count": len(FEATURE_NAMES),
        "training_programs": int(train_frame["program_name"].nunique()),
        "tuning_programs": int(tune_frame["program_name"].nunique()),
        "heldout_test_programs": int(test_frame["program_name"].nunique()),
        "training_schedules": int(len(train_frame)),
        "tuning_schedules": int(len(tune_frame)),
        "heldout_test_schedules": int(len(test_frame)),
        "metrics": summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--tune", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 15, 2026])
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=50)
    args = parser.parse_args()
    if args.rounds < 1 or args.patience < 1 or not args.seeds:
        parser.error("rounds, patience, and seeds must be positive/nonempty")

    print("Loading matched training data...")
    X_train, train_frame = load_split(args.train)
    print("Loading tuning data...")
    X_tune, tune_frame = load_split(args.tune)
    print("Loading held-out test data...")
    X_test, test_frame = load_split(args.test)

    train_names = set(train_frame["program_name"])
    tune_names = set(tune_frame["program_name"])
    test_names = set(test_frame["program_name"])
    if train_names & tune_names or train_names & test_names or tune_names & test_names:
        raise ValueError("Program overlap detected across train/tune/test")
    if X_train.shape[1] != X_tune.shape[1] or X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Feature dimension mismatch across splits")

    reports = []
    for seed in args.seeds:
        reports.append(train_one_seed(
            X_train, train_frame, X_tune, tune_frame, X_test, test_frame,
            seed, args.output, args.rounds, args.patience,
        ))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "all_seed_summary.json").write_text(json.dumps({
        "dataset_protocol": "Pinned LOOPerSet v2.1; common LSTM/Transformer-compatible schedules; program-disjoint tune/test",
        "feature_design": "Order-agnostic handcrafted aggregates; no execution-time or target-derived input features",
        "seeds": [int(seed) for seed in args.seeds],
        "results": reports,
    }, indent=2))


if __name__ == "__main__":
    main()
