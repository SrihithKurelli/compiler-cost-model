import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score
from torch.utils.data import DataLoader

from looper_cost_model.transformer.dataset import TransformerScheduleDataset
from looper_cost_model.transformer.model import TransformationTransformer
from looper_cost_model.transformer.runtime import (
    load_model_state_compatibly,
    resolve_device,
)


def calculate_metrics(df):
    results = []

    for program_name, group in df.groupby("program_name"):
        y_true = group["speedup"].to_numpy()
        y_pred = group["prediction"].to_numpy()
        execution_times = group["execution_time"].to_numpy()

        if len(group) < 2:
            continue

        correlation = spearmanr(y_true, y_pred).statistic
        if not np.isfinite(correlation):
            correlation = np.nan

        true_relevance = np.maximum(y_true, 0)
        ndcg = ndcg_score(
            true_relevance.reshape(1, -1),
            y_pred.reshape(1, -1),
        )
        k = min(5, len(group))
        ndcg5 = ndcg_score(
            true_relevance.reshape(1, -1),
            y_pred.reshape(1, -1),
            k=k,
        )
        ndcg1 = ndcg_score(
            true_relevance.reshape(1, -1),
            y_pred.reshape(1, -1),
            k=1,
        )

        selected_idx = np.argmax(y_pred)
        optimal_idx = np.argmin(execution_times)
        selected_time = execution_times[selected_idx]
        optimal_time = execution_times[optimal_idx]
        regret = (selected_time / optimal_time - 1.0) * 100.0
        top1_optimal = np.isclose(
            selected_time, optimal_time, rtol=1e-5, atol=1e-8
        )

        results.append({
            "program_name": program_name,
            "mape": float(group["ape"].mean()),
            "spearman": correlation,
            "ndcg": ndcg,
            "ndcg_at_5": ndcg5,
            "ndcg_at_1": ndcg1,
            "top1_optimal": float(top1_optimal),
            "regret_percent": regret,
        })

    metrics = pd.DataFrame(results)
    if metrics.empty:
        raise ValueError(
            "No program has at least two schedules; ranking metrics cannot be computed."
        )

    summary = {
        "programs": len(metrics),
        "MAPE": float(df.groupby("program_name")["ape"].mean().mean()),
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


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained LOOPerSet Transformer checkpoint."
    )
    parser.add_argument(
        "--data",
        default="../data/processed/looperset_validation_300_cleaned.jsonl.gz",
    )
    parser.add_argument(
        "--checkpoint",
        default="./runs/transformer_pilot_1000/best_transformer.pt",
    )
    parser.add_argument(
        "--output",
        default="./runs/transformer_pilot_1000/analysis",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, mps, cuda, or cuda:N (default: auto)",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")

    device = resolve_device(args.device)
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    dataset = TransformerScheduleDataset(
        args.data,
        checkpoint["context_mean"],
        checkpoint["context_std"],
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model_config = checkpoint.get("model_config", {})
    model = TransformationTransformer(**model_config).to(device)
    load_model_state_compatibly(model, checkpoint["model_state_dict"])
    model.eval()

    predictions = []
    with torch.no_grad():
        for batch in loader:
            inputs = {
                key: value.to(device)
                for key, value in batch.items()
                if isinstance(value, torch.Tensor)
            }
            outputs = model(inputs).cpu().numpy()
            targets = batch["speedup"].numpy()
            execution_times = batch["execution_time"].numpy()

            for i, prediction in enumerate(outputs):
                predictions.append({
                    "program_name": batch["program_name"][i],
                    "prediction": float(prediction),
                    "speedup": float(targets[i]),
                    "execution_time": float(execution_times[i]),
                    "ape": float(
                        abs(targets[i] - prediction)
                        / (targets[i] + 1e-5)
                        * 100
                    ),
                })

    if not predictions:
        raise RuntimeError("Evaluation dataset produced no schedules.")

    df = pd.DataFrame(predictions)
    summary, program_metrics = calculate_metrics(df)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "transformer_predictions.csv", index=False)
    program_metrics.to_csv(
        output_dir / "transformer_program_metrics.csv",
        index=False,
    )

    print("\nTransformer evaluation")
    print("Checkpoint epoch:", checkpoint.get("epoch", "unknown"))
    print("Device:", device)
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print("\nSaved predictions and program metrics to:", output_dir)


if __name__ == "__main__":
    main()
