
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score
from torch.utils.data import DataLoader

from looper_cost_model.transformer.dataset import (
    TransformerScheduleDataset,
)
from looper_cost_model.transformer.model import (
    TransformationTransformer,
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

        # Relevance scores must be nonnegative.
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

        regret = (
            selected_time / optimal_time - 1.0
        ) * 100.0

        top1_optimal = np.isclose(
            selected_time,
            optimal_time,
            rtol=1e-5,
            atol=1e-8,
        )

        results.append({
            "program_name": program_name,
            "spearman": correlation,
            "ndcg": ndcg,
            "ndcg_at_5": ndcg5,
            "ndcg_at_1": ndcg1,
            "top1_optimal": float(top1_optimal),
            "regret_percent": regret,
        })

    metrics = pd.DataFrame(results)

    summary = {
        "programs": len(metrics),
        "MAPE": float(
            df.groupby("program_name")["ape"].mean().mean()
        ),
        "Spearman": float(
            metrics["spearman"].mean()
        ),
        "nDCG": float(
            metrics["ndcg"].mean()
        ),
        "nDCG@5": float(
            metrics["ndcg_at_5"].mean()
        ),
        "nDCG@1": float(
            metrics["ndcg_at_1"].mean()
        ),
        "Top-1 optimal (%)": float(
            metrics["top1_optimal"].mean() * 100
        ),
        "Mean regret (%)": float(
            metrics["regret_percent"].mean()
        ),
        "Median regret (%)": float(
            metrics["regret_percent"].median()
        ),
        "P90 regret (%)": float(
            metrics["regret_percent"].quantile(0.90)
        ),
    }

    return summary, metrics


def main():
    parser = argparse.ArgumentParser()

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

    args = parser.parse_args()

    device = torch.device("cpu")

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
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
        num_workers=0,
    )

    model = TransformationTransformer().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
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

    df = pd.DataFrame(predictions)

    summary, program_metrics = calculate_metrics(df)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        output_dir / "transformer_predictions.csv",
        index=False,
    )

    program_metrics.to_csv(
        output_dir / "transformer_program_metrics.csv",
        index=False,
    )

    print("\nTransformer evaluation")
    print("Checkpoint epoch:", checkpoint["epoch"])

    for key, value in summary.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

    print("\nSaved predictions and program metrics to:", output_dir)


if __name__ == "__main__":
    main()
