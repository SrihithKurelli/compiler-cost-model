
import argparse
import os

import numpy as np
import pandas as pd
import torch

from looper_cost_model.config import load_config
from looper_cost_model.data.dataset import load_batches
from looper_cost_model.evaluate import predictions_dataframe, score
from looper_cost_model.model import LOOPerCostModel


def evaluate_checkpoint(cfg, checkpoint, split):
    base = cfg.experiment.base_path
    batched_dir = os.path.join(base, "batched")
    meta_path = os.path.join(batched_dir, f"{split}_meta.json")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"{split} batches not found: {meta_path}. "
            "Run the corresponding training experiment first."
        )

    model = LOOPerCostModel(
        input_size=cfg.model.input_size,
        comp_embed_layer_sizes=cfg.model.comp_embed_layer_sizes,
        drops=cfg.model.drops,
    )

    model.load_state_dict(
        torch.load(checkpoint, map_location=cfg.training.device)
    )
    model = model.to(cfg.training.device)

    prepared = load_batches(
        batched_dir,
        split,
        device=cfg.training.device,
    )

    predictions = predictions_dataframe(
        model,
        prepared,
        cfg.training.device,
    )

    return predictions, score(predictions)


def selection_metrics(df):
    rows = []

    for program_name, group in df.groupby("name"):
        selected = group.loc[group["prediction"].idxmax()]
        optimal = group.loc[group["target"].idxmax()]

        selected_time = float(selected["exec_time"])
        optimal_time = float(optimal["exec_time"])

        regret = (
            selected_time / optimal_time - 1.0
        ) * 100.0

        rows.append({
            "name": program_name,
            "schedules": len(group),
            "selected_exec_time": selected_time,
            "optimal_exec_time": optimal_time,
            "selection_regret_pct": regret,
            "top1_optimal": int(
                np.isclose(
                    selected_time,
                    optimal_time,
                    rtol=1e-9,
                    atol=1e-12,
                )
            ),
        })

    return pd.DataFrame(rows)


def summarize_checkpoint(name, predictions, scores):
    selections = selection_metrics(predictions)

    merged = scores.merge(
        selections,
        on="name",
        how="inner",
    )

    print(f"\n{'=' * 65}")
    print(name)
    print(f"{'=' * 65}")

    print("\nPrediction and ranking metrics:")

    for metric in ("MAPE", "Spearman", "nDCG", "nDCG@5", "nDCG@1"):
        print(f"{metric:15s}: {scores[metric].mean():.4f}")

    print("\nPer-program MAPE distribution:")
    print(
        scores["MAPE"].describe(
            percentiles=[0.25, 0.50, 0.75, 0.90]
        ).to_string()
    )

    print("\nSelection metrics:")

    print(
        f"Top-1 optimal selection: "
        f"{merged['top1_optimal'].mean() * 100:.2f}%"
    )

    print(
        f"Mean selection regret: "
        f"{merged['selection_regret_pct'].mean():.4f}%"
    )

    print(
        f"Median selection regret: "
        f"{merged['selection_regret_pct'].median():.4f}%"
    )

    print(
        f"90th percentile regret: "
        f"{merged['selection_regret_pct'].quantile(0.90):.4f}%"
    )

    valid_spearman = scores["Spearman"].notna().sum()

    print(
        f"\nValid Spearman programs: "
        f"{valid_spearman}/{len(scores)}"
    )

    return merged


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/pilot_1000.yaml",
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--label",
        required=True,
    )

    parser.add_argument(
        "--split",
        choices=["train", "valid"],
        default="valid",
    )

    args = parser.parse_args()

    cfg = load_config(["--config", args.config])

    predictions, scores = evaluate_checkpoint(
        cfg,
        args.checkpoint,
        args.split,
    )

    results = summarize_checkpoint(
        args.label,
        predictions,
        scores,
    )

    output_dir = os.path.join(
        cfg.experiment.base_path,
        "analysis",
    )

    os.makedirs(output_dir, exist_ok=True)

    predictions.to_csv(
        os.path.join(
            output_dir,
            f"{args.label}_predictions.csv",
        ),
        index=False,
    )

    results.to_csv(
        os.path.join(
            output_dir,
            f"{args.label}_program_metrics.csv",
        ),
        index=False,
    )

    print(f"\nDetailed results saved to: {output_dir}")


if __name__ == "__main__":
    main()
