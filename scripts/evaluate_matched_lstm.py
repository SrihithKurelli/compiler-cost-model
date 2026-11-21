"""Export per-program recursive-LSTM predictions and ranking metrics for matched runs."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from looper_cost_model.config import load_config
from looper_cost_model.data.dataset import load_batches, prepare_dataset
from looper_cost_model.evaluate import predictions_dataframe, score
from looper_cost_model.model import LOOPerCostModel


def main():
    cfg = load_config()
    base = cfg.experiment.base_path
    batched_dir = os.path.join(base, "batched")
    split = "valid"
    meta = os.path.join(batched_dir, f"{split}_meta.json")
    if not os.path.exists(meta):
        prepare_dataset(
            cfg.data.valid_file,
            cache_dir=os.path.join(base, "cache", split),
            batched_dir=batched_dir,
            split_name=split,
            batch_size=cfg.data.batch_size,
            nb_workers=cfg.data.nb_workers,
            min_functions_per_footprint=cfg.data.min_functions_per_footprint,
            seed=cfg.training.seed,
        )

    model = LOOPerCostModel(
        input_size=cfg.model.input_size,
        comp_embed_layer_sizes=cfg.model.comp_embed_layer_sizes,
        drops=cfg.model.drops,
    )
    weights_path = cfg.eval.weights_path
    if not weights_path:
        raise ValueError("eval.weights_path must point at the matched LSTM checkpoint.")
    model.load_state_dict(torch.load(weights_path, map_location=cfg.training.device))
    model = model.to(cfg.training.device)

    prepared = load_batches(batched_dir, split, device=cfg.training.device)
    predictions = predictions_dataframe(model, prepared, cfg.training.device)
    program_metrics = score(predictions).rename(columns={
        "name": "program_name",
        "MAPE": "mape",
        "Spearman": "spearman",
        "nDCG": "ndcg",
        "nDCG@5": "ndcg_at_5",
        "nDCG@1": "ndcg_at_1",
    })

    # Match the Transformer evaluator's epsilon-stabilized percentage-error denominator.
    epsilon_mape = predictions.assign(
        _ape=np.abs(predictions["target"] - predictions["prediction"])
        / (predictions["target"] + 1e-5) * 100.0
    ).groupby("name")["_ape"].mean()
    program_metrics["mape"] = program_metrics["program_name"].map(epsilon_mape)

    regret_rows = []
    for program_name, group in predictions.groupby("name", sort=False):
        times = group["exec_time"].to_numpy(dtype=float)
        estimates = group["prediction"].to_numpy(dtype=float)
        selected_time = times[int(np.argmax(estimates))]
        optimal_time = float(np.min(times))
        regret_rows.append({
            "program_name": program_name,
            "top1_optimal": float(np.isclose(selected_time, optimal_time, rtol=1e-5, atol=1e-8)),
            "regret_percent": float((selected_time / optimal_time - 1.0) * 100.0),
        })
    program_metrics = program_metrics.merge(
        pd.DataFrame(regret_rows), on="program_name", validate="one_to_one"
    )

    output_dir = Path(os.environ.get("MATCHED_LSTM_ANALYSIS_DIR", Path(cfg.experiment.base_path).parent / "matched_lstm_analysis"))
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.rename(columns={"name": "program_name"}).to_csv(
        output_dir / "lstm_predictions.csv", index=False
    )
    program_metrics.to_csv(output_dir / "lstm_program_metrics.csv", index=False)
    summary = {
        "programs": int(len(program_metrics)),
        "MAPE": float(program_metrics["mape"].mean()),
        "Spearman": float(program_metrics["spearman"].mean()),
        "nDCG": float(program_metrics["ndcg"].mean()),
        "nDCG@5": float(program_metrics["ndcg_at_5"].mean()),
        "nDCG@1": float(program_metrics["ndcg_at_1"].mean()),
        "Top-1 optimal (%)": float(program_metrics["top1_optimal"].mean() * 100),
        "Mean regret (%)": float(program_metrics["regret_percent"].mean()),
        "Median regret (%)": float(program_metrics["regret_percent"].median()),
        "P90 regret (%)": float(program_metrics["regret_percent"].quantile(0.90)),
    }
    (output_dir / "lstm_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Saved LSTM schedule predictions and program metrics to {output_dir}")


if __name__ == "__main__":
    main()
