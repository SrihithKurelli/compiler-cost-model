"""Paired program-level comparison with bootstrap confidence intervals."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


METRICS = {
    "mape": "lower",
    "spearman": "higher",
    "ndcg": "higher",
    "ndcg_at_5": "higher",
    "ndcg_at_1": "higher",
    "top1_optimal": "higher",
    "regret_percent": "lower",
}


def paired_comparison(transformer_path, lstm_path, output_dir, seed=42, n_bootstrap=10000):
    transformer = pd.read_csv(transformer_path)
    lstm = pd.read_csv(lstm_path)
    required = {"program_name", *METRICS.keys()}
    for label, frame in (("Transformer", transformer), ("LSTM", lstm)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} metrics missing columns: {sorted(missing)}")
        if frame["program_name"].duplicated().any():
            raise ValueError(f"{label} metrics contain duplicate program names")

    merged = transformer[list(required)].merge(
        lstm[list(required)], on="program_name", how="outer",
        suffixes=("_transformer", "_lstm"), indicator=True, validate="one_to_one"
    )
    unmatched = merged.loc[merged["_merge"] != "both", ["program_name", "_merge"]]
    if not unmatched.empty:
        raise ValueError(
            f"Program sets differ ({len(unmatched)} unmatched); refusing unpaired comparison."
        )
    merged = merged.drop(columns="_merge").sort_values("program_name").reset_index(drop=True)

    rng = np.random.default_rng(seed)
    report = {
        "programs": int(len(merged)),
        "bootstrap_resamples": int(n_bootstrap),
        "bootstrap_seed": int(seed),
        "difference_convention": "Transformer minus LSTM; negative favors Transformer for lower-is-better metrics",
        "metrics": {},
    }
    for metric, direction in METRICS.items():
        t = merged[f"{metric}_transformer"].to_numpy(dtype=float)
        l = merged[f"{metric}_lstm"].to_numpy(dtype=float)
        valid = np.isfinite(t) & np.isfinite(l)
        differences = t[valid] - l[valid]
        if len(differences) == 0:
            raise ValueError(f"No finite paired observations for {metric}")
        samples = rng.choice(differences, size=(n_bootstrap, len(differences)), replace=True)
        means = samples.mean(axis=1)
        if np.allclose(differences, 0):
            p_value = 1.0
        else:
            p_value = float(wilcoxon(differences).pvalue)
        mean_difference = float(differences.mean())
        report["metrics"][metric] = {
            "direction": direction,
            "paired_programs": int(len(differences)),
            "transformer_mean": float(t[valid].mean()),
            "lstm_mean": float(l[valid].mean()),
            "mean_difference_transformer_minus_lstm": mean_difference,
            "bootstrap_95_ci": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
            "wilcoxon_signed_rank_p": p_value,
        }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_dir / "paired_program_metrics.csv", index=False)
    (output_dir / "paired_comparison.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformer", required=True)
    parser.add_argument("--lstm", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    args = parser.parse_args()
    paired_comparison(
        args.transformer, args.lstm, args.output,
        seed=args.seed, n_bootstrap=args.bootstrap_resamples,
    )


if __name__ == "__main__":
    main()
