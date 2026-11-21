import pandas as pd
import pytest

from scripts.compare_paired_metrics import paired_comparison


def _metrics(names, offset=0.0):
    rows = []
    for i, name in enumerate(names):
        rows.append({
            "program_name": name,
            "mape": 10.0 + i + offset,
            "spearman": 0.5 + i * 0.01 - offset,
            "ndcg": 0.7 + i * 0.01 - offset,
            "ndcg_at_5": 0.6 + i * 0.01 - offset,
            "ndcg_at_1": 0.4 + i * 0.01 - offset,
            "top1_optimal": float(i % 2 == 0),
            "regret_percent": 20.0 + i + offset,
        })
    return pd.DataFrame(rows)


def test_paired_comparison_uses_exact_program_intersection(tmp_path):
    t = tmp_path / "transformer.csv"
    l = tmp_path / "lstm.csv"
    _metrics(["p1", "p2", "p3"]).to_csv(t, index=False)
    _metrics(["p1", "p2", "p3"], offset=2.0).to_csv(l, index=False)

    report = paired_comparison(t, l, tmp_path / "out", n_bootstrap=200)
    assert report["programs"] == 3
    assert report["metrics"]["mape"]["paired_programs"] == 3
    assert (tmp_path / "out" / "paired_program_metrics.csv").exists()
    assert (tmp_path / "out" / "paired_comparison.json").exists()


def test_paired_comparison_rejects_different_program_sets(tmp_path):
    t = tmp_path / "transformer.csv"
    l = tmp_path / "lstm.csv"
    _metrics(["p1", "p2", "p3"]).to_csv(t, index=False)
    _metrics(["p1", "p2", "p4"]).to_csv(l, index=False)

    with pytest.raises(ValueError, match="Program sets differ"):
        paired_comparison(t, l, tmp_path / "out", n_bootstrap=100)


def test_matched_lstm_export_script_imports():
    import scripts.evaluate_matched_lstm
    assert callable(scripts.evaluate_matched_lstm.main)
