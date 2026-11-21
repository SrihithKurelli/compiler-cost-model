import gzip
import json
import sys

from looper_cost_model.transformer.evaluate import main as evaluate_main
from looper_cost_model.transformer.train import main as train_main


def _write_program(path):
    program = {
        "program_name": "end_to_end_program",
        "initial_execution_time": 1.0,
        "program_annotation": {
            "iterators": ["i", "j"],
            "computations": ["C0"],
            "buffers": {"buffers_info": [{"name": "A"}]},
            "memory_size": 128,
        },
        "schedules_list": [
            {
                "execution_times": [1.0],
                "transformations_list": [],
            },
            {
                "execution_times": [0.5],
                "transformations_list": [
                    {
                        "type": "interchange",
                        "loop_levels": [0, 1],
                        "parameters": [],
                        "computations": ["C0"],
                    }
                ],
            },
            {
                "execution_times": [0.8],
                "transformations_list": [
                    {
                        "type": "shifting",
                        "loop_levels": [0],
                        "parameters": [2],
                        "computations": ["C0"],
                    }
                ],
            },
        ],
    }
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        stream.write(json.dumps(program) + "\n")


def test_one_epoch_training_and_checkpoint_evaluation(tmp_path, monkeypatch):
    train_path = tmp_path / "train.jsonl.gz"
    valid_path = tmp_path / "valid.jsonl.gz"
    output_path = tmp_path / "run"
    evaluation_path = tmp_path / "evaluation"
    _write_program(train_path)
    _write_program(valid_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "transformer-train",
            "--train", str(train_path),
            "--valid", str(valid_path),
            "--output", str(output_path),
            "--epochs", "1",
            "--batch-size", "3",
            "--device", "cpu",
            "--seed", "7",
        ],
    )
    train_main()

    checkpoint = output_path / "best_transformer.pt"
    history = output_path / "history.json"
    assert checkpoint.is_file()
    assert history.is_file()
    assert json.loads(history.read_text(encoding="utf-8"))[0]["epoch"] == 1

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "transformer-evaluate",
            "--data", str(valid_path),
            "--checkpoint", str(checkpoint),
            "--output", str(evaluation_path),
            "--batch-size", "3",
            "--device", "cpu",
        ],
    )
    evaluate_main()

    assert (evaluation_path / "transformer_predictions.csv").is_file()
    assert (evaluation_path / "transformer_program_metrics.csv").is_file()
