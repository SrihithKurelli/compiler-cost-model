import gzip
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from looper_cost_model.transformer.dataset import (
    CLS_TOKEN_ID,
    MAX_SEQUENCE_LENGTH,
    TRANSFORM_TO_ID,
    TransformerScheduleDataset,
    calculate_context_statistics,
    read_programs,
)
from looper_cost_model.transformer.model import TransformationTransformer
from looper_cost_model.transformer.runtime import resolve_device


def _program():
    return {
        "program_name": "synthetic_program",
        "initial_execution_time": 1.0,
        "program_annotation": {
            "iterators": ["i", "j"],
            "computations": ["C0"],
            "buffers": {"buffers_info": [{"name": "A"}]},
            "memory_size": 128,
        },
        "schedules_list": [
            {
                "execution_times": [200.0],
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
                "execution_times": [1.0],
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


def test_dataset_clips_speedup_and_encodes_identity(tmp_path):
    path = tmp_path / "synthetic.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        stream.write(json.dumps(_program()) + "\n")

    programs = read_programs(path)
    mean, std = calculate_context_statistics(programs)
    dataset = TransformerScheduleDataset(path, mean, std)

    assert len(dataset) == 3
    identity = dataset[0]
    transformed = dataset[1]

    assert identity["sequence_length"] == 1
    assert identity["type_ids"][0].item() == CLS_TOKEN_ID
    assert identity["padding_mask"][0].item() is False
    assert identity["speedup"].item() == np.float32(0.01)

    assert transformed["sequence_length"] == 2
    assert transformed["type_ids"][1].item() == TRANSFORM_TO_ID["interchange"]
    assert transformed["speedup"].item() == np.float32(2.0)
    assert transformed["padding_mask"][:2].tolist() == [False, False]
    assert transformed["padding_mask"][2:].all()

    shifting = dataset[2]
    assert shifting["type_ids"][1].item() == TRANSFORM_TO_ID["shifting"]
    assert shifting["speedup"].item() == np.float32(1.0)


def test_model_forward_and_backward_with_padding():
    batch = {
        "type_ids": torch.tensor([
            [CLS_TOKEN_ID, 0, 0, 0],
            [CLS_TOKEN_ID, TRANSFORM_TO_ID["shifting"], 0, 0],
        ]),
        "loop_levels": torch.zeros(2, 4, 3),
        "loop_counts": torch.zeros(2, 4, dtype=torch.long),
        "parameters": torch.zeros(2, 4, 3),
        "parameter_counts": torch.zeros(2, 4, dtype=torch.long),
        "computation_counts": torch.zeros(2, 4),
        "padding_mask": torch.tensor([
            [False, True, True, True],
            [False, False, True, True],
        ]),
        "context": torch.zeros(2, 4),
    }
    model = TransformationTransformer(max_sequence_length=4)
    predictions = model(batch)

    assert predictions.shape == (2,)
    assert torch.isfinite(predictions).all()
    assert (predictions > 0).all()

    loss = predictions.mean()
    loss.backward()
    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_dataset_batch_collates_program_names(tmp_path):
    path = tmp_path / "synthetic.jsonl"
    path.write_text(json.dumps(_program()) + "\n", encoding="utf-8")
    mean, std = calculate_context_statistics(read_programs(path))
    dataset = TransformerScheduleDataset(path, mean, std)
    batch = next(iter(DataLoader(dataset, batch_size=2)))

    assert batch["type_ids"].shape == (2, MAX_SEQUENCE_LENGTH)
    assert batch["context"].shape == (2, 4)
    assert batch["program_name"] == ["synthetic_program", "synthetic_program"]


def test_device_resolution_cpu_and_auto():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in {"cpu", "mps", "cuda"}


def test_transform_vocabulary_covers_all_looperset_transformations():
    expected = {
        "interchange",
        "skewing",
        "tiling",
        "reversal",
        "parallelization",
        "unrolling",
        "fusion",
        "shifting",
    }
    assert set(TRANSFORM_TO_ID) == expected


def test_explicit_unavailable_accelerator_fails_loudly():
    if not torch.cuda.is_available():
        import pytest

        with pytest.raises(RuntimeError, match="CUDA was requested"):
            resolve_device("cuda")
    if not (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    ):
        import pytest

        with pytest.raises(RuntimeError, match="MPS was requested"):
            resolve_device("mps")
