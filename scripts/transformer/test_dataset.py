
import torch

from looper_cost_model.transformer.dataset import (
    TransformerScheduleDataset,
    calculate_context_statistics,
    read_programs,
)

TRAIN_PATH = "../data/processed/looperset_train_1000_cleaned.jsonl.gz"
VALID_PATH = "../data/processed/looperset_validation_300_cleaned.jsonl.gz"


def main():
    print("Reading training programs...")

    programs = read_programs(TRAIN_PATH)

    mean, std = calculate_context_statistics(programs)

    print("Context mean:", mean)
    print("Context std:", std)

    train_dataset = TransformerScheduleDataset(
        TRAIN_PATH, mean, std
    )

    valid_dataset = TransformerScheduleDataset(
        VALID_PATH, mean, std
    )

    print("\nDataset sizes:")
    print("Training schedules:", len(train_dataset))
    print("Validation schedules:", len(valid_dataset))

    sample = train_dataset[0]

    print("\nSample tensor shapes:")

    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {tuple(value.shape)}")

    print("\nSample metadata:")
    print("Program:", sample["program_name"])
    print("Sequence length:", sample["sequence_length"])
    print("Speedup:", sample["speedup"].item())
    print("Execution time:", sample["execution_time"].item())

    assert sample["type_ids"].shape == (15,)
    assert sample["loop_levels"].shape == (15, 3)
    assert sample["parameters"].shape == (15, 3)
    assert sample["context"].shape == (4,)
    assert sample["padding_mask"][0].item() is False

    assert torch.isfinite(sample["speedup"])
    assert sample["speedup"].item() > 0

    # Check identity schedule representation.
    identity_indices = [
        i for i, s in enumerate(train_dataset.samples)
        if len(s["transformations"]) == 0
    ]

    assert len(identity_indices) > 0

    identity = train_dataset[identity_indices[0]]

    assert identity["sequence_length"] == 1
    assert identity["type_ids"][0].item() == 8
    assert identity["padding_mask"][0].item() is False
    assert identity["speedup"].item() > 0

    print("\nIdentity schedule check: PASSED")
    print("Dataset validation: PASSED")


if __name__ == "__main__":
    main()
