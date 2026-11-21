
import torch
from torch.utils.data import DataLoader

from looper_cost_model.transformer.dataset import (
    TransformerScheduleDataset,
    calculate_context_statistics,
    read_programs,
)

from looper_cost_model.transformer.model import (
    TransformationTransformer,
)

TRAIN_PATH = "../data/processed/looperset_train_1000_cleaned.jsonl.gz"


def main():
    torch.manual_seed(42)

    print("Loading dataset...")

    programs = read_programs(TRAIN_PATH)
    mean, std = calculate_context_statistics(programs)

    dataset = TransformerScheduleDataset(
        TRAIN_PATH,
        mean,
        std,
    )

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))

    # Move tensor inputs to CPU.
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.cpu()

    print("Initializing Transformer...")

    model = TransformationTransformer()
    model.train()

    print("Model parameters:", sum(
        p.numel() for p in model.parameters()
    ))

    print("\nRunning forward pass...")

    predictions = model(batch)

    print("Prediction shape:", predictions.shape)
    print("Prediction range:",
          predictions.min().item(),
          predictions.max().item())

    assert predictions.shape == (16,)
    assert torch.isfinite(predictions).all()
    assert (predictions > 0).all()

    print("\nRunning MAPE loss and backward pass...")

    targets = batch["speedup"]

    loss = 100.0 * torch.mean(
        torch.abs(targets - predictions) / (targets + 1e-5)
    )

    print("MAPE loss:", loss.item())

    loss.backward()

    gradients = [
        p.grad for p in model.parameters()
        if p.grad is not None
    ]

    assert len(gradients) > 0
    assert all(torch.isfinite(g).all() for g in gradients)

    print("Gradient tensors:", len(gradients))
    print("All gradients finite: True")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=4e-5,
        weight_decay=0.015,
    )

    optimizer.step()

    print("\nTransformer smoke test: PASSED")


if __name__ == "__main__":
    main()
