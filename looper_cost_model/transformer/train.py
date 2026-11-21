
import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from looper_cost_model.transformer.dataset import (
    TransformerScheduleDataset,
    calculate_context_statistics,
    read_programs,
)
from looper_cost_model.transformer.model import TransformationTransformer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def mape_loss(predictions, targets):
    return 100.0 * torch.mean(
        torch.abs(targets - predictions) / (targets + 1e-5)
    )


def run_training_epoch(model, loader, optimizer, device):
    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        inputs = {
            key: value.to(device)
            for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        }

        targets = inputs["speedup"]

        optimizer.zero_grad(set_to_none=True)

        predictions = model(inputs)
        loss = mape_loss(predictions, targets)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


@torch.no_grad()
def evaluate_validation(model, loader, device):
    model.eval()

    total_schedule_ape = 0.0
    total_samples = 0

    # Accumulate absolute percentage errors separately per program.
    program_error_sums = {}
    program_counts = {}

    for batch in loader:
        inputs = {
            key: value.to(device)
            for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        }

        predictions = model(inputs)
        targets = inputs["speedup"]

        ape = (
            torch.abs(targets - predictions)
            / (targets + 1e-5)
            * 100.0
        )

        total_schedule_ape += ape.sum().item()
        total_samples += targets.size(0)

        program_names = batch["program_name"]

        for name, error in zip(program_names, ape.cpu().tolist()):
            program_error_sums[name] = (
                program_error_sums.get(name, 0.0) + error
            )
            program_counts[name] = (
                program_counts.get(name, 0) + 1
            )

    per_program_mape = [
        program_error_sums[name] / program_counts[name]
        for name in program_error_sums
    ]

    if not per_program_mape:
        raise RuntimeError("Validation produced no program metrics.")

    schedule_weighted_mape = total_schedule_ape / total_samples
    program_weighted_mape = float(np.mean(per_program_mape))

    return schedule_weighted_mape, program_weighted_mape


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        default="../data/processed/looperset_train_1000_cleaned.jsonl.gz",
    )
    parser.add_argument(
        "--valid",
        default="../data/processed/looperset_validation_300_cleaned.jsonl.gz",
    )
    parser.add_argument(
        "--output",
        default="./runs/transformer_pilot_1000_program_weighted",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--weight-decay", type=float, default=0.015)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cpu")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Device:", device)
    print("Loading training programs...")

    train_programs = read_programs(args.train)

    context_mean, context_std = calculate_context_statistics(
        train_programs
    )

    print("Loading training dataset...")
    train_dataset = TransformerScheduleDataset(
        args.train,
        context_mean,
        context_std,
    )

    print("Loading validation dataset...")
    valid_dataset = TransformerScheduleDataset(
        args.valid,
        context_mean,
        context_std,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    print("\nDataset summary")
    print("Training programs:", len(train_programs))
    print("Training schedules:", len(train_dataset))
    print("Validation programs:", len(valid_dataset.programs))
    print("Validation schedules:", len(valid_dataset))
    print("Training batches:", len(train_loader))
    print("Validation batches:", len(valid_loader))

    model = TransformationTransformer().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_validation_mape = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        train_mape = run_training_epoch(
            model,
            train_loader,
            optimizer,
            device,
        )

        schedule_valid_mape, program_valid_mape = (
            evaluate_validation(model, valid_loader, device)
        )

        elapsed = time.time() - start_time

        record = {
            "epoch": epoch,
            "train_schedule_mape": train_mape,
            "valid_schedule_mape": schedule_valid_mape,
            "valid_program_mape": program_valid_mape,
            "seconds": elapsed,
        }

        history.append(record)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train MAPE: {train_mape:.4f} | "
            f"Valid Schedule MAPE: {schedule_valid_mape:.4f} | "
            f"Valid Program MAPE: {program_valid_mape:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Select checkpoints using the same aggregation as final evaluation.
        if program_valid_mape < best_validation_mape:
            best_validation_mape = program_valid_mape

            checkpoint = {
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "context_mean": context_mean,
                "context_std": context_std,
                "epoch": epoch,
                "valid_mape": program_valid_mape,
                "valid_schedule_mape": schedule_valid_mape,
                "model_config": {
                    "hidden_dim": 128,
                    "num_heads": 4,
                    "num_layers": 3,
                    "feedforward_dim": 256,
                    "dropout": 0.1,
                },
            }

            torch.save(
                checkpoint,
                output_dir / "best_transformer.pt",
            )

            print("  Saved new best program-weighted checkpoint.")

        with open(output_dir / "history.json", "w") as file:
            json.dump(history, file, indent=2)

    print("\nTraining complete.")
    print("Best program-weighted validation MAPE:", best_validation_mape)
    print("Checkpoint:", output_dir / "best_transformer.pt")
    print("History:", output_dir / "history.json")


if __name__ == "__main__":
    main()
