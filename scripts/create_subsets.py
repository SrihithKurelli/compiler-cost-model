
import argparse
import gzip
import json
import random
from pathlib import Path

from tqdm import tqdm


def create_subset(input_path, output_path, num_programs, seed):
    rng = random.Random(seed)
    programs = []

    with gzip.open(input_path, "rt", encoding="utf-8") as source:
        for line in tqdm(source, desc=f"Reading {Path(input_path).name}"):
            if line.strip():
                programs.append(json.loads(line))

    if num_programs > len(programs):
        raise ValueError(
            f"Requested {num_programs} programs, but only "
            f"{len(programs)} are available."
        )

    selected = rng.sample(programs, num_programs)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(output_path, "wt", encoding="utf-8") as destination:
        for program in selected:
            destination.write(
                json.dumps(program, separators=(",", ":")) + "\n"
            )

    schedule_count = sum(
        len(program["schedules_list"]) for program in selected
    )

    program_names = {program["program_name"] for program in selected}

    print(f"\nCreated: {output_path}")
    print(f"Programs: {len(selected)}")
    print(f"Schedules: {schedule_count}")
    print(f"Seed: {seed}")

    return program_names


def main():
    parser = argparse.ArgumentParser(
        description="Create program-level LOOPerSet subsets."
    )
    parser.add_argument("--train-programs", type=int, default=1000)
    parser.add_argument("--valid-programs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.train_programs <= 0 or args.valid_programs <= 0:
        parser.error("Both program counts must be positive.")

    project_data = Path("../data/processed")

    train_names = create_subset(
        project_data / "looperset_train_cleaned.jsonl.gz",
        project_data / f"looperset_train_{args.train_programs}_cleaned.jsonl.gz",
        args.train_programs,
        args.seed,
    )

    valid_names = create_subset(
        project_data / "looperset_validation_cleaned.jsonl.gz",
        project_data / f"looperset_validation_{args.valid_programs}_cleaned.jsonl.gz",
        args.valid_programs,
        args.seed,
    )

    overlap = train_names.intersection(valid_names)

    if overlap:
        raise RuntimeError(
            f"Program leakage detected: {len(overlap)} overlapping programs."
        )

    print("\nProgram-level split verification: PASSED")
    print(f"Training programs: {len(train_names)}")
    print(f"Validation programs: {len(valid_names)}")
    print(f"Overlapping programs: {len(overlap)}")


if __name__ == "__main__":
    main()
