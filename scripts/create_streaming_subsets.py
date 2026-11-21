"""Create reproducible, memory-bounded program subsets from cleaned LOOPerSet JSONL.

Unlike create_subsets.py, this script uses reservoir sampling and never keeps the
entire split in memory. It is intended for constrained CI/remote experiment runners.
"""
import argparse
import gzip
import json
import random
from pathlib import Path


def open_text(path, mode):
    path = Path(path)
    return gzip.open(path, mode + "t", encoding="utf-8") if path.suffix == ".gz" else open(path, mode, encoding="utf-8")


def reservoir_sample(path, count, seed, label):
    rng = random.Random(seed)
    reservoir = []
    seen = 0

    with open_text(path, "r") as source:
        for line in source:
            if not line.strip():
                continue
            program = json.loads(line)
            seen += 1
            if len(reservoir) < count:
                reservoir.append(program)
            else:
                index = rng.randrange(seen)
                if index < count:
                    reservoir[index] = program

    if seen < count:
        raise ValueError(f"{label}: requested {count} programs, found only {seen}")

    rng.shuffle(reservoir)
    names = [item["program_name"] for item in reservoir]
    if len(names) != len(set(names)):
        raise ValueError(f"{label}: duplicate program names in sampled subset")

    return reservoir, seen


def write_subset(programs, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as output:
        for program in programs:
            schedules = program.get("schedules_list", [])
            identities = sum(not s.get("transformations_list", []) for s in schedules)
            if identities != 1:
                raise ValueError(f"{program.get('program_name')}: expected one identity schedule, got {identities}")
            if len(schedules) < 2:
                raise ValueError(f"{program.get('program_name')}: fewer than two schedules")
            output.write(json.dumps(program, separators=(",", ":")) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--train-programs", type=int, default=1000)
    parser.add_argument("--valid-programs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.train_programs < 1 or args.valid_programs < 1:
        parser.error("program counts must be positive")

    train, train_total = reservoir_sample(args.train, args.train_programs, args.seed, "train")
    valid, valid_total = reservoir_sample(args.valid, args.valid_programs, args.seed + 1, "validation")

    train_names = {p["program_name"] for p in train}
    valid_names = {p["program_name"] for p in valid}
    overlap = train_names & valid_names
    if overlap:
        raise RuntimeError(f"Program leakage detected: {len(overlap)} overlapping names")

    out = Path(args.output_dir)
    train_path = out / f"looperset_train_{args.train_programs}_remote_seed{args.seed}.jsonl.gz"
    valid_path = out / f"looperset_validation_{args.valid_programs}_remote_seed{args.seed}.jsonl.gz"
    write_subset(train, train_path)
    write_subset(valid, valid_path)

    train_schedules = sum(len(p["schedules_list"]) for p in train)
    valid_schedules = sum(len(p["schedules_list"]) for p in valid)
    print("===== STREAMING SUBSET SUMMARY =====")
    print(f"Train source programs: {train_total}")
    print(f"Validation source programs: {valid_total}")
    print(f"Train selected programs/schedules: {len(train)}/{train_schedules}")
    print(f"Validation selected programs/schedules: {len(valid)}/{valid_schedules}")
    print(f"Program overlap: {len(overlap)}")
    print(f"Train output: {train_path}")
    print(f"Validation output: {valid_path}")


if __name__ == "__main__":
    main()
