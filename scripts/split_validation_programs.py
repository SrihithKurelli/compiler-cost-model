"""Create deterministic, program-disjoint tuning and held-out test splits.

The input is the already sampled validation candidate set. The output programs are
assigned as whole groups, so no schedule from a program can cross the split boundary.
"""
import argparse
import gzip
import json
import random
from pathlib import Path


def open_text(path, mode):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def split_programs(programs, seed=314159, tune_fraction=0.5):
    if not 0.0 < tune_fraction < 1.0:
        raise ValueError("tune_fraction must be strictly between 0 and 1")
    names = [p.get("program_name") for p in programs]
    if any(name is None for name in names):
        raise ValueError("Every program must have a program_name")
    if len(names) != len(set(names)):
        raise ValueError("Input contains duplicate program names")
    if len(programs) < 2:
        raise ValueError("At least two programs are required to create disjoint splits")

    shuffled = list(programs)
    random.Random(seed).shuffle(shuffled)
    tune_count = int(len(shuffled) * tune_fraction)
    tune_count = min(max(tune_count, 1), len(shuffled) - 1)
    tune = shuffled[:tune_count]
    test = shuffled[tune_count:]

    tune_names = {p["program_name"] for p in tune}
    test_names = {p["program_name"] for p in test}
    if tune_names & test_names:
        raise RuntimeError("Program leakage detected between tuning and test splits")
    return tune, test


def write_programs(programs, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as output:
        for program in programs:
            output.write(json.dumps(program, separators=(",", ":")) + "\\n")


def split_file(input_path, output_dir, seed=314159, tune_fraction=0.5):
    with open_text(input_path, "r") as source:
        programs = [json.loads(line) for line in source if line.strip()]

    tune, test = split_programs(programs, seed, tune_fraction)
    output_dir = Path(output_dir)
    tune_path = output_dir / f"looperset_validation_tune_seed{seed}.jsonl.gz"
    test_path = output_dir / f"looperset_validation_test_seed{seed}.jsonl.gz"
    write_programs(tune, tune_path)
    write_programs(test, test_path)

    summary = {
        "input": str(input_path),
        "seed": int(seed),
        "tune_fraction": float(tune_fraction),
        "input_programs": len(programs),
        "tuning_programs": len(tune),
        "heldout_test_programs": len(test),
        "tuning_program_names": sorted(p["program_name"] for p in tune),
        "heldout_test_program_names": sorted(p["program_name"] for p in test),
        "program_overlap": 0,
        "tuning_file": str(tune_path),
        "test_file": str(test_path),
    }
    summary_path = output_dir / f"validation_split_seed{seed}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--tune-fraction", type=float, default=0.5)
    args = parser.parse_args()
    split_file(args.input, args.output_dir, args.seed, args.tune_fraction)


if __name__ == "__main__":
    main()
