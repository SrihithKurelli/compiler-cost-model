"""Build LSTM/Transformer datasets containing exactly the same accepted schedules.

The compatibility audit is authoritative: a schedule is retained only when both
implementations accept it. Programs without at least two common schedules, or without
exactly one common identity schedule, are excluded from both models' datasets.
"""
import argparse
import csv
import gzip
import json
from pathlib import Path

from looper_cost_model.data.looperset import stream_jsonl_gz


def schedule_key(schedule):
    return json.dumps(
        schedule.get("transformations_list", []),
        sort_keys=True,
        separators=(",", ":"),
    )


def load_common_schedules(audit_csv):
    common = {}
    audited_programs = set()
    with open(audit_csv, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            name = row["program_name"]
            audited_programs.add(name)
            if row["common_accepted"] == "True":
                common.setdefault(name, set()).add(row["schedule_key"])
    return common, audited_programs


def create_common_dataset(input_path, audit_path, output_path):
    common, audited_programs = load_common_schedules(audit_path)
    total_programs = retained_programs = skipped_programs = 0
    total_schedules = retained_schedules = retained_identity = 0
    seen_programs = set()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(output_path, "wt", encoding="utf-8") as output:
        for program in stream_jsonl_gz(input_path):
            total_programs += 1
            name = program["program_name"]
            if name in seen_programs:
                raise ValueError(f"Duplicate program in input: {name}")
            seen_programs.add(name)
            if name not in audited_programs:
                raise ValueError(f"Program missing from compatibility audit: {name}")

            original_schedules = program["schedules_list"]
            total_schedules += len(original_schedules)
            accepted_keys = common.get(name, set())
            filtered = [s for s in original_schedules if schedule_key(s) in accepted_keys]
            retained_keys = [schedule_key(s) for s in filtered]
            if len(retained_keys) != len(set(retained_keys)):
                raise ValueError(f"Duplicate retained schedule keys in {name}")
            if set(retained_keys) != accepted_keys:
                raise ValueError(
                    f"Schedule mismatch for {name}: expected {len(accepted_keys)}, "
                    f"found {len(retained_keys)}"
                )

            identity_count = sum(not s.get("transformations_list", []) for s in filtered)
            if len(filtered) < 2 or identity_count != 1:
                skipped_programs += 1
                continue

            program["schedules_list"] = filtered
            output.write(json.dumps(program, separators=(",", ":")) + "\n")
            retained_programs += 1
            retained_schedules += len(filtered)
            retained_identity += identity_count

    missing = audited_programs - seen_programs
    if missing:
        raise ValueError(f"{len(missing)} audit programs were not found in the input dataset")
    if retained_programs == 0:
        raise ValueError("No programs remain after applying common compatibility filters")

    summary = {
        "input": str(input_path), "audit": str(audit_path), "output": str(output_path),
        "input_programs": total_programs, "retained_programs": retained_programs,
        "skipped_programs": skipped_programs, "input_schedules": total_schedules,
        "retained_schedules": retained_schedules,
        "retained_identity_schedules": retained_identity,
    }
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n===== COMMON DATASET SUMMARY =====")
    print(json.dumps(summary, indent=2))
    print("Summary:", summary_path)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    create_common_dataset(args.input, args.audit, args.output)


if __name__ == "__main__":
    main()
