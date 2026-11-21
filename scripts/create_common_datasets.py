
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

    with open(audit_csv, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["common_accepted"] == "True":
                name = row["program_name"]
                key = row["schedule_key"]

                common.setdefault(name, set()).add(key)

    return common


def create_common_dataset(input_path, audit_path, output_path):
    common = load_common_schedules(audit_path)

    total_programs = 0
    total_schedules = 0
    retained_schedules = 0
    retained_identity = 0
    missing_programs = set(common)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(output_path, "wt", encoding="utf-8") as output:
        for program in stream_jsonl_gz(input_path):
            total_programs += 1
            name = program["program_name"]

            if name not in common:
                raise ValueError(
                    f"Program missing from compatibility audit: {name}"
                )

            missing_programs.discard(name)
            accepted_keys = common[name]

            original_schedules = program["schedules_list"]
            total_schedules += len(original_schedules)

            filtered_schedules = []

            for schedule in original_schedules:
                key = schedule_key(schedule)

                if key in accepted_keys:
                    filtered_schedules.append(schedule)

            # Verify that every expected schedule was found exactly once.
            retained_keys = [
                schedule_key(schedule)
                for schedule in filtered_schedules
            ]

            if len(retained_keys) != len(set(retained_keys)):
                raise ValueError(
                    f"Duplicate retained schedule keys in {name}"
                )

            if set(retained_keys) != accepted_keys:
                raise ValueError(
                    f"Schedule mismatch for {name}: "
                    f"expected {len(accepted_keys)}, "
                    f"found {len(retained_keys)}"
                )

            identity_count = sum(
                not schedule.get("transformations_list", [])
                for schedule in filtered_schedules
            )

            if identity_count != 1:
                raise ValueError(
                    f"Expected exactly one identity schedule in {name}, "
                    f"found {identity_count}"
                )

            retained_identity += identity_count
            retained_schedules += len(filtered_schedules)

            program["schedules_list"] = filtered_schedules

            output.write(json.dumps(program, separators=(",", ":")) + "\n")

    if missing_programs:
        raise ValueError(
            f"{len(missing_programs)} audit programs were not found "
            "in the input dataset"
        )

    print("\n===== COMMON DATASET SUMMARY =====")
    print("Input:", input_path)
    print("Output:", output_path)
    print("Programs:", total_programs)
    print("Original schedules:", total_schedules)
    print("Retained schedules:", retained_schedules)
    print("Removed schedules:", total_schedules - retained_schedules)
    print("Retained identity schedules:", retained_identity)
    print("All programs matched:", not missing_programs)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    create_common_dataset(
        args.input,
        args.audit,
        args.output,
    )


if __name__ == "__main__":
    main()
