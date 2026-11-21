
import argparse
from collections import Counter

from tqdm import tqdm

from looper_cost_model import constants
from looper_cost_model.data import features as F
from looper_cost_model.data.looperset import (
    ScheduleTooComplexException,
    normalize_schedule,
    stream_jsonl_gz,
)


def diagnose(dataset_path):
    counts = Counter()
    program_counts = Counter()
    examples = Counter()

    for program in tqdm(stream_jsonl_gz(dataset_path), desc="Diagnosing programs"):
        program_name = program["program_name"]

        if F.drop_program(program):
            program_counts["dropped_program_fewer_than_two_schedules"] += 1
            continue

        try:
            template = F.build_representation_template(program)
        except constants.LoopsDepthException:
            program_counts["program_loops_depth"] += 1
            continue
        except constants.NbAccessException:
            program_counts["program_too_many_accesses"] += 1
            continue
        except constants.ExpressionTooLongException:
            program_counts["program_expression_too_long"] += 1
            continue

        if (
            not program["initial_execution_time"]
            or program["initial_execution_time"] <= 0
        ):
            program_counts["program_invalid_initial_time"] += 1
            continue

        program_counts["programs_processed"] += 1

        for schedule_index, raw_schedule in enumerate(program["schedules_list"]):
            if F.drop_schedule(raw_schedule):
                counts["drop_schedule_empty_or_negative_time"] += 1
                continue

            try:
                schedule = normalize_schedule(
                    program["program_annotation"], raw_schedule
                )
            except ScheduleTooComplexException:
                counts["ScheduleTooComplexException"] += 1
                examples["ScheduleTooComplexException"] += 1
                continue

            try:
                F.fill_schedule_representation(
                    program["program_annotation"], schedule, template
                )
            except constants.NbTransformationException:
                counts["NbTransformationException"] += 1
                examples["NbTransformationException"] += 1
                continue

            counts["successfully_extracted"] += 1

    print("\n=== PROGRAM COUNTS ===")
    for key, value in sorted(program_counts.items()):
        print(f"{key}: {value}")

    print("\n=== SCHEDULE COUNTS ===")
    for key, value in sorted(counts.items()):
        print(f"{key}: {value}")

    dropped = sum(
        value
        for key, value in counts.items()
        if key != "successfully_extracted"
    )
    print(f"\nTotal dropped schedules: {dropped}")
    print(f"Successfully extracted schedules: {counts['successfully_extracted']}")
    print(f"Total schedules accounted for: {dropped + counts['successfully_extracted']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="../data/processed/looperset_validation_30_cleaned.jsonl.gz",
    )
    args = parser.parse_args()
    diagnose(args.data)
