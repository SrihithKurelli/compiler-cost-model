
import argparse
import gzip
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def canonical_schedule_key(schedule):
    """Identify schedules by their ordered transformation sequence."""
    transformations = schedule.get("transformations_list", [])
    return json.dumps(
        transformations,
        sort_keys=True,
        separators=(",", ":"),
    )


def valid_measurements(times):
    """Accept only nonempty, finite, strictly positive measurements."""
    if not isinstance(times, list) or not times:
        return False

    try:
        return all(
            math.isfinite(float(t)) and float(t) > 0
            for t in times
        )
    except (TypeError, ValueError):
        return False


def clean_program(program):
    """Aggregate duplicate schedules and construct median-based targets."""

    grouped = {}

    invalid_entries = 0
    duplicate_entries = 0

    for schedule in program["schedules_list"]:
        times = schedule.get("execution_times", [])

        if not valid_measurements(times):
            invalid_entries += 1
            continue

        key = canonical_schedule_key(schedule)

        if key not in grouped:
            grouped[key] = {
                "schedule": dict(schedule),
                "measurements": [],
                "count": 0,
            }
        else:
            duplicate_entries += 1

        grouped[key]["measurements"].extend(
            float(t) for t in times
        )
        grouped[key]["count"] += 1

    # Identify the unique identity schedule.
    identity_key = canonical_schedule_key({
        "transformations_list": []
    })

    if identity_key not in grouped:
        return None, invalid_entries, duplicate_entries

    identity_median = statistics.median(
        grouped[identity_key]["measurements"]
    )

    cleaned_schedules = []

    for key, group in grouped.items():
        schedule = group["schedule"]

        schedule_median = statistics.median(
            group["measurements"]
        )

        # A singleton measurement makes the existing baseline's
        # np.min() calculation equal to our pooled median.
        schedule["execution_times"] = [schedule_median]

        cleaned_schedules.append(schedule)

    # Preserve the baseline's minimum-schedule-count requirement.
    if len(cleaned_schedules) < 2:
        return None, invalid_entries, duplicate_entries

    cleaned_program = dict(program)

    cleaned_program["initial_execution_time"] = identity_median
    cleaned_program["schedules_list"] = cleaned_schedules

    return (
        cleaned_program,
        invalid_entries,
        duplicate_entries,
    )


def process_file(input_path, output_path, max_programs=None):
    input_path = Path(input_path).expanduser()
    output_path = Path(output_path).expanduser()

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    programs_seen = 0
    programs_written = 0
    programs_skipped = 0
    invalid_entries = 0
    duplicate_entries = 0
    schedules_written = 0

    with gzip.open(input_path, "rt", encoding="utf-8") as source:
        with gzip.open(output_path, "wt", encoding="utf-8") as target:

            for line in source:
                line = line.strip()

                if not line:
                    continue

                if max_programs is not None:
                    if programs_seen >= max_programs:
                        break

                program = json.loads(line)
                programs_seen += 1

                cleaned, invalid, duplicates = clean_program(program)

                invalid_entries += invalid
                duplicate_entries += duplicates

                if cleaned is None:
                    programs_skipped += 1
                    continue

                target.write(
                    json.dumps(cleaned, separators=(",", ":")) + "\n"
                )

                programs_written += 1
                schedules_written += len(cleaned["schedules_list"])

                if programs_written % 1000 == 0:
                    print(
                        f"Processed {programs_seen} programs; "
                        f"written {programs_written}"
                    )

    print("\n========== CLEANING SUMMARY ==========")
    print("Input programs processed:", programs_seen)
    print("Programs written:", programs_written)
    print("Programs skipped:", programs_skipped)
    print("Schedules written:", schedules_written)
    print("Duplicate schedule entries merged:", duplicate_entries)
    print("Invalid schedule entries excluded:", invalid_entries)
    print("Output:", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=(
            "~/foml_compiler_project/data/raw/data/pact25/"
            "looperset_v2_pact_train_compact.jsonl.gz"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "../data/processed/"
            "looperset_train_cleaned_sample.jsonl.gz"
        ),
    )

    parser.add_argument(
        "--max-programs",
        type=int,
        default=3,
        help="Maximum number of input programs to process.",
    )

    args = parser.parse_args()

    process_file(
        args.input,
        args.output,
        args.max_programs,
    )
