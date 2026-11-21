
import argparse
import gzip
import json
import math
from pathlib import Path


def canonical_schedule_key(schedule):
    transformations = schedule.get("transformations_list", [])
    return json.dumps(
        transformations,
        sort_keys=True,
        separators=(",", ":"),
    )


def audit_file(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    programs = 0
    schedules = 0
    errors = []
    program_names = set()

    with gzip.open(path, "rt", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            program = json.loads(line)
            programs += 1
            name = program.get("program_name")

            if name is not None:
                if name in program_names:
                    errors.append(f"Duplicate program name: {name}")
                program_names.add(name)

            schedule_list = program.get("schedules_list", [])

            if len(schedule_list) < 2:
                errors.append(f"{name}: fewer than 2 schedules")

            keys = set()
            identity_count = 0

            for schedule in schedule_list:
                schedules += 1
                key = canonical_schedule_key(schedule)

                if key in keys:
                    errors.append(f"{name}: duplicate schedule")
                keys.add(key)

                if key == "[]":
                    identity_count += 1

                times = schedule.get("execution_times", [])

                if len(times) != 1:
                    errors.append(
                        f"{name}: execution_times is not a singleton"
                    )
                    continue

                try:
                    value = float(times[0])
                    if not math.isfinite(value) or value <= 0:
                        errors.append(f"{name}: invalid execution time")
                except (TypeError, ValueError):
                    errors.append(f"{name}: non-numeric execution time")

            if identity_count != 1:
                errors.append(
                    f"{name}: identity schedule count = {identity_count}"
                )
                continue

            identity_schedule = next(
                s for s in schedule_list
                if canonical_schedule_key(s) == "[]"
            )

            identity_time = float(
                identity_schedule["execution_times"][0]
            )
            initial_time = float(program["initial_execution_time"])

            if not math.isclose(
                identity_time,
                initial_time,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                errors.append(
                    f"{name}: initial_execution_time does not "
                    "match identity schedule"
                )

    return {
        "programs": programs,
        "schedules": schedules,
        "program_names": program_names,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Verify cleaned LOOPerSet train and validation datasets."
    )
    parser.add_argument(
        "--train",
        default="../data/processed/looperset_train_cleaned.jsonl.gz",
    )
    parser.add_argument(
        "--valid",
        default="../data/processed/looperset_validation_cleaned.jsonl.gz",
    )
    args = parser.parse_args()

    print("Auditing training split...")
    train = audit_file(args.train)

    print("Auditing validation split...")
    validation = audit_file(args.valid)

    overlap = train["program_names"] & validation["program_names"]

    print("\n========== AUDIT SUMMARY ==========")

    for label, result in (
        ("Training", train),
        ("Validation", validation),
    ):
        print(f"\n{label}:")
        print("Programs:", result["programs"])
        print("Schedules:", result["schedules"])
        print("Errors:", len(result["errors"]))

        for error in result["errors"][:10]:
            print("  ERROR:", error)

    print("\nProgram-name overlap:", len(overlap))

    if overlap:
        print("Examples:", list(overlap)[:10])

    passed = (
        len(train["errors"]) == 0
        and len(validation["errors"]) == 0
        and len(overlap) == 0
    )

    if not passed:
        raise SystemExit("\nCLEANED DATA AUDIT FAILED")

    print("\nCLEANED DATA AUDIT PASSED")


if __name__ == "__main__":
    main()
