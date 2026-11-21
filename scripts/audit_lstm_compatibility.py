
import argparse
import csv
import gzip
import json
from collections import Counter
from pathlib import Path

from looper_cost_model.data import features as F
from looper_cost_model.data.looperset import (
    ScheduleTooComplexException,
    normalize_schedule,
    stream_jsonl_gz,
)
from looper_cost_model.constants import (
    ExpressionTooLongException,
    LoopsDepthException,
    NbAccessException,
    NbTransformationException,
)
from looper_cost_model.transformer.dataset import (
    MAX_TRANSFORMATIONS,
    encode_schedule,
)


PROGRAM_ERRORS = (
    LoopsDepthException,
    NbAccessException,
    ExpressionTooLongException,
)

SCHEDULE_ERRORS = (
    NbTransformationException,
    ScheduleTooComplexException,
)


def schedule_key(schedule):
    """Stable identifier based on the original ordered transformation sequence."""
    return json.dumps(
        schedule.get("transformations_list", []),
        sort_keys=True,
        separators=(",", ":"),
    )


def is_identity(schedule):
    return len(schedule.get("transformations_list", [])) == 0


def audit_program(program):
    name = program["program_name"]
    schedules = program["schedules_list"]

    result = {
        "program_name": name,
        "program_accepted": False,
        "program_reason": "",
        "total_schedules": len(schedules),
        "lstm_accepted": set(),
        "transformer_accepted": set(),
        "lstm_rejected": Counter(),
        "transformer_rejected": Counter(),
        "identity_total": 0,
        "identity_lstm": 0,
        "identity_transformer": 0,
        "unexpected_errors": [],
    }

    # Match the original LSTM's program-level filtering.
    if F.drop_program(program):
        result["program_reason"] = "drop_program"
        return result

    annotation = program["program_annotation"]

    try:
        template = F.build_representation_template(program)
    except PROGRAM_ERRORS as exc:
        result["program_reason"] = type(exc).__name__
        return result
    except Exception as exc:
        result["program_reason"] = "unexpected:" + type(exc).__name__
        result["unexpected_errors"].append(
            f"program_template: {type(exc).__name__}: {exc}"
        )
        return result

    initial_time = program.get("initial_execution_time")

    if initial_time is None or initial_time <= 0:
        result["program_reason"] = "invalid_initial_execution_time"
        return result

    result["program_accepted"] = True

    for schedule in schedules:
        key = schedule_key(schedule)
        identity = is_identity(schedule)

        if identity:
            result["identity_total"] += 1

        # Transformer compatibility.
        try:
            transformations = encode_schedule(schedule)

            if len(transformations) > MAX_TRANSFORMATIONS:
                raise ValueError(
                    f"Schedule exceeds {MAX_TRANSFORMATIONS} transformations"
                )

            result["transformer_accepted"].add(key)

            if identity:
                result["identity_transformer"] += 1

        except Exception as exc:
            reason = type(exc).__name__
            result["transformer_rejected"][reason] += 1

        # LSTM compatibility: reproduce the original extraction path.
        if F.drop_schedule(schedule):
            result["lstm_rejected"]["drop_schedule"] += 1
            continue

        try:
            normalized = normalize_schedule(annotation, schedule)

            F.fill_schedule_representation(
                annotation,
                normalized,
                template,
            )

            result["lstm_accepted"].add(key)

            if identity:
                result["identity_lstm"] += 1

        except SCHEDULE_ERRORS as exc:
            result["lstm_rejected"][type(exc).__name__] += 1

        except Exception as exc:
            reason = type(exc).__name__
            result["lstm_rejected"]["unexpected:" + reason] += 1
            result["unexpected_errors"].append(
                f"schedule: {reason}: {exc}"
            )

    return result


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def audit_split(input_path, output_dir, split_name):
    output_dir.mkdir(parents=True, exist_ok=True)

    program_rows = []
    schedule_rows = []
    unexpected_rows = []

    total_programs = 0
    accepted_programs = 0
    program_rejections = Counter()
    lstm_rejections = Counter()
    transformer_rejections = Counter()

    total_schedules = 0
    lstm_total = 0
    transformer_total = 0
    common_total = 0
    identity_total = 0
    identity_lstm = 0
    identity_transformer = 0

    for program in stream_jsonl_gz(input_path):
        total_programs += 1
        result = audit_program(program)

        name = result["program_name"]
        total_schedules += result["total_schedules"]

        if result["program_accepted"]:
            accepted_programs += 1
        else:
            program_rejections[result["program_reason"]] += 1

        lstm_rejections.update(result["lstm_rejected"])
        transformer_rejections.update(result["transformer_rejected"])

        lstm_keys = result["lstm_accepted"]
        transformer_keys = result["transformer_accepted"]
        common_keys = lstm_keys & transformer_keys

        lstm_total += len(lstm_keys)
        transformer_total += len(transformer_keys)
        common_total += len(common_keys)

        identity_total += result["identity_total"]
        identity_lstm += result["identity_lstm"]
        identity_transformer += result["identity_transformer"]

        program_rows.append({
            "program_name": name,
            "program_accepted": result["program_accepted"],
            "program_reason": result["program_reason"],
            "total_schedules": result["total_schedules"],
            "lstm_accepted": len(lstm_keys),
            "transformer_accepted": len(transformer_keys),
            "common_accepted": len(common_keys),
            "identity_total": result["identity_total"],
            "identity_lstm": result["identity_lstm"],
            "identity_transformer": result["identity_transformer"],
        })

        # Schedule-level comparison, using the original schedule sequence.
        for schedule in program["schedules_list"]:
            key = schedule_key(schedule)
            identity = is_identity(schedule)

            lstm_ok = key in lstm_keys
            transformer_ok = key in transformer_keys

            schedule_rows.append({
                "program_name": name,
                "schedule_key": key,
                "identity": identity,
                "lstm_accepted": lstm_ok,
                "transformer_accepted": transformer_ok,
                "common_accepted": lstm_ok and transformer_ok,
            })

        for error in result["unexpected_errors"]:
            unexpected_rows.append({
                "program_name": name,
                "error": error,
            })

        if total_programs % 500 == 0:
            print(f"{split_name}: audited {total_programs} programs")

    write_csv(
        output_dir / f"{split_name}_programs.csv",
        program_rows,
        [
            "program_name",
            "program_accepted",
            "program_reason",
            "total_schedules",
            "lstm_accepted",
            "transformer_accepted",
            "common_accepted",
            "identity_total",
            "identity_lstm",
            "identity_transformer",
        ],
    )

    write_csv(
        output_dir / f"{split_name}_schedules.csv",
        schedule_rows,
        [
            "program_name",
            "schedule_key",
            "identity",
            "lstm_accepted",
            "transformer_accepted",
            "common_accepted",
        ],
    )

    write_csv(
        output_dir / f"{split_name}_unexpected_errors.csv",
        unexpected_rows,
        ["program_name", "error"],
    )

    summary = {
        "split": split_name,
        "input": str(input_path),
        "programs": total_programs,
        "programs_with_valid_templates": accepted_programs,
        "program_rejections": dict(program_rejections),
        "total_schedules": total_schedules,
        "lstm_accepted_schedules": lstm_total,
        "transformer_accepted_schedules": transformer_total,
        "common_schedules": common_total,
        "lstm_schedule_rejections": dict(lstm_rejections),
        "transformer_schedule_rejections": dict(transformer_rejections),
        "identity_schedules": identity_total,
        "identity_lstm": identity_lstm,
        "identity_transformer": identity_transformer,
        "unexpected_error_count": len(unexpected_rows),
    }

    with open(
        output_dir / f"{split_name}_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    print(f"\n===== {split_name.upper()} SUMMARY =====")
    print(json.dumps(summary, indent=2))

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument("--output", default="./outputs/lstm_compatibility")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_summary = audit_split(
        args.train,
        output_dir,
        "train",
    )

    valid_summary = audit_split(
        args.valid,
        output_dir,
        "validation",
    )

    combined = {
        "train": train_summary,
        "validation": valid_summary,
    }

    with open(
        output_dir / "combined_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(combined, file, indent=2)

    print("\nAudit completed.")
    print(f"Reports saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
