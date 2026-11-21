import csv
import gzip
import json

from scripts.create_common_datasets import create_common_dataset


def _program(name, schedules):
    return {
        "program_name": name,
        "program_annotation": {"fixture": True},
        "initial_execution_time": 10.0,
        "schedules_list": [
            {"transformations_list": transforms, "execution_times": [time]}
            for transforms, time in schedules
        ],
    }


def test_common_dataset_keeps_only_identical_accepted_programs(tmp_path):
    source = tmp_path / "input.jsonl.gz"
    audit = tmp_path / "audit.csv"
    output = tmp_path / "common.jsonl.gz"
    programs = [
        _program("kept", [([], 10.0), ([[1, 0]], 5.0), ([[2, 0]], 4.0)]),
        _program("rejected", [([], 10.0), ([[3, 0]], 6.0)]),
        _program("one_schedule", [([], 10.0), ([[4, 0]], 7.0)]),
    ]
    with gzip.open(source, "wt", encoding="utf-8") as f:
        for program in programs:
            f.write(json.dumps(program) + "\n")
    rows = [
        ("kept", "[]", "True"),
        ("kept", "[[1,0]]", "True"),
        ("kept", "[[2,0]]", "False"),
        ("rejected", "[]", "False"),
        ("rejected", "[[3,0]]", "False"),
        ("one_schedule", "[]", "True"),
        ("one_schedule", "[[4,0]]", "False"),
    ]
    with open(audit, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["program_name", "schedule_key", "common_accepted"])
        writer.writerows(rows)

    summary = create_common_dataset(source, audit, output)
    with gzip.open(output, "rt", encoding="utf-8") as f:
        retained = [json.loads(line) for line in f]

    assert [p["program_name"] for p in retained] == ["kept"]
    assert len(retained[0]["schedules_list"]) == 2
    assert sum(not s["transformations_list"] for s in retained[0]["schedules_list"]) == 1
    assert summary["input_programs"] == 3
    assert summary["retained_programs"] == 1
    assert summary["skipped_programs"] == 2
