import gzip
import json

import pytest

from scripts.split_validation_programs import split_programs, split_file


def make_program(name):
    return {
        "program_name": name,
        "schedules_list": [
            {"transformations_list": [], "execution_times": [1.0]},
            {"transformations_list": [[1, 0]], "execution_times": [0.8]},
        ],
    }


def test_split_is_reproducible_and_program_disjoint():
    programs = [make_program(f"P{i}") for i in range(20)]
    tune_a, test_a = split_programs(programs, seed=17)
    tune_b, test_b = split_programs(programs, seed=17)

    assert [p["program_name"] for p in tune_a] == [p["program_name"] for p in tune_b]
    assert [p["program_name"] for p in test_a] == [p["program_name"] for p in test_b]
    assert {p["program_name"] for p in tune_a}.isdisjoint(
        {p["program_name"] for p in test_a}
    )
    assert len(tune_a) + len(test_a) == len(programs)


def test_split_file_writes_valid_gzip_jsonl_and_metadata(tmp_path):
    source = tmp_path / "validation.jsonl.gz"
    with gzip.open(source, "wt", encoding="utf-8") as output:
        for i in range(10):
            output.write(json.dumps(make_program(f"P{i}")) + "\n")

    summary = split_file(source, tmp_path / "out", seed=23)
    assert summary["tuning_programs"] == 5
    assert summary["heldout_test_programs"] == 5
    assert summary["program_overlap"] == 0

    for key in ("tuning_file", "test_file"):
        with gzip.open(summary[key], "rt", encoding="utf-8") as input_file:
            rows = [json.loads(line) for line in input_file]
        assert len(rows) == 5


def test_split_rejects_duplicate_program_names():
    programs = [make_program("same"), make_program("same")]
    with pytest.raises(ValueError, match="duplicate"):
        split_programs(programs, seed=1)
