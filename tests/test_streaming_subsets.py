import gzip
import json

from scripts.create_streaming_subsets import reservoir_sample, write_subset


def make_program(name, schedule_count=3):
    schedules = [{"transformations_list": [], "execution_times": [1.0]}]
    schedules.extend(
        {
            "transformations_list": [{"type": "interchange", "loop_levels": [0, 1], "parameters": [], "computations": ["C0"]}],
            "execution_times": [0.8],
        }
        for _ in range(schedule_count - 1)
    )
    return {
        "program_name": name,
        "initial_execution_time": 1.0,
        "program_annotation": {
            "iterators": [],
            "computations": {},
            "buffers": {"buffers_info": []},
            "memory_size": 0,
        },
        "schedules_list": schedules,
    }


def write_jsonl_gz(path, programs):
    with gzip.open(path, "wt", encoding="utf-8") as output:
        for program in programs:
            output.write(json.dumps(program) + "\n")


def test_reservoir_sample_is_reproducible_and_bounded(tmp_path):
    source = tmp_path / "source.jsonl.gz"
    write_jsonl_gz(source, [make_program(f"P{i}") for i in range(25)])

    first, seen = reservoir_sample(source, 7, 42, "train")
    second, seen_again = reservoir_sample(source, 7, 42, "train")

    assert seen == seen_again == 25
    assert [p["program_name"] for p in first] == [p["program_name"] for p in second]
    assert len({p["program_name"] for p in first}) == 7


def test_write_subset_preserves_programs_and_requires_one_identity(tmp_path):
    output = tmp_path / "subset.jsonl.gz"
    selected = [make_program("P1"), make_program("P2")]

    write_subset(selected, output)

    with gzip.open(output, "rt", encoding="utf-8") as source:
        loaded = [json.loads(line) for line in source]

    assert [p["program_name"] for p in loaded] == ["P1", "P2"]
    assert all(sum(not s["transformations_list"] for s in p["schedules_list"]) == 1 for p in loaded)
