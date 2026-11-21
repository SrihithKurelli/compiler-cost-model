
import gzip
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

TRANSFORMATION_TYPES = [
    "interchange",
    "skewing",
    "tiling",
    "reversal",
    "parallelization",
    "unrolling",
    "fusion",
    "shifting",
]

TRANSFORM_TO_ID = {
    name: index + 1
    for index, name in enumerate(TRANSFORMATION_TYPES)
}

PAD_TOKEN_ID = 0
CLS_TOKEN_ID = len(TRANSFORMATION_TYPES) + 1

MAX_TRANSFORMATIONS = 14
MAX_SEQUENCE_LENGTH = MAX_TRANSFORMATIONS + 1
MAX_LOOP_LEVELS = 3
MAX_PARAMETERS = 3


def signed_log(value):
    value = float(value)
    return math.copysign(math.log1p(abs(value)), value)


def read_programs(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    programs = []

    with opener(path, "rt", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                programs.append(json.loads(line))

    return programs


def extract_program_context(program):
    annotation = program["program_annotation"]

    values = np.array([
        len(annotation["iterators"]),
        len(annotation["computations"]),
        len(annotation["buffers"]["buffers_info"]),
        float(annotation["memory_size"]),
    ], dtype=np.float32)

    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError(
            f"Invalid program context: {program.get('program_name')}"
        )

    return np.log1p(values).astype(np.float32)


def extract_transformation_features(transformation):
    name = transformation["type"]

    if name not in TRANSFORM_TO_ID:
        raise ValueError(f"Unknown transformation: {name}")

    loops = transformation.get("loop_levels", [])
    params = transformation.get("parameters", [])
    computations = transformation.get("computations", [])

    if len(loops) > MAX_LOOP_LEVELS:
        raise ValueError(f"Too many loop levels: {loops}")

    if len(params) > MAX_PARAMETERS:
        raise ValueError(f"Too many parameters: {params}")

    loop_features = np.zeros(MAX_LOOP_LEVELS, dtype=np.float32)
    parameter_features = np.zeros(MAX_PARAMETERS, dtype=np.float32)

    for i, level in enumerate(loops):
        numeric_level = float(level)
        if not math.isfinite(numeric_level) or numeric_level < 0 or not numeric_level.is_integer():
            raise ValueError(f"Invalid loop level: {level}")
        loop_features[i] = numeric_level

    for i, parameter in enumerate(params):
        numeric_parameter = float(parameter)
        if not math.isfinite(numeric_parameter):
            raise ValueError(f"Invalid transformation parameter: {parameter}")
        parameter_features[i] = signed_log(numeric_parameter)

    return {
        "type_id": TRANSFORM_TO_ID[name],
        "loop_levels": loop_features,
        "loop_count": len(loops),
        "parameters": parameter_features,
        "parameter_count": len(params),
        "computation_count": math.log1p(len(computations)),
    }


def encode_schedule(schedule):
    transformations = schedule["transformations_list"]

    if len(transformations) > MAX_TRANSFORMATIONS:
        raise ValueError(
            f"Schedule exceeds {MAX_TRANSFORMATIONS} transformations"
        )

    return [
        extract_transformation_features(t)
        for t in transformations
    ]


def calculate_context_statistics(programs):
    if not programs:
        raise ValueError("Cannot calculate context statistics from an empty dataset.")

    contexts = np.stack([
        extract_program_context(program)
        for program in programs
    ])

    mean = contexts.mean(axis=0)
    std = contexts.std(axis=0)
    std[std < 1e-8] = 1.0

    return mean.astype(np.float32), std.astype(np.float32)


class TransformerScheduleDataset(Dataset):

    def __init__(
        self,
        path,
        context_mean,
        context_std,
        max_transformations=MAX_TRANSFORMATIONS,
    ):
        self.programs = read_programs(path)
        self.context_mean = np.asarray(context_mean, dtype=np.float32)
        self.context_std = np.asarray(context_std, dtype=np.float32)
        if not isinstance(max_transformations, int) or not (
            0 <= max_transformations <= MAX_TRANSFORMATIONS
        ):
            raise ValueError(
                f"max_transformations must be between 0 and {MAX_TRANSFORMATIONS}"
            )
        if self.context_mean.shape != (4,) or self.context_std.shape != (4,):
            raise ValueError("context_mean and context_std must each have shape (4,)")
        if (
            not np.all(np.isfinite(self.context_mean))
            or not np.all(np.isfinite(self.context_std))
            or np.any(self.context_std <= 0)
        ):
            raise ValueError("Context statistics must be finite with positive standard deviations.")

        self.max_transformations = max_transformations
        self.samples = []

        for program in self.programs:
            program_name = program["program_name"]
            initial_time = float(program["initial_execution_time"])

            if not math.isfinite(initial_time) or initial_time <= 0:
                raise ValueError(f"Invalid initial time: {program_name}")

            context = extract_program_context(program)
            context = (context - self.context_mean) / self.context_std

            for schedule in program["schedules_list"]:
                times = schedule["execution_times"]

                if len(times) != 1:
                    raise ValueError(
                        f"Expected singleton execution_times: {program_name}"
                    )

                execution_time = float(times[0])

                if not math.isfinite(execution_time) or execution_time <= 0:
                    raise ValueError(
                        f"Invalid execution time: {program_name}"
                    )

                speedup = max(initial_time / execution_time, 0.01)

                if not math.isfinite(speedup) or speedup <= 0:
                    raise ValueError(f"Invalid speedup: {program_name}")

                transformations = encode_schedule(schedule)

                if len(transformations) > max_transformations:
                    raise ValueError(
                        f"Too many transformations: {program_name}"
                    )

                self.samples.append({
                    "program_name": program_name,
                    "transformations": transformations,
                    "context": context.copy(),
                    "speedup": np.float32(speedup),
                    "execution_time": np.float32(execution_time),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        transformations = sample["transformations"]

        sequence_length = len(transformations) + 1

        type_ids = np.zeros(MAX_SEQUENCE_LENGTH, dtype=np.int64)
        loop_levels = np.zeros(
            (MAX_SEQUENCE_LENGTH, MAX_LOOP_LEVELS), dtype=np.float32
        )
        loop_counts = np.zeros(MAX_SEQUENCE_LENGTH, dtype=np.int64)
        parameters = np.zeros(
            (MAX_SEQUENCE_LENGTH, MAX_PARAMETERS), dtype=np.float32
        )
        parameter_counts = np.zeros(MAX_SEQUENCE_LENGTH, dtype=np.int64)
        computation_counts = np.zeros(MAX_SEQUENCE_LENGTH, dtype=np.float32)

        type_ids[0] = CLS_TOKEN_ID

        for position, transformation in enumerate(transformations, start=1):
            type_ids[position] = transformation["type_id"]
            loop_levels[position] = transformation["loop_levels"]
            loop_counts[position] = transformation["loop_count"]
            parameters[position] = transformation["parameters"]
            parameter_counts[position] = transformation["parameter_count"]
            computation_counts[position] = transformation["computation_count"]

        padding_mask = np.ones(MAX_SEQUENCE_LENGTH, dtype=bool)
        padding_mask[:sequence_length] = False

        return {
            "type_ids": torch.from_numpy(type_ids),
            "loop_levels": torch.from_numpy(loop_levels),
            "loop_counts": torch.from_numpy(loop_counts),
            "parameters": torch.from_numpy(parameters),
            "parameter_counts": torch.from_numpy(parameter_counts),
            "computation_counts": torch.from_numpy(computation_counts),
            "padding_mask": torch.from_numpy(padding_mask),
            "context": torch.from_numpy(sample["context"]),
            "speedup": torch.tensor(sample["speedup"]),
            "execution_time": torch.tensor(sample["execution_time"]),
            "program_name": sample["program_name"],
            "sequence_length": sequence_length,
        }
