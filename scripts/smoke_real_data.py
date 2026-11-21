
import os
import sys

# Add repository root before importing project modules.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

import numpy as np
import torch

from looper_cost_model.constants import (
    MAX_EXPR_LEN,
    MAX_NUM_TRANSFORMATIONS,
    MAX_TAGS,
    ExpressionTooLongException,
    LoopsDepthException,
    NbAccessException,
    NbTransformationException,
)
from looper_cost_model.data import features as F
from looper_cost_model.data.looperset import (
    normalize_schedule,
    stream_jsonl_gz,
    ScheduleTooComplexException,
)
from looper_cost_model.model import LOOPerCostModel
from looper_cost_model.predict import _schedule_input


DATA_FILE = os.path.expanduser(
    "~/foml_compiler_project/data/raw/data/pact25/"
    "looperset_v2_pact_train_compact.jsonl.gz"
)

MAX_PROGRAMS = 3
MAX_SCHEDULES_PER_PROGRAM = 20


def check_finite_gradients(model):
    """Check that all existing parameter gradients are finite."""
    gradients_checked = 0

    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all():
                raise RuntimeError(
                    f"Non-finite gradient found in parameter: {name}"
                )
            gradients_checked += 1

    if gradients_checked == 0:
        raise RuntimeError("No model gradients were produced")

    return gradients_checked


def test_mps(model_input):
    """Run one real-data forward and backward pass on Apple MPS."""

    if not torch.backends.mps.is_built():
        print("\nMPS TEST SKIPPED: PyTorch was not built with MPS support.")
        return False

    if not torch.backends.mps.is_available():
        print("\nMPS TEST SKIPPED: MPS is not available.")
        return False

    device = torch.device("mps")
    model = LOOPerCostModel().to(device)
    model.train()

    try:
        # Confirm model parameters are on MPS.
        parameter_device = next(model.parameters()).device
        assert parameter_device.type == "mps"

        # Transfer this single real datapoint to MPS.
        tree = model_input[0]
        mps_input = (tree,) + tuple(
            tensor.to(device) for tensor in model_input[1:]
        )

        model.zero_grad(set_to_none=True)

        prediction = model(mps_input)

        if not torch.isfinite(prediction).all():
            raise RuntimeError("MPS produced a non-finite prediction")

        # Use a simple scalar target for the backward compatibility test.
        target = torch.ones_like(prediction)
        loss = torch.mean((prediction - target) ** 2)

        if not torch.isfinite(loss):
            raise RuntimeError("MPS produced a non-finite loss")

        loss.backward()

        gradients_checked = check_finite_gradients(model)

        # Ensure asynchronous MPS operations have completed.
        torch.mps.synchronize()

        print("\n========== MPS TEST ==========")
        print("Device:", parameter_device)
        print("Prediction:", prediction.detach().cpu().item())
        print("Loss:", loss.detach().cpu().item())
        print("Finite gradients checked:", gradients_checked)
        print("MPS FORWARD/BACKWARD TEST PASSED")

        return True

    except Exception as exc:
        print("\n========== MPS TEST FAILED ==========")
        print(type(exc).__name__ + ":", exc)
        raise


def main():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    cpu_model = LOOPerCostModel().eval()

    programs_checked = 0
    schedules_checked = 0
    schedules_skipped = 0
    mps_input = None

    for program in stream_jsonl_gz(DATA_FILE):
        if F.drop_program(program):
            continue

        program_name = program["program_name"]
        annotation = program["program_annotation"]

        try:
            template = F.build_representation_template(program)
        except (
            LoopsDepthException,
            NbAccessException,
            ExpressionTooLongException,
        ) as exc:
            print(f"Skipping program {program_name}: {exc}")
            continue

        identity = [
            s for s in program["schedules_list"]
            if not s.get("transformations_list", [])
        ]

        identity_median = None

        if len(identity) == 1:
            identity_median = float(
                np.median(identity[0]["execution_times"])
            )
            print(f"\nProgram: {program_name}")
            print(f"Identity median: {identity_median:.6g}")

        else:
            print(
                f"WARNING: {program_name} has "
                f"{len(identity)} identity schedules"
            )

        program_schedule_count = 0

        for raw in program["schedules_list"]:
            if program_schedule_count >= MAX_SCHEDULES_PER_PROGRAM:
                break

            if F.drop_schedule(raw):
                schedules_skipped += 1
                continue

            try:
                schedule = normalize_schedule(annotation, raw)

                comps, loops, expr = F.fill_schedule_representation(
                    annotation, schedule, template
                )

                assert comps.shape[0] == 1
                assert comps.shape[2] == (
                    33 + MAX_TAGS * MAX_NUM_TRANSFORMATIONS + 813
                )
                assert expr.shape[0] == 1
                assert expr.shape[2] == MAX_EXPR_LEN
                assert expr.shape[3] == 11
                assert loops.shape[0] == 1
                assert loops.shape[2] == 8

                original_target = (
                    program["initial_execution_time"]
                    / np.min(raw["execution_times"])
                )

                median_target = (
                    identity_median / np.median(raw["execution_times"])
                    if identity_median is not None
                    else float("nan")
                )

                if not np.isfinite(original_target) or original_target <= 0:
                    raise ValueError("Invalid original target")

                if identity_median is not None and (
                    not np.isfinite(median_target) or median_target <= 0
                ):
                    raise ValueError("Invalid median-based target")

                model_input = _schedule_input(
                    annotation, schedule, template
                )

                with torch.no_grad():
                    prediction = cpu_model(model_input)

                assert prediction.numel() == 1
                assert torch.isfinite(prediction).all()

                schedules_checked += 1
                program_schedule_count += 1

                if mps_input is None:
                    mps_input = model_input

                if schedules_checked == 1:
                    print("\nFeature shapes:")
                    print("  Computations:", tuple(comps.shape))
                    print("  Loops:", tuple(loops.shape))
                    print("  Expressions:", tuple(expr.shape))
                    print("  CPU prediction:", prediction.item())

                    print("\nTarget comparison:")
                    print(f"  Original target: {original_target:.6g}")
                    print(f"  Median-based target: {median_target:.6g}")

            except (
                NbTransformationException,
                ScheduleTooComplexException,
                ValueError,
            ) as exc:
                schedules_skipped += 1
                print(f"Skipped schedule: {exc}")

        programs_checked += 1

        if programs_checked >= MAX_PROGRAMS:
            break

    print("\n========== CPU REAL DATA SMOKE TEST ==========")
    print("Programs checked:", programs_checked)
    print("Schedules successfully checked:", schedules_checked)
    print("Schedules skipped:", schedules_skipped)

    if programs_checked == 0 or schedules_checked == 0:
        raise RuntimeError("No usable real-data datapoints found")

    print("CPU REAL DATA SMOKE TEST PASSED")

    if mps_input is not None:
        test_mps(mps_input)


if __name__ == "__main__":
    main()
