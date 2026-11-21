
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

import torch

from looper_cost_model.data import features as F
from looper_cost_model.data.looperset import (
    normalize_schedule,
    stream_jsonl_gz,
    ScheduleTooComplexException,
)
from looper_cost_model.constants import (
    LoopsDepthException,
    NbAccessException,
    ExpressionTooLongException,
    NbTransformationException,
)
from looper_cost_model.model import LOOPerCostModel
from looper_cost_model.predict import _schedule_input


DATA_FILE = os.path.expanduser(
    "~/foml_compiler_project/data/raw/data/pact25/"
    "looperset_v2_pact_train_compact.jsonl.gz"
)


def find_real_datapoint():
    for program in stream_jsonl_gz(DATA_FILE):
        if F.drop_program(program):
            continue

        try:
            template = F.build_representation_template(program)
        except (
            LoopsDepthException,
            NbAccessException,
            ExpressionTooLongException,
        ):
            continue

        for raw in program["schedules_list"]:
            if F.drop_schedule(raw):
                continue

            try:
                schedule = normalize_schedule(
                    program["program_annotation"], raw
                )
                return _schedule_input(
                    program["program_annotation"],
                    schedule,
                    template,
                )
            except (
                NbTransformationException,
                ScheduleTooComplexException,
            ):
                continue

    raise RuntimeError("Could not find a usable datapoint")


def main():
    print("STEP 1: Finding real datapoint...", flush=True)
    model_input = find_real_datapoint()
    print("STEP 1 PASSED", flush=True)

    print("STEP 2: Initializing model on CPU...", flush=True)
    model = LOOPerCostModel().to("cpu")
    model.train()
    print("STEP 2 PASSED", flush=True)

    print("STEP 3: Running forward pass...", flush=True)
    prediction = model(model_input)
    print("Prediction:", prediction.detach().item(), flush=True)
    print("STEP 3 PASSED", flush=True)

    print("STEP 4: Computing loss...", flush=True)
    target = torch.ones_like(prediction)
    loss = torch.mean((prediction - target) ** 2)
    print("Loss:", loss.item(), flush=True)
    print("STEP 4 PASSED", flush=True)

    print("STEP 5: Backward propagation...", flush=True)
    model.zero_grad(set_to_none=True)
    loss.backward()
    print("STEP 5 PASSED", flush=True)

    print("STEP 6: Checking gradients...", flush=True)
    gradients_checked = 0

    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all():
                raise RuntimeError(
                    f"Non-finite gradient in parameter: {name}"
                )
            gradients_checked += 1

    if gradients_checked == 0:
        raise RuntimeError("No gradients were generated")

    print("Finite gradient tensors:", gradients_checked, flush=True)
    print("STEP 6 PASSED", flush=True)

    print("STEP 7: Testing optimizer update...", flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.00004,
        weight_decay=0.015,
    )
    optimizer.step()
    print("STEP 7 PASSED", flush=True)

    print("\nCPU FORWARD + BACKWARD + OPTIMIZER TEST PASSED")


if __name__ == "__main__":
    main()
