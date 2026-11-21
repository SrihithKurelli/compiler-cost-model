
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


def log(message):
    print(message, flush=True)


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
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")

    log("STEP 1: Finding real datapoint...")
    model_input = find_real_datapoint()
    log("STEP 1 PASSED")

    log("STEP 2: Initializing model on MPS...")
    device = torch.device("mps")
    model = LOOPerCostModel().to(device)
    model.train()
    log("STEP 2 PASSED")

    log("STEP 3: Moving input tensors to MPS...")
    tree = model_input[0]
    mps_input = (tree,) + tuple(
        tensor.to(device) for tensor in model_input[1:]
    )
    log("STEP 3 PASSED")

    log("STEP 4: Forward pass...")
    prediction = model(mps_input)
    torch.mps.synchronize()
    log(f"STEP 4 PASSED: prediction = {prediction.detach().cpu().item()}")

    log("STEP 5: Computing loss...")
    target = torch.ones_like(prediction)
    loss = torch.mean((prediction - target) ** 2)
    log(f"Loss: {loss.detach().cpu().item()}")
    log("STEP 5 PASSED")

    log("STEP 6: Backward propagation...")
    model.zero_grad(set_to_none=True)
    loss.backward()
    torch.mps.synchronize()
    log("STEP 6 PASSED")

    log("STEP 7: Checking gradients...")
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

    log(f"Finite gradient tensors: {gradients_checked}")
    log("STEP 7 PASSED")

    log("STEP 8: Testing optimizer update...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.00004,
        weight_decay=0.015,
    )
    optimizer.step()
    torch.mps.synchronize()
    log("STEP 8 PASSED")

    log("\nMPS FORWARD + BACKWARD + OPTIMIZER TEST PASSED")


if __name__ == "__main__":
    main()
