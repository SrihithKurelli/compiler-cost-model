import torch
from torch import nn

from looper_cost_model.data.dataset import PreparedBatches
from looper_cost_model.train import train


class ConstantPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(1.0))

    def forward(self, inputs):
        return self.value.expand(inputs[1].shape[0])


def _prepared(labels, names):
    labels = torch.tensor(labels, dtype=torch.float32)
    inputs = (None, torch.zeros((len(labels), 1)))
    attributes = [
        (name, "schedule", "schedule", 1.0, 0.0, "node", "footprint")
        for name in names
    ]
    return PreparedBatches(
        X=[inputs],
        Y=[labels],
        attributes=[attributes],
        nb_datapoints=len(labels),
    )


def test_lstm_validation_checkpoint_metric_is_program_weighted(tmp_path):
    # Program A has one exact prediction; program B has three 90%-APE predictions.
    # The per-program mean is 45%, while the schedule-weighted mean is 67.5%.
    batches = {
        "train": _prepared([1.0, 1.0, 1.0, 1.0], ["train"] * 4),
        "val": _prepared(
            [1.0, 10.0, 10.0, 10.0],
            ["program_a", "program_b", "program_b", "program_b"],
        ),
    }
    history, state = train(
        ConstantPredictor(),
        batches,
        num_epochs=1,
        lr=1e-8,
        weight_decay=0.0,
        device="cpu",
        batch_size=4,
        checkpoint_path=str(tmp_path / "best.pt"),
    )

    assert state is not None
    assert abs(history[0][2] - 45.0) < 0.01
    assert (tmp_path / "best.pt").exists()
