# LOOPerCostModel

LOOPerCostModel is the learned cost model of the [LOOPer](https://ieeexplore.ieee.org/document/11282943)
polyhedral autoscheduler. Given a program and a candidate sequence of loop
transformations (a *schedule*), it predicts the **speedup** that schedule would achieve,
allowing the autoscheduler's search to rank transformations without compiling and running
each candidate.

The model is a tree-structured neural network that mirrors a program's loop nest. It reads
two kinds of features:

- **Program features** — the untransformed program's structure: the loop-nest tree, each
  computation's iteration domain (as an affine-constraint matrix), its memory-access
  matrices, and its arithmetic expression tree.
- **Schedule features** — the applied transformations: interchange, reversal, skewing,
  tiling, unrolling, parallelization, shifting, and fusion.

These are embedded bottom-up — sequences of affine transformations and expression
operations by LSTMs, then computations and loops by a recursive network that follows the
loop-nest hierarchy — and regressed to a single predicted speedup. The model is trained to
minimize mean absolute percentage error (MAPE) against measured speedups, defined as
`initial_execution_time / min(execution_times)`.

It trains directly on the [LOOPerSet](https://huggingface.co/datasets/Mascinissa/LOOPerSet)
dataset with no manual preprocessing.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # or: pip install -e .
```

Requires Python ≥ 3.9 and PyTorch ≥ 2.0. A GPU is optional — the code runs unchanged on
CPU or GPU (see [Configuration](#configuration)).

## Data

LOOPerCostModel reads LOOPerSet's gzipped JSON-Lines splits directly. Download the
`pact25` compact split (the one used to train the published model) from the
[dataset card](https://huggingface.co/datasets/Mascinissa/LOOPerSet):

```python
from huggingface_hub import hf_hub_download
for name in ["looperset_v2_pact_train_compact.jsonl.gz",
             "looperset_v2_pact_validation_compact.jsonl.gz"]:
    hf_hub_download(repo_id="Mascinissa/LOOPerSet", repo_type="dataset",
                    filename=f"data/pact25/{name}", revision="v2.1",
                    local_dir="LOOPerSet")
```

Point `data.train_file` / `data.valid_file` at any LOOPerSet split (e.g. the larger `full`
split, or the standard variants that retain raw source and IR).

## Usage

Every command reads `configs/default.yaml` and accepts `key=value` overrides on the command
line. All artifacts — the feature cache, batched tensors, checkpoints, and logs — are
written under `experiment.base_path`.

Prepare the batched dataset (feature extraction + batching; run once per split):

```bash
python -m looper_cost_model.prepare_data --config configs/default.yaml
```

Train (any split not yet prepared is prepared automatically):

```bash
python -m looper_cost_model.train --config configs/default.yaml training.device=cuda:0
```

Evaluate a checkpoint on the validation split:

```bash
python -m looper_cost_model.evaluate --config configs/default.yaml \
    eval.weights_path runs/pact25/weights/best_model_looperset_pact25_compact.pt
```

Predict schedule speedups for the programs in a LOOPerSet file:

```bash
python -m looper_cost_model.predict --config configs/default.yaml \
    eval.weights_path <checkpoint.pt> data.valid_file <programs.jsonl.gz>
```

### A quick run on CPU

A 60-program slice is bundled under `.testdata/` for a fast local check:

```bash
python -m looper_cost_model.train \
    experiment.base_path=./runs/mini \
    data.train_file=.testdata/mini.jsonl.gz \
    data.valid_file=.testdata/mini.jsonl.gz \
    data.batch_size=256 data.min_functions_per_footprint=0 \
    training.device=cpu training.max_epochs=5
```

## Configuration

Configuration is a set of typed dataclasses (see `looper_cost_model/config.py`) with
defaults in `configs/default.yaml`, layered as *defaults → `--config` file → CLI
overrides*. The most useful options:

| Key | Description |
| --- | --- |
| `experiment.base_path` | Output root for cache, batches, checkpoints, and logs |
| `data.train_file`, `data.valid_file` | LOOPerSet `*.jsonl.gz` splits |
| `data.batch_size` | Datapoints per batch |
| `data.nb_workers` | Parallel feature-extraction workers |
| `data.min_functions_per_footprint` | Drop rare loop-nest shapes with too few programs (`0` keeps all) |
| `training.device` | `cpu` or `cuda:N` — the only device switch |
| `training.lr`, `training.max_epochs` | Constant learning rate and epoch count |

**Device handling.** `training.device` is the single switch; there is no separate GPU code
path. Cached tensors are stored on CPU and, at load time, as many batches as fit within a
GPU-memory budget are placed on the device while the rest stream from host memory. Moving a
workflow from CPU to GPU requires no code change.

## Project layout

```
looper_cost_model/
  constants.py        Fixed input dimensions and drop-condition exceptions
  config.py           Typed configuration (YAML + CLI overrides)
  data/
    looperset.py      LOOPerSet schema adapter: streaming, tree building, schedule normalization
    features.py       Program + schedule -> fixed-size input tensors
    dataset.py        Parallel feature extraction, caching, and batching
  model.py            The recursive-LSTM cost model
  train.py            Training loop (MAPE objective, constant LR, checkpointing)
  evaluate.py         Per-program metrics (MAPE, Spearman, nDCG)
  predict.py          Schedule-speedup inference
  prepare_data.py     Offline dataset preparation
configs/default.yaml  Default configuration
tests/test_smoke.py   End-to-end smoke test on the bundled slice
```

## Expected results

Trained on the full `pact25` split (100 epochs on an A100 80 GB, ~7 hours), the model
reaches **27.8% MAPE**, **0.77 Spearman rank correlation**, and **0.96 nDCG** on the
held-out validation programs, matching the numbers reported in the LOOPer paper. Short
runs on small slices reproduce the trends — decreasing validation loss and nDCG well above
a constant baseline — but not the absolute figures.

## Transformation-sequence Transformer (research extension)

The repository also contains a separate Transformer cost model under
`looper_cost_model/transformer/`. It consumes the ordered
`transformations_list` sequence and program-level context, and predicts speedup
relative to the unoptimized program. The original recursive-LSTM implementation above
remains unchanged and is retained as the baseline.

The Transformer vocabulary covers all eight transformation types represented by the
LOOPerSet v2.1 schema: interchange, skewing, tiling, reversal, parallelization,
unrolling, fusion, and shifting. The target is
`max(initial_execution_time / execution_time, 0.01)`, matching the lower-bound
clipping convention used by the LSTM pipeline.

### Data expected by the Transformer

The Transformer loader expects the cleaned JSONL files produced by
`scripts/clean_looperset.py`: each schedule must have exactly one pooled
`execution_times` value, and each program must have a valid positive
`initial_execution_time`. Use the pinned LOOPerSet v2.1 PACT compact source
revision rather than the moving dataset `main` branch:

- [LOOPerSet v2.1 PACT data at the pinned revision](https://huggingface.co/datasets/Mascinissa/LOOPerSet/tree/d97fc54/data/pact25)
- Dataset license: CC-BY-4.0.

The Transformer training implementation currently materializes the cleaned split and
its schedule samples in memory. Start with the 100/1,000/5,000-program pilot subsets;
do not launch a full 73,000-program run until the memory-efficient data pipeline has
been completed and profiled.

### Train and evaluate the Transformer

Run these commands from the repository root. Adjust the data paths to wherever the
cleaned files are stored:

```bash
python -m looper_cost_model.transformer.train \
  --train ../data/processed/looperset_train_1000_cleaned.jsonl.gz \
  --valid ../data/processed/looperset_validation_300_cleaned.jsonl.gz \
  --output ./runs/transformer_pilot_1000 \
  --epochs 20 --batch-size 256 --seed 42 --device auto
```

Device selection supports `auto`, `cpu`, `mps`, `cuda`, and `cuda:N`.
`auto` prefers CUDA, then Apple MPS, then CPU. Explicitly requested but
unavailable accelerators raise an error instead of silently falling back.

Evaluate a checkpoint with:

```bash
python -m looper_cost_model.transformer.evaluate \
  --data ../data/processed/looperset_validation_300_cleaned.jsonl.gz \
  --checkpoint ./runs/transformer_pilot_1000/best_transformer.pt \
  --output ./runs/transformer_pilot_1000/analysis \
  --device auto
```

The evaluator reports schedule/program MAPE, Spearman correlation, nDCG,
nDCG@5, nDCG@1, top-1 optimal selection, and schedule-selection regret.

Train with `--no-positional-encoding` to run the order ablation:

```bash
python -m looper_cost_model.transformer.train \
  --train ../data/processed/looperset_train_common.jsonl.gz \
  --valid ../data/processed/looperset_validation_tune_common.jsonl.gz \
  --output ./runs/transformer_orderless \
  --epochs 10 --batch-size 256 --seed 42 --device cpu \
  --no-positional-encoding
```

Without positional encodings, the Transformer encoder is permutation-invariant over
transformation tokens: it retains the multiset of transformation types and their
numerical features, but cannot use their original sequence positions. Checkpoints store
this setting so the evaluator reconstructs the correct architecture automatically.

### Matched Transformer/LSTM research protocol

The GitHub Actions workflow
`.github/workflows/looperset-matched-comparison.yml` runs reproducible CPU comparisons
against the original recursive-LSTM baseline. It pins LOOPerSet v2.1 at revision
`d97fc54`, applies the same compatibility audit and common accepted schedules to both
models, and records the configuration and artifacts for each run.

The matched workflow uses three program-level partitions:

1. **Training programs** fit model parameters.
2. **Tuning programs** select the best checkpoint using program-weighted validation MAPE.
3. **Held-out test programs** are never used for checkpoint selection and are used only
   for final prediction, ranking, and schedule-selection metrics.

The candidate validation sample is split by program (not by schedule), so all schedules
belonging to one program stay in the same partition. The split is deterministic, and
training/validation candidate sampling uses a fixed sampling seed independently of the
model-training seed. Repeated training seeds therefore use the same sampled programs,
making seed-to-seed comparisons more interpretable.

Each matched run also trains an orderless Transformer on the same training and tuning
programs and evaluates it on the same held-out test programs. The paired analysis exports
per-program differences, 95% bootstrap confidence intervals, and Wilcoxon signed-rank
tests for prediction, ranking, and schedule-selection metrics. Interpret the pilot results
carefully: small candidate samples are feasibility evidence, not a substitute for a
larger held-out evaluation.

To trigger a run, push an `experiment-trigger.txt` file to a branch named
`experiment-runner/<name>` with exactly these four settings:

```text
TRAIN_PROGRAMS=1000
VALID_PROGRAMS=1000
EXPERIMENT_SEED=42
EXPERIMENT_EPOCHS=5
```

The validation count is the number of candidates before it is divided equally into tuning
and held-out test programs. Supported training/validation candidate counts and epoch
choices are defined in the workflow file. Each run uploads its metadata, audit reports,
checkpoints, predictions, paired statistics, and split manifest as a GitHub Actions
artifact.

### Tests

The isolated Transformer unit tests do not require LOOPerSet data:

```bash
python -m pytest -q
python -m compileall -q looper_cost_model scripts tests
```

GitHub Actions runs these checks on pushes to the development branches and on pull
requests to `main`.

## Citation

If you use this cost model, please cite the LOOPer paper and the LOOPerSet dataset:

```bibtex
@INPROCEEDINGS{looper,
  author={Merouani, Massinissa and Boudaoud, Afif and Aouadj, Iheb Nassim and Tchoulak, Nassim and Bernou, Islem Kara and Benyamina, Hamza and Tayeb, Fatima Benbouzid-Si and Benatchba, Karima and Leather, Hugh and Baghdadi, Riyadh},
  booktitle={2025 34th International Conference on Parallel Architectures and Compilation Techniques (PACT)},
  title={LOOPer: A Learned Automatic Code Optimizer For Polyhedral Compilers},
  year={2025}, pages={201-215}, doi={10.1109/PACT65351.2025.00028}}

@misc{looperset,
  title={LOOPerSet: A Large-Scale Dataset for Data-Driven Polyhedral Compiler Optimization},
  author={Massinissa Merouani and Afif Boudaoud and Riyadh Baghdadi},
  year={2025}, eprint={2510.10209}, archivePrefix={arXiv}, primaryClass={cs.PL},
  url={https://arxiv.org/abs/2510.10209}}
```

## License

Released under the MIT License (see `LICENSE`). The LOOPerSet dataset is distributed under
CC-BY-4.0.
