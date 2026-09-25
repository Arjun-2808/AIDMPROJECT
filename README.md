# AIDM_PROJECT — IE4483 Mini Project (Option 2): Dogs vs. Cats

Shared repo for a three-person group. Everyone trains a different model family
with a different optimisation strategy, on the same data pipeline, and logs to
the same place — so the comparison at the end is apples to apples.

Deadline: **13 Nov 2026**. Full rubric in [`docs/BRIEF.md`](docs/BRIEF.md).

## The split

| Track | Model family | Optimiser | Files | Owner |
|---|---|---|---|---|
| A | CNN — ResNet/VGG transfer + a from-scratch CNN | SGD + momentum, step decay | `src/models/cnn.py`, `configs/cnn_*.yaml` | *TBD* |
| B | ViT — ViT/DeiT/Swin fine-tune | AdamW + cosine + warmup | `src/models/vit.py`, `configs/vit_*.yaml` | *TBD* |
| C | **not decided yet** | RMSprop + OneCycle (placeholder) | `src/models/baseline.py`, `configs/third_*.yaml` | *TBD* |

Track C options, all defensible against the brief: a modern CNN
(EfficientNet/ConvNeXt), a classical baseline (HOG or CNN features → SVM — the
brief explicitly allows conventional classifiers as comparison), or owning
CIFAR-10 + class imbalance (questions g & h, 20% of the marks).

## Setup

```bash
git clone https://github.com/Arjun-2808/AIDM_PROJECT.git
cd AIDM_PROJECT

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Get the dataset (1.1 GB, **not in the repo** — see [`data/README.md`](data/README.md)):

```bash
python scripts/setup_data.py --zip /path/to/datasets.zip
```

Check your environment works — this runs on CPU in about a minute:

```bash
python train.py --config configs/smoke.yaml --owner <you> --note smoke
```

## Running

```bash
# a real run
python train.py --config configs/vit_s16_adamw.yaml --owner member2

# override anything in the YAML, without editing it
python train.py --config configs/cnn_resnet18_sgd.yaml --owner arjun \
  --set optim.lr=0.003 data.train_per_class=2000 --note "lr sweep 2/5"

python train.py --list-models
```

Then produce the submission:

```bash
python predict.py --run experiments/arjun/20261101-1432_cnn_resnet18_sgd
# -> results/submission.csv   (id, label with 1=dog, 0=cat)
```

## Logs

Every run writes `experiments/<owner>/<timestamp>_<run_name>/` containing
`config.json` (full config + git SHA + command line), `metrics.csv` (per epoch),
`summary.json`, and `train.log` — and appends a row to
[`results/leaderboard.md`](results/leaderboard.md), which is the one page you
check to see where everyone is.

Those files are text and small, so **commit them**. Checkpoints (`.pt`) are
gitignored — keep them local.

```bash
git pull --rebase
git add experiments/<you>/ results/leaderboard.md
git commit -m "run: vit_s16 adamw lr5e-5, val 0.98xx"
git push
```

## Layout

```
configs/        YAML experiment configs (base.yaml + per-track overrides)
data/           dataset goes here — gitignored
docs/           BRIEF.md (rubric), WORKFLOW.md (how we share the repo)
experiments/    one folder per person, committed logs
results/        leaderboard.md, submission.csv
scripts/        setup_data.py
src/
  data.py       dataloaders, augmentation recipes, cat=0/dog=1 assertion
  engine.py     train/eval loop, AMP, early stopping
  optim.py      optimiser + scheduler + loss factory (the axis we vary)
  tracking.py   RunLogger — writes the per-run logs and leaderboard row
  utils.py      seeding, device, run dirs
  models/       registry; add a model with @register("name")
train.py        entry point
predict.py      checkpoint -> submission.csv
```

**Before changing a shared file** (`src/data.py`, `src/engine.py`,
`src/optim.py`, `train.py`) say so in the group chat — changing augmentation or
seed defaults mid-project silently invalidates everyone's earlier numbers.
Conventions and branching: [`docs/WORKFLOW.md`](docs/WORKFLOW.md).
