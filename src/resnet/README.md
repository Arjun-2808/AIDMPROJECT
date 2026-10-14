# IE4483 ResNet experiments

This project implements the IE4483 Dogs-vs-Cats project.
The code is configuration-driven and keeps raw data and generated artifacts outside
the source package.

## Layout

```text
src/aidm_project/       reusable data, model, training, and evaluation code
scripts/                command-line entry points
configs/                reproducible experiment configurations
tests/                  lightweight unit tests
outputs/                generated checkpoints, logs, and submissions
data/                   datasets (not tracked; see note below)
```

Expected data locations are configured on the command line. Dogs-vs-Cats must have
`train/dog`, `train/cat`, `val/dog`, `val/cat`, and `test/` below the supplied
data root. Test images may be directly inside `test/` or inside class-like
subdirectories. For the included dataset, the data root is
`data\dogs_cats\datasets`.

A second data root, `data\dogs_cats_augmented`, holds the offline augmentation
experiment: its `train/` contains the 20,000 originals plus 20,000 generated
copies (40,000 files), while `val/` and `test/` are unchanged copies of the
originals. Pass it to any command with `--data-root` in place of the default
root. No script in this repository regenerates it, so treat it as source data.

## Setup

Use Python 3.11+ and install the dependencies in a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Training, evaluation, and prediction require a CUDA-enabled NVIDIA GPU.
The commands fail clearly if CUDA is unavailable; they do not fall back to CPU.
The project uses PyTorch `2.11.0+cu128` and torchvision `0.26.0+cu128`;
the CUDA 12.8 runtime bundled with these wheels works with the installed
NVIDIA driver.

## Experiments

Two baselines come from the 65-trial Optuna search (`..\..\results\resnet\optuna_search.md`,
raw study in `outputs\optuna\dogs_cats.db`; the tuning code itself has been removed):

| config | Optuna trial | val acc | optimizer | lr | weight decay | batch | dropout | fine-tuning |
|---|---|---|---|---|---|---|---|---|
| `*_best.yaml` | #20 | 98.86% | NAdam | 8.18e-5 | 1.84e-6 | 64 | 0.311 | full network |
| `*_worst.yaml` | #0 | 95.36% | SGD-Nesterov | 1.66e-5 | 8.26e-5 | 16 | 0.279 | classifier only |

Both use ResNet-18 (ImageNet weights) at 160 px, cosine LR and AMP. Dogs vs. Cats trains up to 15 epochs
with early stopping (patience 4) and keeps the best validation epoch. CIFAR-10 trains 5 epochs and
evaluates the final epoch on the official test set (45k train / 5k validation split of the training set;
images are augmented at 32 px and upsampled to 160 px on the GPU). The imbalanced CIFAR-10 study keeps
2% of cat, deer and truck and compares plain CE, class-weighted CE, oversampling, focal loss and logit
adjustment. Every run uses seeds 2026, 2027 and 2028.

Run the whole grid (27 runs; resumable, finished runs are skipped):

```powershell
.\.venv\Scripts\python.exe scripts\sweep.py all        # or: dvc | cifar | imbalance; add --list to preview
```

Turn finished runs into tables, figures, the submission and the error analysis, in the same format
as the ViT track (`..\..\results\vit`):

```powershell
.\.venv\Scripts\python.exe scripts\summarize.py   # -> ..\..\results\resnet\*.md|csv|png
.\.venv\Scripts\python.exe scripts\analyze.py     # -> ..\..\results\resnet\analysis\
```

Train a single run:

```powershell
.\.venv\Scripts\python.exe scripts\train.py --config "configs\dogs_cats_best.yaml" --seed 2027
.\.venv\Scripts\python.exe scripts\train.py --config "configs\cifar10_best.yaml" --imbalance step --method logit_adjust
```

Each run writes `outputs\runs\<dvc|cifar>\<run>\`: `history.csv`, `metrics.json` and, for Dogs vs. Cats,
`val_preds.csv`, `model.pt`, `test_probs.csv` and `submission.csv` (`id,label`, 1 = dog). The default
data roots are `data\dogs_cats\datasets` and `data\cifar10` (CIFAR-10 downloads on first use); pass
`--data-root` to change them, for example to `data\dogs_cats_augmented`. `--limit 200 --epochs 1`
gives a quick smoke test.

Run time on the RTX 4060 Laptop GPU: roughly 1 h per Dogs vs. Cats run with the best parameters, about
twice that with the worst, and about 1 h per CIFAR-10 run. cuDNN is on (`cudnn_safe: false`), which gave
the same accuracy as the Optuna setting and is about 25% faster; `num_workers` stays 0 because worker
processes crash the CUDA driver on this machine.

`predict.py` and `evaluate.py` still work on any checkpoint, for example
`outputs\runs\dvc\resnet18-best-s2026\model.pt`:

```powershell
.\.venv\Scripts\python.exe scripts\predict.py --config "configs\dogs_cats_best.yaml" `
  --data-root "data\dogs_cats\datasets" --checkpoint "outputs\runs\dvc\resnet18-best-s2026\model.pt" `
  --output "outputs\submissions\submission.csv"
.\.venv\Scripts\python.exe scripts\evaluate.py --config "configs\dogs_cats_best.yaml" `
  --data-root "data\dogs_cats\datasets" --checkpoint "outputs\runs\dvc\resnet18-best-s2026\model.pt" `
  --output "outputs\metrics\validation.json"
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests
```

The tests are CPU-only and need no dataset or GPU.

## Notes

The test set must not be used for hyperparameter selection. Final comparisons
should be repeated with seeds 2026, 2027, and 2028 and reported as mean ± standard
deviation. Datasets, checkpoints, databases, and generated CSV files are
excluded by `.gitignore` and are never treated as project sources.
