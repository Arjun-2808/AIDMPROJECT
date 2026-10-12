# IE4483 Mini Project (Option 2): Dogs vs. Cats and CIFAR-10 with Vision Transformers

PyTorch + [timm](https://github.com/huggingface/pytorch-image-models). Main model: ViT-B/16
pretrained on ImageNet-21k (fine-tuned on ImageNet-1k), fine-tuned end to end. It is compared
against ViT-S/16, ResNet-50, a small CNN trained from scratch, a linear probe, and conventional
classifiers (logistic regression, RBF-SVM, kNN) on frozen features and on raw pixels.

## Results

All outputs (tables, figures, `submission.csv`, error analysis) are in [`results/vit/`](../../results/vit).
Mean ± std over 3 seeds.

| Dogs vs. Cats (validation, 5,000 images) | Accuracy |
|---|---|
| ViT-B/16 fine-tuned (main model, `submission.csv`) | **99.73 ± 0.03** |
| ViT-B/16 linear probe (frozen backbone) | 99.77 ± 0.01 |
| Logistic regression on frozen ViT-B/16 features | 99.70 |
| ViT-S/16 fine-tuned | 99.67 ± 0.05 |
| ResNet-50 fine-tuned | 99.31 ± 0.06 |
| Small CNN from scratch | 97.19 ± 0.13 |
| RBF-SVM on raw pixels | 66.08 |

| CIFAR-10 (test, 10,000 images) | Accuracy |
|---|---|
| ViT-B/16 fine-tuned | **98.93 ± 0.07** |
| ViT-S/16 fine-tuned | 98.58 ± 0.05 |
| ResNet-50 fine-tuned | 96.36 ± 0.12 |
| Small CNN from scratch (32×32) | 90.72 ± 0.26 |

| Imbalanced CIFAR-10 (cat, deer, truck at 2 %), test | ViT-B/16 acc / minority recall | Small CNN acc / minority recall |
|---|---|---|
| Plain cross-entropy | 94.66 / 83.14 | 73.81 / 26.36 |
| Class-weighted CE | 94.58 / 82.88 | 75.09 / 33.94 |
| Oversampling | 94.03 / 81.12 | 72.58 / 23.56 |
| Focal loss | 93.76 / 80.26 | 73.25 / 25.80 |
| Logit adjustment | **96.45 / 89.42** | **82.76 / 62.24** |

## Files

| File | Purpose | Report part |
|---|---|---|
| `train_dvc.py` | train/evaluate one Dogs vs. Cats model; writes `submission.csv` with `--save-ckpt` | b, c, d |
| `features_svm.py` | conventional classifiers on frozen features / raw pixels, hyper-parameters by 3-fold CV | b, f |
| `analyze_dvc.py` | test-set contact sheets, validation errors, attention rollout + class-relevance maps | e |
| `train_cifar.py` | CIFAR-10, balanced or with minority classes cut to 2 %, five imbalance methods | g, h |
| `sweep.py` | the full experiment grid (23 Dogs vs. Cats + 14 CIFAR configs, 3 seeds each) | c, f, g, h |
| `summarize.py` | all tables (`results/*.md`, `*.csv`) and figures (`results/fig_*.png`) | all |
| `data.py`, `models.py`, `engine.py` | datasets/augmentation, model zoo, training loop and losses | |
| `setup_nscc.sh`, `jobs/` | one-time NSCC setup and PBS job scripts | |

## Running on NSCC ASPIRE 2A

Everything lives in `~/scratch/AIDM` (code in `IE4483/`, conda env in `env/`, data, runs, caches);
home is only 50 GB. Scratch purges files idle for ~30 days, so copy `results/` back when done.

```bash
# laptop, from the repo root: upload code and data
scp -r src/vit NSCC:scratch/AIDM/IE4483
scp datasets.zip NSCC:scratch/AIDM/

# NSCC login node
cd ~/scratch/AIDM/IE4483
bash setup_nscc.sh            # one time: env, data, CIFAR-10, pretrained weights
bash jobs/submit.sh smoke     # ~10 min end-to-end check; log ends with "SMOKE TEST PASSED"
bash jobs/submit.sh all       # everything; analysis + summary run automatically at the end
bash jobs/status.sh           # progress: finished runs with accuracy, PBS queue

# laptop: fetch tables, figures, submission.csv
scp -r NSCC:scratch/AIDM/IE4483/results results/vit
```

Sweeps are resumable: finished seeds are skipped, so if a job dies just run `bash jobs/submit.sh all` again.
If `qsub` asks for a project, set `PROJECT=<id>` in `jobs/paths.sh`.

## Running a single experiment (any machine with a GPU)

```bash
pip install torch torchvision timm==1.0.30 scikit-learn pandas matplotlib
export IE4483_DATA=/path/to/datasets IE4483_CIFAR=/path/to/cifar10 IE4483_OUT=runs

python train_dvc.py --model vit_b --epochs 5 --lr 2e-5 --save-ckpt   # main model + submission.csv
python features_svm.py
python analyze_dvc.py
python train_cifar.py --model vit_b --epochs 5 --lr 2e-5
python train_cifar.py --model vit_b --imbalance step --method logit_adjust
python summarize.py
python sweep.py dvc --list                                          # every config in the grid
```

## Protocol

- **Data.** Dogs vs. Cats: training subset of up to 10,000 images per class (subsets are nested and
  fixed by `--data-seed 0`), full validation set (2,500 per class) for model selection, 500 unlabelled
  test images. CIFAR-10: official 50k training set split into 45k train / 5k balanced validation
  (stratified, fixed seed), official 10k test set for reporting.
- **Preprocessing.** Resize the short side to `size / crop_pct`, centre-crop to `size` (224), normalise
  with the backbone's pretraining statistics. CIFAR images are augmented at 32×32 and bilinearly
  upsampled to 224 on the GPU.
- **Augmentation.** `basic`: random rotation ±15°, random resized crop (scale 0.6-1), horizontal flip.
  `strong`: + TrivialAugmentWide + random erasing. CIFAR `basic`: random crop with 4-px padding + flip.
- **Training.** AdamW (weight decay 0.05, none on biases/norms/position embeddings), linear warm-up
  for 10 % of steps then cosine decay, batch 64, gradient clipping 1.0, bf16 mixed precision.
  Cross-entropy loss. Dogs vs. Cats keeps the epoch with the best validation accuracy; CIFAR uses the
  final epoch (no selection on test data).
- **Imbalance (part h).** Cat, deer and truck keep 2 % of their training images (90 vs. 4,500).
  Methods: plain CE, class-weighted CE (w_c = N / (C·n_c)), class-balanced oversampling, focal loss
  (γ = 2), and logit-adjusted loss (τ = 1, Menon et al. 2021). Reported: test accuracy, macro-F1,
  minority-class and majority-class recall.
- **Reproducibility.** Every config runs with seeds 0, 1, 2; tables report mean ± std. Seeds fix
  Python/NumPy/PyTorch RNGs, data order and augmentation; cuDNN runs in deterministic mode.

## References

- Dosovitskiy et al., *An Image is Worth 16x16 Words*, ICLR 2021 (ViT).
- Steiner et al., *How to train your ViT?*, TMLR 2022 (AugReg weights).
- He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016.
- Abnar & Zuidema, *Quantifying Attention Flow in Transformers*, ACL 2020 (attention rollout).
- Chefer, Gur & Wolf, *Generic Attention-model Explainability*, ICCV 2021 (class relevance).
- Lin et al., *Focal Loss for Dense Object Detection*, ICCV 2017.
- Menon et al., *Long-tail Learning via Logit Adjustment*, ICLR 2021.
- Müller & Hutter, *TrivialAugment*, ICCV 2021.
