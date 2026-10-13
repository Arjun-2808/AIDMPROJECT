# YOLO track: evaluation report

Ultralytics YOLO classification models, ImageNet-pretrained, trained on the NTU EEE GPU cluster (NVIDIA RTX 6000 Ada). All numbers come from the runs' `result.json` and `*_predictions.csv` files. Every run uses one seed (0), so differences of about 0.5 points or less are within noise. p-values are paired McNemar tests on the same images.

Common recipe: AdamW, lr0 1e-3 (cosine to lrf 0.01), batch 32, strong on-the-fly augmentation (RandomResizedCrop 0.5-1, flip, RandAugment, RandomErasing 0.4). Fine-tuning means all layers trainable, starting from the ImageNet weights (never from another task's checkpoint).

---

## 1. Dogs vs Cats

Setup: 20,000 train images (10k per class), 5,000 validation (2,500 per class), 224 px, 15 epochs. The 500-image test set has no labels, so accuracy is reported on the validation split. Test predictions are produced with `predict.py` for `submission.csv`.

### 1.1 Model comparison (validation, 5,000 images)

| Run | Model | Val accuracy | 95% CI (±) | Errors | Cat recall | Dog recall | Best epoch | Train time |
|---|---|---|---|---|---|---|---|---|
| `cd_l_ft` | yolo26l | **0.9906** | 0.0027 | 47 | 0.9900 | 0.9912 | 13 | 450 s |
| `cd_x_ft` | yolo26x | 0.9894 | 0.0028 | 53 | 0.9888 | 0.9900 | 13 | 451 s |
| `cd_s_ft` | yolo26s | 0.9898 | 0.0028 | 51 | 0.9904 | 0.9892 | 13 | 401 s |
| `cd_y11s_ft` | yolo11s | 0.9882 | 0.0030 | 59 | 0.9860 | 0.9904 | 14 | 400 s |
| `cd_m_ft` | yolo26m | 0.9884 | 0.0030 | 58 | 0.9864 | 0.9904 | 14 | 403 s |
| `cd_n_ft` | yolo26n | 0.9876 | 0.0031 | 62 | 0.9904 | 0.9848 | 14 | 414 s |

Macro-F1 equals accuracy to four decimals in every run (the classes are balanced and the models treat both about equally).

### 1.2 Is the best model really best?

McNemar test of `cd_l_ft` against each other model:

| Compared with | p |
|---|---|
| `cd_n_ft` | 0.04 |
| `cd_y11s_ft` | 0.06 |
| `cd_m_ft` | 0.13 |
| `cd_x_ft` | 0.44 |
| `cd_s_ft` | 0.66 |

`cd_l_ft` is nominally best, but it is only clearly separable from the nano model. Against s, m, x and YOLO11s the gap is within noise. The 95% intervals all overlap (each is about ±0.3 points). Model size above "small" buys very little here, and `cd_s_ft` is the cheaper equal.

### 1.3 Other observations

- **Ensembling:** averaging the probabilities of `cd_l_ft` and `cd_s_ft` gives 99.18% validation accuracy, about 0.1 points above the best single model.
- **Hard images:** 15 validation images are misclassified by all six models. These are most likely label noise or genuinely ambiguous pictures, and they set a floor on what any of these models can reach.
- **Cost:** all six models trained in 400-450 s on one RTX 6000 Ada, so training time barely separates them.

### 1.4 Recommendation

Submit `cd_l_ft` (best on validation) or `cd_s_ft` (statistically tied, smaller). Extra seeds (`--seeds-of cd_l_ft`) are needed to report mean ± std for question (b) and to settle the ranking.

---

## 2. CIFAR-10

Setup: 45,000 train / 5,000 validation (carved from the official train set) / 10,000 official test images, 20 epochs at 128 px, except where noted. Test numbers below are from the official test split (labelled, 1,000 per class).

### 2.1 Balanced training set (question g)

| Run | Model / setting | Val acc | Test acc | Test macro-F1 | Cat recall (test) | Train time |
|---|---|---|---|---|---|---|
| `c10_m_ft` | yolo26m, fine-tune | 0.9662 | **0.9624** | 0.9624 | 0.904 | 560 s |
| `c10_s_ft` | yolo26s, fine-tune | 0.9600 | 0.9553 | 0.9552 | 0.877 | 497 s |
| `c10_n_ft` | yolo26n, fine-tune | 0.9454 | 0.9409 | 0.9408 | 0.854 | 497 s |
| `c10_s_ft_sz64` | yolo26s, 64 px | 0.9346 | 0.9283 | 0.9281 | 0.822 | 429 s |
| `c10_s_scratch` | yolo26s, from scratch, 60 epochs | 0.9160 | 0.9098 | 0.9095 | 0.795 | 1505 s |

McNemar tests on the test split:

| Comparison | Only first right | Only second right | p |
|---|---|---|---|
| m vs s | 200 | 129 | 0.0001 |
| s vs n | 296 | 152 | 9e-12 |
| s: 128 px vs 64 px | 421 | 151 | 2e-30 |
| s: fine-tune vs scratch | 595 | 140 | 1e-67 |

Findings:
- Accuracy rises with model size (n < s < m), and each step is statistically significant.
- Pretraining matters a lot: fine-tuning beats training from scratch by 4.5 points, even though the scratch run trained three times as long.
- Resolution matters: 64 px costs 2.7 points against 128 px. CIFAR images are natively 32 px, and upscaling still helps the pretrained features.
- Cat and dog are the weakest classes in every run, and cat recall is the lowest. For `c10_s_ft` on the test split, 70 of the misclassified cats are called dogs.

Per-class test recall for `c10_s_ft`:

| airplane | automobile | bird | cat | deer | dog | frog | horse | ship | truck |
|---|---|---|---|---|---|---|---|---|---|
| 0.959 | 0.983 | 0.957 | 0.877 | 0.964 | 0.929 | 0.972 | 0.972 | 0.976 | 0.964 |

### 2.2 Imbalanced training set (question h)

The training set keeps only 5% of the cat, deer and truck images (about 225 each, against 4,500 for the other seven classes). The choice of classes and of 5% is ours, not from the brief. Validation and test stay balanced. All runs use yolo26s, fine-tuned.

| Run | Treatment | Test acc | Test macro-F1 | Cat | Deer | Truck | Minority mean | Other-7 mean |
|---|---|---|---|---|---|---|---|---|
| `c10_s_ft` | none (balanced reference) | 0.9553 | 0.9552 | 0.877 | 0.964 | 0.964 | 0.935 | 0.964 |
| `c10imb_s_ce` | plain cross-entropy (baseline) | 0.8739 | 0.8653 | 0.387 | 0.732 | 0.817 | 0.645 | 0.972 |
| `c10imb_s_focal` | focal loss | 0.8785 | 0.8727 | 0.463 | 0.719 | 0.843 | 0.675 | 0.966 |
| `c10imb_s_oversample` | minority images repeated to 4,500 | 0.8898 | 0.8863 | 0.545 | 0.796 | 0.875 | 0.739 | 0.955 |
| `c10imb_s_wce-inv` | class weights 1/n | 0.8871 | 0.8841 | 0.580 | 0.803 | 0.865 | 0.749 | 0.946 |
| `c10imb_s_wce-eff` | effective-number class weights | **0.8970** | **0.8937** | 0.551 | 0.801 | 0.854 | 0.735 | 0.966 |

Validation accuracy for reference: ce 0.8780, focal 0.8796, oversample 0.8946, wce-inv 0.8896, wce-eff 0.9008.

McNemar tests on the test split:

| Comparison | Only first right | Only second right | p |
|---|---|---|---|
| wce-eff vs ce | 403 | 172 | 2e-22 |
| oversample vs ce | 468 | 309 | 1e-8 |
| wce-inv vs ce | 449 | 317 | 2e-6 |
| focal vs ce | 282 | 236 | 0.048 |
| wce-eff vs oversample | 392 | 320 | 0.008 |
| wce-eff vs wce-inv | 353 | 254 | 7e-5 |

Findings:
- **Cost of the imbalance:** accuracy falls 8 points and the minority recall falls from 0.935 to 0.645. The other seven classes are unaffected (0.972), which is why accuracy understates the damage. Cat is hit hardest: recall is 0.387, and 431 test cats are called dogs.
- **Fixes:** every fix beats plain cross-entropy. `wce-eff` is best on accuracy and macro-F1 and keeps the majority classes at 0.966. `wce-inv` and `oversample` push minority recall slightly higher but lose more on the majority classes. Focal loss helps least (p = 0.048).
- **No fix closes the gap:** the best minority recall is about 0.75 against 0.935 for balanced. About 225 real images per class carry limited information.
- **Oversampling:** its best epoch was 9 of 20, consistent with the model memorising the repeated images. The strong augmentation reduces this but does not remove it.

---

## 3. Caveats

- One seed per configuration. Mean ± std over several seeds is still to be added for question (b), and would settle the close calls (for example `cd_l_ft` vs `cd_s_ft`, and `wce-eff` vs `wce-inv`).
- The cats-vs-dogs test set is unlabelled, so no test accuracy exists for it. The validation numbers were also used to pick `best.pt` within each run, so they are slightly optimistic. The CIFAR-10 test numbers were not used for any selection.
- The imbalance setup (which classes, what fraction) is our choice. Changing `--minority` and `--keep` in `prep_cifar10.py` gives a different severity.
- Run times come from one cluster node and will vary with load.

## 4. Reproducing

Numbers come from `~/yolo_project/runs/<name>/result.json` and `{val,test}_predictions.csv`. Re-score a run locally with:

```bash
python evaluate.py --run c10_s_ft --split test
```

Cats vs dogs submission:

```bash
python predict.py --run cd_l_ft
```
