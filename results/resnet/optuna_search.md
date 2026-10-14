# ResNet-18 Optuna hyper-parameter search: Dogs vs. Cats

Exported from `src/resnet/outputs/optuna/dogs_cats.db` (study `dogs_cats_resnet`, direction `maximize`, objective = best validation accuracy over the run).

## Setup

- Model: ResNet-18, ImageNet-pretrained, 160 px input, 2 classes
- Budget per trial: up to 15 epochs, early stopping patience 4, cosine LR schedule, AMP
- Sampler: TPE (seed 2026); pruner: MedianPruner (n_startup_trials=8, n_warmup_steps=3)
- Per-trial seed: 2026 + trial number
- Trials: 65 total, 37 complete, 1 fail, 25 pruned, 2 running
- Run window: 2026-09-27 03:29 to 2026-09-29 04:11

RUNNING trials were interrupted and never finished; their partial curves are listed for completeness.

## Search space

| parameter | distribution | range / choices |
|---|---|---|
| optimizer | categorical | adamw, nadam, sgd_nesterov |
| learning_rate | log-uniform | 1e-05 to 0.003 |
| weight_decay | log-uniform | 1e-06 to 0.01 |
| batch_size | categorical | 16, 32, 64 |
| dropout | uniform | 0.0 to 0.5 |
| fine_tuning | categorical | classifier_only, layer4_plus_classifier, full |

## Best trial

Trial 20: validation accuracy **98.86%** (tied with trial 35).

| parameter | value |
|---|---|
| optimizer | nadam |
| learning_rate | 8.181391538335567e-05 |
| weight_decay | 1.8448923279285054e-06 |
| batch_size | 64 |
| dropout | 0.31103473606481874 |
| fine_tuning | full |

Per-epoch validation accuracy (%) of the tied best trials:

| trial | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 98.10 | 98.22 | 98.26 | 98.56 | 98.34 | 98.36 | 98.60 | 98.66 | 98.54 | 98.74 | 98.86 | 98.80 | 98.78 | 98.72 | 98.82 |
| 35 | 98.10 | 97.88 | 98.34 | 98.62 | 98.52 | 98.62 | 98.78 | 98.82 | 98.70 | 98.76 | 98.86 | 98.74 | 98.80 | 98.84 | 98.80 |

Trial 35 parameters: optimizer=nadam, learning_rate=5.78e-05, weight_decay=4.91e-05, batch_size=64, dropout=0.430, fine_tuning=full.

## Top 10 completed trials

| rank | trial | val_acc_% | epochs | minutes | optimizer | learning_rate | weight_decay | batch_size | dropout | fine_tuning |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 20 | 98.86 | 15 | 68 | nadam | 8.18e-05 | 1.84e-06 | 64 | 0.311 | full |
| 2 | 35 | 98.86 | 15 | 73 | nadam | 5.78e-05 | 4.91e-05 | 64 | 0.430 | full |
| 3 | 15 | 98.84 | 15 | 73 | nadam | 2.31e-05 | 5.51e-06 | 32 | 0.115 | full |
| 4 | 23 | 98.84 | 15 | 69 | nadam | 8.11e-05 | 2.17e-06 | 64 | 0.333 | full |
| 5 | 34 | 98.82 | 12 | 55 | nadam | 3.94e-05 | 1.40e-06 | 64 | 0.419 | full |
| 6 | 42 | 98.82 | 15 | 72 | nadam | 6.00e-05 | 1.15e-05 | 64 | 0.251 | full |
| 7 | 37 | 98.78 | 15 | 78 | nadam | 1.24e-05 | 2.74e-06 | 32 | 0.190 | full |
| 8 | 51 | 98.78 | 15 | 68 | nadam | 7.17e-05 | 2.04e-05 | 64 | 0.468 | full |
| 9 | 58 | 98.78 | 12 | 54 | nadam | 4.50e-05 | 3.37e-06 | 64 | 0.295 | full |
| 10 | 33 | 98.76 | 10 | 46 | adamw | 4.59e-05 | 1.87e-06 | 64 | 0.313 | full |

## Best accuracy by categorical choice (completed trials)

| optimizer | n | best_% | mean_% |
|---|---|---|---|
| adamw | 8 | 98.76 | 98.36 |
| nadam | 27 | 98.86 | 98.69 |
| sgd_nesterov | 2 | 95.40 | 95.38 |

| fine_tuning | n | best_% | mean_% |
|---|---|---|---|
| classifier_only | 2 | 95.40 | 95.38 |
| layer4_plus_classifier | 1 | 98.22 | 98.22 |
| full | 34 | 98.86 | 98.63 |

| batch_size | n | best_% | mean_% |
|---|---|---|---|
| 16 | 3 | 98.14 | 96.30 |
| 32 | 11 | 98.84 | 98.60 |
| 64 | 23 | 98.86 | 98.65 |

## All trials

`val_acc_%` is the objective for completed trials; `best_seen_%` is the highest per-epoch validation accuracy reported before the trial stopped.

| trial | state | val_acc_% | best_seen_% | epochs | minutes | optimizer | learning_rate | weight_decay | batch_size | dropout | fine_tuning |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | COMPLETE | 95.36 | 95.36 | 15 | 138 | sgd_nesterov | 1.66e-05 | 8.26e-05 | 16 | 0.279 | classifier_only |
| 1 | RUNNING |  | 98.54 | 5 |  | nadam | 1.27e-04 | 1.77e-04 | 64 | 0.015 | full |
| 2 | RUNNING |  |  | 0 |  | sgd_nesterov | 1.66e-05 | 8.26e-05 | 16 | 0.279 | classifier_only |
| 3 | COMPLETE | 95.40 | 95.40 | 13 | 54 | sgd_nesterov | 1.66e-05 | 8.26e-05 | 16 | 0.279 | classifier_only |
| 4 | COMPLETE | 98.62 | 98.62 | 8 | 35 | nadam | 1.27e-04 | 1.77e-04 | 64 | 0.015 | full |
| 5 | COMPLETE | 98.68 | 98.68 | 15 | 69 | nadam | 1.31e-05 | 7.62e-05 | 32 | 0.260 | full |
| 6 | COMPLETE | 98.14 | 98.14 | 8 | 40 | adamw | 1.31e-04 | 5.75e-03 | 16 | 0.105 | full |
| 7 | COMPLETE | 97.64 | 97.64 | 15 | 64 | adamw | 2.61e-03 | 1.02e-06 | 64 | 0.498 | full |
| 8 | COMPLETE | 98.22 | 98.22 | 15 | 53 | adamw | 1.74e-05 | 9.76e-03 | 64 | 0.084 | layer4_plus_classifier |
| 9 | COMPLETE | 98.22 | 98.22 | 15 | 69 | adamw | 5.61e-04 | 2.68e-06 | 32 | 0.140 | full |
| 10 | PRUNED | 96.68 | 96.68 | 4 | 17 | adamw | 1.08e-03 | 2.57e-03 | 64 | 0.358 | full |
| 11 | COMPLETE | 98.66 | 98.66 | 13 | 60 | adamw | 3.18e-05 | 5.35e-03 | 32 | 0.162 | full |
| 12 | COMPLETE | 98.40 | 98.40 | 5 | 22 | nadam | 3.14e-05 | 2.04e-03 | 32 | 0.460 | full |
| 13 | COMPLETE | 98.60 | 98.60 | 10 | 46 | nadam | 1.01e-05 | 1.90e-06 | 32 | 0.263 | full |
| 14 | COMPLETE | 98.56 | 98.56 | 9 | 44 | adamw | 5.09e-05 | 3.23e-03 | 32 | 0.251 | full |
| 15 | COMPLETE | 98.84 | 98.84 | 15 | 73 | nadam | 2.31e-05 | 5.51e-06 | 32 | 0.115 | full |
| 16 | COMPLETE | 98.56 | 98.56 | 6 | 30 | nadam | 3.53e-05 | 5.20e-05 | 32 | 0.175 | full |
| 17 | COMPLETE | 98.62 | 98.62 | 15 | 74 | nadam | 1.43e-05 | 4.64e-06 | 32 | 0.002 | full |
| 18 | COMPLETE | 98.56 | 98.56 | 13 | 59 | nadam | 1.32e-05 | 2.47e-05 | 64 | 0.157 | full |
| 19 | PRUNED | 97.88 | 97.94 | 4 | 21 | nadam | 1.07e-04 | 1.60e-06 | 16 | 0.148 | full |
| 20 | COMPLETE | 98.86 | 98.86 | 15 | 68 | nadam | 8.18e-05 | 1.84e-06 | 64 | 0.311 | full |
| 21 | COMPLETE | 98.74 | 98.74 | 10 | 46 | nadam | 3.17e-05 | 1.59e-06 | 64 | 0.248 | full |
| 22 | COMPLETE | 98.50 | 98.50 | 8 | 38 | nadam | 1.60e-05 | 1.31e-05 | 64 | 0.265 | full |
| 23 | COMPLETE | 98.84 | 98.84 | 15 | 69 | nadam | 8.11e-05 | 2.17e-06 | 64 | 0.333 | full |
| 24 | PRUNED | 97.96 | 97.96 | 4 | 18 | nadam | 2.06e-04 | 1.43e-06 | 64 | 0.332 | full |
| 25 | PRUNED | 97.90 | 97.90 | 4 | 15 | nadam | 1.01e-04 | 9.90e-06 | 64 | 0.332 | layer4_plus_classifier |
| 26 | COMPLETE | 98.64 | 98.64 | 14 | 64 | nadam | 2.38e-05 | 4.15e-06 | 64 | 0.456 | full |
| 27 | PRUNED | 98.12 | 98.12 | 4 | 18 | sgd_nesterov | 3.09e-04 | 6.25e-06 | 64 | 0.247 | full |
| 28 | COMPLETE | 98.64 | 98.64 | 9 | 44 | adamw | 2.65e-05 | 2.44e-06 | 32 | 0.097 | full |
| 29 | COMPLETE | 98.74 | 98.74 | 11 | 51 | nadam | 3.53e-05 | 3.54e-06 | 64 | 0.044 | full |
| 30 | PRUNED | 97.46 | 97.46 | 4 | 20 | sgd_nesterov | 4.87e-05 | 4.31e-06 | 32 | 0.238 | full |
| 31 | PRUNED | 98.38 | 98.38 | 4 | 18 | nadam | 8.05e-05 | 6.12e-06 | 64 | 0.390 | full |
| 32 | PRUNED | 97.26 | 97.34 | 4 | 21 | nadam | 1.55e-04 | 1.76e-05 | 16 | 0.189 | full |
| 33 | COMPLETE | 98.76 | 98.76 | 10 | 46 | adamw | 4.59e-05 | 1.87e-06 | 64 | 0.313 | full |
| 34 | COMPLETE | 98.82 | 98.82 | 12 | 55 | nadam | 3.94e-05 | 1.40e-06 | 64 | 0.419 | full |
| 35 | COMPLETE | 98.86 | 98.86 | 15 | 73 | nadam | 5.78e-05 | 4.91e-05 | 64 | 0.430 | full |
| 36 | PRUNED | 98.38 | 98.38 | 4 | 20 | nadam | 1.18e-04 | 1.45e-06 | 64 | 0.419 | full |
| 37 | COMPLETE | 98.78 | 98.78 | 15 | 78 | nadam | 1.24e-05 | 2.74e-06 | 32 | 0.190 | full |
| 38 | PRUNED | 98.44 | 98.44 | 4 | 21 | nadam | 7.15e-05 | 8.02e-06 | 32 | 0.065 | full |
| 39 | PRUNED | 98.40 | 98.40 | 4 | 19 | nadam | 8.30e-05 | 1.83e-06 | 64 | 0.240 | full |
| 40 | PRUNED | 97.84 | 97.84 | 4 | 19 | nadam | 2.75e-04 | 5.66e-06 | 64 | 0.423 | full |
| 41 | COMPLETE | 98.74 | 98.74 | 9 | 43 | nadam | 3.80e-05 | 1.79e-05 | 64 | 0.322 | full |
| 42 | COMPLETE | 98.82 | 98.82 | 15 | 72 | nadam | 6.00e-05 | 1.15e-05 | 64 | 0.251 | full |
| 43 | PRUNED | 98.66 | 98.66 | 14 | 64 | nadam | 4.55e-05 | 1.03e-05 | 64 | 0.463 | full |
| 44 | PRUNED | 98.16 | 98.24 | 4 | 19 | nadam | 6.49e-05 | 2.13e-06 | 32 | 0.389 | full |
| 45 | PRUNED | 98.38 | 98.42 | 4 | 18 | nadam | 3.90e-05 | 1.17e-06 | 64 | 0.474 | full |
| 46 | PRUNED | 98.24 | 98.24 | 4 | 18 | nadam | 1.85e-04 | 2.60e-06 | 64 | 0.269 | full |
| 47 | PRUNED | 97.34 | 97.34 | 4 | 18 | sgd_nesterov | 9.12e-05 | 2.61e-06 | 64 | 0.370 | full |
| 48 | COMPLETE | 98.56 | 98.56 | 8 | 36 | nadam | 7.59e-05 | 1.28e-06 | 64 | 0.175 | full |
| 49 | COMPLETE | 98.68 | 98.68 | 9 | 41 | nadam | 7.34e-05 | 2.53e-05 | 64 | 0.341 | full |
| 50 | COMPLETE | 98.70 | 98.70 | 11 | 50 | nadam | 3.42e-05 | 1.02e-06 | 64 | 0.178 | full |
| 51 | COMPLETE | 98.78 | 98.78 | 15 | 68 | nadam | 7.17e-05 | 2.04e-05 | 64 | 0.468 | full |
| 52 | COMPLETE | 98.70 | 98.70 | 12 | 54 | nadam | 2.18e-05 | 1.30e-06 | 64 | 0.383 | full |
| 53 | COMPLETE | 98.68 | 98.68 | 8 | 37 | nadam | 3.89e-05 | 3.63e-05 | 64 | 0.369 | full |
| 54 | PRUNED | 98.20 | 98.28 | 4 | 20 | nadam | 6.01e-05 | 2.71e-06 | 16 | 0.344 | full |
| 55 | COMPLETE | 98.60 | 98.60 | 8 | 36 | nadam | 4.29e-05 | 1.97e-05 | 64 | 0.424 | full |
| 56 | PRUNED | 98.40 | 98.40 | 4 | 18 | nadam | 9.34e-05 | 4.96e-05 | 64 | 0.423 | full |
| 57 | PRUNED | 98.38 | 98.44 | 4 | 18 | nadam | 2.94e-05 | 1.14e-06 | 64 | 0.312 | full |
| 58 | COMPLETE | 98.78 | 98.78 | 12 | 54 | nadam | 4.50e-05 | 3.37e-06 | 64 | 0.295 | full |
| 59 | PRUNED | 98.08 | 98.08 | 4 | 18 | nadam | 2.07e-04 | 2.02e-06 | 64 | 0.408 | full |
| 60 | PRUNED | 98.18 | 98.18 | 4 | 18 | nadam | 1.48e-04 | 1.81e-05 | 64 | 0.420 | full |
| 61 | PRUNED | 98.32 | 98.48 | 4 | 18 | nadam | 8.80e-05 | 2.56e-06 | 64 | 0.292 | full |
| 62 | PRUNED | 98.44 | 98.44 | 4 | 18 | nadam | 5.10e-05 | 4.58e-05 | 64 | 0.436 | full |
| 63 | PRUNED | 98.36 | 98.42 | 4 | 19 | nadam | 1.17e-05 | 3.42e-06 | 32 | 0.203 | full |
| 64 | FAIL |  | 98.68 | 8 | 39 | nadam | 1.40e-05 | 8.08e-06 | 32 | 0.220 | full |
