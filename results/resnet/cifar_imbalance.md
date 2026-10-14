### Imbalanced CIFAR-10: test metrics (%) and per-class recall

| model | method | test_acc | macro_F1 | minority_recall | airplane | automobile | bird | cat | deer | dog | frog | horse | ship | truck |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ResNet-18 (best params) | Plain CE | 83.72 ± 0.55 | 82.04 ± 0.73 | 50.80 ± 1.65 | 97.3 | 99.4 | 95.9 | 31.5 | 50.1 | 97.7 | 98.9 | 97.8 | 97.9 | 70.8 |
| ResNet-18 (best params) | Weighted CE | 88.86 ± 0.42 | 88.46 ± 0.48 | 72.16 ± 1.30 | 95.8 | 97.9 | 92.6 | 56.1 | 75.5 | 95.0 | 97.7 | 96.1 | 97.1 | 84.9 |
| ResNet-18 (best params) | Oversampling | 85.70 ± 0.65 | 84.43 ± 0.81 | 58.44 ± 2.08 | 96.8 | 99.1 | 95.4 | 33.9 | 63.8 | 96.8 | 98.8 | 97.4 | 97.3 | 77.7 |
| ResNet-18 (best params) | Focal loss | 83.95 ± 0.43 | 82.41 ± 0.52 | 52.11 ± 1.37 | 96.6 | 99.4 | 95.9 | 33.6 | 52.1 | 97.3 | 98.8 | 97.3 | 97.8 | 70.7 |
| ResNet-18 (best params) | Logit adjustment | 92.89 ± 0.08 | 92.84 ± 0.08 | 86.04 ± 0.37 | 96.3 | 97.4 | 93.5 | 77.0 | 89.0 | 92.5 | 97.4 | 96.3 | 97.4 | 92.2 |
