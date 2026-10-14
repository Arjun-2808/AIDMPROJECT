### CIFAR-10 test set (10,000 images), mean/std over seeds

| name | model | img_size | imbalance | method | optimizer | lr | epochs | seeds | test_acc | test_acc_std | macro_f1 | minority_recall | minority_recall_std | majority_recall | minutes_per_run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| best | resnet18 | 160 | none | ce | nadam | 8.18e-05 | 5 | 3 | 95.5 | 0.06 | 95.5 |  |  |  | 4.1 |
| best | resnet18 | 160 | step | ce | nadam | 8.18e-05 | 5 | 3 | 83.72 | 0.55 | 82.04 | 50.8 | 1.65 | 97.82 | 3.1 |
| best | resnet18 | 160 | step | focal | nadam | 8.18e-05 | 5 | 3 | 83.95 | 0.43 | 82.41 | 52.11 | 1.37 | 97.59 | 3.1 |
| best | resnet18 | 160 | step | logit_adjust | nadam | 8.18e-05 | 5 | 3 | 92.89 | 0.08 | 92.84 | 86.04 | 0.37 | 95.82 | 3.0 |
| best | resnet18 | 160 | step | oversample | nadam | 8.18e-05 | 5 | 3 | 85.7 | 0.65 | 84.43 | 58.44 | 2.08 | 97.38 | 3.1 |
| best | resnet18 | 160 | step | wce | nadam | 8.18e-05 | 5 | 3 | 88.86 | 0.42 | 88.46 | 72.16 | 1.3 | 96.01 | 3.1 |
| worst | resnet18 | 160 | none | ce | sgd_nesterov | 1.66e-05 | 5 | 3 | 72.51 | 0.15 | 72.26 |  |  |  | 2.8 |
