### Dogs vs. Cats: validation accuracy (%) per config, mean/std over seeds

| model | mode | n_per_class | aug | img_size | lr | epochs | seeds | val_acc_mean | val_acc_std | final_val_acc_mean | best_epoch | minutes_per_run | trainable_M |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vit_b | linprobe | 0 | basic | 224 | 0.001 | 5 | 3 | 99.77 | 0.01 | 99.75 | 2.7 | 1.3 | 0.0 |
| vit_b | finetune | 0 | basic | 224 | 3e-06 | 5 | 3 | 99.74 | 0.02 | 99.72 | 3.3 | 2.1 | 85.8 |
| vit_b | finetune | 0 | basic | 224 | 2e-05 | 5 | 3 | 99.73 | 0.03 | 99.73 | 4.3 | 2.2 | 85.8 |
| vit_b | finetune | 0 | basic | 224 | 1e-05 | 5 | 3 | 99.71 | 0.02 | 99.69 | 2.7 | 2.1 | 85.8 |
| vit_b | finetune | 1000 | strong | 224 | 2e-05 | 15 | 3 | 99.7 | 0.04 | 99.66 | 9.7 | 1.5 | 85.8 |
| vit_b | finetune | 5000 | basic | 224 | 2e-05 | 6 | 3 | 99.69 | 0.03 | 99.69 | 4.7 | 1.5 | 85.8 |
| vit_s | finetune | 0 | basic | 224 | 2e-05 | 5 | 3 | 99.67 | 0.05 | 99.65 | 4.0 | 1.4 | 21.67 |
| vit_b | finetune | 2500 | basic | 224 | 2e-05 | 8 | 3 | 99.67 | 0.02 | 99.65 | 5.7 | 1.2 | 85.8 |
| vit_b | finetune | 0 | basic | 160 | 2e-05 | 5 | 3 | 99.63 | 0.05 | 99.63 | 4.0 | 1.5 | 85.73 |
| vit_b | finetune | 1000 | basic | 224 | 2e-05 | 15 | 3 | 99.62 | 0.05 | 99.56 | 6.7 | 1.5 | 85.8 |
| vit_b | finetune | 1000 | none | 224 | 2e-05 | 15 | 3 | 99.6 | 0.02 | 99.57 | 5.3 | 1.5 | 85.8 |
| vit_b | finetune | 500 | basic | 224 | 2e-05 | 20 | 3 | 99.58 | 0.02 | 99.5 | 8.0 | 1.6 | 85.8 |
| vit_b | finetune | 0 | basic | 224 | 6e-05 | 5 | 3 | 99.49 | 0.06 | 99.49 | 5.0 | 2.1 | 85.8 |
| vit_b | finetune | 0 | basic | 112 | 2e-05 | 5 | 3 | 99.35 | 0.06 | 99.33 | 4.7 | 1.1 | 85.69 |
| resnet50 | finetune | 0 | basic | 224 | 0.0001 | 5 | 3 | 99.31 | 0.06 | 99.29 | 4.3 | 1.9 | 23.51 |
| vit_b | finetune | 0 | basic | 224 | 0.0002 | 5 | 3 | 98.68 | 0.11 | 98.68 | 5.0 | 2.2 | 85.8 |
| simple_cnn | finetune | 0 | basic | 224 | 0.001 | 30 | 3 | 97.19 | 0.13 | 97.14 | 27.0 | 10.3 | 1.17 |
| simple_cnn | finetune | 0 | strong | 224 | 0.001 | 30 | 3 | 96.4 | 0.22 | 96.34 | 27.7 | 10.2 | 1.17 |
| simple_cnn | finetune | 0 | none | 224 | 0.001 | 30 | 3 | 95.89 | 0.31 | 95.75 | 27.0 | 10.2 | 1.17 |
| simple_cnn | finetune | 5000 | basic | 224 | 0.001 | 30 | 3 | 95.49 | 0.28 | 95.47 | 28.3 | 5.9 | 1.17 |
| simple_cnn | finetune | 2500 | basic | 224 | 0.001 | 30 | 3 | 90.97 | 0.6 | 90.85 | 28.7 | 3.8 | 1.17 |
| simple_cnn | finetune | 1000 | basic | 224 | 0.001 | 30 | 3 | 82.43 | 0.25 | 82.35 | 29.0 | 2.5 | 1.17 |
| simple_cnn | finetune | 500 | basic | 224 | 0.001 | 30 | 3 | 76.65 | 1.23 | 76.48 | 28.3 | 1.9 | 1.17 |
