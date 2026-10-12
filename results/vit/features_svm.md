### Conventional classifiers on fixed features: val acc % (CV-chosen hyper-parameter)

| features | dim | Logistic reg. | RBF-SVM | kNN |
|---|---|---|---|---|
| raw pixels | 3072 | 60.24 (C=0.01) | 66.08 (C=1) | 61.20 (n_neighbors=31) |
| ResNet-50 | 2048 | 99.50 (C=0.01) | 99.22 (C=10) | 99.08 (n_neighbors=5) |
| ViT-S/16 | 384 | 99.56 (C=0.01) | 99.60 (C=1) | 99.56 (n_neighbors=15) |
| ViT-B/16 | 768 | 99.70 (C=0.1) | 99.64 (C=10) | 99.50 (n_neighbors=5) |
