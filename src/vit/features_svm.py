"""Conventional classifiers (logistic regression, RBF-SVM, kNN) on fixed features.

Features are either raw pixels (32x32 RGB) or frozen pretrained backbones (no fine-tuning).
Hyper-parameters (C, k) are chosen by 3-fold cross-validation on the training subset only;
the validation set is used once, for the reported accuracy.
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torchvision import transforms as T

from data import ImageList, dvc_split, dvc_transforms
from engine import autocast, default_workers, get_device, make_loader, save_json
from models import build_model, extract_features
from train_dvc import _limit

CLASSIFIERS = {
    "logreg": (LogisticRegression(max_iter=5000), {"C": [0.01, 0.1, 1, 10]}),
    "svm_rbf": (SVC(kernel="rbf"), {"C": [0.1, 1, 10]}),
    "knn": (KNeighborsClassifier(), {"n_neighbors": [1, 5, 15, 31]}),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbones", default="pixels,resnet50,vit_s,vit_b")
    p.add_argument("--n-per-class", type=int, default=5000)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=default_workers())
    p.add_argument("--data-root", default=os.environ.get("IE4483_DATA", "data/datasets"))
    p.add_argument("--out-root", default=os.environ.get("IE4483_OUT", "runs"))
    p.add_argument("--no-pretrained", action="store_true", help="random init (smoke tests only)")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args(argv)


@torch.no_grad()
def featurize(backbone, splits, a, device):
    if backbone == "pixels":
        tf = T.Compose([T.Resize((32, 32), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor()])
        fn = lambda x: x.flatten(1)
    else:
        model, dcfg = build_model(backbone, 2, a.img_size, pretrained=not a.no_pretrained)
        model.to(device).eval()
        tf = dvc_transforms("none", a.img_size, dcfg["mean"], dcfg["std"], dcfg["crop_pct"])[1]

        def fn(x):
            with autocast(device):
                return extract_features(model, x.to(device)).float()

    feats = []
    for paths, labels in splits:
        loader = make_loader(ImageList(paths, labels, tf), a.batch_size, workers=a.workers)
        feats.append(torch.cat([fn(x).cpu() for x, _ in loader]).numpy())
    return feats


def main(argv=None):
    a = parse_args(argv)
    out = Path(a.out_root) / "features_svm"
    out.mkdir(parents=True, exist_ok=True)
    device = get_device()
    train = _limit(dvc_split(a.data_root, "train", a.n_per_class), a.limit)
    val = _limit(dvc_split(a.data_root, "val"), a.limit)
    y_tr, y_va = np.asarray(train[1]), np.asarray(val[1])
    cv = 3 if not a.limit else 2

    results = {}
    for backbone in a.backbones.split(","):
        t0 = time.time()
        X_tr, X_va = featurize(backbone, [train, val], a, device)
        print(f"{backbone}: features {X_tr.shape} in {time.time() - t0:.0f}s", flush=True)
        results[backbone] = {"feature_dim": int(X_tr.shape[1])}
        for name, (est, grid) in CLASSIFIERS.items():
            t0 = time.time()
            pipe = make_pipeline(StandardScaler(), est)
            step = pipe.steps[-1][0]
            search = GridSearchCV(pipe, {f"{step}__{k}": v for k, v in grid.items()}, cv=cv, n_jobs=max(1, a.workers))
            search.fit(X_tr, y_tr)
            val_acc = float((search.predict(X_va) == y_va).mean())
            results[backbone][name] = dict(
                best_params={k.split("__")[1]: v for k, v in search.best_params_.items()},
                cv_acc=float(search.best_score_), val_acc=val_acc, seconds=time.time() - t0,
                cv_table=[dict(params={k.split("__")[1]: v for k, v in prm.items()}, cv_acc=float(s))
                          for prm, s in zip(search.cv_results_["params"], search.cv_results_["mean_test_score"])])
            print(f"  {name:8s} best {results[backbone][name]['best_params']}  cv {search.best_score_:.4f}  "
                  f"val {val_acc:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        save_json(dict(n_train=len(y_tr), n_val=len(y_va), results=results), out / "results.json")


if __name__ == "__main__":
    main()
