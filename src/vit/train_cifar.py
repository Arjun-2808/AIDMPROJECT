"""Train / evaluate one CIFAR-10 classifier, optionally on a class-imbalanced training set.

Examples:
    python train_cifar.py --model vit_b --epochs 5 --lr 2e-5                     # part (g)
    python train_cifar.py --model vit_b --imbalance step --method logit_adjust   # part (h)
"""
import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler

from data import CIFAR_CLASSES, ArrayDataset, cifar_split, cifar_transforms, load_cifar10, make_step_imbalance
from engine import (FocalLoss, LogitAdjustedLoss, build_optimizer, default_workers, get_device, make_loader,
                    nll, predict, save_json, set_seed, train_one_epoch, warmup_cosine)
from models import MODEL_NAMES, Upsample, build_model, count_params

METHODS = ["ce", "wce", "oversample", "focal", "logit_adjust"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=MODEL_NAMES, default="vit_b")
    p.add_argument("--img-size", type=int, default=None, help="backbone input size (default 224, 32 for simple_cnn)")
    p.add_argument("--aug", choices=["none", "basic", "strong"], default="basic")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--imbalance", choices=["none", "step"], default="none")
    p.add_argument("--minority", default="3,4,9", help="minority class ids (default cat, deer, truck)")
    p.add_argument("--minority-frac", type=float, default=0.02, help="fraction of minority images kept")
    p.add_argument("--method", choices=METHODS, default="ce", help="imbalance handling")
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--la-tau", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=default_workers())
    p.add_argument("--data-root", default=os.environ.get("IE4483_CIFAR", "data/cifar10"))
    p.add_argument("--out-root", default=os.environ.get("IE4483_OUT", "runs"))
    p.add_argument("--name", default=None)
    p.add_argument("--no-pretrained", action="store_true", help="random init (smoke tests only)")
    p.add_argument("--limit", type=int, default=0, help="cap every split at this many images (smoke tests)")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)
    if a.img_size is None:
        a.img_size = 32 if a.model == "simple_cnn" else 224
    return a


def run_name(a):
    data = "balanced" if a.imbalance == "none" else f"step{a.minority_frac:g}-m{a.minority.replace(',', '')}"
    return f"{a.model}-{a.img_size}px-{a.aug}-{data}-{a.method}-lr{a.lr:g}-s{a.seed}"


def _limit(idx, n):
    return idx if not n or n >= len(idx) else idx[np.linspace(0, len(idx) - 1, n).astype(int)]


def classification_report(probs, y, n_classes):
    pred = probs.argmax(1)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (y, pred), 1)
    support, predicted, tp = cm.sum(1), cm.sum(0), np.diag(cm)
    recall = tp / np.maximum(support, 1)
    precision = tp / np.maximum(predicted, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return dict(acc=float((pred == y).mean()), balanced_acc=float(recall.mean()), macro_f1=float(f1.mean()),
                per_class_recall=recall.tolist(), per_class_precision=precision.tolist(),
                confusion_matrix=cm.tolist())


def main(argv=None):
    a = parse_args(argv)
    out = Path(a.out_root) / "cifar" / (a.name or run_name(a))
    if (out / "metrics.json").exists() and not a.overwrite:
        print(f"[skip] {out} already finished")
        return
    out.mkdir(parents=True, exist_ok=True)
    set_seed(a.seed)
    device = get_device()

    x_train, y_train, x_test, y_test = load_cifar10(a.data_root)
    tr_idx, va_idx = cifar_split(y_train, n_val_per_class=500, seed=0)  # 45k train / 5k balanced val
    minority = [int(c) for c in a.minority.split(",")] if a.imbalance == "step" else []
    if minority:
        tr_idx = make_step_imbalance(tr_idx, y_train, minority, a.minority_frac, seed=0)
    tr_idx, va_idx = _limit(tr_idx, a.limit), _limit(va_idx, a.limit)
    te_idx = _limit(np.arange(len(y_test)), a.limit)
    counts = np.bincount(y_train[tr_idx], minlength=10)

    model, dcfg = build_model(a.model, 10, a.img_size, pretrained=not a.no_pretrained)
    if a.img_size != 32:
        model = nn.Sequential(Upsample(a.img_size), model)
    model.to(device)
    train_tf, eval_tf = cifar_transforms(a.aug, dcfg["mean"], dcfg["std"])

    train_ds = ArrayDataset(x_train[tr_idx], y_train[tr_idx], train_tf)
    sampler = None
    if a.method == "oversample":  # draw every class equally often
        g = torch.Generator()
        g.manual_seed(a.seed)
        sampler = WeightedRandomSampler(torch.as_tensor(1.0 / counts[y_train[tr_idx]], dtype=torch.double),
                                        num_samples=len(tr_idx), replacement=True, generator=g)
    train_loader = make_loader(train_ds, a.batch_size, shuffle=True, sampler=sampler, workers=a.workers,
                               seed=a.seed, drop_last=len(tr_idx) > a.batch_size)
    val_loader = make_loader(ArrayDataset(x_train[va_idx], y_train[va_idx], eval_tf), 4 * a.batch_size,
                             workers=a.workers)
    test_loader = make_loader(ArrayDataset(x_test[te_idx], y_test[te_idx], eval_tf), 4 * a.batch_size,
                              workers=a.workers)
    print(f"{out.name}: train {len(tr_idx)} per-class {counts.tolist()}, val {len(va_idx)}, "
          f"test {len(te_idx)}, device {device}", flush=True)

    # "balanced" class weights: w_c = N / (C * n_c), so the average weight per sample is 1
    class_w = torch.as_tensor(counts.sum() / (len(counts) * np.maximum(counts, 1)), dtype=torch.float)
    loss_fn = {
        "ce": lambda: nn.CrossEntropyLoss(label_smoothing=a.label_smoothing),
        "oversample": lambda: nn.CrossEntropyLoss(label_smoothing=a.label_smoothing),
        "wce": lambda: nn.CrossEntropyLoss(weight=class_w, label_smoothing=a.label_smoothing),
        "focal": lambda: FocalLoss(a.focal_gamma),
        "logit_adjust": lambda: LogitAdjustedLoss(np.maximum(counts, 1), a.la_tau),
    }[a.method]().to(device)

    optimizer = build_optimizer(model, a.lr, a.wd)
    scheduler = warmup_cosine(optimizer, a.epochs * len(train_loader))
    history, t_start = [], time.time()
    for epoch in range(1, a.epochs + 1):
        t0 = time.time()
        lr = scheduler.get_last_lr()[0]
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scheduler, loss_fn, device)
        probs, targets = predict(model, val_loader, device)
        rep = classification_report(probs, targets, 10)
        history.append(dict(epoch=epoch, lr=lr, train_loss=tr_loss, train_acc=tr_acc, val_loss=nll(probs, targets),
                            val_acc=rep["acc"], val_macro_f1=rep["macro_f1"], seconds=time.time() - t0))
        print(f"  epoch {epoch}/{a.epochs}  train {tr_loss:.4f}/{tr_acc:.4f}  val acc {rep['acc']:.4f}  "
              f"macro-F1 {rep['macro_f1']:.4f}  ({time.time() - t0:.0f}s)", flush=True)

    with open(out / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)

    # Final-epoch model on the official test set (no test-time model selection).
    probs, targets = predict(model, test_loader, device)
    test = classification_report(probs, targets, 10)
    if minority:
        rec = np.asarray(test["per_class_recall"])
        test["minority_recall"] = float(rec[minority].mean())
        test["majority_recall"] = float(np.delete(rec, minority).mean())
    np.save(out / "test_probs.npy", probs.astype(np.float16))
    save_json(dict(task="cifar", config=vars(a), classes=CIFAR_CLASSES, train_counts=counts.tolist(),
                   minority=minority, n_val=len(va_idx), n_test=len(te_idx), params=count_params(model),
                   final_val_acc=history[-1]["val_acc"], test=test, train_time_s=time.time() - t_start),
              out / "metrics.json")
    print(f"done: test acc {test['acc']:.4f}  macro-F1 {test['macro_f1']:.4f} -> {out}")


if __name__ == "__main__":
    main()
