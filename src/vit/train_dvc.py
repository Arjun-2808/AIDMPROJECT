"""Train / evaluate one Dogs vs. Cats classifier (fine-tune or linear probe).

Example:
    python train_dvc.py --model vit_b --mode finetune --aug basic --epochs 5 --lr 2e-5 --save-ckpt
"""
import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from data import DVC_CLASSES, ImageList, dvc_split, dvc_test, dvc_transforms
from engine import (build_optimizer, default_workers, get_device, make_loader, nll, predict, save_json,
                    set_seed, train_one_epoch, warmup_cosine)
from models import MODEL_NAMES, build_model, count_params, freeze_backbone


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=MODEL_NAMES, default="vit_b")
    p.add_argument("--mode", choices=["finetune", "linprobe"], default="finetune")
    p.add_argument("--n-per-class", type=int, default=0, help="training images per class (0 = all 10000)")
    p.add_argument("--aug", choices=["none", "basic", "strong"], default="basic")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0, help="training seed (init of head, data order, augmentation)")
    p.add_argument("--data-seed", type=int, default=0, help="seed for choosing the training subset")
    p.add_argument("--workers", type=int, default=default_workers())
    p.add_argument("--data-root", default=os.environ.get("IE4483_DATA", "data/datasets"))
    p.add_argument("--out-root", default=os.environ.get("IE4483_OUT", "runs"))
    p.add_argument("--name", default=None, help="run directory name (default: derived from the config)")
    p.add_argument("--save-ckpt", action="store_true", help="save best model and predict the test set")
    p.add_argument("--no-pretrained", action="store_true", help="random init (smoke tests only)")
    p.add_argument("--limit", type=int, default=0, help="cap every split at this many images (smoke tests)")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def run_name(a):
    n = a.n_per_class or "all"
    return f"{a.model}-{a.mode}-n{n}-{a.aug}-{a.img_size}px-lr{a.lr:g}-s{a.seed}"


def _limit(items, n):
    """Evenly spaced subset, so a class-sorted list keeps both classes."""
    if not n or n >= len(items[0]):
        return items
    idx = np.linspace(0, len(items[0]) - 1, n).astype(int)
    return tuple([seq[i] for i in idx] for seq in items)


def main(argv=None):
    a = parse_args(argv)
    out = Path(a.out_root) / "dvc" / (a.name or run_name(a))
    if (out / "metrics.json").exists() and not a.overwrite:
        print(f"[skip] {out} already finished")
        return
    out.mkdir(parents=True, exist_ok=True)
    set_seed(a.seed)
    device = get_device()

    model, dcfg = build_model(a.model, len(DVC_CLASSES), a.img_size, pretrained=not a.no_pretrained)
    if a.mode == "linprobe":
        freeze_backbone(model)
    model.to(device)

    train_tf, eval_tf = dvc_transforms(a.aug, a.img_size, dcfg["mean"], dcfg["std"], dcfg["crop_pct"])
    tr_paths, tr_labels = _limit(dvc_split(a.data_root, "train", a.n_per_class, a.data_seed), a.limit)
    va_paths, va_labels = _limit(dvc_split(a.data_root, "val"), a.limit)
    te_paths, te_ids = _limit(dvc_test(a.data_root), a.limit)
    train_loader = make_loader(ImageList(tr_paths, tr_labels, train_tf), a.batch_size, shuffle=True,
                               workers=a.workers, seed=a.seed, drop_last=len(tr_paths) > a.batch_size)
    val_loader = make_loader(ImageList(va_paths, va_labels, eval_tf), 2 * a.batch_size, workers=a.workers)
    print(f"{out.name}: train {len(tr_paths)} ({np.bincount(tr_labels).tolist()} cat/dog), "
          f"val {len(va_paths)}, test {len(te_paths)}, device {device}", flush=True)

    optimizer = build_optimizer(model, a.lr, a.wd)
    scheduler = warmup_cosine(optimizer, a.epochs * len(train_loader))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=a.label_smoothing)

    history, best, best_state, best_probs = [], {"val_acc": -1.0}, None, None
    t_start = time.time()
    for epoch in range(1, a.epochs + 1):
        t0 = time.time()
        lr = scheduler.get_last_lr()[0]
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scheduler, loss_fn, device,
                                          frozen_backbone=a.mode == "linprobe")
        probs, targets = predict(model, val_loader, device)
        va_acc, va_loss = float((probs.argmax(1) == targets).mean()), nll(probs, targets)
        history.append(dict(epoch=epoch, lr=lr, train_loss=tr_loss, train_acc=tr_acc,
                            val_loss=va_loss, val_acc=va_acc, seconds=time.time() - t0))
        print(f"  epoch {epoch}/{a.epochs}  train {tr_loss:.4f}/{tr_acc:.4f}  "
              f"val {va_loss:.4f}/{va_acc:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if va_acc > best["val_acc"]:
            best = dict(val_acc=va_acc, val_loss=va_loss, epoch=epoch)
            best_probs = probs
            if a.save_ckpt:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    with open(out / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)
    with open(out / "val_preds.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "p_dog", "pred"])
        for path, label, p in zip(va_paths, va_labels, best_probs[:, 1]):
            w.writerow([Path(path).relative_to(a.data_root).as_posix(), label, f"{p:.6f}", int(p > 0.5)])

    metrics = dict(task="dvc", config=vars(a), n_train=len(tr_paths), n_val=len(va_paths),
                   params=count_params(model), trainable_params=count_params(model, trainable_only=True),
                   best_val_acc=best["val_acc"], best_val_loss=best["val_loss"], best_epoch=best["epoch"],
                   final_val_acc=history[-1]["val_acc"], final_val_loss=history[-1]["val_loss"],
                   train_time_s=time.time() - t_start)

    if a.save_ckpt:
        model.load_state_dict(best_state)
        torch.save({"state_dict": best_state, "config": vars(a)}, out / "model.pt")
        test_loader = make_loader(ImageList(te_paths, te_ids, eval_tf), 2 * a.batch_size, workers=a.workers)
        probs, ids = predict(model, test_loader, device, tta_flip=True)
        with open(out / "test_probs.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "p_dog"])
            w.writerows([int(i), f"{p:.6f}"] for i, p in zip(ids, probs[:, 1]))
        with open(out / "submission.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "label"])  # 1 = dog, 0 = cat
            w.writerows([int(i), int(p > 0.5)] for i, p in zip(ids, probs[:, 1]))
        metrics["test_pred_dogs"] = int((probs[:, 1] > 0.5).sum())

    save_json(metrics, out / "metrics.json")  # written last: marks the run as complete
    print(f"done: best val acc {best['val_acc']:.4f} at epoch {best['epoch']} -> {out}")


if __name__ == "__main__":
    main()
