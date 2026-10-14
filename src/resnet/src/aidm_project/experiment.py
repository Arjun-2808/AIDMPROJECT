"""One training run per call, written in the same layout as the ViT track (src/vit/train_*.py):

<out_root>/<dvc|cifar>/<run>/
    history.csv     per-epoch lr, train/val loss and accuracy
    val_preds.csv   (DvC) best-epoch validation probabilities, for error analysis
    metrics.json    config + summary metrics; written last, so it marks the run as finished
    model.pt        (DvC) best-epoch weights
    test_probs.csv / submission.csv   (DvC) flip-TTA predictions for the 500 test images
    test_probs.npy  (CIFAR) final-epoch test probabilities
"""
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .data import CIFAR_CLASSES, build_cifar_loaders, build_loaders, build_test_loader
from .device import get_device
from .losses import build_loss
from .metrics import classification_metrics
from .models import build_model, count_params
from .optimizers import build_optimizer
from .reproducibility import seed_everything
from .training import fit, predict


def dvc_run_name(config: Config) -> str:
    return f"{config.model}-{config.name}-s{config.seed}"


def cifar_run_name(config: Config) -> str:
    data = ("balanced" if config.imbalance == "none"
            else f"step{config.minority_frac:g}-m{''.join(map(str, config.minority))}")
    return f"{config.model}-{config.name}-{data}-{config.method}-s{config.seed}"


def save_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)),
                    encoding="utf-8")


def _write_csv(path: Path, header, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _train(config: Config, train_loader, val_loader, criterion=None, keep="best"):
    model = build_model(config.model, config.num_classes, config.pretrained, config.dropout, config.fine_tuning,
                        upsample_to=config.image_size if config.task == "cifar10" else None)
    optimizer = build_optimizer(model, config.optimizer, config.learning_rate, config.weight_decay, config.momentum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)
    result = fit(model, train_loader, val_loader, optimizer, scheduler, config, criterion=criterion, keep=keep)
    return model, result, result["history"]


def _start(config: Config, out: Path) -> bool:
    if (out / "metrics.json").exists():
        print(f"[skip] {out} already finished")
        return False
    out.mkdir(parents=True, exist_ok=True)
    get_device()
    seed_everything(config.seed)
    return True


def run_dvc(config: Config, data_root: str | Path, out_root: str | Path) -> Path:
    out = Path(out_root) / "dvc" / dvc_run_name(config)
    if not _start(config, out):
        return out
    device = get_device()
    train_loader, val_loader = build_loaders(data_root, config)
    print(f"{out.name}: train {len(train_loader.dataset)}, val {len(val_loader.dataset)}", flush=True)
    model, result, history = _train(config, train_loader, val_loader)

    _write_csv(out / "history.csv", list(history[0]), [list(h.values()) for h in history])
    root = Path(data_root)
    _write_csv(out / "val_preds.csv", ["path", "label", "p_dog", "pred"],
               [[Path(p).relative_to(root).as_posix(), int(y), f"{prob:.6f}", int(prob > 0.5)]
                for p, y, prob in zip(val_loader.dataset.paths, result["val_targets"], result["best_val_probs"][:, 1])])
    torch.save({"model": model.state_dict(), "epoch": result["best_epoch"], "metrics": result["best_val_metrics"],
                "config": config.__dict__}, out / "model.pt")

    probs, ids = predict(model, build_test_loader(data_root, config), device, config.amp, tta_flip=True)
    _write_csv(out / "test_probs.csv", ["id", "p_dog"], [[int(i), f"{p:.6f}"] for i, p in zip(ids, probs[:, 1])])
    _write_csv(out / "submission.csv", ["id", "label"],  # 1 = dog, 0 = cat
               [[int(i), int(p > 0.5)] for i, p in zip(ids, probs[:, 1])])

    val = result["best_val_metrics"]
    save_json({"task": "dvc", "config": config.__dict__, "n_train": len(train_loader.dataset),
               "n_val": len(val_loader.dataset), "params": count_params(model),
               "trainable_params": count_params(model, trainable_only=True),
               "best_val_acc": result["best_accuracy"], "best_val_loss": result["best_val_loss"],
               "best_epoch": result["best_epoch"], "best_val_balanced_acc": val["balanced_accuracy"],
               "best_val_precision": val["precision"], "best_val_recall": val["recall"], "best_val_f1": val["f1"],
               "best_val_confusion": val["confusion_matrix"], "final_val_acc": result["final_val_acc"],
               "final_val_loss": result["final_val_loss"], "epochs_run": len(history),
               "train_time_s": result["train_time_s"], "test_pred_dogs": int((probs[:, 1] > 0.5).sum())},
              out / "metrics.json")
    print(f"done: best val acc {result['best_accuracy']:.4f} at epoch {result['best_epoch']} -> {out}")
    return out


def run_cifar(config: Config, data_root: str | Path, out_root: str | Path) -> Path:
    out = Path(out_root) / "cifar" / cifar_run_name(config)
    if not _start(config, out):
        return out
    device = get_device()
    train_loader, val_loader, test_loader, counts = build_cifar_loaders(data_root, config)
    print(f"{out.name}: train {len(train_loader.dataset)} per-class {counts.tolist()}, "
          f"val {len(val_loader.dataset)}, test {len(test_loader.dataset)}", flush=True)
    criterion = build_loss(config.method, counts, config.label_smoothing)
    model, result, history = _train(config, train_loader, val_loader, criterion, keep="final")
    _write_csv(out / "history.csv", list(history[0]), [list(h.values()) for h in history])

    # Final-epoch model on the official test set (no test-time model selection).
    probs, targets = predict(model, test_loader, device, config.amp)
    pred = probs.argmax(1)
    cm = np.zeros((10, 10), dtype=int)
    np.add.at(cm, (targets, pred), 1)
    recall = np.diag(cm) / np.maximum(cm.sum(1), 1)
    precision = np.diag(cm) / np.maximum(cm.sum(0), 1)
    summary = classification_metrics(targets, pred, "macro")
    test = {"acc": summary["accuracy"], "balanced_acc": summary["balanced_accuracy"], "macro_f1": summary["f1"],
            "per_class_recall": recall.tolist(), "per_class_precision": precision.tolist(),
            "confusion_matrix": cm.tolist()}
    minority = list(config.minority) if config.imbalance == "step" else []
    if minority:
        test["minority_recall"] = float(recall[minority].mean())
        test["majority_recall"] = float(np.delete(recall, minority).mean())
    np.save(out / "test_probs.npy", probs.astype(np.float16))
    save_json({"task": "cifar", "config": config.__dict__, "classes": CIFAR_CLASSES, "train_counts": counts.tolist(),
               "minority": minority, "n_val": len(val_loader.dataset), "n_test": len(test_loader.dataset),
               "params": count_params(model), "trainable_params": count_params(model, trainable_only=True),
               "final_val_acc": result["final_val_acc"], "test": test, "epochs_run": len(history),
               "train_time_s": result["train_time_s"]}, out / "metrics.json")
    print(f"done: test acc {test['acc']:.4f}  macro-F1 {test['macro_f1']:.4f} -> {out}")
    return out


def run(config: Config, data_root: str | Path, out_root: str | Path) -> Path:
    if config.task == "dogs_cats":
        return run_dvc(config, data_root, out_root)
    if config.task == "cifar10":
        return run_cifar(config, data_root, out_root)
    raise ValueError(f"Unsupported task: {config.task}")
