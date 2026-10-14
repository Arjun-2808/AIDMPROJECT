import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .metrics import classification_metrics
from .device import get_device


def run_epoch(model, loader, criterion, device, optimizer: Optional[torch.optim.Optimizer] = None,
              scaler: Optional[torch.amp.GradScaler] = None, amp: bool = True):
    """One pass over `loader`; returns (metrics, softmax probabilities [N, C], targets [N])."""
    training = optimizer is not None
    model.train(training)
    total_loss, targets, probs = 0.0, [], []
    with torch.set_grad_enabled(training):
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                logits = model(images)
            loss = criterion(logits.float(), labels)
            if training:
                if scaler is not None and device.type == "cuda":
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            total_loss += loss.item() * labels.size(0)
            targets.append(labels.detach().cpu())
            probs.append(logits.detach().float().softmax(1).cpu())
    targets, probs = torch.cat(targets).numpy(), torch.cat(probs).numpy()
    average = "binary" if probs.shape[1] == 2 else "macro"
    metrics = classification_metrics(targets, probs.argmax(1), average)
    metrics["loss"] = total_loss / len(targets)
    return metrics, probs, targets


@torch.inference_mode()
def predict(model, loader, device, amp: bool = True, tta_flip: bool = False):
    """Softmax probabilities [N, C] and the loader's second field (labels or ids) as numpy arrays."""
    model.eval()
    probs, keys = [], []
    for images, key in loader:
        images = images.to(device)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            p = model(images).float().softmax(1)
            if tta_flip:
                p = 0.5 * (p + model(torch.flip(images, dims=[3])).float().softmax(1))
        probs.append(p.cpu())
        keys.append(torch.as_tensor(key))
    return torch.cat(probs).numpy(), torch.cat(keys).numpy()


def fit(model, train_loader, val_loader, optimizer, scheduler, config,
        output_path: str | Path | None = None, criterion: nn.Module | None = None, keep: str = "best"):
    """Train with early stopping on validation accuracy.

    keep="best" restores the best-validation epoch (Dogs vs. Cats); keep="final" keeps the last
    epoch (CIFAR-10, where the test set is never used for selection).
    """
    device = get_device()
    if config.cudnn_safe:
        torch.backends.cudnn.enabled = False
    print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    model.to(device)
    criterion = (criterion or nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)).to(device)
    eval_criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    best = {"val_acc": -1.0}
    best_state, stale, history = None, 0, []
    start = time.time()
    for epoch in range(config.epochs):
        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]
        train_metrics, _, _ = run_epoch(model, train_loader, criterion, device, optimizer, scaler, config.amp)
        val_metrics, val_probs, val_targets = run_epoch(model, val_loader, eval_criterion, device,
                                                        None, None, config.amp)
        if scheduler is not None:
            scheduler.step()
        history.append({"epoch": epoch + 1, "lr": lr, "train_loss": train_metrics["loss"],
                        "train_acc": train_metrics["accuracy"], "val_loss": val_metrics["loss"],
                        "val_acc": val_metrics["accuracy"], "val_macro_f1": val_metrics["f1"],
                        "seconds": time.time() - t0})
        print(f"  epoch {epoch + 1}/{config.epochs}  train {train_metrics['loss']:.4f}/"
              f"{train_metrics['accuracy']:.4f}  val {val_metrics['loss']:.4f}/{val_metrics['accuracy']:.4f}"
              f"  ({time.time() - t0:.0f}s)", flush=True)
        if val_metrics["accuracy"] > best["val_acc"]:
            best = {"val_acc": val_metrics["accuracy"], "val_loss": val_metrics["loss"], "epoch": epoch + 1,
                    "metrics": val_metrics, "probs": val_probs}
            stale = 0
            if keep == "best":
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if output_path is not None:
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"model": best_state, "epoch": best["epoch"], "metrics": val_metrics,
                                "config": config.__dict__}, output_path)
        else:
            stale += 1
            if stale >= config.patience:
                break
    if keep == "best":
        model.load_state_dict(best_state)
    return {"best_accuracy": best["val_acc"], "best_val_loss": best["val_loss"], "best_epoch": best["epoch"],
            "best_val_metrics": best["metrics"], "best_val_probs": best["probs"], "val_targets": val_targets,
            "final_val_acc": history[-1]["val_acc"], "final_val_loss": history[-1]["val_loss"],
            "history": history, "train_time_s": time.time() - start}
