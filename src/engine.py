"""Reference train/eval loop.

You are free to ignore this and write your own loop -- but if you do, still use
`RunLogger` so your results land in the shared leaderboard in the same format.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .utils import AverageMeter


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes: int = 2):
    model.eval()
    loss_m, correct, total = AverageMeter(), 0, 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss_m.update(criterion(logits, y).item(), y.size(0))
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        for t, p in zip(y.view(-1).cpu(), pred.view(-1).cpu()):
            confusion[t, p] += 1

    return {
        "loss": loss_m.avg,
        "acc": correct / max(total, 1),
        "confusion": confusion.tolist(),
    }


def train_one_epoch(model, loader, criterion, optimizer, device,
                    scaler=None, scheduler=None, sched_granularity="epoch",
                    grad_clip: float | None = None, log_every: int = 50, logger=None):
    model.train()
    loss_m, correct, total = AverageMeter(), 0, 0
    use_amp = scaler is not None and scaler.is_enabled()

    for i, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)

        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if scheduler is not None and sched_granularity == "batch":
            scheduler.step()

        loss_m.update(loss.item(), y.size(0))
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

        if logger and log_every and i % log_every == 0:
            logger.print(f"    batch {i:4d}/{len(loader)}  loss {loss_m.avg:.4f}  "
                         f"acc {correct / max(total, 1):.4f}")

    return {"loss": loss_m.avg, "acc": correct / max(total, 1)}


def fit(model, train_loader, val_loader, criterion, optimizer, device, epochs,
        scheduler=None, sched_granularity="epoch", scaler=None, logger=None,
        ckpt_path: str | Path | None = None, early_stop_patience: int | None = None,
        grad_clip: float | None = None, num_classes: int = 2):
    """Runs the whole schedule, returns the best-epoch summary dict."""
    best = {"best_val_acc": 0.0, "best_epoch": -1, "best_val_loss": float("inf")}
    since_improved = 0

    for epoch in range(1, epochs + 1):
        lr_now = optimizer.param_groups[0]["lr"]
        if logger:
            logger.print(f"  epoch {epoch}/{epochs}  lr={lr_now:.2e}")

        tr = train_one_epoch(model, train_loader, criterion, optimizer, device,
                             scaler, scheduler, sched_granularity, grad_clip, logger=logger)
        va = evaluate(model, val_loader, criterion, device, num_classes)

        if scheduler is not None:
            if sched_granularity == "metric":
                scheduler.step(va["acc"])
            elif sched_granularity == "epoch":
                scheduler.step()

        if logger:
            logger.print(f"  -> train loss {tr['loss']:.4f} acc {tr['acc']:.4f} | "
                         f"val loss {va['loss']:.4f} acc {va['acc']:.4f}")
            logger.log_epoch(epoch=epoch, lr=lr_now, train_loss=tr["loss"],
                             train_acc=tr["acc"], val_loss=va["loss"], val_acc=va["acc"])

        if va["acc"] > best["best_val_acc"]:
            best.update(best_val_acc=va["acc"], best_epoch=epoch,
                        best_val_loss=va["loss"], confusion=va["confusion"])
            since_improved = 0
            if ckpt_path:
                Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state_dict(), "epoch": epoch,
                            "val_acc": va["acc"]}, ckpt_path)
                best["checkpoint"] = str(ckpt_path).replace("\\", "/")
        else:
            since_improved += 1
            if early_stop_patience and since_improved >= early_stop_patience:
                if logger:
                    logger.print(f"  early stop at epoch {epoch} "
                                 f"(no val improvement for {since_improved})")
                break

    return best
