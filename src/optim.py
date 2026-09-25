"""Optimiser and LR-schedule factory.

This is the axis the three of us vary independently (report question (c)):
each member owns a model family AND an optimisation strategy, and every run
logs exactly which one it used so the comparison table is honest.

Supported optimisers: sgd, sgd_nesterov, adam, adamw, rmsprop, adagrad, adadelta, nadam
Supported schedules : none, step, multistep, cosine, cosine_warm, plateau, onecycle
"""
from __future__ import annotations

import torch
from torch import nn, optim

OPTIMIZERS = {
    "sgd", "sgd_nesterov", "adam", "adamw",
    "rmsprop", "adagrad", "adadelta", "nadam",
}
SCHEDULERS = {
    "none", "step", "multistep", "cosine", "cosine_warm", "plateau", "onecycle",
}


def param_groups(model: nn.Module, weight_decay: float, no_decay_norm_bias: bool = True):
    """Exclude norm/bias params from weight decay.

    Standard practice for transformers; also helps CNNs slightly. Turning it
    off is a legitimate ablation, hence the flag.
    """
    if not no_decay_norm_bias:
        return [{"params": [p for p in model.parameters() if p.requires_grad],
                 "weight_decay": weight_decay}]

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer(
    model: nn.Module,
    name: str = "adamw",
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    momentum: float = 0.9,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    no_decay_norm_bias: bool = True,
) -> optim.Optimizer:
    name = name.lower()
    if name not in OPTIMIZERS:
        raise ValueError(f"unknown optimizer {name!r}; choose from {sorted(OPTIMIZERS)}")

    groups = param_groups(model, weight_decay, no_decay_norm_bias)

    if name in ("sgd", "sgd_nesterov"):
        return optim.SGD(groups, lr=lr, momentum=momentum,
                         nesterov=(name == "sgd_nesterov"))
    if name == "adam":
        return optim.Adam(groups, lr=lr, betas=betas, eps=eps)
    if name == "adamw":
        return optim.AdamW(groups, lr=lr, betas=betas, eps=eps)
    if name == "nadam":
        return optim.NAdam(groups, lr=lr, betas=betas, eps=eps)
    if name == "rmsprop":
        return optim.RMSprop(groups, lr=lr, momentum=momentum, eps=eps)
    if name == "adagrad":
        return optim.Adagrad(groups, lr=lr, eps=eps)
    return optim.Adadelta(groups, lr=lr, eps=eps)


def build_scheduler(
    optimizer: optim.Optimizer,
    name: str = "cosine",
    epochs: int = 10,
    steps_per_epoch: int = 100,
    warmup_epochs: int = 0,
    min_lr: float = 1e-6,
    step_size: int = 5,
    gamma: float = 0.1,
    milestones: list[int] | None = None,
    max_lr: float | None = None,
):
    """Returns (scheduler, step_granularity) where granularity is 'epoch',
    'batch' or 'metric' -- the training loop uses it to know when to step."""
    name = (name or "none").lower()
    if name not in SCHEDULERS:
        raise ValueError(f"unknown scheduler {name!r}; choose from {sorted(SCHEDULERS)}")

    if name == "none":
        return None, "epoch"

    if name == "onecycle":
        peak = max_lr or max(g["lr"] for g in optimizer.param_groups)
        sched = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=peak, epochs=epochs, steps_per_epoch=steps_per_epoch,
        )
        return sched, "batch"

    if name == "plateau":
        sched = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=gamma if gamma < 1 else 0.1,
            patience=2, min_lr=min_lr,
        )
        return sched, "metric"

    if name == "step":
        core = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif name == "multistep":
        core = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones or [epochs // 2, int(epochs * 0.75)], gamma=gamma)
    elif name == "cosine_warm":
        core = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(1, epochs // 3), eta_min=min_lr)
    else:  # cosine
        core = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=min_lr)

    if warmup_epochs > 0:
        warm = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_epochs)
        core = optim.lr_scheduler.SequentialLR(
            optimizer, [warm, core], milestones=[warmup_epochs])

    return core, "epoch"


def build_criterion(name: str = "ce", label_smoothing: float = 0.0,
                    class_weights: torch.Tensor | None = None) -> nn.Module:
    """`focal` is here for the CIFAR-10 imbalance question (h)."""
    name = name.lower()
    if name in ("ce", "cross_entropy"):
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    if name == "focal":
        return FocalLoss(gamma=2.0, weight=class_weights)
    raise ValueError(f"unknown loss {name!r}")


class FocalLoss(nn.Module):
    """Down-weights easy examples; one of the two imbalance remedies for (h)."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else torch.empty(0))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        w = self.weight if self.weight.numel() else None
        ce = nn.functional.cross_entropy(logits, target, weight=w, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()
