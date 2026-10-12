"""Shared training utilities: seeding, loaders, LR schedule, train/eval loops, losses."""
import contextlib
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN kernels; remaining GPU nondeterminism is covered by averaging seeds.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def default_workers():
    try:
        n = len(os.sched_getaffinity(0))  # respects the PBS cpu allocation
    except AttributeError:
        n = os.cpu_count() or 1
    return min(16, n)


def _seed_worker(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def make_loader(ds, batch_size, shuffle=False, workers=0, sampler=None, seed=0, drop_last=False):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle and sampler is None, sampler=sampler,
                      num_workers=workers, pin_memory=torch.cuda.is_available(), drop_last=drop_last,
                      worker_init_fn=_seed_worker, generator=g, persistent_workers=workers > 0)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast(device):
    """bf16 mixed precision on GPUs that support it (A100), fp32 otherwise."""
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def warmup_cosine(optimizer, total_steps, warmup_frac=0.1):
    warmup = max(1, int(warmup_frac * total_steps))

    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def build_optimizer(model, lr, weight_decay):
    """AdamW; no weight decay on biases, norm layers, position embeddings and tokens."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or name.endswith((".pos_embed", ".cls_token")) or name in ("pos_embed", "cls_token")
         else decay).append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}], lr=lr)


def train_one_epoch(model, loader, optimizer, scheduler, loss_fn, device, clip=1.0, frozen_backbone=False):
    model.train(not frozen_backbone)  # linear probe keeps the frozen backbone in eval mode
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast(device):
            logits = model(x)
        loss = loss_fn(logits.float(), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], clip)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


@torch.no_grad()
def predict(model, loader, device, tta_flip=False):
    """Return (softmax probabilities [N, C], targets [N]) as numpy arrays."""
    model.eval()
    probs, targets = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with autocast(device):
            p = model(x).float().softmax(1)
            if tta_flip:
                p = 0.5 * (p + model(torch.flip(x, dims=[3])).float().softmax(1))
        probs.append(p.cpu())
        targets.append(torch.as_tensor(y))
    return torch.cat(probs).numpy(), torch.cat(targets).numpy()


def nll(probs, targets):
    return float(-np.log(np.clip(probs[np.arange(len(targets)), targets], 1e-12, None)).mean())


# ---------------------------------------------------------------- losses for class imbalance

class FocalLoss(nn.Module):
    """Lin et al. 2017: down-weights easy (mostly majority-class) examples."""

    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits, y):
        logp = F.log_softmax(logits, 1)
        logp_t = logp.gather(1, y[:, None]).squeeze(1)
        loss = -((1 - logp_t.exp()) ** self.gamma) * logp_t
        if self.weight is not None:
            loss = loss * self.weight[y]
        return loss.mean()


class LogitAdjustedLoss(nn.Module):
    """Menon et al. 2021: cross-entropy on logits + tau * log(class prior); predict with raw logits."""

    def __init__(self, class_counts, tau=1.0):
        super().__init__()
        prior = torch.as_tensor(class_counts, dtype=torch.float) / sum(class_counts)
        self.register_buffer("adjust", tau * prior.log())

    def forward(self, logits, y):
        return F.cross_entropy(logits + self.adjust, y)


# ---------------------------------------------------------------- io

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def load_json(path):
    with open(path) as f:
        return json.load(f)
