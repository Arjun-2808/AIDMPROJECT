import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

METHODS = ["ce", "wce", "oversample", "focal", "logit_adjust"]


class FocalLoss(nn.Module):
    """Lin et al. 2017: down-weights easy (mostly majority-class) examples."""

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        log_p = F.log_softmax(logits, 1).gather(1, targets[:, None]).squeeze(1)
        return (-((1 - log_p.exp()) ** self.gamma) * log_p).mean()


class LogitAdjustedLoss(nn.Module):
    """Menon et al. 2021: cross-entropy on logits + tau * log(class prior); predict with raw logits."""

    def __init__(self, class_counts, tau: float = 1.0):
        super().__init__()
        prior = torch.as_tensor(class_counts, dtype=torch.float) / float(sum(class_counts))
        self.register_buffer("adjust", tau * prior.log())

    def forward(self, logits, targets):
        return F.cross_entropy(logits + self.adjust, targets)


def build_loss(method: str, class_counts, label_smoothing: float = 0.0) -> nn.Module:
    counts = np.maximum(np.asarray(class_counts), 1)
    if method in {"ce", "oversample"}:
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if method == "wce":
        # "balanced" weights w_c = N / (C * n_c), so the average weight per sample is 1
        weight = torch.as_tensor(counts.sum() / (len(counts) * counts), dtype=torch.float)
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
    if method == "focal":
        return FocalLoss(2.0)
    if method == "logit_adjust":
        return LogitAdjustedLoss(counts, 1.0)
    raise ValueError(f"Unsupported method: {method}")
