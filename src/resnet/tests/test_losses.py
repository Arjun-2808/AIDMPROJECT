import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from aidm_project.losses import FocalLoss, LogitAdjustedLoss, build_loss


def test_focal_loss_with_zero_gamma_is_cross_entropy():
    logits, targets = torch.randn(8, 10), torch.randint(0, 10, (8,))
    assert torch.allclose(FocalLoss(0.0)(logits, targets), F.cross_entropy(logits, targets), atol=1e-6)


def test_logit_adjustment_with_uniform_prior_is_cross_entropy():
    logits, targets = torch.randn(8, 10), torch.randint(0, 10, (8,))
    loss = LogitAdjustedLoss([100] * 10)(logits, targets)
    assert torch.allclose(loss, F.cross_entropy(logits, targets), atol=1e-6)


def test_weighted_ce_upweights_rare_classes():
    weight = build_loss("wce", [900, 100]).weight
    assert weight[1] > weight[0]
    assert torch.isclose((weight * torch.tensor([900.0, 100.0])).sum() / 1000, torch.tensor(1.0))
