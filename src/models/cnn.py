"""CNN track  --  OWNER: <member A>

Covers the brief's suggested route: a CNN trained from scratch, plus pretrained
VGG/ResNet backbones with a new classifier head. The scratch net gives you the
"what does pretraining actually buy us" comparison for question (f).
"""
from __future__ import annotations

import torch
from torch import nn

from . import register


class SimpleCNN(nn.Module):
    """From-scratch CNN: 4 conv blocks -> GAP -> linear.

    Input  (B, 3, H, W)  ->  Output (B, num_classes) logits.
    Global average pooling keeps it resolution-agnostic, so the same net works
    at 64px (fast ablations) and 224px without changing the head.
    """

    def __init__(self, num_classes: int = 2, widths=(32, 64, 128, 256), dropout: float = 0.3):
        super().__init__()
        blocks, in_ch = [], 3
        for w in widths:
            blocks += [
                nn.Conv2d(in_ch, w, 3, padding=1, bias=False),
                nn.BatchNorm2d(w),
                nn.ReLU(inplace=True),
                nn.Conv2d(w, w, 3, padding=1, bias=False),
                nn.BatchNorm2d(w),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            in_ch = w
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(in_ch, num_classes)
        )
        self._init_weights()

    def _init_weights(self) -> None:
        # He init -- the right choice for ReLU nets, and question (b) asks
        # about initialisation explicitly.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))


@register("simple_cnn")
def _simple_cnn(num_classes=2, pretrained=False, dropout=0.3, **kw):
    return SimpleCNN(num_classes=num_classes, dropout=dropout)


def _torchvision_backbone(arch: str, num_classes: int, pretrained: bool, freeze: bool):
    """Swap the final layer of a torchvision classifier for our own head."""
    import torchvision.models as tvm

    weights = "DEFAULT" if pretrained else None
    model = getattr(tvm, arch)(weights=weights)

    if freeze:
        for p in model.parameters():
            p.requires_grad = False

    if hasattr(model, "fc"):                      # ResNet family
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, "classifier"):            # VGG / DenseNet / EfficientNet
        head = model.classifier
        if isinstance(head, nn.Sequential):
            last = head[-1]
            head[-1] = nn.Linear(last.in_features, num_classes)
        else:
            model.classifier = nn.Linear(head.in_features, num_classes)
    else:
        raise RuntimeError(f"don't know how to re-head {arch}")
    return model


@register("resnet18")
def _resnet18(num_classes=2, pretrained=True, freeze_backbone=False, **kw):
    return _torchvision_backbone("resnet18", num_classes, pretrained, freeze_backbone)


@register("resnet50")
def _resnet50(num_classes=2, pretrained=True, freeze_backbone=False, **kw):
    return _torchvision_backbone("resnet50", num_classes, pretrained, freeze_backbone)


@register("vgg16_bn")
def _vgg16(num_classes=2, pretrained=True, freeze_backbone=False, **kw):
    return _torchvision_backbone("vgg16_bn", num_classes, pretrained, freeze_backbone)


@register("densenet121")
def _densenet121(num_classes=2, pretrained=True, freeze_backbone=False, **kw):
    return _torchvision_backbone("densenet121", num_classes, pretrained, freeze_backbone)
