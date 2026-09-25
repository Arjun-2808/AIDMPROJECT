"""Third track  --  OWNER: <member C>   *** TOPIC NOT DECIDED YET ***

Placeholder so the registry imports cleanly. Whoever takes this track replaces
the body with their own model(s); nothing outside this file needs to change.

Candidate topics, all defensible against the brief:
  * EfficientNet / ConvNeXt  -- modern CNN, contrasts with member A's ResNet
  * classical baseline       -- HOG or CNN features + SVM / RandomForest
                                (brief: "conventional classifiers can be used
                                as comparison")
  * CIFAR-10 + class imbalance (questions g & h, 20% of the marks)
"""
from __future__ import annotations

from . import register


@register("efficientnet_b0")
def _effnet_b0(num_classes=2, pretrained=True, freeze_backbone=False, **kw):
    from .cnn import _torchvision_backbone
    return _torchvision_backbone("efficientnet_b0", num_classes, pretrained, freeze_backbone)


@register("convnext_tiny")
def _convnext_tiny(num_classes=2, pretrained=True, freeze_backbone=False, **kw):
    from .cnn import _torchvision_backbone
    return _torchvision_backbone("convnext_tiny", num_classes, pretrained, freeze_backbone)
