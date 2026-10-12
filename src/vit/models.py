"""Model zoo: pretrained timm backbones plus a small CNN trained from scratch."""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

TIMM_IDS = {
    "vit_b": "vit_base_patch16_224.augreg2_in21k_ft_in1k",   # ViT-B/16, 86M params
    "vit_s": "vit_small_patch16_224.augreg_in21k_ft_in1k",   # ViT-S/16, 22M params
    "resnet50": "resnet50.a1_in1k",                          # ResNet-50 CNN, 24M params
}
MODEL_NAMES = [*TIMM_IDS, "simple_cnn"]


class SimpleCNN(nn.Module):
    """VGG-style CNN (4 conv blocks + global pooling + linear), no pretraining."""

    def __init__(self, num_classes, width=32):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2))

        w = width
        self.features = nn.Sequential(block(3, w), block(w, 2 * w), block(2 * w, 4 * w), block(4 * w, 8 * w))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3),
                                  nn.Linear(8 * w, num_classes))

    def forward(self, x):
        return self.head(self.features(x))

    def get_classifier(self):
        return self.head[-1]


class Upsample(nn.Module):
    """Resize the batch on the GPU, e.g. CIFAR 32x32 -> 224x224 for a pretrained ViT."""

    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        if x.shape[-1] == self.size:
            return x
        return F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)


def build_model(name, num_classes, img_size=224, pretrained=True):
    """Return (model, data_cfg) where data_cfg holds the normalisation the backbone expects."""
    if name == "simple_cnn":
        return SimpleCNN(num_classes), dict(mean=(0.5,) * 3, std=(0.5,) * 3, crop_pct=0.875)
    kwargs = dict(pretrained=pretrained, num_classes=num_classes)
    if name.startswith("vit"):
        kwargs["img_size"] = img_size  # position embeddings are interpolated for non-224 inputs
    model = timm.create_model(TIMM_IDS[name], **kwargs)
    cfg = timm.data.resolve_model_data_config(model)
    return model, dict(mean=cfg["mean"], std=cfg["std"], crop_pct=cfg.get("crop_pct", 0.875))


def freeze_backbone(model):
    """Linear probe: train only the classification head."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.get_classifier().parameters():
        p.requires_grad = True


def count_params(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


@torch.no_grad()
def extract_features(model, x):
    """Pooled pre-logit features (CLS token for ViT, global-avg-pool for CNNs)."""
    if isinstance(model, SimpleCNN):
        return model.head[:2](model.features(x))
    return model.forward_head(model.forward_features(x), pre_logits=True)
