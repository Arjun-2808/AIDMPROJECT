import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, ResNet34_Weights, resnet18, resnet34


class Upsample(nn.Module):
    """Bilinear upsampling on the GPU, so small CIFAR images can be augmented at 32x32."""

    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def forward(self, x):
        return F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)


def build_model(name: str, num_classes: int, pretrained: bool, dropout: float,
                fine_tuning: str, upsample_to: int | None = None):
    if name not in {"resnet18", "resnet34"}:
        raise ValueError(f"Unsupported model: {name}")
    if name == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    else:
        model = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)
    if pretrained:
        for parameter in model.parameters():
            parameter.requires_grad = False
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
    if fine_tuning in {"layer4_plus_classifier", "full"}:
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
    if fine_tuning == "full":
        for parameter in model.parameters():
            parameter.requires_grad = True
    if upsample_to is not None:
        return nn.Sequential(Upsample(upsample_to), model)
    return model


def backbone(model):
    """The ResNet itself, whether or not it is wrapped with an Upsample layer."""
    return model[1] if isinstance(model, nn.Sequential) else model


def count_params(model, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
