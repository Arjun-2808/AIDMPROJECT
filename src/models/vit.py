"""Vision Transformer track  --  OWNER: <member B>

Two flavours, because the interesting story for the report is the contrast:

  * fine-tuned pretrained ViT/DeiT  -> very strong on 25k images
  * ViT trained from scratch        -> noticeably worse, because ViTs lack the
    locality/translation-equivariance priors a CNN gets for free and therefore
    need far more data. That contrast is a ready-made answer to question (f).

`timm` is optional -- torchvision's vit_b_16 is the fallback.
"""
from __future__ import annotations

import torch
from torch import nn

from . import register


def _timm_model(arch: str, num_classes: int, pretrained: bool, freeze: bool, **kw):
    import timm

    model = timm.create_model(arch, pretrained=pretrained, num_classes=num_classes, **kw)
    if freeze:
        for name, p in model.named_parameters():
            # Keep the classifier head (and its norm) trainable.
            p.requires_grad = name.startswith(("head", "fc_norm", "norm."))
    return model


@register("vit_b16")
def _vit_b16(num_classes=2, pretrained=True, freeze_backbone=False, **kw):
    """ViT-Base/16 at 224px. ~86M params -- the default heavyweight."""
    try:
        return _timm_model("vit_base_patch16_224", num_classes, pretrained, freeze_backbone)
    except ImportError:
        import torchvision.models as tvm
        m = tvm.vit_b_16(weights="DEFAULT" if pretrained else None)
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
        if freeze_backbone:
            for n, p in m.named_parameters():
                p.requires_grad = n.startswith("heads")
        return m


@register("vit_s16")
def _vit_s16(num_classes=2, pretrained=True, freeze_backbone=False, **kw):
    """ViT-Small/16 -- ~22M params, trains far faster, usually within ~1% of ViT-B here."""
    return _timm_model("vit_small_patch16_224", num_classes, pretrained, freeze_backbone)


@register("deit_s16")
def _deit_s16(num_classes=2, pretrained=True, freeze_backbone=False, **kw):
    """DeiT-Small: ViT architecture with a data-efficient training recipe."""
    return _timm_model("deit_small_patch16_224", num_classes, pretrained, freeze_backbone)


@register("swin_t")
def _swin_t(num_classes=2, pretrained=True, freeze_backbone=False, **kw):
    """Swin-Tiny: hierarchical windowed attention -- the CNN/ViT midpoint."""
    return _timm_model("swin_tiny_patch4_window7_224", num_classes, pretrained, freeze_backbone)


class TinyViT(nn.Module):
    """Minimal ViT written out in full, for training from scratch on small images.

    Kept deliberately readable: patch embed -> [CLS] + learned pos-emb ->
    N transformer encoder blocks -> norm -> linear head on the CLS token.
    Good for CIFAR-10 (32px, patch 4) and for the report's architecture figure.
    """

    def __init__(self, img_size=32, patch_size=4, num_classes=10, dim=192,
                 depth=6, heads=3, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        if img_size % patch_size:
            raise ValueError("img_size must be divisible by patch_size")
        n_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(3, dim, patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)      # (B, N, D)
        x = torch.cat([self.cls_token.expand(b, -1, -1), x], dim=1)
        x = self.drop(x + self.pos_embed)
        x = self.encoder(x)
        return self.head(self.norm(x)[:, 0])


@register("tiny_vit")
def _tiny_vit(num_classes=2, pretrained=False, img_size=32, patch_size=4,
              dim=192, depth=6, heads=3, **kw):
    if pretrained:
        raise ValueError("tiny_vit has no pretrained weights -- set pretrained: false")
    return TinyViT(img_size=img_size, patch_size=patch_size, num_classes=num_classes,
                   dim=dim, depth=depth, heads=heads)
