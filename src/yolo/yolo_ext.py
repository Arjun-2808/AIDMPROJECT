"""Small extensions on top of the Ultralytics classification trainer.

- aug "none"        : real no-augmentation baseline (resize + centre crop only; the stock args
                      cannot switch off RandomResizedCrop)
- aug "strong_rot"  : stock augmentation + random rotation (the brief mentions rotation)
- loss wce_inv/wce_eff/focal : class-weighted / focal loss for the imbalanced CIFAR-10 runs

Settings are passed through the class attribute `ExtTrainer.ext` (Ultralytics rejects unknown
train() args). The custom criterion is attached to the *training* model in a callback so it is
never pickled into checkpoints.
"""

import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from ultralytics.data.augment import classify_transforms
from ultralytics.models.yolo.classify import ClassificationTrainer

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def class_counts(train_dir: Path, names: dict) -> torch.Tensor:
    """Images per class, ordered like `names` (idx -> folder name). Follows symlinks."""
    counts = []
    for i in range(len(names)):
        d = Path(train_dir) / names[i]
        counts.append(sum(1 for f in os.scandir(d) if Path(f.name).suffix.lower() in IMG_EXT))
    return torch.tensor(counts, dtype=torch.float64)


def class_weights(counts: torch.Tensor, scheme: str, beta: float = 0.999) -> torch.Tensor:
    if scheme == "wce_inv":  # w_c ~ 1 / n_c
        w = 1.0 / counts
    elif scheme == "wce_eff":  # class-balanced loss, w_c ~ (1 - beta) / (1 - beta^n_c)  (Cui et al., 2019)
        w = (1.0 - beta) / (1.0 - torch.pow(torch.tensor(beta, dtype=torch.float64), counts))
    else:
        raise ValueError(scheme)
    return (w / w.sum() * len(counts)).float()  # mean weight = 1 so the LR scale is unchanged


class ClsLoss:
    """Drop-in for v8ClassificationLoss: (preds, batch) -> (loss, {"loss": detached})."""

    def __init__(self, kind: str, weight: torch.Tensor | None = None, gamma: float = 2.0):
        self.kind, self.weight, self.gamma = kind, weight, gamma

    def __call__(self, preds, batch):
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        preds, target = preds.float(), batch["cls"]
        w = self.weight.to(preds.device) if self.weight is not None else None
        if self.kind == "focal":
            logp = F.log_softmax(preds, dim=1)
            ce = F.nll_loss(logp, target, weight=w, reduction="none")
            pt = logp.gather(1, target[:, None]).squeeze(1).exp()
            loss = ((1.0 - pt) ** self.gamma * ce).mean()
        else:
            loss = F.cross_entropy(preds, target, weight=w, reduction="mean")
        return loss, {"loss": loss.detach()}


class ExtTrainer(ClassificationTrainer):
    ext: dict = {"aug": "strong", "loss": "ce", "rotation": 15}

    def build_dataset(self, img_path, mode="train", batch=None):
        ds = super().build_dataset(img_path, mode, batch)
        if mode == "train":
            aug = self.ext.get("aug", "strong")
            if aug == "none":
                ds.torch_transforms = classify_transforms(size=self.args.imgsz)
            elif aug == "strong_rot":
                ds.torch_transforms.transforms.insert(0, T.RandomRotation(self.ext.get("rotation", 15)))
        return ds


def install_loss_callback(model, ext: dict):
    """Swap the criterion before the first step (no-op for plain cross-entropy)."""
    kind = ext.get("loss", "ce")
    if kind == "ce":
        return

    def _set_criterion(trainer):
        weight = None
        if kind in ("wce_inv", "wce_eff"):
            counts = class_counts(Path(trainer.data["train"]), trainer.data["names"])
            weight = class_weights(counts, kind)
            print(f"[loss] {kind} class counts={counts.tolist()} weights={[round(x, 3) for x in weight.tolist()]}", flush=True)
        trainer.model.criterion = ClsLoss(kind, weight)

    model.add_callback("on_train_start", _set_criterion)
