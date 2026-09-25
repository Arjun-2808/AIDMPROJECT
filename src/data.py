"""Dogs-vs-Cats data pipeline.

Directory layout expected under `data_root` (see scripts/setup_data.py):

    datasets/
      train/{cat,dog}/*.jpg     10000 each
      val/{cat,dog}/*.jpg        2500 each
      test/*.jpg                  500, unlabelled, named <id>.jpg

Class mapping is pinned to  cat = 0, dog = 1  to match the submission format
required by the brief (1 = dog, 0 = cat). Do not rely on ImageFolder's
alphabetical ordering silently -- we assert it.
"""
from __future__ import annotations

import re
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from .utils import seed_worker

CLASS_TO_IDX = {"cat": 0, "dog": 1}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(img_size: int, train: bool, augment: str = "standard"):
    """Augmentation recipes. Report question (a) -- keep these named and logged.

    none      : resize + normalise only (ablation baseline)
    standard  : flip + small affine + colour jitter (safe for cats/dogs)
    strong    : standard + RandomResizedCrop + erasing (for longer schedules)
    """
    normalise = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    if not train or augment == "none":
        return transforms.Compose([
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalise,
        ])

    if augment == "standard":
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalise,
        ])

    if augment == "strong":
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=20, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            transforms.ToTensor(),
            normalise,
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        ])

    raise ValueError(f"unknown augment recipe: {augment!r}")


class TestImageDataset(Dataset):
    """Flat folder of unlabelled test images -> (tensor, int_id).

    Sorted numerically by filename stem so submission.csv is ordered 1..500.
    """

    def __init__(self, root: str | Path, transform=None):
        self.root = Path(root)
        files = [p for p in self.root.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if not files:
            raise FileNotFoundError(f"no images in {self.root}")

        def key(p: Path):
            m = re.search(r"\d+", p.stem)
            return (int(m.group()) if m else 0, p.stem)

        self.files = sorted(files, key=key)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int):
        path = self.files[i]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        m = re.search(r"\d+", path.stem)
        return img, int(m.group()) if m else i


def _subset(dataset, n_per_class: int | None, seed: int):
    """Take a class-balanced random subset.

    The brief explicitly allows training on fewer images; this makes the
    train-set-size ablation for question (f) a one-flag change.
    """
    if not n_per_class:
        return dataset
    g = torch.Generator().manual_seed(seed)
    targets = torch.tensor(dataset.targets)
    keep: list[int] = []
    for cls in torch.unique(targets):
        idx = (targets == cls).nonzero(as_tuple=True)[0]
        perm = idx[torch.randperm(len(idx), generator=g)][:n_per_class]
        keep.extend(perm.tolist())
    keep.sort()
    return Subset(dataset, keep)


def build_dataloaders(
    data_root: str | Path,
    img_size: int = 224,
    batch_size: int = 32,
    augment: str = "standard",
    train_per_class: int | None = None,
    val_per_class: int | None = None,
    num_workers: int = 4,
    seed: int = 42,
    pin_memory: bool = True,
):
    root = Path(data_root)
    train_ds = ImageFolder(root / "train", transform=build_transforms(img_size, True, augment))
    val_ds = ImageFolder(root / "val", transform=build_transforms(img_size, False))

    for ds, name in ((train_ds, "train"), (val_ds, "val")):
        if ds.class_to_idx != CLASS_TO_IDX:
            raise RuntimeError(
                f"{name} class mapping is {ds.class_to_idx}, expected {CLASS_TO_IDX}. "
                "submission.csv requires cat=0, dog=1."
            )

    train_ds = _subset(train_ds, train_per_class, seed)
    val_ds = _subset(val_ds, val_per_class, seed)

    g = torch.Generator()
    g.manual_seed(seed)
    common = dict(num_workers=num_workers, pin_memory=pin_memory,
                  worker_init_fn=seed_worker, generator=g,
                  persistent_workers=num_workers > 0)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, **common)
    return train_loader, val_loader


def build_test_loader(data_root, img_size=224, batch_size=64, num_workers=4):
    ds = TestImageDataset(Path(data_root) / "test", build_transforms(img_size, False))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
