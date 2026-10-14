from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets, transforms

from .config import Config
from .reproducibility import worker_init_fn

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
DVC_CLASSES = ["cat", "dog"]
CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def _transforms(config: Config, train: bool) -> transforms.Compose:
    size = config.image_size
    ops = ([transforms.RandomResizedCrop(size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(12),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)] if train
           else [transforms.Resize((size, size))])
    ops += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(ops)


def _limit(n: int, limit: int) -> np.ndarray:
    """Evenly spaced subset, so a class-sorted list keeps every class."""
    if not limit or limit >= n:
        return np.arange(n)
    return np.linspace(0, n - 1, limit).astype(int)


class ImageList(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.transform = list(paths), list(labels), transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]).convert("RGB")), self.labels[i]


class ArrayDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform):
        self.images, self.labels, self.transform = images, labels, transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.transform(Image.fromarray(self.images[i])), int(self.labels[i])


def _loader(dataset, config: Config, shuffle: bool = False, sampler=None, batch_size: int | None = None):
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(dataset, batch_size=batch_size or config.batch_size, shuffle=shuffle and sampler is None,
                      sampler=sampler, generator=generator, num_workers=config.num_workers,
                      pin_memory=torch.cuda.is_available(), worker_init_fn=worker_init_fn,
                      persistent_workers=config.num_workers > 0)


def _image_list(root: Path, config: Config, train: bool) -> ImageList:
    folder = datasets.ImageFolder(root)
    keep = _limit(len(folder.samples), config.limit)
    samples = [folder.samples[i] for i in keep]
    return ImageList([p for p, _ in samples], [y for _, y in samples], _transforms(config, train))


def build_loaders(data_root: str | Path, config: Config):
    """Dogs vs. Cats train/val loaders; `val_loader.dataset.paths` lists the validation images."""
    root = Path(data_root)
    train_set = _image_list(root / "train", config, True)
    val_set = _image_list(root / "val", config, False)
    return _loader(train_set, config, shuffle=True), _loader(val_set, config, batch_size=2 * config.batch_size)


def build_test_loader(data_root: str | Path, config: Config):
    """Unlabelled test images (directly in test/ or in sub-folders); yields (image, integer id)."""
    root = Path(data_root) / "test"
    paths = sorted((p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES),
                   key=lambda p: (int(p.stem) if p.stem.isdigit() else 0, p.name))
    paths = [paths[i] for i in _limit(len(paths), config.limit)]
    ids = [int(p.stem) if p.stem.isdigit() else i for i, p in enumerate(paths)]
    return _loader(ImageList(paths, ids, _transforms(config, False)), config, batch_size=2 * config.batch_size)


# ---------------------------------------------------------------- CIFAR-10

def load_cifar10(root: str | Path):
    train = datasets.CIFAR10(root, train=True, download=True)
    test = datasets.CIFAR10(root, train=False, download=True)
    return train.data, np.asarray(train.targets), test.data, np.asarray(test.targets)


def cifar_split(labels: np.ndarray, n_val_per_class: int = 500, seed: int = 0):
    """Stratified train/val split of the official training set -> (train_idx, val_idx)."""
    rng = np.random.RandomState(seed)
    train, val = [], []
    for c in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == c))
        val.append(idx[:n_val_per_class])
        train.append(idx[n_val_per_class:])
    return np.sort(np.concatenate(train)), np.sort(np.concatenate(val))


def make_step_imbalance(idx: np.ndarray, labels: np.ndarray, minority, frac: float, seed: int = 0):
    """Keep only `frac` of the images of each minority class (step imbalance)."""
    rng = np.random.RandomState(seed)
    keep = []
    for c in np.unique(labels[idx]):
        class_idx = idx[labels[idx] == c]
        if c in minority:
            class_idx = rng.permutation(class_idx)[:max(1, int(round(frac * len(class_idx))))]
        keep.append(class_idx)
    return np.sort(np.concatenate(keep))


def cifar_transforms(train: bool):
    """Augment at native 32x32; the model upsamples to config.image_size on the GPU."""
    norm = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    if not train:
        return transforms.Compose(norm)
    return transforms.Compose([transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                               transforms.RandomHorizontalFlip(), *norm])


def build_cifar_loaders(data_root: str | Path, config: Config):
    """45k train / 5k balanced val (fixed split) / official 10k test -> (train, val, test, class counts)."""
    x_train, y_train, x_test, y_test = load_cifar10(data_root)
    train_idx, val_idx = cifar_split(y_train)
    if config.imbalance == "step":
        train_idx = make_step_imbalance(train_idx, y_train, config.minority, config.minority_frac)
    train_idx = train_idx[_limit(len(train_idx), config.limit)]
    val_idx = val_idx[_limit(len(val_idx), config.limit)]
    test_idx = _limit(len(y_test), config.limit)
    counts = np.bincount(y_train[train_idx], minlength=10)
    sampler = None
    if config.method == "oversample":  # draw every class equally often
        weights = torch.as_tensor(1.0 / np.maximum(counts, 1)[y_train[train_idx]], dtype=torch.double)
        sampler = WeightedRandomSampler(weights, len(train_idx), replacement=True,
                                        generator=torch.Generator().manual_seed(config.seed))
    train = _loader(ArrayDataset(x_train[train_idx], y_train[train_idx], cifar_transforms(True)), config,
                    shuffle=True, sampler=sampler)
    val = _loader(ArrayDataset(x_train[val_idx], y_train[val_idx], cifar_transforms(False)), config,
                  batch_size=4 * config.batch_size)
    test = _loader(ArrayDataset(x_test[test_idx], y_test[test_idx], cifar_transforms(False)), config,
                   batch_size=4 * config.batch_size)
    return train, val, test, counts
