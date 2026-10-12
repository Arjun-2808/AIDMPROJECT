"""Datasets and transforms for Dogs vs. Cats and CIFAR-10."""
import pickle
import random
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

DVC_CLASSES = ["cat", "dog"]  # label 0 = cat, 1 = dog (same convention as submission.csv)
CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]
IMG_EXTS = {".jpg", ".jpeg", ".png"}
BICUBIC = T.InterpolationMode.BICUBIC


# ---------------------------------------------------------------- Dogs vs. Cats

def list_images(folder):
    return sorted(p for p in Path(folder).iterdir()
                  if p.suffix.lower() in IMG_EXTS and not p.name.startswith("."))


def dvc_split(root, split, n_per_class=0, data_seed=0):
    """(paths, labels) for train/val. n_per_class > 0 takes a seeded subset per class.

    The subset is the first n of one fixed shuffle, so subsets are nested
    (500 ⊂ 1000 ⊂ ...) and identical across training seeds.
    """
    paths, labels = [], []
    for label, cls in enumerate(DVC_CLASSES):
        files = list_images(Path(root) / split / cls)
        if 0 < n_per_class < len(files):
            random.Random(data_seed).shuffle(files)
            files = sorted(files[:n_per_class])
        paths += files
        labels += [label] * len(files)
    return paths, labels


def dvc_test(root):
    """(paths, ids) for the unlabelled test set, ordered by numeric id."""
    files = sorted(list_images(Path(root) / "test"), key=lambda p: int(p.stem))
    return files, [int(p.stem) for p in files]


class ImageList(Dataset):
    def __init__(self, paths, targets, transform):
        self.paths, self.targets, self.transform = list(paths), list(targets), transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as img:
            img = img.convert("RGB")
        return self.transform(img), self.targets[i]


def dvc_transforms(aug, img_size, mean, std, crop_pct=0.875):
    norm = [T.ToTensor(), T.Normalize(mean, std)]
    eval_tf = T.Compose([T.Resize(int(round(img_size / crop_pct)), interpolation=BICUBIC),
                         T.CenterCrop(img_size), *norm])
    if aug == "none":
        train_tf = eval_tf
    elif aug == "basic":  # scaling + rotation + flipping, as suggested in the brief
        train_tf = T.Compose([T.RandomRotation(15, interpolation=T.InterpolationMode.BILINEAR),
                              T.RandomResizedCrop(img_size, scale=(0.6, 1.0), interpolation=BICUBIC),
                              T.RandomHorizontalFlip(), *norm])
    elif aug == "strong":  # basic + photometric policy + occlusion
        train_tf = T.Compose([T.RandomRotation(15, interpolation=T.InterpolationMode.BILINEAR),
                              T.RandomResizedCrop(img_size, scale=(0.35, 1.0), interpolation=BICUBIC),
                              T.RandomHorizontalFlip(), T.TrivialAugmentWide(), *norm,
                              T.RandomErasing(p=0.25)])
    else:
        raise ValueError(f"unknown aug {aug!r}")
    return train_tf, eval_tf


# ---------------------------------------------------------------- CIFAR-10

def load_cifar10(root):
    """Read the python-pickle batches in <root>/cifar-10-batches-py into uint8 HWC arrays."""
    d = Path(root) / "cifar-10-batches-py"

    def read(name):
        with open(d / name, "rb") as f:
            b = pickle.load(f, encoding="latin1")
        return b["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), np.asarray(b["labels"])

    xs, ys = zip(*(read(f"data_batch_{i}") for i in range(1, 6)))
    x_test, y_test = read("test_batch")
    return np.concatenate(xs), np.concatenate(ys), x_test, y_test


def cifar_split(y, n_val_per_class=500, seed=0):
    """Stratified train/val split of the official training set -> (train_idx, val_idx)."""
    rng = np.random.RandomState(seed)
    tr, va = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        va.append(idx[:n_val_per_class])
        tr.append(idx[n_val_per_class:])
    return np.sort(np.concatenate(tr)), np.sort(np.concatenate(va))


def make_step_imbalance(idx, y, minority, frac, seed=0):
    """Keep only `frac` of the images of each minority class (step imbalance)."""
    rng = np.random.RandomState(seed)
    keep = []
    for c in np.unique(y[idx]):
        ci = idx[y[idx] == c]
        if c in minority:
            ci = rng.permutation(ci)[:max(1, int(round(frac * len(ci))))]
        keep.append(ci)
    return np.sort(np.concatenate(keep))


class ArrayDataset(Dataset):
    def __init__(self, x, y, transform):
        self.x, self.y, self.transform = x, y, transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.transform(Image.fromarray(self.x[i])), int(self.y[i])


def cifar_transforms(aug, mean, std):
    """Augment at native 32x32; upsampling to the backbone resolution happens on the GPU."""
    norm = [T.ToTensor(), T.Normalize(mean, std)]
    crop_flip = [T.RandomCrop(32, padding=4, padding_mode="reflect"), T.RandomHorizontalFlip()]
    eval_tf = T.Compose(norm)
    if aug == "none":
        train_tf = eval_tf
    elif aug == "basic":
        train_tf = T.Compose([*crop_flip, *norm])
    elif aug == "strong":
        train_tf = T.Compose([*crop_flip, T.TrivialAugmentWide(), *norm, T.RandomErasing(p=0.25)])
    else:
        raise ValueError(f"unknown aug {aug!r}")
    return train_tf, eval_tf
