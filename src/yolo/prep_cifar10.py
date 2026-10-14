"""CIFAR-10 -> ImageFolder layout, plus the imbalanced variants for question (h).

    python prep_cifar10.py --download-only     # login node: fetch the archive only (~145 MB)
    python prep_cifar10.py                     # CPU-only job: build everything

Sources (--source):
    zip          (default) Ultralytics' PNG-folder copy of CIFAR-10 hosted on GitHub. The official host
                 (cs.toronto.edu, used by torchvision) was throttled to ~1 KB/s when this was written.
    torchvision  official python tarball via torchvision. If the download is slow, place
                 cifar-10-python.tar.gz in $PROJECT_DIR/data/cifar10_raw/ yourself.

Produces under $PROJECT_DIR/data:
    cifar10/{train,val,test}/<class>/*.png      train 45k / val 5k (stratified, seeded) / test 10k
    cifar10_imb/train                           minority classes cut to --keep of their images (symlinks)
    cifar10_imb_os/train                        same, minority images repeated (symlinks) up to the majority count
    cifar10_imb*/val -> ../cifar10/val          validation stays balanced

The official test split is never used for training or checkpoint selection.
"""

import argparse
import os
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path

import common

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
ZIP_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/cifar10.zip"


def fetch_zip(raw: Path) -> Path:
    """Download + unpack the PNG-folder archive; returns the folder holding train/ and test/."""
    src = raw / "cifar10"
    if (src / "test").exists():
        return src
    raw.mkdir(parents=True, exist_ok=True)
    zpath = raw / "cifar10.zip"
    if not zpath.exists():
        print("downloading", ZIP_URL, flush=True)
        urllib.request.urlretrieve(ZIP_URL, zpath.with_suffix(".part"))
        zpath.with_suffix(".part").rename(zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(raw)
    return src


def fetch_torchvision(raw: Path, download_only: bool) -> Path | None:
    from torchvision.datasets import CIFAR10

    train = CIFAR10(root=raw, train=True, download=True)
    test = CIFAR10(root=raw, train=False, download=True)
    if download_only:
        return None
    assert train.classes == CLASSES, train.classes
    src = raw / "png"
    if not (src / "test").exists():
        for split, ds in (("train", train), ("test", test)):
            for c in CLASSES:
                (src / split / c).mkdir(parents=True, exist_ok=True)
            for i in range(len(ds)):
                img, y = ds[i]
                img.save(src / split / CLASSES[y] / f"{i:05d}.png")
    return src


def build_base(src: Path, out: Path, val_size: int, seed: int):
    """Move files from src/{train,test} into out/{train,val,test}; val is a stratified slice of train."""
    rng = random.Random(seed)
    for c in CLASSES:
        for s in ("train", "val", "test"):
            (out / s / c).mkdir(parents=True, exist_ok=True)
        files = sorted(os.listdir(src / "train" / c))
        val = set(rng.sample(files, val_size // len(CLASSES)))
        for f in files:
            os.rename(src / "train" / c / f, out / ("val" if f in val else "train") / c / f)
        for f in os.listdir(src / "test" / c):
            os.rename(src / "test" / c / f, out / "test" / c / f)
    for s in ("train", "val", "test"):
        print(s, {c: len(os.listdir(out / s / c)) for c in CLASSES}, flush=True)


def link_tree(src_train: Path, dst_train: Path, keep: dict, oversample_to=None, seed=0):
    """Symlink a subset of src_train; optionally repeat images so every class reaches `oversample_to`."""
    rng = random.Random(seed)
    for cls in CLASSES:
        files = sorted(os.listdir(src_train / cls))
        n_keep = max(1, round(len(files) * keep.get(cls, 1.0)))
        chosen = sorted(rng.sample(files, n_keep))
        (dst_train / cls).mkdir(parents=True, exist_ok=True)
        for f in chosen:
            os.symlink(src_train / cls / f, dst_train / cls / f)
        if oversample_to and n_keep < oversample_to:
            for i in range(oversample_to - n_keep):
                f = chosen[i % n_keep]
                os.symlink(src_train / cls / f, dst_train / cls / f"{Path(f).stem}__rep{i}{Path(f).suffix}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["zip", "torchvision"], default="zip")
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--val-size", type=int, default=5000)
    ap.add_argument("--minority", default="cat,deer,truck", help="classes to cut down in the imbalanced variants")
    ap.add_argument("--keep", type=float, default=0.05, help="fraction of a minority class kept (0.05 -> ~225 images)")
    ap.add_argument("--keep-raw", action="store_true", help="do not delete the downloaded archive afterwards")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    data = common.data_dir()
    raw = data / "cifar10_raw"
    out = data / "cifar10"

    if (out / "test").exists():
        print(f"{out} already built, skipping conversion")
    else:
        if a.source == "zip":
            src = fetch_zip(raw)
            if a.download_only:
                print("downloaded to", raw)
                return
        else:
            src = fetch_torchvision(raw, a.download_only)
            if a.download_only:
                print("downloaded to", raw)
                return
        build_base(src, out, a.val_size, a.seed)
        if not a.keep_raw:
            shutil.rmtree(raw, ignore_errors=True)

    if a.download_only:
        return
    minority = [c for c in a.minority.split(",") if c]
    assert set(minority) <= set(CLASSES), minority
    keep = {c: a.keep for c in minority}
    biggest = max(len(os.listdir(out / "train" / c)) for c in CLASSES)
    for name, os_to in (("cifar10_imb", None), ("cifar10_imb_os", biggest)):
        d = data / name
        if d.exists():
            print(f"{d} exists, skipping")
            continue
        link_tree(out / "train", d / "train", keep, oversample_to=os_to, seed=a.seed)
        os.symlink(out / "val", d / "val")
        print(name, {c: len(os.listdir(d / "train" / c)) for c in CLASSES})


if __name__ == "__main__":
    main()
