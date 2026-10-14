"""Arrange and sanity-check the Dogs-vs-Cats data (CPU-only job, not the login node).

Expects the folder from the brief (train/{cat,dog}, val/{cat,dog}, test/*.jpg) uploaded to
$PROJECT_DIR/data/catsdogs_raw and produces

    $PROJECT_DIR/data/catsdogs/{train,val}/{cat,dog}/*.jpg
    $PROJECT_DIR/data/catsdogs/test/*.jpg        (flat, no class subfolders: unlabelled, for submission.csv only)

Directories are moved (same filesystem, instant), not copied. Every image is opened once to catch
corrupt files; nothing is deleted, problems are listed in prep_report.json.
"""

import argparse
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

import common


def check(path):
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            return str(path), im.size, im.mode, None
    except Exception as e:  # noqa: BLE001
        return str(path), None, None, repr(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=None, help="default: $PROJECT_DIR/data/catsdogs_raw")
    a = ap.parse_args()

    data = common.data_dir()
    raw = Path(a.raw) if a.raw else data / "catsdogs_raw"
    dst = data / "catsdogs"
    dst.mkdir(parents=True, exist_ok=True)
    dst_test = dst / "test"  # flat, alongside train/ and val/, but not an ImageFolder split (no class subfolders)

    for split in ("train", "val"):
        if (dst / split).exists():
            print(f"{dst / split} exists, leaving as is")
        elif (raw / split).exists():
            (raw / split).rename(dst / split)
        else:
            raise SystemExit(f"missing {raw / split}")
    if dst_test.exists():
        print(f"{dst_test} exists, leaving as is")
    elif (raw / "test").exists():
        (raw / "test").rename(dst_test)
    else:
        raise SystemExit(f"missing {raw / 'test'}")

    # Fail here, not hours later inside training, if the upload was incomplete.
    brief = {"train/cat": 10000, "train/dog": 10000, "val/cat": 2500, "val/dog": 2500, "test": 500}  # counts in the brief
    n_img = {f"{s}/{c}": sum(1 for _, k in common.list_images(dst / s) if k == c) for s in ("train", "val") for c in ("cat", "dog")}
    n_img["test"] = sum(1 for p in dst_test.iterdir() if p.suffix.lower() in common.IMG_EXT)
    print("images found:", n_img, flush=True)
    empty = [k for k, n in n_img.items() if n == 0]
    if empty:
        raise SystemExit(
            f"ERROR: no images in {empty}. The upload is probably incomplete. Prep never re-reads {raw} once the folders "
            f"exist, so re-upload the missing split straight into {dst}/<split>/ (or {dst_test}/ for test)."
        )
    short = {k: (n, brief[k]) for k, n in n_img.items() if n != brief[k]}
    if short:
        print(f"WARNING: counts differ from the brief (found, expected): {short}. Fine if you deliberately use a subset.")

    files = [p for s in ("train", "val") for p, _ in common.list_images(dst / s)]
    files += [p for p in sorted(dst_test.iterdir()) if p.suffix.lower() in common.IMG_EXT]
    workers = common.num_workers()
    print(f"checking {len(files)} images with {workers} process(es)...", flush=True)
    with Pool(workers) as pool:
        results = pool.map(check, files, chunksize=256)

    bad = [(p, err) for p, _, _, err in results if err]
    sizes = [s for _, s, _, err in results if not err]
    report = {
        "counts": {f"{s}/{c}": len(list((dst / s / c).iterdir())) for s in ("train", "val") for c in ("cat", "dog")},
        "test_images": len(list(dst_test.iterdir())),
        "corrupt": bad,
        "min_side": min(min(s) for s in sizes),
        "max_side": max(max(s) for s in sizes),
        "modes": sorted({m for _, _, m, err in results if not err}),
    }
    common.write_json(dst / "prep_report.json", report)
    print(report)
    if bad:
        print(f"WARNING: {len(bad)} unreadable image(s); Ultralytics will skip them during training.")


if __name__ == "__main__":
    main()
