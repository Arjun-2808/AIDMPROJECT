#!/usr/bin/env python
"""Unpack datasets.zip into data/ and verify the layout.

    python scripts/setup_data.py --zip /path/to/datasets.zip

The zip is ~1.1 GB and is NOT in the repo (GitHub caps files at 100 MB).
Get it from the course Google Drive link in the project brief, or from whoever
already has it. Run this once on your machine and once on NSCC scratch.
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

EXPECTED = {
    "train/cat": 10000, "train/dog": 10000,
    "val/cat": 2500, "val/dog": 2500,
    "test": 500,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zip", default="datasets.zip")
    p.add_argument("--dest", default="data")
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args()

    dest = Path(args.dest)
    root = dest / "datasets"

    if not args.verify_only:
        src = Path(args.zip)
        if not src.exists():
            raise SystemExit(
                f"{src} not found.\n"
                "Download datasets.zip from the Google Drive link in the project brief\n"
                "(see data/README.md) and pass it with --zip."
            )
        dest.mkdir(parents=True, exist_ok=True)
        print(f"extracting {src} -> {dest}/ ...")
        with zipfile.ZipFile(src) as z:
            members = [m for m in z.namelist() if not m.startswith("__MACOSX")]
            for i, m in enumerate(members, 1):
                z.extract(m, dest)
                if i % 2000 == 0:
                    print(f"  {i}/{len(members)}")
        print("extracted.")

    if not root.exists():
        raise SystemExit(f"expected {root} after extraction, not found")

    print(f"\nverifying {root}")
    ok = True
    for rel, want in EXPECTED.items():
        d = root / rel
        if not d.is_dir():
            print(f"  MISSING  {rel}")
            ok = False
            continue
        n = sum(1 for f in d.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
        flag = "ok " if n == want else "!! "
        print(f"  {flag}{rel:12s} {n:6d} images (expected {want})")
        ok &= n == want

    print("\nlayout OK" if ok else "\nlayout does NOT match the brief -- check the zip")
    print(f"point configs at:  data.root: {root.as_posix()}")


if __name__ == "__main__":
    main()
