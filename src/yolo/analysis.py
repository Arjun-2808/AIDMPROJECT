"""Contact sheets for the error analysis in question (e).

    python analysis.py errors --run cd_s_ft            # most confident val mistakes
    python analysis.py test --probs $PROJECT_DIR/submission_probs.csv
        -> test_least_confident.png / test_most_confident.png next to the probs file

The test set has no labels, so `test` shows the images the model is least / most sure about; look at
them and label the errors by eye for the report.
"""

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw

import common


def montage(items, out, cols=6, tile=200):
    """items: [(image_path, caption)] -> one PNG contact sheet."""
    rows = (len(items) + cols - 1) // cols
    cap_h = 26
    sheet = Image.new("RGB", (cols * tile, rows * (tile + cap_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (path, cap) in enumerate(items):
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((tile, tile))
            x, y = (i % cols) * tile, (i // cols) * (tile + cap_h)
            sheet.paste(im, (x + (tile - im.width) // 2, y + (tile - im.height) // 2))
        draw.text((x + 4, y + tile + 2), cap, fill="black")
    sheet.save(out)
    print("wrote", out)


def cmd_errors(a):
    run = common.runs_dir() / a.run
    with open(run / f"{a.split}_predictions.csv") as f:
        rows = list(csv.DictReader(f))
    wrong = sorted((r for r in rows if r["true"] != r["pred"]), key=lambda r: -float(r["conf"]))
    print(f"{len(wrong)} / {len(rows)} misclassified on {a.split}")
    pairs = {}
    for r in wrong:
        pairs[(r["true"], r["pred"])] = pairs.get((r["true"], r["pred"]), 0) + 1
    for (t, p), n in sorted(pairs.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  true {t:<12} -> predicted {p:<12} x{n}")
    if wrong:
        montage([(r["path"], f"T:{r['true']} P:{r['pred']} {float(r['conf']):.2f}") for r in wrong[: a.n]], run / f"errors_{a.split}.png")


def cmd_test(a):
    probs = Path(a.probs)
    with open(probs) as f:
        rows = sorted(csv.DictReader(f), key=lambda r: float(r["conf"]))
    cap = lambda r: f"id {r['id']}  cat {float(r['p_cat']):.2f} dog {float(r['p_dog']):.2f}"  # noqa: E731
    montage([(r["path"], cap(r)) for r in rows[: a.n]], probs.with_name("test_least_confident.png"))
    montage([(r["path"], cap(r)) for r in rows[-a.n :]], probs.with_name("test_most_confident.png"))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("errors")
    e.add_argument("--run", required=True)
    e.add_argument("--split", default="val")
    e.add_argument("--n", type=int, default=24)
    e.set_defaults(fn=cmd_errors)
    t = sub.add_parser("test")
    t.add_argument("--probs", required=True)
    t.add_argument("--n", type=int, default=24)
    t.set_defaults(fn=cmd_test)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
