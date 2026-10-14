"""Classify the 500 unlabelled test images and write submission.csv (id, label; 1 = dog, 0 = cat).

    python predict.py --run cd_s_ft
    python predict.py --run cd_s_ft --run cd_m_ft --out $PROJECT_DIR/submission_ens.csv   # average probabilities

Also writes <out stem>_probs.csv (id, p_cat, p_dog, conf) for the error analysis in question (e).
Check the header of the sample submission.csv from the brief and adjust --id-col / --label-col if
it differs.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

import common
from evaluate import predict_probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="experiment name(s); several = average probs")
    ap.add_argument("--out", default=None)
    ap.add_argument("--test-dir", default=None)
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--batch", type=int, default=32, help="inference batch size; lower = less memory (default 32)")
    ap.add_argument("--device", default=None, help="CUDA device (default: auto)")
    a = ap.parse_args()

    test_dir = Path(a.test_dir) if a.test_dir else common.data_dir() / "catsdogs" / "test"
    if not test_dir.is_dir():
        raise SystemExit(f"Test folder not found: {test_dir}\nCopy it there first (README step 5b: cp -cR ../../datasets/test $PROJECT_DIR/data/catsdogs/test).")
    files = sorted((p for p in test_dir.iterdir() if p.suffix.lower() in common.IMG_EXT), key=lambda p: (len(p.stem), p.stem))
    ids = [int(p.stem) if p.stem.isdigit() else p.stem for p in files]
    if len(files) != 500:
        print(f"WARNING: expected 500 test images, found {len(files)}")

    all_probs = []
    for name in a.run:
        run = common.runs_dir() / name
        model = common.YOLO(str(run / "weights" / "best.pt"))
        cfg = common.run_config(run, model)
        assert {int(k): v for k, v in model.names.items()} == {0: "cat", 1: "dog"}, model.names
        all_probs.append(predict_probs(model, files, cfg["imgsz"], batch=a.batch, device=a.device))
        print(f"predicted with {name} (imgsz={cfg['imgsz']})", flush=True)
    probs = np.mean(all_probs, axis=0)
    label = probs.argmax(1)  # 0 = cat, 1 = dog (alphabetical folder order)

    out = Path(a.out) if a.out else common.project_dir() / "submission.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([a.id_col, a.label_col])
        w.writerows(zip(ids, label.tolist()))
    with open(out.with_name(out.stem + "_probs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([a.id_col, "path", "p_cat", "p_dog", "conf"])
        for i, p, pr in zip(ids, files, probs):
            w.writerow([i, p, f"{pr[0]:.5f}", f"{pr[1]:.5f}", f"{pr.max():.5f}"])
    print(f"wrote {out}  ({int(label.sum())} dog / {int((1 - label).sum())} cat)")


if __name__ == "__main__":
    main()
