"""Score a trained run on a labelled split (val, or a dataset's labelled test split).

    python evaluate.py --run cd_s_ft --split val
    python evaluate.py --run c10_s_ft --split test    # CIFAR-10: the official 10k test split has labels

Writes runs/<run>/<split>_predictions.csv and prints accuracy / per-class recall / macro-F1.

Not every "test" folder has labels: CIFAR-10's does, but the Dogs-vs-Cats test set (data/catsdogs/test)
is the 500 unlabelled images from the brief, laid out flat (no class subfolders) for submission.csv,
not for accuracy. `DATASET_EVAL` in experiments.py is the source of truth for which splits are scored;
this script refuses `--split test` on a dataset where it isn't listed. Use predict.py for that split.
"""

import argparse
import csv

import numpy as np

import common
from experiments import DATASET_EVAL


def predict_probs(model, paths, imgsz, batch=128, device=None):
    """(N, nc) softmax probabilities, in the order of `paths`.

    One predict() call per batch, on purpose: a single call over the whole list makes Ultralytics hold memory that
    grows with the number of images (measured on a laptop: >3.5 GB for 256 images with the "l" model versus a flat
    1.2 GB with per-batch calls; the results are identical).
    """
    device = device or common.get_device()
    out = []
    for i in range(0, len(paths), batch):
        chunk = [str(p) for p in paths[i : i + batch]]
        for r in model.predict(source=chunk, imgsz=imgsz, batch=len(chunk), device=device, verbose=False):
            out.append(r.probs.data.float().cpu().numpy())
    return np.stack(out)


def metrics_from(probs, y_true, names):
    """Accuracy, per-class precision/recall/F1, macro-F1 and confusion matrix (rows = true)."""
    nc = len(names)
    y_pred = probs.argmax(1)
    cm = np.zeros((nc, nc), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return {
        "n": int(len(y_true)),
        "accuracy": float(tp.sum() / max(cm.sum(), 1)),
        "macro_f1": float(f1.mean()),
        "per_class": {names[i]: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]), "support": int(cm[i].sum())} for i in range(nc)},
        "confusion_matrix": {"labels": [names[i] for i in range(nc)], "rows_true_cols_pred": cm.tolist()},
    }


def save_predictions(path, paths, y_true, probs, names):
    y_pred = probs.argmax(1)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "true", "pred", "conf"] + [f"p_{names[i]}" for i in range(len(names))])
        for p, t, yp, pr in zip(paths, y_true, y_pred, probs):
            w.writerow([p, names[t], names[yp], f"{pr[yp]:.5f}"] + [f"{x:.5f}" for x in pr])


def evaluate_split(model, split_dir, imgsz, out_csv=None, batch=128, device=None):
    items = common.list_images(split_dir)
    names = {int(k): v for k, v in model.names.items()}
    inv = {v: k for k, v in names.items()}
    missing = {c for _, c in items} - set(inv)
    if missing:
        raise ValueError(f"folders {missing} in {split_dir} are not model classes {sorted(inv)}")
    paths = [p for p, _ in items]
    y_true = np.array([inv[c] for _, c in items])
    probs = predict_probs(model, paths, imgsz, batch, device)
    if out_csv is not None:
        save_predictions(out_csv, paths, y_true, probs, names)
    return metrics_from(probs, y_true, names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--weights", default="best.pt")
    ap.add_argument("--batch", type=int, default=32, help="inference batch size; lower = less memory (default 32)")
    ap.add_argument("--device", default=None, help="CUDA device (default: cpu)")
    a = ap.parse_args()

    run = common.runs_dir() / a.run
    model = common.YOLO(str(run / "weights" / a.weights))
    cfg = common.run_config(run, model)
    allowed = DATASET_EVAL.get(cfg["dataset"], ["val"])
    if a.split not in allowed:
        raise SystemExit(
            f"'{cfg['dataset']}' has no ground-truth labels for split '{a.split}' (only {allowed} do), so it "
            f"can't be scored.\nUse predict.py instead: python predict.py --run {a.run}"
        )
    m = evaluate_split(model, common.eval_dir(cfg["dataset"], a.split), cfg["imgsz"], run / f"{a.split}_predictions.csv", batch=a.batch, device=a.device)
    print(f"{a.run} [{a.split}] acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f}")
    for c, v in m["per_class"].items():
        print(f"  {c:<12} recall={v['recall']:.4f} precision={v['precision']:.4f} n={v['support']}")


if __name__ == "__main__":
    main()
