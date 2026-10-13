"""Zero-shot baseline: ImageNet-1k classifier, no training.

Sum the softmax mass over the ImageNet dog classes (idx 151-268) and the domestic-cat classes
(idx 281-285: tabby, tiger cat, Persian, Siamese, Egyptian cat) and pick the larger.
Caveat for the report: ImageNet already contains dog/cat photos, so this is a pretrained-features
reference, not a from-scratch result.
"""

import numpy as np

import common
from evaluate import metrics_from, predict_probs, save_predictions

DOG_IDX = range(151, 269)
CAT_IDX = range(281, 286)


def run(exp, out):
    from train import env_info  # local import: train imports this module lazily

    model = common.YOLO(f"{exp.model}.pt")
    names_in = model.names
    assert len(names_in) == 1000, "expected an ImageNet-1k classifier"
    print("[zeroshot] cat idx ->", [names_in[i] for i in CAT_IDX], "| dog idx ends", names_in[151], "...", names_in[268], flush=True)

    items = common.list_images(common.data_dir() / exp.dataset / "val")
    paths = [p for p, _ in items]
    names = {0: "cat", 1: "dog"}
    y_true = np.array([{"cat": 0, "dog": 1}[c] for _, c in items])

    p = predict_probs(model, paths, exp.imgsz)
    two = np.stack([p[:, list(CAT_IDX)].sum(1), p[:, list(DOG_IDX)].sum(1)], axis=1)
    two = two / np.maximum(two.sum(1, keepdims=True), 1e-12)

    out.mkdir(parents=True, exist_ok=True)
    save_predictions(out / "val_predictions.csv", paths, y_true, two, names)
    m = metrics_from(two, y_true, names)
    res = {
        "name": exp.name,
        "config": exp.to_dict(),
        "status": "done",
        **env_info(),
        "splits": {"val": m},
        "val_accuracy": m["accuracy"],
        "val_macro_f1": m["macro_f1"],
    }
    common.write_json(out / "result.json", res)
    print(f"[done] {exp.name} val_acc={m['accuracy']:.4f}", flush=True)
    return res
