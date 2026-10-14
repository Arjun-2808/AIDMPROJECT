"""Error analysis of the best Dogs vs. Cats run, in the ViT track's format (src/vit/analyze_dvc.py).

Writes to <results>/analysis/:
  submission.csv            test predictions of the selected run (1 = dog, 0 = cat)
  test_sheet_*.png          all test images, least confident first, to eyeball mistakes
  val_errors.png            misclassified validation images, most confident mistakes first
  gradcam_*.png             Grad-CAM maps on layer4 (where the ResNet looks); the counterpart of
                            the ViT track's attention maps
  summary.json
"""
import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms as T

PROJECT = Path(__file__).parents[1]
REPO = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from aidm_project.config import Config
from aidm_project.data import DVC_CLASSES, MEAN, STD
from aidm_project.device import get_device
from aidm_project.gradcam import grad_cam
from aidm_project.models import build_model


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", default=None, help="default: finished run of --name with the best val acc")
    p.add_argument("--name", default="best", help="baseline to analyse when --run-dir is not given")
    p.add_argument("--data-root", default=str(PROJECT / "data" / "dogs_cats" / "datasets"))
    p.add_argument("--out-root", default=str(PROJECT / "outputs" / "runs"))
    p.add_argument("--results", default=str(REPO / "results" / "resnet"))
    p.add_argument("--n-show", type=int, default=8, help="images per Grad-CAM panel")
    p.add_argument("--thumb", type=int, default=160)
    return p.parse_args(argv)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pick_run(out_root, name):
    runs = [p.parent for p in (Path(out_root) / "dvc").glob("*/model.pt")
            if (p.parent / "metrics.json").exists() and load_json(p.parent / "metrics.json")["config"]["name"] == name]
    if not runs:
        raise SystemExit(f"no finished '{name}' run with model.pt under {Path(out_root) / 'dvc'}")
    return max(runs, key=lambda r: load_json(r / "metrics.json")["best_val_acc"])


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


FONT = _font(13)


def captioned(img, caption, color="black"):
    canvas = Image.new("RGB", (img.width, img.height + 20), "white")
    canvas.paste(img, (0, 0))
    ImageDraw.Draw(canvas).text((4, img.height + 3), caption, fill=color, font=FONT)
    return canvas


def grid(tiles, cols, pad=4):
    w, h = tiles[0].size
    rows = math.ceil(len(tiles) / cols)
    sheet = Image.new("RGB", (cols * (w + pad) + pad, rows * (h + pad) + pad), (225, 225, 225))
    for i, t in enumerate(tiles):
        sheet.paste(t, (pad + (i % cols) * (w + pad), pad + (i // cols) * (h + pad)))
    return sheet


def label_of(p_dog):
    return f"{DVC_CLASSES[int(p_dog > 0.5)]} {max(p_dog, 1 - p_dog):.2f}"


def overlay(img, mask):
    m = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR)) / 255.0
    heat = Image.fromarray((colormaps["inferno"](m)[..., :3] * 255).astype(np.uint8))
    return Image.blend(img, heat, 0.55)


def gradcam_panel(model, view_tf, norm_tf, paths, captions, device, colors=None):
    """Per image: original | Grad-CAM for the predicted class."""
    views = [view_tf(Image.open(p).convert("RGB")) for p in paths]
    x = torch.stack([norm_tf(v) for v in views]).to(device)
    maps = grad_cam(model, x).cpu().numpy()
    tiles = []
    for v, cam, c, col in zip(views, maps, captions, colors or ["black"] * len(views)):
        row = Image.new("RGB", (2 * v.width, v.height))
        row.paste(v, (0, 0))
        row.paste(overlay(v, cam), (v.width, 0))
        tiles.append(captioned(row, f"{c}   (image | Grad-CAM)", col))
    return grid(tiles, cols=2)


def main(argv=None):
    a = parse_args(argv)
    run = Path(a.run_dir) if a.run_dir else pick_run(a.out_root, a.name)
    metrics = load_json(run / "metrics.json")
    res = Path(a.results) / "analysis"
    res.mkdir(parents=True, exist_ok=True)
    root = Path(a.data_root)
    print(f"analysing {run.name} (best val acc {metrics['best_val_acc']:.4f})")
    shutil.copy(run / "submission.csv", res / "submission.csv")
    thumb_tf = T.Compose([T.Resize(a.thumb), T.CenterCrop(a.thumb)])

    # Test set: no labels, so lay out every image by confidence for manual inspection.
    with open(run / "test_probs.csv", encoding="utf-8") as f:
        test = [(int(r["id"]), float(r["p_dog"])) for r in csv.DictReader(f)]
    test_paths = {int(p.stem): p for p in (root / "test").rglob("*") if p.stem.isdigit()}
    test.sort(key=lambda t: abs(t[1] - 0.5))
    for k in range(0, len(test), 100):
        tiles = [captioned(thumb_tf(Image.open(test_paths[i]).convert("RGB")), f"#{i} {label_of(p)}")
                 for i, p in test[k:k + 100]]
        grid(tiles, cols=10).save(res / f"test_sheet_{k // 100 + 1}.png")

    # Validation set: labels known, so show the actual mistakes.
    with open(run / "val_preds.csv", encoding="utf-8") as f:
        val = [(r["path"], int(r["label"]), float(r["p_dog"])) for r in csv.DictReader(f)]
    errors = sorted([v for v in val if int(v[2] > 0.5) != v[1]], key=lambda v: -abs(v[2] - 0.5))
    if errors:
        tiles = [captioned(thumb_tf(Image.open(root / p).convert("RGB")),
                           f"{DVC_CLASSES[y]} -> {label_of(pd)}", "firebrick") for p, y, pd in errors[:40]]
        grid(tiles, cols=8).save(res / "val_errors.png")

    cfg = Config(**metrics["config"])
    device = get_device()
    model = build_model(cfg.model, cfg.num_classes, False, cfg.dropout, cfg.fine_tuning)
    model.load_state_dict(torch.load(run / "model.pt", map_location="cpu", weights_only=False)["model"])
    model.to(device).eval()
    view_tf = T.Resize((cfg.image_size, cfg.image_size))  # same resize as evaluation
    norm_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    n = a.n_show
    by_p = sorted(test, key=lambda t: t[1])
    confident = by_p[-(n // 2):][::-1] + by_p[:n // 2]  # most confident dogs, then cats
    for name, picks in [("gradcam_test_confident", confident), ("gradcam_test_uncertain", test[:n])]:
        gradcam_panel(model, view_tf, norm_tf, [test_paths[i] for i, _ in picks],
                      [f"#{i} {label_of(p)}" for i, p in picks], device).save(res / f"{name}.png")
    if errors:
        picks = errors[:n]
        gradcam_panel(model, view_tf, norm_tf, [root / p for p, _, _ in picks],
                      [f"{DVC_CLASSES[y]} -> {label_of(pd)}" for _, y, pd in picks], device,
                      ["firebrick"] * len(picks)).save(res / "gradcam_val_errors.png")

    p_test = np.array([p for _, p in test])
    summary = dict(
        run=str(run), best_val_acc=metrics["best_val_acc"], n_val=len(val), n_val_errors=len(errors),
        val_cat_predicted_dog=sum(1 for _, y, _ in errors if y == 0),
        val_dog_predicted_cat=sum(1 for _, y, _ in errors if y == 1),
        test_pred_dog=int((p_test > 0.5).sum()), test_pred_cat=int((p_test <= 0.5).sum()),
        test_uncertain_ids=[i for i, p in test if abs(p - 0.5) < 0.3],  # confidence below 0.8
        val_errors=[dict(path=p, label=DVC_CLASSES[y], p_dog=pd) for p, y, pd in errors],
    )
    (res / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"val errors {len(errors)}/{len(val)}; outputs in {res}")


if __name__ == "__main__":
    main()
