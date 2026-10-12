"""Error analysis of the best Dogs vs. Cats model (part e).

Writes to <results>/analysis/:
  submission.csv            test predictions of the selected run (1 = dog, 0 = cat)
  test_sheet_*.png          all test images, least confident first, to eyeball mistakes
  val_errors.png            misclassified validation images, most confident mistakes first
  attn_*.png                ViT attention-rollout maps (where the model looks)
  summary.json
"""
import argparse
import csv
import math
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms as T

from data import BICUBIC, DVC_CLASSES
from engine import get_device, load_json, save_json
from models import build_model


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", default=None, help="default: finished run with model.pt and best val acc")
    p.add_argument("--data-root", default=os.environ.get("IE4483_DATA", "data/datasets"))
    p.add_argument("--out-root", default=os.environ.get("IE4483_OUT", "runs"))
    p.add_argument("--results", default=os.environ.get("IE4483_RESULTS", "results"))
    p.add_argument("--n-show", type=int, default=8, help="images per attention panel")
    p.add_argument("--thumb", type=int, default=160)
    return p.parse_args(argv)


def pick_run(out_root):
    runs = [p.parent for p in (Path(out_root) / "dvc").glob("*/model.pt") if (p.parent / "metrics.json").exists()]
    if not runs:
        raise SystemExit(f"no finished run with model.pt under {Path(out_root) / 'dvc'}")
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


def _collect_attention(vit, x, with_grad):
    """Run the ViT once, keeping every layer's attention matrix [B, heads, N, N] (and its gradient
    w.r.t. the predicted-class logit when with_grad)."""
    attns, hooks = [], []

    def keep(module, inputs, out):
        if with_grad:
            out.retain_grad()
        attns.append(out)

    for blk in vit.blocks:
        blk.attn.fused_attn = False  # the fused kernel never materialises the attention matrix
        hooks.append(blk.attn.attn_drop.register_forward_hook(keep))
    try:
        with torch.set_grad_enabled(with_grad):
            logits = vit(x)
            if with_grad:
                vit.zero_grad(set_to_none=True)
                logits.gather(1, logits.argmax(1, keepdim=True)).sum().backward()
    finally:
        for h in hooks:
            h.remove()
    return attns


def _cls_to_patches(vit, rel):
    mask = rel[:, 0, vit.num_prefix_tokens:]
    side = int(round(mask.shape[-1] ** 0.5))
    mask = mask.reshape(-1, side, side)
    return (mask / mask.amax((1, 2), keepdim=True).clamp_min(1e-12)).detach().cpu().numpy()


def attention_rollout(vit, x):
    """Abnar & Zuidema (2020): multiply head-averaged attention (+ identity) through all layers and
    read the CLS row - how much each patch flows into the class token. Class-agnostic."""
    attns = _collect_attention(vit, x, with_grad=False)
    eye = torch.eye(attns[0].shape[-1], device=x.device)
    roll = eye.expand(x.shape[0], -1, -1)
    for att in attns:
        att = att.float().mean(1) + eye
        roll = (att / att.sum(-1, keepdim=True)) @ roll
    return _cls_to_patches(vit, roll)


def class_relevance(vit, x):
    """Chefer, Gur & Wolf (2021), gradient-weighted rollout: weight each attention map by its
    gradient w.r.t. the predicted class, keep the positive part, and accumulate R += A_bar @ R.
    Shows the evidence *for the predicted class*, which is what we want for error analysis."""
    attns = _collect_attention(vit, x, with_grad=True)
    rel = torch.eye(attns[0].shape[-1], device=x.device).expand(x.shape[0], -1, -1).clone()
    for att in attns:
        a_bar = (att.grad * att).clamp_min(0).mean(1).detach().float()
        rel = rel + a_bar @ rel
    return _cls_to_patches(vit, rel)


def overlay(img, mask):
    m = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR)) / 255.0
    heat = Image.fromarray((colormaps["inferno"](m)[..., :3] * 255).astype(np.uint8))
    return Image.blend(img, heat, 0.55)


def attention_panel(model, view_tf, norm_tf, paths, captions, device, colors=None):
    """Per image: original | attention rollout | class-specific relevance."""
    views = [view_tf(Image.open(p).convert("RGB")) for p in paths]
    x = torch.stack([norm_tf(v) for v in views]).to(device)
    rollout, relevance = attention_rollout(model, x), class_relevance(model, x)
    tiles = []
    for v, ro, rel, c, col in zip(views, rollout, relevance, captions, colors or ["black"] * len(views)):
        row = Image.new("RGB", (3 * v.width, v.height))
        for k, im in enumerate([v, overlay(v, ro), overlay(v, rel)]):
            row.paste(im, (k * v.width, 0))
        tiles.append(captioned(row, f"{c}   (image | attention rollout | class relevance)", col))
    return grid(tiles, cols=2)


def main(argv=None):
    a = parse_args(argv)
    run = Path(a.run_dir) if a.run_dir else pick_run(a.out_root)
    metrics = load_json(run / "metrics.json")
    cfg = metrics["config"]
    res = Path(a.results) / "analysis"
    res.mkdir(parents=True, exist_ok=True)
    root = Path(a.data_root)
    print(f"analysing {run.name} (best val acc {metrics['best_val_acc']:.4f})")
    shutil.copy(run / "submission.csv", res / "submission.csv")

    thumb_tf = T.Compose([T.Resize(a.thumb, interpolation=BICUBIC), T.CenterCrop(a.thumb)])

    # Test set: no labels, so lay out every image by confidence for manual inspection.
    with open(run / "test_probs.csv") as f:
        test = [(int(r["id"]), float(r["p_dog"])) for r in csv.DictReader(f)]
    test.sort(key=lambda t: abs(t[1] - 0.5))
    for k in range(0, len(test), 100):
        tiles = [captioned(thumb_tf(Image.open(root / "test" / f"{i}.jpg").convert("RGB")), f"#{i} {label_of(p)}")
                 for i, p in test[k:k + 100]]
        grid(tiles, cols=10).save(res / f"test_sheet_{k // 100 + 1}.png")

    # Validation set: labels known, so show the actual mistakes.
    with open(run / "val_preds.csv") as f:
        val = [(r["path"], int(r["label"]), float(r["p_dog"])) for r in csv.DictReader(f)]
    errors = sorted([v for v in val if int(v[2] > 0.5) != v[1]], key=lambda v: -abs(v[2] - 0.5))
    if errors:
        tiles = [captioned(thumb_tf(Image.open(root / p).convert("RGB")),
                           f"{DVC_CLASSES[y]} -> {label_of(pd)}", "firebrick") for p, y, pd in errors[:40]]
        grid(tiles, cols=8).save(res / "val_errors.png")

    if cfg["model"].startswith("vit"):
        device = get_device()
        model, dcfg = build_model(cfg["model"], 2, cfg["img_size"], pretrained=False)
        model.load_state_dict(torch.load(run / "model.pt", map_location="cpu")["state_dict"])
        model.to(device).eval()
        size = cfg["img_size"]  # same resize + centre crop as evaluation
        view_tf = T.Compose([T.Resize(int(round(size / dcfg["crop_pct"])), interpolation=BICUBIC), T.CenterCrop(size)])
        norm_tf = T.Compose([T.ToTensor(), T.Normalize(dcfg["mean"], dcfg["std"])])
        n = a.n_show
        by_p = sorted(test, key=lambda t: t[1])
        confident = by_p[-(n // 2):][::-1] + by_p[:n // 2]  # most confident dogs, then cats
        for name, picks in [("attn_test_confident", confident), ("attn_test_uncertain", test[:n])]:
            attention_panel(model, view_tf, norm_tf, [root / "test" / f"{i}.jpg" for i, _ in picks],
                            [f"#{i} {label_of(p)}" for i, p in picks], device).save(res / f"{name}.png")
        if errors:
            picks = errors[:n]
            attention_panel(model, view_tf, norm_tf, [root / p for p, _, _ in picks],
                            [f"{DVC_CLASSES[y]} -> {label_of(pd)}" for _, y, pd in picks], device,
                            ["firebrick"] * len(picks)).save(res / "attn_val_errors.png")

    p_test = np.array([p for _, p in test])
    save_json(dict(
        run=str(run), best_val_acc=metrics["best_val_acc"], n_val=len(val), n_val_errors=len(errors),
        val_cat_predicted_dog=sum(1 for _, y, _ in errors if y == 0),
        val_dog_predicted_cat=sum(1 for _, y, _ in errors if y == 1),
        test_pred_dog=int((p_test > 0.5).sum()), test_pred_cat=int((p_test <= 0.5).sum()),
        test_uncertain_ids=[i for i, p in test if abs(p - 0.5) < 0.3],  # confidence below 0.8
        val_errors=[dict(path=p, label=DVC_CLASSES[y], p_dog=pd) for p, y, pd in errors],
    ), res / "summary.json")
    print(f"val errors {len(errors)}/{len(val)}; outputs in {res}")


if __name__ == "__main__":
    main()
