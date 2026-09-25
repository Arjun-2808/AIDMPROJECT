#!/usr/bin/env python
"""Write submission.csv from a trained checkpoint.

    python predict.py --run experiments/arjun/20260925-...._cnn_resnet18_sgd

Output format required by the brief: two columns, id and label (1=dog, 0=cat).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from src.data import build_test_loader
from src.models import build_model
from src.utils import get_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="a run dir containing config.json + best.pt")
    p.add_argument("--ckpt", default=None, help="override checkpoint path")
    p.add_argument("--out", default="results/submission.csv")
    p.add_argument("--tta", action="store_true", help="average logits with a horizontal flip")
    args = p.parse_args()

    run = Path(args.run)
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    ckpt_path = Path(args.ckpt) if args.ckpt else run / "best.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"no checkpoint at {ckpt_path}")

    device = get_device(cfg.get("device", "auto"))
    model = build_model(**cfg["model"]).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    print(f"loaded {ckpt_path}  (val acc {state.get('val_acc', float('nan')):.4f})")

    d = cfg["data"]
    loader = build_test_loader(d["root"], d.get("img_size", 224),
                               batch_size=d.get("batch_size", 32) * 2,
                               num_workers=d.get("num_workers", 4))

    rows = []
    with torch.no_grad():
        for x, ids in loader:
            x = x.to(device)
            logits = model(x)
            if args.tta:
                logits = logits + model(torch.flip(x, dims=[3]))
            preds = logits.argmax(1).cpu().tolist()
            rows.extend(zip(ids.tolist(), preds))

    rows.sort(key=lambda r: r[0])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        w.writerows(rows)

    dogs = sum(r[1] for r in rows)
    print(f"wrote {out}  ({len(rows)} rows: {dogs} dog / {len(rows) - dogs} cat)")


if __name__ == "__main__":
    main()
