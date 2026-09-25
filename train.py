#!/usr/bin/env python
"""Entry point. Reads a YAML config, trains, writes logs under experiments/<owner>/.

    python train.py --config configs/cnn_resnet18_sgd.yaml --owner arjun
    python train.py --config configs/vit_s16_adamw.yaml --owner member2 \
        --set optim.lr=1e-4 data.train_per_class=2000 --note "lr sweep"

Anything in the YAML can be overridden with --set a.b=value (no quoting needed
for numbers/bools). The override is recorded in config.json, so the run stays
reproducible.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.data import build_dataloaders
from src.engine import fit
from src.models import available, build_model
from src.optim import build_criterion, build_optimizer, build_scheduler
from src.tracking import RunLogger, load_config
from src.utils import count_params, describe_device, get_device, make_run_dir, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Dogs vs Cats trainer")
    p.add_argument("--config", required=True, help="path to a YAML in configs/")
    p.add_argument("--owner", required=True, help="your folder under experiments/")
    p.add_argument("--run-name", default=None, help="defaults to the config stem")
    p.add_argument("--note", default="", help="one line, lands in the leaderboard")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL",
                   help="dotted-path config overrides")
    p.add_argument("--list-models", action="store_true")
    return p.parse_args()


def apply_overrides(cfg: dict, pairs: list[str]) -> dict:
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--set expects KEY=VAL, got {pair!r}")
        key, raw = pair.split("=", 1)
        try:
            val = json.loads(raw)          # numbers, true/false, null, lists
        except json.JSONDecodeError:
            val = raw                      # plain string
        node = cfg
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = val
    return cfg


def main():
    args = parse_args()
    if args.list_models:
        print("\n".join(available()))
        return

    cfg = apply_overrides(load_config(args.config), args.set)
    run_name = args.run_name or Path(args.config).stem

    seed = cfg.get("seed", 42)
    set_seed(seed, deterministic=cfg.get("deterministic", True))

    run_dir = make_run_dir(cfg.get("experiments_root", "experiments"), args.owner, run_name)
    logger = RunLogger(run_dir, cfg, args.owner, run_name)

    device = get_device(cfg.get("device", "auto"))
    logger.print(f"run dir : {run_dir}")
    logger.print(f"device  : {describe_device(device)}")
    logger.print(f"seed    : {seed}")

    d = cfg["data"]
    train_loader, val_loader = build_dataloaders(
        data_root=d["root"],
        img_size=d.get("img_size", 224),
        batch_size=d.get("batch_size", 32),
        augment=d.get("augment", "standard"),
        train_per_class=d.get("train_per_class"),
        val_per_class=d.get("val_per_class"),
        num_workers=d.get("num_workers", 4),
        seed=seed,
        pin_memory=(device.type == "cuda"),
    )
    logger.print(f"data    : {len(train_loader.dataset)} train / "
                 f"{len(val_loader.dataset)} val  @ {d.get('img_size', 224)}px "
                 f"(augment={d.get('augment', 'standard')})")

    m = cfg["model"]
    model = build_model(**m).to(device)
    total, trainable = count_params(model)
    logger.print(f"model   : {m['name']}  {total/1e6:.1f}M params "
                 f"({trainable/1e6:.1f}M trainable)")

    o = cfg["optim"]
    optimizer = build_optimizer(
        model, name=o.get("name", "adamw"), lr=o.get("lr", 3e-4),
        weight_decay=o.get("weight_decay", 1e-4), momentum=o.get("momentum", 0.9),
        betas=tuple(o.get("betas", (0.9, 0.999))),
        no_decay_norm_bias=o.get("no_decay_norm_bias", True),
    )
    epochs = cfg.get("epochs", 10)
    scheduler, granularity = build_scheduler(
        optimizer, name=o.get("scheduler", "cosine"), epochs=epochs,
        steps_per_epoch=len(train_loader), warmup_epochs=o.get("warmup_epochs", 0),
        min_lr=o.get("min_lr", 1e-6), step_size=o.get("step_size", 5),
        gamma=o.get("gamma", 0.1), milestones=o.get("milestones"),
        max_lr=o.get("max_lr"),
    )
    criterion = build_criterion(o.get("loss", "ce"),
                                label_smoothing=o.get("label_smoothing", 0.0))
    logger.print(f"optim   : {o.get('name','adamw')} lr={o.get('lr',3e-4)} "
                 f"wd={o.get('weight_decay',1e-4)} sched={o.get('scheduler','cosine')} "
                 f"loss={o.get('loss','ce')}")

    use_amp = cfg.get("amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    best = fit(
        model, train_loader, val_loader, criterion, optimizer, device, epochs,
        scheduler=scheduler, sched_granularity=granularity, scaler=scaler,
        logger=logger, ckpt_path=run_dir / "best.pt",
        early_stop_patience=cfg.get("early_stop_patience"),
        grad_clip=o.get("grad_clip"),
    )

    summary = logger.finish(best, notes=args.note)
    print(f"\nbest val acc {summary['best_val_acc']:.4f} @ epoch {summary['best_epoch']}")
    print(f"logs -> {run_dir}")
    print("leaderboard row appended to results/leaderboard.md -- commit it.")


if __name__ == "__main__":
    main()
