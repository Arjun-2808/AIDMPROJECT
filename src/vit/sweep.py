"""Experiment grid. Each PBS array index runs one config for all seeds (finished runs are skipped).

    python sweep.py dvc --count          # number of configs (array size)
    python sweep.py dvc --list
    python sweep.py dvc --index 3        # run config 3 for seeds 0,1,2
    python sweep.py dvc --index 0 --seeds 0 -- --limit 64 --epochs 1   # extra args go to the trainer
"""
import argparse
import sys
import traceback

SEEDS = [0, 1, 2]

# ---------------------------------------------------------------- Dogs vs. Cats (parts b-f)
VIT = dict(model="vit_b", mode="finetune", n_per_class=0, aug="basic", img_size=224, epochs=5, lr=2e-5)
CNN = dict(model="simple_cnn", mode="finetune", n_per_class=0, aug="basic", img_size=224, epochs=30, lr=1e-3)
SIZE_EPOCHS = {500: 20, 1000: 15, 2500: 8, 5000: 6}  # smaller subsets get more epochs (similar step counts)

DVC = [
    # model choice, all 10k images per class
    {**VIT, "save_ckpt": True},                      # main model -> submission.csv
    {**VIT, "model": "vit_s"},
    {**VIT, "model": "resnet50", "lr": 1e-4},
    {**VIT, "mode": "linprobe", "lr": 1e-3},
    CNN,
    # amount of training data
    *({**VIT, "n_per_class": n, "epochs": e} for n, e in SIZE_EPOCHS.items()),
    *({**CNN, "n_per_class": n} for n in SIZE_EPOCHS),
    # augmentation (ViT on a small subset where it matters, CNN on all data)
    *({**VIT, "n_per_class": 1000, "epochs": 15, "aug": aug} for aug in ["none", "strong"]),
    *({**CNN, "aug": aug} for aug in ["none", "strong"]),
    # learning rate
    *({**VIT, "lr": lr} for lr in [3e-6, 1e-5, 6e-5, 2e-4]),
    # input resolution
    *({**VIT, "img_size": s} for s in [112, 160]),
]

# ---------------------------------------------------------------- CIFAR-10 (parts g, h)
C_VIT = dict(model="vit_b", aug="basic", epochs=5, lr=2e-5)
C_CNN = dict(model="simple_cnn", aug="basic", epochs=40, lr=1e-3, batch_size=128)
IMB = dict(imbalance="step", minority="3,4,9", minority_frac=0.02)
IMB_METHODS = ["ce", "wce", "oversample", "focal", "logit_adjust"]

CIFAR = [
    C_VIT,
    {**C_VIT, "model": "vit_s"},
    {**C_VIT, "model": "resnet50", "lr": 1e-4},
    C_CNN,
    *({**C_VIT, **IMB, "method": m} for m in IMB_METHODS),
    *({**C_CNN, **IMB, "method": m} for m in IMB_METHODS),
]

GRIDS = {"dvc": DVC, "cifar": CIFAR}


def to_argv(cfg):
    argv = []
    for k, v in cfg.items():
        flag = "--" + k.replace("_", "-")
        if v is True:
            argv.append(flag)
        elif v is not False and v is not None:
            argv += [flag, str(v)]
    return argv


def main():
    if "--" in sys.argv:
        cut = sys.argv.index("--")
        own, extra = sys.argv[1:cut], sys.argv[cut + 1:]
    else:
        own, extra = sys.argv[1:], []
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("task", choices=GRIDS)
    p.add_argument("--count", action="store_true")
    p.add_argument("--list", action="store_true")
    p.add_argument("--index", type=int)
    p.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    a = p.parse_args(own)
    grid = GRIDS[a.task]

    if a.count:
        print(len(grid))
        return
    if a.list or a.index is None:
        for i, cfg in enumerate(grid):
            print(i, " ".join(to_argv(cfg)))
        return

    if a.task == "dvc":
        from train_dvc import main as train
    else:
        from train_cifar import main as train
    failed = []
    for seed in map(int, a.seeds.split(",")):
        argv = to_argv(grid[a.index]) + ["--seed", str(seed)] + extra
        print("\n=== python", f"train_{a.task}.py", " ".join(argv), flush=True)
        try:
            train(argv)
        except Exception:
            traceback.print_exc()
            failed.append(seed)
    if failed:
        sys.exit(f"config {a.index}: seeds {failed} failed")


if __name__ == "__main__":
    main()
