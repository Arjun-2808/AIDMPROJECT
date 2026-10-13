"""Train (or resume) one experiment from experiments.py and score it.

    python train.py cd_s_ft            # run / resume / skip-if-done
    python train.py cd_s_ft --force    # wipe runs/cd_s_ft and start again

Idempotent and preemption-safe: finished runs are skipped (result.json exists), interrupted runs
continue from runs/<name>/weights/last.pt.
"""

import argparse
import csv
import shutil
import time

import torch

import common
import experiments
from evaluate import evaluate_split
from experiments import DATASET_EVAL, EXPS, Exp

# Extra Ultralytics args per augmentation preset ("strong" = Ultralytics defaults:
# RandomResizedCrop + flip + RandAugment + random erasing). "none" and the rotation in
# "strong_rot" are implemented in yolo_ext.ExtTrainer.
AUG_ARGS = {
    "none": {},
    "basic": {"auto_augment": None, "erasing": 0.0},  # crop + flip + colour jitter
    "strong": {},
    "strong_rot": {},
}


def train_args(exp: Exp, out, device) -> dict:
    a = dict(
        data=str(common.data_dir() / exp.dataset),
        epochs=exp.epochs,
        imgsz=exp.imgsz,
        batch=exp.batch,
        device=device,
        workers=common.num_workers(),
        seed=exp.seed,
        deterministic=True,
        optimizer=exp.optimizer,
        lr0=exp.lr0,
        lrf=exp.lrf,
        cos_lr=True,
        warmup_epochs=1,
        fraction=exp.fraction,
        patience=max(exp.epochs, 1),  # no early stopping: fixed budget, best.pt = best val epoch
        project=str(out.parent),
        name=exp.name,
        exist_ok=True,
        plots=False,
        amp=True,
    )
    if exp.mode == "scratch":
        a["pretrained"] = False
    a.update(AUG_ARGS[exp.aug])
    a.update(exp.extra)
    return a


def _training_finished(last) -> bool:
    """Ultralytics strips the optimizer from last.pt/best.pt only when training completes."""
    ck = torch.load(last, map_location="cpu", weights_only=False)
    return ck.get("optimizer") is None


def _read_results_csv(path):
    if not path.exists():
        return {}
    with open(path) as f:
        rows = list(csv.DictReader(f))
    rows = [{k.strip(): v for k, v in r.items()} for r in rows]
    acc_key = next((k for k in rows[0] if "accuracy_top1" in k), None) if rows else None
    if not rows or acc_key is None:
        return {}
    best = max(rows, key=lambda r: float(r[acc_key]))
    # `time` is cumulative within one session and restarts after a resume: sum the per-session maxima
    total, prev, seg_max = 0.0, 0.0, 0.0
    for r in rows:
        t = float(r.get("time", 0) or 0)
        if t < prev:
            total, seg_max = total + seg_max, 0.0
        seg_max, prev = max(seg_max, t), t
    return {
        "best_epoch": int(float(best["epoch"])),
        "ultralytics_val_top1": float(best[acc_key]),
        "epochs_run": len(rows),
        "train_seconds": total + seg_max,
    }


def env_info():
    import ultralytics

    return {
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def score(exp: Exp, out, model_path, extra=None):
    """Evaluate a checkpoint on this experiment's labelled splits and write result.json."""
    if torch.cuda.is_available():
        # Defensive: release any cached-but-unused CUDA memory from training before the eval-only
        # forward passes below, in case anything upstream (or a future change) leaves it around.
        torch.cuda.empty_cache()
    model = common.YOLO(str(model_path))
    res = {"name": exp.name, "config": exp.to_dict(), "status": "done", **env_info(), **(extra or {})}
    res["splits"] = {}
    for split in DATASET_EVAL[exp.dataset]:
        # Same batch size training itself already ran at (forward+backward): a forward-only pass at
        # the same size cannot need more GPU memory, so this is safe without re-deriving a number.
        res["splits"][split] = evaluate_split(model, common.eval_dir(exp.dataset, split), exp.imgsz, out / f"{split}_predictions.csv", batch=exp.batch)
    v = res["splits"]["val"]
    res["val_accuracy"], res["val_macro_f1"] = v["accuracy"], v["macro_f1"]
    common.write_json(out / "result.json", res)
    return res


def run_experiment(exp: Exp, force=False, device=None) -> dict:
    out = common.runs_dir() / exp.name
    result = out / "result.json"
    if result.exists() and not force:
        print(f"[skip] {exp.name} already done", flush=True)
        return common.read_json(result)
    if force:
        shutil.rmtree(out, ignore_errors=True)

    if exp.mode == "zeroshot":
        import zero_shot

        return zero_shot.run(exp, out)

    device = device or common.get_device()
    common.init_ultralytics()
    from yolo_ext import ExtTrainer, install_loss_callback

    ExtTrainer.ext = {"aug": exp.aug, "loss": exp.loss, "rotation": 15}
    last = out / "weights" / "last.pt"
    t0 = time.time()

    if last.exists() and _training_finished(last):
        print(f"[finish] {exp.name}: training already complete, scoring", flush=True)
    elif last.exists():
        print(f"[resume] {exp.name} from {last}", flush=True)
        model = common.YOLO(str(last))
        install_loss_callback(model, ExtTrainer.ext)
        model.train(resume=True, trainer=ExtTrainer)
    else:
        print(f"[train] {exp.name}", flush=True)
        model = common.YOLO(f"{exp.model}.yaml" if exp.mode == "scratch" else f"{exp.model}.pt")
        args = train_args(exp, out, device)
        if exp.mode == "linear":  # freeze every layer except the classification head
            args["freeze"] = len(model.model.model) - 1
        install_loss_callback(model, ExtTrainer.ext)
        model.train(trainer=ExtTrainer, **args)

    res = score(exp, out, out / "weights" / "best.pt", _read_results_csv(out / "results.csv"))
    print(f"[done] {exp.name} val_acc={res['val_accuracy']:.4f} ({time.time() - t0:.0f}s this session)", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    run_experiment(EXPS[a.name], force=a.force)
