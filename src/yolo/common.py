"""Shared paths and helpers for the YOLO track.

Everything lives under $PROJECT_DIR (the cluster project directory on SSD):

    $PROJECT_DIR/data/catsdogs/{train,val}     Dogs-vs-Cats, ImageFolder layout (class subfolders)
    $PROJECT_DIR/data/catsdogs/test/*.jpg      the 500 unlabelled test images, flat (no class subfolders)
    $PROJECT_DIR/data/cifar10/{train,val,test} CIFAR-10 (val carved from train)
    $PROJECT_DIR/weights/                      pretrained *.pt
    $PROJECT_DIR/runs/<experiment>/            one folder per experiment
"""

import json
import os
import sys
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def project_dir() -> Path:
    p = os.environ.get("PROJECT_DIR")
    if not p:
        sys.exit("PROJECT_DIR is not set. `source slurm/cluster_env.sh` (cluster) or export PROJECT_DIR=<dir>.")
    return Path(p).expanduser()


def data_dir() -> Path:
    return project_dir() / "data"


def runs_dir() -> Path:
    return project_dir() / "runs"


def weights_dir() -> Path:
    return project_dir() / "weights"


def eval_dir(dataset: str, split: str) -> Path:
    """Folder of a labelled split. Every CIFAR-10 variant (balanced / imbalanced / oversampled) is scored on
    the same balanced cifar10/{val,test}, so only that one copy has to exist (e.g. on a laptop)."""
    base = "cifar10" if dataset.startswith("cifar10") else dataset
    return data_dir() / base / split


def num_workers(cap: int = 8) -> int:
    """Dataloader workers = CPUs we were actually given (4 per GPU on the cluster)."""
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:  # macOS
        n = os.cpu_count() or 1
    slurm = os.environ.get("SLURM_CPUS_ON_NODE")
    if slurm and slurm.isdigit():
        n = min(n, int(slurm))
    return max(1, min(n, cap))


def get_device() -> str:
    """CUDA if present, else CPU. Set YOLO_DEVICE=mps (or cpu / 0) to force a device, e.g. on a Mac."""
    forced = os.environ.get("YOLO_DEVICE")
    if forced:
        return forced
    import torch

    return "0" if torch.cuda.is_available() else "cpu"


def init_ultralytics():
    """Point Ultralytics at the project dir (settings, weights, runs) and turn off telemetry.

    Must run before the first `import ultralytics` so the settings file lands in the project dir
    and not in $HOME.
    """
    pd = project_dir()
    os.environ.setdefault("YOLO_CONFIG_DIR", str(pd / ".config" / "Ultralytics"))
    Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
    weights_dir().mkdir(parents=True, exist_ok=True)
    from ultralytics import settings

    settings.update(
        {
            "weights_dir": str(weights_dir()),
            "runs_dir": str(runs_dir()),
            "datasets_dir": str(data_dir()),
            "sync": False,
        }
    )


def YOLO(name: str):
    """Load a model; prefer the pre-downloaded copy in $PROJECT_DIR/weights."""
    init_ultralytics()
    from ultralytics import YOLO as _YOLO

    local = weights_dir() / name
    return _YOLO(str(local) if local.exists() else name)


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)  # atomic: a preempted job never leaves a half-written result


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def run_config(run: Path, model) -> dict:
    """Config of a pulled run: result.json if scoring finished, else recovered from the training args stored in
    the checkpoint (imgsz and data path), so a run whose scoring step never completed is still usable locally."""
    rj = Path(run) / "result.json"
    if rj.exists():
        return read_json(rj)["config"]
    args = getattr(model, "overrides", None) or {}
    if "imgsz" not in args or "data" not in args:
        raise SystemExit(f"{rj} is missing and the checkpoint carries no training args; re-run the experiment on the cluster.")
    cfg = {"imgsz": int(args["imgsz"]), "dataset": Path(str(args["data"])).name}
    print(f"NOTE: {rj.name} not found; using {cfg} from the checkpoint", flush=True)
    return cfg


def list_images(root: Path):
    """[(path, class_name)] for an ImageFolder-style root, sorted and following symlinks."""
    root = Path(root)
    if not root.is_dir():
        raise SystemExit(f"Image folder not found: {root}\nCopy the data there first (README: step 5b on a laptop, steps 1 and 3 on the cluster).")
    out = []
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() in IMG_EXT:
                out.append((f, cls_dir.name))
    return out
