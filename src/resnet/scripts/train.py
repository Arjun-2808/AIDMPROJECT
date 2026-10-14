import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from aidm_project.config import Config
from aidm_project.experiment import run
from aidm_project.losses import METHODS

DEFAULT_DATA = {"dogs_cats": PROJECT / "data" / "dogs_cats" / "datasets", "cifar10": PROJECT / "data" / "cifar10"}


def parse_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes"}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train one run and write it to <out-root>/<dvc|cifar>/<run>/.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None, help="default: data/dogs_cats/datasets or data/cifar10")
    parser.add_argument("--out-root", default=str(PROJECT / "outputs" / "runs"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--imbalance", choices=["none", "step"])
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--limit", type=int, help="cap every split at this many images (smoke tests)")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--cudnn-safe", type=parse_bool, help="true disables cuDNN (as in the Optuna search)")
    args = parser.parse_args(argv)
    config = Config.from_yaml(args.config, seed=args.seed, imbalance=args.imbalance, method=args.method,
                              epochs=args.epochs, limit=args.limit, num_workers=args.num_workers,
                              cudnn_safe=args.cudnn_safe)
    run(config, args.data_root or DEFAULT_DATA[config.task], args.out_root)


if __name__ == "__main__":
    main()
