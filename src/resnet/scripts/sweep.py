"""Run the experiment grid, one subprocess per run; finished runs are skipped, so it can be resumed.

    dvc        Dogs vs. Cats, best and worst Optuna parameters x seeds          (6 runs)
    cifar      CIFAR-10 balanced, best and worst Optuna parameters x seeds      (6 runs)
    imbalance  CIFAR-10 with cat/deer/truck cut to 2 %, best parameters,
               5 imbalance methods x seeds                                      (15 runs)
    all        everything above, in that order

Example:  python scripts/sweep.py dvc          python scripts/sweep.py all --list
"""
import argparse
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from aidm_project.losses import METHODS

SEEDS = [2026, 2027, 2028]
STAGES = ["dvc", "cifar", "imbalance"]


def grid(stage):
    configs = PROJECT / "configs"
    if stage == "dvc":
        return [[f"{configs / f'dogs_cats_{n}.yaml'}", "--seed", str(s)] for n in ("best", "worst") for s in SEEDS]
    if stage == "cifar":
        return [[f"{configs / f'cifar10_{n}.yaml'}", "--seed", str(s)] for n in ("best", "worst") for s in SEEDS]
    if stage == "imbalance":
        return [[f"{configs / 'cifar10_best.yaml'}", "--seed", str(s), "--imbalance", "step", "--method", m]
                for m in METHODS for s in SEEDS]
    raise ValueError(stage)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=[*STAGES, "all"])
    parser.add_argument("--list", action="store_true", help="print the runs without training")
    args, extra = parser.parse_known_args(argv)  # anything else goes to train.py, e.g. --out-root X
    runs = [r for stage in (STAGES if args.stage == "all" else [args.stage]) for r in grid(stage)]
    failed = []
    for i, r in enumerate(runs, 1):
        cmd = [sys.executable, str(PROJECT / "scripts" / "train.py"), "--config", *r, *extra]
        print(f"[{i}/{len(runs)}] {' '.join(Path(r[0]).name if j == 0 else x for j, x in enumerate(r))}", flush=True)
        if args.list:
            continue
        if subprocess.run(cmd).returncode != 0:
            failed.append(r)
    if failed:
        print(f"{len(failed)} run(s) failed; re-run the same command to retry them:")
        for r in failed:
            print("  ", " ".join(r))
        sys.exit(1)


if __name__ == "__main__":
    main()
