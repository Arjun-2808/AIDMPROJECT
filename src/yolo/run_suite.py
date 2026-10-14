"""Run a group of experiments back to back inside ONE Slurm job.

The default QoS allows a single job per user, so sweeps are run sequentially here rather than as
many jobs. Every experiment is cached / resumable, so if the job hits its time limit or is
preempted you simply submit the same command again.

    python run_suite.py --list
    python run_suite.py --group models --require-gpu
    python run_suite.py --seeds-of cd_s_ft --require-gpu      # adds cd_s_ft__seed1, __seed2
"""

import argparse
import sys
import time
import traceback

import common
import experiments
from experiments import EXPS, GROUPS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=sorted(GROUPS))
    ap.add_argument("--only", help="comma-separated experiment names (instead of --group)")
    ap.add_argument("--seeds-of", metavar="NAME", help="run extra seeds of an existing experiment")
    ap.add_argument("--seeds", default="1,2", help="seeds for --seeds-of (default 1,2; base run is seed 0)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--require-gpu", action="store_true", help="abort if no CUDA GPU (never burn a job on CPU)")
    a = ap.parse_args()

    if a.list:
        for g, names in GROUPS.items():
            print(f"{g}: {', '.join(names)}")
        return

    exps = []
    if a.seeds_of:
        exps = experiments.seed_variants(a.seeds_of, tuple(int(s) for s in a.seeds.split(",")))
        for e in exps:
            EXPS[e.name] = e
    elif a.only:
        exps = [EXPS[n] for n in a.only.split(",")]
    elif a.group:
        exps = [EXPS[n] for n in GROUPS[a.group]]
    else:
        ap.error("give --group, --only or --seeds-of")

    if a.require_gpu:
        import torch

        if not torch.cuda.is_available():
            sys.exit("--require-gpu: CUDA is not available in this job; aborting instead of training on CPU.")

    from train import run_experiment  # after arg parsing: keeps --list / --help instant

    failed = []
    t0 = time.time()
    for i, e in enumerate(exps, 1):
        print(f"\n===== [{i}/{len(exps)}] {e.name}  (elapsed {(time.time() - t0) / 60:.1f} min) =====", flush=True)
        try:
            run_experiment(e, force=a.force)
        except Exception:
            traceback.print_exc()
            failed.append(e.name)  # keep going: one bad run must not waste the rest of the allocation
    print(f"\nfinished {len(exps) - len(failed)}/{len(exps)} in {(time.time() - t0) / 60:.1f} min", flush=True)
    if failed:
        print("FAILED:", ", ".join(failed), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
