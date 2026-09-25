"""Per-run logging, shared across all three members.

One run == one directory:

    experiments/<owner>/<timestamp>_<run_name>/
        config.json     exact config + env + git SHA that produced the run
        metrics.csv     one row per epoch
        summary.json    final numbers (this is what the leaderboard reads)
        train.log       stdout copy

Those files are small and text-only, so they are COMMITTED. Checkpoints (.pt)
are gitignored -- keep them on NSCC scratch and note the path in summary.json.

Every finished run also appends a row to results/leaderboard.md, which is the
single page you look at to see where everyone is.
"""
from __future__ import annotations

import csv
import json
import platform
import sys
import time
from pathlib import Path

from .utils import git_commit, save_json

LEADERBOARD = Path("results/leaderboard.md")
LEADERBOARD_HEADER = (
    "| date | owner | run | model | optimiser | sched | lr | bs | epochs | "
    "train imgs | val acc | notes |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
)


class RunLogger:
    def __init__(self, run_dir: str | Path, config: dict, owner: str, run_name: str):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.owner = owner
        self.run_name = run_name
        self.config = dict(config)
        self.rows: list[dict] = []
        self.start = time.time()

        self.config["_meta"] = {
            "owner": owner,
            "run_name": run_name,
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "git_commit": git_commit(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "command": " ".join(sys.argv),
        }
        save_json(self.config, self.dir / "config.json")

        self._log_fh = open(self.dir / "train.log", "a", encoding="utf-8")

    def print(self, *parts) -> None:
        """Print to stdout and tee into train.log."""
        msg = " ".join(str(p) for p in parts)
        print(msg, flush=True)
        self._log_fh.write(msg + "\n")
        self._log_fh.flush()

    def log_epoch(self, **row) -> None:
        self.rows.append(row)
        path = self.dir / "metrics.csv"
        write_header = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if write_header:
                w.writeheader()
            w.writerow(row)

    def finish(self, best: dict, notes: str = "") -> dict:
        summary = {
            "owner": self.owner,
            "run_name": self.run_name,
            "run_dir": str(self.dir).replace("\\", "/"),
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_min": round((time.time() - self.start) / 60, 1),
            "epochs_run": len(self.rows),
            "notes": notes,
            **best,
            "config": self.config,
        }
        save_json(summary, self.dir / "summary.json")
        self._append_leaderboard(summary, notes)
        self._log_fh.close()
        return summary

    def _append_leaderboard(self, summary: dict, notes: str) -> None:
        c = self.config
        LEADERBOARD.parent.mkdir(parents=True, exist_ok=True)
        if not LEADERBOARD.exists():
            LEADERBOARD.write_text(
                "# Results leaderboard\n\n"
                "Auto-appended by `RunLogger.finish()`. Newest rows at the bottom.\n"
                "If two rows conflict on merge, keep both -- they are different runs.\n\n"
                + LEADERBOARD_HEADER,
                encoding="utf-8",
            )
        row = (
            f"| {time.strftime('%m-%d %H:%M')} "
            f"| {self.owner} "
            f"| [{self.run_name}]({summary['run_dir']}) "
            f"| {c.get('model', {}).get('name', '?')} "
            f"| {c.get('optim', {}).get('name', '?')} "
            f"| {c.get('optim', {}).get('scheduler', '?')} "
            f"| {c.get('optim', {}).get('lr', '?')} "
            f"| {c.get('data', {}).get('batch_size', '?')} "
            f"| {summary.get('epochs_run', '?')} "
            f"| {c.get('data', {}).get('train_per_class') or 'all'} "
            f"| **{summary.get('best_val_acc', float('nan')):.4f}** "
            f"| {notes} |\n"
        )
        with open(LEADERBOARD, "a", encoding="utf-8") as f:
            f.write(row)


def load_config(path: str | Path) -> dict:
    """YAML config, with `_base_` single-level inheritance."""
    import yaml

    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = cfg.pop("_base_", None)
    if base:
        parent = load_config(path.parent / base)
        cfg = _deep_merge(parent, cfg)
    return cfg


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out
