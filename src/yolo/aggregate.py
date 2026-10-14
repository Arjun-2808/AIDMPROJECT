"""Collect every runs/*/result.json into one table (+ validation curves per group).

    python aggregate.py

Writes $PROJECT_DIR/runs/summary.csv and $PROJECT_DIR/runs/curves/<group>.png, and prints a markdown
table. Experiments named <base>__seedN are pooled with <base> into a mean +/- std block.
"""

import csv
import re
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import common  # noqa: E402
from experiments import GROUPS  # noqa: E402

COLS = ["name", "dataset", "model", "mode", "imgsz", "lr0", "aug", "loss", "fraction", "seed", "epochs_run", "best_epoch", "train_min", "val_acc", "val_macro_f1", "test_acc", "test_macro_f1"]


def load():
    rows = []
    for p in sorted(common.runs_dir().glob("*/result.json")):
        r = common.read_json(p)
        c = r["config"]
        test = r["splits"].get("test", {})
        rows.append(
            {
                "name": r["name"],
                **{k: c[k] for k in ("dataset", "model", "mode", "imgsz", "lr0", "aug", "loss", "fraction", "seed")},
                "epochs_run": r.get("epochs_run", ""),
                "best_epoch": r.get("best_epoch", ""),
                "train_min": round(r["train_seconds"] / 60, 1) if "train_seconds" in r else "",
                "val_acc": round(r["val_accuracy"], 4),
                "val_macro_f1": round(r["val_macro_f1"], 4),
                "test_acc": round(test["accuracy"], 4) if test else "",
                "test_macro_f1": round(test["macro_f1"], 4) if test else "",
            }
        )
    return rows


def curves():
    out = common.runs_dir() / "curves"
    out.mkdir(exist_ok=True)
    for g, names in GROUPS.items():
        if g in ("all", "catsdogs", "smoke"):
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        drew = False
        for n in names:
            csv_path = common.runs_dir() / n / "results.csv"
            if not csv_path.exists():
                continue
            with open(csv_path) as f:
                rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
            key = next((k for k in rows[0] if "accuracy_top1" in k), None) if rows else None
            if key:
                ax.plot([int(float(r["epoch"])) for r in rows], [float(r[key]) for r in rows], label=n)
                drew = True
        if drew:
            ax.set_xlabel("epoch")
            ax.set_ylabel("val top-1 accuracy")
            ax.set_title(g)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(out / f"{g}.png", dpi=150)
        plt.close(fig)


def main():
    rows = load()
    if not rows:
        raise SystemExit(f"no result.json under {common.runs_dir()}")
    with open(common.runs_dir() / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)

    print("| " + " | ".join(COLS) + " |\n|" + "---|" * len(COLS))
    for r in rows:
        print("| " + " | ".join(str(r[c]) for c in COLS) + " |")

    pooled = {}
    for r in rows:
        pooled.setdefault(re.sub(r"__seed\d+$", "", r["name"]), []).append(r)
    multi = {k: v for k, v in pooled.items() if len(v) > 1}
    if multi:
        print("\nMean +/- std over seeds")
        for k, v in multi.items():
            acc = [x["val_acc"] for x in v]
            print(f"  {k}: val_acc {statistics.mean(acc):.4f} +/- {statistics.stdev(acc):.4f}  (n={len(acc)}, seeds {[x['seed'] for x in v]})")
    curves()


if __name__ == "__main__":
    main()
