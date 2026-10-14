"""Collect every finished run into tables (CSV + Markdown) and figures, in the ViT track's format.

Reads <out-root>/{dvc,cifar}/*/metrics.json; writes to <results> (default ../../results/resnet).
Rows from the ViT track (<reference>/dvc_summary.csv, cifar_summary.csv) are drawn in grey for context.
Works on partial results: anything not run yet is skipped.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).parents[1]
REPO = PROJECT.parents[1]
CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
METHODS = ["ce", "wce", "oversample", "focal", "logit_adjust"]

# ---------------------------------------------------------------- style (same as src/vit/summarize.py)
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"]
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "white",
})

DVC_KEYS = ["name", "model", "optimizer", "lr", "wd", "batch_size", "dropout", "fine_tuning", "img_size", "epochs"]
CIFAR_KEYS = ["name", "model", "img_size", "imbalance", "method", "optimizer", "lr", "epochs"]
LABELS = {"best": "ResNet-18, best Optuna params", "worst": "ResNet-18, worst Optuna params"}
COLORS = {"best": BLUE, "worst": ORANGE}
METHOD_NAMES = {"ce": "Plain CE", "wce": "Weighted CE", "oversample": "Oversampling", "focal": "Focal loss",
                "logit_adjust": "Logit adjustment"}
VIT_DVC = [("vit_b", "finetune", 2e-5, "ViT-B/16 fine-tuned"), ("vit_s", "finetune", 2e-5, "ViT-S/16 fine-tuned"),
           ("resnet50", "finetune", 1e-4, "ResNet-50 fine-tuned"), ("vit_b", "linprobe", 1e-3, "ViT-B/16 linear probe"),
           ("simple_cnn", "finetune", 1e-3, "Small CNN (from scratch)")]
VIT_CIFAR = [("vit_b", "ViT-B/16 fine-tuned"), ("vit_s", "ViT-S/16 fine-tuned"),
             ("resnet50", "ResNet-50 fine-tuned"), ("simple_cnn", "Small CNN (from scratch, 32px)")]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", default=str(PROJECT / "outputs" / "runs"))
    p.add_argument("--results", default=str(REPO / "results" / "resnet"))
    p.add_argument("--reference", default=str(REPO / "results" / "vit"), help="ViT track results ('' to skip)")
    return p.parse_args(argv)


# ---------------------------------------------------------------- loading

def load_runs(root, task):
    rows = []
    for f in sorted((Path(root) / task).glob("*/metrics.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        c = m["config"]
        row = dict(name=c["name"], model=c["model"], optimizer=c["optimizer"], lr=c["learning_rate"],
                   wd=c["weight_decay"], batch_size=c["batch_size"], dropout=round(c["dropout"], 3),
                   fine_tuning=c["fine_tuning"], img_size=c["image_size"], epochs=c["epochs"],
                   imbalance=c["imbalance"], method=c["method"], seed=c["seed"], run=f.parent.name,
                   limit=c.get("limit", 0), train_time_min=m["train_time_s"] / 60, epochs_run=m["epochs_run"],
                   params_M=m["params"] / 1e6, trainable_M=m["trainable_params"] / 1e6)
        if task == "dvc":
            row.update(best_val_acc=m["best_val_acc"], best_epoch=m["best_epoch"], final_val_acc=m["final_val_acc"],
                       balanced_acc=m["best_val_balanced_acc"], precision=m["best_val_precision"],
                       recall=m["best_val_recall"], f1=m["best_val_f1"])
        else:
            t = m["test"]
            row.update(test_acc=t["acc"], macro_f1=t["macro_f1"], minority_recall=t.get("minority_recall"),
                       majority_recall=t.get("majority_recall"), per_class_recall=t["per_class_recall"],
                       confusion=t["confusion_matrix"])
        rows.append(row)
    return pd.DataFrame(rows)


def pick(df, cfg):
    if df.empty:
        return df
    mask = np.ones(len(df), bool)
    for k, v in cfg.items():
        if k in df.columns:
            mask &= np.isclose(df[k], v) if isinstance(v, float) else (df[k] == v).to_numpy()
    return df[mask]


def ms(df, col):
    v = df[col].dropna().to_numpy(float) if not df.empty else np.array([])
    return (v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0, len(v)) if len(v) else (np.nan, np.nan, 0)


def fmt(m, s, n):
    if n == 0:
        return "-"
    return f"{100 * m:.2f} ± {100 * s:.2f}" if n > 1 else f"{100 * m:.2f}"


def write_table(df, path, title):
    df.to_csv(path.with_suffix(".csv"), index=False)
    cells = [[str(c) for c in df.columns]] + [["" if pd.isna(v) else str(v) for v in row] for row in df.to_numpy()]
    lines = ["| " + " | ".join(r) + " |" for r in cells]
    lines.insert(1, "|" + "|".join("---" for _ in df.columns) + "|")
    path.with_suffix(".md").write_text(f"### {title}\n\n" + "\n".join(lines) + "\n", encoding="utf-8")


def reference(ref, file):
    path = Path(ref) / file if ref else None
    return pd.read_csv(path) if path and path.exists() else pd.DataFrame()


# ---------------------------------------------------------------- figure helpers

def dot_rows(ax, rows, xlabel):
    """Horizontal dot plot: rows = [(label, mean, std, color)], top to bottom."""
    y = np.arange(len(rows))[::-1]
    for yi, (label, m, s, color) in zip(y, rows):
        ax.errorbar(m, yi, xerr=s if s else None, fmt="o", ms=7, color=color, ecolor=color, elinewidth=2,
                    capsize=0, mec="white", mew=1.5, zorder=3)
        ax.annotate(f"{100 * m:.2f}%", (m, yi), xytext=(8, 6), textcoords="offset points", fontsize=8.5, color=INK2)
    ax.set_yticks(y, [r[0] for r in rows], color=INK)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    lo = min(r[1] - (r[2] or 0) for r in rows)
    ax.set_xlim(max(0, lo - 0.03), 1.0 + 0.004)


def legend_handles(pairs):
    return [matplotlib.lines.Line2D([], [], marker="o", ls="", color=c, mec="white", ms=8, label=l) for l, c in pairs]


def comparison_figure(rows, xlabel, title, path):
    rows.sort(key=lambda r: -r[1])
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(rows) + 1.4))
    dot_rows(ax, rows, xlabel)
    ax.set_title(title)
    ax.legend(handles=legend_handles([(LABELS["best"], BLUE), (LABELS["worst"], ORANGE),
                                      ("ViT track (reference)", MUTED)]),
              loc="upper center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.4, -0.12 - 0.6 / len(rows)))
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- Dogs vs. Cats

def summarize_dvc(dvc, res, out_root, ref):
    if dvc.empty:
        return
    dvc.to_csv(res / "dvc_runs.csv", index=False)
    g = (dvc.groupby(DVC_KEYS, dropna=False)
         .agg(seeds=("seed", "nunique"), val_acc_mean=("best_val_acc", "mean"), val_acc_std=("best_val_acc", "std"),
              final_val_acc_mean=("final_val_acc", "mean"), balanced_acc=("balanced_acc", "mean"),
              precision=("precision", "mean"), recall=("recall", "mean"), f1=("f1", "mean"),
              best_epoch=("best_epoch", "mean"), epochs_run=("epochs_run", "mean"),
              minutes_per_run=("train_time_min", "mean"), trainable_M=("trainable_M", "first"))
         .reset_index().sort_values("val_acc_mean", ascending=False))
    for c in ["val_acc_mean", "val_acc_std", "final_val_acc_mean", "balanced_acc", "precision", "recall", "f1"]:
        g[c] = (100 * g[c]).round(2)
    g = g.round({"best_epoch": 1, "epochs_run": 1, "minutes_per_run": 1, "trainable_M": 4})
    g[["lr", "wd"]] = g[["lr", "wd"]].map(lambda v: f"{v:.3g}")
    write_table(g, res / "dvc_summary", "Dogs vs. Cats: validation metrics (%) per config, mean/std over seeds")

    rows = []
    for name in ("best", "worst"):
        m, s, n = ms(pick(dvc, {"name": name}), "best_val_acc")
        if n:
            rows.append((LABELS[name], m, s, COLORS[name]))
    vit = reference(ref, "dvc_summary.csv")
    for model, mode, lr, label in VIT_DVC:
        r = pick(vit, {"model": model, "mode": mode, "n_per_class": 0, "aug": "basic", "img_size": 224, "lr": lr})
        if not r.empty:
            rows.append((f"{label} (ViT track)", r.iloc[0]["val_acc_mean"] / 100, r.iloc[0]["val_acc_std"] / 100, MUTED))
    comparison_figure(rows, "Validation accuracy (5,000 images)", "Dogs vs. Cats: model comparison",
                      res / "fig_dvc_models.png")

    # training curves of each baseline (first seed)
    for name in ("best", "worst"):
        r = pick(dvc, {"name": name}).sort_values("seed")
        if r.empty:
            continue
        h = pd.read_csv(Path(out_root) / "dvc" / r.iloc[0]["run"] / "history.csv")
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
        for ax, key, title in [(axes[0], "loss", "Cross-entropy loss"), (axes[1], "acc", "Accuracy")]:
            ax.plot(h.epoch, h[f"train_{key}"], color=BLUE, lw=2, marker="o", ms=5, label="train")
            ax.plot(h.epoch, h[f"val_{key}"], color=ORANGE, lw=2, marker="o", ms=5, label="validation")
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.legend(fontsize=8.5)
        axes[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
        fig.suptitle(f"Training curves: {LABELS[name]} (seed {r.iloc[0]['seed']})", x=0.01, y=1.05, ha="left",
                     fontweight="bold")
        fig.savefig(res / f"fig_dvc_curves_{name}.png")
        plt.close(fig)


# ---------------------------------------------------------------- CIFAR-10

def summarize_cifar(cif, res, ref):
    if cif.empty:
        return
    cif.drop(columns=["per_class_recall", "confusion"]).to_csv(res / "cifar_runs.csv", index=False)
    g = (cif.groupby(CIFAR_KEYS).agg(seeds=("seed", "nunique"), test_acc=("test_acc", "mean"),
                                     test_acc_std=("test_acc", "std"), macro_f1=("macro_f1", "mean"),
                                     minority_recall=("minority_recall", "mean"),
                                     minority_recall_std=("minority_recall", "std"),
                                     majority_recall=("majority_recall", "mean"),
                                     minutes_per_run=("train_time_min", "mean"))
         .reset_index())
    for c in ["test_acc", "test_acc_std", "macro_f1", "minority_recall", "minority_recall_std", "majority_recall"]:
        g[c] = (100 * g[c]).round(2)
    g["lr"] = g["lr"].map(lambda v: f"{v:.3g}")
    write_table(g.round({"minutes_per_run": 1}), res / "cifar_summary",
                "CIFAR-10 test set (10,000 images), mean/std over seeds")

    bal = {"imbalance": "none", "method": "ce"}
    rows = []
    for name in ("best", "worst"):
        m, s, n = ms(pick(cif, {"name": name, **bal}), "test_acc")
        if n:
            rows.append((LABELS[name], m, s, COLORS[name]))
    vit = reference(ref, "cifar_summary.csv")
    for model, label in VIT_CIFAR:
        r = pick(vit, {"model": model, **bal})
        if not r.empty:
            rows.append((f"{label} (ViT track)", r.iloc[0]["test_acc"] / 100, r.iloc[0]["test_acc_std"] / 100, MUTED))
    if any(r[3] != MUTED for r in rows):
        comparison_figure(rows, "Test accuracy (10,000 images)", "CIFAR-10: model comparison",
                          res / "fig_cifar_models.png")

    main = pick(cif, {"name": "best", **bal})
    if not main.empty:
        cm = np.mean([np.asarray(c, float) for c in main.confusion], axis=0)
        cm = cm / np.maximum(cm.sum(1, keepdims=True), 1e-12)
        fig, ax = plt.subplots(figsize=(6.4, 5.4))
        ax.imshow(cm, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("blue", SEQ), vmin=0, vmax=1)
        for i in range(10):
            for j in range(10):
                if cm[i, j] >= 0.005:
                    ax.text(j, i, f"{100 * cm[i, j]:.0f}" if cm[i, j] >= 0.01 else "<1", ha="center", va="center",
                            fontsize=7.5, color="white" if cm[i, j] > 0.5 else INK2)
        ax.set_xticks(range(10), CIFAR_CLASSES, rotation=45, ha="right")
        ax.set_yticks(range(10), CIFAR_CLASSES)
        ax.grid(False)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title("CIFAR-10 confusion matrix, ResNet-18 best params (% of each true class)")
        fig.savefig(res / "fig_cifar_confusion.png")
        plt.close(fig)
        rec = np.mean([np.asarray(r) for r in main.per_class_recall], axis=0)
        write_table(pd.DataFrame({"class": CIFAR_CLASSES, "recall_%": (100 * rec).round(2)}),
                    res / "cifar_per_class", "CIFAR-10 per-class recall, ResNet-18 best params (balanced training)")

    imb = pick(cif, {"name": "best", "imbalance": "step"})
    if imb.empty:
        return
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    x = np.arange(len(METHODS))
    for col, color, label, dx in [("minority_recall", ORANGE, "minority classes (2% data)", -0.12),
                                  ("majority_recall", BLUE, "majority classes", 0.12)]:
        st = [ms(pick(imb, {"method": m}), col) for m in METHODS]
        ax.errorbar(x + dx, [s[0] for s in st], yerr=[s[1] for s in st], fmt="o", ms=7, color=color,
                    elinewidth=2, capsize=0, mec="white", mew=1.5, label=label, zorder=3)
    bal_m = ms(main, "test_acc")[0]
    if not np.isnan(bal_m):
        ax.axhline(bal_m, color=MUTED, lw=1, ls="--", zorder=1)
        ax.annotate("same model, balanced data", (0, bal_m), xycoords=("axes fraction", "data"),
                    xytext=(4, -11), textcoords="offset points", ha="left", fontsize=8, color=MUTED)
    ax.set_xticks(x, [METHOD_NAMES[m] for m in METHODS], rotation=20, ha="right")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("Test recall")
    ax.set_title("ResNet-18 (pretrained), best Optuna params")
    ax.legend(loc="upper left", ncol=2, fontsize=8.5, bbox_to_anchor=(0, 1.2))
    fig.suptitle("CIFAR-10 with cat, deer and truck cut to 2% of their images", x=0.01, y=1.12, ha="left",
                 fontweight="bold")
    fig.savefig(res / "fig_cifar_imbalance.png")
    plt.close(fig)

    tbl = []
    for m in METHODS:
        r = pick(imb, {"method": m})
        if r.empty:
            continue
        rec = np.mean([np.asarray(v) for v in r.per_class_recall], axis=0)
        tbl.append({"model": "ResNet-18 (best params)", "method": METHOD_NAMES[m], "test_acc": fmt(*ms(r, "test_acc")),
                    "macro_F1": fmt(*ms(r, "macro_f1")), "minority_recall": fmt(*ms(r, "minority_recall")),
                    **{c: round(100 * v, 1) for c, v in zip(CIFAR_CLASSES, rec)}})
    write_table(pd.DataFrame(tbl), res / "cifar_imbalance", "Imbalanced CIFAR-10: test metrics (%) and per-class recall")


def main(argv=None):
    args = parse_args(argv)
    res = Path(args.results)
    res.mkdir(parents=True, exist_ok=True)
    dvc, cif = load_runs(args.out_root, "dvc"), load_runs(args.out_root, "cifar")
    summarize_dvc(dvc, res, args.out_root, args.reference)
    summarize_cifar(cif, res, args.reference)
    print(f"{len(dvc)} dvc runs, {len(cif)} cifar runs -> {res}")


if __name__ == "__main__":
    main()
