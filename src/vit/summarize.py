"""Collect every finished run into tables (CSV + Markdown) and report figures.

Reads <out-root>/{dvc,cifar,features_svm}; writes to <results>/ (default ./results).
Works on partial results: anything not run yet is skipped.
"""
import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import CIFAR_CLASSES
from engine import load_json
from sweep import C_CNN, C_VIT, CNN, IMB, IMB_METHODS, SIZE_EPOCHS, VIT

# ---------------------------------------------------------------- style (light report figures)
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3, fixed order
SEQ = ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"]  # one-hue blue ramp
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "white",
})

DVC_KEYS = ["model", "mode", "n_per_class", "aug", "img_size", "lr", "epochs"]
CIFAR_KEYS = ["model", "img_size", "aug", "imbalance", "method", "lr", "epochs"]
NAMES = {("vit_b", "finetune"): "ViT-B/16 fine-tuned", ("vit_s", "finetune"): "ViT-S/16 fine-tuned",
         ("resnet50", "finetune"): "ResNet-50 fine-tuned", ("vit_b", "linprobe"): "ViT-B/16 linear probe",
         ("simple_cnn", "finetune"): "Small CNN (from scratch)"}
BACKBONES = {"vit_b": "ViT-B/16", "vit_s": "ViT-S/16", "resnet50": "ResNet-50", "pixels": "raw pixels"}
CLASSIC = {"logreg": "Logistic reg.", "svm_rbf": "RBF-SVM", "knn": "kNN"}
METHOD_NAMES = {"ce": "Plain CE", "wce": "Weighted CE", "oversample": "Oversampling", "focal": "Focal loss",
                "logit_adjust": "Logit adjustment"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", default=os.environ.get("IE4483_OUT", "runs"))
    p.add_argument("--results", default=os.environ.get("IE4483_RESULTS", "results"))
    return p.parse_args(argv)


# ---------------------------------------------------------------- loading

def load_runs(root, task):
    rows = []
    for f in sorted((Path(root) / task).glob("*/metrics.json")):
        m = load_json(f)
        c = m["config"]
        row = {k: c[k] for k in (DVC_KEYS if task == "dvc" else CIFAR_KEYS)}
        row.update(seed=c["seed"], run=f.parent.name, limit=c.get("limit", 0), train_time_min=m["train_time_s"] / 60)
        if task == "dvc":
            row.update(best_val_acc=m["best_val_acc"], best_epoch=m["best_epoch"], final_val_acc=m["final_val_acc"],
                       params_M=m["params"] / 1e6, trainable_M=m["trainable_params"] / 1e6)
        else:
            t = m["test"]
            row.update(test_acc=t["acc"], macro_f1=t["macro_f1"], minority_recall=t.get("minority_recall"),
                       majority_recall=t.get("majority_recall"), params_M=m["params"] / 1e6,
                       per_class_recall=t["per_class_recall"], confusion=t["confusion_matrix"])
        rows.append(row)
    return pd.DataFrame(rows)


def pick(df, cfg):
    """Rows (all seeds) whose config matches every key of `cfg` that is a column of df."""
    if df.empty:
        return df
    mask = np.ones(len(df), bool)
    for k, v in cfg.items():
        if k in df.columns:
            mask &= np.isclose(df[k], v) if isinstance(v, float) else (df[k] == v).to_numpy()
    return df[mask]


def ms(df, col):
    v = df[col].dropna().to_numpy(float)
    return (v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0, len(v)) if len(v) else (np.nan, np.nan, 0)


def fmt(m, s, n, pct=True):
    if n == 0:
        return "-"
    k = 100 if pct else 1
    return f"{k * m:.2f} ± {k * s:.2f}" if n > 1 else f"{k * m:.2f}"


def write_table(df, path, title):
    df.to_csv(path.with_suffix(".csv"), index=False)
    cells = [[str(c) for c in df.columns]] + [["" if pd.isna(v) else str(v) for v in row] for row in df.to_numpy()]
    lines = ["| " + " | ".join(r) + " |" for r in cells]
    lines.insert(1, "|" + "|".join("---" for _ in df.columns) + "|")
    with open(path.with_suffix(".md"), "w", encoding="utf-8") as f:
        f.write(f"### {title}\n\n" + "\n".join(lines) + "\n")


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


def series(ax, xs, stats, color, label, marker="o"):
    m = np.array([s[0] for s in stats])
    sd = np.array([s[1] for s in stats])
    ax.plot(xs, m, color=color, lw=2, marker=marker, ms=7, mec="white", mew=1.5, label=label, zorder=3)
    ax.fill_between(xs, m - sd, m + sd, color=color, alpha=0.15, lw=0)


def legend_handles(pairs):
    return [matplotlib.lines.Line2D([], [], marker="o", ls="", color=c, mec="white", ms=8, label=l) for l, c in pairs]


# ---------------------------------------------------------------- Dogs vs. Cats

def summarize_dvc(dvc, feats, res, out_root):
    if dvc.empty and not feats:
        return
    if not dvc.empty:
        dvc.to_csv(res / "dvc_runs.csv", index=False)
        g = (dvc.groupby(DVC_KEYS, dropna=False)
             .agg(seeds=("seed", "nunique"), val_acc_mean=("best_val_acc", "mean"), val_acc_std=("best_val_acc", "std"),
                  final_val_acc_mean=("final_val_acc", "mean"), best_epoch=("best_epoch", "mean"),
                  minutes_per_run=("train_time_min", "mean"), trainable_M=("trainable_M", "first"))
             .reset_index().sort_values("val_acc_mean", ascending=False))
        for c in ["val_acc_mean", "val_acc_std", "final_val_acc_mean"]:
            g[c] = (100 * g[c]).round(2)
        g = g.round({"best_epoch": 1, "minutes_per_run": 1, "trainable_M": 2})
        write_table(g, res / "dvc_summary", "Dogs vs. Cats: validation accuracy (%) per config, mean/std over seeds")

    # (b/f) model comparison incl. conventional classifiers on frozen features
    rows = []
    for cfg in [VIT, {**VIT, "model": "vit_s"}, {**VIT, "model": "resnet50", "lr": 1e-4},
                {**VIT, "mode": "linprobe", "lr": 1e-3}, CNN]:
        m, s, n = ms(pick(dvc, cfg), "best_val_acc")
        if n:
            color = BLUE if cfg["mode"] == "finetune" and cfg["model"] != "simple_cnn" else AQUA if cfg["mode"] == "linprobe" else ORANGE
            rows.append((NAMES[(cfg["model"], cfg["mode"])], m, s, color))
    for bb, r in feats.items():
        best = max(CLASSIC, key=lambda c: r[c]["val_acc"])
        rows.append((f"{CLASSIC[best]} on {BACKBONES[bb]}" + ("" if bb == "pixels" else " features"),
                     r[best]["val_acc"], 0.0, ORANGE if bb == "pixels" else AQUA))
    if rows:
        rows.sort(key=lambda r: -r[1])
        fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(rows) + 1.4))
        dot_rows(ax, rows, "Validation accuracy (5,000 images)")
        ax.set_title("Dogs vs. Cats: model comparison")
        ax.legend(handles=legend_handles([("Pretrained, fine-tuned", BLUE), ("Pretrained, frozen features", AQUA),
                                          ("No pretraining", ORANGE)]), loc="upper center", ncol=3, fontsize=8.5,
                  bbox_to_anchor=(0.4, -0.12 - 0.6 / len(rows)))
        fig.savefig(res / "fig_dvc_models.png")
        plt.close(fig)

    if feats:
        t = pd.DataFrame([{"features": BACKBONES[bb], "dim": r["feature_dim"],
                           **{CLASSIC[c]: f"{100 * r[c]['val_acc']:.2f} ({', '.join(f'{k}={v}' for k, v in r[c]['best_params'].items())})"
                              for c in CLASSIC}} for bb, r in feats.items()])
        write_table(t, res / "features_svm", "Conventional classifiers on fixed features: val acc % (CV-chosen hyper-parameter)")

    if dvc.empty:
        return
    # (f) amount of training data
    sizes = [*SIZE_EPOCHS, 10000]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for base, color, label in [(VIT, BLUE, "ViT-B/16 fine-tuned"), (CNN, ORANGE, "Small CNN (from scratch)")]:
        stats = [ms(pick(dvc, {**base, "n_per_class": 0 if n == 10000 else n,
                               "epochs": base["epochs"] if n == 10000 or base is CNN else SIZE_EPOCHS[n]}), "best_val_acc")
                 for n in sizes]
        xs = [2 * n for n, s in zip(sizes, stats) if s[2]]
        stats = [s for s in stats if s[2]]
        if stats:
            series(ax, xs, stats, color, label)
            ax.annotate(label, (xs[-1], stats[-1][0]), xytext=(6, 0), textcoords="offset points", va="center",
                        fontsize=8.5, color=INK2)
    ax.set_xscale("log")
    ax.set_xticks([2 * n for n in sizes], [f"{2 * n:,}" for n in sizes])
    ax.minorticks_off()
    ax.set_xlabel("Training images (both classes)")
    ax.set_ylabel("Validation accuracy")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title("Validation accuracy vs. amount of training data")
    ax.legend(loc="lower right", fontsize=8.5)
    fig.savefig(res / "fig_dvc_train_size.png")
    plt.close(fig)

    # (c) learning rate
    lrs = [3e-6, 1e-5, 2e-5, 6e-5, 2e-4]
    stats = [ms(pick(dvc, {**VIT, "lr": lr}), "best_val_acc") for lr in lrs]
    keep = [(lr, s) for lr, s in zip(lrs, stats) if s[2]]
    if len(keep) > 1:
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        series(ax, [k[0] for k in keep], [k[1] for k in keep], BLUE, "ViT-B/16 fine-tuned")
        ax.set_xscale("log")
        ax.set_xticks([k[0] for k in keep], [f"{k[0]:g}" for k in keep])
        ax.minorticks_off()
        ax.set_xlabel("Peak learning rate (AdamW, warm-up + cosine)")
        ax.set_ylabel("Validation accuracy")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
        ax.set_title("ViT-B/16: learning-rate sweep")
        fig.savefig(res / "fig_dvc_lr.png")
        plt.close(fig)

    # (f) data processing: augmentation and input resolution
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw={"width_ratios": [3, 2]})
    augs = ["none", "basic", "strong"]
    for base, color, label in [({**VIT, "n_per_class": 1000, "epochs": 15}, BLUE, "ViT-B/16, 2,000 images"),
                               (CNN, ORANGE, "Small CNN, 20,000 images")]:
        st = [ms(pick(dvc, {**base, "aug": aug}), "best_val_acc") for aug in augs]
        if any(s[2] for s in st):
            series(axes[0], [i for i, s in enumerate(st) if s[2]], [s for s in st if s[2]], color, label)
    axes[0].set_xticks(range(3), ["none", "basic\n(scale, rotate, flip)", "strong\n(+TrivialAugment,\nrandom erasing)"])
    axes[0].set_ylabel("Validation accuracy")
    axes[0].set_title("Augmentation")
    axes[0].legend(fontsize=8.5, loc="center")
    res_sizes = [112, 160, 224]
    st = [ms(pick(dvc, {**VIT, "img_size": s}), "best_val_acc") for s in res_sizes]
    if any(s[2] for s in st):
        series(axes[1], [r for r, s in zip(res_sizes, st) if s[2]], [s for s in st if s[2]], BLUE, "ViT-B/16")
    axes[1].set_xticks(res_sizes, [f"{s}px" for s in res_sizes])
    axes[1].set_title("Input resolution (ViT-B/16)")
    for ax in axes:
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    fig.savefig(res / "fig_dvc_processing.png")
    plt.close(fig)

    # training curves of the main model and the scratch CNN (seed 0)
    for cfg, name in [(VIT, "vit_b"), (CNN, "cnn")]:
        r = pick(dvc, cfg)
        r = r[r.seed == 0] if not r.empty else r
        if r.empty:
            continue
        run_dir = Path(out_root) / "dvc" / r.iloc[0]["run"]
        h = pd.read_csv(run_dir / "history.csv")
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
        for ax, key, title in [(axes[0], "loss", "Cross-entropy loss"), (axes[1], "acc", "Accuracy")]:
            ax.plot(h.epoch, h[f"train_{key}"], color=BLUE, lw=2, marker="o", ms=5, label="train")
            ax.plot(h.epoch, h[f"val_{key}"], color=ORANGE, lw=2, marker="o", ms=5, label="validation")
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.legend(fontsize=8.5)
        axes[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
        fig.suptitle(f"Training curves: {NAMES[(cfg['model'], cfg['mode'])]}", x=0.01, y=1.05, ha="left", fontweight="bold")
        fig.savefig(res / f"fig_dvc_curves_{name}.png")
        plt.close(fig)


# ---------------------------------------------------------------- CIFAR-10

def summarize_cifar(cif, res):
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
    write_table(g.round({"minutes_per_run": 1}), res / "cifar_summary", "CIFAR-10 test set (10,000 images), mean/std over seeds")

    # (g) balanced CIFAR-10
    bal = {"imbalance": "none", "method": "ce"}
    rows = []
    for cfg, label, color in [(C_VIT, "ViT-B/16 fine-tuned", BLUE), ({**C_VIT, "model": "vit_s"}, "ViT-S/16 fine-tuned", BLUE),
                              ({**C_VIT, "model": "resnet50", "lr": 1e-4}, "ResNet-50 fine-tuned", BLUE),
                              (C_CNN, "Small CNN (from scratch, 32px)", ORANGE)]:
        m, s, n = ms(pick(cif, {**cfg, **bal}), "test_acc")
        if n:
            rows.append((label, m, s, color))
    if rows:
        fig, ax = plt.subplots(figsize=(7, 0.45 * len(rows) + 1.3))
        dot_rows(ax, rows, "Test accuracy (10,000 images)")
        ax.set_title("CIFAR-10: model comparison")
        fig.savefig(res / "fig_cifar_models.png")
        plt.close(fig)

    main = pick(cif, {**C_VIT, **bal})
    if not main.empty:
        cm = np.mean([np.asarray(c, float) for c in main.confusion], axis=0)
        cm = cm / cm.sum(1, keepdims=True)
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
        ax.set_title("CIFAR-10 confusion matrix, ViT-B/16 (% of each true class)")
        fig.savefig(res / "fig_cifar_confusion.png")
        plt.close(fig)
        rec = np.mean([np.asarray(r) for r in main.per_class_recall], axis=0)
        write_table(pd.DataFrame({"class": CIFAR_CLASSES, "recall_%": (100 * rec).round(2)}),
                    res / "cifar_per_class", "CIFAR-10 per-class recall, ViT-B/16 (balanced training)")

    # (h) imbalance: minority vs majority recall per method, one panel per model
    panels = [(C_VIT, "ViT-B/16 (pretrained)"), (C_CNN, "Small CNN (from scratch)")]
    panels = [p for p in panels if not pick(cif, {**p[0], **IMB}).empty]
    if panels:
        fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 3.8), squeeze=False)
        for ax, (base, title) in zip(axes[0], panels):
            x = np.arange(len(IMB_METHODS))
            for col, color, label, dx in [("minority_recall", ORANGE, "minority classes (2% data)", -0.12),
                                          ("majority_recall", BLUE, "majority classes", 0.12)]:
                st = [ms(pick(cif, {**base, **IMB, "method": m}), col) for m in IMB_METHODS]
                ax.errorbar(x + dx, [s[0] for s in st], yerr=[s[1] for s in st], fmt="o", ms=7, color=color,
                            elinewidth=2, capsize=0, mec="white", mew=1.5, label=label, zorder=3)
            bal_m = ms(pick(cif, {**base, **bal}), "test_acc")[0]
            if not np.isnan(bal_m):
                ax.axhline(bal_m, color=MUTED, lw=1, ls="--", zorder=1)
                ax.annotate("same model, balanced data", (0, bal_m), xycoords=("axes fraction", "data"),
                            xytext=(4, -11), textcoords="offset points", ha="left", fontsize=8, color=MUTED)
            ax.set_xticks(x, [METHOD_NAMES[m] for m in IMB_METHODS], rotation=20, ha="right")
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
            ax.set_ylabel("Test recall")
            ax.set_title(title)
        fig.legend(*axes[0][0].get_legend_handles_labels(), loc="upper left", ncol=2, fontsize=8.5,
                   bbox_to_anchor=(0.005, 1.02))
        fig.suptitle("CIFAR-10 with cat, deer and truck cut to 2% of their images", x=0.01, y=1.1, ha="left", fontweight="bold")
        fig.savefig(res / "fig_cifar_imbalance.png")
        plt.close(fig)

        tbl = []
        for base, title in panels:
            for m in IMB_METHODS:
                r = pick(cif, {**base, **IMB, "method": m})
                if r.empty:
                    continue
                rec = np.mean([np.asarray(v) for v in r.per_class_recall], axis=0)
                tbl.append({"model": title, "method": METHOD_NAMES[m], "test_acc": fmt(*ms(r, "test_acc")),
                            "macro_F1": fmt(*ms(r, "macro_f1")), "minority_recall": fmt(*ms(r, "minority_recall")),
                            **{c: round(100 * v, 1) for c, v in zip(CIFAR_CLASSES, rec)}})
        write_table(pd.DataFrame(tbl), res / "cifar_imbalance", "Imbalanced CIFAR-10: test metrics (%) and per-class recall")


def main(argv=None):
    args = parse_args(argv)
    res = Path(args.results)
    res.mkdir(parents=True, exist_ok=True)
    dvc, cif = load_runs(args.out_root, "dvc"), load_runs(args.out_root, "cifar")
    fpath = Path(args.out_root) / "features_svm" / "results.json"
    feats = load_json(fpath)["results"] if fpath.exists() else {}
    summarize_dvc(dvc, feats, res, args.out_root)
    summarize_cifar(cif, res)
    print(f"{len(dvc)} dvc runs, {len(cif)} cifar runs, features: {list(feats)} -> {res}")


if __name__ == "__main__":
    main()
