"""Experiment registry. One `Exp` = one training run; groups are lists of experiment names.

Results are cached by experiment name (runs/<name>/result.json), so an experiment that appears
in several groups is only ever trained once.

Modes:
    finetune  pretrained weights, all layers trainable            (main recommendation)
    linear    pretrained weights, backbone frozen, head trained   (linear probe)
    scratch   random init from the model yaml                     (no pretraining)
    zeroshot  ImageNet-1k head, dog/cat classes summed, no training
"""

from dataclasses import asdict, dataclass, field, replace


@dataclass
class Exp:
    name: str
    dataset: str = "catsdogs"  # folder under $PROJECT_DIR/data
    model: str = "yolo26s-cls"  # weights stem; scratch uses <stem>.yaml
    mode: str = "finetune"
    epochs: int = 15
    imgsz: int = 224
    batch: int = 32
    optimizer: str = "AdamW"
    lr0: float = 1e-3
    lrf: float = 0.01
    aug: str = "strong"  # none | basic | strong | strong_rot
    loss: str = "ce"  # ce | wce_inv | wce_eff | focal
    fraction: float = 1.0  # fraction of the *train* split used
    seed: int = 0
    extra: dict = field(default_factory=dict)  # raw Ultralytics train args (override everything)

    def to_dict(self):
        return asdict(self)


def _build():
    exps = []

    def add(name, **kw):
        exps.append(Exp(name=name, **kw))

    # ---------------------------------------------------------------- Dogs vs Cats
    add("cd_smoke", model="yolo26n-cls", epochs=1, fraction=0.02)  # pipeline check only

    # model size / generation (all full fine-tune, identical recipe)
    for tag, m in [("n", "yolo26n-cls"), ("s", "yolo26s-cls"), ("m", "yolo26m-cls"), ("l", "yolo26l-cls"), ("x", "yolo26x-cls")]:
        add(f"cd_{tag}_ft", model=m)
    add("cd_y11s_ft", model="yolo11s-cls")

    # training method (on yolo26s-cls)
    add("cd_s_zeroshot", mode="zeroshot", epochs=0)
    add("cd_s_linear", mode="linear")
    add("cd_s_scratch", mode="scratch", epochs=60)

    # learning-rate sweep (1e-3 is cd_s_ft)
    for lr in (3e-3, 3e-4, 1e-4):
        add(f"cd_s_ft_lr{lr:g}", lr0=lr)

    # augmentation ablation (strong is cd_s_ft)
    for a in ("none", "basic", "strong_rot"):
        add(f"cd_s_ft_aug-{a}", aug=a)

    # input resolution (224 is cd_s_ft)
    for sz in (128, 160, 320):
        add(f"cd_s_ft_sz{sz}", imgsz=sz)

    # amount of training data (1.0 is cd_s_ft): 1000 / 2000 / 5000 / 10000 images
    for fr in (0.05, 0.1, 0.25, 0.5):
        add(f"cd_s_ft_frac{fr:g}", fraction=fr)

    # ---------------------------------------------------------------- CIFAR-10 (balanced)
    c10 = dict(dataset="cifar10", epochs=20, imgsz=128)
    add("c10_s_ft_sz64", **{**c10, "imgsz": 64})
    add("c10_s_ft", **c10)
    add("c10_n_ft", model="yolo26n-cls", **c10)
    add("c10_m_ft", model="yolo26m-cls", **c10)
    add("c10_s_scratch", mode="scratch", **{**c10, "epochs": 60})

    # ---------------------------------------------------------------- CIFAR-10 (imbalanced train set)
    imb = dict(epochs=20, imgsz=128)
    add("c10imb_s_ce", dataset="cifar10_imb", **imb)  # baseline: plain cross-entropy
    add("c10imb_s_oversample", dataset="cifar10_imb_os", **imb)  # minority images repeated
    add("c10imb_s_wce-inv", dataset="cifar10_imb", loss="wce_inv", **imb)  # 1/n_c class weights
    add("c10imb_s_wce-eff", dataset="cifar10_imb", loss="wce_eff", **imb)  # effective-number weights
    add("c10imb_s_focal", dataset="cifar10_imb", loss="focal", **imb)

    return {e.name: e for e in exps}


EXPS = _build()

GROUPS = {
    "smoke": ["cd_smoke"],
    "models": ["cd_n_ft", "cd_s_ft", "cd_m_ft", "cd_l_ft", "cd_x_ft", "cd_y11s_ft"],
    "methods": ["cd_s_zeroshot", "cd_s_linear", "cd_s_ft", "cd_s_scratch"],
    "lr": ["cd_s_ft", "cd_s_ft_lr0.003", "cd_s_ft_lr0.0003", "cd_s_ft_lr0.0001"],
    "aug": ["cd_s_ft", "cd_s_ft_aug-none", "cd_s_ft_aug-basic", "cd_s_ft_aug-strong_rot"],
    "imgsz": ["cd_s_ft_sz128", "cd_s_ft_sz160", "cd_s_ft", "cd_s_ft_sz320"],
    "data": ["cd_s_ft_frac0.05", "cd_s_ft_frac0.1", "cd_s_ft_frac0.25", "cd_s_ft_frac0.5", "cd_s_ft"],
    "cifar": ["c10_s_ft_sz64", "c10_s_ft", "c10_n_ft", "c10_m_ft", "c10_s_scratch"],
    "cifar_imb": ["c10_s_ft", "c10imb_s_ce", "c10imb_s_oversample", "c10imb_s_wce-inv", "c10imb_s_wce-eff", "c10imb_s_focal"],
}
# Dogs-vs-Cats ablations end to end, then CIFAR-10
GROUPS["catsdogs"] = list(dict.fromkeys(sum((GROUPS[g] for g in ("models", "methods", "lr", "aug", "imgsz", "data")), [])))
GROUPS["all"] = list(dict.fromkeys(GROUPS["catsdogs"] + GROUPS["cifar"] + GROUPS["cifar_imb"]))

# Splits scored after training. CIFAR-10 has a labelled test split; Dogs-vs-Cats test is unlabelled.
DATASET_EVAL = {
    "catsdogs": ["val"],
    "cifar10": ["val", "test"],
    "cifar10_imb": ["val", "test"],
    "cifar10_imb_os": ["val", "test"],
}


def seed_variants(base: str, seeds=(1, 2)):
    """Extra seeds of an existing experiment, for the mean +/- std over runs required by the brief."""
    b = EXPS[base]
    return [replace(b, name=f"{base}__seed{s}", seed=s) for s in seeds]
