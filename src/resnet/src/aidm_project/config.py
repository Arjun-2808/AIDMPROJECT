from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    task: str = "dogs_cats"
    name: str = "best"
    model: str = "resnet18"
    pretrained: bool = True
    num_classes: int = 2
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 20
    patience: int = 5
    optimizer: str = "adamw"
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    momentum: float = 0.9
    dropout: float = 0.2
    fine_tuning: str = "layer4_plus_classifier"
    scheduler: str = "cosine"
    warmup_epochs: int = 2
    label_smoothing: float = 0.0
    seed: int = 2026
    num_workers: int = 2
    amp: bool = True
    cudnn_safe: bool = True
    imbalance: str = "none"
    method: str = "ce"
    minority: list[int] = field(default_factory=lambda: [3, 4, 9])
    minority_frac: float = 0.02
    limit: int = 0

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides: Any) -> "Config":
        values = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**values)
