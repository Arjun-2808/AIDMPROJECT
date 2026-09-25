"""Model registry.

Every model is a plain `nn.Module` returning logits of shape (B, num_classes).
Add yours with @register("your_name") in your own file, then reference it from
a YAML config -- no other file needs to change. That is the whole point: three
people build in parallel without touching each other's code.
"""
from __future__ import annotations

from torch import nn

_REGISTRY: dict[str, callable] = {}


def register(name: str):
    def wrap(fn):
        if name in _REGISTRY:
            raise KeyError(f"model {name!r} already registered")
        _REGISTRY[name] = fn
        return fn
    return wrap


def build_model(name: str, num_classes: int = 2, pretrained: bool = True,
                **kwargs) -> nn.Module:
    if name not in _REGISTRY:
        raise KeyError(f"unknown model {name!r}. Available: {available()}")
    return _REGISTRY[name](num_classes=num_classes, pretrained=pretrained, **kwargs)


def available() -> list[str]:
    return sorted(_REGISTRY)


# Importing the modules is what populates the registry.
from . import cnn, vit, baseline  # noqa: E402,F401
