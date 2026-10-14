import json
from pathlib import Path

import torch
import torch.nn as nn

from .metrics import classification_metrics


def evaluate_checkpoint(model, loader, checkpoint: str | Path, device: torch.device) -> dict:
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    targets, predictions = [], []
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += criterion(logits, labels).item() * labels.size(0)
            targets.extend(labels.cpu().tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
    average = "binary" if len(set(targets)) <= 2 else "macro"
    result = classification_metrics(targets, predictions, average)
    result["loss"] = total_loss / len(loader.dataset)
    result["checkpoint_epoch"] = state.get("epoch")
    return result


def save_metrics(metrics: dict, output: str | Path) -> None:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
