import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from aidm_project.config import Config
from aidm_project.data import build_loaders
from aidm_project.device import get_device
from aidm_project.evaluation import evaluate_checkpoint, save_metrics
from aidm_project.models import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="outputs/metrics/validation.json")
    args = parser.parse_args()
    device = get_device()
    config = Config.from_yaml(args.config)
    _, val_loader = build_loaders(args.data_root, config)
    model = build_model(config.model, config.num_classes, config.pretrained,
                        config.dropout, config.fine_tuning)
    metrics = evaluate_checkpoint(model, val_loader, args.checkpoint, device)
    save_metrics(metrics, args.output)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
