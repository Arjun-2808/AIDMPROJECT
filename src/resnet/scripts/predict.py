import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from aidm_project.config import Config
from aidm_project.data import build_test_loader
from aidm_project.device import get_device
from aidm_project.models import build_model
from aidm_project.training import predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = Config.from_yaml(args.config)
    device = get_device()
    model = build_model(config.model, config.num_classes, config.pretrained,
                        config.dropout, config.fine_tuning)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    probs, ids = predict(model, build_test_loader(args.data_root, config), device, config.amp, tta_flip=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "label"])  # 1 = dog, 0 = cat
        writer.writerows([int(i), int(p > 0.5)] for i, p in zip(ids, probs[:, 1]))


if __name__ == "__main__":
    main()
