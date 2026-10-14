import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from aidm_project.gradcam import grad_cam
from aidm_project.models import build_model


def test_resnet_binary_output():
    model = build_model("resnet18", 2, False, 0.2, "full")
    output = model(torch.randn(2, 3, 224, 224))
    assert output.shape == (2, 2)


def test_cifar_model_upsamples_small_images():
    model = build_model("resnet18", 10, False, 0.2, "full", upsample_to=64)
    output = model(torch.randn(2, 3, 32, 32))
    assert output.shape == (2, 10)


def test_grad_cam_maps_are_normalised():
    model = build_model("resnet18", 2, False, 0.2, "classifier_only")
    for p in model.parameters():  # mimic a frozen pretrained backbone
        p.requires_grad = False
    cam = grad_cam(model, torch.randn(2, 3, 160, 160))
    assert cam.shape == (2, 5, 5)
    assert float(cam.min()) >= 0 and float(cam.max()) <= 1
