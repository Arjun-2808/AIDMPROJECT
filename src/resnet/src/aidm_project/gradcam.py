import torch

from .models import backbone


def grad_cam(model, images: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
    """Selvaraju et al. 2017: layer4 activations weighted by the spatially averaged gradient of the
    target-class logit (default: predicted class). Returns maps in [0, 1], shape [B, h, w]."""
    layer = backbone(model).layer4
    store = {}

    def keep(module, inputs, output):
        store["acts"] = output
        output.register_hook(lambda grad: store.__setitem__("grads", grad))

    handle = layer.register_forward_hook(keep)
    model.eval()
    try:
        with torch.enable_grad():
            x = images.clone().requires_grad_(True)  # gradients reach layer4 even when it is frozen
            logits = model(x)
            target = logits.argmax(1) if target is None else target
            model.zero_grad(set_to_none=True)
            logits.gather(1, target[:, None]).sum().backward()
    finally:
        handle.remove()
    weights = store["grads"].mean((2, 3), keepdim=True)
    cam = torch.relu((weights * store["acts"]).sum(1)).detach()
    return cam / cam.amax((1, 2), keepdim=True).clamp_min(1e-12)
