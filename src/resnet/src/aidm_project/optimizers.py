import torch


def build_optimizer(model, name: str, lr: float, weight_decay: float, momentum: float = 0.9,
                    beta1: float = 0.9, beta2: float = 0.999):
    parameters = [p for p in model.parameters() if p.requires_grad]
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay, betas=(beta1, beta2))
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay, betas=(beta1, beta2))
    if name == "nadam":
        return torch.optim.NAdam(parameters, lr=lr, weight_decay=weight_decay, betas=(beta1, beta2))
    if name in {"sgd", "sgd_nesterov"}:
        return torch.optim.SGD(parameters, lr=lr, momentum=momentum,
                               nesterov=True, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")

