import torch


def get_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this project, but no CUDA-enabled GPU is available. "
            "Install a CUDA-enabled PyTorch build and verify your NVIDIA drivers."
        )
    return torch.device("cuda")
