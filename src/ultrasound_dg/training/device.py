import torch


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        if torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")

    return torch.device(requested)
