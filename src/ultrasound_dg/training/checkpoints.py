from pathlib import Path
from typing import Any

import torch
from torch import nn


def read_checkpoint(
    path: Path, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    """Read checkpoint contents without mutating a model or optimizer."""
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}.")

    return torch.load(path, map_location=map_location, weights_only=True)


def restore_checkpoint(
    checkpoint: dict[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    """Restore model weights and, when supplied, optimizer state in place."""

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> dict:
    """Read a checkpoint, restore supplied objects, and return its metadata."""

    checkpoint = read_checkpoint(path=path, map_location=device)

    restore_checkpoint(checkpoint=checkpoint, model=model, optimizer=optimizer)

    return checkpoint


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    configs: dict[str, Any],
) -> None:
    """Persist training state, validation metrics, and experiment configuration."""

    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "configs": configs,
    }

    torch.save(checkpoint, path)
