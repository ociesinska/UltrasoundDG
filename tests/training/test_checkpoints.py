from pathlib import Path

import torch
from torch import nn

from ultrasound_dg.training.checkpoints import (
    read_checkpoint,
    restore_checkpoint,
    save_checkpoint,
)


def test_checkpoint_can_be_saved_without_optimizer(tmp_path: Path) -> None:
    """Support lightweight evaluation checkpoints without optimizer state."""
    model = nn.Linear(2, 1)
    path = tmp_path / "model.pt"

    save_checkpoint(
        path=path,
        model=model,
        optimizer=None,
        epoch=3,
        metrics={"macro_source_lesion_dice": 0.8},
        configs={"name": "test_experiment"},
    )

    checkpoint = read_checkpoint(path)

    assert checkpoint["epoch"] == 3
    assert "optimizer_state_dict" not in checkpoint
    assert checkpoint["metrics"]["macro_source_lesion_dice"] == 0.8


def test_lightweight_checkpoint_restores_model_weights(tmp_path: Path) -> None:
    """Restore model parameters from a checkpoint without optimizer state."""
    source_model = nn.Linear(2, 1)
    restored_model = nn.Linear(2, 1)
    path = tmp_path / "model.pt"

    save_checkpoint(
        path=path,
        model=source_model,
        optimizer=None,
        epoch=1,
        metrics={},
        configs={},
    )

    checkpoint = read_checkpoint(path)
    restore_checkpoint(
        checkpoint=checkpoint,
        model=restored_model,
        optimizer=None,
    )

    for source_parameter, restored_parameter in zip(
        source_model.parameters(),
        restored_model.parameters(),
        strict=True,
    ):
        assert torch.equal(source_parameter, restored_parameter)
