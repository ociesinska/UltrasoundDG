import torch
from torch import nn

from ultrasound_dg.configs.config_schemas import TrainingConfig


def create_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    if config.optimizer == "AdamW":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    if config.optimizer == "Adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    raise ValueError(f"Unsupported optimizer: {config.optimizer}")
