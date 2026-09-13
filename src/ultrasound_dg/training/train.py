import torch
from torch import nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:

    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["mask"].to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)

        loss = loss_fn(logits, targets)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples  # comparable average loss per epoch sample


def evaluate_loss(
    model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device
) -> float:
    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["mask"].to(device)

            logits = model(images)

            loss = loss_fn(logits, targets)

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples
