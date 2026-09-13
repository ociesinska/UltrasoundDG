import torch
from torch import nn
from torch.nn import functional as F


class BCEDiceLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        smooth: float = 1e-6,
    ):

        super().__init__()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        bce = F.binary_cross_entropy_with_logits(logits, targets)

        dice = self._dice_loss(logits, targets)

        return self.bce_weight * bce + self.dice_weight * dice

    def _dice_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        probabilities = torch.sigmoid(logits)
        dims = tuple(range(1, probabilities.ndim))

        intersection = (probabilities * targets).sum(dim=dims)

        denominator = probabilities.sum(dim=dims) + targets.sum(dim=dims)

        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)

        return 1.0 - dice.mean()
