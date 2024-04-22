"""BirdCLEF 2024 loss functions."""

import torch
from torch import nn
from torchvision.ops.focal_loss import sigmoid_focal_loss


class FocalLoss(nn.Module):
    """Focal loss."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return sigmoid_focal_loss(
            inputs, targets, self.alpha, self.gamma, reduction="mean"
        )
