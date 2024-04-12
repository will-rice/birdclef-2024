"""Modules."""

from typing import Union

import torch
from torch import nn


class AvgPool(nn.Module):
    """Average pooling."""

    def __init__(self, dim: Union[int, tuple[int, ...]] = 1):
        super().__init__()

        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return x.mean(dim=self.dim)
