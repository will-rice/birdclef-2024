"""Modules."""

from typing import Union

import torch
from torch import nn


class GlobalAvgPool(nn.Module):
    """Average pooling."""

    def __init__(self, dim: Union[int, tuple[int, ...]] = 1):
        super().__init__()

        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return x.mean(dim=self.dim)


class GlobalAvgPool1D(GlobalAvgPool):
    """Average pooling."""

    def __init__(self) -> None:
        super().__init__(dim=2)


class GlobalAvgPool2D(GlobalAvgPool):
    """Average pooling."""

    def __init__(self) -> None:
        super().__init__(dim=(2, 3))


class GeMPool2D(nn.Module):
    """GeM Pooling.

    From: https://amaarora.github.io/posts/2020-08-30-gempool.html#gem-pooling

    """

    def __init__(self, p: int = 3, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return (
            nn.functional.avg_pool2d(
                x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))
            )
            .pow(1.0 / self.p)
            .squeeze()
        )


class GeMPool1D(nn.Module):
    """GeM Pooling.

    From: https://amaarora.github.io/posts/2020-08-30-gempool.html#gem-pooling

    """

    def __init__(self, p: int = 3, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return (
            nn.functional.avg_pool1d(x.clamp(min=self.eps).pow(self.p), (x.size(1)))
            .pow(1.0 / self.p)
            .squeeze()
        )
