"""Transforms for the data augmentation."""

import torch
from audiomentations import Mp3Compression
from torch import nn


class TorchMP3Compression(nn.Module):
    """MP3 compression transform."""

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.transform = Mp3Compression(p=p)

    def forward(self, x: torch.Tensor, sample_rate: int = 32000) -> torch.Tensor:
        """Forward pass."""
        x = x.numpy()
        x = self.transform(x, sample_rate=sample_rate)
        x = torch.from_numpy(x.copy())
        return x
