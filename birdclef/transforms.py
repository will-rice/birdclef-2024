"""Transforms for the data augmentation."""

import random

import torch
import torchaudio
from torch import nn


class SpecFreqMask(nn.Module):
    """Frequency Mask."""

    def __init__(self, num_masks: int, size: int, p: float = 0.5) -> None:
        super().__init__()
        self.num_masks = num_masks
        self.size = size
        self.p = p
        self.transform = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=self.size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward Pass."""
        if random.random() <= self.p:
            for _ in range(self.num_masks):
                x = self.transform(x)
        return x


class SpecTimeMask(nn.Module):
    """Time Mask."""

    def __init__(self, num_masks: int, size: int, p: float = 0.5) -> None:
        super().__init__()
        self.num_masks = num_masks
        self.size = size
        self.p = p
        self.transform = torchaudio.transforms.TimeMasking(
            time_mask_param=self.size,
            p=1.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward Pass."""
        if random.random() <= self.p:
            for _ in range(self.num_masks):
                x = self.transform(x)
        return x
