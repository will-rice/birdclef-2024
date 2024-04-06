"""Conv2d model for birdclef competition."""

from typing import Optional

import torch
from torch import nn

from birdclef.modeling.conv2d.modules import DownsampleBlock2D
from birdclef.signal import MelSpectrogram


class Conv2DModel(nn.Module):
    """Conv2d model."""

    def __init__(
        self, num_classes: int = 182, dropout: float = 0.2, in_kernel: int = 7
    ) -> None:
        super().__init__()
        self.mel_fn = MelSpectrogram()
        self.in_conv = nn.Conv2d(
            1, 32, kernel_size=in_kernel, stride=1, padding=in_kernel // 2
        )
        self.blocks = nn.Sequential(
            DownsampleBlock2D(32, 32, dropout=dropout),
            DownsampleBlock2D(32, 64, dropout=dropout),
            DownsampleBlock2D(64, 128, dropout=dropout),
            DownsampleBlock2D(128, 256, dropout=dropout),
            DownsampleBlock2D(256, 512, dropout=dropout),
            DownsampleBlock2D(512, 1024, dropout=dropout),
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor]) -> torch.Tensor:
        """Forward pass."""
        with torch.no_grad():
            x = self.mel_fn(x)
        x = self.in_conv(x)
        x = self.blocks(x)
        x = x.mean([2, 3])
        x = self.dropout(x)
        x = self.head(x)
        return x
