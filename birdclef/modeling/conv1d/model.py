"""Conv1D model for BirdCLEF."""

import torch
from torch import nn

from birdclef.modeling.conv1d.modules import ResidualBlock1D
from birdclef.modeling.modules import GlobalAvgPool


class Conv1DClassifier(nn.Module):
    """Conv1D classifier model."""

    def __init__(self, num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv_in = nn.Conv1d(1, 32, kernel_size=1, stride=4)
        self.encoder = nn.Sequential(
            ResidualBlock1D(32, 64, kernel_size=7, dropout=dropout),
            ResidualBlock1D(64, 128, kernel_size=5, dropout=dropout),
            ResidualBlock1D(128, 256, kernel_size=3, dropout=dropout),
            ResidualBlock1D(256, 512, kernel_size=3, dropout=dropout),
            ResidualBlock1D(512, 1024, kernel_size=3, dropout=dropout),
        )
        self.pool = GlobalAvgPool(dim=-1)
        self.norm = nn.LayerNorm(1024)
        self.head = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor, from_audio: bool = True) -> torch.Tensor:
        """Forward pass."""
        x = self.conv_in(x)
        x = self.encoder(x)
        x = self.pool(x)
        x = x.squeeze(-1)
        x = self.norm(x)
        x = self.head(x)
        return x
