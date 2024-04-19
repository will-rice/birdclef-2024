"""ConvNext model definition."""

from typing import Optional

import timm
import torch
from torch import nn
from torchvision.transforms import v2

from birdclef.signal import MelSpectrogram


class EfficientViTClassifier(nn.Module):
    """ConvNext model."""

    def __init__(self, dropout: float = 0.2, num_classes: int = 182):
        super().__init__()
        self.mel_fn = MelSpectrogram()
        self.normalize = v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        self.encoder = timm.create_model(
            "efficientvit_b1.r224_in1k",
            pretrained=True,
            num_classes=num_classes,
            drop_rate=dropout,
        )

    def forward(
        self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass."""
        with torch.no_grad():
            x = self.normalize(x)
        x = self.encoder(x)
        return x

    def infer(
        self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass."""
        with torch.no_grad():
            x = self.mel_fn(x)
            x = self.normalize(x)
        x = self.encoder(x)
        return x
