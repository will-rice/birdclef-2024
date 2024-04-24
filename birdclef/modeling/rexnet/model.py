"""ConvNext model definition."""

import timm
import torch
from torch import nn
from torchvision.transforms import v2

from birdclef.modeling.modules import GeMPool2D
from birdclef.signal import MelSpectrogram


class ReXNetClassifier(nn.Module):
    """ConvNext model."""

    def __init__(self, dropout: float = 0.2, num_classes: int = 182):
        super().__init__()
        self.mel_fn = MelSpectrogram()

        self.encoder = timm.create_model(
            "timm/rexnet_150.nav_in1k",
            pretrained=True,
            num_classes=0,
            drop_rate=dropout,
        )
        self.pool = GeMPool2D()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(self.encoder.num_features)
        self.head = nn.Linear(self.encoder.num_features, num_classes)
        self.normalize = v2.Normalize(
            self.encoder.pretrained_cfg["mean"], self.encoder.pretrained_cfg["std"]
        )
        self.image_size = self.encoder.default_cfg["input_size"][-2:]

    def forward(self, x: torch.Tensor, from_audio: bool = True) -> torch.Tensor:
        """Forward pass."""
        if from_audio:
            x = self.mel_fn(x)
            x = self.normalize(x)
        x = nn.functional.interpolate(x, size=self.image_size, mode="bilinear")
        x = self.encoder.forward_features(x)
        x = self.pool(x)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.head(x)
        return x
