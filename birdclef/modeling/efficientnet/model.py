"""EfficientNet model definition."""

import torch
from torch import nn
from torchvision.transforms import v2
from transformers import EfficientNetModel
from transformers.utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from birdclef.modeling.modules import GeMPool2D
from birdclef.signal import MelSpectrogram


class EfficientNetClassifier(nn.Module):
    """ConvNext model."""

    def __init__(self, dropout: float = 0.2, num_classes: int = 182):
        super().__init__()
        self.mel_fn = MelSpectrogram()
        self.normalize = v2.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        self.encoder = EfficientNetModel.from_pretrained(
            "google/efficientnet-b0", dropout_rate=dropout
        )
        self.pool = GeMPool2D()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(self.encoder.config.hidden_dim)
        self.head = nn.Linear(self.encoder.config.hidden_dim, num_classes)
        self.image_size = (
            self.encoder.config.image_size,
            self.encoder.config.image_size,
        )

    def forward(self, x: torch.Tensor, from_audio: bool = True) -> torch.Tensor:
        """Forward pass."""
        if from_audio:
            x = self.mel_fn(x)
            x = self.normalize(x)
        x = nn.functional.interpolate(x, size=self.image_size, mode="bilinear")
        x = self.encoder(x, return_dict=False)[0]
        x = self.pool(x)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.head(x)
        return x
