"""ConvNext model definition."""

from typing import Optional

import torch
from torch import nn
from torchvision.transforms import v2
from transformers import EfficientNetModel
from transformers.utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD

from birdclef.signal import MelSpectrogram


class EfficientNetClassifier(nn.Module):
    """ConvNext model."""

    def __init__(self, dropout: float = 0.2, num_classes: int = 182):
        super().__init__()
        self.mel_fn = MelSpectrogram()
        self.normalize = v2.Normalize(IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD)
        self.encoder = EfficientNetModel.from_pretrained("google/efficientnet-b2")
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.encoder.config.hidden_dim, num_classes)

    def forward(
        self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass."""
        x = self.normalize(x)
        x = self.encoder(x, return_dict=False)[1]
        x = self.dropout(x)
        x = self.head(x)
        return x

    def infer(self, x: torch.Tensor) -> torch.Tensor:
        """Inference pass."""
        with torch.no_grad():
            x = self.mel_fn(x)
            x = self.normalize(x)
        x = self.encoder(x, return_dict=False)[1]
        x = self.dropout(x)
        x = self.head(x)
        return x
