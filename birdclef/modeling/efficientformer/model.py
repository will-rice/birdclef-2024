"""ConvNext model definition."""

import torch
from torch import nn
from torchvision.transforms import v2
from transformers import EfficientFormerModel
from transformers.utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD

from birdclef.signal import MelSpectrogram


class EfficientFormerClassifier(nn.Module):
    """EfficientFormer model."""

    def __init__(self, dropout: float = 0.2, num_classes: int = 182):
        super().__init__()
        self.mel_fn = MelSpectrogram()
        self.normalize = v2.Normalize(IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD)
        self.encoder = EfficientFormerModel.from_pretrained(
            "snap-research/efficientformer-l1-300", drop_path_rate=0.2
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.encoder.config.hidden_sizes[-1], num_classes)

    def forward(self, x: torch.Tensor, from_audio: bool = True) -> torch.Tensor:
        """Forward pass."""
        if from_audio:
            x = self.mel_fn(x)
        x = self.normalize(x)
        x = self.encoder(x, return_dict=False)[0]
        x = x.mean(1)
        x = self.dropout(x)
        x = self.head(x)
        return x
