"""ConvNext model definition."""

import torch
from torch import nn
from transformers import AutoModel

from birdclef.modeling.modules import GeMPool2D
from birdclef.signal import MelSpectrogram


class ConvNextV2Classifier(nn.Module):
    """ConvNext model."""

    def __init__(
        self,
        dropout: float = 0.1,
        num_classes: int = 182,
        model_name: str = "facebook/convnextv2-nano-22k-224",
    ) -> None:
        super().__init__()
        self.spec_fn = MelSpectrogram()
        self.encoder = AutoModel.from_pretrained(model_name, drop_path_rate=dropout)
        self.pool = GeMPool2D()
        self.norm = nn.LayerNorm(self.encoder.config.hidden_sizes[-1])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.encoder.config.hidden_sizes[-1], num_classes)
        self.image_size = (
            self.encoder.config.image_size,
            self.encoder.config.image_size,
        )

    def forward(self, x: torch.Tensor, from_audio: bool = True) -> torch.Tensor:
        """Forward pass."""
        if from_audio:
            x = self.spec_fn(x)
        x = nn.functional.interpolate(x, size=self.image_size, mode="bilinear")
        x = self.encoder(x, return_dict=False)[0]
        x = self.pool(x)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.head(x)
        return x
