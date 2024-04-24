"""ConvNext model definition."""

import torch
from torch import nn
from transformers import ASTConfig, ASTModel

from birdclef.modeling.modules import GeMPool1D
from birdclef.signal import MelSpectrogram


class ASTClassifier(nn.Module):
    """ConvNext model."""

    def __init__(self, dropout: float = 0.1, num_classes: int = 182) -> None:
        super().__init__()
        self.spec_fn = MelSpectrogram(image_normalize=False)
        self.config = ASTConfig(
            num_hidden_layers=4,
            hidden_size=256,
            num_attention_heads=6,
            intermediate_size=1024,
            attention_probs_dropout_prob=0.1,
            hidden_dropout_prob=dropout,
        )
        self.encoder = ASTModel(self.config)
        self.pool = GeMPool1D()
        self.norm = nn.LayerNorm(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.encoder.config.hidden_size, num_classes)

    def forward(self, x: torch.Tensor, from_audio: bool = True) -> torch.Tensor:
        """Forward pass."""
        if from_audio:
            x = self.spec_fn(x)
        x = x.squeeze(1)
        x = x.transpose(2, 1)
        x = self.encoder(x, return_dict=False)[0]
        x = self.pool(x)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.head(x)
        return x
