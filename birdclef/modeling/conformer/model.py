"""Conformer model."""

import torch
import torchaudio
from torch import nn
from torchvision.transforms import v2
from transformers.utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD

from birdclef.modeling.modules import GlobalAvgPool
from birdclef.signal import MelSpectrogram


class ConformerClassifier(nn.Module):
    """Conformer classifier."""

    def __init__(self, num_classes: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.mel_fn = MelSpectrogram()
        self.encoder = torchaudio.models.Conformer(
            input_dim=self.mel_fn.n_mels,
            num_heads=8,
            ffn_dim=192,
            num_layers=12,
            depthwise_conv_kernel_size=31,
            dropout=dropout,
            use_group_norm=True,
        )
        self.normalize = v2.Normalize(IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD)
        self.avg_pool = GlobalAvgPool(dim=1)
        self.layer_norm = nn.LayerNorm(self.mel_fn.n_mels)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.mel_fn.n_mels, num_classes)

    def forward(self, x: torch.Tensor, from_audio: bool = True) -> torch.Tensor:
        """Forward Pass."""
        if from_audio:
            x = self.mel_fn(x)
            x = self.normalize(x)
        x = x.mean(1).transpose(2, 1)
        lengths = torch.tensor([x.size(1)], device=x.device)
        lengths = lengths.repeat(x.size(0))
        x = self.encoder(x, lengths)[0]
        x = self.avg_pool(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


class ConformerModelForPreTraining(nn.Module):
    """Conformer model for pre-training."""

    def __init__(self, dropout: float = 0.2) -> None:
        super().__init__()
        self.encoder = torchaudio.models.Conformer(
            input_dim=128,
            num_heads=8,
            ffn_dim=192,
            num_layers=12,
            depthwise_conv_kernel_size=31,
            dropout=dropout,
            use_group_norm=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward Pass."""
        lengths = torch.tensor([x.size(1)], device=x.device)
        lengths = lengths.repeat(x.size(0))
        x = self.encoder(x, lengths)[0]
        return x
