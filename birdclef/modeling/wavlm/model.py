"""WavLM classifier model."""

import torch
import torchaudio
from torch import nn
from transformers import AutoModel

from birdclef.modeling.modules import AvgPool


class WavLMClassifier(nn.Module):
    """WavLM classifier model."""

    def __init__(
        self, model_name: str = "microsoft/wavlm-large", num_classes: int = 182
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, apply_spec_augment=False)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.resample = torchaudio.transforms.Resample(32000, 16000)

        self.pool = AvgPool()
        self.head = nn.Linear(self.encoder.config.hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.resample(x)
        with torch.no_grad():
            x = self.encoder(x).last_hidden_state
        x = self.pool(x)
        x = self.head(x)
        return x
