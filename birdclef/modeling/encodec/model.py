"""Encodec based classifier model."""

from typing import Optional

import torch
import torchaudio
from torch import nn
from transformers import AutoModel, T5Config, T5EncoderModel

from birdclef import utils
from birdclef.modeling.modules import AvgPool


class EncodecClassifier(nn.Module):
    """Encodec based classifier model."""

    def __init__(
        self,
        model_name: str = "facebook/encodec_24khz",
        num_classes: int = 182,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.resample = torchaudio.transforms.Resample(
            32000, self.encoder.config.sampling_rate
        )

        self.lm = T5EncoderModel(
            T5Config(
                vocab_size=self.encoder.config.codebook_size * 2,
                d_model=64,
                d_ff=int(64 * 4),
                num_layers=2,
                dropout_rate=dropout,
            )
        )
        self.lm = torch.compile(self.lm)
        self.pool = AvgPool()
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.lm.config.hidden_size, num_classes)

    def forward(
        self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass."""
        x = self.resample(x)
        codec_masks = (
            utils.sequence_mask(lengths, x.size(1)) if lengths is not None else None
        )

        with torch.no_grad():
            x = self.encoder.encode(x, padding_mask=codec_masks, return_dict=False)[0][
                0
            ].sum(1)

        attn_masks = (
            utils.sequence_mask(lengths // 320, x.size(1))
            if lengths is not None
            else None
        )
        x = self.lm(x, attention_mask=attn_masks, return_dict=False)[0]
        x = self.pool(x)
        x = self.dropout(x)
        x = self.head(x)
        return x
