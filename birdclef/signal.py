"""Signal processing functions."""

from typing import Optional

import torch
import torchaudio
from torch import nn


class MelSpectrogram(nn.Module):
    """Mel spectrogram computation."""

    def __init__(
        self,
        sample_rate: int = 32000,
        n_fft: int = 2048,
        win_length: int = 2048,
        hop_length: int = 512,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        n_mels: int = 128,
        normalized: bool = True,
    ):
        super().__init__()
        self.mel_fn = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            normalized=normalized,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.mel_fn(x)
        return x
