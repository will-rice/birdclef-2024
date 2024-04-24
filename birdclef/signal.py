"""Signal processing functions."""

from typing import Optional

import torch
import torchaudio
from kymatio.scattering1d.frontend.torch_frontend import ScatteringTorch1D
from torch import nn
from torchvision.transforms import v2
from transformers.utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class MelSpectrogram(nn.Module):
    """Mel spectrogram computation."""

    def __init__(
        self,
        sample_rate: int = 32000,
        n_fft: int = 2048,
        win_length: int = 2048,
        hop_length: int = 512,
        f_min: float = 20.0,
        f_max: Optional[float] = 16000.0,
        n_mels: int = 128,
        normalized: bool = True,
        norm: str = "slaney",
        mel_scale: str = "slaney",
        center: bool = True,
        image_normalize: bool = True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.n_mels = n_mels
        self.normalized = normalized
        self.mel_fn = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            normalized=normalized,
            norm=norm,
            mel_scale=mel_scale,
            center=center,
        )
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
        self.image_normalize = image_normalize
        self.image_norm = v2.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.mel_fn(x)
        x = self.amp_to_db(x)
        x = self.normalize(x)
        if self.image_normalize:
            x = self.image_norm(x)
        return x

    @staticmethod
    def normalize(x: torch.Tensor) -> torch.Tensor:
        """Scale mel spectrogram."""
        mean = x.mean()
        std = x.std()
        x = torch.where(std == 0, x - mean, (x - mean) / std)
        min_val = x.min()
        max_val = x.max()
        x = torch.where(
            (max_val - min_val) == 0,
            x - min_val,
            (x - min_val) / (max_val - min_val),
        )
        return x


class Spectrogram(nn.Module):
    """Spectrogram computation."""

    def __init__(
        self,
        sample_rate: int = 32000,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 320,
        normalized: bool = True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.normalized = normalized
        self.spectrogram_fn = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            normalized=normalized,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.spectrogram_fn(x)
        x = self.normalize(x)
        return x

    @staticmethod
    def normalize(x: torch.Tensor) -> torch.Tensor:
        """Scale spectrogram."""
        mean = x.mean()
        std = x.std()
        x = torch.where(std == 0, x - mean, (x - mean) / std)
        min_val = x.min()
        max_val = x.max()
        x = torch.where(
            (max_val - min_val) == 0,
            x - min_val,
            (x - min_val) / (max_val - min_val),
        )
        return x


class JointTimeFrequencyScatter(nn.Module):
    """Joint time-frequency scattering."""

    def __init__(self, j: int = 6, q: int = 16, t: int = 160000) -> None:
        super().__init__()
        self.scatter_fn = ScatteringTorch1D(J=j, shape=t, Q=q)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.scatter_fn(x)
        x = self.normalize(x)
        x = x.tile(3, 1, 1)
        return x

    @staticmethod
    def normalize(x: torch.Tensor) -> torch.Tensor:
        """Scale spectrogram."""
        mean = x.mean()
        std = x.std()
        x = torch.where(std == 0, x - mean, (x - mean) / std)
        min_val = x.min()
        max_val = x.max()
        x = torch.where(
            (max_val - min_val) == 0,
            x - min_val,
            (x - min_val) / (max_val - min_val),
        )
        return x
