"""Transforms for the data augmentation."""

import random
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torchaudio
from audiomentations import (
    AddBackgroundNoise,
    AddColorNoise,
    AddGaussianNoise,
    AddShortNoises,
    Compose,
    Gain,
    PitchShift,
    Reverse,
    Shift,
    TimeMask,
    TimeStretch,
)
from torch import nn


class SpecFreqMask(nn.Module):
    """Frequency Mask."""

    def __init__(self, num_masks: int, size: int, p: float = 0.5) -> None:
        super().__init__()
        self.num_masks = num_masks
        self.size = size
        self.p = p
        self.transform = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=self.size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward Pass."""
        if random.random() <= self.p:
            for _ in range(self.num_masks):
                x = self.transform(x)
        return x


class SpecTimeMask(nn.Module):
    """Time Mask."""

    def __init__(self, num_masks: int, size: int, p: float = 0.5) -> None:
        super().__init__()
        self.num_masks = num_masks
        self.size = size
        self.p = p
        self.transform = torchaudio.transforms.TimeMasking(
            time_mask_param=self.size,
            p=1.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward Pass."""
        if random.random() <= self.p:
            for _ in range(self.num_masks):
                x = self.transform(x)
        return x


class NoiseFromFile(nn.Module):
    """Add background noise from random file."""

    def __init__(
        self,
        root: Path,
        p: float = 0.5,
        sample_rate: int = 32000,
        num_samples: int = 1000,
    ) -> None:
        super().__init__()
        self.root = root
        self.p = p
        self.sample_rate = sample_rate
        self.noises = list(root.glob("**/*.ogg"))
        print(f"Loaded {len(self.noises)} noises")

    def forward(
        self, x: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """Forward Pass."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        if random.random() < self.p:
            noise_path = str(random.choice(self.noises))
            noise = torchaudio.load(noise_path)[0].to(x.device)
            start_idx = random.randint(0, noise.size(1) - x.size(1))
            noise = noise[:, start_idx : start_idx + x.size(1)]
            x = x + noise

        return x


class MultilabelMixUp(nn.Module):
    """Mixup for multilabel classification."""

    def __init__(
        self, p: float = 0.5, alpha: float = 1.0, beta: Optional[float] = None
    ) -> None:
        super().__init__()
        self.p = p
        self.alpha = alpha
        if not beta:
            beta = alpha
        self.dist = torch.distributions.beta.Beta(alpha, beta)

    @torch.no_grad()
    def forward(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        if random.random() <= self.p:
            lam = self.dist.sample()
            mixed_x = lam * images + (1 - lam) * images.roll(1, 0)
            mixed_y = lam * labels + (1 - lam) * labels.roll(1, 0)
            return mixed_x, mixed_y
        return images, labels


class AudiomentationsTransform:
    """Audiomentations transform."""

    def __init__(
        self,
        add_pitch_shift: bool = False,
        add_time_stretch: bool = False,
        add_background_noise: bool = False,
        add_short_noises: bool = False,
        add_time_mask: bool = False,
        add_gain: bool = False,
        add_gaussian_noise: bool = False,
        add_color_noise: bool = False,
        background_noise_root: Optional[Path] = None,
        short_noises_root: Optional[Path] = None,
    ) -> None:
        transforms = [Reverse(p=0.5), Shift(p=0.5)]
        if add_pitch_shift:
            transforms.append(PitchShift(p=0.5))
        if add_time_stretch:
            transforms.append(TimeStretch(p=0.5))
        if add_background_noise:
            transforms.append(AddBackgroundNoise(background_noise_root, p=0.1))
        if add_short_noises:
            transforms.append(AddShortNoises(short_noises_root, p=0.1))
        if add_time_mask:
            transforms.append(TimeMask(p=0.5))
        if add_gain:
            transforms.append(Gain(p=0.5))
        if add_gaussian_noise:
            transforms.append(AddGaussianNoise(p=0.5))
        if add_color_noise:
            transforms.append(AddColorNoise(p=0.5))

        self.transform = Compose(transforms=transforms, p=1.0, shuffle=True)

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """Call."""
        audio = self.transform(audio.numpy().squeeze(0), sample_rate=32000)
        audio = torch.from_numpy(audio.copy()).unsqueeze(0)
        return audio
