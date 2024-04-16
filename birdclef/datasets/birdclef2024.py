"""BirdCLEF Dataset."""

import random
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import torch
import torchaudio
from audiomentations import (
    AddColorNoise,
    AddGaussianNoise,
    Compose,
    Gain,
    PitchShift,
    PolarityInversion,
    Reverse,
    Shift,
    TimeMask,
)
from torch import nn
from torch.utils.data import Dataset

from birdclef.signal import MelSpectrogram
from birdclef.transforms import SpecFreqMask, SpecTimeMask


class Batch(NamedTuple):
    """Batch of inputs."""

    audio: torch.Tensor
    specs: torch.Tensor
    label_id: torch.Tensor
    lengths: torch.Tensor


class BirdCLEF2024Dataset(Dataset):
    """BirdCLEF Dataset."""

    def __init__(self, root: Path) -> None:
        super().__init__()
        self.root = root
        self.metadata = pd.read_csv(root / "train_metadata.csv")
        self.label_map = {
            v: k for k, v in enumerate(sorted(self.metadata.primary_label.unique()))
        }
        self.labels = list(self.label_map.keys())
        self.max_length = 32000 * 30
        self.waveform_augmentations = Compose(
            transforms=[
                Gain(p=0.5),
                PolarityInversion(p=0.5),
                AddGaussianNoise(p=0.5),
                AddColorNoise(p=0.5),
                Shift(p=0.5),
                TimeMask(p=0.5),
                Reverse(p=0.5),
                PitchShift(min_semitones=-5.0, max_semitones=5.0, p=0.5),
            ]
        )
        self.spec_augmentations = nn.Sequential(
            SpecFreqMask(num_masks=3, size=10, p=0.3),
            SpecTimeMask(num_masks=3, size=20, p=0.3),
        )
        self.mel_fn = MelSpectrogram()
        self.transform = True

    def __len__(self) -> int:
        """Length."""
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Batch:
        """Get item."""
        sample = self.metadata.iloc[idx]
        audio, sr = torchaudio.load(str(self.root / "train_audio" / sample.filename))

        if audio.shape[1] < self.max_length:
            diff = self.max_length // audio.shape[1] + 1
            audio = audio.repeat(1, diff)
            audio = audio[:, : self.max_length]
        else:
            start_idx = random.randint(0, audio.shape[1] - self.max_length)
            audio = audio[:, start_idx : start_idx + self.max_length]

        audio_length = self.max_length

        spec = self.mel_fn(audio)

        if self.transform:
            # Apply augmentation
            # audio = self.waveform_augmentations(audio.numpy(), sample_rate=32000)
            # audio = torch.from_numpy(audio.copy())
            spec = self.spec_augmentations(spec)

        label_id = self.label_map[sample.primary_label]

        return Batch(
            audio=audio,
            specs=spec,
            label_id=torch.tensor(label_id),
            lengths=torch.tensor(audio_length),
        )
