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
    Mp3Compression,
    PolarityInversion,
    Reverse,
    Shift,
    TimeMask,
)
from torch.utils.data import Dataset


class Batch(NamedTuple):
    """Batch of inputs."""

    audio: torch.Tensor
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
        self.max_length = 32000 * 5
        self.apply_augmentation = Compose(
            transforms=[
                Gain(p=0.5),
                PolarityInversion(p=0.5),
                AddGaussianNoise(p=0.5),
                AddColorNoise(p=0.5),
                Shift(p=0.5),
                TimeMask(p=0.5),
                Mp3Compression(p=0.5),
                Reverse(p=0.5),
            ]
        )
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

        if self.transform:
            # Apply augmentation
            audio = self.apply_augmentation(audio.numpy(), sample_rate=32000)
            audio = torch.from_numpy(audio.copy())

        label_id = self.label_map[sample.primary_label]

        return Batch(
            audio=audio,
            label_id=torch.tensor(label_id),
            lengths=torch.tensor(audio_length),
        )
