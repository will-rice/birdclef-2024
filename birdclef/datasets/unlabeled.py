"""Unlabeled dataset for BirdCLEF 2024."""

import random
from pathlib import Path
from typing import NamedTuple

import torch
import torchaudio
from torch.utils.data import Dataset
from torchvision.transforms import v2

from birdclef.signal import MelSpectrogram


class Batch(NamedTuple):
    """Batch of inputs."""

    clean_specs: torch.Tensor
    noisy_specs: torch.Tensor


class UnlabeledDataset(Dataset):
    """Unlabeled dataset for BirdCLEF 2024."""

    def __init__(self, root: Path) -> None:
        super().__init__()
        self.root = root
        self.paths = list(self.root.glob("**/*.ogg"))
        self.mel_fn = MelSpectrogram()
        self.max_length = 32000 * 5
        self.spec_augmentations = v2.Compose(
            [v2.RandomErasing(p=1.0, scale=(0.5, 0.6))]
        )

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.paths)

    def __getitem__(self, idx: int) -> Batch:
        """Get a sample from the dataset."""
        path = self.paths[idx]
        audio, sr = torchaudio.load(str(path))

        if audio.shape[1] < self.max_length:
            num_repeats = self.max_length // audio.shape[1] + 1
            audio = audio.repeat(1, num_repeats)
            audio = audio[:, : self.max_length]
        else:
            start_idx = random.randint(0, audio.shape[1] - self.max_length)
            audio = audio[:, start_idx : start_idx + self.max_length]

        clean_spec = self.mel_fn(audio)
        noisy_spec = self.spec_augmentations(clean_spec.clone())

        return Batch(clean_specs=clean_spec, noisy_specs=noisy_spec)
