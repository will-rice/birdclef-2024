"""Pseudo labeled dataset."""

import random
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import torch
import torchaudio
from audiomentations import Compose, PitchShift, Reverse, Shift, TimeStretch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from transformers.utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from birdclef.signal import MelSpectrogram


class Batch(NamedTuple):
    """Batch of inputs."""

    audio: torch.Tensor
    specs: torch.Tensor
    labels: torch.Tensor


class PseudoLabeledDataset(Dataset):
    """Pseudo Labeled Dataset."""

    def __init__(
        self,
        root: Path,
        augment: bool = False,
        max_seconds: int = 5,
        multi_label_augment: bool = False,
        use_secondary_labels: bool = False,
        secondary_coefficient: float = 1.0,
        random_crop: bool = False,
        balanced: bool = False,
        use_end: bool = False,
    ) -> None:
        super().__init__()
        self.root = root
        train_metadata = pd.read_csv(root / "train_metadata.csv")
        self.labels = sorted(train_metadata.primary_label.unique())
        metadata = pd.read_csv(root / "labeled_soundscapes" / "metadata.csv")
        counts = metadata.primary_label.value_counts()
        self.metadata = metadata[~metadata.primary_label.isin(counts[counts < 2].index)]
        self.all_labels = self.metadata.primary_label.tolist()
        self.max_length = 32000 * max_seconds
        self.multi_label_augment = multi_label_augment
        self.use_secondary_labels = use_secondary_labels
        self.secondary_coefficient = secondary_coefficient
        self.random_crop = random_crop
        self.balanced = balanced
        self.use_end = use_end
        self.waveform_augmentations = Compose(
            transforms=[
                PitchShift(p=0.5),
                TimeStretch(p=0.5),
                Reverse(p=0.5),
                Shift(p=0.5),
            ],
            p=1.0,
            shuffle=True,
        )
        self.spec_augmentations = v2.Compose(
            [v2.RandomHorizontalFlip(p=0.5), v2.RandomErasing(p=0.5)]
        )

        self.normalize = v2.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        self.spec_fn = MelSpectrogram()
        self.augment = augment

    def __len__(self) -> int:
        """Length."""
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Batch:
        """Get item."""
        if self.balanced:
            random_label = random.choice(self.labels)
            sample = (
                self.metadata.loc[self.metadata.primary_label == random_label]
                .sample()
                .iloc[0]
            )
        else:
            sample = self.metadata.iloc[idx]

        label = sample.primary_label
        label_one_hot = torch.zeros(len(self.labels))
        if label in set(self.labels):
            label_id = torch.tensor(self.labels.index(label))
            label_one_hot[label_id] = 1.0

        if self.use_secondary_labels:
            for secondary_label in eval(sample.secondary_labels):
                if secondary_label in set(self.labels):
                    secondary_label_id = self.labels.index(secondary_label)
                    label_one_hot[secondary_label_id] = self.secondary_coefficient

        audio = self.load_audio(self.root / "labeled_soundscapes" / sample.filepath)

        if self.multi_label_augment:
            # This allows leakage from validation set
            num_secondary_samples = random.randint(0, 3)
            labels = self.labels.copy()
            labels.pop(labels.index(label))
            for _ in range(num_secondary_samples):
                random_label = random.choice(labels)
                secondary_sample = (
                    self.metadata.loc[self.metadata.primary_label == random_label]
                    .sample()
                    .iloc[0]
                )
                secondary_audio = self.load_audio(
                    self.root / "labeled_soundscapes" / secondary_sample.filepath
                )
                audio += secondary_audio
                label_one_hot[self.labels.index(random_label)] = (
                    self.secondary_coefficient
                )
                labels.pop(labels.index(random_label))

        if self.augment:
            audio = self.waveform_augmentations(
                audio.numpy().copy().squeeze(0), sample_rate=32000
            )
            audio = torch.from_numpy(audio.copy()).unsqueeze(0)

        spec = self.spec_fn(audio)
        spec = self.normalize(spec)

        if self.augment:
            spec = self.spec_augmentations(spec)

        return Batch(audio=audio, specs=spec, labels=label_one_hot)

    def load_audio(self, path: Path) -> torch.Tensor:
        """Load sample."""
        audio, sr = torchaudio.load(
            str(path), num_frames=-1 if self.random_crop else self.max_length
        )

        if audio.shape[1] < self.max_length:
            num_repeats = self.max_length // audio.shape[1] + 1
            audio = audio.repeat(1, num_repeats)
            audio = audio[:, : self.max_length]
        else:

            if self.random_crop:
                start_idx = random.randint(0, audio.shape[1] - self.max_length)
            elif self.use_end and random.random() <= 0.5:
                start_idx = audio.shape[1] - self.max_length
            else:
                start_idx = 0

            audio = audio[:, start_idx : start_idx + self.max_length]

        return audio

    @property
    def num_classes(self) -> int:
        """Number of classes."""
        return len(self.labels)
