"""BirdCLEF Dataset."""

import random
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import v2

from birdclef import utils
from birdclef.signal import MelSpectrogram
from birdclef.transforms import AudiomentationsTransform


class Batch(NamedTuple):
    """Batch of inputs."""

    audio: torch.Tensor
    specs: torch.Tensor
    labels: torch.Tensor


class BirdCLEF2024DatasetPlus(Dataset):
    """BirdCLEF Dataset."""

    def __init__(
        self,
        root: Path,
        augment: bool = False,
        max_seconds: int = 5,
        random_crop: bool = False,
        use_end: bool = False,
        use_secondary_labels: bool = False,
        secondary_coefficient: float = 1.0,
        add_nocall: bool = False,
        nocall_p: float = 0.2,
    ) -> None:
        super().__init__()
        self.root = root
        self.metadata = pd.read_csv(root / "train_metadata.csv")
        self.samples = list(root.glob("train_audio/**/*.ogg"))
        self.samples += list(root.glob("train_audio_plus/**/*.wav"))
        self.samples = [p for p in self.samples if p.parent.name != "nocall"]
        self.labels = sorted({p.parent.name for p in self.samples})
        assert len(self.labels) == 182
        self.all_labels = [p.parent.name for p in self.samples]
        self.max_length = 32000 * max_seconds
        self.random_crop = random_crop
        self.use_secondary_labels = use_secondary_labels
        self.secondary_coefficient = secondary_coefficient
        self.use_end = use_end
        self.add_nocall = add_nocall
        self.nocall_p = nocall_p
        self.augment = augment

        if self.add_nocall:
            self.nocall = sorted(
                (root / "labeled_soundscapes" / "nocall").glob("**/*.flac")
            )

        self.waveform_augmentations = AudiomentationsTransform(
            background_noise_root=root / "labeled_soundscapes" / "nocall",
            short_noises_root=root / "labeled_soundscapes" / "nocall",
            add_pitch_shift=True,
            add_time_stretch=True,
            add_background_noise=False,
            add_short_noises=False,
            add_time_mask=False,
            add_gain=False,
            add_gaussian_noise=False,
            add_color_noise=False,
        )
        self.spec_augmentations = v2.Compose(
            [v2.RandomHorizontalFlip(p=0.5), v2.RandomErasing(p=0.5)]
        )

        self.spec_fn = MelSpectrogram()

    def __len__(self) -> int:
        """Length."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Batch:
        """Get item."""
        path = self.samples[idx]

        label = path.parent.stem
        label_id = torch.tensor(self.labels.index(label))
        label_one_hot = nn.functional.one_hot(
            label_id, num_classes=len(self.labels)
        ).float()

        if self.use_secondary_labels:
            sample = self.metadata[
                self.metadata.filename == path.parent.name + "/" + path.name
            ]
            if len(sample) > 0:
                sample = sample.iloc[0]
                for secondary_label in eval(sample.secondary_labels):
                    if secondary_label in set(self.labels):
                        secondary_label_id = self.labels.index(secondary_label)
                        label_one_hot[secondary_label_id] = self.secondary_coefficient

        audio = self.load_audio(path)

        if self.add_nocall and random.random() <= self.nocall_p:
            path = random.choice(self.nocall)
            audio = self.load_audio(path)
            label_one_hot = torch.zeros_like(label_one_hot)

        if self.augment:
            audio = self.waveform_augmentations(audio)

        spec = self.spec_fn(audio)

        if self.augment:
            spec = self.spec_augmentations(spec)

        return Batch(audio=audio, specs=spec, labels=label_one_hot)

    def load_audio(self, path: Path) -> torch.Tensor:
        """Load sample."""
        audio = utils.load_audio(
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
