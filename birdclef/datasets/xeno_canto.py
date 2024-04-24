"""Xeno Canto dataset."""

from pathlib import Path
from typing import Dict, NamedTuple

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from birdclef import utils
from birdclef.signal import MelSpectrogram
from birdclef.transforms import AudiomentationsTransform


class Batch(NamedTuple):
    """Batch of inputs."""

    audio: torch.Tensor
    labels: torch.Tensor
    specs: torch.Tensor


class XenoCantoDataset(Dataset):
    """Xeno Canto dataset for bird audio."""

    def __init__(
        self,
        root: Path,
        augment: bool = True,
        max_seconds: int = 5,
        random_crop: bool = False,
        min_samples: int = 100,
    ) -> None:
        self.root = root
        self.paths = list(root.glob("**/*.flac"))
        counts: Dict[str, int] = {}
        for path in self.paths:
            counts[path.parent.name] = counts.get(path.parent.name, 0) + 1
        self.paths = [p for p in self.paths if counts[p.parent.name] >= min_samples]

        self.labels = sorted({p.parent.name for p in self.paths})
        self.all_labels = [p.parent.name for p in self.paths]

        self.max_length = 32000 * max_seconds
        self.random_crop = random_crop
        self.mel_fn = MelSpectrogram()
        self.augment = augment
        self.sample_weights = [1.0 for _ in self.labels]
        self.waveform_augmentations = AudiomentationsTransform(
            background_noise_root=root.parent
            / "birdclef-2024"
            / "labeled_soundscapes"
            / "nocall",
            short_noises_root=root.parent
            / "birdclef-2024"
            / "labeled_soundscapes"
            / "nocall",
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

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.paths)

    def __getitem__(self, idx: int) -> Batch:
        """Get a sample from the dataset."""
        path = self.paths[idx]
        audio = utils.load_audio(str(path), num_frames=self.max_length)

        if audio.shape[1] < self.max_length:
            num_repeats = self.max_length // audio.shape[1] + 1
            audio = audio.repeat(1, num_repeats)
            audio = audio[:, : self.max_length]
        else:
            audio = audio[:, : self.max_length]

        if self.augment:
            audio = self.waveform_augmentations(audio)

        spec = self.mel_fn(audio)

        if self.augment:
            spec = self.spec_augmentations(spec)

        label = self.labels.index(path.parent.name)
        label_one_hot = torch.nn.functional.one_hot(
            torch.tensor(label), num_classes=len(self.labels)
        ).float()

        return Batch(audio=audio, labels=label_one_hot, specs=spec)

    @property
    def num_classes(self) -> int:
        """Get the number of classes in the dataset."""
        return len(self.labels)
