"""BirdCLEF DataModule."""

from typing import Any

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split


class BirdCLEFDataModule(LightningDataModule):
    """BirdCLEF DataModule."""

    def __init__(self, dataset: Dataset, num_workers: int = 4, batch_size: int = 12):
        super().__init__()

        self._dataset = dataset
        self._batch_size = batch_size
        self._num_workers = num_workers

    def setup(self, stage: str) -> None:
        """Initialize datasets."""
        self._train_dataset, self._val_dataset = split_dataset(self._dataset, 0.8)

    def train_dataloader(self) -> DataLoader:
        """Train dataloader."""
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=False,
        )

    def val_dataloader(self) -> DataLoader:
        """Validate dataloader."""
        return DataLoader(
            self._val_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            drop_last=True,
            shuffle=False,
            pin_memory=False,
        )


def split_dataset(dataset: Any, split: Any) -> Any:
    """Split dataset into train and validation."""
    train_set_size = int(len(dataset) * split)
    val_set_size = len(dataset) - train_set_size
    train_samples, val_samples = random_split(
        dataset, lengths=[train_set_size, val_set_size]
    )
    return train_samples, val_samples
