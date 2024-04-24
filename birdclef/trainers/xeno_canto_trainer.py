"""Xeno Canto trainer."""

import torch
import wandb
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SubsetRandomSampler

from birdclef.trainers import Trainer


class XenoCantoTrainer(Trainer):
    """Xeno Canto trainer."""

    def fit(self) -> None:
        """Fit the model."""
        self.on_fit_begin()
        for _ in range(self.num_epochs):
            self.on_epoch_begin()
            train_ids, val_ids = train_test_split(
                range(len(self.train_dataset)),
                test_size=0.2,
                random_state=self.seed,
                shuffle=True,
                stratify=self.train_dataset.all_labels,
            )
            train_sampler = SubsetRandomSampler(train_ids, generator=self.rng)
            val_sampler = SubsetRandomSampler(val_ids, generator=self.rng)
            train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                sampler=train_sampler,
            )
            val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                sampler=val_sampler,
            )

            self.train(train_dataloader)
            self.validate(val_dataloader)
            self.on_epoch_end()

        self.on_fit_end()

    def on_fit_begin(self) -> None:
        """Fit begin."""
        wandb.init(
            project="birdclef-2024",
            group=self.log_path.name,
            name=self.log_path.name,
            mode="offline" if self.debug else "online",
        )

    def save_model(self) -> None:
        """Save model."""
        torch.save(self.model.state_dict(), self.log_path / "xeno_canto_model.pt")
