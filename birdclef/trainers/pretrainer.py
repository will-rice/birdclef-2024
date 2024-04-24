"""Trainer module for training the model."""

import gc
from pathlib import Path

import torch
import wandb
from sklearn.model_selection import train_test_split
from timm.scheduler import CosineLRScheduler
from torch import GradScaler, nn
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torchmetrics import MeanMetric
from tqdm import tqdm

from birdclef.datasets.unlabeled import Batch


class PreTrainer:
    """Trainer for BirdCLEF2024 dataset."""

    def __init__(
        self,
        model: nn.Module,
        dataset: Dataset,
        log_path: Path,
        num_epochs: int = 40,
        batch_size: int = 16,
        num_workers: int = 12,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-6,
        seed: int = 1234,
        task: str = "multilabel",
        use_mixup: bool = True,
        debug: bool = False,
        from_audio: bool = False,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.log_path = log_path
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.seed = seed
        self.task = task
        self.use_mixup = use_mixup
        self.debug = debug
        self.from_audio = from_audio
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.loss_fn = nn.MSELoss()

        self.optimizer, self.lr_scheduler = self.configure_optimizers()

        self.ema_model = torch.optim.swa_utils.AveragedModel(
            self.model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999)
        )

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.global_step = 0
        self.epoch = 0
        self.model.to(self.device)
        self.ema_model.to(self.device)
        self.rng = torch.Generator().manual_seed(seed)
        self.grad_scaler = GradScaler()
        self.lowest_val_loss = float("inf")

    def configure_optimizers(self) -> tuple[torch.optim.Optimizer, CosineLRScheduler]:
        """Configure optimizer."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(), self.learning_rate, weight_decay=self.weight_decay
        )
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=self.num_epochs,
            lr_min=1e-5,
            warmup_t=int(0.2 * self.num_epochs),
            warmup_lr_init=3e-4,
        )
        return optimizer, lr_scheduler

    def fit(self) -> None:
        """Fit the model."""
        self.on_fit_begin()
        for _ in range(self.num_epochs):
            self.on_epoch_begin()
            train_ids, val_ids = train_test_split(
                range(len(self.dataset)),
                test_size=0.2,
                random_state=self.seed,
                shuffle=True,
            )
            train_sampler = SubsetRandomSampler(train_ids, generator=self.rng)
            val_sampler = SubsetRandomSampler(val_ids, generator=self.rng)
            train_dataloader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                sampler=train_sampler,
                pin_memory=True,
            )
            val_dataloader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                sampler=val_sampler,
                pin_memory=True,
            )

            if self.epoch == 0:
                self.validate(val_dataloader)
                self.reset_metrics()

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

    def train(self, dataloader: DataLoader) -> None:
        """Train the model."""
        self.model.train()
        with tqdm(dataloader, desc=self.log_path.name) as pbar:
            for batch in dataloader:
                loss = self.train_step(batch)
                pbar.set_postfix({"epoch": self.epoch, "loss": loss.detach().item()})
                pbar.update()
                self.global_step += 1

    def train_step(self, batch: Batch) -> torch.Tensor:
        """Train step."""
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            enabled=torch.cuda.is_available(),
            device_type=self.device.type,
            dtype=torch.bfloat16,
        ):
            logits = self.model(batch.noisy_specs.to(self.device), from_audio=False)
            loss = self.loss_fn(logits, batch.clean_specs.to(self.device))
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        self.ema_model.update_parameters(self.model)
        train_loss = self.train_loss(loss.detach().cpu())
        wandb.log({"train_loss": train_loss}, step=self.global_step)
        return train_loss

    def validate(self, dataloader: DataLoader) -> None:
        """Validate the model."""
        self.model.eval()
        with tqdm(dataloader, desc=self.log_path.name) as pbar:
            for batch in dataloader:
                loss = self.validate_step(batch)
                pbar.set_postfix(
                    {"epoch": self.epoch, "val_loss": loss.detach().item()}
                )
                pbar.update()
        self.on_validate_end()

    def validate_step(self, batch: Batch) -> torch.Tensor:
        """Validate step."""
        with torch.autocast(
            enabled=torch.cuda.is_available(),
            device_type=self.device.type,
            dtype=torch.bfloat16,
        ):
            with torch.no_grad():
                logits = self.ema_model(
                    batch.noisy_specs.to(self.device), from_audio=False
                )
            loss = self.loss_fn(logits, batch.clean_specs.to(self.device))
        val_loss = self.val_loss(loss.cpu())
        return val_loss

    def on_validate_end(self) -> None:
        """On validation end."""
        current_score = self.val_loss.compute()
        if current_score > self.lowest_val_loss:
            self.lowest_val_loss = current_score
            self.save_model()
        wandb.log({"val_loss": current_score}, step=self.global_step)

    def on_epoch_begin(self) -> None:
        """Epoch begin."""
        wandb.log({"epoch": self.epoch}, step=self.global_step)

    def on_epoch_end(self) -> None:
        """Reset epoch."""
        self.lr_scheduler.step(epoch=self.epoch)
        self.reset_metrics()
        self.epoch += 1
        self.garbage_collection()

    def reset_metrics(self) -> None:
        """Reset metrics."""
        self.train_loss.reset()
        self.val_loss.reset()

    @staticmethod
    def garbage_collection() -> None:
        """Garbage collection for cuda."""
        gc.collect()
        torch.cuda.empty_cache()

    def on_fit_end(self) -> None:
        """Fit end."""
        self.save_model()
        wandb.join()

    def save_model(self) -> None:
        """Save model."""
        torch.save(self.ema_model.state_dict(), self.log_path / "ema_model.pt")
