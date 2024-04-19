"""StratifiedKFold trainer for birdCLEF dataset."""

import gc
from copy import deepcopy
from pathlib import Path

import kagglehub
import torch
import wandb
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchmetrics import (
    AUROC,
    Accuracy,
    AveragePrecision,
    ExactMatch,
    F1Score,
    MetricCollection,
    Precision,
    Recall,
)
from torchmetrics.aggregation import MeanMetric
from tqdm import tqdm

from birdclef.datasets.birdclef2024 import Batch, BirdCLEF2024Dataset


class StratifiedKFoldTrainer:
    """StratifiedKFold trainer for birdCLEF dataset."""

    def __init__(
        self,
        model: nn.Module,
        dataset: BirdCLEF2024Dataset,
        log_path: Path,
        num_folds: int = 5,
        num_epochs: int = 1,
        batch_size: int = 16,
        num_workers: int = 12,
        debug: bool = False,
    ) -> None:
        self.models = [deepcopy(model) for _ in range(num_folds)]
        if torch.cuda.is_available():
            for m in self.models:
                m.cuda()
        self.dataset = dataset
        self.log_path = log_path
        self.num_folds = num_folds
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.debug = debug
        self.splitter = StratifiedKFold(n_splits=num_folds, shuffle=True)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.2)
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.cv_score = MeanMetric()
        metrics = MetricCollection(
            [
                Accuracy(task="multiclass", num_classes=182, average="macro"),
                Precision(task="multiclass", num_classes=182, average="macro"),
                Recall(task="multiclass", num_classes=182, average="macro"),
                AUROC(task="multiclass", num_classes=182, average="macro"),
                AveragePrecision(task="multiclass", num_classes=182, average="macro"),
                F1Score(task="multiclass", num_classes=182, average="macro"),
                ExactMatch(task="multiclass", num_classes=182),
            ]
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.global_step = 0
        self.fold = 0
        self.epoch = 0
        self.model = self.models[self.fold]
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.ema_model = torch.optim.swa_utils.AveragedModel(
            self.model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999)
        )

    def fit(self) -> None:
        """Train the model."""
        for train_ids, val_ids in self.splitter.split(
            self.dataset, self.dataset.metadata.primary_label
        ):
            self.on_fold_begin()

            train_sampler = SubsetRandomSampler(train_ids)
            val_sampler = SubsetRandomSampler(val_ids)
            self.dataset.transform = True
            self.dataset.max_length = 32000 * 5
            train_loader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=False,
                sampler=train_sampler,
            )
            self.dataset.transform = False
            self.dataset.max_length = 32000 * 5
            val_loader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=False,
                sampler=val_sampler,
            )
            for _ in range(self.num_epochs):
                self.train(train_loader)
                self.validate(val_loader)
                self.on_epoch_end()
            self.on_fold_end()

        wandb.finish()

    def on_fold_begin(self) -> None:
        """Start fold."""
        self.model = self.models[self.fold]
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.ema_model = torch.optim.swa_utils.AveragedModel(
            self.model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999)
        )

        wandb.init(
            project="birdclef-2024",
            group=self.log_path.name,
            name=f"{self.log_path.name}-{self.fold}",
            mode="offline" if self.debug else "online",
        )

    def train(self, dataloader: DataLoader) -> None:
        """Train the model."""
        with tqdm(dataloader, desc=self.log_path.name) as pbar:
            for batch in dataloader:
                with torch.autocast(
                    enabled=torch.cuda.is_available(),
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):
                    loss = self.train_step(batch)
                    pbar.set_postfix(
                        {"fold": self.fold, "epoch": self.epoch, "loss": loss.item()}
                    )
                    pbar.update()
                    self.global_step += 1

    def train_step(self, batch: Batch) -> torch.Tensor:
        """Train step."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(batch.specs.cuda(), batch.lengths.cuda(), from_audio=False)
        loss = self.loss_fn(logits, batch.label_id.cuda())
        loss.backward()
        self.optimizer.step()
        self.ema_model.update_parameters(self.model)
        train_loss = self.train_loss(loss.cpu())
        metrics = self.train_metrics(logits.softmax(dim=1).cpu(), batch.label_id)
        wandb.log({"train_loss": train_loss, **metrics}, step=self.global_step)
        return train_loss

    def validate(self, dataloader: DataLoader) -> None:
        """Validate the model."""
        with tqdm(dataloader, desc=self.log_path.name) as pbar:
            for batch in dataloader:
                with torch.autocast(
                    enabled=torch.cuda.is_available(),
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):
                    self.validate_step(batch)
                    pbar.set_postfix(
                        {
                            "fold": self.fold,
                            "epoch": self.epoch,
                            "val_loss": self.val_loss.compute().item(),
                        }
                    )
                    pbar.update()
        self.on_validate_end()

    def validate_step(self, batch: Batch) -> torch.Tensor:
        """Validate step."""
        self.ema_model.eval()
        with torch.no_grad():
            logits = self.ema_model(
                batch.specs.cuda(), batch.lengths.cuda(), from_audio=False
            )
        loss = self.loss_fn(logits, batch.label_id.cuda())
        val_loss = self.val_loss(loss.cpu())
        self.val_metrics.update(logits.softmax(dim=1).cpu(), batch.label_id)
        return val_loss

    def on_validate_end(self) -> None:
        """On validate end."""
        val_metrics = self.val_metrics.compute()
        self.cv_score.update(val_metrics["val_MulticlassAUROC"])
        wandb.log(
            {
                "val_loss": self.val_loss.compute(),
                **self.val_metrics.compute(),
            },
            step=self.global_step,
        )

    def on_fold_end(self) -> None:
        """Reset fold."""
        wandb.log({"cv_score": self.cv_score.compute()})
        self.save_model()
        self.cv_score.reset()
        self.global_step = 0
        self.epoch = 0
        self.fold += 1
        wandb.join()

    def save_model(self) -> None:
        """Save model."""
        traced_model = torch.jit.trace(
            self.ema_model.to("cpu").eval(),
            torch.randn(1, 1, 32000 * 5, device="cpu"),
        )
        model_name = f"{self.log_path.name}_{self.fold}_{self.cv_score.compute():.4f}"
        save_path = self.log_path / f"{model_name}.pt"
        torch.jit.save(traced_model, save_path)
        kagglehub.model_upload(
            f"willrice/{self.log_path.name}/pyTorch/fold-{self.fold}",
            str(save_path),
            "Apache 2.0",
        )
        self.ema_model.cuda()

    def on_epoch_end(self) -> None:
        """Reset epoch."""
        self.train_loss.reset()
        self.val_loss.reset()
        self.train_metrics.reset()
        self.val_metrics.reset()
        self.garbage_collection()
        self.epoch += 1

    @staticmethod
    def garbage_collection() -> None:
        """Garbage collection for cuda."""
        gc.collect()
        torch.cuda.empty_cache()
