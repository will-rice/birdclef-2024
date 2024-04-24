"""Trainer module for training the model."""

import gc
from pathlib import Path

import kagglehub
import matplotlib.pyplot as plt
import torch
import wandb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from timm.scheduler import CosineLRScheduler
from torch import GradScaler, nn
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torchmetrics import MeanMetric, StatScores
from torchmetrics.classification import MultilabelConfusionMatrix
from tqdm import tqdm

from birdclef.datasets.birdclef2024 import Batch
from birdclef.losses import FocalBCELoss
from birdclef.metrics import Metrics
from birdclef.transforms import MultilabelMixUp


class Trainer:
    """Trainer for BirdCLEF2024 dataset."""

    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Dataset,
        log_path: Path,
        num_epochs: int = 35,
        batch_size: int = 16,
        num_workers: int = 12,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-6,
        seed: int = 1234,
        task: str = "multilabel",
        use_mixup: bool = True,
        debug: bool = False,
        from_audio: bool = False,
        use_lr_scheduler: bool = True,
    ) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
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
        self.use_lr_scheduler = use_lr_scheduler
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        if task == "multiclass":
            self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.02)
        elif task == "multilabel":
            self.loss_fn = FocalBCELoss(alpha=1.0, gamma=5.0)
        else:
            raise ValueError(f"Invalid task: {task}")

        self.optimizer, self.lr_scheduler = self.configure_optimizers()

        self.ema_model = torch.optim.swa_utils.AveragedModel(
            self.model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999)
        )

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.train_metrics = Metrics(
            num_classes=self.train_dataset.num_classes, task=task, phase="train"
        )
        self.val_metrics = Metrics(
            num_classes=self.train_dataset.num_classes, task=task, phase="val"
        )
        self.confusion_matrix = MultilabelConfusionMatrix(
            self.train_dataset.num_classes
        )
        self.sklearn_roc_auc = MeanMetric()
        if task == "multiclass":
            self.stat_scores = StatScores(
                task="multiclass", num_classes=self.train_dataset.num_classes
            )
        else:
            self.stat_scores = StatScores(
                task="multilabel", num_labels=self.train_dataset.num_classes
            )

        self.global_step = 0
        self.epoch = 0
        self.model.to(self.device)
        self.ema_model.to(self.device)
        self.rng = torch.Generator().manual_seed(seed)
        self.mixup = MultilabelMixUp(p=0.5, alpha=1.0, beta=1.0)
        self.grad_scaler = GradScaler()
        self.cv_score = 0.0

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
                pin_memory=True,
            )
            val_dataloader = DataLoader(
                self.val_dataset,
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
        wandb.log({"seed": self.seed}, step=self.global_step)

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

        if self.from_audio:
            inputs, labels = batch.audio, batch.labels
        else:
            inputs, labels = batch.specs, batch.labels

            if self.use_mixup:
                inputs, labels = self.mixup(inputs, labels)

        with torch.autocast(
            enabled=torch.cuda.is_available(),
            device_type=self.device.type,
            dtype=torch.bfloat16,
        ):
            logits = self.model(inputs.to(self.device), from_audio=False)
            loss = self.loss_fn(logits, labels.to(self.device))
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        self.ema_model.update_parameters(self.model)
        train_loss = self.train_loss(loss.cpu())
        metrics = self.train_metrics(logits.cpu(), batch.labels)
        wandb.log({"train_loss": train_loss, **metrics}, step=self.global_step)
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
        if self.from_audio:
            inputs, labels = batch.audio, batch.labels
        else:
            inputs, labels = batch.specs, batch.labels

        with torch.autocast(
            enabled=torch.cuda.is_available(),
            device_type=self.device.type,
            dtype=torch.bfloat16,
        ):
            with torch.no_grad():
                logits = self.ema_model(inputs.to(self.device), from_audio=False)
                loss = self.loss_fn(logits, labels.to(self.device))
        val_loss = self.val_loss(loss.detach().cpu())
        self.val_metrics.update(logits.detach().cpu(), labels)

        if self.task == "multiclass":
            probs = logits.detach().cpu().softmax(dim=1).float()
        else:
            probs = logits.detach().cpu().sigmoid().float()

        sklearn_roc_auc = 0.0
        count = 0
        for label, prob in zip(labels, probs):
            if label.sum() == 0.0:
                continue
            sklearn_roc_auc += roc_auc_score(label, prob, average="macro")
            count += 1

        if count > 0:
            sklearn_roc_auc /= count

        self.sklearn_roc_auc.update(sklearn_roc_auc)
        self.confusion_matrix.update(probs, labels.int())
        self.stat_scores.update(probs, labels.int())

        return val_loss

    def on_validate_end(self) -> None:
        """On validation end."""
        metrics = self.val_metrics.compute()
        current_score = metrics["val_auroc"]
        if current_score > self.cv_score:
            self.cv_score = current_score
            self.save_model()
        wandb.log(
            {
                "val_loss": self.val_loss.compute(),
                "sklearn_roc_auc": self.sklearn_roc_auc.compute(),
                **metrics,
            },
            step=self.global_step,
        )
        save_path = self.log_path / "weights.pt"
        torch.save(self.model.state_dict(), save_path)

        fig, ax = self.confusion_matrix.plot(labels=self.train_dataset.labels)
        fig.set_figwidth(30)
        fig.set_figheight(30)
        wandb.log({"confusion_matrix": fig}, step=self.global_step)
        plt.close(fig)

        true_positives, false_positives, true_negatives, false_negatives, support = (
            self.stat_scores.compute()
        )
        wandb.log(
            {
                "true_positives": true_positives,
                "false_positives": false_positives,
                "true_negatives": true_negatives,
                "false_negatives": false_negatives,
                "sensitivity": true_positives / (true_positives + false_negatives),
                "specificity": true_negatives / (true_negatives + false_positives),
                "support": support,
            },
            step=self.global_step,
        )

    def on_epoch_begin(self) -> None:
        """Epoch begin."""
        wandb.log({"epoch": self.epoch}, step=self.global_step)

    def on_epoch_end(self) -> None:
        """Reset epoch."""
        if self.use_lr_scheduler:
            self.lr_scheduler.step(epoch=self.epoch)
        self.reset_metrics()
        self.epoch += 1
        self.seed += 1
        self.garbage_collection()

    def reset_metrics(self) -> None:
        """Reset metrics."""
        self.train_loss.reset()
        self.val_loss.reset()
        self.train_metrics.reset()
        self.val_metrics.reset()
        self.sklearn_roc_auc.reset()
        self.confusion_matrix.reset()
        self.stat_scores.reset()

    @staticmethod
    def garbage_collection() -> None:
        """Garbage collection for cuda."""
        gc.collect()
        torch.cuda.empty_cache()

    def on_fit_end(self) -> None:
        """Fit end."""
        self.upload_model()
        wandb.log({"cv_score": self.cv_score}, step=self.global_step)
        wandb.join()

    def upload_model(self) -> None:
        """Upload model to kagglehub."""
        save_path = self.log_path / f"{self.log_path.name}.pt"
        kagglehub.model_upload(
            f"willrice/{self.log_path.name}/pyTorch/1",
            str(save_path),
            "Apache 2.0",
        )

    def save_model(self) -> None:
        """Save model."""
        self.ema_model.cpu()
        traced_model = torch.jit.trace(
            self.ema_model.eval(),
            torch.randn(1, 1, 160000, device="cpu"),
        )
        save_path = self.log_path / f"{self.log_path.name}.pt"
        torch.jit.save(traced_model, save_path)

        self.ema_model.cuda()
