"""StratifiedKFold trainer for birdCLEF dataset."""

import gc
from copy import deepcopy
from pathlib import Path
from typing import Tuple

import kagglehub
import matplotlib.pyplot as plt
import torch
import wandb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch import GradScaler, nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchmetrics.aggregation import MeanMetric, MinMetric
from torchmetrics.classification import MultilabelConfusionMatrix, StatScores
from tqdm import tqdm

from birdclef.datasets.birdclef2024 import Batch, BirdCLEF2024Dataset
from birdclef.losses import FocalBCELoss
from birdclef.metrics import Metrics
from birdclef.transforms import MultilabelMixUp


class StratifiedKFoldTrainer:
    """StratifiedKFold trainer for birdCLEF dataset."""

    def __init__(
        self,
        model: nn.Module,
        train_dataset: BirdCLEF2024Dataset,
        val_dataset: BirdCLEF2024Dataset,
        log_path: Path,
        num_folds: int = 5,
        num_epochs: int = 35,
        batch_size: int = 16,
        num_workers: int = 12,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-6,
        seed: int = 1234,
        task: str = "multilabel",
        use_lr_scheduler: bool = True,
        use_mixup: bool = True,
        debug: bool = False,
        from_audio: bool = False,
        use_class_weights: bool = False,
    ) -> None:
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.models = [deepcopy(model) for _ in range(num_folds)]
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.log_path = log_path
        self.num_folds = num_folds
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_lr_scheduler = use_lr_scheduler
        self.use_mixup = use_mixup
        self.task = task
        self.debug = debug
        self.from_audio = from_audio
        self.splitter = StratifiedKFold(
            n_splits=num_folds, shuffle=True, random_state=seed
        )
        if task == "multiclass":
            self.loss_fn = nn.CrossEntropyLoss(
                label_smoothing=0.1,
            )
        elif task == "multilabel":
            self.loss_fn = FocalBCELoss(
                weight=(
                    torch.tensor(self.train_dataset.sample_weights, device=self.device)
                    if use_class_weights
                    else None
                ),
                label_smoothing=0.0,
            )
        else:
            raise ValueError(f"Invalid task: {task}")

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.min_max_prob = MinMetric()
        self.sklearn_roc_auc = MeanMetric()
        self.cv_score = 0.0
        self.train_metrics = Metrics(
            num_classes=self.train_dataset.num_classes, task=task, phase="train"
        )
        self.val_metrics = Metrics(
            num_classes=self.train_dataset.num_classes, task=task, phase="val"
        )
        self.confusion_matrix = MultilabelConfusionMatrix(
            num_labels=self.train_dataset.num_classes
        )
        if task == "multiclass":
            self.stat_scores = StatScores(
                task="multiclass", num_classes=self.train_dataset.num_classes
            )
        else:
            self.stat_scores = StatScores(
                task="multilabel", num_labels=self.train_dataset.num_classes
            )
        self.global_step = 1
        self.fold = 0
        self.epoch = 0
        self.model = self.models[self.fold]
        self.ema_model = torch.optim.swa_utils.AveragedModel(
            self.model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999)
        )
        self.rng = torch.Generator().manual_seed(seed)
        self.mixup = MultilabelMixUp()
        self.grad_scaler = GradScaler()
        self.optimizer, self.lr_scheduler = self.configure_optimizers()

    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, CosineLRScheduler]:
        """Configure optimizers."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
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
        """Train the model."""
        for train_ids, val_ids in self.splitter.split(
            self.train_dataset, self.train_dataset.all_labels
        ):
            self.on_fold_begin()

            train_sampler = SubsetRandomSampler(train_ids, generator=self.rng)
            val_sampler = SubsetRandomSampler(val_ids, generator=self.rng)
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                sampler=train_sampler,
            )
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                sampler=val_sampler,
            )
            for _ in range(self.num_epochs):
                self.on_epoch_begin()

                if self.epoch == 0:
                    self.validate(val_loader)
                    self.reset_metrics()

                self.train(train_loader)
                self.validate(val_loader)

                self.on_epoch_end()

            self.on_fold_end()

        wandb.finish()

    def on_fold_begin(self) -> None:
        """Start fold."""
        self.model = self.models[self.fold].to(self.device)

        self.optimizer, self.lr_scheduler = self.configure_optimizers()

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
        self.model.train()
        with tqdm(dataloader, desc=self.log_path.name) as pbar:
            for batch in dataloader:
                loss = self.train_step(batch)
                pbar.set_postfix(
                    {"fold": self.fold, "epoch": self.epoch, "loss": loss.item()}
                )
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
        metrics = self.train_metrics(logits.cpu().float(), batch.labels)
        wandb.log(
            {"train_loss": train_loss, **metrics},
            step=self.global_step,
        )
        return train_loss

    def validate(self, dataloader: DataLoader) -> None:
        """Validate the model."""
        self.model.eval()
        with tqdm(dataloader, desc=self.log_path.name) as pbar:
            for batch in dataloader:
                loss = self.validate_step(batch)
                pbar.set_postfix(
                    {"fold": self.fold, "epoch": self.epoch, "val_loss": loss.item()}
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
        val_loss = self.val_loss(loss.cpu())
        self.val_metrics.update(logits.cpu().float(), labels)

        if self.task == "multiclass":
            probs = logits.cpu().softmax(dim=1).float()
        else:
            probs = logits.cpu().sigmoid().float()

        self.min_max_prob.update(probs.max())

        sklearn_roc_auc = 0.0
        for label, prob in zip(labels, probs):
            sklearn_roc_auc += roc_auc_score(label, prob, average="macro")

        sklearn_roc_auc /= len(labels)

        self.sklearn_roc_auc.update(sklearn_roc_auc)
        self.confusion_matrix.update(probs, labels.int())
        self.stat_scores.update(probs, labels.int())
        return val_loss

    def on_validate_end(self) -> None:
        """On validate end."""
        metrics = self.val_metrics.compute()
        current_score = metrics["val_auroc"]
        if current_score >= self.cv_score:
            self.cv_score = current_score
            self.save_model()

        wandb.log(
            {
                "val_loss": self.val_loss.compute(),
                "min_max_prob": self.min_max_prob.compute(),
                "sklearn_roc_auc": self.sklearn_roc_auc.compute(),
                **metrics,
            },
            step=self.global_step,
        )
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
        self.garbage_collection()
        self.epoch += 1

    def reset_metrics(self) -> None:
        """Reset metrics."""
        self.train_loss.reset()
        self.val_loss.reset()
        self.train_metrics.reset()
        self.val_metrics.reset()
        self.min_max_prob.reset()
        self.sklearn_roc_auc.reset()
        self.confusion_matrix.reset()
        self.stat_scores.reset()

    @staticmethod
    def garbage_collection() -> None:
        """Garbage collection for cuda."""
        gc.collect()
        torch.cuda.empty_cache()

    def on_fold_end(self) -> None:
        """Reset fold."""
        self.upload_model()
        wandb.log({"cv_score": self.cv_score}, step=self.global_step)
        self.cv_score = 0.0
        self.global_step = 1
        self.epoch = 0
        self.fold += 1
        wandb.join()

    def upload_model(self) -> None:
        """Upload model to kagglehub."""
        model_name = f"{self.log_path.name}_{self.fold}"
        save_path = self.log_path / f"{model_name}.pt"
        kagglehub.model_upload(
            f"willrice/{self.log_path.name}/pyTorch/fold-{self.fold}",
            str(save_path),
            "Apache 2.0",
        )

    def save_model(self) -> None:
        """Save model."""
        traced_model = torch.jit.trace(
            self.ema_model.to("cpu").eval(),
            torch.randn(1, 1, 32000 * 5, device="cpu"),
        )
        model_name = f"{self.log_path.name}_{self.fold}"
        save_path = self.log_path / f"{model_name}.pt"
        torch.jit.save(traced_model, save_path)

        self.ema_model.cuda()
