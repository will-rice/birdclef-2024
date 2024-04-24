"""Distillation trainer."""

from pathlib import Path

import torch
import wandb
from torch import nn
from torch.utils.data import Dataset

from birdclef.datasets.birdclef2024 import Batch
from birdclef.trainers import Trainer


class DistillationTrainer(Trainer):
    """Distillation trainer."""

    def __init__(
        self,
        teacher: nn.Module,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Dataset,
        log_path: Path,
        num_epochs: int = 30,
        batch_size: int = 16,
        num_workers: int = 12,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-6,
        seed: int = 1234,
        task: str = "multilabel",
        use_mixup: bool = True,
        debug: bool = False,
        temperature: float = 2.0,
        alpha: float = 0.5,
    ) -> None:
        super().__init__(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            log_path=log_path,
            num_epochs=num_epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            seed=seed,
            task=task,
            use_mixup=use_mixup,
            debug=debug,
        )
        self.teacher = teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.to(self.device)
        self.temperature = temperature
        self.alpha = alpha
        self.distillation_loss_fn = nn.KLDivLoss()

    def train_step(self, batch: Batch) -> torch.Tensor:
        """Train step."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        specs, labels = batch.specs, batch.labels
        if self.use_mixup:
            specs, labels = self.mixup(batch.specs, batch.labels)

        with torch.autocast(
            enabled=torch.cuda.is_available(),
            device_type=self.device.type,
            dtype=torch.bfloat16,
        ):
            with torch.no_grad():
                teacher_logits = self.teacher(specs.to(self.device), from_audio=False)
            student_logits = self.model(specs.to(self.device), from_audio=False)
            student_loss = self.loss_fn(student_logits, labels.to(self.device))

        distillation_loss = (
            self.distillation_loss_fn(
                nn.functional.log_softmax(student_logits / self.temperature, -1),
                nn.functional.softmax(teacher_logits / self.temperature, -1),
            )
            * self.temperature**2
        )
        loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        self.ema_model.update_parameters(self.model)
        train_loss = self.train_loss(loss.cpu())
        metrics = self.train_metrics(student_logits.cpu(), batch.labels)
        wandb.log(
            {
                "train_loss": train_loss,
                "distillation_loss": distillation_loss,
                **metrics,
            },
            step=self.global_step,
        )
        return train_loss
