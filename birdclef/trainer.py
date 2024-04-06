"""Trainer module for training the model."""

from typing import Any

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import grad_norm, rank_zero_only
from pytorch_lightning.utilities.memory import garbage_collection_cuda
from torch import nn
from torchmetrics import AUROC, Accuracy, Precision, Recall

from birdclef.datasets.birdclef2024 import Batch


class ClassifierLightningModule(LightningModule):
    """Lightning module for training the model."""

    def __init__(self, model: nn.Module, sync_dist: bool = False) -> None:
        super().__init__()

        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.acc_fn = Accuracy(task="multiclass", num_classes=182, average="weighted")
        self.pre_fn = Precision(task="multiclass", num_classes=182, average="weighted")
        self.rec_fn = Recall(task="multiclass", num_classes=182, average="weighted")
        self.auroc_fn = AUROC(task="multiclass", num_classes=182, average="weighted")
        self.sync_dist = sync_dist

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        """Train step."""
        logits = self.model(batch.audio, batch.lengths)
        probs = logits.softmax(dim=1)

        loss = self.loss_fn(logits, batch.label_id)
        self.log("train_loss", loss, prog_bar=True, sync_dist=self.sync_dist)

        accuracy = self.acc_fn(probs, batch.label_id)
        precision = self.pre_fn(probs, batch.label_id)
        recall = self.rec_fn(probs, batch.label_id)
        auroc = self.auroc_fn(probs, batch.label_id)

        self.log_dict(
            {
                "train_accuracy": accuracy,
                "train_precision": precision,
                "train_recall": recall,
                "train_auroc": auroc,
            },
            sync_dist=self.sync_dist,
        )

        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        """Validate step."""
        with torch.no_grad():
            logits = self.model(batch.audio, batch.lengths)
        probs = logits.softmax(dim=1)
        loss = self.loss_fn(logits, batch.label_id)
        self.log("val_loss", loss, prog_bar=True, sync_dist=self.sync_dist)

        accuracy = self.acc_fn(probs, batch.label_id)
        precision = self.pre_fn(probs, batch.label_id)
        recall = self.rec_fn(probs, batch.label_id)
        auroc = self.auroc_fn(probs, batch.label_id)

        self.log_dict(
            {
                "val_accuracy": accuracy,
                "val_precision": precision,
                "val_recall": recall,
                "val_auroc": auroc,
            },
            sync_dist=self.sync_dist,
        )
        return loss

    @rank_zero_only
    def on_validation_end(self) -> None:
        """On validation end."""
        traced_model = torch.jit.trace(
            self.model.to("cpu"), torch.randn(1, 1, 32000 * 5, device="cpu")
        )
        torch.jit.save(traced_model, self.trainer.default_root_dir + "/model.pt")

        garbage_collection_cuda()
        self.model.cuda()

    def on_before_optimizer_step(self, optimizer: Any) -> None:
        """Before optimizer step."""
        self.log_dict(grad_norm(self, norm_type=1), sync_dist=self.sync_dist)

    def configure_optimizers(self) -> Any:
        """Configure optimizers for the model."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer
