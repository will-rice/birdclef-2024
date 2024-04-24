"""BirdCLEF 2024 metrics."""

from typing import Any, Optional

import torch
from torchmetrics import (
    AUROC,
    Accuracy,
    AveragePrecision,
    ExactMatch,
    F1Score,
    Metric,
    MetricCollection,
    Precision,
    Recall,
)
from torchmetrics.classification import (
    MultilabelAveragePrecision,
    MultilabelRankingAveragePrecision,
    MultilabelRankingLoss,
)


class Metrics:
    """Classification metrics for BirdCLEF 2024."""

    def __init__(
        self,
        num_classes: int,
        phase: str = "train",
        task: str = "multiclass",
    ) -> None:
        self.num_classes = num_classes
        self.task = task

        if task == "multiclass":
            self.metrics = MetricCollection(
                {
                    f"{phase}_accuracy": Accuracy(
                        task="multiclass",
                        num_classes=num_classes,
                        average="macro",
                    ),
                    f"{phase}_precision": Precision(
                        task="multiclass",
                        num_classes=num_classes,
                        average="macro",
                    ),
                    f"{phase}_recall": Recall(
                        task="multiclass",
                        num_classes=num_classes,
                        average="macro",
                    ),
                    f"{phase}_auroc": AUROC(
                        task="multiclass",
                        num_classes=num_classes,
                        average="macro",
                    ),
                    f"{phase}_average_precision": AveragePrecision(
                        task="multiclass",
                        num_classes=num_classes,
                        average="macro",
                    ),
                    f"{phase}_f1score": F1Score(
                        task="multiclass",
                        num_classes=num_classes,
                        average="macro",
                    ),
                    f"{phase}_exact_match": ExactMatch(
                        task="multiclass",
                        num_classes=num_classes,
                    ),
                }
            )
        elif task == "multilabel":
            self.metrics = MetricCollection(
                {
                    f"{phase}_accuracy": Accuracy(
                        task="multilabel",
                        num_labels=num_classes,
                        average="macro",
                    ),
                    f"{phase}_precision": Precision(
                        task="multilabel",
                        num_labels=num_classes,
                        average="macro",
                    ),
                    f"{phase}_recall": Recall(
                        task="multilabel",
                        num_labels=num_classes,
                        average="macro",
                    ),
                    f"{phase}_auroc": AUROC(
                        task="multilabel",
                        num_labels=num_classes,
                        average="macro",
                    ),
                    f"{phase}_average_precision": AveragePrecision(
                        task="multilabel",
                        num_labels=num_classes,
                        average="macro",
                    ),
                    f"{phase}_f1score": F1Score(
                        task="multilabel",
                        num_labels=num_classes,
                        average="macro",
                    ),
                    f"{phase}_exact_match": ExactMatch(
                        task="multilabel",
                        num_labels=num_classes,
                    ),
                    f"{phase}_ranking_loss": MultilabelRankingLoss(
                        num_labels=num_classes,
                    ),
                    f"{phase}_ranking_avg_precision": MultilabelRankingAveragePrecision(
                        num_labels=num_classes
                    ),
                    f"{phase}_cmap5": CMAP5(
                        num_labels=num_classes,
                    ),
                }
            )

        else:
            raise ValueError(f"{task} task is not supported.")

    def __call__(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute metrics."""
        if self.task == "multiclass":
            inputs = inputs.softmax(dim=1)
            targets = targets.argmax(dim=1)
        elif self.task == "multilabel":
            inputs = inputs.sigmoid()
            targets = targets.int()

        return self.metrics(inputs, targets)

    def reset(self) -> None:
        """Reset metrics."""
        self.metrics.reset()

    def update(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        """Update metrics."""
        if self.task == "multiclass":
            inputs = inputs.softmax(dim=1)
            targets = targets.argmax(dim=1)
        elif self.task == "multilabel":
            inputs = inputs.sigmoid()
            targets = targets.int()

        self.metrics.update(inputs, targets)

    def compute(self) -> dict[str, float]:
        """Compute metrics."""
        return self.metrics.compute()


class CMAP5(Metric):
    """CMAP5 metric for BirdCLEF 2024."""

    def __init__(
        self,
        num_labels: int,
        sample_threshold: int = 5,
        thresholds: Optional[torch.Tensor] = None,
        dist_sync_on_step: bool = False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.num_labels = num_labels
        self.sample_threshold = sample_threshold
        self.thresholds = thresholds

        self.multilabel_ap = MultilabelAveragePrecision(
            average="macro", num_labels=self.num_labels, thresholds=self.thresholds
        )

        # State variable to accumulate predictions and labels across batches
        self.add_state("accumulated_predictions", default=[], dist_reduce_fx="cat")
        self.add_state("accumulated_labels", default=[], dist_reduce_fx="cat")

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """Update metric."""
        self.accumulated_predictions.append(logits)
        self.accumulated_labels.append(labels)

    def compute(self) -> torch.Tensor:
        """Compute metric."""
        # Ensure that accumulated variables are lists
        if not isinstance(self.accumulated_predictions, list):
            self.accumulated_predictions: Any = [self.accumulated_predictions]
        if not isinstance(self.accumulated_labels, list):
            self.accumulated_labels: Any = [self.accumulated_labels]

        # Concatenate accumulated predictions and labels along the batch dimension
        all_predictions = torch.cat(self.accumulated_predictions, dim=0)
        all_labels = torch.cat(self.accumulated_labels, dim=0)

        # Calculate class-wise AP
        class_aps = self.multilabel_ap(all_predictions, all_labels)

        if self.sample_threshold > 1:
            mask = all_labels.sum(dim=0) >= self.sample_threshold
            class_aps = torch.where(mask, class_aps, torch.nan)

        # Compute macro AP by taking the mean of class-wise APs, ignoring NaNs
        macro_cmap = torch.nanmean(class_aps)
        return macro_cmap

    def reset(self) -> None:
        """Reset metric."""
        self.accumulated_predictions = []
        self.accumulated_labels = []
        self.multilabel_ap.reset()
