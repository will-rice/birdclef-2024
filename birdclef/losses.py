"""BirdCLEF 2024 loss functions."""

from typing import Optional

import torch
from torch import nn
from torchvision.ops.focal_loss import sigmoid_focal_loss


class FocalLoss(nn.Module):
    """Focal loss."""

    def __init__(
        self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return sigmoid_focal_loss(
            inputs, targets, self.alpha, self.gamma, reduction=self.reduction
        )


class FocalBCELoss(nn.Module):
    """Focal loss with BCE."""

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
        bce_weight: float = 1.0,
        focal_weight: float = 1.0,
        label_smoothing: float = 0.0,
        weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction=reduction, weight=weight)
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if self.label_smoothing:
            targets = (
                targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
            )
        focal_loss = sigmoid_focal_loss(
            inputs=inputs,
            targets=targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )
        bce_loss = self.bce(inputs, targets)
        return self.bce_weight * bce_loss + self.focal_weight * focal_loss


class FocalCELoss(nn.Module):
    """Focal loss with CE."""

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        # use standard CE loss without reduction as basis
        self.cross_entropy = nn.CrossEntropyLoss(
            reduction="none", ignore_index=ignore_index, label_smoothing=label_smoothing
        )
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        minus_logpt = self.cross_entropy(inputs, target)
        pt = torch.exp(-minus_logpt)  # don't forget the minus here
        focal_loss = (1 - pt) ** self.gamma * minus_logpt

        # apply class weights
        if self.alpha is not None:
            focal_loss *= self.alpha.gather(0, target.argmax(1))

        if self.reduction == "mean":
            focal_loss = focal_loss.mean()
        elif self.reduction == "sum":
            focal_loss = focal_loss.sum()
        return focal_loss


class MultiLabelDistillationLoss(nn.Module):
    """Multi-label distillation loss.

    https://arxiv.org/pdf/2308.06453
    """

    def __init__(self, reduction: str = "batchmean", eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.criterion = nn.KLDivLoss(reduction="none", log_target=False)

    def forward(
        self, student_logits: torch.Tensor, teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass."""
        batch_size = student_logits.size(0)
        student_logits = torch.sigmoid(student_logits)
        teacher_logits = torch.sigmoid(teacher_logits)
        student_logits = torch.clamp(student_logits, min=self.eps, max=1 - self.eps)
        teacher_logits = torch.clamp(teacher_logits, min=self.eps, max=1 - self.eps)
        loss = self.criterion(
            torch.log(student_logits), teacher_logits
        ) + self.criterion(torch.log(1 - student_logits), 1 - teacher_logits)
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "batchmean":
            loss = loss.sum() / batch_size
        return loss
