"""GeM model definition."""

import numpy as np
import timm
import torch
from torch import nn
from torchvision.transforms import v2

from birdclef.signal import MelSpectrogram


class GeMClassifier(torch.nn.Module):
    """GeM model."""

    def __init__(
        self,
        backbone: str = "timm/eca_nfnet_l0",
        pretrained: bool = True,
        num_classes: int = 183,
    ):
        super().__init__()
        self.spec_fn = MelSpectrogram()
        out_indices = (3, 4)
        self.backbone = timm.create_model(
            backbone,
            features_only=True,
            pretrained=pretrained,
            in_chans=3,
            num_classes=num_classes,
            out_indices=out_indices,
        )
        feature_dims = self.backbone.feature_info.channels()
        print(f"feature dims: {feature_dims}")

        self.global_pools = torch.nn.ModuleList([GeM() for _ in out_indices])
        self.mid_features = np.sum(feature_dims)
        self.neck = torch.nn.BatchNorm1d(self.mid_features)
        self.head = torch.nn.Linear(self.mid_features, num_classes)
        self.image_size = self.backbone.default_cfg["input_size"][1:]
        self.normalize = v2.Normalize(
            self.backbone.default_cfg["mean"], self.backbone.default_cfg["std"]
        )

    def forward(self, x: torch.Tensor, from_audio: bool = True) -> torch.Tensor:
        """Forward pass."""
        if from_audio:
            x = self.spec_fn(x)
            x = self.normalize(x)
        x = nn.functional.interpolate(x, size=self.image_size, mode="bilinear")
        ms = self.backbone(x)
        h = torch.cat(
            [global_pool(m) for m, global_pool in zip(ms, self.global_pools)], dim=1
        )
        x = self.neck(h)
        x = self.head(x)
        return x


class GeM(torch.nn.Module):
    """GeM pooling."""

    def __init__(self, p: int = 3, eps: float = 1e-6):
        super(GeM, self).__init__()
        self.p = torch.nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        bs, ch, h, w = x.shape
        x = torch.nn.functional.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p)
        x = x.view(bs, ch)
        return x
