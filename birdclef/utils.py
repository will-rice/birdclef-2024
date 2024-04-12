"""Utility functions for the BirdCLEF 2024 competition."""

from typing import Any

import torch


def sequence_mask(length: torch.Tensor, max_length: Any = None) -> torch.Tensor:
    """Create a boolean mask from sequence lengths."""
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)
