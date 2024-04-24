"""Utility functions for the BirdCLEF 2024 competition."""

import random
from typing import Any

import numpy as np
import torch
from torchaudio.io import StreamReader


def sequence_mask(length: torch.Tensor, max_length: Any = None) -> torch.Tensor:
    """Create a boolean mask from sequence lengths."""
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def seed_everything(seed: int) -> None:
    """Seed everything for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available:
        torch.backends.cudnn.deterministic = True


def load_audio(path: str, start_idx: int = 0, num_frames: int = -1) -> torch.Tensor:
    """Load a video with audio from a file."""
    reader = StreamReader(path)
    reader.add_basic_audio_stream(num_frames)
    metadata = reader.get_src_stream_info(0)
    start_seconds = start_idx / metadata.sample_rate
    reader.seek(start_seconds)
    reader.fill_buffer()
    (audio,) = reader.pop_chunks()
    return audio.transpose(0, 1)
