"""Config files."""

from pathlib import Path
from typing import NamedTuple, Optional


class TransformConfig(NamedTuple):
    """Transform configuration."""

    # PitchShift
    add_pitch_shift: bool = False
    pitch_shift_p: float = 0.5
    # TimeStretch
    add_time_stretch: bool = False
    time_stretch_p: float = 0.5
    # AddBackgroundNoise
    add_background_noise: bool = False
    background_noise_p: float = 0.5
    background_noise_path: Optional[Path] = None
    # AddShortNoises
    add_short_noises: bool = False
    short_noises_p: float = 0.5
    short_noises_path: Optional[Path] = None
    # TimeMask
    add_time_mask: bool = False
    time_mask_p: float = 0.5
    # Gain
    add_gain: bool = False
    gain_p: float = 0.5
    # AddGaussianNoise
    add_gaussian_noise: bool = False
    gaussian_noise_p: float = 0.5
    # AddColorNoise
    add_color_noise: bool = False
    color_noise_p: float = 0.5
