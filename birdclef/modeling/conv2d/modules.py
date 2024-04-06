"""Conv2d modules."""

from typing import Optional

import torch
from torch import nn


class DownsampleBlock2D(nn.Module):
    """2d downsample block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        num_groups: int = 32,
        dropout: float = 0.0,
        activation: str = "silu",
        n_res_blocks: int = 1,
    ) -> None:
        super().__init__()
        self.res_blocks = nn.Sequential(
            *[
                ResidualBlock2D(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    num_groups=num_groups,
                    activation=activation,
                    dropout=dropout,
                )
                for i in range(n_res_blocks)
            ]
        )
        self.downsample = Downsample2D(
            out_channels, out_channels=out_channels, use_conv=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.res_blocks(x)
        x = self.downsample(x)
        return x


class ResidualBlock2D(nn.Module):
    """2d residual block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        num_groups: int = 32,
        activation: str = "silu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm_1 = nn.GroupNorm(num_groups, in_channels)
        self.activation_1 = Activation(activation)
        self.conv_1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )

        self.norm_2 = nn.GroupNorm(num_groups, out_channels)
        self.activation_2 = Activation(activation)
        self.conv_2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.residual_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=True
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        residual = self.residual_conv(x)
        x = self.norm_1(x)
        x = self.activation_1(x)
        x = self.conv_1(x)
        x = self.norm_2(x)
        x = self.activation_2(x)
        x = self.dropout(x)
        x = self.conv_2(x)
        return x + residual


class Downsample2D(nn.Module):
    """A 2D downsampling layer with an optional convolution.

    Parameters
    ----------
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        padding (`int`, default `1`):
            padding for the convolution.
        name (`str`, default `conv`):
            name of the downsampling 2D layer.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: Optional[int] = None,
        padding: int = 1,
        name: str = "conv",
        kernel_size: int = 3,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name

        if use_conv:
            conv = nn.Conv2d(
                self.channels,
                self.out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )
        else:
            assert self.channels == self.out_channels
            conv = nn.AvgPool2d(kernel_size=stride, stride=stride)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if name == "conv":
            self.Conv2d_0 = conv
            self.conv = conv
        elif name == "Conv2d_0":
            self.conv = conv
        else:
            self.conv = conv

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        """Forward pass."""
        hidden_states = self.conv(hidden_states)
        return hidden_states


class Activation(nn.Module):
    """Activation function."""

    def __init__(self, name: str):
        super().__init__()
        if name == "silu":
            self.activation: nn.Module = nn.SiLU(inplace=True)
        elif name == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif name == "leaky_relu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise ValueError(f"{name} activation is not supported.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward Pass."""
        x = self.activation(x)
        return x
