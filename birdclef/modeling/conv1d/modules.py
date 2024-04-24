"""Conv2d modules."""

import math

import torch
from torch import nn

KERNELS = {
    "linear": [1 / 8, 3 / 8, 3 / 8, 1 / 8],
    "cubic": [
        -0.01171875,
        -0.03515625,
        0.11328125,
        0.43359375,
        0.43359375,
        0.11328125,
        -0.03515625,
        -0.01171875,
    ],
    "lanczos3": [
        0.003689131001010537,
        0.015056144446134567,
        -0.03399861603975296,
        -0.066637322306633,
        0.13550527393817902,
        0.44638532400131226,
        0.44638532400131226,
        0.13550527393817902,
        -0.066637322306633,
        -0.03399861603975296,
        0.015056144446134567,
        0.003689131001010537,
    ],
}


class DownsampleBlock1D(nn.Module):
    """2d downsample block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
        activation: str = "silu",
        n_res_blocks: int = 1,
    ) -> None:
        super().__init__()
        self.res_blocks = nn.Sequential(
            *[
                ResidualBlock1D(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    activation=activation,
                    dropout=dropout,
                )
                for i in range(n_res_blocks)
            ]
        )
        self.downsample = Downsample1d(kernel="linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.res_blocks(x)
        x = self.downsample(x)
        return x


class ResidualBlock1D(nn.Module):
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
        self.norm_1 = nn.GroupNorm(num_channels=in_channels, num_groups=num_groups)
        self.activation_1 = Activation(activation)
        self.conv_1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )

        self.norm_2 = nn.GroupNorm(num_channels=out_channels, num_groups=num_groups)
        self.activation_2 = Activation(activation)
        self.conv_2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.residual_conv = nn.Conv1d(
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


class LayerNorm(nn.LayerNorm):
    """Channels first LayerNorm."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = x.transpose(1, 2)
        x = super().forward(x)
        x = x.transpose(1, 2)
        return x


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


class Downsample1d(nn.Module):
    """1D downsampling layer."""

    def __init__(self, kernel: str = "linear", pad_mode: str = "reflect"):
        super().__init__()
        self.pad_mode = pad_mode
        kernel_1d = torch.tensor(KERNELS[kernel])
        self.pad = kernel_1d.shape[0] // 2 - 1
        self.register_buffer("kernel", kernel_1d)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        hidden_states = nn.functional.pad(hidden_states, (self.pad,) * 2, self.pad_mode)
        weight = hidden_states.new_zeros(
            [hidden_states.shape[1], hidden_states.shape[1], self.kernel.shape[0]]
        )
        indices = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        kernel = self.kernel.to(weight)[None, :].expand(hidden_states.shape[1], -1)
        weight[indices, indices] = kernel
        return nn.functional.conv1d(hidden_states, weight, stride=4)


class SelfAttention1d(nn.Module):
    """1D self attention layer."""

    def __init__(self, in_channels: int, n_head: int = 1, dropout_rate: float = 0.0):
        super().__init__()
        self.channels = in_channels
        self.group_norm = nn.GroupNorm(1, num_channels=in_channels)
        self.num_heads = n_head

        self.query = nn.Linear(self.channels, self.channels)
        self.key = nn.Linear(self.channels, self.channels)
        self.value = nn.Linear(self.channels, self.channels)

        self.proj_attn = nn.Linear(self.channels, self.channels, bias=True)

        self.dropout = nn.Dropout(dropout_rate, inplace=True)

    def transpose_for_scores(self, projection: torch.Tensor) -> torch.Tensor:
        """Transpose projection."""
        new_projection_shape = projection.size()[:-1] + (self.num_heads, -1)
        # move heads to 2nd position (B, T, H * D) -> (B, T, H, D) -> (B, H, T, D)
        new_projection = projection.view(new_projection_shape).permute(0, 2, 1, 3)
        return new_projection

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        residual = hidden_states
        batch, channel_dim, seq = hidden_states.shape

        hidden_states = self.group_norm(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)

        query_proj = self.query(hidden_states)
        key_proj = self.key(hidden_states)
        value_proj = self.value(hidden_states)

        query_states = self.transpose_for_scores(query_proj)
        key_states = self.transpose_for_scores(key_proj)
        value_states = self.transpose_for_scores(value_proj)

        scale = 1 / math.sqrt(math.sqrt(key_states.shape[-1]))

        attention_scores = torch.matmul(
            query_states * scale, key_states.transpose(-1, -2) * scale
        )
        attention_probs = torch.softmax(attention_scores, dim=-1)

        # compute attention output
        hidden_states = torch.matmul(attention_probs, value_states)

        hidden_states = hidden_states.permute(0, 2, 1, 3).contiguous()
        new_hidden_states_shape = hidden_states.size()[:-2] + (self.channels,)
        hidden_states = hidden_states.view(new_hidden_states_shape)

        # compute next hidden_states
        hidden_states = self.proj_attn(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.dropout(hidden_states)

        output = hidden_states + residual

        return output


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d: int, p: float = -1.0, eps: float = 1e-8, bias: bool = False):
        """Root Mean Square Layer Normalization."""
        super().__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = x.transpose(2, 1)
        if self.p < 0.0 or self.p > 1.0:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1.0 / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return (self.scale * x_normed + self.offset).transpose(2, 1)
        return (self.scale * x_normed).transpose(2, 1)
