from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from .scaling import ActivationDropoutAndLinear, Balancer, ChunkCausalDepthwiseConv1d, FloatLike, Identity, ScaledLinear, ScheduledFloat, Whiten
from .schedules import _whitening_schedule


class FeedforwardModule(nn.Module):
    """Feedforward module in Zipformer2 model."""

    def __init__(
        self,
        embed_dim: int,
        feedforward_dim: int,
        dropout: FloatLike,
    ) -> None:
        super(FeedforwardModule, self).__init__()
        self.in_proj = nn.Linear(embed_dim, feedforward_dim)

        self.hidden_balancer = Balancer(
            feedforward_dim,
            channel_dim=-1,
            min_positive=0.3,
            max_positive=1.0,
            min_abs=0.75,
            max_abs=5.0,
        )

        # shared_dim=0 means we share the dropout mask along the time axis
        self.out_proj = ActivationDropoutAndLinear(
            feedforward_dim,
            embed_dim,
            activation="SwooshL",
            dropout_p=dropout,
            dropout_shared_dim=0,
            bias=True,
            initial_scale=0.1,
        )

        self.out_whiten = Whiten(
            num_groups=1,
            whitening_limit=_whitening_schedule(7.5),
            prob=(0.025, 0.25),
            grad_scale=0.01,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_proj(x)
        x = self.hidden_balancer(x)
        # out_proj contains SwooshL activation, then dropout, then linear.
        x = self.out_proj(x)
        x = self.out_whiten(x)
        return x


class NonlinAttention(nn.Module):
    """This is like the ConvolutionModule, but refactored so that we use multiplication by attention weights (borrowed
       from the attention module) in place of actual convolution.  We also took out the second nonlinearity, the
       one after the attention mechanism.

    Args:
        channels (int): The number of channels of conv layers.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
    ) -> None:
        super().__init__()

        self.hidden_channels = hidden_channels

        self.in_proj = nn.Linear(channels, hidden_channels * 3, bias=True)

        # balancer that goes before the sigmoid.  Have quite a large min_abs value, at 2.0,
        # because we noticed that well-trained instances of this module have abs-value before the sigmoid
        # starting from about 3, and poorly-trained instances of the module have smaller abs values
        # before the sigmoid.
        self.balancer = Balancer(
            hidden_channels,
            channel_dim=-1,
            min_positive=ScheduledFloat((0.0, 0.25), (20000.0, 0.05)),
            max_positive=ScheduledFloat((0.0, 0.75), (20000.0, 0.95)),
            min_abs=0.5,
            max_abs=5.0,
        )
        self.tanh = nn.Tanh()

        self.identity1 = Identity()  # for diagnostics.
        self.identity2 = Identity()  # for diagnostics.
        self.identity3 = Identity()  # for diagnostics.

        self.out_proj = ScaledLinear(hidden_channels, channels, bias=True, initial_scale=0.05)

        self.whiten1 = Whiten(
            num_groups=1,
            whitening_limit=_whitening_schedule(5.0),
            prob=(0.025, 0.25),
            grad_scale=0.01,
        )

        self.whiten2 = Whiten(
            num_groups=1,
            whitening_limit=_whitening_schedule(5.0, ratio=3.0),
            prob=(0.025, 0.25),
            grad_scale=0.01,
        )

    def forward(
        self,
        x: Tensor,
        attn_weights: Tensor,
    ) -> Tensor:
        """.
                Args:
                   x: a Tensor of shape (seq_len, batch_size, num_channels)
        attn_weights: a Tensor of shape (num_heads, batch_size, seq_len, seq_len)
                Returns:
                   a Tensor with the same shape as x
        """
        x = self.in_proj(x)

        (seq_len, batch_size, _) = x.shape
        hidden_channels = self.hidden_channels

        s, x, y = x.chunk(3, dim=2)

        # s will go through tanh.

        s = self.balancer(s)
        s = self.tanh(s)

        s = s.unsqueeze(-1).reshape(seq_len, batch_size, hidden_channels)
        x = self.whiten1(x)
        x = x * s
        x = self.identity1(x)  # diagnostics only, it's the identity.

        (seq_len, batch_size, embed_dim) = x.shape
        num_heads = attn_weights.shape[0]
        assert attn_weights.shape == (num_heads, batch_size, seq_len, seq_len)

        x = x.reshape(seq_len, batch_size, num_heads, -1).permute(2, 1, 0, 3)
        # now x: (num_heads, batch_size, seq_len, head_dim)
        x = torch.matmul(attn_weights, x)
        # now x: (num_heads, batch_size, seq_len, head_dim)
        x = x.permute(2, 1, 0, 3).reshape(seq_len, batch_size, -1)

        y = self.identity2(y)
        x = x * y
        x = self.identity3(x)

        x = self.out_proj(x)
        x = self.whiten2(x)
        return x

    def streaming_forward(
        self,
        x: Tensor,
        attn_weights: Tensor,
        cached_x: Tensor,
        left_context_len: int,
    ) -> Tuple[Tensor, Tensor]:
        """.
        Args:
            x: a Tensor of shape (seq_len, batch_size, num_channels)
            attn_weights: a Tensor of shape (num_heads, batch_size, seq_len, seq_len)
            cached_x: left context, a Tensor of shape
              (num_heads, batch_size, left_context_len, head_dim)
            left_context_len: number of left context frames.
        Returns:
            - a Tensor with the same shape as x
            - updated left context with same shape as cached_x
        """
        x = self.in_proj(x)

        (seq_len, batch_size, _) = x.shape
        hidden_channels = self.hidden_channels

        s, x, y = x.chunk(3, dim=2)

        # s will go through tanh.
        s = self.tanh(s)

        s = s.unsqueeze(-1).reshape(seq_len, batch_size, hidden_channels)
        x = x * s

        (seq_len, batch_size, embed_dim) = x.shape
        num_heads = attn_weights.shape[0]
        assert attn_weights.shape == (
            num_heads,
            batch_size,
            seq_len,
            left_context_len + seq_len,
        )

        x = x.reshape(seq_len, batch_size, num_heads, -1).permute(2, 1, 0, 3)
        # now x: (num_heads, batch_size, seq_len, head_dim)

        # Pad cached tensor
        assert cached_x.shape[2] == left_context_len, (
            cached_x.shape[2],
            left_context_len,
        )
        x_pad = torch.cat([cached_x, x], dim=2)
        # Update cached tensor
        cached_x = x_pad[:, :, -left_context_len:, :]

        x = torch.matmul(attn_weights, x_pad)
        # now x: (num_heads, batch_size, seq_len, head_dim)
        x = x.permute(2, 1, 0, 3).reshape(seq_len, batch_size, -1)

        x = x * y

        x = self.out_proj(x)
        return x, cached_x


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Zipformer2 model.
    Modified from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/zipformer/convolution.py

    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernerl size of conv layers.
        bias (bool): Whether to use bias in conv layers (default=True).

    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        causal: bool,
    ) -> None:
        """Construct a ConvolutionModule object."""
        super(ConvolutionModule, self).__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0

        bottleneck_dim = channels
        self.causal = causal

        self.in_proj = nn.Linear(
            channels,
            2 * bottleneck_dim,
        )
        # the gradients on in_proj are a little noisy, likely to do with the
        # sigmoid in glu.

        # after in_proj we put x through a gated linear unit (nn.functional.glu).
        # For most layers the normal rms value of channels of x seems to be in the range 1 to 4,
        # but sometimes, for some reason, for layer 0 the rms ends up being very large,
        # between 50 and 100 for different channels.  This will cause very peaky and
        # sparse derivatives for the sigmoid gating function, which will tend to make
        # the loss function not learn effectively.  (for most layers the average absolute values
        # are in the range 0.5..9.0, and the average p(x>0), i.e. positive proportion,
        # at the output of pointwise_conv1.output is around 0.35 to 0.45 for different
        # layers, which likely breaks down as 0.5 for the "linear" half and
        # 0.2 to 0.3 for the part that goes into the sigmoid.  The idea is that if we
        # constrain the rms values to a reasonable range via a constraint of max_abs=10.0,
        # it will be in a better position to start learning something, i.e. to latch onto
        # the correct range.
        self.balancer1 = Balancer(
            bottleneck_dim,
            channel_dim=-1,
            min_positive=ScheduledFloat((0.0, 0.05), (8000.0, 0.025)),
            max_positive=1.0,
            min_abs=1.5,
            max_abs=ScheduledFloat((0.0, 5.0), (8000.0, 10.0), default=1.0),
        )

        self.activation1 = Identity()  # for diagnostics

        self.sigmoid = nn.Sigmoid()

        self.activation2 = Identity()  # for diagnostics

        assert kernel_size % 2 == 1

        self.depthwise_conv = (
            ChunkCausalDepthwiseConv1d(channels=bottleneck_dim, kernel_size=kernel_size)
            if causal
            else nn.Conv1d(
                in_channels=bottleneck_dim,
                out_channels=bottleneck_dim,
                groups=bottleneck_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
        )

        self.balancer2 = Balancer(
            bottleneck_dim,
            channel_dim=1,
            min_positive=ScheduledFloat((0.0, 0.1), (8000.0, 0.05)),
            max_positive=1.0,
            min_abs=ScheduledFloat((0.0, 0.2), (20000.0, 0.5)),
            max_abs=10.0,
        )

        self.whiten = Whiten(
            num_groups=1,
            whitening_limit=_whitening_schedule(7.5),
            prob=(0.025, 0.25),
            grad_scale=0.01,
        )

        self.out_proj = ActivationDropoutAndLinear(
            bottleneck_dim,
            channels,
            activation="SwooshR",
            dropout_p=0.0,
            initial_scale=0.05,
        )

    def forward(
        self,
        x: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
        chunk_size: int = -1,
    ) -> Tensor:
        """Compute convolution module.

        Args:
            x: Input tensor (#time, batch, channels).
           src_key_padding_mask: the mask for the src keys per batch (optional):
               (batch, #time), contains True in masked positions.

        Returns:
            Tensor: Output tensor (#time, batch, channels).

        """

        x = self.in_proj(x)  # (time, batch, 2*channels)

        x, s = x.chunk(2, dim=2)
        s = self.balancer1(s)
        s = self.sigmoid(s)
        x = self.activation1(x)  # identity.
        x = x * s
        x = self.activation2(x)  # identity

        # (time, batch, channels)

        # exchange the temporal dimension and the feature dimension
        x = x.permute(1, 2, 0)  # (#batch, channels, time).

        if src_key_padding_mask is not None:
            x = x.masked_fill(src_key_padding_mask.unsqueeze(1).expand_as(x), 0.0)

        if not torch.jit.is_scripting() and not torch.jit.is_tracing() and chunk_size >= 0:
            # Not support exporting a model for simulated streaming decoding
            assert self.causal, "Must initialize model with causal=True if you use chunk_size"
            x = self.depthwise_conv(x, chunk_size=chunk_size)
        else:
            x = self.depthwise_conv(x)

        x = self.balancer2(x)
        x = x.permute(2, 0, 1)  # (time, batch, channels)

        x = self.whiten(x)  # (time, batch, channels)
        x = self.out_proj(x)  # (time, batch, channels)

        return x

    def streaming_forward(
        self,
        x: Tensor,
        cache: Tensor,
        src_key_padding_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Compute convolution module in streaming forward mode.

        Args:
            x: Input tensor (#time, batch, channels).
            cache: cached left context for depthwise_conv of shape
              (#batch, channels, left_pad)
            src_key_padding_mask: the mask for the src keys per batch (optional):
              (batch, #time), contains True in masked positions.

        Returns:
            - Output tensor (#time, batch, channels).
            - Updated cache (#batch, channels, left_pad)
        """

        x = self.in_proj(x)  # (time, batch, 2*channels)

        x, s = x.chunk(2, dim=2)
        s = self.sigmoid(s)
        x = x * s
        # (time, batch, channels)

        # exchange the temporal dimension and the feature dimension
        x = x.permute(1, 2, 0)  # (#batch, channels, time).

        if src_key_padding_mask is not None:
            x = x.masked_fill(src_key_padding_mask.unsqueeze(1).expand_as(x), 0.0)

        x, cache = self.depthwise_conv.streaming_forward(x, cache=cache)

        x = x.permute(2, 0, 1)  # (time, batch, channels)

        x = self.out_proj(x)  # (time, batch, channels)

        return x, cache
