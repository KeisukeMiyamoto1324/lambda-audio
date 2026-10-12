import math

import mlx.core as mx
import mlx.nn as nn


def positional_encoding(length: int, dimension: int, dtype: mx.Dtype) -> mx.array:
    # ---------------------------------------------------------
    # Encode compact relative offsets in the original Zipformer order.
    # ---------------------------------------------------------
    offsets = mx.arange(1 - length, length, dtype=mx.float32)[:, None]
    compression = math.sqrt(dimension)
    compressed = compression * mx.sign(offsets) * (mx.log(mx.abs(offsets) + compression) - math.log(compression))
    angles = mx.arctan(compressed / (dimension / (2 * math.pi))) * mx.arange(1, dimension // 2 + 1)
    encoding = mx.stack((mx.cos(angles), mx.sin(angles)), axis=-1).reshape(2 * length - 1, dimension)
    return mx.concatenate((encoding[:, :-1], mx.ones((2 * length - 1, 1))), axis=-1).astype(dtype)


class BiasNorm(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.bias = mx.zeros((dimension,))
        self.log_scale = mx.array(1.0)

    def __call__(self, x: mx.array) -> mx.array:
        scale = mx.rsqrt(mx.mean(mx.square(x.astype(mx.float32) - self.bias.astype(mx.float32)), axis=-1, keepdims=True))
        return (x.astype(mx.float32) * scale * mx.exp(self.log_scale.astype(mx.float32))).astype(x.dtype)


class Bypass(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.bypass_scale = mx.full((dimension,), 0.5)

    def __call__(self, original: mx.array, x: mx.array) -> mx.array:
        return original + (x - original) * self.bypass_scale


class FeedForward(nn.Module):
    def __init__(self, dimension: int, hidden: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(dimension, hidden)
        self.out_proj = nn.Linear(hidden, dimension)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.in_proj(x)
        return self.out_proj((nn.softplus(x.astype(mx.float32) - 4) - 0.08 * x.astype(mx.float32) - 0.035).astype(x.dtype))


class Convolution(nn.Module):
    def __init__(self, dimension: int, kernel: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(dimension, 2 * dimension)
        self.depthwise_conv = nn.Conv1d(dimension, dimension, kernel, padding=kernel // 2, groups=dimension)
        self.out_proj = nn.Linear(dimension, dimension)

    def __call__(self, x: mx.array, padding_mask: mx.array) -> mx.array:
        x, gate = mx.split(self.in_proj(x), 2, axis=-1)
        x = mx.where(padding_mask[..., None], 0, x * mx.sigmoid(gate))
        x = self.depthwise_conv(x)
        return self.out_proj((nn.softplus(x.astype(mx.float32) - 1) - 0.08 * x.astype(mx.float32) - 0.313261687).astype(x.dtype))


class AttentionWeights(nn.Module):
    def __init__(self, dimension: int, pos_dim: int, heads: int, query_dim: int, pos_head_dim: int) -> None:
        super().__init__()
        self.heads = heads
        self.query_dim = query_dim
        self.pos_head_dim = pos_head_dim
        self.in_proj = nn.Linear(dimension, heads * (2 * query_dim + pos_head_dim))
        self.linear_pos = nn.Linear(pos_dim, heads * pos_head_dim, bias=False)

    def __call__(self, x: mx.array, positions: mx.array, padding_mask: mx.array) -> mx.array:
        batch, length, _ = x.shape
        query, key, pos_query = mx.split(self.in_proj(x), [self.heads * self.query_dim, 2 * self.heads * self.query_dim], axis=-1)
        query = query.reshape(batch, length, self.heads, self.query_dim).transpose(0, 2, 1, 3)
        key = key.reshape(batch, length, self.heads, self.query_dim).transpose(0, 2, 3, 1)
        pos_query = pos_query.reshape(batch, length, self.heads, self.pos_head_dim).transpose(0, 2, 1, 3)
        positions = self.linear_pos(positions).reshape(2 * length - 1, self.heads, self.pos_head_dim).transpose(1, 2, 0)
        relative_scores = pos_query @ positions
        indices = length - 1 - mx.arange(length)[:, None] + mx.arange(length)[None, :]
        scores = query @ key + mx.take_along_axis(relative_scores, indices[None, None], axis=-1)
        return mx.softmax(mx.where(padding_mask[:, None, None, :], -1000, scores), axis=-1, precise=True)


class SelfAttention(nn.Module):
    def __init__(self, dimension: int, heads: int, value_dim: int) -> None:
        super().__init__()
        self.heads = heads
        self.in_proj = nn.Linear(dimension, heads * value_dim)
        self.out_proj = nn.Linear(heads * value_dim, dimension)

    def __call__(self, x: mx.array, weights: mx.array) -> mx.array:
        batch, length, _ = x.shape
        values = self.in_proj(x).reshape(batch, length, self.heads, -1).transpose(0, 2, 1, 3)
        output = (weights @ values).transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.out_proj(output)


class NonlinearAttention(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        hidden = 3 * dimension // 4
        self.in_proj = nn.Linear(dimension, 3 * hidden)
        self.out_proj = nn.Linear(hidden, dimension)

    def __call__(self, x: mx.array, weights: mx.array) -> mx.array:
        gate, values, multiplier = mx.split(self.in_proj(x), 3, axis=-1)
        output = (weights[:, :1] @ (values * mx.tanh(gate))[:, None])[:, 0]
        return self.out_proj(output * multiplier)


class EncoderLayer(nn.Module):
    def __init__(self, dimension: int, pos_dim: int, heads: int, query_dim: int, pos_head_dim: int, value_dim: int, feedforward_dim: int, kernel: int) -> None:
        super().__init__()
        self.self_attn_weights = AttentionWeights(dimension, pos_dim, heads, query_dim, pos_head_dim)
        self.self_attn1 = SelfAttention(dimension, heads, value_dim)
        self.self_attn2 = SelfAttention(dimension, heads, value_dim)
        self.nonlin_attention = NonlinearAttention(dimension)
        self.feed_forward1 = FeedForward(dimension, feedforward_dim * 3 // 4)
        self.feed_forward2 = FeedForward(dimension, feedforward_dim)
        self.feed_forward3 = FeedForward(dimension, feedforward_dim * 5 // 4)
        self.conv_module1 = Convolution(dimension, kernel)
        self.conv_module2 = Convolution(dimension, kernel)
        self.bypass_mid = Bypass(dimension)
        self.bypass = Bypass(dimension)
        self.bypass_scale = mx.full((dimension,), 0.5)
        self.norm = BiasNorm(dimension)

    def __call__(self, x: mx.array, positions: mx.array, padding_mask: mx.array) -> mx.array:
        # ---------------------------------------------------------
        # Preserve the original residual and bypass order at inference.
        # ---------------------------------------------------------
        original = x
        weights = self.self_attn_weights(x, positions, padding_mask)
        x = x + self.feed_forward1(x)
        x = x + self.nonlin_attention(x, weights)
        x = x + self.self_attn1(x, weights)
        x = x + self.conv_module1(x, padding_mask)
        x = x + self.feed_forward2(x)
        x = self.bypass_mid(original, x)
        x = x + self.self_attn2(x, weights)
        x = x + self.conv_module2(x, padding_mask)
        x = x + self.feed_forward3(x)
        return self.bypass(original, self.norm(x))


class Downsample(nn.Module):
    def __init__(self, factor: int) -> None:
        super().__init__()
        self.bias = mx.zeros((factor,))
        self.factor = factor

    def __call__(self, x: mx.array, padding_mask: mx.array) -> mx.array:
        # ---------------------------------------------------------
        # Extend the last valid frame before reducing each sequence.
        # ---------------------------------------------------------
        batch, length, dimension = x.shape
        last_indices = mx.sum(~padding_mask, axis=-1) - 1
        last = mx.take_along_axis(x, last_indices[:, None, None], axis=1)
        x = mx.where(padding_mask[..., None], last, x)
        pad = -length % self.factor
        x = mx.concatenate((x, mx.broadcast_to(x[:, -1:], (batch, pad, dimension))), axis=1)
        x = x.reshape(batch, -1, self.factor, dimension)
        return mx.sum(x * mx.softmax(self.bias)[None, None, :, None], axis=2)
