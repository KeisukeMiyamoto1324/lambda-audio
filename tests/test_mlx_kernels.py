import numpy as np
import pytest

pytest.importorskip("mlx.core")

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

from mlx.kernel.attention import relative_softmax  # noqa: E402
from mlx.kernel.convolution import depthwise_swoosh, short_conv  # noqa: E402
from mlx.kernel.normalization import masked_group_norm  # noqa: E402


@pytest.mark.parametrize("dtype", [mx.float16, mx.float32])
@pytest.mark.parametrize("length", [1, 7, 257, 769])
def test_relative_softmax_matches_gather_and_mask(dtype: mx.Dtype, length: int) -> None:
    mx.random.seed(23)
    scores = mx.random.normal((2, 3, length, length)).astype(dtype)
    relative = mx.random.normal((2, 3, length, 2 * length - 1)).astype(dtype)
    mask = mx.arange(length)[None, :] >= mx.array([max(1, length // 2), length])[:, None]
    indices = length - 1 - mx.arange(length)[:, None] + mx.arange(length)[None, :]
    combined = scores + mx.take_along_axis(relative, indices[None, None], axis=-1)
    expected = mx.softmax(mx.where(mask[:, None, None, :], -1000, combined), axis=-1, precise=True)
    output = relative_softmax(scores, relative, mask)
    tolerance = 0.001 if dtype == mx.float16 else 2e-6
    np.testing.assert_allclose(np.array(output), np.array(expected), rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dtype", [mx.float16, mx.float32])
@pytest.mark.parametrize("batch,length,dimension,kernel", [(1, 1, 16, 3), (3, 1, 64, 3), (2, 5, 33, 3), (1, 2, 16, 5), (2, 4, 32, 1)])
@pytest.mark.parametrize("with_bias", [False, True])
def test_short_conv_matches_gated_convolution(dtype: mx.Dtype, batch: int, length: int, dimension: int, kernel: int, with_bias: bool) -> None:
    mx.random.seed(7)
    projected = mx.random.normal((batch, length, 3 * dimension)).astype(dtype)
    state = mx.random.normal((batch, kernel - 1, dimension)).astype(dtype)
    conv = nn.Conv1d(dimension, dimension, kernel, groups=dimension, bias=with_bias)
    conv.set_dtype(dtype)
    gate, multiplier, values = mx.split(projected, 3, axis=-1)
    sequence = mx.concatenate((state, gate * values), axis=1)
    expected = multiplier * conv(sequence)
    output, next_state = short_conv(projected, state, conv.weight, conv.bias if with_bias else None)
    tolerance = 0.002 if dtype == mx.float16 else 2e-6
    np.testing.assert_allclose(np.array(output), np.array(expected), rtol=tolerance, atol=tolerance)
    np.testing.assert_array_equal(np.array(next_state), np.array(sequence[:, length:]))


@pytest.mark.parametrize("dtype", [mx.float16, mx.float32])
@pytest.mark.parametrize("length,dimension", [(11, 33), (513, 64), (4097, 16)])
def test_masked_group_norm_ignores_padding(dtype: mx.Dtype, length: int, dimension: int) -> None:
    mx.random.seed(19)
    x = mx.random.normal((2, length, dimension)).astype(dtype)
    lengths = mx.array([length // 2, length])
    weight = mx.random.normal((dimension,)).astype(dtype)
    bias = mx.random.normal((dimension,)).astype(dtype)
    valid = (mx.arange(length)[None, :] < lengths[:, None])[..., None]
    values = x.astype(mx.float32)
    count = lengths[:, None, None]
    mean = mx.sum(values * valid, axis=1, keepdims=True) / count
    variance = mx.sum(mx.square(values - mean) * valid, axis=1, keepdims=True) / count
    expected = mx.where(valid, ((values - mean) * mx.rsqrt(variance + 1e-5) * weight + bias).astype(dtype), 0)
    output = masked_group_norm(x, lengths, weight, bias)
    tolerance = 0.002 if dtype == mx.float16 else 3e-6
    np.testing.assert_allclose(np.array(output), np.array(expected), rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dtype", [mx.float16, mx.float32])
@pytest.mark.parametrize("batch,length,dimension,kernel", [(1, 7, 16, 3), (2, 11, 33, 7), (3, 1, 64, 31)])
def test_depthwise_swoosh_matches_padded_convolution(dtype: mx.Dtype, batch: int, length: int, dimension: int, kernel: int) -> None:
    mx.random.seed(11)
    x = mx.random.normal((batch, length, dimension)).astype(dtype)
    conv = nn.Conv1d(dimension, dimension, kernel, padding=kernel // 2, groups=dimension)
    conv.set_dtype(dtype)
    value = conv(x).astype(mx.float32)
    expected = (nn.softplus(value - 1) - 0.08 * value - 0.313261687).astype(dtype)
    output = depthwise_swoosh(x, conv.weight, conv.bias)
    tolerance = 0.002 if dtype == mx.float16 else 2e-6
    np.testing.assert_allclose(np.array(output), np.array(expected), rtol=tolerance, atol=tolerance)
