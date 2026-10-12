import mlx.core as mx

# ---------------------------------------------------------
# Fuse the decoder gates, causal depthwise convolution, and state update.
# Keep fp16 rounding at the same boundaries as the unfused operators.
# ---------------------------------------------------------
_SHORT_CONV = mx.fast.metal_kernel(
    name="lambda_short_conv",
    input_names=["projected", "state", "weight", "bias"],
    output_names=["output", "next_state"],
    source="""
        const uint L = projected_shape[1];
        uint channel = thread_position_in_grid.x % D;
        uint batch = thread_position_in_grid.x / D;
        uint time = thread_position_in_grid.y;
        uint base = (batch * L + time) * 3 * D + channel;
        float sum = 0.0f;
        for (uint k = 0; k < K; ++k) {
            int position = int(time + k) - K + 1;
            T value;
            if (position < 0) {
                value = state[(batch * (K - 1) + position + K - 1) * D + channel];
            } else {
                uint i = (batch * L + position) * 3 * D + channel;
                value = T(projected[i] * projected[i + 2 * D]);
            }
            sum = metal::fma(float(value), float(weight[channel * K + k]), sum);
        }
        T convolved = T(sum);
        if (BIAS) convolved = T(convolved + bias[channel]);
        output[(batch * L + time) * D + channel] = T(projected[base + D] * convolved);
        if (time == 0) {
            for (uint k = 0; k < K - 1; ++k) {
                int position = int(L + k) - K + 1;
                T value;
                if (position < 0) {
                    value = state[(batch * (K - 1) + position + K - 1) * D + channel];
                } else {
                    uint i = (batch * L + position) * 3 * D + channel;
                    value = T(projected[i] * projected[i + 2 * D]);
                }
                next_state[(batch * (K - 1) + k) * D + channel] = value;
            }
        }
    """,
)


def short_conv(projected: mx.array, state: mx.array, weight: mx.array, bias: mx.array | None) -> tuple[mx.array, mx.array]:
    batch, length, channels = projected.shape
    dimension = channels // 3
    kernel = weight.shape[1]
    output, next_state = _SHORT_CONV(
        inputs=[projected, state, weight, weight if bias is None else bias],
        template=[("T", projected.dtype), ("D", dimension), ("K", kernel), ("BIAS", bias is not None)],
        grid=(batch * dimension, length, 1),
        threadgroup=(min(256, batch * dimension), 1, 1),
        output_shapes=[(batch, length, dimension), (batch, kernel - 1, dimension)],
        output_dtypes=[projected.dtype, projected.dtype],
    )
    return output, next_state


# ---------------------------------------------------------
# Adjacent threads process adjacent channels instead of separate groups.
# Apply SwooshR immediately after the encoder depthwise convolution.
# ---------------------------------------------------------
_DEPTHWISE_SWOOSH = mx.fast.metal_kernel(
    name="lambda_depthwise_swoosh",
    input_names=["x", "weight", "bias"],
    output_names=["output"],
    source="""
        const int L = x_shape[1];
        uint channel = thread_position_in_grid.x % D;
        uint batch = thread_position_in_grid.x / D;
        uint time = thread_position_in_grid.y;
        float sum = 0.0f;
        for (uint k = 0; k < K; ++k) {
            int position = int(time + k) - K / 2;
            if (position >= 0 && position < L) {
                sum = metal::fma(float(x[(batch * L + position) * D + channel]), float(weight[channel * K + k]), sum);
            }
        }
        float value = float(T(T(sum) + bias[channel]));
        float shifted = value - 1.0f;
        output[(batch * L + time) * D + channel] = T(metal::max(shifted, 0.0f) + metal::log(1.0f + metal::exp(-metal::abs(shifted))) - 0.08f * value - 0.313261687f);
    """,
)


def depthwise_swoosh(x: mx.array, weight: mx.array, bias: mx.array) -> mx.array:
    batch, length, dimension = x.shape
    return _DEPTHWISE_SWOOSH(
        inputs=[x, weight, bias],
        template=[("T", x.dtype), ("D", dimension), ("K", weight.shape[1])],
        grid=(batch * dimension, length, 1),
        threadgroup=(min(256, batch * dimension), 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]
