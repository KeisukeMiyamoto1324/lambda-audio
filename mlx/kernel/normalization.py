import mlx.core as mx

# ---------------------------------------------------------
# Reduce time in small tiles with coalesced channel reads.
# Center each channel before accumulating to retain small variances.
# ---------------------------------------------------------
_MOMENTS = mx.fast.metal_kernel(
    name="lambda_masked_moments",
    input_names=["x", "lengths"],
    output_names=["moments"],
    source="""
        const uint length = x_shape[1];
        const uint chunks = (length + 255) / 256;
        uint channel = thread_position_in_grid.x;
        uint batch = thread_position_in_grid.z;
        uint chunk = threadgroup_position_in_grid.y;
        uint lane = thread_position_in_threadgroup.x;
        uint part = thread_position_in_threadgroup.y;
        float center = channel < D ? float(x[batch * length * D + channel]) : 0.0f;
        float sum = 0.0f;
        float square = 0.0f;
        for (uint t = chunk * 256 + part; t < (chunk + 1) * 256 && t < uint(lengths[batch]); t += 8) {
            float value = channel < D ? float(x[(batch * length + t) * D + channel]) - center : 0.0f;
            sum += value;
            square = metal::fma(value, value, square);
        }
        threadgroup float sums[256];
        threadgroup float squares[256];
        sums[part * 32 + lane] = sum;
        squares[part * 32 + lane] = square;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (part == 0 && channel < D) {
            for (uint p = 1; p < 8; ++p) {
                sum += sums[p * 32 + lane];
                square += squares[p * 32 + lane];
            }
            moments[((batch * chunks + chunk) * 2) * D + channel] = sum;
            moments[((batch * chunks + chunk) * 2 + 1) * D + channel] = square;
        }
    """,
)

_NORMALIZE = mx.fast.metal_kernel(
    name="lambda_masked_group_norm",
    input_names=["x", "lengths", "moments", "weight", "bias"],
    output_names=["output"],
    source="""
        uint index = thread_position_in_grid.x;
        if (index >= x_shape[0] * x_shape[1] * D) return;
        uint channel = index % D;
        uint time = index / D % x_shape[1];
        uint batch = index / (D * x_shape[1]);
        float count = float(lengths[batch]);
        float mean = moments[batch * 2 * D + channel] / count;
        float variance = metal::max(moments[(batch * 2 + 1) * D + channel] / count - mean * mean, 0.0f);
        float centered = float(x[index]) - float(x[batch * x_shape[1] * D + channel]);
        float value = (centered - mean) * metal::rsqrt(variance + 1e-5f);
        output[index] = time < uint(lengths[batch]) ? T(value * float(weight[channel]) + float(bias[channel])) : T(0);
    """,
)


def masked_group_norm(x: mx.array, lengths: mx.array, weight: mx.array, bias: mx.array) -> mx.array:
    batch, length, dimension = x.shape
    chunks = (length + 255) // 256
    moments = _MOMENTS(
        inputs=[x, lengths],
        template=[("D", dimension)],
        grid=((dimension + 31) // 32 * 32, chunks * 8, batch),
        threadgroup=(32, 8, 1),
        output_shapes=[(batch, chunks, 2, dimension)],
        output_dtypes=[mx.float32],
    )[0]
    return _NORMALIZE(
        inputs=[x, lengths, mx.sum(moments, axis=1), weight, bias],
        template=[("T", x.dtype), ("D", dimension)],
        grid=(x.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]
