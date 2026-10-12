import mlx.core as mx

# ---------------------------------------------------------
# Gather relative positions and mask padding inside a row-wise softmax.
# Accumulate the normalization in float32 while preserving score rounding.
# ---------------------------------------------------------
_RELATIVE_SOFTMAX = mx.fast.metal_kernel(
    name="lambda_relative_softmax",
    input_names=["scores", "relative", "padding"],
    output_names=["output"],
    source="""
        const uint length = scores_shape[3];
        uint row = thread_position_in_grid.y;
        uint tid = thread_position_in_threadgroup.x;
        uint query = row % length;
        uint batch = row / (H * length);
        float values[ITEMS];
        float maximum = -INFINITY;
        for (uint i = 0; i < ITEMS; ++i) {
            uint key = tid + i * 256;
            float value = -INFINITY;
            if (key < length) {
                value = padding[batch * length + key] ? -1000.0f : float(T(scores[row * length + key] + relative[row * (2 * length - 1) + length - 1 - query + key]));
            }
            values[i] = value;
            maximum = metal::max(maximum, value);
        }
        threadgroup float shared[8];
        maximum = simd_max(maximum);
        if (tid % 32 == 0) shared[tid / 32] = maximum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        maximum = simd_max(tid % 32 < 8 ? shared[tid % 32] : -INFINITY);
        float total = 0.0f;
        for (uint i = 0; i < ITEMS; ++i) {
            values[i] = metal::exp(values[i] - maximum);
            total += values[i];
        }
        total = simd_sum(total);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid % 32 == 0) shared[tid / 32] = total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        total = simd_sum(tid % 32 < 8 ? shared[tid % 32] : 0.0f);
        for (uint i = 0; i < ITEMS; ++i) {
            uint key = tid + i * 256;
            if (key < length) output[row * length + key] = T(values[i] / total);
        }
    """,
)


def relative_softmax(scores: mx.array, relative: mx.array, padding_mask: mx.array) -> mx.array:
    batch, heads, length, _ = scores.shape
    return _RELATIVE_SOFTMAX(
        inputs=[scores, relative, padding_mask],
        template=[("T", scores.dtype), ("H", heads), ("ITEMS", (length + 255) // 256)],
        grid=(256, batch * heads * length, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[scores.shape],
        output_dtypes=[scores.dtype],
    )[0]
