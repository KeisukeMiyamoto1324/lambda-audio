from typing import Any

import torch
import triton
import triton.language as tl


@triton.jit
def _statistics(X: tl.tensor, MASK: tl.tensor, PART: tl.tensor, T: tl.constexpr, C: tl.constexpr, CHUNKS: tl.constexpr, BLOCK: tl.constexpr) -> None:
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    i = chunk * BLOCK + tl.arange(0, BLOCK)
    valid = (i < T) & ~tl.load(MASK + (row // C) * T + i, i < T, 1)
    x = tl.load(X + row * T + i, i < T, 0).to(tl.float32)
    count = tl.sum(valid.to(tl.float32), 0)
    mean = tl.sum(tl.where(valid, x, 0), 0) / tl.maximum(count, 1)
    delta = x - mean
    m2 = tl.sum(tl.where(valid, delta * delta, 0), 0)
    base = (row * CHUNKS + chunk) * 3
    tl.store(PART + base, mean)
    tl.store(PART + base + 1, m2)
    tl.store(PART + base + 2, count)


@triton.jit
def _merge(PART: tl.tensor, STATS: tl.tensor, CHUNKS: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr) -> None:
    row = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    base = (row * CHUNKS + i) * 3
    means = tl.load(PART + base, i < CHUNKS, 0)
    m2 = tl.load(PART + base + 1, i < CHUNKS, 0)
    counts = tl.load(PART + base + 2, i < CHUNKS, 0)
    count = tl.sum(counts, 0)
    mean = tl.sum(means * counts, 0) / count
    d = means - mean
    variance = tl.sum(m2 + d * d * counts, 0) / count
    tl.store(STATS + row * 3, mean)
    tl.store(STATS + row * 3 + 1, tl.rsqrt(variance + EPS))
    tl.store(STATS + row * 3 + 2, count)


@triton.jit
def _forward(X: tl.tensor, W: tl.tensor, B: tl.tensor, STATS: tl.tensor, Y: tl.tensor, T: tl.constexpr, C: tl.constexpr, BLOCK: tl.constexpr) -> None:
    row = tl.program_id(0)
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + row * T + i, i < T, 0).to(tl.float32)
    mean = tl.load(STATS + row * 3)
    inv = tl.load(STATS + row * 3 + 1)
    w = tl.load(W + row % C).to(tl.float32)
    b = tl.load(B + row % C).to(tl.float32)
    y = ((x - mean) * inv) * w + b
    tl.store(Y + row * T + i, y, i < T)


@triton.jit
def _backward_statistics(X: tl.tensor, G: tl.tensor, STATS: tl.tensor, PART: tl.tensor, T: tl.constexpr, CHUNKS: tl.constexpr, BLOCK: tl.constexpr) -> None:
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    i = chunk * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + row * T + i, i < T, 0).to(tl.float32)
    g = tl.load(G + row * T + i, i < T, 0).to(tl.float32)
    mean = tl.load(STATS + row * 3)
    inv = tl.load(STATS + row * 3 + 1)
    normalized = (x - mean) * inv
    base = row * 2 * CHUNKS + chunk
    tl.store(PART + base, tl.sum(g, 0))
    tl.store(PART + base + CHUNKS, tl.sum(g * normalized, 0))


@triton.jit
def _backward(
    X: tl.tensor, MASK: tl.tensor, G: tl.tensor, W: tl.tensor, STATS: tl.tensor, SUMS: tl.tensor, DX: tl.tensor, T: tl.constexpr, C: tl.constexpr, BLOCK: tl.constexpr
) -> None:
    row = tl.program_id(0)
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + row * T + i, i < T, 0).to(tl.float32)
    g = tl.load(G + row * T + i, i < T, 0).to(tl.float32)
    valid = (i < T) & ~tl.load(MASK + (row // C) * T + i, i < T, 1)
    mean = tl.load(STATS + row * 3)
    inv = tl.load(STATS + row * 3 + 1)
    count = tl.load(STATS + row * 3 + 2)
    w = tl.load(W + row % C).to(tl.float32)
    sum_g = tl.load(SUMS + row * 2)
    sum_gn = tl.load(SUMS + row * 2 + 1)
    normalized = (x - mean) * inv
    correction = (sum_g + normalized * sum_gn) / count
    dx = w * inv * (g - tl.where(valid, correction, 0))
    tl.store(DX + row * T + i, dx, i < T)


class _MaskedGroupNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
        x = x.contiguous()
        mask = mask.contiguous()
        b, c, t = x.shape
        chunks = triton.cdiv(t, 2048)
        partial = torch.empty((b * c, chunks, 3), device=x.device, dtype=torch.float32)
        stats = torch.empty((b * c, 3), device=x.device, dtype=torch.float32)
        y = torch.empty_like(x)
        _statistics[(b * c, chunks)](x, mask, partial, t, c, chunks, 2048, enable_fp_fusion=False)
        _merge[(b * c,)](partial, stats, chunks, eps, triton.next_power_of_2(chunks), enable_fp_fusion=False)
        _forward[(b * c, chunks)](x, weight, bias, stats, y, t, c, 2048, enable_fp_fusion=False)
        ctx.save_for_backward(x, mask, weight, stats)
        ctx.bias_dtype = bias.dtype
        return y

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None, torch.Tensor, torch.Tensor, None]:
        x, mask, weight, stats = ctx.saved_tensors
        grad = grad.contiguous()
        b, c, t = x.shape
        chunks = triton.cdiv(t, 2048)
        partial = torch.empty((b, c, 2, chunks), device=x.device, dtype=torch.float32)
        _backward_statistics[(b * c, chunks)](x, grad, stats, partial, t, chunks, 2048, enable_fp_fusion=False)
        sums = partial.sum(-1)
        dx = torch.empty_like(x)
        _backward[(b * c, chunks)](x, mask, grad, weight, stats, sums, dx, t, c, 2048, enable_fp_fusion=False)
        parameter_grads = sums.sum(0)
        return dx, None, parameter_grads[:, 1].to(weight.dtype), parameter_grads[:, 0].to(ctx.bias_dtype), None


def masked_group_norm(x: torch.Tensor, mask: torch.Tensor | None, weight: torch.Tensor, bias: torch.Tensor, groups: int, eps: float) -> torch.Tensor:
    if not x.is_cuda or x.ndim != 3 or groups != x.shape[1]:
        raise ValueError("The Triton masked GroupNorm requires CUDA BCT input and one group per channel.")
    if mask is None:
        mask = torch.zeros((x.shape[0], x.shape[2]), device=x.device, dtype=torch.bool)
    return _MaskedGroupNorm.apply(x, mask, weight, bias, eps)
