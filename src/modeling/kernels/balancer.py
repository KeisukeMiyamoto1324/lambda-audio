from typing import Any

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["N", "PARTS"])
def _moments(X: tl.tensor, P: tl.tensor, N: tl.tensor, PARTS: tl.tensor, BLOCK: tl.constexpr) -> None:
    c = tl.program_id(0)
    k = tl.program_id(1)
    i = k * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + c * N + i, i < N, 0).to(tl.float32)
    tl.store(P + (c * PARTS + k) * 2, tl.sum(x, 0))
    tl.store(P + (c * PARTS + k) * 2 + 1, tl.sum(x * x, 0))


@triton.jit(do_not_specialize=["N", "PARTS"])
def _coefficients(
    P: tl.tensor, COEF: tl.tensor, N: tl.tensor, PARTS: tl.tensor, LO: tl.tensor, HI: tl.tensor, RLO: tl.tensor, RHI: tl.tensor, SCALE: tl.tensor, BLOCK: tl.constexpr
) -> None:
    c = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    mean = tl.sum(tl.load(P + (c * PARTS + i) * 2, i < PARTS, 0), 0) / N
    second = tl.sum(tl.load(P + (c * PARTS + i) * 2 + 1, i < PARTS, 0), 0) / N
    variance = second - mean * mean
    std = tl.sqrt(tl.maximum(variance, 1.0e-20))
    rms = tl.sqrt(tl.maximum(second, 1.0e-20))
    ratio = mean / std
    direction = tl.where(ratio < LO, -1.0, tl.where(ratio > HI, 1.0, 0.0))
    rdir = tl.where(rms < RLO, -1.0, tl.where(rms > RHI, 1.0, 0.0))
    a = -direction * mean / (N * std * std * std) * (variance >= 1.0e-20)
    r = rdir / (N * tl.maximum(second, 1.0e-20)) * (second >= 1.0e-20)
    a += r
    b = direction / (N * std) + r * mean
    norm = tl.maximum(tl.sqrt(a * a * tl.maximum(variance, 0.0) + b * b), 1.0e-20)
    tl.store(COEF + c * 3, mean)
    tl.store(COEF + c * 3 + 1, a * (SCALE / norm))
    tl.store(COEF + c * 3 + 2, b * (SCALE / norm))


@triton.jit(do_not_specialize=["N"])
def _apply(X: tl.tensor, G: tl.tensor, COEF: tl.tensor, Y: tl.tensor, N: tl.tensor, BLOCK: tl.constexpr) -> None:
    c = tl.program_id(0)
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + c * N + i, i < N, 0).to(tl.float32)
    g = tl.load(G + c * N + i, i < N, 0).to(tl.float32)
    mean = tl.load(COEF + c * 3)
    a = tl.load(COEF + c * 3 + 1)
    b = tl.load(COEF + c * 3 + 2)
    tl.store(Y + c * N + i, g + tl.abs(g) * (a * (x - mean) + b), i < N)


def balancer_backward(
    x: torch.Tensor, grad: torch.Tensor, min_mean: float, max_mean: float, min_rms: float, max_rms: float, grad_scale: float, channel_dim: int
) -> torch.Tensor:
    xc = x.movedim(channel_dim, 0).contiguous()
    gc = grad.movedim(channel_dim, 0).contiguous()
    channels = xc.shape[0]
    n = xc.numel() // channels
    parts = triton.cdiv(n, 1024)
    partial = torch.empty((channels, parts, 2), device=x.device)
    coefficients = torch.empty((channels, 3), device=x.device)
    output = torch.empty_like(gc)
    _moments[(channels, parts)](xc, partial, n, parts, 1024, enable_fp_fusion=False)
    _coefficients[(channels,)](partial, coefficients, n, parts, min_mean, max_mean, min_rms, max_rms, grad_scale, triton.next_power_of_2(parts), enable_fp_fusion=False)
    _apply[(channels, parts)](xc, gc, coefficients, output, n, 1024, enable_fp_fusion=False)
    return output.movedim(0, channel_dim)


class _Balancer(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, config: tuple[float, float, float, float, float, int]) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.config = config
        return x

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        (x,) = ctx.saved_tensors
        return balancer_backward(x, grad, *ctx.config), None


def balance(x: torch.Tensor, min_mean: float, max_mean: float, min_rms: float, max_rms: float, grad_scale: float, channel_dim: int) -> torch.Tensor:
    return _Balancer.apply(x, (min_mean, max_mean, min_rms, max_rms, grad_scale, channel_dim))
