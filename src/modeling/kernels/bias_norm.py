from typing import Any

import torch
import triton
import triton.language as tl


@triton.jit
def _forward(X: tl.tensor, B: tl.tensor, L: tl.tensor, Y: tl.tensor, S: tl.tensor, C: tl.constexpr, BLOCK: tl.constexpr) -> None:
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * C + col, col < C, 0).to(tl.float32)
    b = tl.load(B + col, col < C, 0).to(tl.float32)
    d = x - b
    variance = tl.sum(tl.where(col < C, d * d, 0), 0) / C
    s = tl.rsqrt(variance) * tl.exp(tl.load(L).to(tl.float32))
    tl.store(Y + row * C + col, x * s, col < C)
    tl.store(S + row, s)


@triton.jit
def _backward(
    X: tl.tensor,
    B: tl.tensor,
    L: tl.tensor,
    S: tl.tensor,
    G: tl.tensor,
    DX: tl.tensor,
    DB: tl.tensor,
    DL: tl.tensor,
    C: tl.constexpr,
    STORED_OUTPUT: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * C + col, col < C, 0).to(tl.float32)
    if STORED_OUTPUT:
        x = x / tl.load(S + row)
    b = tl.load(B + col, col < C, 0).to(tl.float32)
    g = tl.load(G + row * C + col, col < C, 0).to(tl.float32)
    d = x - b
    variance = tl.sum(tl.where(col < C, d * d, 0), 0) / C
    s = tl.rsqrt(variance) * tl.exp(tl.load(L).to(tl.float32))
    dot = tl.sum(g * x, 0)
    db = d * (s * dot / (C * variance))
    tl.store(DX + row * C + col, g * s - db, col < C)
    tl.store(DB + row * C + col, db, col < C)
    tl.store(DL + row, s * dot)


class _BiasNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, bias: torch.Tensor, log_scale: torch.Tensor, store_output: bool) -> torch.Tensor:
        x = x.contiguous()
        c = x.shape[-1]
        y = torch.empty(x.shape, device=x.device, dtype=torch.float32)
        scales = torch.empty(x.numel() // c, device=x.device, dtype=torch.float32)
        _forward[(scales.numel(),)](x, bias, log_scale, y, scales, c, triton.next_power_of_2(c), enable_fp_fusion=False)
        ctx.save_for_backward(y if store_output else x, bias, log_scale, scales)
        ctx.store_output = store_output
        ctx.input_dtype = x.dtype
        return y

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        x, bias, log_scale, scales = ctx.saved_tensors
        c = x.shape[-1]
        dx = torch.empty(x.shape, device=x.device, dtype=ctx.input_dtype)
        db = torch.empty((scales.numel(), c), device=x.device, dtype=torch.float32)
        dl = torch.empty_like(scales)
        _backward[(scales.numel(),)](x, bias, log_scale, scales, grad.contiguous(), dx, db, dl, c, ctx.store_output, triton.next_power_of_2(c), enable_fp_fusion=False)
        return dx, db.sum(0).to(bias.dtype), dl.sum().reshape_as(log_scale).to(log_scale.dtype), None


def bias_norm(x: torch.Tensor, bias: torch.Tensor, log_scale: torch.Tensor, channel_dim: int, store_output: bool) -> torch.Tensor:
    if not x.is_cuda or bias.dtype != torch.float32 or log_scale.dtype != torch.float32:
        raise ValueError("The Triton BiasNorm requires CUDA and FP32 parameters.")
    return _BiasNorm.apply(x.movedim(channel_dim, -1), bias, log_scale, store_output).movedim(-1, channel_dim)
