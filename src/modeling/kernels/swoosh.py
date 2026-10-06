from typing import Any

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit(do_not_specialize=["N"])
def _forward(X: tl.tensor, Y: tl.tensor, N: tl.tensor, OFFSET: tl.constexpr, CONSTANT: tl.constexpr, BLOCK: tl.constexpr) -> None:
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, 0)
    # ---------------------------------------------------------
    # Evaluate softplus in FP32 with nonpositive exponential arguments.
    # Preserve input subtraction and output rounding in the input dtype.
    # ---------------------------------------------------------
    z = (x.to(tl.float32) - OFFSET).to(x.dtype).to(tl.float32)
    s = (tl.maximum(z, 0.0) + libdevice.log1p(tl.exp(-tl.abs(z)))).to(x.dtype)
    linear = (0.08 * x.to(tl.float32)).to(x.dtype)
    y = (s.to(tl.float32) - linear.to(tl.float32)).to(x.dtype)
    y = (y.to(tl.float32) - CONSTANT).to(x.dtype)
    tl.store(Y + i, y, i < N)


@triton.jit(do_not_specialize=["N"])
def _backward(X: tl.tensor, G: tl.tensor, DX: tl.tensor, N: tl.tensor, OFFSET: tl.constexpr, BLOCK: tl.constexpr) -> None:
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, 0)
    g = tl.load(G + i, i < N, 0)
    z = (x.to(tl.float32) - OFFSET).to(x.dtype).to(tl.float32)
    # ---------------------------------------------------------
    # Compute sigmoid with bounded exponentials for every input.
    # ---------------------------------------------------------
    sigmoid = tl.exp(tl.minimum(z, 0.0)) / (1.0 + tl.exp(-tl.abs(z)))
    nonlinear = (g.to(tl.float32) * sigmoid).to(x.dtype)
    linear = (-g.to(tl.float32) * 0.08).to(x.dtype)
    dx = nonlinear.to(tl.float32) + linear.to(tl.float32)
    tl.store(DX + i, dx, i < N)


class _Swoosh(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, offset: float, constant: float) -> torch.Tensor:
        x = x.contiguous()
        y = torch.empty_like(x)
        _forward[(triton.cdiv(x.numel(), 512),)](x, y, x.numel(), offset, constant, 512, enable_fp_fusion=False)
        ctx.save_for_backward(x)
        ctx.offset = offset
        return y

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        (x,) = ctx.saved_tensors
        dx = torch.empty_like(x)
        _backward[(triton.cdiv(x.numel(), 512),)](x, grad.contiguous(), dx, x.numel(), ctx.offset, 512, enable_fp_fusion=False)
        return dx, None, None


def swoosh(x: torch.Tensor, activation: str) -> torch.Tensor:
    if not x.is_cuda:
        raise ValueError("The Triton Swoosh requires CUDA.")
    if activation not in ("SwooshL", "SwooshR"):
        raise ValueError(f"Unsupported activation: {activation}")
    return _Swoosh.apply(x, 4.0 if activation == "SwooshL" else 1.0, 0.035 if activation == "SwooshL" else 0.313261687)
