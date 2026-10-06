from typing import Any

import torch
import triton
import triton.language as tl


@triton.jit
def _backward(Y: tl.tensor, G: tl.tensor, DX: tl.tensor, N: tl.constexpr, BLOCK: tl.constexpr) -> None:
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    y = tl.load(Y + row * N + col, col < N, 0).to(tl.float32)
    g = tl.load(G + row * N + col, col < N, 0).to(tl.float32)
    product = y * g
    dx = product - y * tl.sum(product, 0)
    tl.store(DX + row * N + col, dx, col < N)


class _Softmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        y = x.softmax(dim=-1)
        if x.requires_grad and torch.is_autocast_enabled("cuda"):
            y = y.to(torch.get_autocast_dtype("cuda"))
        ctx.save_for_backward(y)
        ctx.input_dtype = x.dtype
        return y

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> torch.Tensor:
        (y,) = ctx.saved_tensors
        grad = grad.contiguous()
        dx = torch.empty(y.shape, device=y.device, dtype=ctx.input_dtype)
        n = y.shape[-1]
        _backward[(y.numel() // n,)](y, grad, dx, n, triton.next_power_of_2(n), enable_fp_fusion=False)
        return dx


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    if not x.is_cuda or dim not in (-1, x.ndim - 1):
        raise ValueError("The Triton softmax requires CUDA and the last dimension.")
    return _Softmax.apply(x)
