from typing import Any

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["M", "K", "N"])
def _gated_mm(A: tl.tensor, B: tl.tensor, OUT: tl.tensor, ACTIVE: tl.tensor, M: tl.tensor, K: tl.tensor, N: tl.tensor, TRANSPOSE: tl.constexpr, BLOCK: tl.constexpr) -> None:
    group = tl.program_id(0)
    rows = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    cols = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    inner = tl.arange(0, BLOCK)
    acc = tl.full((BLOCK, BLOCK), 0, tl.float32)
    if tl.load(ACTIVE):
        for start in range(tl.cdiv(K, BLOCK)):
            k = start * BLOCK + inner
            if TRANSPOSE:
                a = tl.load(A + group * M * K + k[None, :] * M + rows[:, None], (rows[:, None] < M) & (k[None, :] < K), 0)
            else:
                a = tl.load(A + group * M * K + rows[:, None] * K + k[None, :], (rows[:, None] < M) & (k[None, :] < K), 0)
            b = tl.load(B + group * K * N + k[:, None] * N + cols[None, :], (k[:, None] < K) & (cols[None, :] < N), 0)
            acc += tl.dot(a, b, input_precision="tf32x3")
    tl.store(OUT + group * M * N + rows[:, None] * N + cols[None, :], acc, (rows[:, None] < M) & (cols[None, :] < N))


@triton.jit(do_not_specialize=["D", "TOTAL"])
def _matrix(
    COV: tl.tensor,
    DMEAN: tl.tensor,
    QMEAN: tl.tensor,
    SELECTED: tl.tensor,
    PROB: tl.tensor,
    ACTIVE: tl.tensor,
    OUT: tl.tensor,
    D: tl.tensor,
    TOTAL: tl.tensor,
    LIMIT: tl.tensor,
    PMIN: tl.tensor,
    PMAX: tl.tensor,
    BLOCK: tl.constexpr,
) -> None:
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    d = tl.load(DMEAN)
    q = tl.load(QMEAN)
    denom = d * d + 1.0e-20
    selected = tl.load(SELECTED)
    active = selected & (q / denom >= LIMIT)
    if tl.program_id(0) == 0:
        tl.store(ACTIVE, active)
        if selected:
            tl.store(PROB, tl.where(active, PMAX, PMIN))
    cov = tl.load(COV + i, i < TOTAL * D, 0)
    diagonal = (i % D) == ((i // D) % D)
    value = (4.0 / (TOTAL * denom)) * (cov - tl.where(diagonal, (q / denom) * d, 0.0))
    tl.store(OUT + i, value, i < TOTAL * D)


@triton.jit(do_not_specialize=["SIZE"])
def _norms(P: tl.tensor, G: tl.tensor, ACTIVE: tl.tensor, SUMS: tl.tensor, SIZE: tl.tensor, BLOCK: tl.constexpr) -> None:
    block = tl.program_id(0)
    i = block * BLOCK + tl.arange(0, BLOCK)
    ps = tl.full((), 0, tl.float32)
    gs = tl.full((), 0, tl.float32)
    if tl.load(ACTIVE):
        p = tl.load(P + i, i < SIZE, 0)
        g = tl.load(G + i, i < SIZE, 0).to(tl.float32)
        ps = tl.sum(p * p, 0)
        gs = tl.sum(g * g, 0)
    tl.store(SUMS + block * 2, ps)
    tl.store(SUMS + block * 2 + 1, gs)


@triton.jit(do_not_specialize=["PARTS"])
def _scale(SUMS: tl.tensor, OUT: tl.tensor, PARTS: tl.tensor, SCALE: tl.tensor, BLOCK: tl.constexpr) -> None:
    i = tl.arange(0, BLOCK)
    p = tl.sum(tl.load(SUMS + i * 2, i < PARTS, 0), 0)
    g = tl.sum(tl.load(SUMS + i * 2 + 1, i < PARTS, 0), 0)
    tl.store(OUT, SCALE * (tl.sqrt(g) / (tl.sqrt(p) + 1.0e-20)))


@triton.jit(do_not_specialize=["N", "D", "GROUPS", "SIZE"])
def _apply(
    P: tl.tensor, G: tl.tensor, SCALE: tl.tensor, ACTIVE: tl.tensor, OUT: tl.tensor, N: tl.tensor, D: tl.tensor, GROUPS: tl.tensor, SIZE: tl.tensor, BLOCK: tl.constexpr
) -> None:
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    g = tl.load(G + i, i < SIZE, 0)
    y = g.to(tl.float32)
    if tl.load(ACTIVE):
        offset = ((i // D) % GROUPS) * N * D + (i // (D * GROUPS)) * D + i % D
        p = tl.load(P + offset, i < SIZE, 0)
        correction = (p * tl.load(SCALE)).to(g.dtype)
        y += correction.to(tl.float32)
    tl.store(OUT + i, y, i < SIZE)


def whitening_backward(
    x: torch.Tensor, grad: torch.Tensor, probability: torch.Tensor, selected: torch.Tensor, groups: int, limit: float, min_prob: float, max_prob: float, scale: float
) -> torch.Tensor:
    channels = x.shape[-1]
    n = x.numel() // channels
    d = channels // groups
    with torch.autocast("cuda", enabled=False):
        centered = x.float().reshape(n, groups, d).permute(1, 0, 2).contiguous()
        centered = centered - centered.mean(1, keepdim=True)
        covariance = torch.empty((groups, d, d), device=x.device)
        _gated_mm[(groups, triton.cdiv(d, 32), triton.cdiv(d, 32))](centered, centered, covariance, selected, d, n, d, True, 32)
        diagonal_mean = covariance.diagonal(dim1=-2, dim2=-1).mean()
        square_mean = covariance.square().sum() / channels
        matrix = torch.empty_like(covariance)
        active = torch.empty((), device=x.device, dtype=torch.bool)
        _matrix[(triton.cdiv(covariance.numel(), 256),)](
            covariance, diagonal_mean, square_mean, selected, probability, active, matrix, d, channels, limit, min_prob, max_prob, 256, enable_fp_fusion=False
        )
        penalty = torch.empty_like(centered)
        _gated_mm[(groups, triton.cdiv(n, 32), triton.cdiv(d, 32))](centered, matrix, penalty, active, n, d, d, False, 32)
        penalty = penalty - penalty.mean(1, keepdim=True)
        g = grad.contiguous()
        parts = triton.cdiv(g.numel(), 1024)
        sums = torch.empty((parts, 2), device=x.device)
        factor = torch.empty((), device=x.device)
        result = torch.empty_like(g)
        _norms[(parts,)](penalty, g, active, sums, g.numel(), 1024)
        _scale[(1,)](sums, factor, parts, scale, triton.next_power_of_2(parts))
        _apply[(parts,)](penalty, g, factor, active, result, n, d, groups, g.numel(), 1024)
        return result


class _Whitening(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, module: Any, draw: float) -> torch.Tensor:
        ctx.save_for_backward(x, module.kernel_probability >= draw)
        ctx.module = module
        return x

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        x, selected = ctx.saved_tensors
        w = ctx.module
        result = whitening_backward(x, grad, w.kernel_probability, selected, w.num_groups, float(w.whitening_limit), w.min_prob, w.max_prob, float(w.grad_scale))
        return result, None, None


def whiten(x: torch.Tensor, module: Any, draw: float) -> torch.Tensor:
    return _Whitening.apply(x, module, draw)
