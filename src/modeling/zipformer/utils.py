from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class TransposeLast(nn.Module):
    def __init__(
        self,
        deconstruct_idx: int | None = None,
        transpose_dim: int = -2,
    ) -> None:
        super().__init__()
        self.deconstruct_idx = deconstruct_idx
        self.transpose_dim = transpose_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(self.transpose_dim, -1)


class Fp32LayerNorm(nn.LayerNorm):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


class Fp32GroupNorm(nn.GroupNorm):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = F.group_norm(
            input.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return x.new(x)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad * ctx.scale, None
