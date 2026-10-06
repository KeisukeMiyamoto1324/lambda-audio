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
    use_triton: bool = False

    def forward(
        self,
        input: torch.Tensor,
        padding_mask: torch.BoolTensor | None = None,
    ) -> torch.Tensor:
        if self.use_triton:
            from src.modeling.kernels.group_norm import masked_group_norm

            return masked_group_norm(input, padding_mask, self.weight, self.bias, self.num_groups, self.eps)

        if padding_mask is not None:
            batch_size, channels, sequence_length = input.shape
            grouped = input.float().view(
                batch_size,
                self.num_groups,
                channels // self.num_groups,
                sequence_length,
            )
            valid = (~padding_mask).float()[:, None, None, :]
            count = valid.sum(dim=-1, keepdim=True) * grouped.shape[2]
            mean = (grouped * valid).sum(dim=(2, 3), keepdim=True) / count
            variance = ((grouped - mean).square() * valid).sum(dim=(2, 3), keepdim=True) / count
            output = ((grouped - mean) * torch.rsqrt(variance + self.eps)).view_as(input)
            if self.affine:
                output = output * self.weight.float()[None, :, None]
                output = output + self.bias.float()[None, :, None]
            return output.type_as(input)

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
