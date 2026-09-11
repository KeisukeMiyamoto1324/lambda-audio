# Copyright (c) Facebook, Inc. and its affiliates.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn

from .utils import Fp32GroupNorm, Fp32LayerNorm, TransposeLast


class ConvFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        conv_layers: list[tuple[int, int, int]] | tuple[tuple[int, int, int], ...],
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False,
    ) -> None:
        super().__init__()

        if mode not in {"default", "layer_norm"}:
            raise ValueError(f"Unsupported extractor mode: {mode}")

        def block(
            n_in: int,
            n_out: int,
            k: int,
            stride: int,
            is_layer_norm: bool = False,
            is_group_norm: bool = False,
            conv_bias: bool = False,
        ) -> nn.Sequential:
            def make_conv() -> nn.Conv1d:
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            if is_layer_norm and is_group_norm:
                raise ValueError("Layer norm and group norm are exclusive.")

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            if is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.BoolTensor | None = None,
    ) -> torch.Tensor:
        # BxT -> BxCxT
        x = x.unsqueeze(1)
        lengths = None if padding_mask is None else (~padding_mask).sum(dim=-1)
        feature_padding_mask = None

        for index, conv in enumerate(self.conv_layers):
            convolution = conv[0]
            x = convolution(x)
            if lengths is not None:
                kernel_size = convolution.kernel_size[0]
                stride = convolution.stride[0]
                lengths = torch.div(lengths - kernel_size, stride, rounding_mode="floor") + 1
                positions = torch.arange(x.shape[-1], device=x.device)
                feature_padding_mask = positions.unsqueeze(0) >= lengths.unsqueeze(1)

            if index == 0 and len(conv) == 4 and isinstance(conv[2], Fp32GroupNorm) and feature_padding_mask is not None:
                x = conv[1](x)
                x = conv[2](x, feature_padding_mask)
                x = conv[3](x)
            else:
                x = conv[1:](x)

            if feature_padding_mask is not None:
                x = x.masked_fill(feature_padding_mask.unsqueeze(1), 0.0)

        return x
