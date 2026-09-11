from typing import TypedDict

import torch


class CerBatch(TypedDict):
    input_values: torch.Tensor
    audio_padding_mask: torch.BoolTensor
    references: list[str]
