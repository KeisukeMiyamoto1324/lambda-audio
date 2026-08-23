import torch
from torch import nn
from torch.nn import functional as F


class AudioAdapter(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.conv = nn.Conv1d(
            input_size,
            input_size,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.activation = nn.GELU()
        self.projection = nn.Linear(input_size, hidden_size)
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_mask: torch.BoolTensor,
    ) -> tuple[torch.Tensor, torch.BoolTensor]:
        # ---------------------------------------------------------
        # Compress the audio sequence and change its hidden size.
        # ---------------------------------------------------------
        hidden_states = self.input_norm(hidden_states)
        hidden_states = self.conv(hidden_states.transpose(1, 2)).transpose(1, 2)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.projection(hidden_states)
        hidden_states = self.output_norm(hidden_states)

        # ---------------------------------------------------------
        # Keep positions that contain at least one valid input.
        # ---------------------------------------------------------
        valid_mask = (~padding_mask).to(hidden_states.dtype).unsqueeze(1)
        valid_mask = F.max_pool1d(valid_mask, kernel_size=3, stride=2, padding=1)
        output_padding_mask = valid_mask.squeeze(1) == 0

        return hidden_states, output_padding_mask
