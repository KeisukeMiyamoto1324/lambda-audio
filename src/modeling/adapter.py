import torch
from torch import nn


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
        hidden_states = hidden_states.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        hidden_states = self.conv(hidden_states.transpose(1, 2)).transpose(1, 2)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.projection(hidden_states)
        hidden_states = self.output_norm(hidden_states)

        # ---------------------------------------------------------
        # Derive exact convolution output lengths before batch padding.
        # ---------------------------------------------------------
        input_lengths = (~padding_mask).sum(dim=-1)
        output_lengths = (input_lengths + 1) // 2
        positions = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        output_padding_mask = positions.unsqueeze(0) >= output_lengths.unsqueeze(1)

        return hidden_states, output_padding_mask
