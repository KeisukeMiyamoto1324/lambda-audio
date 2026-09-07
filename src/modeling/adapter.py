import torch
from torch import nn


class AudioAdapter(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.projection = nn.Linear(input_size, hidden_size)
        self.activation = nn.GELU()
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_mask: torch.BoolTensor,
    ) -> tuple[torch.Tensor, torch.BoolTensor]:
        # ---------------------------------------------------------
        # Preserve the audio sequence length and change its hidden size.
        # ---------------------------------------------------------
        hidden_states = self.input_norm(hidden_states)
        hidden_states = self.projection(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.output_norm(hidden_states)

        return hidden_states, padding_mask
