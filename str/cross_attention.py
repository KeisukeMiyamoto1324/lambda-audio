import torch
from torch import nn


class GatedCrossAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, norm_eps: float) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            bias=False,
            batch_first=True,
        )
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_features: torch.Tensor,
        audio_padding_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        # ---------------------------------------------------------
        # Use text states as queries and audio states as keys/values.
        # ---------------------------------------------------------
        attention_output, _ = self.attention(
            query=self.norm(hidden_states),
            key=audio_features,
            value=audio_features,
            key_padding_mask=audio_padding_mask,
            need_weights=False,
        )

        return hidden_states + torch.tanh(self.gate) * attention_output
