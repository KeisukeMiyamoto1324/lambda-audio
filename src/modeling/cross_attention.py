import torch
from torch import nn
from torch.nn import functional as F


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

    def project_key_value(self, audio_features: torch.Tensor) -> torch.Tensor:
        # ---------------------------------------------------------
        # Project audio once and retain the checkpoint-compatible MHA weights.
        # ---------------------------------------------------------
        _, key_weight, value_weight = self.attention.in_proj_weight.chunk(3)
        batch_size, audio_length, _ = audio_features.shape
        projection_shape = (
            batch_size,
            audio_length,
            self.attention.num_heads,
            self.attention.head_dim,
        )
        key = F.linear(audio_features, key_weight).view(projection_shape)
        value = F.linear(audio_features, value_weight).view(projection_shape)
        return torch.stack(
            (key.transpose(1, 2), value.transpose(1, 2)),
            dim=1,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_key_values: torch.Tensor,
        audio_padding_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        # ---------------------------------------------------------
        # Attend from text queries to the precomputed audio keys and values.
        # ---------------------------------------------------------
        query_weight, _, _ = self.attention.in_proj_weight.chunk(3)
        batch_size, text_length, hidden_size = hidden_states.shape
        query = F.linear(self.norm(hidden_states), query_weight)
        query = query.view(
            batch_size,
            text_length,
            self.attention.num_heads,
            self.attention.head_dim,
        ).transpose(1, 2)
        attention_output = F.scaled_dot_product_attention(
            query,
            audio_key_values[:, 0],
            audio_key_values[:, 1],
            attn_mask=~audio_padding_mask[:, None, None, :],
            dropout_p=0.0,
        )
        attention_output = attention_output.transpose(1, 2).reshape(
            batch_size,
            text_length,
            hidden_size,
        )
        attention_output = self.attention.out_proj(attention_output)

        return hidden_states + torch.tanh(self.gate) * attention_output
