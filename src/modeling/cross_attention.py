import torch
from torch import nn
from torch.nn import functional as F

AudioKeyValue = tuple[torch.Tensor, torch.Tensor]


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

    def project_key_value(self, audio_features: torch.Tensor) -> AudioKeyValue:
        # ---------------------------------------------------------
        # Project audio once and retain the checkpoint-compatible MHA weights.
        # ---------------------------------------------------------
        key_value_weight = self.attention.in_proj_weight[self.attention.embed_dim :]
        batch_size, audio_length, _ = audio_features.shape
        projection_shape = (
            2,
            batch_size,
            self.attention.num_heads,
            audio_length,
            self.attention.head_dim,
        )
        key_value = F.linear(audio_features, key_value_weight)
        key_value = key_value.view(batch_size, audio_length, 2, self.attention.num_heads, self.attention.head_dim)
        key_value = key_value.permute(2, 0, 3, 1, 4).reshape(projection_shape)
        key, value = key_value.unbind(dim=0)
        return key, value

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_key_value: AudioKeyValue,
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
            audio_key_value[0],
            audio_key_value[1],
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
