from typing import Any

import torch
from cut_cross_entropy import linear_cross_entropy
from kernels import LayerRepository
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.lfm2 import modeling_lfm2
from transformers.models.lfm2.configuration_lfm2 import Lfm2Config
from transformers.models.lfm2.modeling_lfm2 import Lfm2DecoderLayer, Lfm2ForCausalLM

from src.modeling.cross_attention import GatedCrossAttention


def enable_causal_convolution_kernels() -> None:
    # ---------------------------------------------------------
    # Replace LFM reference convolutions with CUDA 13 Hub kernels.
    # ---------------------------------------------------------
    for kernel_name in ("causal_conv1d_fn", "causal_conv1d_update"):
        kernel_class = LayerRepository(
            repo_id="kernels-community/mamba-ssm",
            layer_name=kernel_name,
            version=3,
        ).load()
        setattr(modeling_lfm2, kernel_name, kernel_class())


class AudioLfm2DecoderLayer(Lfm2DecoderLayer):
    def __init__(
        self,
        config: Lfm2Config,
        layer_idx: int,
        audio_cache_index: int,
    ) -> None:
        super().__init__(config, layer_idx)
        self.audio_cache_index = audio_cache_index
        self.cross_attention = GatedCrossAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            norm_eps=config.norm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        *,
        audio_key_values: torch.Tensor,
        audio_padding_mask: torch.BoolTensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        # ---------------------------------------------------------
        # Keep the original LFM operator and residual connection.
        # Based on Transformers 5.8.1 Lfm2DecoderLayer.
        # ---------------------------------------------------------
        residual = hidden_states
        hidden_states, _ = self.self_attn(
            hidden_states=self.operator_norm(hidden_states),
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = hidden_states + residual

        # ---------------------------------------------------------
        # Add audio information before the original feed-forward block.
        # ---------------------------------------------------------
        hidden_states = self.cross_attention(
            hidden_states,
            audio_key_values[:, self.audio_cache_index],
            audio_padding_mask,
        )
        hidden_states = hidden_states + self.feed_forward(self.ffn_norm(hidden_states))

        return hidden_states


class AudioLfm2ForCausalLM(Lfm2ForCausalLM):
    def project_audio_key_values(
        self,
        audio_features: torch.Tensor,
    ) -> torch.Tensor:
        # ---------------------------------------------------------
        # Build one reusable audio K/V cache for every cross-attention layer.
        # ---------------------------------------------------------
        key_values = [layer.cross_attention.project_key_value(audio_features) for layer in self.model.layers if isinstance(layer, AudioLfm2DecoderLayer)]
        return torch.stack(key_values, dim=1)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        *,
        audio_key_values: torch.Tensor,
        audio_padding_mask: torch.BoolTensor,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            audio_key_values=audio_key_values,
            audio_padding_mask=audio_padding_mask,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state

        # ---------------------------------------------------------
        # Fuse the classifier and training loss without materializing logits.
        # ---------------------------------------------------------
        if labels is not None:
            loss = linear_cross_entropy(
                hidden_states.to(torch.bfloat16),
                self.lm_head.weight,
                labels,
                shift=True,
                filter_eps="high",
                impl="cce",
            )
            logits = None
        else:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])
            loss = None

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def add_cross_attention(model: Lfm2ForCausalLM) -> None:
    # ---------------------------------------------------------
    # Replace only full-attention layers and preserve LFM weights.
    # ---------------------------------------------------------
    audio_cache_index = 0
    for layer_idx, layer_type in enumerate(model.config.layer_types):
        if layer_type != "full_attention":
            continue

        original_layer = model.model.layers[layer_idx]
        new_layer = AudioLfm2DecoderLayer(
            model.config,
            layer_idx,
            audio_cache_index,
        )
        new_layer.load_state_dict(original_layer.state_dict(), strict=False)

        parameter = next(original_layer.parameters())
        new_layer.to(device=parameter.device, dtype=parameter.dtype)
        model.model.layers[layer_idx] = new_layer
        audio_cache_index += 1
