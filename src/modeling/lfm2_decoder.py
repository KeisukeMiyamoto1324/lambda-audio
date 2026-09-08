from typing import Any

import torch
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.lfm2.configuration_lfm2 import Lfm2Config
from transformers.models.lfm2.modeling_lfm2 import Lfm2DecoderLayer, Lfm2ForCausalLM

from src.modeling.cross_attention import GatedCrossAttention


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
        audio_key_values: tuple[torch.Tensor, ...],
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
            audio_key_values[self.audio_cache_index],
            audio_padding_mask,
        )
        hidden_states = hidden_states + self.feed_forward(self.ffn_norm(hidden_states))

        return hidden_states


class AudioLfm2ForCausalLM(Lfm2ForCausalLM):
    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: torch.LongTensor | None = None,
        **model_kwargs: Any,
    ) -> tuple[torch.LongTensor | None, dict[str, Any]]:
        input_ids, model_kwargs = Lfm2ForCausalLM._expand_inputs_for_generation(
            expand_size=expand_size,
            is_encoder_decoder=is_encoder_decoder,
            input_ids=input_ids,
            **model_kwargs,
        )
        # ---------------------------------------------------------
        # Expand tuple-held audio caches in the same order as text inputs.
        # ---------------------------------------------------------
        if expand_size != 1:
            model_kwargs["audio_key_values"] = tuple(cache.repeat_interleave(expand_size, dim=0) for cache in model_kwargs["audio_key_values"])
        return input_ids, model_kwargs

    def project_audio_key_values(
        self,
        audio_features: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        # ---------------------------------------------------------
        # Build one reusable audio K/V cache for every cross-attention layer.
        # ---------------------------------------------------------
        return tuple(layer.cross_attention.project_key_value(audio_features) for layer in self.model.layers if isinstance(layer, AudioLfm2DecoderLayer))

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
        audio_key_values: tuple[torch.Tensor, ...],
        audio_padding_mask: torch.BoolTensor,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            audio_key_values=audio_key_values,
            audio_padding_mask=audio_padding_mask,
            **kwargs,
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
