from typing import Any

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from src.adapter import AudioAdapter
from src.cross_attention import GatedCrossAttention
from src.lfm2_decoder import AudioLfm2ForCausalLM, add_cross_attention


class LambdaAudioModel(nn.Module):
    def __init__(
        self,
        encoder: PreTrainedModel,
        decoder: AudioLfm2ForCausalLM,
        encoder_hidden_size: int = 768,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.adapter = AudioAdapter(encoder_hidden_size, decoder.config.hidden_size)
        add_cross_attention(self.decoder)

    def encode(
        self,
        input_values: torch.Tensor,
        audio_padding_mask: torch.BoolTensor,
    ) -> tuple[torch.Tensor, torch.BoolTensor]:
        # ---------------------------------------------------------
        # Encode audio and create the mask for encoder output frames.
        # ---------------------------------------------------------
        encoder_output = self.encoder(
            input_values=input_values,
            padding_mask=audio_padding_mask,
        )
        encoder_padding_mask = self.encoder.forward_padding_mask(
            encoder_output.last_hidden_state,
            audio_padding_mask,
        )

        return self.adapter(
            encoder_output.last_hidden_state,
            encoder_padding_mask,
        )

    def forward(
        self,
        input_values: torch.Tensor,
        audio_padding_mask: torch.BoolTensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        labels: torch.LongTensor,
    ) -> CausalLMOutputWithPast:
        audio_features, encoded_padding_mask = self.encode(
            input_values,
            audio_padding_mask,
        )

        return self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            audio_features=audio_features,
            audio_padding_mask=encoded_padding_mask,
        )

    def generate(
        self,
        input_values: torch.Tensor,
        audio_padding_mask: torch.BoolTensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        **kwargs: Any,
    ) -> torch.LongTensor:
        audio_features, encoded_padding_mask = self.encode(
            input_values,
            audio_padding_mask,
        )

        return self.decoder.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_features=audio_features,
            audio_padding_mask=encoded_padding_mask,
            **kwargs,
        )

    def set_backbones_trainable(self, trainable: bool) -> None:
        # ---------------------------------------------------------
        # Freeze only pretrained weights during the first stage.
        # ---------------------------------------------------------
        self.encoder.requires_grad_(trainable)
        self.decoder.requires_grad_(trainable)
        self.adapter.requires_grad_(True)

        if trainable:
            return

        for module in self.decoder.modules():
            if isinstance(module, GatedCrossAttention):
                module.requires_grad_(True)
