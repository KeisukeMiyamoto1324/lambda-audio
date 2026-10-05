from typing import Any, Self

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from src.modeling.adapter import AudioAdapter
from src.modeling.cross_attention import GatedCrossAttention
from src.modeling.lfm2_decoder import AudioLfm2ForCausalLM, add_cross_attention


class LambdaAudioModel(nn.Module):
    def __init__(
        self,
        encoder: PreTrainedModel,
        decoder: AudioLfm2ForCausalLM,
        encoder_hidden_size: int = 768,
    ) -> None:
        super().__init__()
        self.backbones_trainable = True
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
        audio_key_values = self.decoder.project_audio_key_values(audio_features)

        return self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            audio_key_values=audio_key_values,
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
        audio_key_values = self.decoder.project_audio_key_values(audio_features)

        return self.decoder.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_key_values=audio_key_values,
            audio_padding_mask=encoded_padding_mask,
            **kwargs,
        )

    def set_backbones_trainable(self, trainable: bool) -> None:
        # ---------------------------------------------------------
        # Keep audio conditioning trainable while freezing pretrained weights.
        # ---------------------------------------------------------
        self.backbones_trainable = trainable
        self.requires_grad_(trainable)
        self.adapter.requires_grad_(True)
        for module in self.decoder.modules():
            if isinstance(module, GatedCrossAttention):
                module.requires_grad_(True)
        self.train(self.training)

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        if mode and not self.backbones_trainable:
            self.encoder.eval()
            self.decoder.eval()
            for module in self.decoder.modules():
                if isinstance(module, GatedCrossAttention):
                    module.train()
        return self
