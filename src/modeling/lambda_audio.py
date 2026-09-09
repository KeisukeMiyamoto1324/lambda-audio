from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers import Wav2Vec2BertModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from src.modeling.adapter import AudioAdapter
from src.modeling.cross_attention import GatedCrossAttention
from src.modeling.lfm2_decoder import AudioLfm2ForCausalLM, add_cross_attention


class LambdaAudioModel(nn.Module):
    def __init__(
        self,
        encoder: Wav2Vec2BertModel,
        decoder: AudioLfm2ForCausalLM,
        ctc_vocab_size: int,
        ctc_blank_token_id: int,
        ctc_loss_weight: float,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.ctc_blank_token_id = ctc_blank_token_id
        self.ctc_loss_weight = ctc_loss_weight
        self.adapter = AudioAdapter(encoder.config.hidden_size, decoder.config.hidden_size)
        self.ctc_output = nn.Linear(decoder.config.hidden_size, ctc_vocab_size)
        add_cross_attention(self.decoder)

    def encode(
        self,
        input_features: torch.Tensor,
        audio_padding_mask: torch.BoolTensor,
    ) -> tuple[torch.Tensor, torch.BoolTensor]:
        # ---------------------------------------------------------
        # Encode Fbank features using their frame-level padding mask.
        # ---------------------------------------------------------
        encoder_output = self.encoder(
            input_features=input_features,
            attention_mask=~audio_padding_mask,
        )

        return self.adapter(
            encoder_output.last_hidden_state,
            audio_padding_mask,
        )

    def forward(
        self,
        input_features: torch.Tensor,
        audio_padding_mask: torch.BoolTensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        labels: torch.LongTensor,
        ctc_labels: torch.LongTensor,
        ctc_label_lengths: torch.LongTensor,
    ) -> CausalLMOutputWithPast:
        audio_features, encoded_padding_mask = self.encode(
            input_features,
            audio_padding_mask,
        )
        audio_key_values = self.decoder.project_audio_key_values(audio_features)

        decoder_output = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            audio_key_values=audio_key_values,
            audio_padding_mask=encoded_padding_mask,
        )
        if decoder_output.loss is None:
            raise RuntimeError("The decoder did not return a training loss.")

        # ---------------------------------------------------------
        # Add monotonic alignment supervision to the decoder loss.
        # ---------------------------------------------------------
        ctc_logits = self.ctc_output(audio_features)
        ctc_input_lengths = (~encoded_padding_mask).sum(dim=-1)
        label_positions = torch.arange(ctc_labels.shape[1] - 1, device=ctc_labels.device)
        repeated_labels = ctc_labels[:, 1:].eq(ctc_labels[:, :-1])
        repeated_labels.logical_and_(label_positions.unsqueeze(0) < (ctc_label_lengths - 1).unsqueeze(1))
        required_ctc_frames = ctc_label_lengths + repeated_labels.sum(dim=-1)
        valid_ctc_examples = ctc_input_lengths >= required_ctc_frames
        if valid_ctc_examples.any():
            ctc_loss = F.ctc_loss(
                ctc_logits[valid_ctc_examples].log_softmax(dim=-1, dtype=torch.float32).transpose(0, 1),
                ctc_labels[valid_ctc_examples],
                ctc_input_lengths[valid_ctc_examples],
                ctc_label_lengths[valid_ctc_examples],
                blank=self.ctc_blank_token_id,
            )
        else:
            ctc_loss = ctc_logits.sum() * 0.0
        decoder_output.loss = decoder_output.loss + self.ctc_loss_weight * ctc_loss
        return decoder_output

    def cast_frozen_weights(self, dtype: torch.dtype) -> None:
        # ---------------------------------------------------------
        # Store large frozen weights in the autocast dtype.
        # Keep trainable layers and normalization parameters in FP32.
        # ---------------------------------------------------------
        for module in self.modules():
            if not isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
                continue
            parameters = list(module.parameters(recurse=False))
            if any(parameter.requires_grad for parameter in parameters):
                continue
            module.to(dtype=dtype)

    def generate(
        self,
        input_features: torch.Tensor,
        audio_padding_mask: torch.BoolTensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        **kwargs: Any,
    ) -> torch.LongTensor:
        audio_features, encoded_padding_mask = self.encode(
            input_features,
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

    def set_encoder_trainable_layers(self, first_layer_index: int) -> None:
        # ---------------------------------------------------------
        # Freeze the frontend and train only the selected later layers.
        # ---------------------------------------------------------
        layers = self.encoder.encoder.layers
        if not 0 <= first_layer_index < len(layers):
            raise ValueError(f"Invalid first trainable layer index: {first_layer_index}")

        self.encoder.requires_grad_(False)
        for layer in layers[first_layer_index:]:
            layer.requires_grad_(True)

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
