import re

import mlx.core as mx
import mlx.nn as nn
from transformers.models.lfm2.configuration_lfm2 import Lfm2Config

from mlx.model.lfm2 import AudioLfm2ForCausalLM
from mlx.model.zipformer import ZipformerModel
from src.modeling.zipformer.configuration_zipformer import ZipformerConfig


class AudioAdapter(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.conv = nn.Conv1d(input_size, input_size, 3, stride=2, padding=1)
        self.projection = nn.Linear(input_size, hidden_size)
        self.output_norm = nn.LayerNorm(hidden_size)

    def __call__(self, x: mx.array, padding_mask: mx.array) -> tuple[mx.array, mx.array]:
        x = mx.where(padding_mask[..., None], 0, self.input_norm(x))
        x = self.output_norm(self.projection(nn.gelu(self.conv(x))))
        lengths = (mx.sum(~padding_mask, axis=-1) + 1) // 2
        return x, mx.arange(x.shape[1])[None, :] >= lengths[:, None]


class LambdaAudioModel(nn.Module):
    def __init__(self, encoder_config: ZipformerConfig, decoder_config: Lfm2Config) -> None:
        super().__init__()
        self.encoder = ZipformerModel(encoder_config)
        self.adapter = AudioAdapter(max(encoder_config.encoder_dim), decoder_config.hidden_size)
        self.decoder = AudioLfm2ForCausalLM(decoder_config)

    def encode(self, input_values: mx.array, padding_mask: mx.array) -> tuple[mx.array, mx.array]:
        features, padding_mask = self.encoder(input_values, padding_mask)
        return self.adapter(features, padding_mask)

    @staticmethod
    def sanitize(weights: dict[str, mx.array]) -> dict[str, mx.array]:
        # ---------------------------------------------------------
        # Map PyTorch feature blocks and convolution layouts to MLX.
        # Preserve all weights so strict loading catches incompatible files.
        # ---------------------------------------------------------
        converted = {}
        for name, tensor in weights.items():
            name = name.removeprefix("model.")
            name = re.sub(r"(encoder.feature_extractor.conv_layers.\d+)\.0\.", r"\1.conv.", name)
            name = re.sub(r"(encoder.feature_extractor.conv_layers.\d+)\.2\.(?:1\.)?", r"\1.norm.", name)
            if tensor.ndim == 3 and name.endswith(".weight"):
                tensor = tensor.transpose(0, 2, 1)
            converted[name] = tensor
        return converted
