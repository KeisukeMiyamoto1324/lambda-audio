import mlx.core as mx
import mlx.nn as nn

from mlx.kernel.normalization import masked_group_norm
from mlx.model.zipformer_layers import Bypass, Downsample, EncoderLayer, positional_encoding
from src.modeling.zipformer.configuration_zipformer import ZipformerConfig


class FeatureConv(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, kernel: int, stride: int, bias: bool, norm: str | None) -> None:
        super().__init__()
        self.conv = nn.Conv1d(input_dim, output_dim, kernel, stride=stride, bias=bias)
        self.norm_type = norm
        self.norm = nn.LayerNorm(output_dim) if norm is not None else None

    def __call__(self, x: mx.array, lengths: mx.array) -> tuple[mx.array, mx.array]:
        x = self.conv(x)
        lengths = (lengths - self.conv.weight.shape[1]) // self.conv.stride + 1
        mask = mx.arange(x.shape[1])[None, :] >= lengths[:, None]
        if self.norm_type == "group_norm":
            x = masked_group_norm(x, lengths, self.norm.weight, self.norm.bias)
        if self.norm_type == "layer_norm":
            x = self.norm(x)
        return mx.where(mask[..., None], 0, nn.gelu(x)), lengths


class FeatureExtractor(nn.Module):
    def __init__(self, config: ZipformerConfig) -> None:
        super().__init__()
        if config.extractor_mode not in {"default", "layer_norm"}:
            raise ValueError(f"Unsupported extractor mode: {config.extractor_mode}")
        self.conv_layers = []
        input_dim = 1
        for index, (dimension, kernel, stride) in enumerate(config.conv_feature_layers):
            norm = "layer_norm" if config.extractor_mode == "layer_norm" else "group_norm" if index == 0 else None
            self.conv_layers.append(FeatureConv(input_dim, dimension, kernel, stride, config.conv_bias, norm))
            input_dim = dimension

    def __call__(self, x: mx.array, lengths: mx.array) -> tuple[mx.array, mx.array]:
        x = x[..., None]
        for layer in self.conv_layers:
            x, lengths = layer(x, lengths)
        return x, lengths


class EncoderStack(nn.Module):
    def __init__(self, config: ZipformerConfig, index: int) -> None:
        super().__init__()
        self.pos_dim = config.pos_dim
        self.layers = [
            EncoderLayer(
                config.encoder_dim[index],
                config.pos_dim,
                config.num_heads[index],
                config.query_head_dim,
                config.pos_head_dim,
                config.value_head_dim,
                config.feedforward_dim[index],
                config.cnn_module_kernel[index],
            )
            for _ in range(config.num_encoder_layers[index])
        ]

    def __call__(self, x: mx.array, padding_mask: mx.array) -> mx.array:
        positions = positional_encoding(x.shape[1], self.pos_dim, x.dtype)
        for layer in self.layers:
            x = layer(x, positions, padding_mask)
        return x


class DownsampledEncoder(nn.Module):
    def __init__(self, config: ZipformerConfig, index: int) -> None:
        super().__init__()
        self.factor = config.downsampling_factor[index]
        self.downsample = Downsample(self.factor)
        self.encoder = EncoderStack(config, index)
        self.out_combiner = Bypass(config.encoder_dim[index])

    def __call__(self, x: mx.array, padding_mask: mx.array) -> mx.array:
        output = self.encoder(self.downsample(x, padding_mask), padding_mask[:, :: self.factor])
        output = mx.repeat(output, self.factor, axis=1)[:, : x.shape[1]]
        return self.out_combiner(x, output)


class Zipformer(nn.Module):
    def __init__(self, config: ZipformerConfig) -> None:
        super().__init__()
        self.dimensions = tuple(config.encoder_dim)
        self.encoders = [EncoderStack(config, index) if factor == 1 else DownsampledEncoder(config, index) for index, factor in enumerate(config.downsampling_factor)]
        self.downsample_output = Downsample(1)

    def __call__(self, x: mx.array, padding_mask: mx.array) -> mx.array:
        outputs = []
        for dimension, encoder in zip(self.dimensions, self.encoders, strict=True):
            x = x[..., :dimension] if dimension <= x.shape[-1] else mx.pad(x, [(0, 0), (0, 0), (0, dimension - x.shape[-1])])
            x = encoder(x, padding_mask)
            outputs.append(x)
        # ---------------------------------------------------------
        # Keep each channel from the last stack that contains it.
        # ---------------------------------------------------------
        pieces = [outputs[-1]]
        dimension = self.dimensions[-1]
        for output in reversed(outputs[:-1]):
            if output.shape[-1] <= dimension:
                continue
            pieces.append(output[..., dimension:])
            dimension = output.shape[-1]
        return mx.concatenate(pieces, axis=-1)


class ZipformerModel(nn.Module):
    def __init__(self, config: ZipformerConfig) -> None:
        super().__init__()
        self.feature_extractor = FeatureExtractor(config)
        feature_dim = config.conv_feature_layers[-1][0]
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.post_extract_proj = nn.Linear(feature_dim, config.encoder_dim[0]) if feature_dim != config.encoder_dim[0] else None
        self.mask_emb = mx.zeros((config.encoder_dim[0],))
        self.encoder = Zipformer(config)
        self.receptive_field = 1
        stride = 1
        for _, kernel, step in config.conv_feature_layers:
            self.receptive_field += (kernel - 1) * stride
            stride *= step

    def __call__(self, input_values: mx.array, padding_mask: mx.array) -> tuple[mx.array, mx.array]:
        lengths = mx.sum(~padding_mask, axis=-1)
        if bool(mx.any(lengths < self.receptive_field).item()):
            raise ValueError(f"Audio must contain at least {self.receptive_field} samples.")
        x, lengths = self.feature_extractor(input_values, lengths)
        x = self.layer_norm(x)
        padding_mask = mx.arange(x.shape[1])[None, :] >= lengths[:, None]
        if self.post_extract_proj is not None:
            x = self.post_extract_proj(x)
        return self.encoder(x, padding_mask), padding_mask
