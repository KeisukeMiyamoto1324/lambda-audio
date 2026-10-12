import mlx.core as mx
import mlx.nn as nn
from transformers.models.lfm2.configuration_lfm2 import Lfm2Config

LayerCache = tuple[mx.array, mx.array] | mx.array | None
AudioKeyValues = list[tuple[mx.array, mx.array] | None]


class Attention(nn.Module):
    def __init__(self, config: Lfm2Config) -> None:
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.q_proj = nn.Linear(config.hidden_size, self.heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.heads * self.head_dim, config.hidden_size, bias=False)
        self.q_layernorm = nn.RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_layernorm = nn.RMSNorm(self.head_dim, eps=config.norm_eps)
        if config.rope_parameters["rope_type"] != "default":
            raise ValueError("MLX LambdaAudio requires default RoPE.")
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=config.rope_parameters["rope_theta"])

    def __call__(self, x: mx.array, cache: tuple[mx.array, mx.array] | None) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        batch, length, _ = x.shape
        offset = 0 if cache is None else cache[0].shape[2]
        query = self.q_layernorm(self.q_proj(x).reshape(batch, length, self.heads, self.head_dim)).transpose(0, 2, 1, 3)
        key = self.k_layernorm(self.k_proj(x).reshape(batch, length, self.kv_heads, self.head_dim)).transpose(0, 2, 1, 3)
        value = self.v_proj(x).reshape(batch, length, self.kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        query, key = self.rope(query, offset=offset), self.rope(key, offset=offset)
        if cache is not None:
            key = mx.concatenate((cache[0], key), axis=2)
            value = mx.concatenate((cache[1], value), axis=2)
        mask = mx.arange(key.shape[2])[None, :] <= (offset + mx.arange(length))[:, None] if length > 1 else None
        output = mx.fast.scaled_dot_product_attention(query, key, value, scale=self.head_dim**-0.5, mask=mask)
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.out_proj(output), (key, value)


class ShortConv(nn.Module):
    def __init__(self, config: Lfm2Config) -> None:
        super().__init__()
        dimension = config.hidden_size
        self.kernel = config.conv_L_cache
        self.in_proj = nn.Linear(dimension, 3 * dimension, bias=config.conv_bias)
        self.conv = nn.Conv1d(dimension, dimension, self.kernel, groups=dimension, bias=config.conv_bias)
        self.out_proj = nn.Linear(dimension, dimension, bias=config.conv_bias)

    def __call__(self, x: mx.array, cache: mx.array | None) -> tuple[mx.array, mx.array]:
        gate, multiplier, values = mx.split(self.in_proj(x), 3, axis=-1)
        values = gate * values
        if cache is None:
            cache = mx.zeros((x.shape[0], self.kernel - 1, x.shape[-1]), dtype=x.dtype)
        values = mx.concatenate((cache, values), axis=1)
        new_cache = values[:, values.shape[1] - self.kernel + 1 :]
        return self.out_proj(multiplier * self.conv(values)), new_cache


class FeedForward(nn.Module):
    def __init__(self, config: Lfm2Config) -> None:
        super().__init__()
        hidden = config.intermediate_size
        if config.block_auto_adjust_ff_dim:
            hidden = int(2 * hidden / 3)
            if config.block_ffn_dim_multiplier is not None:
                hidden = int(config.block_ffn_dim_multiplier * hidden)
                hidden = config.block_multiple_of * ((hidden + config.block_multiple_of - 1) // config.block_multiple_of)
        self.w1 = nn.Linear(config.hidden_size, hidden, bias=False)
        self.w2 = nn.Linear(hidden, config.hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, hidden, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class CrossAttentionProjection(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.in_proj_weight = mx.zeros((3 * dimension, dimension))
        self.out_proj = nn.Linear(dimension, dimension, bias=False)


class GatedCrossAttention(nn.Module):
    def __init__(self, config: Lfm2Config) -> None:
        super().__init__()
        self.heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.heads
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attention = CrossAttentionProjection(config.hidden_size)
        self.gate = mx.array(0.0)

    def project_key_value(self, features: mx.array) -> tuple[mx.array, mx.array]:
        _, key_weight, value_weight = mx.split(self.attention.in_proj_weight, 3, axis=0)
        batch, length, _ = features.shape
        shape = (batch, length, self.heads, self.head_dim)
        key = (features @ key_weight.T).reshape(shape).transpose(0, 2, 1, 3)
        value = (features @ value_weight.T).reshape(shape).transpose(0, 2, 1, 3)
        return key, value

    def __call__(self, x: mx.array, key_values: tuple[mx.array, mx.array], padding_mask: mx.array) -> mx.array:
        query_weight = self.attention.in_proj_weight[: x.shape[-1]]
        query = (self.norm(x) @ query_weight.T).reshape(x.shape[0], x.shape[1], self.heads, self.head_dim).transpose(0, 2, 1, 3)
        output = mx.fast.scaled_dot_product_attention(query, *key_values, scale=self.head_dim**-0.5, mask=~padding_mask[:, None, None, :])
        output = output.transpose(0, 2, 1, 3).reshape(x.shape)
        return x + mx.tanh(self.gate) * self.attention.out_proj(output)


class DecoderLayer(nn.Module):
    def __init__(self, config: Lfm2Config, layer_type: str) -> None:
        super().__init__()
        if layer_type not in {"full_attention", "conv"}:
            raise ValueError(f"Unsupported LFM2 layer: {layer_type}")
        self.self_attn = Attention(config) if layer_type == "full_attention" else None
        self.conv = ShortConv(config) if layer_type == "conv" else None
        self.cross_attention = GatedCrossAttention(config) if layer_type == "full_attention" else None
        self.operator_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.feed_forward = FeedForward(config)

    def __call__(self, x: mx.array, cache: LayerCache, audio_key_values: tuple[mx.array, mx.array] | None, audio_padding_mask: mx.array) -> tuple[mx.array, LayerCache]:
        normalized = self.operator_norm(x)
        if self.self_attn is not None:
            output, cache = self.self_attn(normalized, cache)
        else:
            output, cache = self.conv(normalized, cache)
        x = x + output
        if self.cross_attention is not None:
            x = self.cross_attention(x, audio_key_values, audio_padding_mask)
        return x + self.feed_forward(self.ffn_norm(x)), cache


class Decoder(nn.Module):
    def __init__(self, config: Lfm2Config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [DecoderLayer(config, layer_type) for layer_type in config.layer_types]
        self.embedding_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)

    def __call__(self, input_ids: mx.array, audio_key_values: AudioKeyValues, audio_padding_mask: mx.array, cache: list[LayerCache] | None) -> mx.array:
        # ---------------------------------------------------------
        # Keep decoder caches outside the model, one list per request.
        # ---------------------------------------------------------
        x = self.embed_tokens(input_ids)
        states = [None] * len(self.layers) if cache is None else cache
        for index, layer in enumerate(self.layers):
            x, state = layer(x, states[index], audio_key_values[index], audio_padding_mask)
            states[index] = state
        return self.embedding_norm(x)


class AudioLfm2ForCausalLM(nn.Module):
    def __init__(self, config: Lfm2Config) -> None:
        super().__init__()
        self.model = Decoder(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def project_audio_key_values(self, audio_features: mx.array) -> AudioKeyValues:
        return [layer.cross_attention.project_key_value(audio_features) if layer.cross_attention is not None else None for layer in self.model.layers]

    def __call__(self, input_ids: mx.array, audio_key_values: AudioKeyValues, audio_padding_mask: mx.array, cache: list[LayerCache] | None = None) -> mx.array:
        return self.lm_head(self.model(input_ids, audio_key_values, audio_padding_mask, cache))
