from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, PositiveInt


class ModelConfig(Protocol):
    encoder_model_id: str
    decoder_model_id: str
    sampling_rate: int


class TranscriptionConfig(Protocol):
    sampling_rate: int
    max_audio_seconds: float
    max_new_tokens: int


class InferenceConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    encoder_model_id: str = "KeisukeMiyamoto/zipformer-cpt"
    decoder_model_id: str = "KeisukeMiyamoto/lfm2.5-230m-cpt"
    sampling_rate: PositiveInt = 16000
    max_audio_seconds: float = Field(default=30.0, gt=0, le=30)
    max_new_tokens: PositiveInt = 256


class ZipformerConfig(BaseModel):
    conv_feature_layers: tuple[tuple[int, int, int], ...] = ((512, 10, 5), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 2, 2), (512, 2, 2))
    extractor_mode: str = "default"
    conv_bias: bool = False
    encoder_dim: tuple[int, ...] = (192, 256, 448, 768, 448, 192)
    downsampling_factor: tuple[int, ...] = (1, 2, 4, 8, 4, 2)
    num_encoder_layers: tuple[int, ...] = (2, 2, 3, 4, 3, 2)
    query_head_dim: int = 32
    pos_head_dim: int = 4
    value_head_dim: int = 12
    pos_dim: int = 48
    num_heads: tuple[int, ...] = (4, 4, 4, 8, 4, 4)
    feedforward_dim: tuple[int, ...] = (512, 768, 1024, 1536, 1024, 768)
    cnn_module_kernel: tuple[int, ...] = (31, 31, 15, 15, 15, 31)


class Lfm2Config(BaseModel):
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    layer_types: list[str]
    norm_eps: float = 1e-5
    conv_L_cache: int = 3
    conv_bias: bool = False
    block_auto_adjust_ff_dim: bool = False
    block_ffn_dim_multiplier: float | None = 1.0
    block_multiple_of: int = 256
    rope_parameters: dict[str, str | float] = Field(default_factory=lambda: {"rope_type": "default", "rope_theta": 1000000.0})
