import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
from huggingface_hub import hf_hub_download

from mlx.config import Lfm2Config, ModelConfig, ZipformerConfig
from mlx.inference.checkpoint import load_checkpoint
from mlx.inference.preprocessing import FeatureExtractor, Tokenizer
from mlx.model import LambdaAudioModel


@dataclass(frozen=True)
class ModelComponents:
    model: LambdaAudioModel
    feature_extractor: FeatureExtractor
    tokenizer: Tokenizer


def load_model_components(config: ModelConfig, checkpoint_path: Path, token: str) -> ModelComponents:
    if checkpoint_path.suffix != ".safetensors":
        raise ValueError("MLX LambdaAudio requires a .safetensors checkpoint.")
    if not mx.metal.is_available():
        raise RuntimeError("MLX LambdaAudio requires an Apple Silicon Mac with Metal.")
    # ---------------------------------------------------------
    # Release reusable Metal buffers between calls.
    # Download only configurations and the tokenizer with HF_TOKEN.
    # ---------------------------------------------------------
    mx.set_cache_limit(0)
    mx.clear_cache()
    encoder_path = hf_hub_download(config.encoder_model_id, "config.json", token=token)
    decoder_path = hf_hub_download(config.decoder_model_id, "config.json", token=token)
    feature_path = hf_hub_download(config.encoder_model_id, "preprocessor_config.json", token=token)
    tokenizer_path = hf_hub_download(config.decoder_model_id, "tokenizer.json", token=token)
    tokenizer_config_path = hf_hub_download(config.decoder_model_id, "tokenizer_config.json", token=token)
    encoder_config = ZipformerConfig.model_validate_json(Path(encoder_path).read_text())
    decoder_config = Lfm2Config.model_validate_json(Path(decoder_path).read_text())
    feature_extractor = FeatureExtractor(json.loads(Path(feature_path).read_text()))
    if feature_extractor.sampling_rate != config.sampling_rate:
        raise ValueError("The feature extractor sampling rate does not match the inference configuration.")
    tokenizer = Tokenizer(tokenizer_path, json.loads(Path(tokenizer_config_path).read_text()))
    model = LambdaAudioModel(encoder_config, decoder_config)
    load_checkpoint(model, checkpoint_path)
    model.eval()
    mx.eval(model.parameters())
    return ModelComponents(model, feature_extractor, tokenizer)
