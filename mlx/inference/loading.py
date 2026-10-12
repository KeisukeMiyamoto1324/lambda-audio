from dataclasses import dataclass
from pathlib import Path
from typing import cast

import mlx.core as mx
from transformers import AutoFeatureExtractor, AutoTokenizer, PreTrainedTokenizerBase, SequenceFeatureExtractor
from transformers.models.lfm2.configuration_lfm2 import Lfm2Config

from mlx.inference.checkpoint import load_checkpoint
from mlx.model import LambdaAudioModel
from src.modeling.config import ModelConfig
from src.modeling.zipformer.configuration_zipformer import ZipformerConfig


@dataclass(frozen=True)
class ModelComponents:
    model: LambdaAudioModel
    feature_extractor: SequenceFeatureExtractor
    tokenizer: PreTrainedTokenizerBase


def load_model_components(config: ModelConfig, checkpoint_path: Path, token: str) -> ModelComponents:
    if checkpoint_path.suffix != ".safetensors":
        raise ValueError("MLX LambdaAudio requires a .safetensors checkpoint.")
    if not mx.metal.is_available():
        raise RuntimeError("MLX LambdaAudio requires an Apple Silicon Mac with Metal.")
    # ---------------------------------------------------------
    # Fetch configurations and preprocessing only, not backbone weights.
    # Use the explicitly provided HF_TOKEN for every Hub request.
    # ---------------------------------------------------------
    encoder_config = ZipformerConfig.from_pretrained(config.encoder_model_id, token=token)
    decoder_config = Lfm2Config.from_pretrained(config.decoder_model_id, token=token)
    feature_extractor = cast(SequenceFeatureExtractor, AutoFeatureExtractor.from_pretrained(config.encoder_model_id, token=token))
    tokenizer = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(config.decoder_model_id, token=token))
    model = LambdaAudioModel(encoder_config, decoder_config)
    load_checkpoint(model, checkpoint_path)
    model.eval()
    mx.eval(model.parameters())
    return ModelComponents(model, feature_extractor, tokenizer)
