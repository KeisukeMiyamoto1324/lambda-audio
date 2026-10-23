from functools import partial

import torch

from src.benchmark.config import BenchmarkConfig
from src.benchmark.models.base import BenchmarkModel
from src.modeling.checkpoint import load_model_checkpoint
from src.modeling.loading import load_model_components
from src.shared.transcription import transcribe


def load_model(config: BenchmarkConfig, device: torch.device, token: str) -> BenchmarkModel:
    if config.checkpoint_path is None:
        raise ValueError("LambdaAudio requires a checkpoint.")
    components = load_model_components(config, token)
    load_model_checkpoint(components.model, config.checkpoint_path)
    components.model.to(device).eval()
    return partial(transcribe, components.model, components.feature_extractor, components.tokenizer, config, device, output_language=config.output_language)
