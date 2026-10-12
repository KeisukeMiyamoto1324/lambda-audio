from functools import partial

import torch

from src.benchmark.config import BenchmarkConfig
from src.benchmark.models.base import BenchmarkModel


def load_model(config: BenchmarkConfig, device: torch.device, token: str) -> BenchmarkModel:
    # ---------------------------------------------------------
    # Import MLX only when selected, preserving other benchmark backends.
    # The shared loader signature is kept; MLX uses its Metal device.
    # ---------------------------------------------------------
    from mlx.inference import load_model_components, transcribe

    if config.checkpoint_path is None:
        raise ValueError("MLX LambdaAudio requires a checkpoint.")
    components = load_model_components(config, config.checkpoint_path, token)
    return partial(transcribe, components.model, components.feature_extractor, components.tokenizer, config)
