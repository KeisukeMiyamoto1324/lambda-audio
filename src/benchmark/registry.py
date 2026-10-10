from collections.abc import Callable

import torch

from src.benchmark.config import BenchmarkConfig
from src.benchmark.models.base import BenchmarkModel
from src.benchmark.models.lambda_audio import load_model as load_lambda_audio
from src.benchmark.models.whisper import load_model as load_whisper

MODEL_LOADERS: dict[str, Callable[[BenchmarkConfig, torch.device, str], BenchmarkModel]] = {
    "lambda-audio": load_lambda_audio,
    "whisper": load_whisper,
}
