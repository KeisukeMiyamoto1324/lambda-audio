from collections.abc import Callable
from functools import partial

import torch

from src.benchmark.config import BenchmarkConfig
from src.benchmark.models.base import BenchmarkModel
from src.benchmark.models.cohere import load_model as load_cohere
from src.benchmark.models.lambda_audio import load_model as load_lambda_audio
from src.benchmark.models.parakeet import load_model as load_parakeet
from src.benchmark.models.reazonspeech import load_model as load_reazonspeech
from src.benchmark.models.whisper import load_model as load_whisper

MODEL_LOADERS: dict[str, Callable[[BenchmarkConfig, torch.device, str], BenchmarkModel]] = {
    "lambda-audio": load_lambda_audio,
    "whisper": load_whisper,
    "kotoba-whisper": partial(load_whisper, model_id="kotoba-tech/kotoba-whisper-v2.2"),
    "reazonspeech": load_reazonspeech,
    "parakeet": load_parakeet,
    "cohere-transcribe": load_cohere,
}
