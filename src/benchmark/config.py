from pathlib import Path
from typing import Self

from pydantic import FilePath, PositiveInt, model_validator

from src.modeling.config import ModelConfig


class BenchmarkConfig(ModelConfig):
    model: str = "lambda-audio"
    checkpoint_path: FilePath | None = None
    input_path: FilePath = Path("data/data.parquet")
    batch_size: PositiveInt = 16
    max_new_tokens: PositiveInt = 256
    max_audio_seconds: float = 30.0

    @model_validator(mode="after")
    def validate_checkpoint(self) -> Self:
        if self.model == "lambda-audio" and self.checkpoint_path is None:
            raise ValueError("LambdaAudio requires --checkpoint.")
        if not 0 < self.max_audio_seconds <= 30:
            raise ValueError("Audio duration must be between 0 and 30 seconds.")
        return self
