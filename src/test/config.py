from pathlib import Path
from typing import Self

from pydantic import Field, FilePath, PositiveFloat, PositiveInt, model_validator

from src.modeling.config import ModelConfig


class EvaluationConfig(ModelConfig):
    checkpoint_path: FilePath
    output_path: Path = Path("outputs/test/results.jsonl")
    max_new_tokens: PositiveInt = 256
    max_audio_seconds: PositiveFloat = 30.0
    batch_size: PositiveInt = 16
    bucket_buffer_size: PositiveInt = 256
    num_decode_threads: PositiveInt = 16
    dataset_ratio: float = Field(default=0.1, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_bucket_buffer_size(self) -> Self:
        if self.bucket_buffer_size < self.batch_size:
            raise ValueError("The bucket buffer size must be at least the batch size.")
        return self
