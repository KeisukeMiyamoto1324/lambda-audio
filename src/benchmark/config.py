from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, FilePath, PositiveInt


class WhisperBenchmarkConfig(BaseModel):
    model_config = ConfigDict(frozen=True, validate_default=True, extra="forbid")

    input_path: FilePath = Path("data/data.parquet")
    batch_size: PositiveInt = 8
    max_new_tokens: PositiveInt = 256
    sampling_rate: int = Field(default=16_000, frozen=True)
    max_audio_seconds: float = Field(default=30.0, gt=0, le=30.0)
