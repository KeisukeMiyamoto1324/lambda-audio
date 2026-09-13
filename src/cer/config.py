from pathlib import Path

from pydantic import Field, FilePath, PositiveFloat, PositiveInt

from src.modeling.config import ModelConfig


class CerConfig(ModelConfig):
    checkpoint_path: FilePath
    dataset_percentage: int = Field(default=10, ge=1, le=100)
    cache_dir: Path = Path("data/cache")
    max_new_tokens: PositiveInt = 256
    max_audio_seconds: PositiveFloat = 30.0
