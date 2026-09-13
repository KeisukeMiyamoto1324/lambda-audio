from pathlib import Path

from pydantic import FilePath, PositiveFloat, PositiveInt

from src.modeling.config import ModelConfig


class CerConfig(ModelConfig):
    checkpoint_path: FilePath
    cache_dir: Path = Path("data/cache")
    max_new_tokens: PositiveInt = 256
    max_audio_seconds: PositiveFloat = 30.0
