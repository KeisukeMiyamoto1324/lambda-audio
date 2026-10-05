from pathlib import Path

from pydantic import FilePath, PositiveFloat, PositiveInt

from src.modeling.config import ModelConfig


class CerConfig(ModelConfig):
    checkpoint_path: FilePath
    cache_dir: Path = Path("data/cache")
    output_path: Path = Path("outputs/cer/results.csv")
    max_new_tokens: PositiveInt = 256
    beam_search: PositiveInt = 5
    max_audio_seconds: PositiveFloat = 30.0
