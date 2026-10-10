from pathlib import Path

from pydantic import FilePath, PositiveFloat, PositiveInt

from src.modeling.config import ModelConfig


class InferenceConfig(ModelConfig):
    checkpoint_path: FilePath
    input_path: FilePath = Path("data/data.parquet")
    max_new_tokens: PositiveInt = 256
    max_audio_seconds: PositiveFloat = 30.0
    batch_size: PositiveInt = 8
