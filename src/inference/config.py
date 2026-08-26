from pathlib import Path

from pydantic import DirectoryPath, FilePath, PositiveFloat, PositiveInt

from src.modeling.config import ModelConfig


class InferenceConfig(ModelConfig):
    checkpoint_path: FilePath
    input_dir: DirectoryPath
    max_new_tokens: PositiveInt = 256
    max_audio_seconds: PositiveFloat = 30.0
