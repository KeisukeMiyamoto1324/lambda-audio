from pathlib import Path

from pydantic import FilePath, PositiveFloat, PositiveInt

from src.modeling.config import ModelConfig


class EvaluationConfig(ModelConfig):
    checkpoint_path: FilePath
    output_path: Path = Path("outputs/test/results.jsonl")
    max_new_tokens: PositiveInt = 256
    max_audio_seconds: PositiveFloat = 30.0
