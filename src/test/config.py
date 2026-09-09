from pathlib import Path

from pydantic import Field, FilePath, PositiveFloat, PositiveInt

from src.modeling.config import ModelConfig


class EvaluationConfig(ModelConfig):
    checkpoint_path: FilePath
    dataset_dir: Path = Path("outputs/datasets/lambda-audio-corpus-1/test")
    output_path: Path = Path("outputs/test/results.jsonl")
    max_new_tokens: PositiveInt = 150
    max_audio_seconds: PositiveFloat = 30.0
    batch_size: PositiveInt = 32
    dataset_ratio: float = Field(default=0.1, gt=0.0, le=1.0)
