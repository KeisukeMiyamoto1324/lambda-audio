from pathlib import Path

from pydantic import PositiveFloat, PositiveInt

from src.training.config import BaseTrainingConfig


class Stage1Config(BaseTrainingConfig):
    output_dir: Path = Path("outputs/stage1")
    num_workers: PositiveInt = 8
    prefetch_factor: PositiveInt = 2
    train_max_batch_padded_audio_seconds: PositiveFloat = 1_400.0
    validation_max_batch_padded_audio_seconds: PositiveFloat = 120.0
    train_max_batch_attention_cost: PositiveInt = 56_000_000
    validation_max_batch_attention_cost: PositiveInt = 12_000_000
    gradient_accumulation_steps: PositiveInt = 2
    max_steps: PositiveInt = 204_800
    learning_rate: PositiveFloat = 1e-4
