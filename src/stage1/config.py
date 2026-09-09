from pathlib import Path

from pydantic import Field, PositiveFloat, PositiveInt

from src.training.config import BaseTrainingConfig


class Stage1Config(BaseTrainingConfig):
    output_dir: Path = Path("outputs/stage1")

    gradient_accumulation_steps: PositiveInt = 1
    max_steps: PositiveInt = 25600
    learning_rate: PositiveFloat = 2e-4
    min_learning_rate_ratio: float = Field(default=0.5, ge=0.0, le=1.0)

    train_max_batch_padded_audio_seconds: PositiveFloat = 1400

    validation_batches: PositiveInt = 2_560
    validation_max_batch_padded_audio_seconds: PositiveFloat = 700

    num_workers: PositiveInt = 8
    prefetch_factor: PositiveInt = 2

    early_stopping_patience: PositiveInt = 3
