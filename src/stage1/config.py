from pathlib import Path

from pydantic import PositiveFloat, PositiveInt

from src.training.config import BaseTrainingConfig


class Stage1Config(BaseTrainingConfig):
    output_dir: Path = Path("outputs/stage1")

    gradient_accumulation_steps: PositiveInt = 2
    max_steps: PositiveInt = 38400
    learning_rate: PositiveFloat = 5e-4

    train_max_batch_padded_audio_seconds: PositiveFloat = 1200

    validation_max_batch_padded_audio_seconds: PositiveFloat = 700

    num_workers: PositiveInt = 8
    prefetch_factor: PositiveInt = 4

    early_stopping_patience: PositiveInt = 5
