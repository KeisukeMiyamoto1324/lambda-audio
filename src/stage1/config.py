from pathlib import Path

from pydantic import PositiveFloat, PositiveInt

from src.training.config import BaseTrainingConfig


class Stage1Config(BaseTrainingConfig):
    output_dir: Path = Path("outputs/stage1")

    gradient_accumulation_steps: PositiveInt = 1
    max_steps: PositiveInt = 12800
    learning_rate: PositiveFloat = 2e-4

    train_max_batch_padded_audio_seconds: PositiveFloat = 1400
    train_max_batch_attention_cost: PositiveInt = 56_000_000

    validation_batches: PositiveInt = 2_560
    validation_max_batch_padded_audio_seconds: PositiveFloat = 700
    validation_max_batch_attention_cost: PositiveInt = 28_000_000

    num_workers: PositiveInt = 8
    prefetch_factor: PositiveInt = 2

    early_stopping_patience: PositiveInt = 5
