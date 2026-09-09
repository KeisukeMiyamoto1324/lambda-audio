from pathlib import Path
from typing import Self

from pydantic import Field, PositiveFloat, PositiveInt, model_validator

from src.training.config import BaseTrainingConfig


class Stage2Config(BaseTrainingConfig):
    output_dir: Path = Path("outputs/stage2")
    initialization_checkpoint: Path = Path("outputs/stage1/model.ckpt")

    gradient_accumulation_steps: PositiveInt = 4
    max_steps: PositiveInt = 51200
    adapter_learning_rate: PositiveFloat = 5e-5
    encoder_learning_rate: PositiveFloat = 5e-6
    backbone_learning_rate: PositiveFloat = 5e-5
    min_learning_rate_ratio: float = Field(default=0.2, ge=0.0, le=1.0)
    train_max_batch_padded_audio_seconds: PositiveFloat = 900.0
    train_max_batch_attention_cost: PositiveInt = 32_000_000

    validation_batches: PositiveInt = 2_560
    validation_max_batch_padded_audio_seconds: PositiveFloat = 750.0
    validation_max_batch_attention_cost: PositiveInt = 20_000_000

    num_workers: PositiveInt = 8
    prefetch_factor: PositiveInt = 2

    early_stopping_patience: PositiveInt = 3

    @model_validator(mode="after")
    def validate_initialization_checkpoint(self) -> Self:
        if self.resume_checkpoint is None and not self.initialization_checkpoint.is_file():
            raise ValueError(f"The initialization checkpoint does not exist: {self.initialization_checkpoint}")
        return self
