from pathlib import Path
from typing import Self

from pydantic import NonNegativeFloat, PositiveFloat, PositiveInt, model_validator

from src.training.config import BaseTrainingConfig


class Stage2Config(BaseTrainingConfig):
    output_dir: Path = Path("outputs/stage2")
    initialization_checkpoint: Path = Path("outputs/stage1/model.ckpt")

    gradient_accumulation_steps: PositiveInt = 4
    max_steps: PositiveInt = 51200
    adapter_learning_rate: PositiveFloat = 1e-5
    backbone_learning_rate: PositiveFloat = 2e-5
    early_stopping_min_delta: NonNegativeFloat = 0.01

    train_max_batch_padded_audio_seconds: PositiveFloat = 900

    validation_max_batch_padded_audio_seconds: PositiveFloat = 750.0

    num_workers: PositiveInt = 8
    prefetch_factor: PositiveInt = 4

    early_stopping_patience: PositiveInt = 5

    @model_validator(mode="after")
    def validate_initialization_checkpoint(self) -> Self:
        if self.resume_checkpoint is None and not self.initialization_checkpoint.is_file():
            raise ValueError(f"The initialization checkpoint does not exist: {self.initialization_checkpoint}")
        return self
