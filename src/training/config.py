from pathlib import Path
from typing import Self

from pydantic import Field, FilePath, NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt, model_validator

from src.modeling.config import ModelConfig


class TrainingConfig(ModelConfig):
    # ---------------------------------------------------------
    # Output, resume checkpoint, and data paths.
    # ---------------------------------------------------------
    output_dir: Path = Path("outputs/training")
    resume_checkpoint: FilePath | None = None
    dataset_cache_dir: Path = Path("outputs/datasets/lambda-audio-courpus")
    inference_samples_path: FilePath = Path("data/data.parquet")

    # ---------------------------------------------------------
    # Audio and generated text length limits.
    # ---------------------------------------------------------
    max_audio_seconds: PositiveFloat = 30.0
    max_new_tokens: PositiveInt = 256

    # ---------------------------------------------------------
    # Data loading, batching, and gradient accumulation.
    # ---------------------------------------------------------
    bucket_buffer_size: PositiveInt = 512
    num_workers: PositiveInt = 12
    prefetch_factor: PositiveInt = 4
    max_batch_padded_audio_seconds: PositiveFloat = 1800
    gradient_accumulation_steps: PositiveInt = 1

    # ---------------------------------------------------------
    # Training duration and the step to unfreeze all weights.
    # ---------------------------------------------------------
    max_steps: PositiveInt = 64000
    unfreeze_step: NonNegativeInt = 4000

    # ---------------------------------------------------------
    # Learning rates, schedule, and gradient stability.
    # ---------------------------------------------------------
    adapter_learning_rate: PositiveFloat = 5e-5
    backbone_learning_rate: PositiveFloat = 5e-5
    warmup_steps: PositiveInt = 2000
    min_learning_rate_ratio: float = Field(default=0.2, ge=0.0, le=1.0)
    weight_decay: NonNegativeFloat = 0.01
    max_grad_norm: PositiveFloat = 1.0

    # ---------------------------------------------------------
    # Logging, validation, checkpointing, and early stopping.
    # ---------------------------------------------------------
    log_steps: PositiveInt = 10
    evaluation_steps: PositiveInt = 2000
    save_steps: PositiveInt = 500
    early_stopping_patience: PositiveInt = 5
    early_stopping_min_delta: NonNegativeFloat = 0.01

    # ---------------------------------------------------------
    # Random seed for reproducibility.
    # ---------------------------------------------------------
    seed: int = Field(default=42, ge=0)

    @model_validator(mode="after")
    def validate_unfreeze_step(self) -> Self:
        if self.unfreeze_step >= self.max_steps:
            raise ValueError("unfreeze_step must be less than max_steps.")
        return self
