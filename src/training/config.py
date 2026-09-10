from pathlib import Path

from pydantic import Field, FilePath, NonNegativeFloat, PositiveFloat, PositiveInt

from src.modeling.config import ModelConfig


class BaseTrainingConfig(ModelConfig):
    output_dir: Path
    compile_model: bool = False
    resume_checkpoint: FilePath | None = None
    inference_samples_path: FilePath = Path("data/data.parquet")
    max_audio_seconds: PositiveFloat = 30.0
    max_new_tokens: PositiveInt = 256
    dataset_cache_dir: Path = Path("outputs/datasets/lambda-audio-corpus-1")
    max_text_length: PositiveInt = 256
    bucket_buffer_size: PositiveInt = 128
    num_workers: PositiveInt = 8
    prefetch_factor: PositiveInt = 2
    train_max_batch_padded_audio_seconds: PositiveFloat = 1_400.0
    validation_max_batch_padded_audio_seconds: PositiveFloat = 120.0
    gradient_accumulation_steps: PositiveInt = 8
    max_steps: PositiveInt = 10_000
    warmup_steps: PositiveInt = 500
    min_learning_rate_ratio: float = Field(default=0.2, ge=0.0, le=1.0)
    weight_decay: NonNegativeFloat = 0.01
    max_grad_norm: PositiveFloat = 1.0
    log_steps: PositiveInt = 10
    evaluation_steps: PositiveInt = 2000
    save_steps: PositiveInt = 2000
    validation_batches: PositiveInt = 32
    early_stopping_patience: PositiveInt = 5
    early_stopping_min_delta: NonNegativeFloat = 0.01
    seed: int = Field(default=42, ge=0)
