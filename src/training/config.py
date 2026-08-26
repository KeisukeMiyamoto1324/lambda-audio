from pathlib import Path

from pydantic import Field, NonNegativeFloat, PositiveFloat, PositiveInt

from src.modeling.config import ModelConfig


class BaseTrainingConfig(ModelConfig):
    output_dir: Path
    metadata_index_path: Path = Path(
        "outputs/datasets/reazon-speech-v2-metadata.sqlite3"
    )
    shard_cache_dir: Path = Path("outputs/datasets/reazon-speech-v2-shards")
    max_text_length: PositiveInt = 256
    validation_samples: PositiveInt = 256
    shuffle_buffer_size: PositiveInt = 1_000
    bucket_buffer_size: PositiveInt = 512
    num_workers: PositiveInt = 8
    prefetch_factor: PositiveInt = 2
    encoder_frame_samples: PositiveInt = 320
    train_max_batch_padded_audio_seconds: PositiveFloat = 1_400.0
    validation_max_batch_padded_audio_seconds: PositiveFloat = 120.0
    train_max_batch_attention_cost: PositiveInt = 56_000_000
    validation_max_batch_attention_cost: PositiveInt = 12_000_000
    gradient_accumulation_steps: PositiveInt = 8
    max_steps: PositiveInt = 10_000
    warmup_steps: PositiveInt = 500
    weight_decay: NonNegativeFloat = 0.01
    max_grad_norm: PositiveFloat = 1.0
    log_steps: PositiveInt = 10
    evaluation_steps: PositiveInt = 500
    save_steps: PositiveInt = 500
    validation_batches: PositiveInt = 32
    early_stopping_patience: PositiveInt = 5
    seed: int = Field(default=42, ge=0)
