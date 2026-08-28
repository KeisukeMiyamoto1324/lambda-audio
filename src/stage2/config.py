from pathlib import Path

from pydantic import FilePath, PositiveFloat, PositiveInt

from src.training.config import BaseTrainingConfig


class Stage2Config(BaseTrainingConfig):
    output_dir: Path = Path("outputs/stage2")
    initialization_checkpoint: FilePath = Path("outputs/stage1/model.ckpt")

    gradient_accumulation_steps: PositiveInt = 1
    max_steps: PositiveInt = 204_800
    adapter_learning_rate: PositiveFloat = 2e-5
    backbone_learning_rate: PositiveFloat = 2e-5

    train_max_batch_padded_audio_seconds: PositiveFloat = 1_500.0
    train_max_batch_attention_cost: PositiveInt = 40_000_000

    validation_samples: PositiveInt = 2_560
    validation_batches: PositiveInt = 2_560
    validation_max_batch_padded_audio_seconds: PositiveFloat = 750.0
    validation_max_batch_attention_cost: PositiveInt = 20_000_000

    num_workers: PositiveInt = 8
    prefetch_factor: PositiveInt = 4

    early_stopping_patience: PositiveInt = 5
