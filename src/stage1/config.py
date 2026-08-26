from pathlib import Path

from pydantic import PositiveFloat, PositiveInt

from src.training.config import BaseTrainingConfig


class Stage1Config(BaseTrainingConfig):
    output_dir: Path = Path("outputs/stage1")

    gradient_accumulation_steps: PositiveInt = 2
    max_steps: PositiveInt = 204_800
    learning_rate: PositiveFloat = 1e-4
    
    train_max_batch_padded_audio_seconds: PositiveFloat = 1_400.0
    train_max_batch_attention_cost: PositiveInt = 56_000_000

    validation_max_batch_padded_audio_seconds: PositiveFloat = 256
    validation_max_batch_attention_cost: PositiveInt = 36_000_000
    
    num_workers: PositiveInt = 16
    prefetch_factor: PositiveInt = 4
        