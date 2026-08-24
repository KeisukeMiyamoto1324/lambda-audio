from dataclasses import dataclass, field
from pathlib import Path

from src.shared.config import BaseTrainingConfig


@dataclass(frozen=True)
class Stage2Config(BaseTrainingConfig):
    initialization_checkpoint: Path = field(kw_only=True)
    train_max_batch_padded_audio_seconds: float = 120.0
    validation_max_batch_padded_audio_seconds: float = 60.0
    train_max_batch_attention_cost: int = 12_000_000
    validation_max_batch_attention_cost: int = 12_000_000
    adapter_learning_rate: float = 1e-4
    backbone_learning_rate: float = 1e-6
