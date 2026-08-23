from dataclasses import dataclass, field
from pathlib import Path

from src.shared.config import BaseTrainingConfig


@dataclass(frozen=True)
class Stage2Config(BaseTrainingConfig):
    initialization_checkpoint: Path = field(kw_only=True)
    adapter_learning_rate: float = 1e-4
    backbone_learning_rate: float = 1e-6
