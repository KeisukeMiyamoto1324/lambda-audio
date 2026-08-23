from dataclasses import dataclass

from src.shared.config import BaseTrainingConfig


@dataclass(frozen=True)
class Stage1Config(BaseTrainingConfig):
    learning_rate: float = 1e-4
