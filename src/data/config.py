from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataPreparationConfig:
    output_dir: Path
    dataset_name: str = "reazon-research/reazonspeech"
    dataset_config: str = "tiny"
    sampling_rate: int = 16_000
    max_audio_seconds: float = 30.0
    validation_samples: int = 256
    seed: int = 42
    num_proc: int = 4
    max_shard_size: str = "1GB"
