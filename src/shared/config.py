from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BaseTrainingConfig:
    output_dir: Path
    dataset_name: str = "reazon-research/reazonspeech"
    dataset_config: str = "tiny"
    encoder_model_id: str = "reazon-research/japanese-zipformer-base-k2"
    decoder_model_id: str = "LiquidAI/LFM2.5-230M-Base"
    sampling_rate: int = 16_000
    max_audio_seconds: float = 30.0
    max_text_length: int = 256
    validation_samples: int = 256
    shuffle_buffer_size: int = 10_000
    train_batch_size: int = 2
    validation_batch_size: int = 2
    dataloader_workers: int = 4
    gradient_accumulation_steps: int = 8
    max_steps: int = 10_000
    warmup_steps: int = 500
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    log_steps: int = 10
    evaluation_steps: int = 500
    save_steps: int = 500
    validation_batches: int = 32
    seed: int = 42
