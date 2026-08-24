from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BaseTrainingConfig:
    output_dir: Path
    metadata_index_path: Path = Path(
        "outputs/datasets/reazon-speech-v2-metadata.sqlite3"
    )
    encoder_model_id: str = "reazon-research/japanese-zipformer-base-k2"
    decoder_model_id: str = "LiquidAI/LFM2.5-230M-Base"
    sampling_rate: int = 16_000
    max_text_length: int = 256
    validation_samples: int = 256
    shuffle_buffer_size: int = 1_000
    bucket_buffer_size: int = 512
    num_workers: int = 8
    prefetch_factor: int = 2
    encoder_frame_samples: int = 320
    train_max_batch_padded_audio_seconds: float = 300.0
    validation_max_batch_padded_audio_seconds: float = 120.0
    train_max_batch_attention_cost: int = 12_000_000
    validation_max_batch_attention_cost: int = 12_000_000
    gradient_accumulation_steps: int = 8
    max_steps: int = 10_000
    warmup_steps: int = 500
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    log_steps: int = 10
    evaluation_steps: int = 500
    save_steps: int = 500
    validation_batches: int = 32
    early_stopping_patience: int = 5
    seed: int = 42
