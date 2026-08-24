from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InferenceConfig:
    checkpoint_path: Path
    input_dir: Path
    max_new_tokens: int = 256
    encoder_model_id: str = "reazon-research/japanese-zipformer-base-k2"
    decoder_model_id: str = "LiquidAI/LFM2.5-230M-Base"
    sampling_rate: int = 16_000
    max_audio_seconds: float = 30.0
