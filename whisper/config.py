from typing import Protocol

from pydantic import BaseModel, Field, PositiveInt

MODEL_ID = "mlx-community/whisper-large-v3-turbo-4bit"


class TranscriptionConfig(Protocol):
    sampling_rate: int
    max_audio_seconds: float
    max_new_tokens: int


class InferenceConfig(BaseModel):
    sampling_rate: PositiveInt = 16000
    max_audio_seconds: float = Field(default=30.0, gt=0, le=30)
    max_new_tokens: int = Field(default=256, ge=1, le=444)
