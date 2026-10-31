from io import BytesIO

import av
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, NonNegativeFloat

from whisper.config import TranscriptionConfig


class TranscriptionResult(BaseModel):
    text: str
    audio_seconds: NonNegativeFloat
    inference_seconds: NonNegativeFloat


def decode_audio_batch(audio: list[bytes], config: TranscriptionConfig) -> list[NDArray[np.float32]]:
    if not audio:
        raise ValueError("An inference batch must contain at least one audio clip.")
    arrays = [decode_audio(encoded, config.sampling_rate) for encoded in audio]
    if any(array.size > config.sampling_rate * config.max_audio_seconds for array in arrays):
        raise ValueError(f"The audio file must be {config.max_audio_seconds:g} seconds or shorter.")
    return arrays


def decode_audio(audio: bytes, sampling_rate: int) -> NDArray[np.float32]:
    # ---------------------------------------------------------
    # Decode and resample mono audio without a training framework.
    # ---------------------------------------------------------
    with av.open(BytesIO(audio), mode="r") as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=sampling_rate)
        arrays = [converted.to_ndarray()[0] for frame in container.decode(stream) for converted in resampler.resample(frame)]
        arrays.extend(converted.to_ndarray()[0] for converted in resampler.resample(None))
    if not arrays:
        raise ValueError("The audio file is empty.")
    return np.concatenate(arrays).astype(np.float32, copy=False)
