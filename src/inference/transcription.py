from io import BytesIO
from typing import Protocol

import av
import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, NonNegativeFloat
from transformers import PreTrainedTokenizerBase, SequenceFeatureExtractor

from src.modeling.lambda_audio import LambdaAudioModel


class TranscriptionConfig(Protocol):
    sampling_rate: int
    max_audio_seconds: float
    max_new_tokens: int


class TranscriptionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    audio_seconds: NonNegativeFloat


def transcribe(
    model: LambdaAudioModel,
    feature_extractor: SequenceFeatureExtractor,
    tokenizer: PreTrainedTokenizerBase,
    config: TranscriptionConfig,
    device: torch.device,
    audio: bytes,
    *,
    num_beams: int = 1,
) -> TranscriptionResult:
    # ---------------------------------------------------------
    # Load mono audio at the encoder sampling rate and validate its length.
    # ---------------------------------------------------------
    audio_array = decode_audio(audio, config.sampling_rate)
    if audio_array.size > config.sampling_rate * config.max_audio_seconds:
        raise ValueError(f"The audio file must be {config.max_audio_seconds:g} seconds or shorter.")

    audio_batch = feature_extractor(
        audio_array,
        sampling_rate=config.sampling_rate,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_values = audio_batch["input_values"].to(device)
    audio_padding_mask = audio_batch["attention_mask"].eq(0).to(device)

    if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must define BOS and EOS tokens.")

    # ---------------------------------------------------------
    # Generate from BOS and decode only newly generated transcription tokens.
    # ---------------------------------------------------------
    input_ids = torch.tensor([[tokenizer.bos_token_id]], device=device)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        generated_ids = model.generate(
            input_values=input_values,
            audio_padding_mask=audio_padding_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=config.max_new_tokens,
            do_sample=False,
            num_beams=num_beams,
            eos_token_id=tokenizer.eos_token_id,
        )

    transcription_ids = generated_ids[:, input_ids.shape[1] :]
    text = tokenizer.batch_decode(
        transcription_ids,
        skip_special_tokens=True,
    )[0].strip()
    return TranscriptionResult(
        text=text,
        audio_seconds=audio_array.size / config.sampling_rate,
    )


def decode_audio(audio: bytes, sampling_rate: int) -> NDArray[np.float32]:
    container = av.open(BytesIO(audio), mode="r")
    with container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(
            format="fltp",
            layout="mono",
            rate=sampling_rate,
        )
        arrays = [converted.to_ndarray()[0] for frame in container.decode(stream) for converted in resampler.resample(frame)]
        arrays.extend(converted.to_ndarray()[0] for converted in resampler.resample(None))
    if not arrays:
        raise ValueError("The audio file is empty.")
    return np.concatenate(arrays).astype(np.float32, copy=False)
