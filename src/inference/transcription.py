from io import BytesIO
from typing import Protocol

import librosa
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
) -> TranscriptionResult:
    # ---------------------------------------------------------
    # Load mono audio at the encoder sampling rate and validate its length.
    # ---------------------------------------------------------
    audio_array, sampling_rate = librosa.load(
        BytesIO(audio),
        sr=config.sampling_rate,
        mono=True,
        dtype="float32",
    )
    return transcribe_audio_array(
        model,
        feature_extractor,
        tokenizer,
        config,
        device,
        audio_array,
        int(sampling_rate),
    )


def transcribe_audio_array(
    model: LambdaAudioModel,
    feature_extractor: SequenceFeatureExtractor,
    tokenizer: PreTrainedTokenizerBase,
    config: TranscriptionConfig,
    device: torch.device,
    audio_array: NDArray[np.float32],
    sampling_rate: int,
) -> TranscriptionResult:
    return transcribe_audio_batch(
        model,
        feature_extractor,
        tokenizer,
        config,
        device,
        [audio_array],
        [sampling_rate],
    )[0]


def transcribe_audio_batch(
    model: LambdaAudioModel,
    feature_extractor: SequenceFeatureExtractor,
    tokenizer: PreTrainedTokenizerBase,
    config: TranscriptionConfig,
    device: torch.device,
    audio_arrays: list[NDArray[np.float32]],
    sampling_rates: list[int],
) -> list[TranscriptionResult]:
    # ---------------------------------------------------------
    # Validate decoded audio before padded batch feature extraction.
    # ---------------------------------------------------------
    if not audio_arrays:
        raise ValueError("The audio batch is empty.")
    if len(audio_arrays) != len(sampling_rates):
        raise ValueError("Audio arrays and sampling rates must have the same length.")
    if any(audio_array.size == 0 for audio_array in audio_arrays):
        raise ValueError("The audio file is empty.")
    if any(sampling_rate != config.sampling_rate for sampling_rate in sampling_rates):
        raise ValueError(f"The audio sampling rate must be {config.sampling_rate} Hz.")
    if any(audio_array.size > config.sampling_rate * config.max_audio_seconds for audio_array in audio_arrays):
        raise ValueError(f"The audio file must be {config.max_audio_seconds:g} seconds or shorter.")

    audio_batch = feature_extractor(
        audio_arrays,
        sampling_rate=config.sampling_rate,
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_features = audio_batch["input_features"].to(device)
    audio_padding_mask = audio_batch["attention_mask"].eq(0).to(device)

    if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must define BOS and EOS tokens.")

    # ---------------------------------------------------------
    # Generate from BOS and decode only newly generated transcription tokens.
    # ---------------------------------------------------------
    input_ids = torch.full(
        (len(audio_arrays), 1),
        tokenizer.bos_token_id,
        device=device,
    )
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        generated_ids = model.generate(
            input_features=input_features,
            audio_padding_mask=audio_padding_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=config.max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )

    transcription_ids = generated_ids[:, input_ids.shape[1] :]
    texts = tokenizer.batch_decode(
        transcription_ids,
        skip_special_tokens=True,
    )
    return [
        TranscriptionResult(
            text=text.strip(),
            audio_seconds=audio_array.size / config.sampling_rate,
        )
        for text, audio_array in zip(texts, audio_arrays, strict=True)
    ]
