from io import BytesIO
from typing import Protocol

import librosa
import torch
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
    if audio_array.size == 0:
        raise ValueError("The audio file is empty.")
    if audio_array.size > config.sampling_rate * config.max_audio_seconds:
        raise ValueError(f"The audio file must be {config.max_audio_seconds:g} seconds or shorter.")

    audio_batch = feature_extractor(
        audio_array,
        sampling_rate=int(sampling_rate),
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
