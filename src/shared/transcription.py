from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from io import BytesIO
from itertools import batched
from time import perf_counter
from typing import Protocol

import av
import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, NonNegativeFloat
from transformers import PreTrainedTokenizerBase, SequenceFeatureExtractor

from src.modeling.lambda_audio import LambdaAudioModel
from src.shared.languages import OutputLanguage, language_token_id
from src.shared.samples import InferenceSample


class TranscriptionConfig(Protocol):
    sampling_rate: int
    max_audio_seconds: float
    max_new_tokens: int


class TranscriptionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    audio_seconds: NonNegativeFloat
    inference_seconds: NonNegativeFloat


@dataclass(frozen=True)
class TranscriptionBatch:
    inputs: dict[str, torch.Tensor]
    audio_seconds: list[float]


def prepare_transcription_batch(
    feature_extractor: SequenceFeatureExtractor,
    tokenizer: PreTrainedTokenizerBase,
    config: TranscriptionConfig,
    audio: list[bytes],
    output_language: OutputLanguage | list[OutputLanguage] = "ja",
) -> TranscriptionBatch:
    # ---------------------------------------------------------
    # Decode, normalize and pad audio once into reusable CPU tensors.
    # ---------------------------------------------------------
    audio_arrays = decode_audio_batch(audio, config)
    audio_batch = feature_extractor(
        audio_arrays,
        sampling_rate=config.sampling_rate,
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must define BOS and EOS tokens.")

    languages = [output_language] * len(audio) if isinstance(output_language, str) else output_language
    if len(languages) != len(audio):
        raise ValueError("Each audio clip must have one output language.")
    prompts = [[tokenizer.bos_token_id, language_token_id(tokenizer, language)] for language in languages]
    input_ids = torch.tensor(prompts, dtype=torch.long)
    return TranscriptionBatch(
        inputs={
            "input_values": audio_batch["input_values"],
            "audio_padding_mask": audio_batch["attention_mask"].eq(0),
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        },
        audio_seconds=[array.size / config.sampling_rate for array in audio_arrays],
    )


def transcribe_prepared_batch(
    model: LambdaAudioModel,
    tokenizer: PreTrainedTokenizerBase,
    config: TranscriptionConfig,
    device: torch.device,
    batch: TranscriptionBatch,
) -> list[TranscriptionResult]:
    # ---------------------------------------------------------
    # Transfer cached inputs and generate using the current model weights.
    # ---------------------------------------------------------
    inference_start = perf_counter()
    inputs = {name: tensor.to(device) for name, tensor in batch.inputs.items()}
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    transcription_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
    texts = tokenizer.batch_decode(transcription_ids, skip_special_tokens=True)
    inference_seconds = (perf_counter() - inference_start) / len(batch.audio_seconds)
    return [
        TranscriptionResult(text=text.strip(), audio_seconds=seconds, inference_seconds=inference_seconds) for text, seconds in zip(texts, batch.audio_seconds, strict=True)
    ]


def transcribe(
    model: LambdaAudioModel,
    feature_extractor: SequenceFeatureExtractor,
    tokenizer: PreTrainedTokenizerBase,
    config: TranscriptionConfig,
    device: torch.device,
    audio: list[bytes],
    output_language: OutputLanguage | list[OutputLanguage] = "ja",
) -> list[TranscriptionResult]:
    # ---------------------------------------------------------
    # Include preprocessing time for one-off transcription requests.
    # ---------------------------------------------------------
    inference_start = perf_counter()
    batch = prepare_transcription_batch(feature_extractor, tokenizer, config, audio, output_language)
    results = transcribe_prepared_batch(model, tokenizer, config, device, batch)
    inference_seconds = (perf_counter() - inference_start) / len(audio)
    return [result.model_copy(update={"inference_seconds": inference_seconds}) for result in results]


def iter_transcriptions(
    model: LambdaAudioModel,
    feature_extractor: SequenceFeatureExtractor,
    tokenizer: PreTrainedTokenizerBase,
    config: TranscriptionConfig,
    device: torch.device,
    samples: Iterable[InferenceSample],
    batch_size: int,
) -> Iterator[tuple[InferenceSample, TranscriptionResult]]:
    # ---------------------------------------------------------
    # Generate each batch together while preserving sample order.
    # ---------------------------------------------------------
    for batch in batched(samples, batch_size):
        results = transcribe(
            model,
            feature_extractor,
            tokenizer,
            config,
            device,
            [sample.audio for sample in batch],
            output_language=[sample.lang for sample in batch],
        )
        yield from zip(batch, results, strict=True)


def iter_batch_transcriptions(
    samples: Iterable[InferenceSample],
    batch_size: int,
    transcribe_batch: Callable[[list[bytes]], list[TranscriptionResult]],
) -> Iterator[tuple[InferenceSample, TranscriptionResult]]:
    for batch in batched(samples, batch_size):
        results = transcribe_batch([sample.audio for sample in batch])
        yield from zip(batch, results, strict=True)


def decode_audio_batch(audio: list[bytes], config: TranscriptionConfig) -> list[NDArray[np.float32]]:
    if not audio:
        raise ValueError("An inference batch must contain at least one audio clip.")
    arrays = [decode_audio(encoded, config.sampling_rate) for encoded in audio]
    if any(array.size > config.sampling_rate * config.max_audio_seconds for array in arrays):
        raise ValueError(f"The audio file must be {config.max_audio_seconds:g} seconds or shorter.")
    return arrays


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
