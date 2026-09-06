from typing import Any

import numpy as np
import pytest
from datasets import IterableDataset

from src.data.dataset import AudioBatchDataset


def create_audio_example(transcription: str, audio_samples: int) -> dict[str, Any]:
    return {
        "audio_array": np.ones(audio_samples, dtype=np.float32),
        "transcription": transcription,
    }


def create_transcribed_dataset(
    examples: list[tuple[str, int]],
) -> IterableDataset:
    return IterableDataset.from_generator(lambda: iter([create_audio_example(transcription, samples) for transcription, samples in examples]))


def test_audio_batch_dataset_limits_padded_audio_samples() -> None:
    source = create_transcribed_dataset(
        [
            ("音声", 8),
            ("音声", 8),
            ("音声", 4),
            ("音声", 4),
        ],
    )
    dataset = AudioBatchDataset(
        source,
        max_batch_padded_audio_samples=12,
        max_batch_attention_cost=1_000,
        encoder_frame_samples=1,
        bucket_buffer_size=4,
        seed=42,
    )

    batches = list(dataset)

    assert sorted(len(batch) for batch in batches) == [1, 1, 2]
    assert all(len(batch) * max(len(example["audio_array"]) for example in batch) <= 12 for batch in batches)
    assert all(example["transcription"] == "音声" for batch in batches for example in batch)


def test_audio_batch_dataset_limits_attention_cost() -> None:
    source = create_transcribed_dataset(
        [
            ("000/first", 8),
            ("000/second", 8),
            ("000/third", 4),
            ("000/fourth", 4),
        ],
    )
    dataset = AudioBatchDataset(
        source,
        max_batch_padded_audio_samples=1_000,
        max_batch_attention_cost=64,
        encoder_frame_samples=1,
        bucket_buffer_size=4,
        seed=42,
    )

    batches = list(dataset)

    assert sorted(len(batch) for batch in batches) == [1, 1, 2]
    assert all(len(batch) * max(len(example["audio_array"]) for example in batch) ** 2 <= 64 for batch in batches)


def test_audio_batch_dataset_reduces_weighted_padding_ratio() -> None:
    examples = [(f"000/{index:03d}", samples) for index, samples in enumerate(range(10, 522))]
    source = create_transcribed_dataset(examples)
    dataset = AudioBatchDataset(
        source,
        max_batch_padded_audio_samples=20_000,
        max_batch_attention_cost=1_000_000_000,
        encoder_frame_samples=1,
        bucket_buffer_size=512,
        seed=42,
    )

    batches = list(dataset)
    valid_samples = sum(len(example["audio_array"]) for batch in batches for example in batch)
    padded_samples = sum(len(batch) * max(len(example["audio_array"]) for example in batch) for batch in batches)

    assert padded_samples / valid_samples < 1.5


def test_audio_batch_dataset_refills_between_ready_batches() -> None:
    examples = [(f"000/{index}", 4) for index in range(8)]
    reads: list[str] = []

    def generate() -> Any:
        for key, samples in examples:
            reads.append(key)
            yield create_audio_example(key, samples)

    source = IterableDataset.from_generator(generate)
    dataset = AudioBatchDataset(
        source,
        max_batch_padded_audio_samples=8,
        max_batch_attention_cost=1_000,
        encoder_frame_samples=1,
        bucket_buffer_size=4,
        seed=42,
    )
    batches = iter(dataset)

    next(batches)
    assert len(reads) == 4
    next(batches)
    assert len(reads) == 6
    next(batches)
    assert len(reads) == 8


def test_audio_batch_dataset_rejects_oversized_audio() -> None:
    source = create_transcribed_dataset(
        [("000/audio", 13)],
    )
    dataset = AudioBatchDataset(
        source,
        max_batch_padded_audio_samples=12,
        max_batch_attention_cost=1_000,
        encoder_frame_samples=1,
        bucket_buffer_size=4,
        seed=42,
    )

    with pytest.raises(ValueError, match="exceeds the batch limits"):
        list(dataset)
