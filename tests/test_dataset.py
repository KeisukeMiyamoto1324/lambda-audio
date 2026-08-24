from pathlib import Path
from typing import Any

import numpy as np
import pytest
from datasets import IterableDataset
from pytest import MonkeyPatch
from torch.utils.data import DataLoader

from src.shared.dataset import (
    AudioBatchDataset,
    TranscribedAudioDataset,
    load_reazon_speech_datasets,
)
from src.shared.metadata import create_metadata_index


def create_audio_example(key: str, audio_samples: int) -> dict[str, Any]:
    return {
        "__key__": key,
        "flac": {"array": np.ones(audio_samples, dtype=np.float32)},
    }


def create_metadata_example(key: str, transcription: str) -> dict[str, Any]:
    return {
        "file_name": f"data/{key}.flac",
        "transcription": transcription,
    }


def create_transcribed_dataset(
    path: Path,
    examples: list[tuple[str, int]],
) -> TranscribedAudioDataset:
    create_metadata_index(
        [create_metadata_example(key, " 音声 ") for key, _ in examples],
        path,
        revision="revision",
    )
    source = IterableDataset.from_generator(
        lambda: iter(
            [create_audio_example(key, samples) for key, samples in examples]
        )
    )
    return TranscribedAudioDataset(source, path)


def test_audio_batch_dataset_limits_padded_audio_samples(
    tmp_path: Path,
) -> None:
    source = create_transcribed_dataset(
        tmp_path / "metadata.sqlite3",
        [
            ("000/first", 8),
            ("000/second", 8),
            ("000/third", 4),
            ("000/fourth", 4),
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
    assert all(
        len(batch) * max(len(example["audio_array"]) for example in batch) <= 12
        for batch in batches
    )
    assert all(
        example["transcription"] == "音声"
        for batch in batches
        for example in batch
    )


def test_audio_batch_dataset_limits_attention_cost(tmp_path: Path) -> None:
    source = create_transcribed_dataset(
        tmp_path / "metadata.sqlite3",
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
    assert all(
        len(batch)
        * max(len(example["audio_array"]) for example in batch) ** 2
        <= 64
        for batch in batches
    )


def test_audio_batch_dataset_reduces_weighted_padding_ratio(
    tmp_path: Path,
) -> None:
    examples = [
        (f"000/{index:03d}", samples)
        for index, samples in enumerate(range(10, 522))
    ]
    source = create_transcribed_dataset(tmp_path / "metadata.sqlite3", examples)
    dataset = AudioBatchDataset(
        source,
        max_batch_padded_audio_samples=20_000,
        max_batch_attention_cost=1_000_000_000,
        encoder_frame_samples=1,
        bucket_buffer_size=512,
        seed=42,
    )

    batches = list(dataset)
    valid_samples = sum(
        len(example["audio_array"])
        for batch in batches
        for example in batch
    )
    padded_samples = sum(
        len(batch) * max(len(example["audio_array"]) for example in batch)
        for batch in batches
    )

    assert padded_samples / valid_samples < 1.5


def test_transcribed_dataset_preserves_worker_sharding(tmp_path: Path) -> None:
    keys = [f"{shard:03d}/{index}" for shard in range(4) for index in range(2)]
    index_path = tmp_path / "metadata.sqlite3"
    create_metadata_index(
        [create_metadata_example(key, key) for key in keys],
        index_path,
        revision="revision",
    )

    def generate(shards: list[int]) -> Any:
        for shard in shards:
            for index in range(2):
                key = f"{shard:03d}/{index}"
                yield create_audio_example(key, index + 1)

    audio_dataset = IterableDataset.from_generator(
        generate,
        gen_kwargs={"shards": [0, 1, 2, 3]},
    )
    dataset = TranscribedAudioDataset(audio_dataset, index_path)
    loader = DataLoader(dataset, batch_size=None, num_workers=2)

    transcriptions = [example["transcription"] for example in loader]

    assert sorted(transcriptions) == keys


def test_load_reazon_speech_datasets_preserves_audio_shards(
    monkeypatch: MonkeyPatch,
) -> None:
    audio_dataset = IterableDataset.from_generator(
        lambda: iter(
            [
                create_audio_example("000/first", 1),
                create_audio_example("000/second", 2),
                create_audio_example("000/third", 3),
            ]
        )
    )

    def load_dataset_stub(*args: Any, **kwargs: Any) -> IterableDataset:
        return audio_dataset

    monkeypatch.setattr("src.shared.dataset.load_dataset", load_dataset_stub)

    training, validation = load_reazon_speech_datasets(
        token="token",
        validation_samples=1,
        shuffle_buffer_size=2,
        seed=42,
    )

    assert [example["__key__"] for example in validation] == ["000/first"]
    assert {example["__key__"] for example in training} == {
        "000/second",
        "000/third",
    }


def test_audio_batch_dataset_rejects_oversized_audio(tmp_path: Path) -> None:
    source = create_transcribed_dataset(
        tmp_path / "metadata.sqlite3",
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
