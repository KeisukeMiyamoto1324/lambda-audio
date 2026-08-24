from typing import Any

import numpy as np
import pytest
from datasets import IterableDataset
from pytest import MonkeyPatch

from src.shared.dataset import AudioBatchDataset, load_reazon_speech_datasets


def create_example(key: str, audio_samples: int) -> dict[str, Any]:
    return {
        "__key__": key,
        "file_name": f"data/{key}.flac",
        "flac": {"array": np.ones(audio_samples, dtype=np.float32)},
        "transcription": " 音声 ",
    }


def test_audio_batch_dataset_limits_audio_samples_and_examples() -> None:
    source = IterableDataset.from_generator(
        lambda: iter(
            [
                create_example("000/first", 8),
                create_example("000/second", 8),
                create_example("000/third", 4),
                create_example("000/fourth", 4),
            ]
        )
    )
    dataset = AudioBatchDataset(
        source,
        max_batch_audio_samples=12,
        max_batch_examples=2,
    )

    batches = list(dataset)

    assert [len(batch) for batch in batches] == [1, 2, 1]
    assert all(
        sum(len(example["audio_array"]) for example in batch) <= 12
        for batch in batches
    )
    assert batches[0][0]["transcription"] == "音声"


def test_audio_batch_dataset_rejects_mismatched_keys() -> None:
    example = create_example("000/audio", 4)
    example["file_name"] = "data/000/text.flac"
    source = IterableDataset.from_generator(lambda: iter([example]))
    dataset = AudioBatchDataset(source, 4, 1)

    with pytest.raises(ValueError, match="keys do not match"):
        list(dataset)


def test_load_reazon_speech_datasets_joins_and_splits_rows(
    monkeypatch: MonkeyPatch,
) -> None:
    audio_dataset = IterableDataset.from_generator(
        lambda: iter(
            [
                {"__key__": "000/first", "flac": {"array": [1]}},
                {"__key__": "000/second", "flac": {"array": [2]}},
                {"__key__": "000/third", "flac": {"array": [3]}},
            ]
        )
    )
    metadata_dataset = IterableDataset.from_generator(
        lambda: iter(
            [
                {
                    "file_name": "data/000/first.flac",
                    "transcription": "一",
                },
                {
                    "file_name": "data/000/second.flac",
                    "transcription": "二",
                },
                {
                    "file_name": "data/000/third.flac",
                    "transcription": "三",
                },
            ]
        )
    )
    datasets = iter([audio_dataset, metadata_dataset])

    def load_dataset_stub(*args: Any, **kwargs: Any) -> IterableDataset:
        return next(datasets)

    monkeypatch.setattr("src.shared.dataset.load_dataset", load_dataset_stub)

    training, validation = load_reazon_speech_datasets(
        token="token",
        validation_samples=1,
        shuffle_buffer_size=2,
        seed=42,
    )

    assert [example["transcription"] for example in validation] == ["一"]
    assert {example["transcription"] for example in training} == {"二", "三"}
