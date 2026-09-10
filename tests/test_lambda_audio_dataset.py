from collections.abc import Iterator
from io import BytesIO
from typing import Any, cast

import numpy as np
import soundfile as sf
from datasets import IterableDataset
from pytest import MonkeyPatch

from src.data.lambda_audio_dataset import (
    DATASET_ID,
    LambdaAudioDataset,
    load_lambda_audio_dataset,
)


class FakeDataset:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.selected_columns: list[str] = []
        self.cast_sampling_rate: int | None = None
        self.shuffle_arguments: tuple[int, int] | None = None
        self.epoch: int | None = None

    def __iter__(self) -> Iterator[dict[str, Any]]:
        yield from self.rows

    def select_columns(self, columns: list[str]) -> "FakeDataset":
        self.selected_columns = columns
        return self

    def cast_column(self, column: str, feature: Any) -> "FakeDataset":
        assert column == "audio"
        self.cast_sampling_rate = cast(int, feature.sampling_rate)
        return self

    def shuffle(self, buffer_size: int, seed: int) -> "FakeDataset":
        self.shuffle_arguments = (buffer_size, seed)
        return self

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


def test_lambda_audio_dataset_reads_audio_and_text_columns() -> None:
    audio_file = BytesIO()
    sf.write(audio_file, np.array([0.25, -0.5]), 16_000, format="FLAC")
    source = FakeDataset(
        [{"audio": {"bytes": audio_file.getvalue()}, "text": " 音声 ", "unused": 1}]
    )
    dataset = LambdaAudioDataset(cast(IterableDataset, source), 16_000)

    example = next(iter(dataset))
    dataset.set_epoch(3)

    np.testing.assert_allclose(
        example["audio_array"],
        np.array([0.25, -0.5], dtype=np.float32),
        atol=1e-4,
    )
    assert example["transcription"] == "音声"
    assert source.epoch == 3


def test_load_lambda_audio_dataset_configures_streaming_and_sharding(
    monkeypatch: MonkeyPatch,
) -> None:
    source = FakeDataset([])
    load_arguments: dict[str, Any] = {}
    shard_arguments: tuple[int, int] | None = None

    def load_stub(dataset_id: str, **kwargs: Any) -> IterableDataset:
        load_arguments.update({"dataset_id": dataset_id, **kwargs})
        return cast(IterableDataset, source)

    def shard_stub(
        dataset: IterableDataset,
        rank: int,
        world_size: int,
    ) -> IterableDataset:
        nonlocal shard_arguments
        shard_arguments = (rank, world_size)
        return dataset

    monkeypatch.setattr("src.data.lambda_audio_dataset.load_dataset", load_stub)
    monkeypatch.setattr(
        "src.data.lambda_audio_dataset.split_dataset_by_node",
        shard_stub,
    )

    load_lambda_audio_dataset("token", "train", 1_000, 16_000, 42, 1, 4)

    assert load_arguments == {
        "dataset_id": DATASET_ID,
        "split": "train",
        "streaming": True,
        "token": "token",
    }
    assert source.selected_columns == ["audio", "text"]
    assert source.cast_sampling_rate == 16_000
    assert source.shuffle_arguments == (1_000, 42)
    assert shard_arguments == (1, 4)
