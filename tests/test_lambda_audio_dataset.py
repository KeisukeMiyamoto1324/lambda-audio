from collections.abc import Iterator
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from datasets import Dataset, IterableDataset
from pytest import MonkeyPatch

from src.data.lambda_audio_dataset import (
    DATASET_ID,
    LambdaAudioDataset,
    TrainLambdaAudioDataset,
    list_train_shard_files,
    list_validation_cache_files,
    load_lambda_audio_dataset,
    prepare_validation_dataset,
)


class FakeDataset:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.selected_columns: list[str] = []
        self.cast_sampling_rate: int | None = None
        self.decode_audio: bool | None = None
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
        self.decode_audio = cast(bool, feature.decode)
        return self

    def shuffle(self, buffer_size: int, seed: int) -> "FakeDataset":
        self.shuffle_arguments = (buffer_size, seed)
        return self

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


def test_lambda_audio_dataset_reads_audio_and_text_columns() -> None:
    audio_file = BytesIO()
    sf.write(audio_file, np.array([0.25, -0.5]), 16_000, format="FLAC")
    source = FakeDataset([{"audio": {"bytes": audio_file.getvalue()}, "text": " 音声 ", "unused": 1}])
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


def test_lambda_audio_dataset_skips_invalid_audio() -> None:
    audio_file = BytesIO()
    sf.write(audio_file, np.array([0.25, -0.5]), 16_000, format="FLAC")
    source = FakeDataset(
        [
            {"audio": {"bytes": b"invalid"}, "text": "破損"},
            {"audio": {"bytes": audio_file.getvalue()}, "text": "正常"},
        ]
    )
    dataset = LambdaAudioDataset(cast(IterableDataset, source), 16_000)

    examples = list(dataset)

    assert [example["transcription"] for example in examples] == ["正常"]


def test_load_training_dataset_uses_local_shards(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "src.data.lambda_audio_dataset.list_train_shard_files",
        lambda token: ["train/part-000000.parquet"],
    )

    dataset = load_lambda_audio_dataset(
        "token",
        "train",
        tmp_path,
        1_000,
        16_000,
        42,
        1,
        4,
        8,
    )

    assert isinstance(dataset, TrainLambdaAudioDataset)
    assert dataset.shard_files == ["train/part-000000.parquet"]
    assert dataset.shuffle_buffer_size == 1_000
    assert dataset.global_rank == 1
    assert dataset.world_size == 4


def test_training_dataset_downloads_reads_and_deletes_temporary_shard(
    monkeypatch: MonkeyPatch,
) -> None:
    audio_file = BytesIO()
    sf.write(audio_file, np.array([0.25, -0.5]), 16_000, format="FLAC")
    downloaded_paths: list[Path] = []

    def download_stub(
        *,
        repo_id: str,
        filename: str,
        repo_type: str,
        token: str,
        local_dir: Path,
    ) -> str:
        assert repo_id == DATASET_ID
        assert repo_type == "dataset"
        assert token == "token"
        path = local_dir / filename
        path.parent.mkdir(parents=True)
        table = pa.Table.from_pylist(
            [
                {
                    "audio": {"bytes": b"invalid", "path": "invalid.flac"},
                    "text": "破損",
                },
                {
                    "audio": {"bytes": audio_file.getvalue(), "path": "audio.flac"},
                    "text": " 音声 ",
                },
                {
                    "audio": {"bytes": audio_file.getvalue(), "path": "audio.flac"},
                    "text": " 音声 ",
                },
            ]
        )
        pq.write_table(table, path)
        downloaded_paths.append(path)
        return str(path)

    monkeypatch.setattr(
        "src.data.lambda_audio_dataset.hf_hub_download",
        download_stub,
    )
    dataset = TrainLambdaAudioDataset(
        "token",
        ["train/part-000000.parquet"],
        1,
        16_000,
        42,
        0,
        1,
    )

    iterator = iter(dataset)
    example = next(iterator)
    iterator.close()

    assert example["transcription"] == "音声"
    np.testing.assert_allclose(
        example["audio_array"],
        np.array([0.25, -0.5], dtype=np.float32),
        atol=1e-4,
    )
    assert all(not path.exists() for path in downloaded_paths)


def test_validation_dataset_uses_cached_ten_percent(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = FakeDataset([])
    load_calls: list[dict[str, Any]] = []
    iterable_shards: int | None = None

    class FakeCachedDataset:
        def to_iterable_dataset(self, num_shards: int) -> IterableDataset:
            nonlocal iterable_shards
            iterable_shards = num_shards
            return cast(IterableDataset, source)

    def load_stub(dataset_id: str, **kwargs: Any) -> Dataset:
        load_calls.append({"dataset_id": dataset_id, **kwargs})
        return cast(Dataset, FakeCachedDataset())

    monkeypatch.setattr("src.data.lambda_audio_dataset.load_dataset", load_stub)
    monkeypatch.setattr(
        "src.data.lambda_audio_dataset.list_validation_cache_files",
        lambda token: ["validation/part-000000.parquet"],
    )
    monkeypatch.setattr(
        "src.data.lambda_audio_dataset.split_dataset_by_node",
        lambda dataset, rank, world_size: dataset,
    )

    prepare_validation_dataset("token", tmp_path)
    load_lambda_audio_dataset(
        "token",
        "validation",
        tmp_path,
        1_000,
        16_000,
        42,
        0,
        2,
        4,
    )

    assert load_calls == [
        {
            "dataset_id": DATASET_ID,
            "split": "validation",
            "data_files": {"validation": ["validation/part-000000.parquet"]},
            "streaming": False,
            "token": "token",
            "cache_dir": str(tmp_path),
            "verification_mode": "no_checks",
        },
        {
            "dataset_id": DATASET_ID,
            "split": "validation",
            "data_files": {"validation": ["validation/part-000000.parquet"]},
            "streaming": False,
            "token": "token",
            "cache_dir": str(tmp_path),
            "verification_mode": "no_checks",
        },
    ]
    assert iterable_shards == 8
    assert source.shuffle_arguments is None


def test_validation_cache_selects_only_ten_percent_of_validation_shards(
    monkeypatch: MonkeyPatch,
) -> None:
    class FakeApi:
        def list_repo_files(self, dataset_id: str, repo_type: str) -> list[str]:
            assert dataset_id == DATASET_ID
            assert repo_type == "dataset"
            return [
                *[f"train/part-{index:06d}.parquet" for index in range(1_030)],
                *[f"validation/part-{index:06d}.parquet" for index in range(10)],
            ]

    monkeypatch.setattr(
        "src.data.lambda_audio_dataset.HfApi",
        lambda token: FakeApi(),
    )

    files = list_validation_cache_files("token")

    assert files == ["validation/part-000000.parquet"]


def test_training_shards_select_only_train_parquet_files(
    monkeypatch: MonkeyPatch,
) -> None:
    class FakeApi:
        def list_repo_files(self, dataset_id: str, repo_type: str) -> list[str]:
            assert dataset_id == DATASET_ID
            assert repo_type == "dataset"
            return [
                "train/part-000001.parquet",
                "validation/part-000000.parquet",
                "train/part-000000.parquet",
            ]

    monkeypatch.setattr(
        "src.data.lambda_audio_dataset.HfApi",
        lambda token: FakeApi(),
    )

    files = list_train_shard_files("token")

    assert files == [
        "train/part-000000.parquet",
        "train/part-000001.parquet",
    ]
