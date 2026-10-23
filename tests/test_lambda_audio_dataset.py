import random
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf
from datasets import Dataset, IterableDataset
from pytest import MonkeyPatch

from src.data.lambda_audio_dataset import (
    DATASET_COLUMNS,
    DATASET_ID,
    LambdaAudioDataset,
    TrainLambdaAudioDataset,
    _read_examples,
    list_shard_files,
    list_validation_cache_files,
    load_lambda_audio_dataset,
    prepare_validation_dataset,
)
from src.shared.languages import OUTPUT_LANGUAGES


class FakeDataset:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.selected_columns: list[str] = []
        self.cast_sampling_rate: int | None = None
        self.decode_audio: bool | None = None
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

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


def test_lambda_audio_dataset_reads_audio_and_text_columns() -> None:
    audio_file = BytesIO()
    sf.write(audio_file, np.array([0.25, -0.5]), 16_000, format="FLAC")
    source = FakeDataset([{"audio": {"bytes": audio_file.getvalue()}, **dict.fromkeys(OUTPUT_LANGUAGES, " 音声 "), "id": "audio", "unused": 1}])
    dataset = LambdaAudioDataset(cast(IterableDataset, source), 16_000, 42)

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
            {"audio": {"bytes": b"invalid"}, **dict.fromkeys(OUTPUT_LANGUAGES, "破損"), "id": "invalid"},
            {"audio": {"bytes": audio_file.getvalue()}, **dict.fromkeys(OUTPUT_LANGUAGES, "正常"), "id": "valid"},
        ]
    )
    dataset = LambdaAudioDataset(cast(IterableDataset, source), 16_000, 42)

    examples = list(dataset)

    assert [example["transcription"] for example in examples] == ["正常"]


def test_load_training_dataset_uses_local_shards(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "src.data.lambda_audio_dataset.list_shard_files",
        lambda token, split: ["train/part-000000.parquet"],
    )

    dataset = load_lambda_audio_dataset(
        "token",
        "train",
        tmp_path,
        16_000,
        42,
        1,
        4,
        8,
    )

    assert isinstance(dataset, TrainLambdaAudioDataset)
    assert dataset.shard_files == ["train/part-000000.parquet"]
    assert dataset.global_rank == 1
    assert dataset.world_size == 4


@pytest.mark.parametrize("consume_all", [False, True])
def test_training_dataset_downloads_reads_and_deletes_temporary_shard(
    monkeypatch: MonkeyPatch,
    consume_all: bool,
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
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(
            [
                {
                    "audio": {"bytes": b"invalid", "path": "invalid.flac"},
                    **dict.fromkeys(OUTPUT_LANGUAGES, "破損"),
                    "id": "invalid",
                },
                {
                    "audio": {"bytes": audio_file.getvalue(), "path": "audio.flac"},
                    **dict.fromkeys(OUTPUT_LANGUAGES, " 音声 "),
                    "id": "audio",
                },
                {
                    "audio": {"bytes": audio_file.getvalue(), "path": "audio.flac"},
                    **dict.fromkeys(OUTPUT_LANGUAGES, " 続き "),
                    "id": "next",
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
        ["train/part-000000.parquet", "train/part-000001.parquet"],
        16_000,
        42,
        0,
        1,
    )

    iterator = iter(dataset)
    example = next(iterator)
    if consume_all:
        remaining = list(iterator)
        assert [item["transcription"] for item in remaining] == ["続き", "音声", "続き"]
    iterator.close()

    assert example["transcription"] == "音声"
    np.testing.assert_allclose(
        example["audio_array"],
        np.array([0.25, -0.5], dtype=np.float32),
        atol=1e-4,
    )
    assert len(downloaded_paths) == 2
    assert all(not path.parent.parent.exists() for path in downloaded_paths)


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
    assert source.selected_columns == DATASET_COLUMNS


def test_validation_cache_selects_fixed_ten_percent_of_validation_shards(
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

    assert files == ["validation/part-000001.parquet"]


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

    files = list_shard_files("token", "train")

    assert files == [
        "train/part-000000.parquet",
        "train/part-000001.parquet",
    ]


@pytest.mark.parametrize("sampling_rate, channels", [(8_000, 1), (16_000, 2)])
def test_dataset_skips_unsupported_audio_format(sampling_rate: int, channels: int) -> None:
    audio_file = BytesIO()
    sf.write(audio_file, np.zeros((32, channels)), sampling_rate, format="FLAC")
    source = FakeDataset([{"audio": {"bytes": audio_file.getvalue()}, **dict.fromkeys(OUTPUT_LANGUAGES, "音声"), "id": "audio"}])
    dataset = LambdaAudioDataset(cast(IterableDataset, source), 16_000, 42)

    assert list(dataset) == []


@pytest.mark.parametrize("fail_download", [False, True])
def test_training_dataset_cleans_up_after_shard_failure(monkeypatch: MonkeyPatch, fail_download: bool) -> None:
    directories: list[Path] = []

    def download_stub(self: TrainLambdaAudioDataset, shard_file: str, directory: Path) -> Path:
        directories.append(directory)
        if fail_download:
            raise OSError("Download failed")
        path = directory / "invalid.parquet"
        path.write_bytes(b"invalid parquet")
        return path

    monkeypatch.setattr(TrainLambdaAudioDataset, "_download_shard", download_stub)
    dataset = TrainLambdaAudioDataset("token", ["train/part.parquet"], 16_000, 42, 0, 1)
    with pytest.raises((OSError, pa.ArrowInvalid)):
        list(dataset)

    assert len(directories) == 1
    assert not directories[0].exists()


def test_training_workers_partition_shards_and_change_order_each_epoch(monkeypatch: MonkeyPatch) -> None:
    from types import SimpleNamespace

    from src.data.types import TranscribedAudioExample

    def download_stub(self: TrainLambdaAudioDataset, shard_file: str, directory: Path) -> Path:
        return directory / shard_file

    def read_stub(self: TrainLambdaAudioDataset, shard_path: Path, generator: random.Random) -> Iterator[TranscribedAudioExample]:
        yield {"audio_array": np.zeros(1, dtype=np.float32), "transcription": shard_path.name, "output_language": "ja"}

    monkeypatch.setattr(TrainLambdaAudioDataset, "_download_shard", download_stub)
    monkeypatch.setattr(TrainLambdaAudioDataset, "_read_shard", read_stub)
    shards = [f"part-{index}.parquet" for index in range(11)]
    orders: list[list[str]] = []
    for epoch in (0, 0, 1):
        observed: list[str] = []
        for rank in range(2):
            dataset = TrainLambdaAudioDataset("token", shards, 16_000, 42, rank, 2)
            dataset.set_epoch(epoch)
            for worker_id in range(2):
                worker = SimpleNamespace(id=worker_id, num_workers=2)
                monkeypatch.setattr("src.data.lambda_audio_dataset.get_worker_info", lambda: worker)
                observed.extend(example["transcription"] for example in dataset)
        assert sorted(observed) == sorted(shards)
        orders.append(observed)

    assert orders[0] == orders[1]
    assert orders[0] != orders[2]
    assert shards == [f"part-{index}.parquet" for index in range(11)]


def test_validation_language_is_fixed_per_row_across_order_and_epoch() -> None:
    audio_file = BytesIO()
    sf.write(audio_file, np.ones(4), 16_000, format="FLAC")
    rows = [{"id": str(index), "audio": {"bytes": audio_file.getvalue()}, **{language: f"{index}:{language}" for language in OUTPUT_LANGUAGES}} for index in range(40)]
    dataset = LambdaAudioDataset(cast(IterableDataset, FakeDataset(rows)), 16_000, 42)
    first = list(dataset)
    dataset.set_epoch(7)
    repeated = list(dataset)
    reversed_rows = list(LambdaAudioDataset(cast(IterableDataset, FakeDataset(rows[::-1])), 16_000, 42))

    assert [example["transcription"] for example in first] == [example["transcription"] for example in repeated]
    assert [example["transcription"] for example in first] == [example["transcription"] for example in reversed_rows[::-1]]
    assert 1 < len({example["output_language"] for example in first}) <= len(OUTPUT_LANGUAGES)
    assert all(example["transcription"] == f"{index}:{example['output_language']}" for index, example in enumerate(first))


def test_training_language_sampling_is_reproducible_and_changes_each_epoch(monkeypatch: MonkeyPatch) -> None:
    audio_file = BytesIO()
    sf.write(audio_file, np.ones(4), 16_000, format="FLAC")
    rows = [{"id": str(index), "audio": {"bytes": audio_file.getvalue()}, **{language: language for language in OUTPUT_LANGUAGES}} for index in range(40)]

    def download_stub(self: TrainLambdaAudioDataset, shard_file: str, directory: Path) -> Path:
        return directory / shard_file

    def read_stub(self: TrainLambdaAudioDataset, shard_path: Path, generator: random.Random) -> Iterator[dict[str, Any]]:
        yield from _read_examples(rows, self.sampling_rate, shard_path.name, self.seed, generator)

    monkeypatch.setattr(TrainLambdaAudioDataset, "_download_shard", download_stub)
    monkeypatch.setattr(TrainLambdaAudioDataset, "_read_shard", read_stub)
    dataset = TrainLambdaAudioDataset("token", ["part.parquet"], 16_000, 42, 0, 1)
    sequences: list[list[str]] = []
    for epoch in (0, 0, 1):
        dataset.set_epoch(epoch)
        examples = list(dataset)
        assert len(examples) == len(rows)
        assert all(example["transcription"] == example["output_language"] for example in examples)
        sequences.append([example["output_language"] for example in examples])

    assert sequences[0] == sequences[1]
    assert sequences[0] != sequences[2]
    assert 1 < len(set(sequences[0])) <= len(OUTPUT_LANGUAGES)
