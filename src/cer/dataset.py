from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

from datasets import Audio, Dataset, load_dataset

DATASET_ID = "google/fleurs"
TEST_DATA_FILES = {"test": "parquet-data/ja_jp/test-*.parquet"}


def load_test_samples(
    cache_dir: Path,
    sampling_rate: int,
) -> Dataset:
    dataset = cast(
        Dataset,
        load_dataset(
            DATASET_ID,
            data_files=TEST_DATA_FILES,
            split="test",
            cache_dir=str(cache_dir),
        ),
    )
    dataset = dataset.select_columns(["id", "audio", "transcription"])
    dataset = dataset.cast_column(
        "audio",
        Audio(sampling_rate=sampling_rate, decode=False),
    )
    return dataset


def iterate_test_samples(dataset: Dataset) -> Iterator[tuple[int, bytes, str]]:
    for row in dataset:
        audio = cast(dict[str, Any], row["audio"])
        yield cast(int, row["id"]), cast(bytes, audio["bytes"]), cast(str, row["transcription"])
