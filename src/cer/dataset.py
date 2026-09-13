from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

from datasets import Audio, Dataset, load_dataset

DATASET_ID = "google/fleurs"
DATASET_CONFIG = "ja_jp"


def load_test_samples(
    cache_dir: Path,
    sampling_rate: int,
) -> Dataset:
    dataset = cast(
        Dataset,
        load_dataset(
            DATASET_ID,
            DATASET_CONFIG,
            split="test",
            cache_dir=str(cache_dir),
        ),
    )
    dataset = dataset.select_columns(["audio", "transcription"])
    dataset = dataset.cast_column(
        "audio",
        Audio(sampling_rate=sampling_rate, decode=False),
    )
    return dataset


def iterate_test_samples(dataset: Dataset) -> Iterator[tuple[bytes, str]]:
    for row in dataset:
        audio = cast(dict[str, Any], row["audio"])
        yield cast(bytes, audio["bytes"]), cast(str, row["transcription"])
