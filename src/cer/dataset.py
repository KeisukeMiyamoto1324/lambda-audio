from collections.abc import Iterator
from math import ceil
from pathlib import Path
from typing import Any, cast

from datasets import Audio, IterableDataset, get_dataset_config_info, load_dataset

from src.data.lambda_audio_dataset import DATASET_ID

TEST_DATA_FILES = {"test": "test/*.parquet"}


def load_test_samples(
    token: str,
    percentage: int,
    cache_dir: Path,
    sampling_rate: int,
) -> tuple[IterableDataset, int]:
    dataset_info = get_dataset_config_info(
        DATASET_ID,
        data_files=TEST_DATA_FILES,
        token=token,
    )
    total_examples = cast(int, dataset_info.splits["test"].num_examples)
    sample_count = ceil(total_examples * percentage / 100)
    dataset = cast(
        IterableDataset,
        load_dataset(
            DATASET_ID,
            data_files=TEST_DATA_FILES,
            split="test",
            streaming=True,
            token=token,
            cache_dir=str(cache_dir),
        ),
    )
    dataset = dataset.select_columns(["audio", "text"])
    dataset = dataset.cast_column(
        "audio",
        Audio(sampling_rate=sampling_rate, decode=False),
    )
    return dataset.take(sample_count), sample_count


def iterate_test_samples(dataset: IterableDataset) -> Iterator[tuple[bytes, str]]:
    for row in dataset:
        audio = cast(dict[str, Any], row["audio"])
        yield cast(bytes, audio["bytes"]), cast(str, row["text"])
