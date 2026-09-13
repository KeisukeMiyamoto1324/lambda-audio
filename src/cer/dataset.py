from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

from datasets import Audio, Dataset, load_dataset

from src.data.lambda_audio_dataset import DATASET_ID


def load_test_samples(
    token: str,
    percentage: int,
    cache_dir: Path,
    sampling_rate: int,
) -> Dataset:
    dataset = cast(
        Dataset,
        load_dataset(
            DATASET_ID,
            split=f"test[:{percentage}%]",
            token=token,
            cache_dir=str(cache_dir),
        ),
    )
    dataset = dataset.select_columns(["audio", "text"])
    return dataset.cast_column(
        "audio",
        Audio(sampling_rate=sampling_rate, decode=False),
    )


def iterate_test_samples(dataset: Dataset) -> Iterator[tuple[bytes, str]]:
    for row in dataset:
        audio = cast(dict[str, Any], row["audio"])
        yield cast(bytes, audio["bytes"]), cast(str, row["text"])
