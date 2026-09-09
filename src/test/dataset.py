from pathlib import Path
from typing import cast

from datasets import IterableDataset, load_dataset


def load_local_test_dataset(dataset_dir: Path) -> IterableDataset:
    test_files = sorted(dataset_dir.glob("*.parquet"))
    if not test_files:
        raise FileNotFoundError("The test dataset is not cached.")

    rows = load_dataset(
        "parquet",
        data_files={"test": [str(path) for path in test_files]},
        split="test",
        streaming=True,
    ).select_columns(["id", "audio", "text"])
    return cast(IterableDataset, rows)
