from pathlib import Path
from unittest.mock import Mock

import pytest
from datasets import IterableDataset

from src.test.dataset import load_local_test_dataset


def test_load_local_test_dataset_uses_cached_parquet_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_file = tmp_path / "part-000000.parquet"
    second_file = tmp_path / "part-000001.parquet"
    first_file.touch()
    second_file.touch()
    selected = IterableDataset.from_generator(lambda: iter([]))
    dataset = Mock()
    dataset.select_columns.return_value = selected
    load = Mock(return_value=dataset)
    monkeypatch.setattr("src.test.dataset.load_dataset", load)

    result = load_local_test_dataset(tmp_path)

    assert result is selected
    load.assert_called_once_with(
        "parquet",
        data_files={"test": [str(first_file), str(second_file)]},
        split="test",
        streaming=True,
    )
    dataset.select_columns.assert_called_once_with(["id", "audio", "text"])


def test_load_local_test_dataset_requires_cached_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="test dataset is not cached"):
        load_local_test_dataset(tmp_path)
