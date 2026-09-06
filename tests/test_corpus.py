from pathlib import Path
from unittest.mock import Mock

import numpy as np
import torch
from datasets import IterableDataset
from pytest import MonkeyPatch

from src.data.corpus import (
    DATASET_ID,
    LambdaAudioDataset,
    load_lambda_audio_datasets,
    prepare_validation_data,
)


def test_lambda_audio_dataset_reads_audio_and_text() -> None:
    samples = Mock()
    samples.data = torch.ones((1, 4), dtype=torch.float32)
    decoder = Mock()
    decoder.get_all_samples.return_value = samples
    source = IterableDataset.from_generator(
        lambda: iter([{"audio": decoder, "text": "（音声）"}])
    )

    example = next(iter(LambdaAudioDataset(source)))

    np.testing.assert_array_equal(
        example["audio_array"],
        np.ones(4, dtype=np.float32),
    )
    assert example["transcription"] == "（音声）"


def test_prepare_validation_data_uses_local_cache(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    download = Mock()
    monkeypatch.setattr("src.data.corpus.snapshot_download", download)

    path = prepare_validation_data("token", tmp_path)

    assert path == tmp_path / "validation"
    download.assert_called_once_with(
        repo_id=DATASET_ID,
        repo_type="dataset",
        token="token",
        allow_patterns="validation/*.parquet",
        local_dir=tmp_path,
    )


def test_load_datasets_uses_remote_train_and_local_validation(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    validation_dir = tmp_path / "validation"
    validation_dir.mkdir()
    (validation_dir / "part-000000.parquet").touch()
    datasets = [
        IterableDataset.from_generator(lambda: iter([])),
        IterableDataset.from_generator(lambda: iter([])),
    ]
    load = Mock(side_effect=datasets)
    monkeypatch.setattr("src.data.corpus.load_dataset", load)
    monkeypatch.setattr(
        "src.data.corpus.split_dataset_by_node",
        lambda dataset, rank, world_size: dataset,
    )

    training, validation = load_lambda_audio_datasets(
        "token",
        tmp_path,
        shuffle_buffer_size=10,
        seed=42,
        global_rank=0,
        world_size=1,
    )

    assert isinstance(training, LambdaAudioDataset)
    assert isinstance(validation, LambdaAudioDataset)
    assert load.call_args_list[0].args == (DATASET_ID,)
    assert load.call_args_list[0].kwargs == {
        "split": "train",
        "streaming": True,
        "token": "token",
    }
    assert load.call_args_list[1].args == ("parquet",)
    assert load.call_args_list[1].kwargs == {
        "data_files": {"validation": [str(validation_dir / "part-000000.parquet")]},
        "split": "validation",
        "streaming": True,
    }
