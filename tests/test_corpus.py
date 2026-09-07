import logging
import os
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import torch
from datasets import IterableDataset
from pytest import LogCaptureFixture, MonkeyPatch

from src.data.corpus import (
    DATASET_ID,
    LambdaAudioDataset,
    load_lambda_audio_datasets,
    prepare_dataset,
)


def test_lambda_audio_dataset_reads_audio_and_text() -> None:
    samples = Mock()
    samples.data = torch.ones((1, 4), dtype=torch.float32)
    decoder = Mock()
    decoder.get_all_samples.return_value = samples
    source = IterableDataset.from_generator(lambda: iter([{"id": "audio-1", "audio": decoder, "text": "（音声）"}]))

    example = next(iter(LambdaAudioDataset(source)))

    np.testing.assert_array_equal(
        example["audio_array"],
        np.ones(4, dtype=np.float32),
    )
    assert example["transcription"] == "（音声）"


def test_lambda_audio_dataset_skips_and_logs_undecodable_audio(
    caplog: LogCaptureFixture,
) -> None:
    invalid_decoder = Mock()
    invalid_decoder.get_all_samples.side_effect = RuntimeError("Invalid audio")
    samples = Mock()
    samples.data = torch.ones((1, 4), dtype=torch.float32)
    valid_decoder = Mock()
    valid_decoder.get_all_samples.return_value = samples
    source = IterableDataset.from_generator(
        lambda: iter(
            [
                {"id": "invalid-audio", "audio": invalid_decoder, "text": "失敗"},
                {"id": "valid-audio", "audio": valid_decoder, "text": "成功"},
            ]
        )
    )

    with caplog.at_level(logging.ERROR, logger="src.data.corpus"):
        examples = list(LambdaAudioDataset(source))

    assert len(examples) == 1
    assert examples[0]["transcription"] == "成功"
    assert "id=invalid-audio" in caplog.text
    assert "Invalid audio" in caplog.text


def test_prepare_dataset_downloads_entire_repository(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    download = Mock()
    monkeypatch.delenv("HF_XET_HIGH_PERFORMANCE", raising=False)
    monkeypatch.setattr("src.data.corpus.snapshot_download", download)

    path = prepare_dataset("token", tmp_path)

    assert path == tmp_path
    assert os.environ["HF_XET_HIGH_PERFORMANCE"] == "1"
    download.assert_called_once_with(
        repo_id=DATASET_ID,
        repo_type="dataset",
        token="token",
        local_dir=tmp_path,
        max_workers=32,
    )


def test_load_datasets_uses_local_train_and_validation(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    training_dir = tmp_path / "train"
    training_dir.mkdir()
    (training_dir / "part-000000.parquet").touch()
    validation_dir = tmp_path / "validation"
    validation_dir.mkdir()
    (validation_dir / "part-000000.parquet").touch()
    (validation_dir / "part-000001.parquet").touch()
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
        tmp_path,
        global_rank=0,
        world_size=1,
    )

    assert isinstance(training, LambdaAudioDataset)
    assert isinstance(validation, LambdaAudioDataset)
    assert load.call_args_list[0].args == ("parquet",)
    assert load.call_args_list[0].kwargs == {
        "data_files": {"train": [str(training_dir / "part-000000.parquet")]},
        "split": "train",
        "streaming": True,
    }
    assert load.call_args_list[1].args == ("parquet",)
    assert load.call_args_list[1].kwargs == {
        "data_files": {"validation": [str(validation_dir / "part-000000.parquet")]},
        "split": "validation",
        "streaming": True,
    }
