import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from datasets import Audio, Dataset, IterableDataset

from src.cer.cli import parse_arguments
from src.cer.dataset import iterate_test_samples, load_test_samples
from src.cer.metric import CerMetric
from src.cer.text import normalize_cer_text, should_skip_text


def test_cer_metric_accumulates_corpus_errors() -> None:
    metric = CerMetric()

    metric.update("音声認識", "音声認知")
    metric.update("日本語", "日本")

    assert metric.errors == 2
    assert metric.reference_characters == 7
    assert metric.value == 2 / 7


def test_normalize_cer_text_removes_all_punctuation() -> None:
    assert normalize_cer_text("「今日は、晴れ。」") == "今日は晴れ"


def test_should_skip_text_uses_non_japanese_ratio() -> None:
    assert should_skip_text("日本a1234567")
    assert not should_skip_text("日abc")
    assert not should_skip_text("「日本語です。」")


def test_load_test_samples_streams_only_test_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = Dataset.from_dict(
        {
            "audio": [{"bytes": b"audio", "path": "audio.flac"}],
            "text": ["音声"],
            "unused": [1],
        }
    ).to_iterable_dataset()
    arguments: dict[str, Any] = {}

    def fake_load_dataset(*args: Any, **kwargs: Any) -> IterableDataset:
        arguments.update(kwargs)
        return source

    def fake_get_dataset_config_info(*args: Any, **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(splits={"test": SimpleNamespace(num_examples=4)})

    monkeypatch.setattr("src.cer.dataset.load_dataset", fake_load_dataset)
    monkeypatch.setattr("src.cer.dataset.get_dataset_config_info", fake_get_dataset_config_info)

    dataset, sample_count = load_test_samples("token", 25, tmp_path, 16_000)

    assert arguments["data_files"] == {"test": "test/*.parquet"}
    assert arguments["split"] == "test"
    assert arguments["streaming"] is True
    assert sample_count == 1
    assert dataset.column_names == ["audio", "text"]
    assert isinstance(dataset.features["audio"], Audio)
    assert list(iterate_test_samples(dataset)) == [(b"audio", "音声")]


def test_cer_cli_uses_ten_percent_by_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "model.ckpt"
    checkpoint_path.touch()
    monkeypatch.setattr(sys, "argv", ["cer", str(checkpoint_path)])

    config = parse_arguments()

    assert config.checkpoint_path == checkpoint_path
    assert config.dataset_percentage == 10
