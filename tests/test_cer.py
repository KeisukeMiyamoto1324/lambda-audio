import sys
from pathlib import Path
from typing import Any

from datasets import Audio, Dataset

from src.cer.cli import parse_arguments
from src.cer.dataset import iterate_test_samples, load_test_samples
from src.cer.metric import CerMetric


def test_cer_metric_accumulates_corpus_errors() -> None:
    metric = CerMetric()

    metric.update("音声認識", "音声認知")
    metric.update("日本語", "日本")

    assert metric.errors == 2
    assert metric.reference_characters == 7
    assert metric.value == 2 / 7


def test_load_test_samples_uses_requested_percentage(monkeypatch: Any, tmp_path: Path) -> None:
    source = Dataset.from_dict(
        {
            "audio": [{"bytes": b"audio", "path": "audio.flac"}],
            "text": ["音声"],
            "unused": [1],
        }
    )
    arguments: dict[str, Any] = {}

    def fake_load_dataset(*args: Any, **kwargs: Any) -> Dataset:
        arguments.update(kwargs)
        return source

    monkeypatch.setattr("src.cer.dataset.load_dataset", fake_load_dataset)

    dataset = load_test_samples("token", 25, tmp_path, 16_000)

    assert arguments["split"] == "test[:25%]"
    assert dataset.column_names == ["audio", "text"]
    assert isinstance(dataset.features["audio"], Audio)
    assert list(iterate_test_samples(dataset)) == [(b"audio", "音声")]


def test_cer_cli_uses_ten_percent_by_default(monkeypatch: Any, tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "model.ckpt"
    checkpoint_path.touch()
    monkeypatch.setattr(sys, "argv", ["cer", str(checkpoint_path)])

    config = parse_arguments()

    assert config.checkpoint_path == checkpoint_path
    assert config.dataset_percentage == 10
