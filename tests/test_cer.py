import sys
from pathlib import Path
from typing import Any

import pytest
from datasets import Audio, Dataset

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


def test_load_test_samples_uses_complete_japanese_fleurs_test_split(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = Dataset.from_dict(
        {
            "audio": [{"bytes": b"audio", "path": "audio.flac"}],
            "transcription": ["音声"],
            "unused": [1],
        }
    )
    arguments: dict[str, Any] = {}

    def fake_load_dataset(*args: Any, **kwargs: Any) -> Dataset:
        arguments["args"] = args
        arguments.update(kwargs)
        return source

    monkeypatch.setattr("src.cer.dataset.load_dataset", fake_load_dataset)

    dataset = load_test_samples(tmp_path, 16_000)

    assert arguments["args"] == ("google/fleurs", "ja_jp")
    assert arguments["split"] == "test"
    assert arguments["cache_dir"] == str(tmp_path)
    assert dataset.column_names == ["audio", "transcription"]
    assert isinstance(dataset.features["audio"], Audio)
    assert list(iterate_test_samples(dataset)) == [(b"audio", "音声")]


def test_cer_cli_uses_default_cache_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "model.ckpt"
    checkpoint_path.touch()
    monkeypatch.setattr(sys, "argv", ["cer", str(checkpoint_path)])

    config = parse_arguments()

    assert config.checkpoint_path == checkpoint_path
    assert config.cache_dir == Path("data/cache")
