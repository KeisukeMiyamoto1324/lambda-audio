import csv
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from datasets import Audio, Dataset

from src.cer.config import CerConfig
from src.cer.dataset import iterate_test_samples, load_test_samples
from src.cer.evaluate import evaluate_cer
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


def test_normalize_cer_text_removes_whitespace() -> None:
    assert normalize_cer_text("日 本\t語\nです") == "日本語です"


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
            "id": [42],
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

    assert arguments["args"] == ("google/fleurs",)
    assert arguments["data_files"] == {"test": "parquet-data/ja_jp/test-*.parquet"}
    assert arguments["split"] == "test"
    assert arguments["cache_dir"] == str(tmp_path)
    assert dataset.column_names == ["id", "audio", "transcription"]
    assert isinstance(dataset.features["audio"], Audio)
    assert list(iterate_test_samples(dataset)) == [(42, b"audio", "音声")]


def test_evaluate_cer_writes_sample_results(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.touch()
    output = tmp_path / "results" / "cer.csv"
    config = CerConfig(checkpoint_path=checkpoint, output_path=output)
    model = SimpleNamespace(to=lambda device: None, eval=lambda: None)
    monkeypatch.setattr("src.cer.evaluate.select_device", lambda: torch.device("cpu"))
    monkeypatch.setattr("src.cer.evaluate.load_model_components", lambda config, token: SimpleNamespace(model=model, feature_extractor=None, tokenizer=None))
    monkeypatch.setattr("src.cer.evaluate.load_model_checkpoint", lambda model, checkpoint: None)
    monkeypatch.setattr("src.cer.evaluate.load_test_samples", lambda cache_dir, sampling_rate: [None, None])
    monkeypatch.setattr("src.cer.evaluate.iterate_test_samples", lambda dataset: iter([(7, b"audio", "「日 本語。」"), (8, b"audio", "12345678")]))
    monkeypatch.setattr("src.cer.evaluate.transcribe", lambda *args: SimpleNamespace(text="日 本", audio_seconds=1.0))

    metric = evaluate_cer(config, "token")

    with output.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert rows == [{"id": "7", "reference": "「日 本語。」", "prediction": "日 本", "cer": str(1 / 3)}]
    assert metric.value == 1 / 3
