from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from pydantic import ValidationError

from src.benchmark.cli import parse_arguments
from src.benchmark.config import BenchmarkConfig
from src.benchmark.registry import MODEL_LOADERS


def test_lambda_audio_requires_checkpoint_but_whisper_does_not(tmp_path: Path) -> None:
    input_path = tmp_path / "audio.wav"
    input_path.touch()
    with pytest.raises(ValidationError, match="requires --checkpoint"):
        BenchmarkConfig(model="lambda-audio", input_path=input_path)
    config = BenchmarkConfig(model="whisper", input_path=input_path)
    assert config.checkpoint_path is None


def test_cli_accepts_newly_registered_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    input_path = tmp_path / "audio.wav"
    input_path.touch()
    monkeypatch.setitem(MODEL_LOADERS, "custom-model", lambda config, device, token: lambda audio: [])
    monkeypatch.setattr("sys.argv", ["benchmark", "--model", "custom-model", "--input", str(input_path), "--batch-size", "2"])

    config = parse_arguments()

    assert config.model == "custom-model"
    assert config.input_path == input_path
    assert config.batch_size == 2


@pytest.mark.parametrize(("arguments", "expected"), [([], True), (["--show-table"], True), (["--no-show-table"], False)])
def test_cli_controls_result_table(arguments: list[str], expected: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "audio.wav"
    path.touch()
    monkeypatch.setattr("sys.argv", ["benchmark", "--model", "whisper", "--input", str(path), *arguments])
    assert parse_arguments().show_table is expected


@pytest.mark.parametrize("language", ["ja", "en", "ar", "zh", "fr", "de", "ko", "pt", "es"])
def test_cli_accepts_output_language(language: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "model.ckpt"
    path.touch()
    monkeypatch.setattr("sys.argv", ["benchmark", "--model", "lambda-audio", "--checkpoint", str(path), "--input", str(path), "--output-language", language])
    assert parse_arguments().output_language == language


def test_cli_rejects_unsupported_output_language(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "model.ckpt"
    path.touch()
    monkeypatch.setattr("sys.argv", ["benchmark", "--model", "lambda-audio", "--checkpoint", str(path), "--input", str(path), "--output-language", "invalid"])
    with pytest.raises(SystemExit):
        parse_arguments()


def test_lambda_audio_loader_passes_output_language(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.benchmark.models import lambda_audio

    path = tmp_path / "model.ckpt"
    path.touch()
    config = BenchmarkConfig(checkpoint_path=path, input_path=path, output_language="en")
    model = torch.nn.Linear(1, 1)
    components = SimpleNamespace(model=model, feature_extractor=object(), tokenizer=object())
    calls: list[dict[str, Any]] = []

    def transcribe_stub(*args: Any, **kwargs: Any) -> list[Any]:
        calls.append(kwargs)
        return []

    monkeypatch.setattr(lambda_audio, "load_model_components", lambda config, token: components)
    monkeypatch.setattr(lambda_audio, "load_model_checkpoint", lambda model, checkpoint: None)
    monkeypatch.setattr(lambda_audio, "transcribe", transcribe_stub)
    transcriber = lambda_audio.load_model(config, torch.device("cpu"), "token")
    transcriber([b"audio"])

    assert calls == [{"output_language": "en"}]
