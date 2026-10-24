from pathlib import Path

import pytest
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
