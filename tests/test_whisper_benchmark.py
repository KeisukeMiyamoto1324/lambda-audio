from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from rich.console import Console
from rich.progress import Progress
from torch import nn

from src.benchmark import whisper as benchmark_module
from src.benchmark.config import WhisperBenchmarkConfig
from src.shared import inference_display as display_module
from src.shared.samples import InferenceSample
from src.shared.whisper import WhisperTranscriber


class DummyWhisper(nn.Module):
    dtype = torch.float32

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> torch.Tensor:
        assert torch.is_inference_mode_enabled()
        self.calls.append(kwargs)
        return torch.tensor([[1, 4, 2]]).repeat(kwargs["input_features"].shape[0], 1)


class DummyWhisperProcessor:
    def feature_extractor(self, arrays: list[np.ndarray], **kwargs: Any) -> dict[str, torch.Tensor]:
        assert kwargs["sampling_rate"] == 16_000
        assert kwargs["padding"] == "max_length"
        assert kwargs["truncation"] is False
        assert kwargs["return_attention_mask"] is True
        mask = torch.zeros(len(arrays), 3_000, dtype=torch.long)
        for index, array in enumerate(arrays):
            mask[index, : array.size // 160] = 1
        return {"input_features": torch.ones(len(arrays), 128, 3_000), "attention_mask": mask}

    def batch_decode(self, token_ids: torch.Tensor, skip_special_tokens: bool) -> list[str]:
        assert skip_special_tokens
        assert all(ids == [1, 4, 2] for ids in token_ids.tolist())
        return [" 日本語 "] * token_ids.shape[0]


def test_whisper_batches_audio_with_matching_generation_conditions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.wav"
    input_path.touch()
    config = WhisperBenchmarkConfig(input_path=input_path, max_new_tokens=32)
    model = DummyWhisper()
    transcriber = WhisperTranscriber(model, DummyWhisperProcessor(), config, torch.device("cpu"))  # type: ignore[arg-type]
    monkeypatch.setattr("src.shared.transcription.decode_audio", lambda audio, sampling_rate: np.ones(len(audio), dtype=np.float32))

    results = transcriber([b"a" * 160, b"a" * 320])

    assert [result.text for result in results] == ["日本語", "日本語"]
    assert [result.audio_seconds for result in results] == [0.01, 0.02]
    assert results[0].inference_seconds == results[1].inference_seconds
    assert len(model.calls) == 1
    arguments = model.calls[0]
    assert arguments["input_features"].shape == (2, 128, 3_000)
    assert arguments["input_features"].dtype == torch.float32
    assert arguments["attention_mask"].sum(dim=1).tolist() == [1, 2]
    assert arguments["max_new_tokens"] == 32
    assert arguments["num_beams"] == 1
    assert arguments["do_sample"] is False
    assert arguments["language"] == "ja"
    assert arguments["task"] == "transcribe"
    assert arguments["temperature"] == 0.0
    assert arguments["return_timestamps"] is False


def test_whisper_rejects_overlong_clips_without_truncation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.wav"
    input_path.touch()
    config = WhisperBenchmarkConfig(input_path=input_path, max_audio_seconds=0.01)
    model = DummyWhisper()
    transcriber = WhisperTranscriber(model, DummyWhisperProcessor(), config, torch.device("cpu"))  # type: ignore[arg-type]
    monkeypatch.setattr("src.shared.transcription.decode_audio", lambda audio, sampling_rate: np.ones(len(audio), dtype=np.float32))

    with pytest.raises(ValueError, match="0.01 seconds"):
        transcriber([b"a" * 161])

    assert not model.calls


def test_whisper_entrypoint_uses_hf_token_and_shared_progress(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.wav"
    input_path.touch()
    config = WhisperBenchmarkConfig(input_path=input_path, batch_size=2)
    model = DummyWhisper()
    processor = DummyWhisperProcessor()
    loading_arguments: list[dict[str, Any]] = []

    def load_model(model_id: str, **kwargs: Any) -> DummyWhisper:
        loading_arguments.append(kwargs)
        assert model_id == "openai/whisper-large-v3-turbo"
        return model

    def load_processor(model_id: str, **kwargs: Any) -> DummyWhisperProcessor:
        loading_arguments.append(kwargs)
        return processor

    samples = [InferenceSample(audio_name=str(index), audio=b"a" * 160, reference="日本語") for index in range(3)]
    output = StringIO()
    console = Console(file=output, width=100, color_system=None)
    progress = Progress(console=console, disable=True)
    monkeypatch.setattr(benchmark_module, "parse_arguments", lambda: config)
    monkeypatch.setattr(benchmark_module, "load_hf_token", lambda: "test-token")
    monkeypatch.setattr(benchmark_module, "select_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(benchmark_module, "WhisperForConditionalGeneration", SimpleNamespace(from_pretrained=load_model))
    monkeypatch.setattr(benchmark_module, "WhisperProcessor", SimpleNamespace(from_pretrained=load_processor))
    monkeypatch.setattr(benchmark_module, "load_inference_samples", lambda path: samples)
    monkeypatch.setattr(display_module, "Console", lambda: console)
    monkeypatch.setattr(display_module, "Progress", lambda *columns: progress)
    monkeypatch.setattr("src.shared.transcription.decode_audio", lambda audio, sampling_rate: np.ones(len(audio), dtype=np.float32))

    benchmark_module.main()

    assert [call["input_features"].shape[0] for call in model.calls] == [2, 1]
    assert all(arguments["token"] == "test-token" for arguments in loading_arguments)
    assert loading_arguments[1]["dtype"] == torch.float32
    assert not model.training
    assert progress.tasks[0].completed == 3
    assert progress.tasks[0].fields == {"cer": "0.00%", "skipped": 0}
    assert "Whisper large-v3-turbo" in output.getvalue()
    assert "CER: 0.00%" in output.getvalue()
    assert "Elapsed time:" in output.getvalue()


def test_whisper_config_rejects_invalid_batch_size(tmp_path: Path) -> None:
    input_path = tmp_path / "input.wav"
    input_path.touch()
    with pytest.raises(ValidationError):
        WhisperBenchmarkConfig(input_path=input_path, batch_size=0)
