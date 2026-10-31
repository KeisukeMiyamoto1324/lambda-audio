import os
import shutil
import subprocess
import sys
import wave
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_audio.stt.models.whisper")

import mlx.core as mx  # noqa: E402

from whisper.config import MODEL_ID, InferenceConfig  # noqa: E402
from whisper.inference import WhisperEngine, load_engine  # noqa: E402


def test_whisper_decodes_audio_and_uses_comparison_settings() -> None:
    buffer = BytesIO()
    values = np.arange(160, dtype=np.int16)
    with wave.open(buffer, "wb") as output:
        output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        output.writeframes(values.tobytes())
    calls: list[dict[str, Any]] = []

    def generate(audio: np.ndarray, **kwargs: Any) -> SimpleNamespace:
        np.testing.assert_allclose(audio, values.astype(np.float32) / 32768)
        calls.append(kwargs)
        return SimpleNamespace(text=" 日本語の音声 ")

    engine = WhisperEngine(SimpleNamespace(generate=generate), InferenceConfig(max_new_tokens=123))  # type: ignore[arg-type]
    result = engine([buffer.getvalue()])[0]
    assert result.text == "日本語の音声"
    assert result.audio_seconds == 0.01
    assert result.inference_seconds > 0
    assert calls == [
        {
            "language": "ja",
            "task": "transcribe",
            "sample_len": 123,
            "temperature": 0.0,
            "without_timestamps": True,
            "word_timestamps": False,
            "condition_on_previous_text": False,
            "compression_ratio_threshold": None,
            "logprob_threshold": None,
            "no_speech_threshold": None,
            "verbose": None,
        }
    ]
    with pytest.raises(ValueError, match="batch size 1"):
        engine([buffer.getvalue(), buffer.getvalue()])
    with pytest.raises(ValueError, match="batch size 1"):
        engine([])


def test_whisper_loading_uses_explicit_token_and_local_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "multilingual.tiktoken").touch()
    model = SimpleNamespace(is_multilingual=True, num_languages=100, eval=lambda: None, parameters=lambda: {})
    events: list[str] = []

    def download(repo_id: str, *, allow_patterns: list[str], token: str) -> str:
        assert repo_id == MODEL_ID and token == "explicit-token"
        assert set(allow_patterns) == {"config.json", "model.safetensors", "multilingual.tiktoken"}
        events.append("download")
        return str(tmp_path)

    def load(path: str, *, dtype: mx.Dtype) -> Any:
        assert path == str(tmp_path) and dtype == mx.float16
        events.append("load")
        return model

    def tokenizer(multilingual: bool, **kwargs: Any) -> None:
        assert multilingual
        assert kwargs == {"num_languages": 100, "language": "ja", "task": "transcribe", "model_path": tmp_path}
        events.append("tokenizer")

    monkeypatch.setattr("whisper.inference.snapshot_download", download)
    monkeypatch.setattr("whisper.inference.Model.from_pretrained", load)
    monkeypatch.setattr("whisper.inference.get_tokenizer", tokenizer)
    engine = load_engine(InferenceConfig(), "explicit-token")
    assert engine.model is model
    assert events == ["download", "load", "tokenizer"]
    assert mx.get_cache_memory() == 0
    (tmp_path / "multilingual.tiktoken").unlink()
    with pytest.raises(FileNotFoundError, match="multilingual.tiktoken"):
        load_engine(InferenceConfig(), "explicit-token")
    assert events.count("load") == 1


def test_whisper_cli_runs_from_an_independent_copy(tmp_path: Path) -> None:
    shutil.copytree(Path(__file__).resolve().parents[1] / "whisper", tmp_path / "whisper", ignore=shutil.ignore_patterns("__pycache__"))
    code = """
import importlib.abc
import runpy
import sys

class BlockProjectImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: object = None, target: object = None) -> None:
        if fullname.split(".")[0] in {"src", "torch", "transformers", "triton", "cut_cross_entropy"}:
            raise AssertionError(f"Forbidden dependency: {fullname}")
        if fullname in {"mlx.model", "mlx.inference", "mlx.evaluation", "mlx.config"}:
            raise AssertionError(f"Forbidden project dependency: {fullname}")

sys.meta_path.insert(0, BlockProjectImports())
sys.argv = ["benchmark", "--help"]
runpy.run_module("whisper.benchmark", run_name="__main__")
"""
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=environment, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--max-new-tokens" in result.stdout


@pytest.mark.parametrize("arguments", [["--batch-size", "2"], ["--max-new-tokens", "0"], ["--max-new-tokens", "445"]])
def test_whisper_cli_rejects_unsupported_settings(arguments: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    from whisper.benchmark import main

    monkeypatch.setattr(sys, "argv", ["benchmark", "--input", "unused.wav", *arguments])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
