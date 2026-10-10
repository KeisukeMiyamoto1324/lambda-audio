import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.app.app import AppConfig, create_app


class DummyModel:
    def to(self, device: torch.device) -> "DummyModel":
        return self

    def eval(self) -> None:
        pass


class DummyFeatureExtractor:
    pass


class DummyTokenizer:
    pass


def test_create_app_transcribes_when_recording_stops(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "model.ckpt"
    checkpoint_path.touch()
    audio_path = tmp_path / "recording.wav"
    audio_path.write_bytes(b"audio")
    model = DummyModel()
    components = SimpleNamespace(
        model=model,
        feature_extractor=DummyFeatureExtractor(),
        tokenizer=DummyTokenizer(),
    )
    monkeypatch.setattr("src.app.app.load_hf_token", lambda: "token")
    monkeypatch.setattr("src.app.app.select_device", lambda: torch.device("cpu"))
    monkeypatch.setattr("src.app.app.load_model_components", lambda config, token: components)
    monkeypatch.setattr("src.app.app.load_model_checkpoint", lambda loaded_model, path: None)
    monkeypatch.setattr(
        "src.app.app.transcribe",
        lambda loaded_model, feature_extractor, tokenizer, config, device, audio: [
            type(
                "Result",
                (),
                {"text": "テスト"},
            )()
        ],
    )

    app = create_app(AppConfig(checkpoint_path=checkpoint_path))
    result = asyncio.run(
        app.process_api(
            block_fn=0,
            inputs=[
                {
                    "path": str(audio_path),
                    "meta": {"_type": "gradio.FileData"},
                }
            ],
            explicit_call=True,
        )
    )

    assert result["data"] == ["テスト"]
