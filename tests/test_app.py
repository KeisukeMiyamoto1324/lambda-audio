import asyncio
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
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
    sf.write(audio_path, np.array([0.1, -0.2, 0.5], dtype=np.float32), 16_000)
    model = DummyModel()
    transcribed_audio = b""
    components = SimpleNamespace(
        model=model,
        feature_extractor=DummyFeatureExtractor(),
        tokenizer=DummyTokenizer(),
    )
    monkeypatch.setattr("src.app.app.load_hf_token", lambda: "token")
    monkeypatch.setattr("src.app.app.select_device", lambda: torch.device("cpu"))
    monkeypatch.setattr("src.app.app.load_model_components", lambda config, token: components)
    monkeypatch.setattr("src.app.app.load_model_checkpoint", lambda loaded_model, path: None)

    def transcribe_audio(
        loaded_model: DummyModel,
        feature_extractor: DummyFeatureExtractor,
        tokenizer: DummyTokenizer,
        config: AppConfig,
        device: torch.device,
        audio: bytes,
    ) -> object:
        nonlocal transcribed_audio
        transcribed_audio = audio
        return type(
            "Result",
            (),
            {"text": "テスト"},
        )()

    monkeypatch.setattr("src.app.app.transcribe", transcribe_audio)

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

    amplified_audio, _ = sf.read(BytesIO(transcribed_audio), dtype="float32")

    assert result["data"][1] == "テスト"
    assert result["data"][0]["path"].endswith("_amplified.wav")
    assert amplified_audio == pytest.approx([0.3, -0.6, 1.0], abs=1e-4)
