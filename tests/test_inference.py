from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from src.inference.config import InferenceConfig
from src.inference.loader import select_device
from src.inference.samples import load_inference_samples
from src.inference.transcription import transcribe
from src.modeling.checkpoint import load_model_checkpoint


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.generation_arguments: dict[str, Any] = {}

    def generate(self, **kwargs: Any) -> torch.LongTensor:
        self.generation_arguments = kwargs
        return torch.tensor([[1, 4, 2]])


class DummyFeatureExtractor:
    def __call__(self, audio: np.ndarray, **kwargs: Any) -> dict[str, torch.Tensor]:
        return {
            "input_values": torch.from_numpy(audio).unsqueeze(0),
            "attention_mask": torch.ones(1, audio.size, dtype=torch.long),
        }


class DummyTokenizer:
    bos_token_id = 1
    eos_token_id = 2

    def batch_decode(
        self,
        token_ids: torch.LongTensor,
        skip_special_tokens: bool,
    ) -> list[str]:
        assert token_ids.tolist() == [[4, 2]]
        assert skip_special_tokens
        return [" transcription "]


@pytest.mark.parametrize(
    ("cuda_available", "mps_available", "expected_device"),
    [
        (True, True, "cuda"),
        (False, True, "mps"),
        (False, False, "cpu"),
    ],
)
def test_select_device_uses_requested_priority(
    monkeypatch: pytest.MonkeyPatch,
    cuda_available: bool,
    mps_available: bool,
    expected_device: str,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps_available)

    assert select_device().type == expected_device


def test_load_checkpoint_weights_removes_lightning_prefix(tmp_path: Path) -> None:
    source_model = nn.Linear(2, 2)
    target_model = nn.Linear(2, 2)
    checkpoint_path = tmp_path / "model.ckpt"
    state_dict = {
        f"model.{name}": parameter
        for name, parameter in source_model.state_dict().items()
    }
    torch.save({"state_dict": state_dict}, checkpoint_path)

    load_model_checkpoint(target_model, checkpoint_path)

    for source_parameter, target_parameter in zip(
        source_model.parameters(),
        target_model.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(source_parameter, target_parameter)


def test_transcribe_generates_from_bos(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "audio.wav"
    checkpoint_path = tmp_path / "model.ckpt"
    checkpoint_path.touch()
    config = InferenceConfig(
        checkpoint_path=checkpoint_path,
        input_dir=tmp_path,
    )
    model = DummyModel()
    monkeypatch.setattr(
        "src.inference.transcription.librosa.load",
        lambda *args, **kwargs: (np.ones(160, dtype=np.float32), 16_000),
    )

    result = transcribe(
        model,  # type: ignore[arg-type]
        DummyFeatureExtractor(),  # type: ignore[arg-type]
        DummyTokenizer(),  # type: ignore[arg-type]
        config,
        torch.device("cpu"),
        audio_path,
    )

    assert result.text == "transcription"
    assert result.audio_seconds == 0.01
    assert model.generation_arguments["input_ids"].tolist() == [[1]]
    assert model.generation_arguments["max_new_tokens"] == 256
    assert not model.generation_arguments["audio_padding_mask"].any()


def test_transcribe_rejects_audio_longer_than_30_seconds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "model.ckpt"
    checkpoint_path.touch()
    config = InferenceConfig(
        checkpoint_path=checkpoint_path,
        input_dir=tmp_path,
    )
    monkeypatch.setattr(
        "src.inference.transcription.librosa.load",
        lambda *args, **kwargs: (
            np.ones(16_000 * 30 + 1, dtype=np.float32),
            16_000,
        ),
    )

    with pytest.raises(ValueError, match="30 seconds or shorter"):
        transcribe(
            DummyModel(),  # type: ignore[arg-type]
            DummyFeatureExtractor(),  # type: ignore[arg-type]
            DummyTokenizer(),  # type: ignore[arg-type]
            config,
            torch.device("cpu"),
            tmp_path / "audio.wav",
        )


def test_load_inference_samples_pairs_audio_and_reference(tmp_path: Path) -> None:
    (tmp_path / "b.flac").touch()
    (tmp_path / "b.txt").write_text("二\n", encoding="utf-8")
    (tmp_path / "a.flac").touch()
    (tmp_path / "a.txt").write_text("一\n", encoding="utf-8")

    samples = load_inference_samples(tmp_path)

    assert [sample.audio_path.name for sample in samples] == ["a.flac", "b.flac"]
    assert [sample.reference for sample in samples] == ["一", "二"]


def test_load_inference_samples_rejects_missing_reference(tmp_path: Path) -> None:
    (tmp_path / "audio.flac").touch()

    with pytest.raises(FileNotFoundError, match="audio.txt"):
        load_inference_samples(tmp_path)
