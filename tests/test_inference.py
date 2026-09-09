from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch import nn

from src.inference.config import InferenceConfig
from src.inference.loader import select_device
from src.inference.samples import load_inference_samples
from src.inference.transcription import transcribe, transcribe_audio_batch
from src.modeling.checkpoint import load_model_checkpoint


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.generation_arguments: dict[str, Any] = {}

    def generate(self, **kwargs: Any) -> torch.LongTensor:
        self.generation_arguments = kwargs
        return torch.tensor([[1, 4, 2]]).repeat(kwargs["input_ids"].shape[0], 1)


class DummyFeatureExtractor:
    def __call__(
        self,
        audio: np.ndarray | list[np.ndarray],
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        audio_arrays = audio if isinstance(audio, list) else [audio]
        lengths = [audio_array.size for audio_array in audio_arrays]
        input_values = torch.nn.utils.rnn.pad_sequence(
            [torch.from_numpy(audio_array) for audio_array in audio_arrays],
            batch_first=True,
        )
        attention_mask = torch.zeros_like(input_values, dtype=torch.long)
        for index, length in enumerate(lengths):
            attention_mask[index, :length] = 1
        return {
            "input_values": input_values,
            "attention_mask": attention_mask,
        }


class DummyTokenizer:
    bos_token_id = 1
    eos_token_id = 2

    def batch_decode(
        self,
        token_ids: torch.LongTensor,
        skip_special_tokens: bool,
    ) -> list[str]:
        assert all(sequence == [4, 2] for sequence in token_ids.tolist())
        assert skip_special_tokens
        return [" transcription "] * token_ids.shape[0]


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
    state_dict = {f"model.{name}": parameter for name, parameter in source_model.state_dict().items()}
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
    checkpoint_path = tmp_path / "model.ckpt"
    checkpoint_path.touch()
    config = InferenceConfig(
        checkpoint_path=checkpoint_path,
        input_path=checkpoint_path,
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
        b"audio",
    )

    assert result.text == "transcription"
    assert result.audio_seconds == 0.01
    assert model.generation_arguments["input_ids"].tolist() == [[1]]
    assert model.generation_arguments["max_new_tokens"] == 256
    assert not model.generation_arguments["audio_padding_mask"].any()


def test_transcribe_audio_batch_pads_and_generates_together(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "model.ckpt"
    checkpoint_path.touch()
    config = InferenceConfig(
        checkpoint_path=checkpoint_path,
        input_path=checkpoint_path,
    )
    model = DummyModel()

    results = transcribe_audio_batch(
        model,  # type: ignore[arg-type]
        DummyFeatureExtractor(),  # type: ignore[arg-type]
        DummyTokenizer(),  # type: ignore[arg-type]
        config,
        torch.device("cpu"),
        [np.ones(160, dtype=np.float32), np.ones(80, dtype=np.float32)],
        [16_000, 16_000],
    )

    assert [result.text for result in results] == ["transcription", "transcription"]
    assert model.generation_arguments["input_ids"].shape == (2, 1)
    assert model.generation_arguments["audio_padding_mask"][1, 80:].all()


def test_transcribe_rejects_audio_longer_than_30_seconds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "model.ckpt"
    checkpoint_path.touch()
    config = InferenceConfig(
        checkpoint_path=checkpoint_path,
        input_path=checkpoint_path,
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
            b"audio",
        )


def test_load_inference_samples_reads_parquet(tmp_path: Path) -> None:
    input_path = tmp_path / "data.parquet"
    table = pa.table(
        {
            "audio": [
                {"bytes": b"audio-a", "path": "a.flac"},
                {"bytes": b"audio-b", "path": "b.flac"},
            ],
            "text": ["一", "二"],
        }
    )
    pq.write_table(table, input_path)

    samples = load_inference_samples(input_path)

    assert [sample.audio_name for sample in samples] == ["a.flac", "b.flac"]
    assert [sample.audio for sample in samples] == [b"audio-a", b"audio-b"]
    assert [sample.reference for sample in samples] == ["一", "二"]


def test_load_inference_samples_reads_audio_file(tmp_path: Path) -> None:
    input_path = tmp_path / "sample.wav"
    input_path.write_bytes(b"audio")

    samples = load_inference_samples(input_path)

    assert len(samples) == 1
    assert samples[0].audio_name == "sample.wav"
    assert samples[0].audio == b"audio"
    assert samples[0].reference == ""


def test_load_inference_samples_rejects_empty_parquet(tmp_path: Path) -> None:
    input_path = tmp_path / "data.parquet"
    audio_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    table = pa.table(
        {
            "audio": pa.array([], type=audio_type),
            "text": pa.array([], type=pa.string()),
        }
    )
    pq.write_table(table, input_path)

    with pytest.raises(ValueError, match="contains no samples"):
        load_inference_samples(input_path)
