from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from transformers import BatchFeature

from src.benchmark.config import BenchmarkConfig
from src.benchmark.models.cohere import CohereTranscriber
from src.benchmark.models.parakeet import ParakeetTranscriber
from src.benchmark.models.reazonspeech import ReazonSpeechTranscriber
from src.benchmark.registry import MODEL_LOADERS


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> BenchmarkConfig:
    path = tmp_path / "input.wav"
    path.touch()
    monkeypatch.setattr("src.shared.transcription.decode_audio", lambda audio, sampling_rate: np.ones(len(audio), dtype=np.float32))
    return BenchmarkConfig(model="reazonspeech", input_path=path, max_new_tokens=32)


class CtcModel:
    dtype = torch.float32

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        assert torch.is_inference_mode_enabled()
        assert set(kwargs) == {"input_values"}
        assert kwargs["input_values"].shape == (2, 320)
        return SimpleNamespace(logits=torch.tensor([[[0, 1], [1, 0]], [[1, 0], [0, 1]]]))

    def _get_feat_extract_output_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        return lengths // 160


class CtcProcessor:
    def __call__(self, arrays: list[np.ndarray], **kwargs: Any) -> dict[str, torch.Tensor]:
        mask = torch.zeros(2, 320, dtype=torch.long)
        for index, array in enumerate(arrays):
            mask[index, : array.size] = 1
        assert kwargs["padding"] and kwargs["return_attention_mask"]
        return {"input_values": torch.ones(2, 320), "attention_mask": mask}

    def batch_decode(self, ids: list[torch.Tensor], **kwargs: Any) -> list[str]:
        assert [item.tolist() for item in ids] == [[1], [0, 1]]
        return [" 一 ", " 二 "]


def test_ctc_crops_padding_before_decoding(config: BenchmarkConfig) -> None:
    engine = ReazonSpeechTranscriber(CtcModel(), CtcProcessor(), config, torch.device("cpu"))  # type: ignore[arg-type]
    results = engine([b"a" * 160, b"a" * 320])
    assert [item.text for item in results] == ["一", "二"]
    assert [item.audio_seconds for item in results] == [0.01, 0.02]


class CohereModel:
    dtype = torch.float32

    def generate(self, **kwargs: Any) -> torch.Tensor:
        assert torch.is_inference_mode_enabled()
        assert kwargs["max_new_tokens"] == 32
        assert kwargs["num_beams"] == 1 and kwargs["do_sample"] is False
        assert kwargs["audio_chunk_index"].dtype == torch.long
        assert kwargs["decoder_input_ids"].dtype == torch.long
        return torch.tensor([[1], [2]])


class CohereProcessor:
    def __call__(self, arrays: list[np.ndarray], **kwargs: Any) -> BatchFeature:
        assert len(arrays) == 2
        assert kwargs["language"] == "ja" and kwargs["punctuation"] is True
        return BatchFeature({"input_features": torch.ones(2, 4, 8), "audio_chunk_index": torch.tensor([0, 1]), "decoder_input_ids": torch.tensor([[1], [1]])})

    def decode(self, outputs: torch.Tensor, **kwargs: Any) -> list[str]:
        assert kwargs["audio_chunk_index"].tolist() == [0, 1]
        assert kwargs["language"] == "ja" and kwargs["skip_special_tokens"]
        return [" 一 ", " 二 "]


def test_cohere_preserves_chunk_indices_and_integer_prompt(config: BenchmarkConfig) -> None:
    engine = CohereTranscriber(CohereModel(), CohereProcessor(), config, torch.device("cpu"))  # type: ignore[arg-type]
    results = engine([b"a" * 160, b"a" * 320])
    assert [item.text for item in results] == ["一", "二"]
    assert [item.audio_seconds for item in results] == [0.01, 0.02]


class ParakeetModel:
    def transcribe(self, **kwargs: Any) -> list[SimpleNamespace]:
        assert torch.is_inference_mode_enabled()
        assert kwargs["batch_size"] == 2
        assert [array.size for array in kwargs["audio"]] == [160, 320]
        assert kwargs["return_hypotheses"] and kwargs["num_workers"] == 0
        return [SimpleNamespace(text=" 一 "), SimpleNamespace(text=" 二 ")]


def test_parakeet_batches_arrays_in_input_order(config: BenchmarkConfig) -> None:
    engine = ParakeetTranscriber(ParakeetModel(), config)  # type: ignore[arg-type]
    results = engine([b"a" * 160, b"a" * 320])
    assert [item.text for item in results] == ["一", "二"]
    assert [item.audio_seconds for item in results] == [0.01, 0.02]


@pytest.mark.parametrize("model", ["reazonspeech", "kotoba-whisper", "parakeet", "cohere-transcribe"])
def test_additional_models_can_be_selected(model: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.benchmark.cli import parse_arguments

    path = tmp_path / "input.wav"
    path.touch()
    monkeypatch.setattr("sys.argv", ["benchmark", "--model", model, "--input", str(path)])
    result = parse_arguments()
    assert result.model == model
    assert result.checkpoint_path is None
    assert callable(MODEL_LOADERS[result.model])


@pytest.mark.parametrize(
    ("name", "module_name", "model_class", "processor_class", "model_id"),
    [
        ("reazonspeech", "reazonspeech", "Wav2Vec2ForCTC", "AutoProcessor", "reazon-research/japanese-wav2vec2-base-rs35kh"),
        ("cohere-transcribe", "cohere", "CohereAsrForConditionalGeneration", "AutoProcessor", "CohereLabs/cohere-transcribe-03-2026"),
        ("kotoba-whisper", "whisper", "WhisperForConditionalGeneration", "WhisperProcessor", "kotoba-tech/kotoba-whisper-v2.2"),
    ],
)
def test_model_loaders_pass_explicit_hf_token(
    name: str, module_name: str, model_class: str, processor_class: str, model_id: str, config: BenchmarkConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from importlib import import_module

    calls: list[dict[str, Any]] = []
    model = torch.nn.Linear(1, 1)

    def load(identifier: str, **kwargs: Any) -> Any:
        assert identifier == model_id
        calls.append(kwargs)
        return model if "dtype" in kwargs else SimpleNamespace()

    module = import_module("src.benchmark.models." + module_name)
    monkeypatch.setattr(module, model_class, SimpleNamespace(from_pretrained=load))
    monkeypatch.setattr(module, processor_class, SimpleNamespace(from_pretrained=load))
    engine = MODEL_LOADERS[name](config, torch.device("cpu"), "explicit-token")
    assert engine.model is model  # type: ignore[attr-defined]
    assert not model.training
    assert len(calls) == 2
    assert all(call["token"] == "explicit-token" for call in calls)
    assert calls[1]["dtype"] == torch.float32


def test_parakeet_downloads_with_token_and_selects_batched_tdt(config: BenchmarkConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    from omegaconf import OmegaConf

    from src.benchmark.models import parakeet

    model = torch.nn.Linear(1, 1)
    model.cfg = SimpleNamespace(decoding=OmegaConf.create({"strategy": "beam", "model_type": "tdt"}))
    decoding_calls: list[Any] = []

    def change_decoding(decoding: Any, **kwargs: Any) -> None:
        assert kwargs == {"decoder_type": "rnnt", "verbose": False}
        decoding_calls.append(decoding)

    model.change_decoding_strategy = change_decoding

    def download(repo: str, filename: str, token: str) -> str:
        assert repo == "nvidia/parakeet-tdt_ctc-0.6b-ja"
        assert filename.endswith(".nemo") and token == "explicit-token"
        return "/tmp/test.nemo"

    def restore(path: str, map_location: torch.device) -> Any:
        assert path == "/tmp/test.nemo" and map_location == torch.device("cpu")
        return model

    monkeypatch.setattr(parakeet, "hf_hub_download", download)
    monkeypatch.setitem(sys.modules, "nemo.collections.asr.models", SimpleNamespace(EncDecHybridRNNTCTCBPEModel=SimpleNamespace(restore_from=restore)))
    engine = parakeet.load_model(config, torch.device("cpu"), "explicit-token")
    assert engine.model is model  # type: ignore[attr-defined]
    assert not model.training
    assert decoding_calls[0].strategy == "greedy_batch"
    assert decoding_calls[0].model_type == "tdt"
