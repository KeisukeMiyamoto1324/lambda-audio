import csv
from io import BytesIO, StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf
import torch
from lightning.pytorch.plugins.precision import MixedPrecision
from rich.console import Console
from rich.progress import Progress
from torch import nn

from src.benchmark import runner as infer_module
from src.benchmark.config import BenchmarkConfig
from src.modeling.checkpoint import load_model_checkpoint
from src.shared import inference_display as display_module
from src.shared.device import select_device
from src.shared.languages import OUTPUT_LANGUAGES, OutputLanguage
from src.shared.samples import InferenceSample, load_inference_samples, load_multilingual_inference_samples
from src.shared.transcription import decode_audio, iter_transcriptions, transcribe
from src.training.config import TrainingConfig
from src.training.data_module import LambdaAudioDataModule
from src.training.inference_monitor import ValidationInferenceMonitor
from src.training.lightning import LambdaAudioLightningModule


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.generation_arguments: dict[str, Any] = {}
        self.batch_sizes: list[int] = []
        self.autocast_states: list[bool] = []

    def generate(self, **kwargs: Any) -> torch.LongTensor:
        self.generation_arguments = kwargs
        batch_size = kwargs["input_ids"].shape[0]
        self.batch_sizes.append(batch_size)
        self.autocast_states.append(torch.is_autocast_enabled("cpu"))
        return torch.cat([kwargs["input_ids"], torch.tensor([[4, 2]]).repeat(batch_size, 1)], dim=1)


class DummyFeatureExtractor:
    def __call__(self, audio: list[np.ndarray], **kwargs: Any) -> dict[str, torch.Tensor]:
        assert kwargs["padding"] is True
        input_values = torch.zeros(len(audio), max(array.size for array in audio))
        attention_mask = torch.zeros_like(input_values, dtype=torch.long)
        for index, array in enumerate(audio):
            input_values[index, : array.size] = torch.from_numpy(array)
            attention_mask[index, : array.size] = 1
        return {
            "input_values": input_values,
            "attention_mask": attention_mask,
        }


class DummyTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    all_special_tokens = [f"<|target_{language}|>" for language in OUTPUT_LANGUAGES]

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.all_special_tokens.index(token) + 5

    def batch_decode(
        self,
        token_ids: torch.LongTensor,
        skip_special_tokens: bool,
    ) -> list[str]:
        assert all(tokens == [4, 2] for tokens in token_ids.tolist())
        assert skip_special_tokens
        return [" transcription "] * token_ids.shape[0]


def test_decode_audio_reads_flac_with_unknown_total_sample_count() -> None:
    audio_file = BytesIO()
    source = np.array([0.25, -0.5], dtype=np.float32)
    sf.write(audio_file, source, 16_000, format="FLAC")
    encoded = bytearray(audio_file.getvalue())
    stream_properties = int.from_bytes(encoded[18:26], "big")
    encoded[18:26] = (stream_properties & ~((1 << 36) - 1)).to_bytes(8, "big")

    decoded = decode_audio(bytes(encoded), 16_000)

    np.testing.assert_allclose(decoded, source, atol=1e-4)


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


@pytest.mark.parametrize("output_language", OUTPUT_LANGUAGES)
def test_transcribe_generates_from_language_prompt(
    output_language: OutputLanguage,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "model.ckpt"
    checkpoint_path.touch()
    config = BenchmarkConfig(
        checkpoint_path=checkpoint_path,
        input_path=checkpoint_path,
    )
    model = DummyModel()
    monkeypatch.setattr(
        "src.shared.transcription.decode_audio",
        lambda audio, sampling_rate: np.ones(160, dtype=np.float32),
    )

    result = transcribe(
        model,  # type: ignore[arg-type]
        DummyFeatureExtractor(),  # type: ignore[arg-type]
        DummyTokenizer(),  # type: ignore[arg-type]
        config,
        torch.device("cpu"),
        [b"audio"],
        output_language=output_language,
    )[0]

    assert result.text == "transcription"
    assert result.audio_seconds == 0.01
    assert model.generation_arguments["input_ids"].tolist() == [[1, DummyTokenizer().convert_tokens_to_ids(f"<|target_{output_language}|>")]]
    assert not model.generation_arguments["audio_padding_mask"].any()
    assert model.generation_arguments["use_cache"] is True


@pytest.mark.parametrize("batch_size", [1, 2, 8])
def test_iter_transcriptions_batches_variable_length_audio_in_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    batch_size: int,
) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.touch()
    config = BenchmarkConfig(checkpoint_path=checkpoint, input_path=checkpoint)
    model = DummyModel()
    samples = [InferenceSample(audio_name=str(index), audio=b"a" * (index + 1), reference="音声") for index in range(5)]
    monkeypatch.setattr("src.shared.transcription.decode_audio", lambda audio, sampling_rate: np.ones(len(audio), dtype=np.float32))

    results = list(
        iter_transcriptions(
            model,  # type: ignore[arg-type]
            DummyFeatureExtractor(),  # type: ignore[arg-type]
            DummyTokenizer(),  # type: ignore[arg-type]
            config,
            torch.device("cpu"),
            samples,
            batch_size,
        )
    )

    assert [sample for sample, result in results] == samples
    assert [result.audio_seconds for sample, result in results] == [(index + 1) / 16_000 for index in range(5)]
    assert all(result.text == "transcription" for sample, result in results)
    assert model.batch_sizes == [min(batch_size, 5 - index) for index in range(0, 5, batch_size)]
    final_batch_size = model.batch_sizes[-1]
    final_lengths = torch.arange(6 - final_batch_size, 6)
    expected_padding_mask = torch.arange(5).unsqueeze(0) >= final_lengths.unsqueeze(1)
    assert torch.equal(model.generation_arguments["audio_padding_mask"], expected_padding_mask)
    assert model.generation_arguments["input_ids"].tolist() == [[1, 5]] * final_batch_size


def test_transcribe_rejects_overlong_audio_before_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.touch()
    config = BenchmarkConfig(checkpoint_path=checkpoint, input_path=checkpoint, max_audio_seconds=0.01)
    model = DummyModel()
    monkeypatch.setattr("src.shared.transcription.decode_audio", lambda audio, sampling_rate: np.ones(len(audio), dtype=np.float32))

    with pytest.raises(ValueError, match="0.01 seconds"):
        transcribe(
            model,  # type: ignore[arg-type]
            DummyFeatureExtractor(),  # type: ignore[arg-type]
            DummyTokenizer(),  # type: ignore[arg-type]
            config,
            torch.device("cpu"),
            [b"a", b"a" * 161],
        )

    assert model.batch_sizes == []


def test_inference_progress_tracks_batched_results_and_cer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.touch()
    config = BenchmarkConfig(checkpoint_path=checkpoint, input_path=checkpoint, batch_size=2)
    samples = [
        InferenceSample(audio_name="first.wav", audio=b"a", reference="あいう"),
        InferenceSample(audio_name="second.wav", audio=b"aa", reference="かきくけこ"),
        InferenceSample(audio_name="third.wav", audio=b"aaa", reference=""),
    ]
    model = DummyModel()
    tokenizer = DummyTokenizer()
    monkeypatch.setattr(tokenizer, "batch_decode", lambda token_ids, skip_special_tokens: ["あいえ", "かきくけこ"] if len(token_ids) == 2 else ["任意"])
    components = SimpleNamespace(model=model, feature_extractor=DummyFeatureExtractor(), tokenizer=tokenizer)
    output = StringIO()
    console = Console(file=output, width=100, color_system=None)
    progress = Progress(console=console, disable=True)
    monkeypatch.setattr(infer_module, "parse_arguments", lambda: config)
    monkeypatch.setattr(infer_module, "load_hf_token", lambda: "token")
    monkeypatch.setattr(infer_module, "select_device", lambda: torch.device("cpu"))
    monkeypatch.setitem(
        infer_module.MODEL_LOADERS,
        "lambda-audio",
        lambda config, device, token: lambda audio: transcribe(model, components.feature_extractor, tokenizer, config, device, audio),
    )
    monkeypatch.setattr(infer_module, "load_inference_samples", lambda path: samples)
    monkeypatch.setattr(display_module, "Console", lambda: console)
    monkeypatch.setattr(display_module, "Progress", lambda *columns, **kwargs: progress)
    monkeypatch.setattr("src.shared.transcription.decode_audio", lambda audio, sampling_rate: np.ones(len(audio), dtype=np.float32))

    infer_module.main()

    assert model.batch_sizes == [2, 1]
    assert progress.tasks[0].completed == progress.tasks[0].total == 3
    assert progress.tasks[0].fields == {"cer": "12.50%", "skipped": 1}
    assert "CER: 12.50%" in output.getvalue()
    assert "Elapsed time:" in output.getvalue()
    assert "first.wav" not in output.getvalue()
    assert "second.wav" not in output.getvalue()
    assert "third.wav" not in output.getvalue()


def test_training_inference_batches_samples_and_preserves_history(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "samples.parquet"
    input_path.touch()
    config = TrainingConfig(output_dir=tmp_path, inference_samples_path=input_path, inference_batch_size=2)
    model = DummyModel()
    module = LambdaAudioLightningModule(model, config)  # type: ignore[arg-type]
    data_module = LambdaAudioDataModule(DummyFeatureExtractor(), DummyTokenizer(), config, "token")  # type: ignore[arg-type]
    samples = [InferenceSample(audio_name=str(index), audio=b"a" * (index + 1), reference="日本語", lang=language) for index, language in enumerate(("ja", "en", "ar"))]
    output = StringIO()
    console = Console(file=output, width=100, color_system=None)
    progress = Progress(console=console, disable=True)
    monkeypatch.setattr(display_module, "Progress", lambda *columns, **kwargs: progress)
    barriers: list[str] = []
    trainer = SimpleNamespace(
        datamodule=data_module,
        global_step=42,
        is_global_zero=True,
        strategy=SimpleNamespace(barrier=barriers.append),
        precision_plugin=MixedPrecision("bf16-mixed", device="cpu"),
    )
    progress_bar = SimpleNamespace(progress=progress)
    monitor = ValidationInferenceMonitor(config, progress_bar)  # type: ignore[arg-type]
    monkeypatch.setattr("src.training.inference_monitor.load_multilingual_inference_samples", lambda path: samples)
    monkeypatch.setattr("src.shared.transcription.decode_audio", lambda audio, sampling_rate: np.ones(len(audio), dtype=np.float32))

    monitor.on_validation_end(trainer, module)  # type: ignore[arg-type]

    with monitor.history_path.open(encoding="utf-8", newline="") as file:
        records = list(csv.DictReader(file))
    assert model.batch_sizes == [2, 1]
    assert [record["audio"] for record in records] == ["0", "1", "2"]
    assert [record["step"] for record in records] == ["42"] * 3
    assert [record["lang"] for record in records] == ["ja", "en", "ar"]
    assert model.generation_arguments["input_ids"].tolist() == [[1, 7]]
    rendered = output.getvalue()
    assert "Validation Inference — Step 42" in rendered
    assert rendered.index("Lang") < rendered.index("正解") < rendered.index("推論")
    assert all(language in rendered for language in ("ja", "en", "ar"))
    assert records[0]["inference_seconds"] == records[1]["inference_seconds"]
    assert progress.tasks[0].completed == 3
    assert "CER:" in rendered
    assert "Elapsed time:" in rendered
    assert barriers == ["validation_inference"]
    assert model.autocast_states == [True, True]
    assert not torch.is_autocast_enabled("cpu")

    # ---------------------------------------------------------
    # Reuse CPU inputs but run generation again at the next validation step.
    # ---------------------------------------------------------
    assert monitor.batches is not None
    cached_inputs = [{name: tensor.clone() for name, tensor in batch.inputs.items()} for _, batch in monitor.batches]
    monkeypatch.setattr("src.training.inference_monitor.load_multilingual_inference_samples", Mock(side_effect=AssertionError("Parquet was read twice")))
    monkeypatch.setattr("src.shared.transcription.decode_audio", Mock(side_effect=AssertionError("Audio was decoded twice")))
    monkeypatch.setattr(data_module.collator, "feature_extractor", Mock(side_effect=AssertionError("Audio was processed twice")))
    monkeypatch.setattr(data_module.collator.tokenizer, "convert_tokens_to_ids", Mock(side_effect=AssertionError("Prompts were converted twice")))
    trainer.global_step = 84
    monitor.on_validation_end(trainer, module)  # type: ignore[arg-type]

    assert model.batch_sizes == [2, 1, 2, 1]
    assert model.autocast_states == [True] * 4
    assert not torch.is_autocast_enabled("cpu")
    for (_, batch), original_inputs in zip(monitor.batches, cached_inputs, strict=True):
        for name, tensor in batch.inputs.items():
            assert tensor.device.type == "cpu"
            torch.testing.assert_close(tensor, original_inputs[name])
    with monitor.history_path.open(encoding="utf-8", newline="") as file:
        records = list(csv.DictReader(file))
    assert [record["step"] for record in records] == ["42"] * 3 + ["84"] * 3
    assert [record["audio"] for record in records] == ["0", "1", "2"] * 2
    assert [record["lang"] for record in records] == ["ja", "en", "ar"] * 2
    assert barriers == ["validation_inference"] * 2


@pytest.mark.parametrize("audio_paths", [("a.flac", "b.flac"), (None, None), ("a.flac", None)])
def test_load_inference_samples_reads_parquet(
    tmp_path: Path,
    audio_paths: tuple[str | None, str | None],
) -> None:
    input_path = tmp_path / "data.parquet"
    audio_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    table = pa.table(
        {
            "audio": pa.array(
                [
                    {"bytes": b"audio-a", "path": audio_paths[0]},
                    {"bytes": b"audio-b", "path": audio_paths[1]},
                ],
                type=audio_type,
            ),
            "text": ["一", "二"],
        }
    )
    pq.write_table(table, input_path)

    samples = load_inference_samples(input_path)

    assert [sample.audio_name for sample in samples] == ["data.parquet:1", "data.parquet:2"]
    assert [sample.audio for sample in samples] == [b"audio-a", b"audio-b"]
    assert [sample.reference for sample in samples] == ["一", "二"]


def test_load_inference_samples_reads_wav(tmp_path: Path) -> None:
    input_path = tmp_path / "audio.wav"
    input_path.write_bytes(b"wav-audio")

    samples = load_inference_samples(input_path)

    assert len(samples) == 1
    assert samples[0].audio_name == "audio.wav"
    assert samples[0].audio == b"wav-audio"
    assert samples[0].reference == ""


def test_load_inference_samples_rejects_unsupported_file(tmp_path: Path) -> None:
    input_path = tmp_path / "audio.mp3"
    input_path.touch()

    with pytest.raises(ValueError, match="WAV or Parquet"):
        load_inference_samples(input_path)


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


def test_load_multilingual_samples_preserves_ids_languages_and_references(tmp_path: Path) -> None:
    input_path = tmp_path / "samples.parquet"
    rows = [
        {"id": f"sample-{index}", "audio": {"bytes": language.encode(), "path": None}, "lang": language, "text": f"reference-{language}"}
        for index, language in enumerate(OUTPUT_LANGUAGES)
    ]
    pq.write_table(pa.Table.from_pylist(rows), input_path)

    samples = load_multilingual_inference_samples(input_path)

    assert [sample.audio_name for sample in samples] == [row["id"] for row in rows]
    assert [sample.lang for sample in samples] == list(OUTPUT_LANGUAGES)
    assert [sample.audio for sample in samples] == [row["audio"]["bytes"] for row in rows]
    assert [sample.reference for sample in samples] == [row["text"] for row in rows]


def test_iter_transcriptions_uses_each_sample_language_in_one_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    model = DummyModel()
    tokenizer = DummyTokenizer()
    samples = [InferenceSample(audio_name=language, audio=b"audio", reference=language, lang=language) for language in OUTPUT_LANGUAGES]
    monkeypatch.setattr("src.shared.transcription.decode_audio", lambda audio, sampling_rate: np.ones(160, dtype=np.float32))
    results = list(
        iter_transcriptions(
            model,
            DummyFeatureExtractor(),
            tokenizer,
            TrainingConfig(),
            torch.device("cpu"),
            samples,
            9,
        )
    )  # type: ignore[arg-type]

    assert [sample for sample, result in results] == samples
    assert model.batch_sizes == [9]
    assert model.generation_arguments["input_ids"].tolist() == [[1, tokenizer.convert_tokens_to_ids(f"<|target_{language}|>")] for language in OUTPUT_LANGUAGES]


def test_transcribe_rejects_language_count_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    model = DummyModel()
    monkeypatch.setattr("src.shared.transcription.decode_audio", lambda audio, sampling_rate: np.ones(160, dtype=np.float32))
    with pytest.raises(ValueError, match="one output language"):
        transcribe(model, DummyFeatureExtractor(), DummyTokenizer(), TrainingConfig(), torch.device("cpu"), [b"audio"], output_language=[])  # type: ignore[arg-type]
    assert model.batch_sizes == []
