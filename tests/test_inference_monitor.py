from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from rich.table import Table

from src.inference.samples import InferenceSample
from src.inference.transcription import TranscriptionResult
from src.stage1.config import Stage1Config
from src.training import inference_monitor as inference_monitor_module
from src.training.inference_monitor import ValidationInferenceMonitor


class RecordingPrecisionPlugin:
    def __init__(self) -> None:
        self.active = False
        self.entries = 0

    @contextmanager
    def forward_context(self) -> Iterator[None]:
        self.active = True
        self.entries += 1
        try:
            yield
        finally:
            self.active = False


class DummyProgressBar:
    def print_table(self, table: Table) -> None:
        pass


class DummyModel:
    def __init__(self) -> None:
        self.decoder = SimpleNamespace(config=SimpleNamespace(use_cache=False))

    def eval(self) -> None:
        pass


def test_validation_inference_uses_training_precision_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sample_path = tmp_path / "sample.wav"
    sample_path.touch()
    config = Stage1Config(
        output_dir=tmp_path,
        inference_samples_path=sample_path,
    )
    precision_plugin = RecordingPrecisionPlugin()
    data_module = SimpleNamespace(
        collator=SimpleNamespace(feature_extractor=object(), tokenizer=object()),
    )
    trainer = SimpleNamespace(
        datamodule=data_module,
        global_step=2_000,
        precision_plugin=precision_plugin,
    )
    pl_module = SimpleNamespace(
        model=DummyModel(),
        device=torch.device("cpu"),
    )

    monkeypatch.setattr(
        inference_monitor_module,
        "LambdaAudioDataModule",
        type(data_module),
    )
    monkeypatch.setattr(
        inference_monitor_module,
        "load_inference_samples",
        lambda path: [InferenceSample(audio_name="sample.wav", audio=b"audio", reference="正解")],
    )

    def transcribe_in_context(*args: Any, **kwargs: Any) -> TranscriptionResult:
        assert precision_plugin.active
        return TranscriptionResult(text="推論", audio_seconds=1.0)

    monkeypatch.setattr(
        inference_monitor_module,
        "transcribe",
        transcribe_in_context,
    )
    monitor = ValidationInferenceMonitor(config, DummyProgressBar())  # type: ignore[arg-type]

    monitor._run_inference(trainer, pl_module)  # type: ignore[arg-type]

    assert precision_plugin.entries == 1
    assert not precision_plugin.active
    assert (tmp_path / "inference_history.csv").exists()
