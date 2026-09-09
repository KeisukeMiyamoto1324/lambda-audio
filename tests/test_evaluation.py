import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from src.inference.transcription import TranscriptionResult
from src.test.config import EvaluationConfig
from src.test.evaluation import character_edit_distance, evaluate_test_dataset, normalize_for_cer


class DummyAudio:
    def __init__(self) -> None:
        self.samples = SimpleNamespace(
            data=torch.ones(1, 160, dtype=torch.float32),
            sample_rate=16_000,
        )

    def get_all_samples(self) -> SimpleNamespace:
        return self.samples


def test_normalize_for_cer_removes_spaces_and_punctuation() -> None:
    assert normalize_for_cer(" 日本語、です。 A! ") == "日本語ですA"


@pytest.mark.parametrize(
    ("reference", "prediction", "expected"),
    [
        ("", "", 0),
        ("abc", "abc", 0),
        ("abc", "axc", 1),
        ("abc", "ab", 1),
        ("ab", "abc", 1),
    ],
)
def test_character_edit_distance(
    reference: str,
    prediction: str,
    expected: int,
) -> None:
    assert character_edit_distance(reference, prediction) == expected


def test_evaluate_test_dataset_writes_jsonl_and_corpus_cer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "model.ckpt"
    checkpoint_path.touch()
    output_path = tmp_path / "results.jsonl"
    config = EvaluationConfig(checkpoint_path=checkpoint_path, output_path=output_path)
    predictions = iter(["あい", "う"])

    def transcribe(*args: Any, **kwargs: Any) -> list[TranscriptionResult]:
        batch_size = len(args[5])
        return [TranscriptionResult(text=next(predictions), audio_seconds=0.01) for _ in range(batch_size)]

    monkeypatch.setattr("src.test.evaluation.transcribe_audio_batch", transcribe)
    rows = [
        {"id": "first", "audio": DummyAudio(), "text": "あ、い"},
        {"id": "second", "audio": DummyAudio(), "text": "うえ"},
    ]

    summary = evaluate_test_dataset(
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        config,
        torch.device("cpu"),
        rows,
    )

    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert summary.samples == 2
    assert summary.character_errors == 1
    assert summary.reference_characters == 4
    assert summary.cer == 0.25
    assert [record["id"] for record in records] == ["first", "second"]
    assert records[0]["normalized_reference"] == "あい"


def test_evaluate_test_dataset_rejects_empty_references(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "model.ckpt"
    checkpoint_path.touch()
    config = EvaluationConfig(
        checkpoint_path=checkpoint_path,
        output_path=tmp_path / "results.jsonl",
    )

    monkeypatch.setattr(
        "src.test.evaluation.transcribe_audio_batch",
        lambda *args, **kwargs: [TranscriptionResult(text="", audio_seconds=0.01)],
    )

    with pytest.raises(ValueError, match="no reference characters"):
        evaluate_test_dataset(
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            config,
            torch.device("cpu"),
            [{"id": "empty", "audio": DummyAudio(), "text": "。、 "}],
        )


def test_evaluate_test_dataset_limits_rows_by_ratio(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "model.ckpt"
    checkpoint_path.touch()
    config = EvaluationConfig(
        checkpoint_path=checkpoint_path,
        output_path=tmp_path / "results.jsonl",
        dataset_ratio=0.0001,
    )
    processed_batch_sizes: list[int] = []

    def transcribe(*args: Any, **kwargs: Any) -> list[TranscriptionResult]:
        batch_size = len(args[5])
        processed_batch_sizes.append(batch_size)
        return [TranscriptionResult(text="正解", audio_seconds=0.01)] * batch_size

    monkeypatch.setattr("src.test.evaluation.transcribe_audio_batch", transcribe)
    rows = [{"id": str(index), "audio": DummyAudio(), "text": "正解"} for index in range(10)]

    summary = evaluate_test_dataset(
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        config,
        torch.device("cpu"),
        rows,
    )

    assert summary.samples == 3
    assert sum(processed_batch_sizes) == 3
