import sys
from pathlib import Path

import pytest

from src.stage1.cli import parse_arguments as parse_stage1_arguments
from src.stage2.cli import parse_arguments as parse_stage2_arguments
from src.test.cli import parse_arguments as parse_test_arguments


def test_stage2_cli_uses_renamed_training_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "model.ckpt"
    checkpoint_path.touch()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage2",
            "--initialization-checkpoint",
            str(checkpoint_path),
            "--train-max-batch-padded-audio-seconds",
            "800",
            "--ctc-loss-weight",
            "0.2",
        ],
    )

    config = parse_stage2_arguments()

    assert config.train_max_batch_padded_audio_seconds == 800
    assert config.ctc_loss_weight == 0.2


def test_stage1_cli_reports_pydantic_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["stage1", "--num-workers", "0"])

    with pytest.raises(SystemExit):
        parse_stage1_arguments()


def test_stage1_cli_accepts_resume_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "last.ckpt"
    checkpoint_path.touch()
    monkeypatch.setattr(
        sys,
        "argv",
        ["stage1", "--resume-checkpoint", str(checkpoint_path)],
    )

    config = parse_stage1_arguments()

    assert config.resume_checkpoint == checkpoint_path


def test_stage2_cli_accepts_resume_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "last.ckpt"
    checkpoint_path.touch()
    monkeypatch.setattr(
        sys,
        "argv",
        ["stage2", "--resume-checkpoint", str(checkpoint_path)],
    )

    config = parse_stage2_arguments()

    assert config.resume_checkpoint == checkpoint_path


def test_test_cli_accepts_checkpoint_and_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "model.ckpt"
    checkpoint_path.touch()
    output_path = tmp_path / "results.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "test",
            "--checkpoint",
            str(checkpoint_path),
            "--output",
            str(output_path),
        ],
    )

    config = parse_test_arguments()

    assert config.checkpoint_path == checkpoint_path
    assert config.output_path == output_path
    assert config.batch_size == 16
    assert config.bucket_buffer_size == 256
    assert config.num_decode_threads == 16
    assert config.dataset_ratio == 0.1
