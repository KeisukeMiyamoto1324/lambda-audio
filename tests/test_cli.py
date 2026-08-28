import sys
from pathlib import Path

import pytest

from src.stage1.cli import parse_arguments as parse_stage1_arguments
from src.stage2.cli import parse_arguments as parse_stage2_arguments


def test_stage1_cli_uses_config_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["stage1"])

    config = parse_stage1_arguments()

    assert config.output_dir == Path("outputs/stage1")
    assert config.max_steps == 204_800
    assert config.gradient_accumulation_steps == 2


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
            "--train-max-batch-attention-cost",
            "20000000",
        ],
    )

    config = parse_stage2_arguments()

    assert config.train_max_batch_padded_audio_seconds == 800
    assert config.train_max_batch_attention_cost == 20_000_000


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
