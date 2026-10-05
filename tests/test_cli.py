import sys
from pathlib import Path

import pytest

from src.training.cli import parse_arguments


def test_training_cli_accepts_batch_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "training",
            "--max-batch-padded-audio-seconds",
            "600",
        ],
    )

    config = parse_arguments()

    assert config.max_batch_padded_audio_seconds == 600


def test_training_cli_reports_pydantic_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["training", "--num-workers", "0"])

    with pytest.raises(SystemExit):
        parse_arguments()


def test_training_cli_accepts_resume_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "last.ckpt"
    checkpoint_path.touch()
    monkeypatch.setattr(
        sys,
        "argv",
        ["training", "--resume-checkpoint", str(checkpoint_path)],
    )

    config = parse_arguments()

    assert config.resume_checkpoint == checkpoint_path


def test_cli_accepts_unfreeze_step(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["training", "--unfreeze-step", "0", "--max-steps", "10"])
    config = parse_arguments()
    assert config.unfreeze_step == 0
    assert config.max_steps == 10
