import sys
from pathlib import Path

import pytest

from src.cer.cli import parse_arguments as parse_cer_arguments
from src.stage1.cli import parse_arguments as parse_stage1_arguments
from src.stage2.cli import parse_arguments as parse_stage2_arguments


@pytest.mark.parametrize("arguments, expected", [([], 5), (["--beam-search", "1"], 1), (["--beam-search", "3"], 3)])
def test_cer_cli_accepts_beam_search(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    arguments: list[str],
    expected: int,
) -> None:
    checkpoint_path = tmp_path / "model.ckpt"
    checkpoint_path.touch()
    monkeypatch.setattr(sys, "argv", ["cer", str(checkpoint_path), *arguments])

    assert parse_cer_arguments().beam_search == expected


@pytest.mark.parametrize("num_beams", ["0", "-1"])
def test_cer_cli_rejects_nonpositive_beam_search(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    num_beams: str,
) -> None:
    checkpoint_path = tmp_path / "model.ckpt"
    checkpoint_path.touch()
    monkeypatch.setattr(sys, "argv", ["cer", str(checkpoint_path), "--beam-search", num_beams])

    with pytest.raises(SystemExit):
        parse_cer_arguments()


def test_stage2_cli_accepts_training_batch_limit(
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
        ],
    )

    config = parse_stage2_arguments()

    assert config.train_max_batch_padded_audio_seconds == 800


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
