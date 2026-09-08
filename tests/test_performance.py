from types import SimpleNamespace
from typing import cast

import lightning as L
import torch

from src.training.performance import TrainingProgressBar, audio_batch_sizes


def test_training_progress_bar_hides_logger_version() -> None:
    trainer = cast(
        L.Trainer,
        SimpleNamespace(
            loggers=[SimpleNamespace(version=3)],
            progress_bar_metrics={"train_loss": 1.0},
        ),
    )
    pl_module = cast(L.LightningModule, object())

    metrics = TrainingProgressBar().get_metrics(trainer, pl_module)

    assert metrics == {"train_loss": 1.0}


def test_audio_batch_sizes_counts_valid_and_padded_samples() -> None:
    input_values = torch.zeros(2, 4)

    valid_samples, padded_samples = audio_batch_sizes(input_values, 6)

    assert valid_samples == 6
    assert padded_samples == 8
