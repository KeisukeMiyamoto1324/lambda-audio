from types import SimpleNamespace
from typing import cast

import lightning as L

from src.data.types import AudioTextBatch
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
    batch = cast(
        AudioTextBatch,
        {
            "audio_valid_samples": 6,
            "audio_padded_samples": 8,
        },
    )

    valid_samples, padded_samples = audio_batch_sizes(batch)

    assert valid_samples == 6
    assert padded_samples == 8


def test_progress_loss_is_read_only_at_logging_boundary() -> None:
    from unittest.mock import Mock, patch

    loss = Mock()
    loss.item.return_value = 2.5
    trainer = cast(L.Trainer, SimpleNamespace(accumulate_grad_batches=3, callback_metrics={"train_loss": loss}, loggers=[], progress_bar_metrics={}))
    module = cast(L.LightningModule, object())
    bar = TrainingProgressBar(refresh_rate=10)
    with patch("lightning.pytorch.callbacks.RichProgressBar.on_train_batch_end"):
        for index in range(60):
            bar.on_train_batch_end(trainer, module, None, None, index)
            bar.get_metrics(trainer, module)
    assert loss.item.call_count == 2
    assert bar.get_metrics(trainer, module)["train_loss"] == 2.5
