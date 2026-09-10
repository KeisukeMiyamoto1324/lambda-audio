from types import SimpleNamespace
from typing import cast

import lightning as L

from src.training.performance import TrainingProgressBar


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


def test_training_progress_bar_hides_unbounded_epoch_limit() -> None:
    progress_bar = TrainingProgressBar()
    progress_bar._trainer = cast(L.Trainer, SimpleNamespace(max_epochs=-1))

    assert progress_bar._get_train_description(0) == "Epoch 0"
