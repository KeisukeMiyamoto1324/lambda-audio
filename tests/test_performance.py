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


def test_training_loss_plot_replaces_loss_within_same_optimizer_step() -> None:
    progress_bar = TrainingProgressBar()

    progress_bar.loss_plot.update(3, 2.0)
    progress_bar.loss_plot.update(3, 1.5)
    progress_bar.loss_plot.update(4, 1.0)

    assert progress_bar.loss_plot.steps == [3, 4]
    assert progress_bar.loss_plot.losses == [1.5, 1.0]


def test_training_loss_plot_displays_latest_500_steps() -> None:
    progress_bar = TrainingProgressBar()
    for step in range(1, 502):
        progress_bar.loss_plot.update(step, float(step))

    steps, losses = progress_bar.loss_plot.visible_data()

    assert steps == list(range(2, 502))
    assert losses == [float(step) for step in range(2, 502)]


def test_training_loss_plot_is_rendered_above_progress() -> None:
    progress_bar = TrainingProgressBar()
    progress_bar.progress_renderables = lambda: iter(["progress"])

    renderables = list(progress_bar._get_progress_renderables())

    assert renderables == [progress_bar.loss_plot, "progress"]
