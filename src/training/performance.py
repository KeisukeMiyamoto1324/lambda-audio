import time
from collections.abc import Iterable
from typing import Any, Callable

import lightning as L
import torch
from lightning.pytorch.callbacks import RichProgressBar
from rich.console import RenderableType
from rich.table import Table

from src.training.loss_plot import TrainingLossPlot

DATA_STALL_SECONDS = 1.0


class TrainingProgressBar(RichProgressBar):
    def __init__(self) -> None:
        super().__init__()
        self.loss_plot = TrainingLossPlot()
        self.progress_renderables: Callable[[], Iterable[RenderableType]] | None = None

    def _init_progress(self, trainer: L.Trainer) -> None:
        previous_progress = self.progress
        super()._init_progress(trainer)
        if self.progress is None or self.progress is previous_progress:
            return

        self.progress_renderables = self.progress.get_renderables
        self.progress.get_renderables = self._get_progress_renderables

    def _get_progress_renderables(self) -> Iterable[RenderableType]:
        if self.progress_renderables is None:
            raise RuntimeError("Progress renderables are not initialized.")
        yield self.loss_plot
        yield from self.progress_renderables()

    def _get_train_description(self, current_epoch: int) -> str:
        if self.trainer.max_epochs == -1:
            return f"Epoch {current_epoch}"
        return super()._get_train_description(current_epoch)

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: torch.Tensor | dict[str, torch.Tensor] | None,
        batch: Any,
        batch_idx: int,
    ) -> None:
        loss = trainer.progress_bar_metrics.get("train_loss")
        if isinstance(loss, torch.Tensor):
            loss = loss.item()
        if loss is not None:
            if not isinstance(loss, (int, float)):
                raise TypeError("train_loss must be numeric.")
            self.loss_plot.update(trainer.global_step, float(loss))
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def get_metrics(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> dict[str, int | str | float | dict[str, float]]:
        metrics = super().get_metrics(trainer, pl_module)
        metrics.pop("v_num", None)
        return metrics

    def print_table(self, table: Table) -> None:
        if self.progress is None:
            raise RuntimeError("The training progress display is not initialized.")
        self.progress.console.print(table)


class TrainingPerformanceMonitor(L.Callback):
    def __init__(self, sampling_rate: int, log_steps: int) -> None:
        self.sampling_rate = sampling_rate
        self.log_steps = log_steps
        self.gpu_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self.window_valid_samples = 0
        self.window_padded_samples = 0
        self.window_data_time = 0.0
        self.window_data_time_max = 0.0
        self.window_data_stalls = 0
        self.window_batches = 0
        self.window_start_time = 0.0
        self.previous_batch_end_time = 0.0
        self.start_event: torch.cuda.Event | None = None

    def on_train_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        now = time.perf_counter()
        self.window_start_time = now
        self.previous_batch_end_time = now

    def on_train_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch: dict[str, torch.Tensor | int],
        batch_idx: int,
    ) -> None:
        now = time.perf_counter()
        data_time = now - self.previous_batch_end_time
        self.window_data_time += data_time
        self.window_data_time_max = max(self.window_data_time_max, data_time)
        if data_time >= DATA_STALL_SECONDS:
            self.window_data_stalls += 1
        valid_samples_value = batch["audio_valid_samples"]
        padded_samples_value = batch["audio_padded_samples"]
        if not isinstance(valid_samples_value, int) or not isinstance(padded_samples_value, int):
            raise TypeError("Training batch audio statistics are invalid.")
        self.window_valid_samples += valid_samples_value
        self.window_padded_samples += padded_samples_value
        self.window_batches += 1
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record()

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: torch.Tensor | dict[str, torch.Tensor] | None,
        batch: dict[str, torch.Tensor | int],
        batch_idx: int,
    ) -> None:
        if self.start_event is None:
            raise RuntimeError("The GPU start event is unavailable.")
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        self.gpu_events.append((self.start_event, end_event))

        if self.window_batches < self.log_steps:
            self.previous_batch_end_time = time.perf_counter()
            return

        end_event.synchronize()
        now = time.perf_counter()
        gpu_seconds = sum(start.elapsed_time(end) / 1_000.0 for start, end in self.gpu_events)
        elapsed_seconds = now - self.window_start_time
        metrics = {
            "data_time": self.window_data_time / self.window_batches,
            "data_time_max": self.window_data_time_max,
            "data_stalls": self.window_data_stalls,
            "gpu_step_time": gpu_seconds / self.window_batches,
            "padding_ratio": (self.window_padded_samples / self.window_valid_samples),
            "audio_seconds_per_second": (self.window_valid_samples / self.sampling_rate / elapsed_seconds),
        }
        for name, value in metrics.items():
            pl_module.log(
                name,
                value,
                on_step=True,
                prog_bar=False,
                logger=True,
                sync_dist=False,
            )

        self.gpu_events.clear()
        self.window_valid_samples = 0
        self.window_padded_samples = 0
        self.window_data_time = 0.0
        self.window_data_time_max = 0.0
        self.window_data_stalls = 0
        self.window_batches = 0
        self.window_start_time = now
        self.previous_batch_end_time = now
