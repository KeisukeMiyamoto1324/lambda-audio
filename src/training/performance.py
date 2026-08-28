import time

import lightning as L
import torch
from lightning.pytorch.callbacks import RichProgressBar

DATA_STALL_SECONDS = 1.0


class TrainingProgressBar(RichProgressBar):
    def get_metrics(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> dict[str, int | str | float | dict[str, float]]:
        metrics = super().get_metrics(trainer, pl_module)
        metrics.pop("v_num", None)
        return metrics


def audio_batch_sizes(audio_padding_mask: torch.Tensor) -> tuple[int, int]:
    padded_samples = audio_padding_mask.numel()
    valid_samples = padded_samples - torch.count_nonzero(audio_padding_mask).item()
    return valid_samples, padded_samples


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
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        now = time.perf_counter()
        data_time = now - self.previous_batch_end_time
        self.window_data_time += data_time
        self.window_data_time_max = max(self.window_data_time_max, data_time)
        if data_time >= DATA_STALL_SECONDS:
            self.window_data_stalls += 1
        valid_samples, padded_samples = audio_batch_sizes(batch["audio_padding_mask"])
        self.window_valid_samples += valid_samples
        self.window_padded_samples += padded_samples
        self.window_batches += 1
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record()

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: torch.Tensor | dict[str, torch.Tensor] | None,
        batch: dict[str, torch.Tensor],
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
        gpu_seconds = sum(
            start.elapsed_time(end) / 1_000.0 for start, end in self.gpu_events
        )
        elapsed_seconds = now - self.window_start_time
        metrics = {
            "data_time": self.window_data_time / self.window_batches,
            "data_time_max": self.window_data_time_max,
            "data_stalls": self.window_data_stalls,
            "gpu_step_time": gpu_seconds / self.window_batches,
            "padding_ratio": (self.window_padded_samples / self.window_valid_samples),
            "audio_seconds_per_second": (
                self.window_valid_samples / self.sampling_rate / elapsed_seconds
            ),
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
