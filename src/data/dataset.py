import multiprocessing
from collections.abc import Iterator
from typing import Protocol

from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from torch.utils.data import IterableDataset as TorchIterableDataset
from torch.utils.data import get_worker_info

from src.data.types import TranscribedAudioExample


class EpochTranscribedAudioDataset(Protocol):
    def __iter__(self) -> Iterator[TranscribedAudioExample]: ...

    def set_epoch(self, epoch: int) -> None: ...


class AudioBatchDataset(TorchIterableDataset[list[TranscribedAudioExample]]):
    def __init__(
        self,
        dataset: EpochTranscribedAudioDataset,
        *,
        max_batch_padded_audio_samples: int,
        bucket_buffer_size: int,
        show_initial_buffer_progress: bool = False,
        repeat: bool = False,
    ) -> None:
        self.dataset = dataset
        self.max_batch_padded_audio_samples = max_batch_padded_audio_samples
        self.bucket_buffer_size = bucket_buffer_size
        self.show_initial_buffer_progress = show_initial_buffer_progress
        self.repeat = repeat
        self.epoch = multiprocessing.Value("q", 0)

        if max_batch_padded_audio_samples < 1:
            raise ValueError("The padded audio limit must be positive.")
        if bucket_buffer_size < 1:
            raise ValueError("The bucket buffer size must be positive.")

    def __iter__(self) -> Iterator[list[TranscribedAudioExample]]:
        epoch = self.epoch.value
        self.dataset.set_epoch(epoch)
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        source = iter(self.dataset)
        window: list[TranscribedAudioExample] = []
        source_exhausted = self._fill_window(
            window,
            source,
            self.bucket_buffer_size,
            self.show_initial_buffer_progress and worker_id == 0,
        )

        while window:
            batches = self._create_batches(window)
            next_window: list[TranscribedAudioExample] = []
            for batch_index, batch in enumerate(batches, start=1):
                yield batch
                if source_exhausted:
                    continue
                refill_target = batch_index * self.bucket_buffer_size // len(batches)
                source_exhausted = self._fill_window(
                    next_window,
                    source,
                    refill_target,
                    False,
                )
            if self.repeat and source_exhausted and not next_window:
                epoch += 1
                self.dataset.set_epoch(epoch)
                source = iter(self.dataset)
                source_exhausted = self._fill_window(
                    next_window,
                    source,
                    self.bucket_buffer_size,
                    False,
                )
            window = next_window

    def _create_batches(
        self,
        window: list[TranscribedAudioExample],
    ) -> list[list[TranscribedAudioExample]]:
        window.sort(key=lambda example: len(example["audio_array"]))
        batches: list[list[TranscribedAudioExample]] = []
        batch: list[TranscribedAudioExample] = []

        for example in window:
            # ---------------------------------------------------------
            # Bound padded feature extraction by total audio length.
            # ---------------------------------------------------------
            audio_samples = len(example["audio_array"])
            if audio_samples > self.max_batch_padded_audio_samples:
                raise ValueError("An audio example exceeds the padded audio limit.")

            batch_size = len(batch) + 1
            padded_audio_samples = batch_size * audio_samples

            if batch and padded_audio_samples > self.max_batch_padded_audio_samples:
                batches.append(batch)
                batch = []

            batch.append(example)

        if batch:
            batches.append(batch)

        return batches

    @staticmethod
    def _fill_window(
        window: list[TranscribedAudioExample],
        source: Iterator[TranscribedAudioExample],
        target_size: int,
        show_progress: bool,
    ) -> bool:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("Preparing training buffer"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            disable=not show_progress,
        )
        with progress:
            task = progress.add_task("buffer", total=target_size)
            while len(window) < target_size:
                try:
                    window.append(next(source))
                except StopIteration:
                    return True
                progress.update(task, completed=len(window))
        return False

    def set_epoch(self, epoch: int) -> None:
        with self.epoch.get_lock():
            self.epoch.value = epoch
