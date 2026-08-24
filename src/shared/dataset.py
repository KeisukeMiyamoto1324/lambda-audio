import multiprocessing
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol, cast

from torch.utils.data import IterableDataset as TorchIterableDataset, get_worker_info

from src.shared.metadata import MetadataReader


class EpochAudioDataset(Protocol):
    def __iter__(self) -> Iterator[dict[str, Any]]: ...

    def set_epoch(self, epoch: int) -> None: ...


class TranscribedAudioDataset(TorchIterableDataset[dict[str, Any]]):
    def __init__(
        self,
        dataset: EpochAudioDataset,
        metadata_index_path: Path,
    ) -> None:
        self.dataset = dataset
        self.metadata_index_path = metadata_index_path

    def __iter__(self) -> Iterator[dict[str, Any]]:
        reader = MetadataReader(self.metadata_index_path)
        try:
            for example in self.dataset:
                audio_key = cast(str, example["__key__"])
                yield {
                    "audio_array": example["flac"]["array"],
                    "transcription": reader.transcription(audio_key),
                }
        finally:
            reader.close()

    def set_epoch(self, epoch: int) -> None:
        self.dataset.set_epoch(epoch)


class AudioBatchDataset(TorchIterableDataset[list[dict[str, Any]]]):
    def __init__(
        self,
        dataset: TranscribedAudioDataset,
        *,
        max_batch_padded_audio_samples: int,
        max_batch_attention_cost: int,
        encoder_frame_samples: int,
        bucket_buffer_size: int,
        seed: int,
    ) -> None:
        self.dataset = dataset
        self.max_batch_padded_audio_samples = max_batch_padded_audio_samples
        self.max_batch_attention_cost = max_batch_attention_cost
        self.encoder_frame_samples = encoder_frame_samples
        self.bucket_buffer_size = bucket_buffer_size
        self.seed = seed
        self.epoch = multiprocessing.Value("q", 0)

        if max_batch_padded_audio_samples < 1:
            raise ValueError("The padded audio limit must be positive.")
        if max_batch_attention_cost < 1:
            raise ValueError("The attention cost limit must be positive.")
        if encoder_frame_samples < 1:
            raise ValueError("The encoder frame size must be positive.")
        if bucket_buffer_size < 1:
            raise ValueError("The bucket buffer size must be positive.")

    def __iter__(self) -> Iterator[list[dict[str, Any]]]:
        epoch = self.epoch.value
        self.dataset.set_epoch(epoch)
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        generator = random.Random(self.seed + epoch + worker_id)
        source = iter(self.dataset)
        window: list[dict[str, Any]] = []
        source_exhausted = self._fill_window(
            window,
            source,
            self.bucket_buffer_size,
        )

        while window:
            batches = self._create_batches(window, generator)
            next_window: list[dict[str, Any]] = []
            for batch_index, batch in enumerate(batches, start=1):
                yield batch
                if source_exhausted:
                    continue
                refill_target = (
                    batch_index * self.bucket_buffer_size // len(batches)
                )
                source_exhausted = self._fill_window(
                    next_window,
                    source,
                    refill_target,
                )
            window = next_window

    def _create_batches(
        self,
        window: list[dict[str, Any]],
        generator: random.Random,
    ) -> list[list[dict[str, Any]]]:
        window.sort(key=lambda example: len(example["audio_array"]))
        batches: list[list[dict[str, Any]]] = []
        batch: list[dict[str, Any]] = []

        for example in window:
            # ---------------------------------------------------------
            # Bound padded feature extraction and quadratic attention.
            # ---------------------------------------------------------
            audio_samples = len(example["audio_array"])
            encoder_frames = (
                audio_samples + self.encoder_frame_samples - 1
            ) // self.encoder_frame_samples
            if (
                audio_samples > self.max_batch_padded_audio_samples
                or encoder_frames**2 > self.max_batch_attention_cost
            ):
                raise ValueError("An audio example exceeds the batch limits.")

            batch_size = len(batch) + 1
            padded_audio_samples = batch_size * audio_samples
            attention_cost = batch_size * encoder_frames**2

            if (
                batch
                and (
                    padded_audio_samples > self.max_batch_padded_audio_samples
                    or attention_cost > self.max_batch_attention_cost
                )
            ):
                batches.append(batch)
                batch = []

            batch.append(example)

        if batch:
            batches.append(batch)

        generator.shuffle(batches)
        return batches

    @staticmethod
    def _fill_window(
        window: list[dict[str, Any]],
        source: Iterator[dict[str, Any]],
        target_size: int,
    ) -> bool:
        while len(window) < target_size:
            try:
                window.append(next(source))
            except StopIteration:
                return True
        return False

    def set_epoch(self, epoch: int) -> None:
        with self.epoch.get_lock():
            self.epoch.value = epoch
