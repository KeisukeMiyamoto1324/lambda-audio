import multiprocessing
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

from datasets import IterableDataset, load_dataset
from torch.utils.data import IterableDataset as TorchIterableDataset, get_worker_info

from src.shared.metadata import DATASET_ID, MetadataReader


class TranscribedAudioDataset(TorchIterableDataset[dict[str, Any]]):
    def __init__(
        self,
        dataset: IterableDataset,
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
        max_batch_audio_samples: int,
        max_batch_examples: int,
        bucket_buffer_size: int,
        seed: int,
    ) -> None:
        self.dataset = dataset
        self.max_batch_audio_samples = max_batch_audio_samples
        self.max_batch_examples = max_batch_examples
        self.bucket_buffer_size = bucket_buffer_size
        self.seed = seed
        self.epoch = multiprocessing.Value("q", 0)

    def __iter__(self) -> Iterator[list[dict[str, Any]]]:
        epoch = self.epoch.value
        self.dataset.set_epoch(epoch)
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        generator = random.Random(self.seed + epoch + worker_id)
        window: list[dict[str, Any]] = []

        for example in self.dataset:
            window.append(example)
            if len(window) < self.bucket_buffer_size:
                continue
            yield from self._batch_window(window, generator)
            window = []

        if window:
            yield from self._batch_window(window, generator)

    def _batch_window(
        self,
        window: list[dict[str, Any]],
        generator: random.Random,
    ) -> Iterator[list[dict[str, Any]]]:
        window.sort(key=lambda example: len(example["audio_array"]))
        batches: list[list[dict[str, Any]]] = []
        batch: list[dict[str, Any]] = []
        batch_audio_samples = 0

        for example in window:
            audio_samples = len(example["audio_array"])
            if audio_samples > self.max_batch_audio_samples:
                raise ValueError("An audio example exceeds the batch audio limit.")

            if batch and (
                batch_audio_samples + audio_samples > self.max_batch_audio_samples
                or len(batch) >= self.max_batch_examples
            ):
                batches.append(batch)
                batch = []
                batch_audio_samples = 0

            batch.append(example)
            batch_audio_samples += audio_samples

        if batch:
            batches.append(batch)

        generator.shuffle(batches)
        yield from batches

    def set_epoch(self, epoch: int) -> None:
        with self.epoch.get_lock():
            self.epoch.value = epoch


def load_reazon_speech_datasets(
    token: str,
    validation_samples: int,
    shuffle_buffer_size: int,
    seed: int,
) -> tuple[IterableDataset, IterableDataset]:
    audio_dataset = cast(
        IterableDataset,
        load_dataset(
            DATASET_ID,
            split="train",
            streaming=True,
            token=token,
        ),
    )
    validation_dataset = audio_dataset.take(validation_samples)
    training_dataset = audio_dataset.skip(validation_samples).shuffle(
        seed=seed,
        buffer_size=shuffle_buffer_size,
    )
    return training_dataset, validation_dataset
