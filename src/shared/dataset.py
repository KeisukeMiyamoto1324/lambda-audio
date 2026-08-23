import math
import random
from collections.abc import Iterator
from typing import Any

from datasets import IterableDataset, load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import IterableDataset as TorchIterableDataset
from torch.utils.data import get_worker_info

from src.shared.config import BaseTrainingConfig


class ValidAudioExample:
    def __init__(self, max_audio_samples: int) -> None:
        self.max_audio_samples = max_audio_samples

    def __call__(self, example: dict[str, Any]) -> bool:
        audio = example["audio"]
        transcription = example["transcription"]
        sample_count = len(audio["array"])

        return 0 < sample_count <= self.max_audio_samples and bool(transcription.strip())


class LengthBucketBatchDataset(TorchIterableDataset[list[dict[str, Any]]]):
    def __init__(
        self,
        dataset: IterableDataset,
        batch_size: int,
        buffer_multiplier: int,
        seed: int,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.buffer_size = batch_size * buffer_multiplier
        self.audio_window_size = batch_size * math.isqrt(buffer_multiplier)
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterator[list[dict[str, Any]]]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        random_generator = random.Random(self.seed + self.epoch + worker_id)
        buffer: list[dict[str, Any]] = []

        # ---------------------------------------------------------
        # Group similar audio lengths, then similar text lengths.
        # ---------------------------------------------------------
        for example in self.dataset:
            buffer.append(example)
            if len(buffer) < self.buffer_size:
                continue

            yield from self._create_batches(buffer, random_generator)
            buffer = []

        if buffer:
            yield from self._create_batches(buffer, random_generator)

    def _create_batches(
        self,
        buffer: list[dict[str, Any]],
        random_generator: random.Random,
    ) -> Iterator[list[dict[str, Any]]]:
        buffer.sort(key=lambda example: len(example["audio"]["array"]))
        batches: list[list[dict[str, Any]]] = []

        for start in range(0, len(buffer), self.audio_window_size):
            audio_window = buffer[start : start + self.audio_window_size]
            audio_window.sort(key=lambda example: len(example["transcription"]))

            for batch_start in range(0, len(audio_window), self.batch_size):
                batches.append(audio_window[batch_start : batch_start + self.batch_size])

        random_generator.shuffle(batches)
        yield from batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.dataset.set_epoch(epoch)


def load_streaming_datasets(
    config: BaseTrainingConfig,
    rank: int,
    world_size: int,
) -> tuple[IterableDataset, IterableDataset]:
    # ---------------------------------------------------------
    # Stream and shuffle samples without storing the corpus locally.
    # ---------------------------------------------------------
    dataset = load_dataset(
        config.dataset_name,
        config.dataset_config,
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    if not isinstance(dataset, IterableDataset):
        raise TypeError("A streaming IterableDataset is required.")

    max_audio_samples = int(config.sampling_rate * config.max_audio_seconds)
    dataset = dataset.filter(ValidAudioExample(max_audio_samples))
    dataset = dataset.shuffle(
        seed=config.seed,
        buffer_size=config.shuffle_buffer_size,
    )

    # ---------------------------------------------------------
    # Reserve the first shuffled samples for deterministic validation.
    # ---------------------------------------------------------
    validation_dataset = dataset.take(config.validation_samples)
    training_dataset = dataset.skip(config.validation_samples)
    training_dataset = split_dataset_by_node(training_dataset, rank, world_size)

    return training_dataset, validation_dataset
