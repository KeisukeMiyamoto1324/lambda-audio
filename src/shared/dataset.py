import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import Sampler


class PreparedAudioDataset(TorchDataset[dict[str, Any]]):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.dataset[index]
        return {
            "audio_array": example["audio"]["array"],
            "transcription": example["transcription"],
        }


class LengthBucketBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        audio_lengths: list[int],
        text_lengths: list[int],
        max_batch_audio_samples: int,
        max_batch_examples: int,
        bucket_window_size: int,
        seed: int,
        rank: int,
        world_size: int,
        shuffle: bool,
    ) -> None:
        self.audio_lengths = audio_lengths
        self.text_lengths = text_lengths
        self.max_batch_audio_samples = max_batch_audio_samples
        self.max_batch_examples = max_batch_examples
        self.bucket_window_size = bucket_window_size
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.epoch = 0

        if len(audio_lengths) != len(text_lengths):
            raise ValueError("Audio and text length counts must match.")
        if max_batch_examples < 1:
            raise ValueError("The maximum batch example count must be positive.")
        if max(audio_lengths, default=0) > max_batch_audio_samples:
            raise ValueError("An audio example exceeds the batch audio limit.")

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._create_batches()

    def __len__(self) -> int:
        return len(self._create_batches())

    def _create_batches(self) -> list[list[int]]:
        random_generator = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.audio_lengths)))
        if self.shuffle:
            random_generator.shuffle(indices)

        batches: list[list[int]] = []
        indices.sort(key=self.audio_lengths.__getitem__)
        for start in range(0, len(indices), self.bucket_window_size):
            window = indices[start : start + self.bucket_window_size]
            window.sort(key=self.text_lengths.__getitem__)
            batch: list[int] = []
            batch_audio_samples = 0

            for index in window:
                audio_samples = self.audio_lengths[index]
                if batch and (
                    batch_audio_samples + audio_samples
                    > self.max_batch_audio_samples
                    or len(batch) >= self.max_batch_examples
                ):
                    batches.append(batch)
                    batch = []
                    batch_audio_samples = 0

                batch.append(index)
                batch_audio_samples += audio_samples

            if batch:
                batches.append(batch)

        if self.shuffle:
            random_generator.shuffle(batches)
        distributed_batch_count = len(batches) - len(batches) % self.world_size
        return batches[self.rank : distributed_batch_count : self.world_size]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


def load_prepared_datasets(data_dir: Path) -> tuple[Dataset, Dataset]:
    dataset = load_from_disk(data_dir)
    if not isinstance(dataset, DatasetDict):
        raise TypeError("A prepared DatasetDict is required.")
    if "train" not in dataset or "validation" not in dataset:
        raise ValueError("Prepared data must contain train and validation splits.")

    return dataset["train"], dataset["validation"]
