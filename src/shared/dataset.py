import math
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
        batch_size: int,
        buffer_multiplier: int,
        seed: int,
        rank: int,
        world_size: int,
        shuffle: bool,
    ) -> None:
        self.audio_lengths = audio_lengths
        self.text_lengths = text_lengths
        self.batch_size = batch_size
        self.buffer_size = batch_size * buffer_multiplier
        self.audio_window_size = batch_size * math.isqrt(buffer_multiplier)
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.epoch = 0

        if len(audio_lengths) != len(text_lengths):
            raise ValueError("Audio and text length counts must match.")

    def __iter__(self) -> Iterator[list[int]]:
        random_generator = random.Random(self.seed + self.epoch)
        indices = list(range(self.rank, len(self.audio_lengths), self.world_size))
        if self.shuffle:
            random_generator.shuffle(indices)

        batches: list[list[int]] = []
        for start in range(0, len(indices), self.buffer_size):
            buffer = indices[start : start + self.buffer_size]
            buffer.sort(key=self.audio_lengths.__getitem__)

            for window_start in range(0, len(buffer), self.audio_window_size):
                audio_window = buffer[
                    window_start : window_start + self.audio_window_size
                ]
                audio_window.sort(key=self.text_lengths.__getitem__)

                for batch_start in range(0, len(audio_window), self.batch_size):
                    batches.append(
                        audio_window[batch_start : batch_start + self.batch_size]
                    )

        if self.shuffle:
            random_generator.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        sample_count = len(range(self.rank, len(self.audio_lengths), self.world_size))
        return math.ceil(sample_count / self.batch_size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


def load_prepared_datasets(data_dir: Path) -> tuple[Dataset, Dataset]:
    dataset = load_from_disk(data_dir)
    if not isinstance(dataset, DatasetDict):
        raise TypeError("A prepared DatasetDict is required.")
    if "train" not in dataset or "validation" not in dataset:
        raise ValueError("Prepared data must contain train and validation splits.")

    return dataset["train"], dataset["validation"]
