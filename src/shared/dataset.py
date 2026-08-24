from collections.abc import Iterator
from typing import Any, cast

from datasets import IterableDataset, concatenate_datasets, load_dataset
from torch.utils.data import IterableDataset as TorchIterableDataset


DATASET_ID = "litagin/reazon-speech-v2-clone"


class AudioBatchDataset(TorchIterableDataset[list[dict[str, Any]]]):
    def __init__(
        self,
        dataset: IterableDataset,
        max_batch_audio_samples: int,
        max_batch_examples: int,
    ) -> None:
        self.dataset = dataset
        self.max_batch_audio_samples = max_batch_audio_samples
        self.max_batch_examples = max_batch_examples

    def __iter__(self) -> Iterator[list[dict[str, Any]]]:
        batch: list[dict[str, Any]] = []
        batch_audio_samples = 0

        for example in self.dataset:
            expected_file_name = f"data/{example['__key__']}.flac"
            if example["file_name"] != expected_file_name:
                raise ValueError("Audio and transcription keys do not match.")

            audio_array = example["flac"]["array"]
            audio_samples = len(audio_array)
            if audio_samples > self.max_batch_audio_samples:
                raise ValueError("An audio example exceeds the batch audio limit.")

            if batch and (
                batch_audio_samples + audio_samples > self.max_batch_audio_samples
                or len(batch) >= self.max_batch_examples
            ):
                yield batch
                batch = []
                batch_audio_samples = 0

            batch.append(
                {
                    "audio_array": audio_array,
                    "transcription": example["transcription"].strip(),
                }
            )
            batch_audio_samples += audio_samples

        if batch:
            yield batch

    def set_epoch(self, epoch: int) -> None:
        self.dataset.set_epoch(epoch)


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
    metadata_dataset = cast(
        IterableDataset,
        load_dataset(
            "csv",
            data_files=f"hf://datasets/{DATASET_ID}/metadata.csv",
            split="train",
            streaming=True,
            token=token,
        ),
    )
    dataset = concatenate_datasets([audio_dataset, metadata_dataset], axis=1)
    validation_dataset = dataset.take(validation_samples)
    training_dataset = dataset.skip(validation_samples).shuffle(
        seed=seed,
        buffer_size=shuffle_buffer_size,
    )
    return training_dataset, validation_dataset
