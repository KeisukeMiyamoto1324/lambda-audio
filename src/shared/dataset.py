from collections.abc import Iterator
from typing import Any, cast

from datasets import IterableDataset, load_dataset
from torch.utils.data import IterableDataset as TorchIterableDataset


DATASET_ID = "litagin/reazon-speech-v2-clone"


def metadata_key(example: dict[str, Any]) -> str:
    file_name = cast(str, example["file_name"])
    prefix = "data/"
    suffix = ".flac"
    if not file_name.startswith(prefix) or not file_name.endswith(suffix):
        raise ValueError(f"Invalid metadata file name: {file_name}")
    return file_name[len(prefix) : -len(suffix)]


def join_audio_and_transcriptions(
    audio_dataset: IterableDataset,
    metadata_dataset: IterableDataset,
) -> Iterator[dict[str, Any]]:
    metadata_iterator = iter(metadata_dataset)
    metadata_example = next(metadata_iterator, None)

    for audio_example in audio_dataset:
        audio_key = cast(str, audio_example["__key__"])
        while (
            metadata_example is not None
            and metadata_key(metadata_example) < audio_key
        ):
            metadata_example = next(metadata_iterator, None)

        if (
            metadata_example is None
            or metadata_key(metadata_example) != audio_key
        ):
            raise ValueError(f"Transcription not found for audio key: {audio_key}")

        yield {**audio_example, **metadata_example}
        metadata_example = next(metadata_iterator, None)


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
    dataset = IterableDataset.from_generator(
        join_audio_and_transcriptions,
        gen_kwargs={
            "audio_dataset": audio_dataset,
            "metadata_dataset": metadata_dataset,
        },
    )
    validation_dataset = dataset.take(validation_samples)
    training_dataset = dataset.skip(validation_samples).shuffle(
        seed=seed,
        buffer_size=shuffle_buffer_size,
    )
    return training_dataset, validation_dataset
