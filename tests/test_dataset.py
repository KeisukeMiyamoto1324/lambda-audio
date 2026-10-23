from collections.abc import Iterator

import numpy as np
import pytest

from src.data.dataset import AudioBatchDataset
from src.data.types import TranscribedAudioExample


def create_audio_example(
    transcription: str,
    audio_samples: int,
) -> TranscribedAudioExample:
    return {
        "audio_array": np.ones(audio_samples, dtype=np.float32),
        "transcription": transcription,
        "output_language": "ja",
    }


class ExampleDataset:
    def __init__(self, examples: list[tuple[str, int]]) -> None:
        self.examples = examples
        self.epoch = 0

    def __iter__(self) -> Iterator[TranscribedAudioExample]:
        for transcription, samples in self.examples:
            yield create_audio_example(transcription, samples)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


def test_audio_batch_dataset_limits_padded_audio_samples() -> None:
    source = ExampleDataset([("音声", 8), ("音声", 8), ("音声", 4), ("音声", 4)])
    dataset = AudioBatchDataset(
        source,
        max_batch_padded_audio_samples=12,
        bucket_buffer_size=4,
        seed=42,
    )

    batches = list(dataset)

    assert sorted(len(batch) for batch in batches) == [1, 1, 2]
    assert all(len(batch) * max(len(example["audio_array"]) for example in batch) <= 12 for batch in batches)
    assert all(example["transcription"] == "音声" for batch in batches for example in batch)


def test_audio_batch_dataset_reduces_weighted_padding_ratio() -> None:
    examples = [(f"{index:03d}", samples) for index, samples in enumerate(range(10, 522))]
    dataset = AudioBatchDataset(
        ExampleDataset(examples),
        max_batch_padded_audio_samples=20_000,
        bucket_buffer_size=512,
        seed=42,
    )

    batches = list(dataset)
    valid_samples = sum(len(example["audio_array"]) for batch in batches for example in batch)
    padded_samples = sum(len(batch) * max(len(example["audio_array"]) for example in batch) for batch in batches)

    assert padded_samples / valid_samples < 1.5


def test_audio_batch_dataset_refills_between_ready_batches() -> None:
    examples = [(str(index), 4) for index in range(8)]
    reads: list[str] = []

    class TrackingDataset(ExampleDataset):
        def __iter__(self) -> Iterator[TranscribedAudioExample]:
            for transcription, samples in self.examples:
                reads.append(transcription)
                yield create_audio_example(transcription, samples)

    dataset = AudioBatchDataset(
        TrackingDataset(examples),
        max_batch_padded_audio_samples=8,
        bucket_buffer_size=4,
        seed=42,
    )
    batches = iter(dataset)

    next(batches)
    assert len(reads) == 4
    next(batches)
    assert len(reads) == 6
    next(batches)
    assert len(reads) == 8


def test_audio_batch_dataset_rejects_oversized_audio() -> None:
    dataset = AudioBatchDataset(
        ExampleDataset([("audio", 13)]),
        max_batch_padded_audio_samples=12,
        bucket_buffer_size=4,
        seed=42,
    )

    with pytest.raises(ValueError, match="exceeds the padded audio limit"):
        list(dataset)


@pytest.mark.parametrize("buffer_size", [1, 3, 7])
def test_audio_batches_carry_remainder_without_losing_examples(buffer_size: int) -> None:
    source = ExampleDataset([(str(i), 3) for i in range(23)])
    dataset = AudioBatchDataset(source, max_batch_padded_audio_samples=12, bucket_buffer_size=buffer_size, seed=42)
    batches = list(dataset)
    assert sorted(len(batch) for batch in batches) == [3, 4, 4, 4, 4, 4]
    assert sorted(int(x["transcription"]) for batch in batches for x in batch) == list(range(23))
    assert all(len(batch) * max(len(x["audio_array"]) for x in batch) <= 12 for batch in batches)


def test_audio_batches_eventually_emit_longest_remainder() -> None:
    examples = [("long", 11), *[(str(i), 2) for i in range(31)]]
    dataset = AudioBatchDataset(ExampleDataset(examples), max_batch_padded_audio_samples=12, bucket_buffer_size=3, seed=42)
    batches = list(dataset)
    emitted = [x["transcription"] for batch in batches for x in batch]
    assert sorted(emitted) == sorted(name for name, _ in examples)
    assert all(len(batch) * max(len(x["audio_array"]) for x in batch) <= 12 for batch in batches)


def test_audio_batches_do_not_defer_long_examples_until_epoch_end() -> None:
    from itertools import islice

    examples = [("long", 11), *[(str(i), 2) for i in range(1000)]]
    dataset = AudioBatchDataset(ExampleDataset(examples), max_batch_padded_audio_samples=12, bucket_buffer_size=3, seed=42)
    first_batches = list(islice(dataset, 3))
    assert any(x["transcription"] == "long" for batch in first_batches for x in batch)
