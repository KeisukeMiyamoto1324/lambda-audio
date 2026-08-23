import numpy as np
from datasets import IterableDataset

from src.shared.dataset import LengthBucketBatchDataset, ValidAudioExample


def test_valid_audio_example_checks_length_and_text() -> None:
    validator = ValidAudioExample(max_audio_samples=4)

    assert validator(
        {
            "audio_array": np.ones(4, dtype=np.float32),
            "transcription": "音声",
        }
    )
    assert not validator(
        {
            "audio_array": np.ones(5, dtype=np.float32),
            "transcription": "音声",
        }
    )


def test_length_bucket_batch_dataset_groups_audio_and_text_lengths() -> None:
    examples = [
        {
            "audio_array": np.ones(audio_length, dtype=np.float32),
            "transcription": "a" * text_length,
        }
        for audio_length, text_length in zip(
            range(1, 9),
            [40, 30, 20, 10, 80, 70, 60, 50],
            strict=True,
        )
    ]
    source = IterableDataset.from_generator(lambda: iter(examples))
    dataset = LengthBucketBatchDataset(
        source,
        batch_size=2,
        buffer_multiplier=4,
        seed=42,
    )

    batches = list(dataset)
    length_pairs = {
        tuple(len(example["transcription"]) for example in batch) for batch in batches
    }

    assert length_pairs == {(10, 20), (30, 40), (50, 60), (70, 80)}
