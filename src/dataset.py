from typing import Any

from datasets import IterableDataset, load_dataset

from src.config import TrainingConfig


class ValidAudioExample:
    def __init__(self, max_audio_samples: int) -> None:
        self.max_audio_samples = max_audio_samples

    def __call__(self, example: dict[str, Any]) -> bool:
        audio = example["audio"]
        transcription = example["transcription"]
        sample_count = len(audio["array"])

        return 0 < sample_count <= self.max_audio_samples and bool(transcription.strip())


def load_streaming_datasets(
    config: TrainingConfig,
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

    return training_dataset, validation_dataset
