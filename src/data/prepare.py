import argparse
from pathlib import Path
from typing import Any

from datasets import Audio, Dataset, DatasetDict, load_dataset

from src.data.config import DataPreparationConfig
from src.data.schema import create_prepared_features
from src.shared.environment import load_hf_token


class PrepareAudioExample:
    def __init__(self, sampling_rate: int) -> None:
        self.sampling_rate = sampling_rate

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        audio_array = example["audio"]["array"]
        transcription = example["transcription"].strip()
        return {
            "audio": {
                "array": audio_array,
                "sampling_rate": self.sampling_rate,
            },
            "transcription": transcription,
            "audio_length": len(audio_array),
            "text_length": len(transcription),
        }


class ValidPreparedExample:
    def __init__(self, max_audio_samples: int) -> None:
        self.max_audio_samples = max_audio_samples

    def __call__(self, example: dict[str, Any]) -> bool:
        return (
            0 < example["audio_length"] <= self.max_audio_samples
            and example["text_length"] > 0
        )


def parse_arguments() -> DataPreparationConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/prepared/reazonspeech-small"),
    )
    parser.add_argument("--dataset-name", default="reazon-research/reazonspeech")
    parser.add_argument("--dataset-config", default="small")
    parser.add_argument("--sampling-rate", type=int, default=16_000)
    parser.add_argument("--max-audio-seconds", type=float, default=30.0)
    parser.add_argument("--validation-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-proc", type=int, default=4)
    parser.add_argument("--max-shard-size", default="1GB")
    arguments = parser.parse_args()

    if arguments.validation_samples < 1:
        parser.error("--validation-samples must be at least 1.")
    if arguments.num_proc < 1:
        parser.error("--num-proc must be at least 1.")

    return DataPreparationConfig(
        output_dir=arguments.output_dir,
        dataset_name=arguments.dataset_name,
        dataset_config=arguments.dataset_config,
        sampling_rate=arguments.sampling_rate,
        max_audio_seconds=arguments.max_audio_seconds,
        validation_samples=arguments.validation_samples,
        seed=arguments.seed,
        num_proc=arguments.num_proc,
        max_shard_size=arguments.max_shard_size,
    )


def prepare_dataset(config: DataPreparationConfig) -> None:
    source = load_dataset(
        config.dataset_name,
        config.dataset_config,
        split="train",
        token=load_hf_token(),
    )
    if not isinstance(source, Dataset):
        raise TypeError("A map-style Dataset is required.")

    source = source.cast_column("audio", Audio(sampling_rate=config.sampling_rate))
    prepared = source.map(
        PrepareAudioExample(config.sampling_rate),
        remove_columns=source.column_names,
        features=create_prepared_features(config.sampling_rate),
        num_proc=config.num_proc,
        desc="Preparing audio",
    )
    max_audio_samples = int(config.sampling_rate * config.max_audio_seconds)
    prepared = prepared.filter(
        ValidPreparedExample(max_audio_samples),
        num_proc=config.num_proc,
        desc="Filtering audio",
    )
    if len(prepared) <= config.validation_samples:
        raise ValueError("The prepared dataset is too small for the validation split.")

    prepared = prepared.shuffle(seed=config.seed)
    validation = prepared.select(range(config.validation_samples))
    training = prepared.select(range(config.validation_samples, len(prepared)))
    dataset = DatasetDict({"train": training, "validation": validation})
    dataset.save_to_disk(
        config.output_dir,
        max_shard_size=config.max_shard_size,
        num_proc=config.num_proc,
    )


def main() -> None:
    prepare_dataset(parse_arguments())


if __name__ == "__main__":
    main()
