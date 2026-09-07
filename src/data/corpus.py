from collections.abc import Iterator
from pathlib import Path
from typing import cast

import numpy as np
from datasets import IterableDataset, load_dataset
from datasets.distributed import split_dataset_by_node
from huggingface_hub import snapshot_download
from torch.utils.data import IterableDataset as TorchIterableDataset

from src.data.types import TranscribedAudioExample

DATASET_ID = "KeisukeMiyamoto/lambda-audio-courpus-1"


class LambdaAudioDataset(TorchIterableDataset[TranscribedAudioExample]):
    def __init__(self, dataset: IterableDataset) -> None:
        self.dataset = dataset

    def __iter__(self) -> Iterator[TranscribedAudioExample]:
        for row in self.dataset:
            samples = row["audio"].get_all_samples()
            yield {
                "audio_array": samples.data.squeeze(0)
                .numpy()
                .astype(
                    np.float32,
                    copy=False,
                ),
                "transcription": cast(str, row["text"]),
            }

    def set_epoch(self, epoch: int) -> None:
        self.dataset.set_epoch(epoch)


def prepare_dataset(token: str, cache_dir: Path) -> Path:
    snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        token=token,
        local_dir=cache_dir,
    )
    return cache_dir


def load_lambda_audio_datasets(
    cache_dir: Path,
    global_rank: int,
    world_size: int,
) -> tuple[LambdaAudioDataset, LambdaAudioDataset]:
    training_files = sorted((cache_dir / "train").glob("*.parquet"))
    validation_files = sorted((cache_dir / "validation").glob("*.parquet"))
    if not training_files:
        raise FileNotFoundError("The training dataset is not cached.")
    if not validation_files:
        raise FileNotFoundError("The validation dataset is not cached.")

    training = load_dataset(
        "parquet",
        data_files={"train": [str(path) for path in training_files]},
        split="train",
        streaming=True,
    ).select_columns(["audio", "text"])
    validation = load_dataset(
        "parquet",
        data_files={"validation": [str(path) for path in validation_files]},
        split="validation",
        streaming=True,
    ).select_columns(["audio", "text"])

    training = cast(
        IterableDataset,
        split_dataset_by_node(training, global_rank, world_size),
    )
    validation = cast(
        IterableDataset,
        split_dataset_by_node(validation, global_rank, world_size),
    )
    return LambdaAudioDataset(training), LambdaAudioDataset(validation)
