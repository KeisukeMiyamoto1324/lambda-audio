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


def prepare_validation_data(token: str, cache_dir: Path) -> Path:
    snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        token=token,
        allow_patterns="validation/*.parquet",
        local_dir=cache_dir,
    )
    return cache_dir / "validation"


def load_lambda_audio_datasets(
    token: str,
    cache_dir: Path,
    shuffle_buffer_size: int,
    seed: int,
    global_rank: int,
    world_size: int,
) -> tuple[LambdaAudioDataset, LambdaAudioDataset]:
    validation_files = sorted((cache_dir / "validation").glob("*.parquet"))
    if not validation_files:
        raise FileNotFoundError("The validation dataset is not cached.")

    training = load_dataset(
        DATASET_ID,
        split="train",
        streaming=True,
        token=token,
    ).select_columns(["audio", "text"])
    validation = load_dataset(
        "parquet",
        data_files={"validation": [str(path) for path in validation_files]},
        split="validation",
        streaming=True,
    ).select_columns(["audio", "text"])

    training = training.shuffle(buffer_size=shuffle_buffer_size, seed=seed)
    training = cast(
        IterableDataset,
        split_dataset_by_node(training, global_rank, world_size),
    )
    validation = cast(
        IterableDataset,
        split_dataset_by_node(validation, global_rank, world_size),
    )
    return LambdaAudioDataset(training), LambdaAudioDataset(validation)
