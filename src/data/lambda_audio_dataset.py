import logging
import multiprocessing
import random
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from itertools import chain
from math import ceil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, cast

import av
import numpy as np
import pyarrow.parquet as pq
from datasets import Audio, Dataset, IterableDataset, load_dataset
from datasets.distributed import split_dataset_by_node
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import disable_progress_bars
from torch.utils.data import get_worker_info

from src.data.text import normalize_transcription
from src.data.types import TranscribedAudioExample
from src.shared.languages import OUTPUT_LANGUAGES, OutputLanguage

DATASET_ID = "KeisukeMiyamoto/lambda-audio-courpus-multilingual"
VALIDATION_CACHE_RATIO = 0.1
VALIDATION_SELECTION_SEED = 42
PARQUET_BATCH_SIZE = 128
DATASET_COLUMNS = ["id", "audio", *OUTPUT_LANGUAGES]
DatasetSplit = Literal["train", "validation"]
LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------
# Read cached validation data and temporary training shards.
# ---------------------------------------------------------
class LambdaAudioDataset:
    def __init__(self, dataset: IterableDataset, sampling_rate: int, seed: int) -> None:
        self.dataset = dataset
        self.sampling_rate = sampling_rate
        self.seed = seed

    def __iter__(self) -> Iterator[TranscribedAudioExample]:
        yield from _read_examples(self.dataset, self.sampling_rate, "validation", self.seed, None)

    def set_epoch(self, epoch: int) -> None:
        self.dataset.set_epoch(epoch)


class TrainLambdaAudioDataset:
    def __init__(
        self,
        token: str,
        shard_files: list[str],
        sampling_rate: int,
        seed: int,
        global_rank: int,
        world_size: int,
    ) -> None:
        self.token = token
        self.shard_files = shard_files
        self.sampling_rate = sampling_rate
        self.seed = seed
        self.global_rank = global_rank
        self.world_size = world_size
        self.epoch = multiprocessing.Value("q", 0)

    def __iter__(self) -> Iterator[TranscribedAudioExample]:
        # ---------------------------------------------------------
        # Shuffle shards once per epoch and divide them across workers.
        # ---------------------------------------------------------
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers
        consumer_index = self.global_rank * worker_count + worker_id
        consumer_count = self.world_size * worker_count
        shards = list(self.shard_files)
        random.Random(self.seed + self.epoch.value).shuffle(shards)
        shards = shards[consumer_index::consumer_count]
        if not shards:
            return

        # ---------------------------------------------------------
        # Download the next shard while reading the current one.
        # Wait for downloads before removing the temporary directory.
        # ---------------------------------------------------------
        generator = random.Random(f"{self.seed}:{self.epoch.value}:{self.global_rank}:{worker_id}")
        with TemporaryDirectory(prefix="lambda-audio-train-") as directory, ThreadPoolExecutor(max_workers=1) as executor:
            download_dir = Path(directory)
            pending = executor.submit(self._download_shard, shards[0], download_dir)
            for index in range(len(shards)):
                shard_path = pending.result()
                if index + 1 < len(shards):
                    pending = executor.submit(self._download_shard, shards[index + 1], download_dir)
                yield from self._read_shard(shard_path, generator)

    def _download_shard(self, shard_file: str, directory: Path) -> Path:
        with disable_progress_bars():
            path = hf_hub_download(
                repo_id=DATASET_ID,
                filename=shard_file,
                repo_type="dataset",
                token=self.token,
                local_dir=directory,
            )
        return Path(path)

    def _read_shard(self, shard_path: Path, generator: random.Random) -> Iterator[TranscribedAudioExample]:
        # ---------------------------------------------------------
        # Read bounded batches and delete the shard even on early exit.
        # ---------------------------------------------------------
        try:
            with pq.ParquetFile(shard_path) as parquet_file:
                batches = parquet_file.iter_batches(batch_size=PARQUET_BATCH_SIZE, columns=DATASET_COLUMNS)
                rows = chain.from_iterable(batch.to_pylist() for batch in batches)
                yield from _read_examples(rows, self.sampling_rate, shard_path.name, self.seed, generator)
        finally:
            shard_path.unlink()

    def set_epoch(self, epoch: int) -> None:
        with self.epoch.get_lock():
            self.epoch.value = epoch


# ---------------------------------------------------------
# Build datasets and cache the fixed validation subset.
# ---------------------------------------------------------
def load_lambda_audio_dataset(
    token: str,
    split: DatasetSplit,
    cache_dir: Path,
    sampling_rate: int,
    seed: int,
    global_rank: int,
    world_size: int,
    num_workers: int,
) -> LambdaAudioDataset | TrainLambdaAudioDataset:
    if split == "train":
        return TrainLambdaAudioDataset(
            token=token,
            shard_files=list_shard_files(token, "train"),
            sampling_rate=sampling_rate,
            seed=seed,
            global_rank=global_rank,
            world_size=world_size,
        )

    cached = prepare_validation_dataset(token, cache_dir)
    dataset = cached.to_iterable_dataset(num_shards=world_size * num_workers)
    dataset = dataset.select_columns(DATASET_COLUMNS)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate, decode=False))
    dataset = split_dataset_by_node(dataset, global_rank, world_size)
    return LambdaAudioDataset(dataset, sampling_rate, seed)


def prepare_validation_dataset(token: str, cache_dir: Path) -> Dataset:
    return cast(
        Dataset,
        load_dataset(
            DATASET_ID,
            split="validation",
            data_files={"validation": list_validation_cache_files(token)},
            streaming=False,
            token=token,
            cache_dir=str(cache_dir),
            verification_mode="no_checks",
        ),
    )


def list_shard_files(token: str, split: DatasetSplit) -> list[str]:
    files = HfApi(token=token).list_repo_files(DATASET_ID, repo_type="dataset")
    shards = sorted(path for path in files if path.startswith(f"{split}/") and path.endswith(".parquet"))
    if not shards:
        raise RuntimeError(f"The {split} split has no Parquet shards.")
    return shards


def list_validation_cache_files(token: str) -> list[str]:
    # ---------------------------------------------------------
    # Use the same validation subset for every run and worker.
    # ---------------------------------------------------------
    shards = list_shard_files(token, "validation")
    count = ceil(len(shards) * VALIDATION_CACHE_RATIO)
    return sorted(random.Random(VALIDATION_SELECTION_SEED).sample(shards, count))


# ---------------------------------------------------------
# Decode both splits through the same audio and text handling.
# ---------------------------------------------------------
def _read_examples(
    rows: Iterable[dict[str, Any]],
    sampling_rate: int,
    source: str,
    seed: int,
    generator: random.Random | None,
) -> Iterator[TranscribedAudioExample]:
    # ---------------------------------------------------------
    # Sample training languages and keep validation choices stable by ID.
    # ---------------------------------------------------------
    for index, row in enumerate(rows):
        selector = random.Random(f"{seed}:{row['id']}") if generator is None else generator
        language = selector.choice(OUTPUT_LANGUAGES)
        text = normalize_transcription(row[language])
        try:
            example = _decode_example(row["audio"]["bytes"], text, language, sampling_rate)
        except (av.error.FFmpegError, ValueError, IndexError) as error:
            LOGGER.warning("Skipping invalid audio at %s row %s: %s", source, index, error)
            continue
        yield example


def _decode_example(audio: bytes, text: str, language: OutputLanguage, sampling_rate: int) -> TranscribedAudioExample:
    # ---------------------------------------------------------
    # Require mono audio at the expected rate and decode to float32.
    # ---------------------------------------------------------
    with av.open(BytesIO(audio), mode="r") as container:
        stream = container.streams.audio[0]
        if stream.sample_rate != sampling_rate:
            raise ValueError(f"Expected {sampling_rate} Hz audio, got {stream.sample_rate} Hz.")
        if stream.channels != 1:
            raise ValueError("Audio must be mono.")

        resampler = av.AudioResampler(format="fltp", layout="mono", rate=sampling_rate)
        arrays = []
        for frame in container.decode(stream):
            arrays.extend(converted.to_ndarray()[0] for converted in resampler.resample(frame))
        arrays.extend(converted.to_ndarray()[0] for converted in resampler.resample(None))

    if not arrays:
        raise ValueError("Audio is empty.")
    return {
        "audio_array": np.concatenate(arrays).astype(np.float32, copy=False),
        "transcription": text,
        "output_language": language,
    }
