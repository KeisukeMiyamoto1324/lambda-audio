import logging
import multiprocessing
import random
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import closing
from io import BytesIO
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

DATASET_ID = "KeisukeMiyamoto/lambda-audio-courpus-1"
VALIDATION_CACHE_RATIO = 0.1
VALIDATION_SELECTION_SEED = 42
PARQUET_BATCH_SIZE = 128
DatasetSplit = Literal["train", "validation"]
LOGGER = logging.getLogger(__name__)


class LambdaAudioDataset:
    def __init__(self, dataset: IterableDataset, sampling_rate: int) -> None:
        self.dataset = dataset
        self.sampling_rate = sampling_rate

    def __iter__(self) -> Iterator[TranscribedAudioExample]:
        for row_index, row in enumerate(self.dataset):
            audio = cast(Any, row["audio"])
            example = _create_example_or_none(
                cast(bytes, audio["bytes"]),
                cast(str, row["text"]),
                self.sampling_rate,
                f"validation row {row_index}",
            )
            if example is not None:
                yield example

    def set_epoch(self, epoch: int) -> None:
        self.dataset.set_epoch(epoch)


class TrainLambdaAudioDataset:
    def __init__(
        self,
        token: str,
        shard_files: list[str],
        shuffle_buffer_size: int,
        sampling_rate: int,
        seed: int,
        global_rank: int,
        world_size: int,
    ) -> None:
        self.token = token
        self.shard_files = shard_files
        self.shuffle_buffer_size = shuffle_buffer_size
        self.sampling_rate = sampling_rate
        self.seed = seed
        self.global_rank = global_rank
        self.world_size = world_size
        self.epoch = multiprocessing.Value("q", 0)

    def __iter__(self) -> Iterator[TranscribedAudioExample]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers
        consumer_index = self.global_rank * worker_count + worker_id
        consumer_count = self.world_size * worker_count
        generator = random.Random(self.seed + self.epoch.value)
        shards = list(self.shard_files)
        generator.shuffle(shards)
        assigned_shards = shards[consumer_index::consumer_count]

        with TemporaryDirectory(prefix="lambda-audio-train-") as directory:
            with closing(self._iterate_shards(assigned_shards, Path(directory))) as examples:
                with closing(self._shuffle_examples(examples, generator)) as shuffled_examples:
                    yield from shuffled_examples

    def _iterate_shards(
        self,
        shard_files: list[str],
        directory: Path,
    ) -> Iterator[TranscribedAudioExample]:
        if not shard_files:
            return

        executor = ThreadPoolExecutor(max_workers=1)
        future: Future[Path] = executor.submit(
            self._download_shard,
            shard_files[0],
            directory,
        )
        try:
            for index, shard_file in enumerate(shard_files):
                shard_path = future.result()
                if index + 1 < len(shard_files):
                    future = executor.submit(
                        self._download_shard,
                        shard_files[index + 1],
                        directory,
                    )
                try:
                    yield from self._read_shard(shard_path)
                finally:
                    shard_path.unlink()
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    def _read_shard(self, shard_path: Path) -> Iterator[TranscribedAudioExample]:
        row_index = 0
        with pq.ParquetFile(shard_path) as parquet_file:
            for batch in parquet_file.iter_batches(
                batch_size=PARQUET_BATCH_SIZE,
                columns=["audio", "text"],
            ):
                for row in batch.to_pylist():
                    audio = cast(dict[str, Any], row["audio"])
                    example = _create_example_or_none(
                        cast(bytes, audio["bytes"]),
                        cast(str, row["text"]),
                        self.sampling_rate,
                        f"{shard_path.name} row {row_index}",
                    )
                    row_index += 1
                    if example is not None:
                        yield example

    def _shuffle_examples(
        self,
        examples: Iterator[TranscribedAudioExample],
        generator: random.Random,
    ) -> Iterator[TranscribedAudioExample]:
        buffer: list[TranscribedAudioExample] = []
        for example in examples:
            if len(buffer) < self.shuffle_buffer_size:
                buffer.append(example)
                continue
            index = generator.randrange(len(buffer))
            yield buffer[index]
            buffer[index] = example

        generator.shuffle(buffer)
        yield from buffer

    def _download_shard(self, shard_file: str, directory: Path) -> Path:
        with disable_progress_bars():
            path = hf_hub_download(
                repo_id=DATASET_ID,
                filename=shard_file,
                repo_type="dataset",
                token=self.token,
                local_dir=directory,
            )
        return Path(cast(str, path))

    def set_epoch(self, epoch: int) -> None:
        with self.epoch.get_lock():
            self.epoch.value = epoch


def load_lambda_audio_dataset(
    token: str,
    split: DatasetSplit,
    cache_dir: Path,
    shuffle_buffer_size: int,
    sampling_rate: int,
    seed: int,
    global_rank: int,
    world_size: int,
    num_workers: int,
) -> LambdaAudioDataset | TrainLambdaAudioDataset:
    if split == "train":
        return TrainLambdaAudioDataset(
            token,
            list_train_shard_files(token),
            shuffle_buffer_size,
            sampling_rate,
            seed,
            global_rank,
            world_size,
        )

    validation_files = list_validation_cache_files(token)
    cached_dataset = cast(
        Dataset,
        load_dataset(
            DATASET_ID,
            split="validation",
            data_files={"validation": validation_files},
            streaming=False,
            token=token,
            cache_dir=str(cache_dir),
            verification_mode="no_checks",
        ),
    )
    dataset = cached_dataset.to_iterable_dataset(num_shards=world_size * num_workers)
    dataset = dataset.select_columns(["audio", "text"])
    dataset = dataset.cast_column(
        "audio",
        Audio(sampling_rate=sampling_rate, decode=False),
    )
    dataset = split_dataset_by_node(dataset, global_rank, world_size)
    return LambdaAudioDataset(dataset, sampling_rate)


def prepare_validation_dataset(token: str, cache_dir: Path) -> None:
    validation_files = list_validation_cache_files(token)
    load_dataset(
        DATASET_ID,
        split="validation",
        data_files={"validation": validation_files},
        streaming=False,
        token=token,
        cache_dir=str(cache_dir),
        verification_mode="no_checks",
    )


def list_train_shard_files(token: str) -> list[str]:
    return _list_shard_files(token, "train")


def list_validation_cache_files(token: str) -> list[str]:
    validation_files = _list_shard_files(token, "validation")
    cache_shards = ceil(len(validation_files) * VALIDATION_CACHE_RATIO)
    selected_files = random.Random(VALIDATION_SELECTION_SEED).sample(validation_files, cache_shards)
    return sorted(selected_files)


def _list_shard_files(token: str, split: DatasetSplit) -> list[str]:
    files = HfApi(token=token).list_repo_files(DATASET_ID, repo_type="dataset")
    shard_files = sorted(path for path in files if path.startswith(f"{split}/") and path.endswith(".parquet"))
    if not shard_files:
        raise RuntimeError(f"The {split} split has no Parquet shards.")
    return shard_files


def _create_example(
    audio: bytes,
    text: str,
    sampling_rate: int,
) -> TranscribedAudioExample:
    container = av.open(BytesIO(audio), mode="r")
    with container:
        stream = container.streams.audio[0]
        if stream.sample_rate != sampling_rate:
            raise ValueError(f"Expected {sampling_rate} Hz audio, got {stream.sample_rate} Hz.")
        if stream.channels != 1:
            raise ValueError("Audio must be mono.")
        resampler = av.AudioResampler(
            format="fltp",
            layout="mono",
            rate=sampling_rate,
        )
        arrays = [converted.to_ndarray()[0] for frame in container.decode(stream) for converted in resampler.resample(frame)]
        arrays.extend(converted.to_ndarray()[0] for converted in resampler.resample(None))
    if not arrays:
        raise ValueError("Audio is empty.")
    return {
        "audio_array": np.concatenate(arrays).astype(np.float32, copy=False),
        "transcription": normalize_transcription(text),
    }


def _create_example_or_none(
    audio: bytes,
    text: str,
    sampling_rate: int,
    source: str,
) -> TranscribedAudioExample | None:
    try:
        return _create_example(audio, text, sampling_rate)
    except (av.error.FFmpegError, ValueError, IndexError) as error:
        LOGGER.warning("Skipping invalid audio at %s: %s", source, error)
        return None
