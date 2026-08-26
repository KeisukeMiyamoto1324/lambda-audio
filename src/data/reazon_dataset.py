import multiprocessing
import random
import tarfile
import warnings
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Literal

import av
import numpy as np
from numpy.typing import NDArray
from torch.utils.data import IterableDataset, get_worker_info

from src.data.reazon_repository import (
    dataset_revision,
    download_reazon_speech_file,
    list_reazon_speech_shards,
)
from src.data.types import AudioExample


DatasetSplit = Literal["train", "validation"]


class InvalidAudioError(RuntimeError):
    pass


class ReazonSpeechShardDataset(IterableDataset[AudioExample]):
    def __init__(
        self,
        *,
        token: str,
        revision: str,
        shard_files: list[str],
        cache_dir: Path,
        split: DatasetSplit,
        validation_samples: int,
        shuffle_buffer_size: int,
        sampling_rate: int,
        seed: int,
        global_rank: int,
        world_size: int,
    ) -> None:
        self.token = token
        self.revision = revision
        self.shard_files = shard_files
        self.cache_dir = cache_dir
        self.split = split
        self.validation_samples = validation_samples
        self.shuffle_buffer_size = shuffle_buffer_size
        self.sampling_rate = sampling_rate
        self.seed = seed
        self.global_rank = global_rank
        self.world_size = world_size
        self.epoch = multiprocessing.Value("q", 0)

        if not shard_files:
            raise ValueError("At least one audio shard is required.")
        if validation_samples < 1:
            raise ValueError("The validation sample count must be positive.")
        if shuffle_buffer_size < 1:
            raise ValueError("The shuffle buffer size must be positive.")
        if sampling_rate < 1:
            raise ValueError("The sampling rate must be positive.")
        if global_rank < 0 or global_rank >= world_size:
            raise ValueError("The distributed rank is invalid.")

    def __iter__(self) -> Iterator[AudioExample]:
        if self.split == "validation":
            yield from self._iterate_validation()
            return

        yield from self._iterate_training()

    def _iterate_training(self) -> Iterator[AudioExample]:
        worker_id, worker_count = self._worker_identity()
        consumer_index = self.global_rank * worker_count + worker_id
        consumer_count = self.world_size * worker_count
        generator = random.Random(self.seed + self.epoch.value)
        shards = list(self.shard_files)
        generator.shuffle(shards)
        assigned_shards = shards[consumer_index::consumer_count]
        worker_cache = (
            self.cache_dir
            / "train"
            / f"rank-{self.global_rank}"
            / f"worker-{worker_id}"
        )
        examples = self._iterate_training_shards(assigned_shards, worker_cache)
        yield from self._shuffle_examples(examples, generator)

    def _iterate_training_shards(
        self,
        shard_files: list[str],
        worker_cache: Path,
    ) -> Iterator[AudioExample]:
        worker_cache.mkdir(parents=True, exist_ok=True)
        self._clear_training_cache(worker_cache)
        if not shard_files:
            return

        executor = ThreadPoolExecutor(max_workers=1)
        future: Future[Path] = executor.submit(
            self._download_shard,
            shard_files[0],
            worker_cache,
        )
        try:
            for index, shard_file in enumerate(shard_files):
                shard_path = future.result()
                if index + 1 < len(shard_files):
                    future = executor.submit(
                        self._download_shard,
                        shard_files[index + 1],
                        worker_cache,
                    )
                yield from self._read_training_shard(shard_file, shard_path)
                shard_path.unlink()
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
            self._clear_training_cache(worker_cache)

    def _iterate_validation(self) -> Iterator[AudioExample]:
        worker_id, worker_count = self._worker_identity()
        consumer_index = self.global_rank * worker_count + worker_id
        consumer_count = self.world_size * worker_count
        shard_path = self.cache_dir / "validation" / self.shard_files[0]
        if not shard_path.is_file():
            raise FileNotFoundError(
                f"Prepared validation shard not found: {shard_path}"
            )

        for position, example in enumerate(self._read_shard(shard_path)):
            if position >= self.validation_samples:
                return
            if position % consumer_count == consumer_index:
                yield example

    def _read_training_shard(
        self,
        shard_file: str,
        shard_path: Path,
    ) -> Iterator[AudioExample]:
        for position, example in enumerate(self._read_shard(shard_path)):
            if (
                shard_file == self.shard_files[0]
                and position < self.validation_samples
            ):
                continue
            yield example

    def _read_shard(self, shard_path: Path) -> Iterator[AudioExample]:
        with tarfile.open(shard_path, mode="r") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".flac"):
                    continue
                audio_file = archive.extractfile(member)
                if audio_file is None:
                    raise RuntimeError(f"Unable to read {member.name}.")
                try:
                    audio_array = self._decode_audio(
                        audio_file.read(),
                        member.name,
                    )
                except InvalidAudioError:
                    warnings.warn(
                        f"Skipping invalid audio: {member.name}.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    continue
                yield {
                    "audio_key": member.name.removesuffix(".flac"),
                    "audio_array": audio_array,
                }

    def _decode_audio(
        self,
        audio: bytes,
        member_name: str,
    ) -> NDArray[np.float32]:
        try:
            container = av.open(BytesIO(audio), mode="r", format="flac")
            with container:
                stream = container.streams.audio[0]
                if stream.sample_rate != self.sampling_rate:
                    raise InvalidAudioError(
                        f"Unexpected sampling rate in {member_name}: "
                        f"{stream.sample_rate}."
                    )
                if stream.channels != 1:
                    raise InvalidAudioError(
                        f"Audio must be mono: {member_name}."
                    )
                resampler = av.AudioResampler(
                    format="fltp",
                    layout="mono",
                    rate=self.sampling_rate,
                )
                arrays = [
                    converted.to_ndarray()[0]
                    for frame in container.decode(stream)
                    for converted in resampler.resample(frame)
                ]
                arrays.extend(
                    converted.to_ndarray()[0]
                    for converted in resampler.resample(None)
                )
        except (av.FFmpegError, IndexError) as error:
            raise InvalidAudioError(
                f"Unable to decode FLAC audio: {member_name}."
            ) from error

        if not arrays:
            raise InvalidAudioError(f"Audio is empty: {member_name}.")
        return np.concatenate(arrays).astype(np.float32, copy=False)

    def _shuffle_examples(
        self,
        examples: Iterator[AudioExample],
        generator: random.Random,
    ) -> Iterator[AudioExample]:
        buffer: list[AudioExample] = []
        for example in examples:
            if len(buffer) < self.shuffle_buffer_size:
                buffer.append(example)
                continue
            index = generator.randrange(len(buffer))
            yield buffer[index]
            buffer[index] = example

        generator.shuffle(buffer)
        yield from buffer

    def _download_shard(self, shard_file: str, cache_dir: Path) -> Path:
        return download_reazon_speech_file(
            self.token,
            self.revision,
            shard_file,
            cache_dir,
        )

    @staticmethod
    def _clear_training_cache(cache_dir: Path) -> None:
        for pattern in ("*.tar", "*.incomplete"):
            for path in cache_dir.rglob(pattern):
                path.unlink()

    @staticmethod
    def _worker_identity() -> tuple[int, int]:
        worker = get_worker_info()
        if worker is None:
            return 0, 1
        return worker.id, worker.num_workers

    def set_epoch(self, epoch: int) -> None:
        with self.epoch.get_lock():
            self.epoch.value = epoch


def load_reazon_speech_datasets(
    token: str,
    validation_samples: int,
    shuffle_buffer_size: int,
    sampling_rate: int,
    cache_dir: Path,
    seed: int,
    global_rank: int,
    world_size: int,
) -> tuple[ReazonSpeechShardDataset, ReazonSpeechShardDataset]:
    revision = dataset_revision(token)
    shard_files = list_reazon_speech_shards(token, revision)
    common_arguments = {
        "token": token,
        "revision": revision,
        "shard_files": shard_files,
        "cache_dir": cache_dir,
        "validation_samples": validation_samples,
        "shuffle_buffer_size": shuffle_buffer_size,
        "sampling_rate": sampling_rate,
        "seed": seed,
        "global_rank": global_rank,
        "world_size": world_size,
    }
    return (
        ReazonSpeechShardDataset(split="train", **common_arguments),
        ReazonSpeechShardDataset(split="validation", **common_arguments),
    )
