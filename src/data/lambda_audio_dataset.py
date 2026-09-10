from collections.abc import Iterator
from io import BytesIO
from typing import Any, Literal, cast

import av
import numpy as np
from datasets import Audio, IterableDataset, load_dataset
from datasets.distributed import split_dataset_by_node

from src.data.text import normalize_transcription
from src.data.types import TranscribedAudioExample

DATASET_ID = "KeisukeMiyamoto/lambda-audio-courpus-1"
DatasetSplit = Literal["train", "validation"]


class LambdaAudioDataset:
    def __init__(self, dataset: IterableDataset, sampling_rate: int) -> None:
        self.dataset = dataset
        self.sampling_rate = sampling_rate

    def __iter__(self) -> Iterator[TranscribedAudioExample]:
        for row in self.dataset:
            audio = cast(Any, row["audio"])
            yield {
                "audio_array": self._decode_audio(cast(bytes, audio["bytes"])),
                "transcription": normalize_transcription(cast(str, row["text"])),
            }

    def _decode_audio(self, audio: bytes) -> np.ndarray[Any, np.dtype[np.float32]]:
        container = av.open(BytesIO(audio), mode="r")
        with container:
            stream = container.streams.audio[0]
            if stream.sample_rate != self.sampling_rate:
                raise ValueError(
                    f"Expected {self.sampling_rate} Hz audio, got {stream.sample_rate} Hz."
                )
            if stream.channels != 1:
                raise ValueError("Audio must be mono.")
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
                converted.to_ndarray()[0] for converted in resampler.resample(None)
            )
        if not arrays:
            raise ValueError("Audio is empty.")
        return np.concatenate(arrays).astype(np.float32, copy=False)

    def set_epoch(self, epoch: int) -> None:
        self.dataset.set_epoch(epoch)


def load_lambda_audio_dataset(
    token: str,
    split: DatasetSplit,
    shuffle_buffer_size: int,
    sampling_rate: int,
    seed: int,
    global_rank: int,
    world_size: int,
) -> LambdaAudioDataset:
    dataset = cast(
        IterableDataset,
        load_dataset(
            DATASET_ID,
            split=split,
            streaming=True,
            token=token,
        ),
    )
    dataset = dataset.select_columns(["audio", "text"])
    dataset = dataset.cast_column(
        "audio",
        Audio(sampling_rate=sampling_rate, decode=False),
    )
    dataset = split_dataset_by_node(dataset, global_rank, world_size)
    if split == "train":
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, seed=seed)
    return LambdaAudioDataset(dataset, sampling_rate)
