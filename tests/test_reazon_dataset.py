import shutil
import tarfile
import threading
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest
import soundfile as sf
from pytest import MonkeyPatch

from src.shared.reazon_dataset import ReazonSpeechShardDataset


def create_shard(path: Path, keys: list[str], sampling_rate: int = 16_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, mode="w") as archive:
        for key in keys:
            audio_file = BytesIO()
            sf.write(
                audio_file,
                np.ones(160, dtype=np.float32),
                sampling_rate,
                format="FLAC",
            )
            audio = audio_file.getvalue()
            member = tarfile.TarInfo(f"{key}.flac")
            member.size = len(audio)
            archive.addfile(member, BytesIO(audio))


def add_invalid_audio(path: Path, key: str) -> None:
    audio = b"invalid flac"
    member = tarfile.TarInfo(f"{key}.flac")
    member.size = len(audio)
    with tarfile.open(path, mode="a") as archive:
        archive.addfile(member, BytesIO(audio))


def install_download_stub(
    monkeypatch: MonkeyPatch,
    source_dir: Path,
    calls: list[str],
    second_download_started: threading.Event | None = None,
) -> None:
    def download_stub(
        repo_id: str,
        filename: str,
        *,
        local_dir: Path,
        **kwargs: Any,
    ) -> str:
        calls.append(filename)
        if second_download_started is not None and len(calls) == 2:
            second_download_started.set()
        destination = Path(local_dir) / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_dir / filename, destination)
        return str(destination)

    monkeypatch.setattr(
        "src.shared.reazon_dataset.hf_hub_download",
        download_stub,
    )


def create_dataset(
    tmp_path: Path,
    shard_files: list[str],
    *,
    split: Literal["train", "validation"],
    global_rank: int = 0,
    world_size: int = 1,
) -> ReazonSpeechShardDataset:
    return ReazonSpeechShardDataset(
        token="token",
        revision="revision",
        shard_files=shard_files,
        cache_dir=tmp_path / "cache",
        split=split,
        validation_samples=1,
        shuffle_buffer_size=1,
        sampling_rate=16_000,
        seed=42,
        global_rank=global_rank,
        world_size=world_size,
    )


def test_reazon_dataset_separates_training_and_validation(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    remote = tmp_path / "remote"
    shard_files = ["data/000.tar", "data/001.tar"]
    create_shard(remote / shard_files[0], ["000/validation", "000/train"])
    create_shard(remote / shard_files[1], ["001/train"])
    install_download_stub(monkeypatch, remote, [])

    training = create_dataset(tmp_path, shard_files, split="train")
    validation = create_dataset(tmp_path, shard_files, split="validation")

    assert {example["__key__"] for example in training} == {
        "000/train",
        "001/train",
    }
    assert [example["__key__"] for example in validation] == [
        "000/validation"
    ]
    assert len(list((tmp_path / "cache" / "validation").rglob("*.tar"))) == 1


def test_reazon_dataset_prefetches_next_shard_and_bounds_cache(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    remote = tmp_path / "remote"
    shard_files = ["data/000.tar", "data/001.tar", "data/002.tar"]
    for index, shard_file in enumerate(shard_files):
        create_shard(
            remote / shard_file,
            [f"{index:03d}/first", f"{index:03d}/second"],
        )
    calls: list[str] = []
    second_download_started = threading.Event()
    install_download_stub(
        monkeypatch,
        remote,
        calls,
        second_download_started,
    )
    dataset = create_dataset(tmp_path, shard_files, split="train")
    examples = iter(dataset)

    next(examples)

    assert second_download_started.wait(timeout=1.0)
    assert len(set(calls[:2])) == 2
    assert len(list((tmp_path / "cache" / "train").rglob("*.tar"))) <= 2

    list(examples)
    assert not list((tmp_path / "cache" / "train").rglob("*.tar"))


def test_reazon_dataset_partitions_shards_between_ranks(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    remote = tmp_path / "remote"
    shard_files = [f"data/{index:03d}.tar" for index in range(4)]
    for index, shard_file in enumerate(shard_files):
        create_shard(remote / shard_file, [f"{index:03d}/audio"])
    install_download_stub(monkeypatch, remote, [])

    rank_zero = create_dataset(
        tmp_path / "rank-zero",
        shard_files,
        split="train",
        global_rank=0,
        world_size=2,
    )
    rank_one = create_dataset(
        tmp_path / "rank-one",
        shard_files,
        split="train",
        global_rank=1,
        world_size=2,
    )
    rank_zero_keys = {example["__key__"] for example in rank_zero}
    rank_one_keys = {example["__key__"] for example in rank_one}

    assert rank_zero_keys.isdisjoint(rank_one_keys)
    assert rank_zero_keys | rank_one_keys == {
        "001/audio",
        "002/audio",
        "003/audio",
    }


def test_reazon_dataset_skips_invalid_audio(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    remote = tmp_path / "remote"
    shard_file = "data/000.tar"
    create_shard(
        remote / shard_file,
        ["000/validation", "000/training"],
    )
    add_invalid_audio(remote / shard_file, "000/invalid")
    install_download_stub(monkeypatch, remote, [])
    dataset = create_dataset(tmp_path, [shard_file], split="train")

    with pytest.warns(RuntimeWarning, match="Skipping invalid audio"):
        keys = [example["__key__"] for example in dataset]

    assert keys == ["000/training"]
