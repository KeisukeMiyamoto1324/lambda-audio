import random
from collections import defaultdict
from math import floor
from pathlib import Path
from typing import Any, cast

from datasets import Audio, Dataset, IterableDataset
from datasets.distributed import split_dataset_by_node

from src.data.lambda_audio_dataset import LambdaAudioDataset, load_cached_validation_dataset

CER_SAMPLE_COUNT = 500
CER_SELECTION_SEED = 42


def load_cer_dataset(
    token: str,
    cache_dir: Path,
    sampling_rate: int,
    global_rank: int,
    world_size: int,
    num_workers: int,
) -> LambdaAudioDataset:
    dataset = load_cached_validation_dataset(token, cache_dir)
    indices = select_stratified_indices(dataset, CER_SAMPLE_COUNT, CER_SELECTION_SEED)
    selected = dataset.select(indices)
    iterable = selected.to_iterable_dataset(num_shards=world_size * num_workers)
    iterable = iterable.select_columns(["audio", "text"])
    iterable = iterable.cast_column(
        "audio",
        Audio(sampling_rate=sampling_rate, decode=False),
    )
    iterable = split_dataset_by_node(iterable, global_rank, world_size)
    return LambdaAudioDataset(cast(IterableDataset, iterable), sampling_rate)


def select_stratified_indices(
    dataset: Dataset,
    sample_count: int,
    seed: int,
) -> list[int]:
    if len(dataset) < sample_count:
        raise ValueError(f"CER requires at least {sample_count} validation examples, got {len(dataset)}.")

    strata: dict[tuple[str, int], list[int]] = defaultdict(list)
    origins = cast(list[Any], dataset["origin"])
    durations = cast(list[float], dataset["duration"])
    for index, (origin, duration) in enumerate(zip(origins, durations, strict=True)):
        strata[(str(origin), _duration_bucket(duration))].append(index)

    quotas = {key: floor(sample_count * len(indices) / len(dataset)) for key, indices in strata.items()}
    remaining = sample_count - sum(quotas.values())
    ranked_keys = sorted(
        strata,
        key=lambda key: (
            -(sample_count * len(strata[key]) / len(dataset) - quotas[key]),
            key,
        ),
    )
    for key in ranked_keys[:remaining]:
        quotas[key] += 1

    generator = random.Random(seed)
    selected: list[int] = []
    for key in sorted(strata):
        indices = strata[key]
        generator.shuffle(indices)
        selected.extend(indices[: quotas[key]])
    generator.shuffle(selected)
    return selected


def _duration_bucket(duration: float) -> int:
    if duration < 5.0:
        return 0
    if duration < 10.0:
        return 1
    if duration < 20.0:
        return 2
    return 3
