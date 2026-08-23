from src.shared.dataset import LengthBucketBatchSampler


def test_length_bucket_batch_sampler_limits_total_audio_samples() -> None:
    audio_lengths = [8, 8, 4, 4, 4]
    text_lengths = [8, 8, 4, 4, 4]
    sampler = LengthBucketBatchSampler(
        audio_lengths,
        text_lengths,
        max_batch_audio_samples=12,
        max_batch_examples=4,
        bucket_window_size=5,
        seed=42,
        rank=0,
        world_size=1,
        shuffle=False,
    )

    batches = list(sampler)

    assert [len(batch) for batch in batches] == [3, 1, 1]
    assert all(
        sum(audio_lengths[index] for index in batch) <= 12 for batch in batches
    )
    assert len(sampler) == 3


def test_length_bucket_batch_sampler_partitions_distributed_samples() -> None:
    samplers = [
        LengthBucketBatchSampler(
            [4] * 8,
            list(range(8)),
            max_batch_audio_samples=8,
            max_batch_examples=2,
            bucket_window_size=8,
            seed=42,
            rank=rank,
            world_size=2,
            shuffle=False,
        )
        for rank in range(2)
    ]

    rank_indices = [
        {index for batch in sampler for index in batch} for sampler in samplers
    ]

    assert rank_indices[0] == {0, 1, 4, 5}
    assert rank_indices[1] == {2, 3, 6, 7}


def test_length_bucket_batch_sampler_limits_example_count() -> None:
    sampler = LengthBucketBatchSampler(
        [1] * 10,
        list(range(10)),
        max_batch_audio_samples=100,
        max_batch_examples=3,
        bucket_window_size=10,
        seed=42,
        rank=0,
        world_size=1,
        shuffle=False,
    )

    assert [len(batch) for batch in sampler] == [3, 3, 3, 1]
