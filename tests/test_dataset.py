from src.shared.dataset import LengthBucketBatchSampler


def test_length_bucket_batch_sampler_groups_audio_and_text_lengths() -> None:
    audio_lengths = list(range(1, 9))
    text_lengths = [40, 30, 20, 10, 80, 70, 60, 50]
    sampler = LengthBucketBatchSampler(
        audio_lengths,
        text_lengths,
        batch_size=2,
        buffer_multiplier=4,
        seed=42,
        rank=0,
        world_size=1,
        shuffle=False,
    )

    batches = list(sampler)
    length_pairs = {
        tuple(text_lengths[index] for index in batch) for batch in batches
    }

    assert length_pairs == {(10, 20), (30, 40), (50, 60), (70, 80)}


def test_length_bucket_batch_sampler_partitions_distributed_samples() -> None:
    samplers = [
        LengthBucketBatchSampler(
            list(range(8)),
            list(range(8)),
            batch_size=2,
            buffer_multiplier=4,
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

    assert rank_indices[0] == {0, 2, 4, 6}
    assert rank_indices[1] == {1, 3, 5, 7}
