import torch

from src.shared.performance import audio_batch_sizes


def test_audio_batch_sizes_counts_valid_and_padded_samples() -> None:
    padding_mask = torch.tensor(
        [
            [False, False, False, False],
            [False, False, True, True],
        ]
    )

    valid_samples, padded_samples = audio_batch_sizes(padding_mask)

    assert valid_samples == 6
    assert padded_samples == 8
