from typing import Any

import numpy as np
from datasets import IterableDataset

from src.data.dataset import AudioBatchDataset


def test_audio_batch_dataset_shows_initial_buffer_progress(capsys: Any) -> None:
    source = IterableDataset.from_generator(
        lambda: iter(
            [
                {
                    "audio_array": np.ones(4, dtype=np.float32),
                    "transcription": "音声",
                }
            ]
        )
    )
    dataset = AudioBatchDataset(
        source,
        max_batch_padded_audio_samples=8,
        max_batch_attention_cost=1_000,
        encoder_frame_samples=1,
        bucket_buffer_size=1,
        seed=42,
        show_initial_buffer_progress=True,
    )

    list(dataset)

    assert "Preparing training buffer" in capsys.readouterr().out
