import numpy as np

from src.shared.dataset import ValidAudioExample


def test_valid_audio_example_checks_length_and_text() -> None:
    validator = ValidAudioExample(max_audio_samples=4)

    assert validator(
        {
            "audio": {"array": np.ones(4, dtype=np.float32)},
            "transcription": "音声",
        }
    )
    assert not validator(
        {
            "audio": {"array": np.ones(5, dtype=np.float32)},
            "transcription": "音声",
        }
    )
