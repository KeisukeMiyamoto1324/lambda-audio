import numpy as np

from src.data.prepare import PrepareAudioExample, ValidPreparedExample


def test_prepare_audio_example_adds_lengths() -> None:
    prepare = PrepareAudioExample(sampling_rate=16_000)

    example = prepare(
        {
            "audio": {"array": np.ones(4, dtype=np.float32)},
            "transcription": " 音声 ",
        }
    )

    assert example["audio_length"] == 4
    assert example["text_length"] == 2
    assert example["transcription"] == "音声"
    assert example["audio"]["sampling_rate"] == 16_000


def test_valid_prepared_example_checks_length_and_text() -> None:
    validator = ValidPreparedExample(max_audio_samples=4)

    assert validator({"audio_length": 4, "text_length": 2})
    assert not validator({"audio_length": 5, "text_length": 2})
    assert not validator({"audio_length": 4, "text_length": 0})
