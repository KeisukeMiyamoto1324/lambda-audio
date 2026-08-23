from datasets import Audio, Features, Value


def create_prepared_features(sampling_rate: int) -> Features:
    return Features(
        {
            "audio": Audio(sampling_rate=sampling_rate),
            "transcription": Value("string"),
            "audio_length": Value("int32"),
            "text_length": Value("int32"),
        }
    )
