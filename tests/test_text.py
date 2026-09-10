from src.data.text import normalize_transcription


def test_normalize_transcription_normalizes_and_removes_enclosing_punctuation() -> None:
    assert normalize_transcription("[<ＡＢＣ>]") == "ABC"


def test_normalize_transcription_removes_matching_quotes() -> None:
    assert normalize_transcription('"音声"') == "音声"


def test_normalize_transcription_strips_whitespace() -> None:
    assert normalize_transcription(" 音声 ") == "音声"
