from src.shared.cer_metric import CerMetric
from src.shared.cer_text import normalize_cer_text, should_skip_text


def test_cer_metric_accumulates_corpus_errors() -> None:
    metric = CerMetric()

    metric.update("音声認識", "音声認知")
    metric.update("日本語", "日本")

    assert metric.errors == 2
    assert metric.reference_characters == 7
    assert metric.value == 2 / 7


def test_normalize_cer_text_removes_all_punctuation() -> None:
    assert normalize_cer_text("「今日は、晴れ。」") == "今日は晴れ"


def test_normalize_cer_text_removes_whitespace() -> None:
    assert normalize_cer_text("日 本\t語\nです") == "日本語です"


def test_should_skip_text_uses_non_japanese_ratio() -> None:
    assert should_skip_text("日本a1234567")
    assert not should_skip_text("日abc")
    assert not should_skip_text("「日本語です。」")
