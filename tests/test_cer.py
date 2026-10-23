import pytest

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


@pytest.mark.parametrize("text", ["日本a1234567", "English text", "مرحبا", "안녕하세요", "1234567", "「日本語です。」"])
def test_should_skip_text_accepts_all_languages(text: str) -> None:
    assert not should_skip_text(text)


@pytest.mark.parametrize("text", ["", " \t\n", "!? 。"])
def test_should_skip_text_rejects_empty_normalized_text(text: str) -> None:
    assert should_skip_text(text)
