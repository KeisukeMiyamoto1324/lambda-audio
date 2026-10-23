from io import StringIO

from rich.console import Console

from src.shared.inference_report import InferenceReport


def test_report_displays_differences_and_corpus_cer() -> None:
    report = InferenceReport("Inference")
    report.add_sample("ja", "あいう。", "あいえ!")
    report.add_sample("en", "かきくけこ", "かきくけこ")
    table = report.build_table()
    output = StringIO()
    Console(file=output, width=100, color_system=None).print(table)
    rendered = output.getvalue()

    assert "Lang" in rendered
    assert "en" in rendered
    assert "正解" in rendered
    assert "推論" in rendered
    assert "Audio" not in rendered
    assert rendered.index("Lang") < rendered.index("正解") < rendered.index("推論")
    assert "あいう。" in rendered
    assert "あいえ!" in rendered
    assert "CER: 12.50%" in rendered
    assert rendered.index("CER:") > rendered.index("かきくけこ")
    assert table.columns[1]._cells[0].spans
    assert table.columns[2]._cells[0].spans


def test_report_skips_ineligible_references_for_cer() -> None:
    report = InferenceReport("Inference")
    report.add_sample("ja", "", "音声")
    report.add_sample("ja", "!? ", "音声")
    report.add_sample("en", "English text", "English text")
    report.add_sample("ja", "あいう", "あいう")

    table = report.build_table()

    assert table.row_count == 4
    assert table.caption == "CER: 0.00%"
    assert report.skipped_samples == 2


def test_report_includes_multilingual_references_in_corpus_cer() -> None:
    report = InferenceReport("Inference")
    report.add_sample("en", "hello", "hell")
    report.add_sample("ar", "مرحبا", "مرحبا")
    report.add_sample("ko", "안녕하세요", "안녕하세요")
    report.add_sample("ja", "あいうえお", "あいうえお")

    assert report.cer_text == "5.00%"
    assert report.skipped_samples == 0


def test_report_without_eligible_reference_displays_unavailable_cer() -> None:
    report = InferenceReport("Inference")
    report.add_sample("ja", "", "推論結果")

    table = report.build_table()

    assert table.row_count == 1
    assert "CER: N/A" in str(table.caption)
