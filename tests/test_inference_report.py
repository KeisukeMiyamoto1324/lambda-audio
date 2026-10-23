from io import StringIO

from rich.console import Console

from src.shared.inference_report import InferenceReport


def test_report_displays_differences_and_corpus_cer() -> None:
    report = InferenceReport("Inference")
    report.add_sample("data.parquet:1", "あいう。", "あいえ!")
    report.add_sample("data.parquet:2", "かきくけこ", "かきくけこ", lang="en")
    table = report.build_table()
    output = StringIO()
    Console(file=output, width=100, color_system=None).print(table)
    rendered = output.getvalue()

    assert "Lang" in rendered
    assert "en" in rendered
    assert "正解" in rendered
    assert "推論" in rendered
    assert "data.parquet:1" in rendered
    assert "あいう。" in rendered
    assert "あいえ!" in rendered
    assert "CER: 12.50%" in rendered
    assert rendered.index("CER:") > rendered.index("data.parquet:2")
    assert table.columns[2]._cells[0].spans
    assert table.columns[3]._cells[0].spans


def test_report_skips_ineligible_references_for_cer() -> None:
    report = InferenceReport("Inference")
    report.add_sample("audio.wav", "", "音声")
    report.add_sample("data.parquet:1", "!? ", "音声")
    report.add_sample("data.parquet:2", "English text", "音声")
    report.add_sample("data.parquet:3", "あいう", "あいう")

    table = report.build_table()

    assert table.row_count == 4
    assert table.caption == "CER: 0.00%"


def test_report_without_eligible_reference_displays_unavailable_cer() -> None:
    report = InferenceReport("Inference")
    report.add_sample("audio.wav", "", "推論結果")

    table = report.build_table()

    assert table.row_count == 1
    assert "CER: N/A" in str(table.caption)
