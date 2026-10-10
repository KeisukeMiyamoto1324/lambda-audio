from io import StringIO

import pytest
from rich.console import Console
from rich.progress import Progress

from src.shared import inference_display
from src.shared.samples import InferenceSample
from src.shared.transcription import TranscriptionResult


@pytest.mark.parametrize("show_table", [True, False])
def test_table_visibility_preserves_metrics_and_progress(show_table: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    output = StringIO()
    console = Console(file=output, width=100, color_system=None)
    progress = Progress(console=console, disable=True)
    monkeypatch.setattr(inference_display, "Console", lambda: console)
    monkeypatch.setattr(inference_display, "Progress", lambda *columns: progress)
    sample = InferenceSample(audio_name="example.wav", audio=b"audio", reference="あいう")
    result = TranscriptionResult(text="あいえ", audio_seconds=1.0, inference_seconds=0.1)

    inference_display.display_inference([(sample, result)], 1, "Test", show_table=show_table)

    rendered = output.getvalue()
    assert ("example.wav" in rendered) is show_table
    assert ("正解" in rendered) is show_table
    assert rendered.count("CER: 33.33%") == 1
    assert "Elapsed time:" in rendered
    assert progress.tasks[0].completed == 1
    assert progress.tasks[0].fields["cer"] == "33.33%"
