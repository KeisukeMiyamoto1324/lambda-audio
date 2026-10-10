from rich.table import Table

from src.cer.metric import CerMetric
from src.cer.text import normalize_cer_text, should_skip_text
from src.shared.text_diff import create_text_diff


class InferenceReport:
    def __init__(self, title: str) -> None:
        self._table = Table(title=title, caption_justify="left")
        self._table.add_column("Audio", no_wrap=True)
        self._table.add_column("正解", overflow="fold", ratio=1)
        self._table.add_column("推論", overflow="fold", ratio=1)
        self._metric = CerMetric()

    def add_sample(self, audio_name: str, reference: str, prediction: str) -> None:
        reference_text, prediction_text = create_text_diff(reference, prediction)
        self._table.add_row(audio_name, reference_text, prediction_text)
        if should_skip_text(reference):
            return
        self._metric.update(normalize_cer_text(reference), normalize_cer_text(prediction))

    def build_table(self) -> Table:
        if self._metric.reference_characters == 0:
            self._table.caption = "CER: N/A（評価対象の正解テキストなし）"
        else:
            self._table.caption = f"CER: {self._metric.value:.2%}"
        return self._table
