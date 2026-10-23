from rich.table import Table

from src.shared.cer_metric import CerMetric
from src.shared.cer_text import normalize_cer_text, should_skip_text
from src.shared.languages import OutputLanguage
from src.shared.text_diff import create_text_diff


class InferenceReport:
    def __init__(self, title: str) -> None:
        self._table = Table(title=title, caption_justify="left")
        self._table.add_column("Audio", no_wrap=True)
        self._table.add_column("Lang", no_wrap=True)
        self._table.add_column("正解", overflow="fold", ratio=1)
        self._table.add_column("推論", overflow="fold", ratio=1)
        self._metric = CerMetric()
        self.skipped_samples = 0

    def add_sample(self, audio_name: str, reference: str, prediction: str, lang: OutputLanguage = "ja") -> None:
        reference_text, prediction_text = create_text_diff(reference, prediction)
        self._table.add_row(audio_name, lang, reference_text, prediction_text)
        if should_skip_text(reference):
            self.skipped_samples += 1
            return
        self._metric.update(normalize_cer_text(reference), normalize_cer_text(prediction))

    @property
    def cer_text(self) -> str:
        if self._metric.reference_characters == 0:
            return "N/A（評価対象の正解テキストなし）"
        return f"{self._metric.value:.2%}"

    def build_table(self) -> Table:
        self._table.caption = f"CER: {self.cer_text}"
        return self._table
