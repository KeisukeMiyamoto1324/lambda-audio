from typing import Protocol

from src.shared.transcription import TranscriptionResult


class BenchmarkModel(Protocol):
    def __call__(self, audio: list[bytes]) -> list[TranscriptionResult]: ...
