from unicodedata import category

from src.data.text import normalize_transcription


def normalize_cer_text(text: str) -> str:
    normalized = normalize_transcription(text)
    return "".join(character for character in normalized if not character.isspace() and not category(character).startswith("P"))


def should_skip_text(text: str) -> bool:
    return not normalize_cer_text(text)
