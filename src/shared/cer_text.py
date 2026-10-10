from unicodedata import category, name

from src.data.text import normalize_transcription

JAPANESE_MARKS = "々〆〇"


def normalize_cer_text(text: str) -> str:
    normalized = normalize_transcription(text)
    return "".join(character for character in normalized if not character.isspace() and not category(character).startswith("P"))


def should_skip_text(text: str) -> bool:
    characters = normalize_cer_text(text)
    if not characters:
        return True

    japanese_characters = sum(_is_japanese_character(character) for character in characters)
    non_japanese_characters = len(characters) - japanese_characters
    return non_japanese_characters * 4 > len(characters) * 3


def _is_japanese_character(character: str) -> bool:
    unicode_name = name(character, "")
    return (
        "HIRAGANA" in unicode_name
        or "KATAKANA" in unicode_name
        or "CJK UNIFIED IDEOGRAPH" in unicode_name
        or "CJK COMPATIBILITY IDEOGRAPH" in unicode_name
        or character in JAPANESE_MARKS
    )
