from unicodedata import category, mirrored, normalize


def normalize_transcription(transcription: str) -> str:
    text = normalize("NFKC", transcription).strip()

    # ---------------------------------------------------------
    # Remove enclosing punctuation at the text boundaries.
    # ---------------------------------------------------------
    while text and (category(text[0]) in {"Ps", "Pi"} or mirrored(text[0])):
        text = text[1:]
    while text and (category(text[-1]) in {"Pe", "Pf"} or mirrored(text[-1])):
        text = text[:-1]
    if len(text) >= 2 and text[0] == text[-1] and category(text[0])[0] in "PS":
        text = text[1:-1]

    return text
