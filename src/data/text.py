from unicodedata import normalize

BRACKET_PAIRS = {
    "(": ")",
    "[": "]",
    "{": "}",
    "<": ">",
    "「": "」",
    "『": "』",
    "【": "】",
    "〔": "〕",
    "〖": "〗",
    "〘": "〙",
    "〚": "〛",
    "〈": "〉",
    "《": "》",
    "“": "”",
    "‘": "’",
    "«": "»",
    "‹": "›",
    '"': '"',
    "'": "'",
}


def normalize_transcription(transcription: str) -> str:
    text = normalize("NFKC", transcription).strip()

    # ---------------------------------------------------------
    # Remove only matching brackets or quotes at both boundaries.
    # ---------------------------------------------------------
    while len(text) >= 2 and BRACKET_PAIRS.get(text[0]) == text[-1]:
        text = text[1:-1]

    return text
