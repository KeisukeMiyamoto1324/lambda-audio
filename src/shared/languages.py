from typing import Literal, get_args

from transformers import PreTrainedTokenizerBase

OutputLanguage = Literal["ja", "en", "ar", "zh", "fr", "de", "ko", "pt", "es"]
OUTPUT_LANGUAGES: tuple[OutputLanguage, ...] = get_args(OutputLanguage)


def language_token_id(tokenizer: PreTrainedTokenizerBase, language: OutputLanguage) -> int:
    # ---------------------------------------------------------
    # Require a registered language token from the pretrained tokenizer.
    # ---------------------------------------------------------
    if language not in OUTPUT_LANGUAGES:
        raise ValueError(f"Unsupported output language: {language}")
    token = f"<|target_{language}|>"
    if token not in tokenizer.all_special_tokens:
        raise ValueError(f"Missing language token: {token}")
    return tokenizer.convert_tokens_to_ids(token)
