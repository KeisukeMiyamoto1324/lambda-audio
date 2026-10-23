import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Wav2Vec2FeatureExtractor

from src.data.collator import AudioTextCollator
from src.shared.languages import language_token_id


def test_collator_pads_audio_and_masks_text_labels() -> None:
    backend = Tokenizer(
        WordLevel(
            {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, "test": 4},
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        additional_special_tokens=["<|target_ja|>", "<|target_en|>"],
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        sampling_rate=16_000,
        return_attention_mask=True,
    )
    collator = AudioTextCollator(feature_extractor, tokenizer, 16_000)

    batch = collator(
        [
            {
                "audio_array": np.ones(4, dtype=np.float32),
                "output_language": "ja",
                "transcription": "test test",
            },
            {
                "audio_array": np.ones(2, dtype=np.float32),
                "output_language": "en",
                "transcription": "test",
            },
        ]
    )

    assert batch["input_values"].shape == (2, 4)
    torch.testing.assert_close(
        batch["audio_padding_mask"],
        torch.tensor([[False, False, False, False], [False, False, True, True]]),
    )
    assert batch["labels"][:, :2].eq(-100).all()
    assert batch["input_ids"][:, :2].tolist() == [[1, tokenizer.convert_tokens_to_ids(f"<|target_{language}|>")] for language in ("ja", "en")]
    assert batch["labels"][1, -1].item() == -100
    assert batch["target_token_count"] == 5
    assert batch["audio_valid_samples"] == 6
    assert batch["audio_padded_samples"] == 8


def test_collator_keeps_full_transcription() -> None:
    backend = Tokenizer(
        WordLevel(
            {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, "test": 4},
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        additional_special_tokens=["<|target_ja|>", "<|target_en|>"],
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        sampling_rate=16_000,
        return_attention_mask=True,
    )
    collator = AudioTextCollator(feature_extractor, tokenizer, 16_000)
    transcription = " ".join(["test"] * 300)

    batch = collator(
        [
            {
                "audio_array": np.ones(4, dtype=np.float32),
                "output_language": "ja",
                "transcription": transcription,
            }
        ]
    )

    assert batch["input_ids"].shape == (1, 303)
    assert batch["input_ids"][0, -1].item() == tokenizer.eos_token_id


@pytest.mark.parametrize("language", ["ja", "invalid"])
def test_language_token_rejects_missing_or_unsupported_language(language: str) -> None:
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"<unk>": 0}, unk_token="<unk>")), unk_token="<unk>")
    with pytest.raises(ValueError, match="Missing language token|Unsupported output language"):
        language_token_id(tokenizer, language)  # type: ignore[arg-type]
