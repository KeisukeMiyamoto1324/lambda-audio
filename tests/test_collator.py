import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Wav2Vec2FeatureExtractor

from src.data.collator import AudioTextCollator


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
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        sampling_rate=16_000,
        return_attention_mask=True,
    )
    collator = AudioTextCollator(feature_extractor, tokenizer, 16_000, 8)

    batch = collator(
        [
            {
                "audio_array": np.ones(4, dtype=np.float32),
                "transcription": "test test",
            },
            {
                "audio_array": np.ones(2, dtype=np.float32),
                "transcription": "test",
            },
        ]
    )

    assert batch["input_values"].shape == (2, 4)
    torch.testing.assert_close(
        batch["audio_padding_mask"],
        torch.tensor([[False, False, False, False], [False, False, True, True]]),
    )
    assert batch["labels"][1, -1].item() == -100
