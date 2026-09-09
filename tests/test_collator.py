import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, SeamlessM4TFeatureExtractor

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
    feature_extractor = SeamlessM4TFeatureExtractor(
        sampling_rate=16_000,
        return_attention_mask=True,
    )
    collator = AudioTextCollator(
        feature_extractor,
        tokenizer,
        tokenizer,
        16_000,
        8,
    )

    batch = collator(
        [
            {
                "audio_array": np.ones(16_000, dtype=np.float32),
                "transcription": "test test",
            },
            {
                "audio_array": np.ones(8_000, dtype=np.float32),
                "transcription": "test",
            },
        ]
    )

    assert batch["input_features"].shape == (2, 49, 160)
    assert batch["audio_valid_samples"] == 24_000
    assert batch["audio_padded_samples"] == 32_000
    assert not batch["audio_padding_mask"][0].any()
    assert batch["audio_padding_mask"][1, 24:].all()
    assert batch["labels"][1, -1].item() == -100
    torch.testing.assert_close(batch["ctc_label_lengths"], torch.tensor([2, 1]))
    assert batch["ctc_labels"][0, :2].tolist() == [4, 4]
