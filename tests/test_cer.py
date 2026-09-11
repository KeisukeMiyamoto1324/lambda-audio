from typing import Any

import numpy as np
import pytest
import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, Wav2Vec2FeatureExtractor

from src.cer.collator import CerCollator
from src.cer.dataset import select_stratified_indices
from src.cer.inference import generate_transcriptions
from src.cer.metric import CerCounts, compute_cer_counts


class DummyModel:
    def __init__(self) -> None:
        self.arguments: dict[str, Any] = {}

    def generate(self, **kwargs: Any) -> torch.LongTensor:
        self.arguments = kwargs
        return torch.tensor(
            [
                [1, 4, 2],
                [1, 4, 4],
            ]
        )


def create_tokenizer() -> PreTrainedTokenizerFast:
    backend = Tokenizer(
        WordLevel(
            {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, "音声": 4},
            unk_token="<unk>",
        )
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
    )


def test_cer_counts_corpus_edits_after_training_normalization() -> None:
    counts = compute_cer_counts(
        ["（音声）", "abc", ""],
        ["音生", "ab", "追加"],
    )

    assert counts == CerCounts(
        substitutions=1,
        deletions=1,
        insertions=2,
        reference_characters=5,
        samples=3,
    )
    assert counts.errors == 4


def test_cer_counts_rejects_mismatched_batches() -> None:
    with pytest.raises(ValueError, match="same length"):
        compute_cer_counts(["音声"], [])


def test_cer_collator_pads_audio_and_keeps_references() -> None:
    collator = CerCollator(
        Wav2Vec2FeatureExtractor(sampling_rate=16_000, return_attention_mask=True),
        16_000,
    )

    batch = collator(
        [
            {"audio_array": np.ones(4, dtype=np.float32), "transcription": "長い"},
            {"audio_array": np.ones(2, dtype=np.float32), "transcription": "短い"},
        ]
    )

    assert batch["input_values"].shape == (2, 4)
    torch.testing.assert_close(
        batch["audio_padding_mask"],
        torch.tensor([[False, False, False, False], [False, False, True, True]]),
    )
    assert batch["references"] == ["長い", "短い"]


def test_generate_transcriptions_runs_one_batch() -> None:
    model = DummyModel()
    tokenizer = create_tokenizer()
    batch = {
        "input_values": torch.ones(2, 4),
        "audio_padding_mask": torch.tensor([[False, False, False, False], [False, False, True, True]]),
        "references": ["音声", "音声音声"],
    }

    hypotheses = generate_transcriptions(  # type: ignore[arg-type]
        model,
        tokenizer,
        batch,
        16,
    )

    assert hypotheses == ["音声", "音声 音声"]
    assert model.arguments["input_values"].shape[0] == 2
    assert model.arguments["max_new_tokens"] == 16
    assert model.arguments["do_sample"] is False
    assert model.arguments["use_cache"] is True


def test_stratified_selection_is_fixed_and_proportional() -> None:
    dataset = Dataset.from_dict(
        {
            "origin": ["a"] * 80 + ["b"] * 20,
            "duration": [3.0] * 50 + [8.0] * 30 + [12.0] * 20,
        }
    )

    first = select_stratified_indices(dataset, 50, 42)
    second = select_stratified_indices(dataset, 50, 42)
    selected = dataset.select(first)

    assert first == second
    assert len(first) == len(set(first)) == 50
    assert selected["origin"].count("a") == 40
    assert selected["origin"].count("b") == 10


def test_stratified_selection_requires_requested_sample_count() -> None:
    dataset = Dataset.from_dict({"origin": ["a"], "duration": [1.0]})

    with pytest.raises(ValueError, match="at least 2"):
        select_stratified_indices(dataset, 2, 42)
