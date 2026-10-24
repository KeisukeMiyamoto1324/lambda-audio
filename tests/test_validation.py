from collections.abc import Iterator
from io import BytesIO
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf
import torch
from datasets import Dataset
from datasets.distributed import split_dataset_by_node
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Wav2Vec2FeatureExtractor

from src.data.collator import AudioTextCollator
from src.data.dataset import AudioBatchDataset
from src.data.lambda_audio_dataset import LambdaAudioDataset
from src.data.types import TranscribedAudioExample
from src.data.validation import CachedValidationDataset
from src.shared.languages import OUTPUT_LANGUAGES
from src.training.config import TrainingConfig
from src.training.data_module import LambdaAudioDataModule


class CountingValidationSource:
    def __init__(self, dataset: LambdaAudioDataset, single_pass: bool = False) -> None:
        self.dataset = dataset
        self.single_pass = single_pass
        self.iterations = 0

    def __iter__(self) -> Iterator[TranscribedAudioExample]:
        self.iterations += 1
        if self.single_pass and self.iterations > 1:
            raise AssertionError("Validation source was read twice in the same worker.")
        yield from self.dataset

    def set_epoch(self, epoch: int) -> None:
        self.dataset.set_epoch(epoch)


@pytest.fixture
def validation_components() -> tuple[LambdaAudioDataset, AudioTextCollator]:
    rows = []
    for index in range(8):
        audio = BytesIO()
        sf.write(audio, np.linspace(-0.5, 0.5, 16 + index, dtype=np.float32), 16_000, format="FLAC")
        rows.append({"id": str(index), "audio": {"bytes": audio.getvalue()}, **{language: str(index) for language in OUTPUT_LANGUAGES}})
    source = LambdaAudioDataset(Dataset.from_list(rows).to_iterable_dataset(num_shards=4), 16_000, 42)
    vocabulary = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, **{str(index): index + 4 for index in range(8)}}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        additional_special_tokens=[f"<|target_{language}|>" for language in OUTPUT_LANGUAGES],
    )
    collator = AudioTextCollator(Wav2Vec2FeatureExtractor(sampling_rate=16_000, return_attention_mask=True), tokenizer, 16_000)
    return source, collator


def test_validation_reuses_preprocessing_without_changing_batches(validation_components: tuple[LambdaAudioDataset, AudioTextCollator]) -> None:
    source, collator = validation_components
    expected = [collator(examples) for examples in AudioBatchDataset(source, max_batch_padded_audio_samples=48, bucket_buffer_size=4, seed=42)]
    counted_source = CountingValidationSource(source, single_pass=True)
    counted_collator = Mock(wraps=collator)
    dataset = CachedValidationDataset(AudioBatchDataset(counted_source, max_batch_padded_audio_samples=48, bucket_buffer_size=4, seed=42), counted_collator)

    first = list(dataset)
    second = list(dataset)

    assert counted_source.iterations == 1
    assert counted_collator.call_count == len(first)
    assert len(first) == len(second) == len(expected)
    for original, prepared, cached in zip(expected, first, second, strict=True):
        assert prepared is cached
        for name, value in original.items():
            if isinstance(value, torch.Tensor):
                assert prepared[name].device.type == "cpu"
                torch.testing.assert_close(prepared[name], value)
            else:
                assert prepared[name] == value


def test_interrupted_validation_does_not_reuse_partial_cache(validation_components: tuple[LambdaAudioDataset, AudioTextCollator]) -> None:
    source, collator = validation_components
    counted_source = CountingValidationSource(source)
    dataset = CachedValidationDataset(AudioBatchDataset(counted_source, max_batch_padded_audio_samples=48, bucket_buffer_size=4, seed=42), collator)
    partial = iter(dataset)
    next(partial)
    partial.close()
    assert dataset.batches is None

    complete = list(dataset)

    assert counted_source.iterations == 2
    assert sum(batch["input_values"].shape[0] for batch in complete) == 8
    assert [token for batch in complete for token in batch["input_ids"][:, 2].tolist()] == [token for batch in dataset for token in batch["input_ids"][:, 2].tolist()]
    assert counted_source.iterations == 2


def test_validation_cache_preserves_rank_and_worker_partitioning(validation_components: tuple[LambdaAudioDataset, AudioTextCollator]) -> None:
    source, collator = validation_components
    all_tokens: list[int] = []
    for rank in range(2):
        rank_source = LambdaAudioDataset(split_dataset_by_node(source.dataset, rank, 2), 16_000, 42)
        dataset = CachedValidationDataset(
            AudioBatchDataset(CountingValidationSource(rank_source, single_pass=True), max_batch_padded_audio_samples=48, bucket_buffer_size=4, seed=42 + rank), collator
        )
        module = LambdaAudioDataModule(collator.feature_extractor, collator.tokenizer, TrainingConfig(num_workers=2, prefetch_factor=1), "unused")
        module.validation_dataset = dataset
        loader = module.val_dataloader()
        try:
            first = list(loader)
            second = list(loader)
            first_tokens = [token for batch in first for token in batch["input_ids"][:, 2].tolist()]
            second_tokens = [token for batch in second for token in batch["input_ids"][:, 2].tolist()]
            assert first_tokens == second_tokens
            assert len(first_tokens) == 4
            all_tokens.extend(first_tokens)
            for original, cached in zip(first, second, strict=True):
                for name, value in original.items():
                    if isinstance(value, torch.Tensor):
                        torch.testing.assert_close(cached[name], value)
                    else:
                        assert cached[name] == value
        finally:
            if loader._iterator is not None:
                loader._iterator._shutdown_workers()
    assert sorted(all_tokens) == list(range(4, 12))
