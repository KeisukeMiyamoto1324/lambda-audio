from collections.abc import Iterator

from torch.utils.data import IterableDataset

from src.data.collator import AudioTextCollator
from src.data.dataset import AudioBatchDataset
from src.data.types import AudioTextBatch


class CachedValidationDataset(IterableDataset[AudioTextBatch]):
    def __init__(self, dataset: AudioBatchDataset, collator: AudioTextCollator) -> None:
        self.dataset = dataset
        self.collator = collator
        self.batches: list[AudioTextBatch] | None = None

    def __iter__(self) -> Iterator[AudioTextBatch]:
        # ---------------------------------------------------------
        # Reuse complete CPU batches inside each persistent worker.
        # ---------------------------------------------------------
        if self.batches is not None:
            yield from self.batches
            return

        batches: list[AudioTextBatch] = []
        for examples in self.dataset:
            batch = self.collator(examples)
            batches.append(batch)
            yield batch

        # ---------------------------------------------------------
        # Publish only a complete pass so interrupted reads cannot lose data.
        # ---------------------------------------------------------
        self.batches = batches
