from typing import Any

import lightning as L
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase, SequenceFeatureExtractor

from src.data.collator import AudioTextCollator
from src.data.corpus import (
    load_lambda_audio_datasets,
    prepare_validation_data,
)
from src.data.dataset import (
    AudioBatchDataset,
    EpochTranscribedAudioDataset,
)
from src.training.config import BaseTrainingConfig


class LambdaAudioDataModule(L.LightningDataModule):
    def __init__(
        self,
        feature_extractor: SequenceFeatureExtractor,
        tokenizer: PreTrainedTokenizerBase,
        config: BaseTrainingConfig,
        token: str,
    ) -> None:
        super().__init__()
        self.config = config
        self.token = token
        self.collator = AudioTextCollator(
            feature_extractor,
            tokenizer,
            config.sampling_rate,
            config.max_text_length,
        )
        self.training_dataset: AudioBatchDataset | None = None
        self.validation_dataset: AudioBatchDataset | None = None

    def prepare_data(self) -> None:
        prepare_validation_data(
            self.token,
            self.config.dataset_cache_dir,
        )

    def setup(self, stage: str | None = None) -> None:
        if stage not in ("fit", None):
            return

        training_source, validation_source = load_lambda_audio_datasets(
            self.token,
            self.config.dataset_cache_dir,
            self.trainer.global_rank,
            self.trainer.world_size,
        )
        self.training_dataset = self._batch_dataset(
            training_source,
            self.config.train_max_batch_padded_audio_seconds,
            self.config.train_max_batch_attention_cost,
            True,
        )
        self.validation_dataset = self._batch_dataset(
            validation_source,
            self.config.validation_max_batch_padded_audio_seconds,
            self.config.validation_max_batch_attention_cost,
            False,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        if self.training_dataset is None:
            raise RuntimeError("Training dataset is not initialized.")
        return self._data_loader(self.training_dataset, persistent_workers=True)

    def val_dataloader(self) -> DataLoader[Any]:
        if self.validation_dataset is None:
            raise RuntimeError("Validation dataset is not initialized.")
        return self._data_loader(self.validation_dataset, persistent_workers=False)

    def set_epoch(self, epoch: int) -> None:
        if self.training_dataset is None:
            raise RuntimeError("Training dataset is not initialized.")
        self.training_dataset.set_epoch(epoch)

    def _batch_dataset(
        self,
        source: EpochTranscribedAudioDataset,
        max_padded_audio_seconds: float,
        max_attention_cost: int,
        show_initial_buffer_progress: bool,
    ) -> AudioBatchDataset:
        return AudioBatchDataset(
            source,
            max_batch_padded_audio_samples=int(max_padded_audio_seconds * self.config.sampling_rate),
            max_batch_attention_cost=max_attention_cost,
            encoder_frame_samples=self.config.encoder_frame_samples,
            bucket_buffer_size=self.config.bucket_buffer_size,
            show_initial_buffer_progress=show_initial_buffer_progress,
        )

    def _data_loader(
        self,
        dataset: AudioBatchDataset,
        persistent_workers: bool,
    ) -> DataLoader[Any]:
        return DataLoader(
            dataset,
            batch_size=None,
            collate_fn=self.collator,
            pin_memory=True,
            num_workers=self.config.num_workers,
            persistent_workers=persistent_workers,
            prefetch_factor=self.config.prefetch_factor,
            in_order=False,
        )
