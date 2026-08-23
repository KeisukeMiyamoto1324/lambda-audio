from typing import Any

import lightning as L
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase, SequenceFeatureExtractor

from src.shared.collator import AudioTextCollator
from src.shared.config import BaseTrainingConfig
from src.shared.dataset import (
    LengthBucketBatchSampler,
    PreparedAudioDataset,
    load_prepared_datasets,
)
from src.shared.lambda_audio import LambdaAudioModel


class LambdaAudioDataModule(L.LightningDataModule):
    def __init__(
        self,
        feature_extractor: SequenceFeatureExtractor,
        tokenizer: PreTrainedTokenizerBase,
        config: BaseTrainingConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.collator = AudioTextCollator(
            feature_extractor,
            tokenizer,
            config.sampling_rate,
            config.max_text_length,
        )
        self.training_dataset: PreparedAudioDataset | None = None
        self.validation_dataset: PreparedAudioDataset | None = None
        self.training_sampler: LengthBucketBatchSampler | None = None
        self.validation_sampler: LengthBucketBatchSampler | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage not in ("fit", None):
            return

        training_dataset, validation_dataset = load_prepared_datasets(
            self.config.data_dir
        )
        self.training_dataset = PreparedAudioDataset(training_dataset)
        self.validation_dataset = PreparedAudioDataset(validation_dataset)
        self.training_sampler = LengthBucketBatchSampler(
            training_dataset["audio_length"],
            training_dataset["text_length"],
            int(
                self.config.train_max_batch_audio_seconds
                * self.config.sampling_rate
            ),
            self.config.train_max_batch_examples,
            self.config.bucket_window_size,
            self.config.seed,
            self.trainer.global_rank,
            self.trainer.world_size,
            shuffle=True,
        )
        self.validation_sampler = LengthBucketBatchSampler(
            validation_dataset["audio_length"],
            validation_dataset["text_length"],
            int(
                self.config.validation_max_batch_audio_seconds
                * self.config.sampling_rate
            ),
            self.config.validation_max_batch_examples,
            self.config.bucket_window_size,
            self.config.seed,
            self.trainer.global_rank,
            self.trainer.world_size,
            shuffle=False,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        if self.training_dataset is None or self.training_sampler is None:
            raise RuntimeError("Training dataset is not initialized.")

        return DataLoader(
            self.training_dataset,
            batch_sampler=self.training_sampler,
            collate_fn=self.collator,
            num_workers=self.config.dataloader_workers,
            prefetch_factor=self.config.prefetch_factor,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        if self.validation_dataset is None or self.validation_sampler is None:
            raise RuntimeError("Validation dataset is not initialized.")

        return DataLoader(
            self.validation_dataset,
            batch_sampler=self.validation_sampler,
            collate_fn=self.collator,
            num_workers=self.config.dataloader_workers,
            prefetch_factor=self.config.prefetch_factor,
            pin_memory=True,
            persistent_workers=True,
        )

    def set_epoch(self, epoch: int) -> None:
        if self.training_sampler is None:
            raise RuntimeError("Training dataset is not initialized.")
        self.training_sampler.set_epoch(epoch)


class BaseLambdaAudioLightningModule(L.LightningModule):
    def __init__(self, model: LambdaAudioModel, config: BaseTrainingConfig) -> None:
        super().__init__()
        self.model = model
        self.config = config

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        output = self.model(**batch)
        if output.loss is None:
            raise RuntimeError("The model did not return training loss.")

        self.log(
            "train_loss",
            output.loss,
            on_step=True,
            prog_bar=True,
            sync_dist=False,
            batch_size=batch["input_ids"].shape[0],
        )
        return output.loss

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        output = self.model(**batch)
        if output.loss is None:
            raise RuntimeError("The model did not return validation loss.")

        self.log(
            "validation_loss",
            output.loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch["input_ids"].shape[0],
        )
        return output.loss

    def on_train_epoch_start(self) -> None:
        data_module = self.trainer.datamodule
        if not isinstance(data_module, LambdaAudioDataModule):
            raise TypeError("LambdaAudioDataModule is required.")
        data_module.set_epoch(self.current_epoch)
