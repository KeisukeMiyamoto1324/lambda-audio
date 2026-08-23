from typing import Any

import lightning as L
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase, SequenceFeatureExtractor

from src.shared.collator import AudioTextCollator
from src.shared.config import BaseTrainingConfig
from src.shared.dataset import LengthBucketBatchDataset, load_streaming_datasets
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
        self.training_dataset: LengthBucketBatchDataset | None = None
        self.validation_dataset: LengthBucketBatchDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage not in ("fit", None):
            return

        training_dataset, validation_dataset = load_streaming_datasets(
            self.config,
            self.trainer.global_rank,
            self.trainer.world_size,
        )
        self.training_dataset = LengthBucketBatchDataset(
            training_dataset,
            self.config.train_batch_size,
            self.config.bucket_buffer_multiplier,
            self.config.seed,
        )
        self.validation_dataset = LengthBucketBatchDataset(
            validation_dataset,
            self.config.validation_batch_size,
            self.config.bucket_buffer_multiplier,
            self.config.seed,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        if self.training_dataset is None:
            raise RuntimeError("Training dataset is not initialized.")

        return DataLoader(
            self.training_dataset,
            batch_size=None,
            collate_fn=self.collator,
            num_workers=self.config.dataloader_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        if self.validation_dataset is None:
            raise RuntimeError("Validation dataset is not initialized.")

        return DataLoader(
            self.validation_dataset,
            batch_size=None,
            collate_fn=self.collator,
            num_workers=self.config.dataloader_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def set_epoch(self, epoch: int) -> None:
        if self.training_dataset is None:
            raise RuntimeError("Training dataset is not initialized.")
        self.training_dataset.set_epoch(epoch)


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
