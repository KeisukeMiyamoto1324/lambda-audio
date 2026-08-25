from typing import Any

import lightning as L
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase, SequenceFeatureExtractor

from src.shared.collator import AudioTextCollator
from src.shared.config import BaseTrainingConfig
from src.shared.dataset import (
    AudioBatchDataset,
    TranscribedAudioDataset,
)
from src.shared.environment import load_hf_token
from src.shared.lambda_audio import LambdaAudioModel
from src.shared.metadata import prepare_metadata_index
from src.shared.reazon_dataset import (
    load_reazon_speech_datasets,
    prepare_reazon_speech_validation_shard,
)
from src.shared.zipformer.scaling import ScheduledFloat


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
        self.training_dataset: AudioBatchDataset | None = None
        self.validation_dataset: AudioBatchDataset | None = None

    def prepare_data(self) -> None:
        token = load_hf_token()
        prepare_metadata_index(token, self.config.metadata_index_path)
        prepare_reazon_speech_validation_shard(
            token,
            self.config.shard_cache_dir,
        )

    def setup(self, stage: str | None = None) -> None:
        if stage not in ("fit", None):
            return

        training_dataset, validation_dataset = load_reazon_speech_datasets(
            load_hf_token(),
            self.config.validation_samples,
            self.config.shuffle_buffer_size,
            self.config.sampling_rate,
            self.config.shard_cache_dir,
            self.config.seed,
            self.trainer.global_rank,
            self.trainer.world_size,
        )
        self.training_dataset = AudioBatchDataset(
            TranscribedAudioDataset(
                training_dataset,
                self.config.metadata_index_path,
            ),
            max_batch_padded_audio_samples=int(
                self.config.train_max_batch_padded_audio_seconds
                * self.config.sampling_rate
            ),
            max_batch_attention_cost=self.config.train_max_batch_attention_cost,
            encoder_frame_samples=self.config.encoder_frame_samples,
            bucket_buffer_size=self.config.bucket_buffer_size,
            seed=self.config.seed + self.trainer.global_rank,
        )
        self.validation_dataset = AudioBatchDataset(
            TranscribedAudioDataset(
                validation_dataset,
                self.config.metadata_index_path,
            ),
            max_batch_padded_audio_samples=int(
                self.config.validation_max_batch_padded_audio_seconds
                * self.config.sampling_rate
            ),
            max_batch_attention_cost=(
                self.config.validation_max_batch_attention_cost
            ),
            encoder_frame_samples=self.config.encoder_frame_samples,
            bucket_buffer_size=self.config.bucket_buffer_size,
            seed=self.config.seed + self.trainer.global_rank,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        if self.training_dataset is None:
            raise RuntimeError("Training dataset is not initialized.")

        return DataLoader(
            self.training_dataset,
            batch_size=None,
            collate_fn=self.collator,
            pin_memory=True,
            num_workers=self.config.num_workers,
            persistent_workers=True,
            prefetch_factor=self.config.prefetch_factor,
            in_order=False,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        if self.validation_dataset is None:
            raise RuntimeError("Validation dataset is not initialized.")

        return DataLoader(
            self.validation_dataset,
            batch_size=None,
            collate_fn=self.collator,
            pin_memory=True,
            num_workers=self.config.num_workers,
            persistent_workers=True,
            prefetch_factor=self.config.prefetch_factor,
            in_order=False,
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
        if not torch.isfinite(output.loss):
            raise FloatingPointError(
                "Non-finite training loss at "
                f"step={self.global_step}, batch={batch_idx}."
            )

        self.log(
            "train_loss",
            output.loss,
            on_step=True,
            prog_bar=True,
            sync_dist=False,
            batch_size=batch["input_ids"].shape[0],
        )
        return output.loss

    def on_train_batch_start(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        # ---------------------------------------------------------
        # Advance the numerical schedules used inside Zipformer.
        # ---------------------------------------------------------
        for name, module in self.model.encoder.named_modules():
            if isinstance(module, ScheduledFloat):
                module.batch_count = self.global_step
                module.name = f"encoder.{name}"

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        # ---------------------------------------------------------
        # Reject non-finite gradients before they can corrupt weights.
        # ---------------------------------------------------------
        parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.grad is not None
        ]
        try:
            torch.nn.utils.clip_grad_norm_(
                parameters,
                max_norm=float("inf"),
                error_if_nonfinite=True,
                foreach=True,
            )
        except RuntimeError as error:
            for name, parameter in self.model.named_parameters():
                if parameter.grad is None or torch.isfinite(parameter.grad).all():
                    continue
                raise FloatingPointError(
                    "Non-finite gradient at "
                    f"step={self.global_step}, parameter={name}."
                ) from error
            raise FloatingPointError(
                f"Non-finite gradient norm at step={self.global_step}."
            ) from error

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
