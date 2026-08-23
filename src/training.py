from typing import Any

import lightning as L
import torch
from datasets import IterableDataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    PreTrainedTokenizerBase,
    SequenceFeatureExtractor,
    get_cosine_schedule_with_warmup,
)

from src.collator import AudioTextCollator
from src.config import TrainingConfig
from src.dataset import load_streaming_datasets
from src.lambda_audio import LambdaAudioModel


class LambdaAudioDataModule(L.LightningDataModule):
    def __init__(
        self,
        feature_extractor: SequenceFeatureExtractor,
        tokenizer: PreTrainedTokenizerBase,
        config: TrainingConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.collator = AudioTextCollator(
            feature_extractor,
            tokenizer,
            config.sampling_rate,
            config.max_text_length,
        )
        self.training_dataset: IterableDataset | None = None
        self.validation_dataset: IterableDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage not in ("fit", None):
            return

        self.training_dataset, self.validation_dataset = load_streaming_datasets(
            self.config,
            self.trainer.global_rank,
            self.trainer.world_size,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        if self.training_dataset is None:
            raise RuntimeError("Training dataset is not initialized.")

        return DataLoader(
            self.training_dataset,
            batch_size=self.config.train_batch_size,
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
            batch_size=self.config.validation_batch_size,
            collate_fn=self.collator,
            num_workers=self.config.dataloader_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def set_epoch(self, epoch: int) -> None:
        if self.training_dataset is None:
            raise RuntimeError("Training dataset is not initialized.")
        self.training_dataset.set_epoch(epoch)


class LambdaAudioLightningModule(L.LightningModule):
    def __init__(self, model: LambdaAudioModel, config: TrainingConfig) -> None:
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
            sync_dist=True,
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

    def configure_optimizers(self) -> dict[str, Any]:
        # ---------------------------------------------------------
        # Use a lower learning rate for pretrained backbone weights.
        # ---------------------------------------------------------
        parameter_groups: dict[tuple[float, float], list[torch.nn.Parameter]] = {}
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue

            is_audio_module = name.startswith("adapter.") or ".cross_attention." in name
            learning_rate = (
                self.config.adapter_learning_rate
                if is_audio_module
                else self.config.backbone_learning_rate
            )
            weight_decay = self.config.weight_decay if parameter.ndim >= 2 else 0.0
            group_key = (learning_rate, weight_decay)
            parameter_groups.setdefault(group_key, []).append(parameter)

        groups = [
            {
                "params": parameters,
                "lr": learning_rate,
                "weight_decay": weight_decay,
            }
            for (learning_rate, weight_decay), parameters in parameter_groups.items()
        ]
        optimizer = AdamW(groups, betas=(0.9, 0.95))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.config.max_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
