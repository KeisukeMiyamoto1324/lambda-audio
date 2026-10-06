from typing import Any

import lightning as L
import torch

from src.data.types import AudioTextBatch
from src.modeling.lambda_audio import LambdaAudioModel
from src.modeling.zipformer.scaling import ScheduledFloat
from src.training.config import TrainingConfig
from src.training.data_module import LambdaAudioDataModule
from src.training.optimizer import create_optimizer_configuration


class LambdaAudioLightningModule(L.LightningModule):
    def __init__(self, model: LambdaAudioModel, config: TrainingConfig) -> None:
        super().__init__()
        self.model = model
        self.config = config
        self.training_token_count = 0

    def configure_optimizers(self) -> dict[str, Any]:
        return create_optimizer_configuration(self.model, self.config)

    def on_train_start(self) -> None:
        # ---------------------------------------------------------
        # Freeze only after DDP registers all weights and restores the step.
        # ---------------------------------------------------------
        self.model.set_backbones_trainable(self.global_step >= self.config.unfreeze_step)

    def training_step(
        self,
        batch: AudioTextBatch,
        batch_idx: int,
    ) -> torch.Tensor:
        # ---------------------------------------------------------
        # Compute training loss and track the number of target tokens.
        # ---------------------------------------------------------
        output = self.model(
            input_values=batch["input_values"],
            audio_padding_mask=batch["audio_padding_mask"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        if output.loss is None:
            raise RuntimeError("The model did not return training loss.")

        self.training_token_count += batch["target_token_count"]
        mean_loss = output.loss / batch["target_token_count"]
        self.log(
            "train_loss",
            mean_loss,
            on_step=True,
            prog_bar=False,
            sync_dist=False,
            batch_size=batch["target_token_count"],
        )
        return output.loss

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        # ---------------------------------------------------------
        # Normalize accumulated and distributed gradients per target token.
        # ---------------------------------------------------------
        gradient_scale: float | torch.Tensor
        if self.trainer.world_size == 1:
            gradient_scale = self.trainer.accumulate_grad_batches / self.training_token_count
        else:
            token_count = torch.tensor(self.training_token_count, device=self.device, dtype=torch.float64)
            token_count = self.trainer.strategy.reduce(token_count, reduce_op="sum")
            gradient_scale = self.trainer.accumulate_grad_batches * self.trainer.world_size / token_count
        gradients = [parameter.grad for group in optimizer.param_groups for parameter in group["params"] if parameter.grad is not None]
        torch._foreach_mul_(gradients, gradient_scale)
        self.training_token_count = 0

    def on_train_batch_start(
        self,
        batch: AudioTextBatch,
        batch_idx: int,
    ) -> None:
        # ---------------------------------------------------------
        # Unfreeze at an optimizer boundary and advance encoder schedules.
        # ---------------------------------------------------------
        if not self.model.backbones_trainable and self.global_step >= self.config.unfreeze_step:
            self.model.set_backbones_trainable(True)

        for name, module in self.model.encoder.named_modules():
            if isinstance(module, ScheduledFloat):
                module.batch_count = self.global_step
                module.name = f"encoder.{name}"

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: float | int | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        # ---------------------------------------------------------
        # Validate and clip gradients with one norm calculation.
        # ---------------------------------------------------------
        if gradient_clip_val is None:
            raise RuntimeError("A gradient clipping value is required.")
        parameters = [parameter for group in optimizer.param_groups for parameter in group["params"] if parameter.grad is not None]
        try:
            torch.nn.utils.clip_grad_norm_(
                parameters,
                max_norm=float(gradient_clip_val),
                error_if_nonfinite=True,
                foreach=True,
            )
        except RuntimeError as error:
            for name, parameter in self.model.named_parameters():
                if parameter.grad is None or torch.isfinite(parameter.grad).all():
                    continue
                raise FloatingPointError(f"Non-finite gradient at step={self.global_step}, parameter={name}.") from error
            raise FloatingPointError(f"Non-finite gradient norm at step={self.global_step}.") from error

    def validation_step(
        self,
        batch: AudioTextBatch,
        batch_idx: int,
    ) -> torch.Tensor:
        # ---------------------------------------------------------
        # Compute and log validation loss normalized by target tokens.
        # ---------------------------------------------------------
        output = self.model(
            input_values=batch["input_values"],
            audio_padding_mask=batch["audio_padding_mask"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        if output.loss is None:
            raise RuntimeError("The model did not return validation loss.")

        mean_loss = output.loss / batch["target_token_count"]
        self.log(
            "validation_loss",
            mean_loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch["target_token_count"],
        )
        return output.loss

    def on_train_epoch_start(self) -> None:
        # ---------------------------------------------------------
        # Pass the current epoch to the training data sampler.
        # ---------------------------------------------------------
        data_module = self.trainer.datamodule
        if not isinstance(data_module, LambdaAudioDataModule):
            raise TypeError("LambdaAudioDataModule is required.")
        data_module.set_epoch(self.current_epoch)
