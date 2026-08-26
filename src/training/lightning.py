from typing import Generic, TypeVar

import lightning as L
import torch
from src.modeling.lambda_audio import LambdaAudioModel
from src.modeling.zipformer.scaling import ScheduledFloat
from src.training.config import BaseTrainingConfig
from src.training.data_module import LambdaAudioDataModule


TrainingConfigT = TypeVar("TrainingConfigT", bound=BaseTrainingConfig)


class BaseLambdaAudioLightningModule(
    L.LightningModule,
    Generic[TrainingConfigT],
):
    def __init__(self, model: LambdaAudioModel, config: TrainingConfigT) -> None:
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
