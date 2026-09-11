from typing import Generic, TypeVar, cast

import lightning as L
import torch

from src.cer.inference import generate_transcriptions
from src.cer.metric import compute_cer_counts
from src.cer.types import CerBatch
from src.data.types import AudioTextBatch
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
        self.training_token_count = 0
        self.register_buffer(
            "cer_counts",
            torch.zeros(5, dtype=torch.long),
            persistent=False,
        )

    def training_step(
        self,
        batch: AudioTextBatch,
        batch_idx: int,
    ) -> torch.Tensor:
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
            prog_bar=True,
            sync_dist=False,
            batch_size=batch["target_token_count"],
        )
        return output.loss

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        # ---------------------------------------------------------
        # Normalize accumulated and distributed gradients per target token.
        # ---------------------------------------------------------
        token_count = torch.tensor(
            self.training_token_count,
            device=self.device,
            dtype=torch.float64,
        )
        token_count = self.trainer.strategy.reduce(token_count, reduce_op="sum")
        gradient_scale = self.trainer.accumulate_grad_batches * self.trainer.world_size / token_count.item()
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    parameter.grad.mul_(gradient_scale)
        self.training_token_count = 0

    def on_train_batch_start(
        self,
        batch: AudioTextBatch,
        batch_idx: int,
    ) -> None:
        # ---------------------------------------------------------
        # Advance the numerical schedules used inside Zipformer.
        # ---------------------------------------------------------
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
        batch: AudioTextBatch | CerBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor | None:
        if dataloader_idx == 1:
            self._cer_step(cast(CerBatch, batch))
            return None

        audio_text_batch = cast(AudioTextBatch, batch)
        output = self.model(
            input_values=audio_text_batch["input_values"],
            audio_padding_mask=audio_text_batch["audio_padding_mask"],
            input_ids=audio_text_batch["input_ids"],
            attention_mask=audio_text_batch["attention_mask"],
            labels=audio_text_batch["labels"],
        )
        if output.loss is None:
            raise RuntimeError("The model did not return validation loss.")

        mean_loss = output.loss / audio_text_batch["target_token_count"]
        self.log(
            "validation_loss",
            mean_loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=audio_text_batch["target_token_count"],
            add_dataloader_idx=False,
        )
        return output.loss

    def on_validation_epoch_start(self) -> None:
        self.cer_counts.zero_()

    def on_validation_epoch_end(self) -> None:
        counts = self.trainer.strategy.reduce(self.cer_counts, reduce_op="sum")
        reference_characters = counts[3]
        if reference_characters.item() == 0:
            raise RuntimeError("CER validation references contain no characters.")
        cer = counts[:3].sum().to(torch.float64) / reference_characters
        self.log(
            "validation_cer",
            cer,
            prog_bar=True,
            sync_dist=False,
        )
        self.log("validation_cer_samples", counts[4], sync_dist=False)

    def _cer_step(self, batch: CerBatch) -> None:
        data_module = self.trainer.datamodule
        if not isinstance(data_module, LambdaAudioDataModule):
            raise TypeError("LambdaAudioDataModule is required.")
        hypotheses = generate_transcriptions(
            self.model,
            data_module.collator.tokenizer,
            batch,
            self.config.max_new_tokens,
        )
        counts = compute_cer_counts(batch["references"], hypotheses)
        self.cer_counts += torch.tensor(
            [
                counts.substitutions,
                counts.deletions,
                counts.insertions,
                counts.reference_characters,
                counts.samples,
            ],
            dtype=torch.long,
            device=self.cer_counts.device,
        )

    def on_train_epoch_start(self) -> None:
        data_module = self.trainer.datamodule
        if not isinstance(data_module, LambdaAudioDataModule):
            raise TypeError("LambdaAudioDataModule is required.")
        data_module.set_epoch(self.current_epoch)
