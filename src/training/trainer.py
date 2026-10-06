import warnings

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.utilities.warnings import PossibleUserWarning

from src.environment import configure_cuda_allocator
from src.training.config import TrainingConfig
from src.training.inference_monitor import ValidationInferenceMonitor
from src.training.performance import TrainingPerformanceMonitor, TrainingProgressBar


class TrainingEarlyStopping(EarlyStopping):
    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        # ---------------------------------------------------------
        # Delay early stopping until backbone weights are unfrozen.
        # ---------------------------------------------------------
        if trainer.global_step <= pl_module.config.unfreeze_step:
            return
        super().on_validation_end(trainer, pl_module)


def configure_training_environment(seed: int) -> None:
    # ---------------------------------------------------------
    # Check accelerator support and configure reproducible execution.
    # ---------------------------------------------------------
    configure_cuda_allocator()
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires an NVIDIA CUDA GPU.")

    # ---------------------------------------------------------
    # Ignore a known Lightning 2.6 incompatibility with PyTorch 2.14.
    # ---------------------------------------------------------
    warnings.filterwarnings(
        "ignore",
        message=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*",
        category=FutureWarning,
        module=r"lightning\.pytorch\.utilities\._pytree",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Found \d+ module\(s\) in eval mode at the start of training.*",
        category=PossibleUserWarning,
    )
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    L.seed_everything(seed, workers=True)


def create_trainer(config: TrainingConfig) -> L.Trainer:
    # ---------------------------------------------------------
    # Create the output folder and save the resolved configuration.
    # ---------------------------------------------------------
    config.output_dir.mkdir(parents=True, exist_ok=False)
    (config.output_dir / "training_config.json").write_text(
        f"{config.model_dump_json(indent=2)}\n",
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # Configure checkpointing, stopping, progress, and monitoring.
    # ---------------------------------------------------------
    checkpoint = ModelCheckpoint(
        dirpath=config.output_dir,
        filename="model",
        monitor="validation_loss",
        mode="min",
        every_n_train_steps=config.save_steps,
        save_top_k=1,
        save_last=True,
        save_weights_only=False,
        save_on_train_epoch_end=False,
        enable_version_counter=False,
    )
    early_stopping = TrainingEarlyStopping(
        monitor="validation_loss",
        mode="min",
        patience=config.early_stopping_patience,
        check_on_train_epoch_end=False,
        min_delta=config.early_stopping_min_delta,
    )
    progress_bar = TrainingProgressBar(refresh_rate=config.log_steps)
    callbacks = [
        checkpoint,
        early_stopping,
        TrainingPerformanceMonitor(config.sampling_rate, config.log_steps),
        progress_bar,
        ValidationInferenceMonitor(config, progress_bar),
    ]

    # ---------------------------------------------------------
    # Assemble the trainer with distributed, precision, and logging options.
    # ---------------------------------------------------------
    return L.Trainer(
        accelerator="gpu",
        devices="auto",
        strategy=DDPStrategy(find_unused_parameters=True) if torch.cuda.device_count() > 1 else "auto",
        precision="bf16-mixed",
        max_epochs=-1,
        max_steps=config.max_steps,
        callbacks=callbacks,
        logger=CSVLogger(config.output_dir, name="logs"),
        accumulate_grad_batches=config.gradient_accumulation_steps,
        gradient_clip_val=config.max_grad_norm,
        log_every_n_steps=config.log_steps,
        val_check_interval=(config.evaluation_steps * config.gradient_accumulation_steps),
        check_val_every_n_epoch=None,
        limit_val_batches=1.0,
        num_sanity_val_steps=0,
        use_distributed_sampler=False,
        default_root_dir=config.output_dir,
    )
