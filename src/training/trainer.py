import warnings

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy

from src.environment import configure_cuda_allocator
from src.training.config import BaseTrainingConfig
from src.training.inference_monitor import ValidationInferenceMonitor
from src.training.performance import TrainingPerformanceMonitor, TrainingProgressBar


def configure_training_environment(seed: int) -> None:
    # ---------------------------------------------------------
    # Suppress a harmless incompatibility warning in Lightning's PyTree shim.
    # Lightning still uses LeafSpec internally with newer PyTorch releases.
    # ---------------------------------------------------------
    warnings.filterwarnings(
        "ignore",
        message=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*",
        category=FutureWarning,
        module=r"lightning\.pytorch\.utilities\._pytree",
    )
    configure_cuda_allocator()
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires an NVIDIA CUDA GPU.")

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    L.seed_everything(seed, workers=True)


def create_trainer(config: BaseTrainingConfig) -> L.Trainer:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "training_config.json").write_text(
        f"{config.model_dump_json(indent=2)}\n",
        encoding="utf-8",
    )
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
    early_stopping = EarlyStopping(
        monitor="validation_loss",
        mode="min",
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta,
    )
    progress_bar = TrainingProgressBar()
    callbacks = [
        checkpoint,
        early_stopping,
        TrainingPerformanceMonitor(config.sampling_rate, config.log_steps),
        progress_bar,
        ValidationInferenceMonitor(config, progress_bar),
    ]
    strategy = DDPStrategy(gradient_as_bucket_view=True) if torch.cuda.device_count() > 1 else "auto"
    return L.Trainer(
        accelerator="gpu",
        devices="auto",
        strategy=strategy,
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
        limit_val_batches=config.validation_batches,
        num_sanity_val_steps=0,
        use_distributed_sampler=False,
        default_root_dir=config.output_dir,
    )
