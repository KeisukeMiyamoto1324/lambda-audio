from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from src.config import TrainingConfig


def create_optimizer(model: nn.Module, config: TrainingConfig) -> Optimizer:
    # ---------------------------------------------------------
    # Use a lower learning rate for pretrained backbone weights.
    # ---------------------------------------------------------
    parameter_groups: dict[tuple[float, float], list[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        is_audio_module = name.startswith("adapter.") or ".cross_attention." in name
        learning_rate = (
            config.adapter_learning_rate
            if is_audio_module
            else config.backbone_learning_rate
        )
        weight_decay = config.weight_decay if parameter.ndim >= 2 else 0.0
        group_key = (learning_rate, weight_decay)
        parameter_groups.setdefault(group_key, []).append(parameter)

    groups = [
        {"params": parameters, "lr": learning_rate, "weight_decay": weight_decay}
        for (learning_rate, weight_decay), parameters in parameter_groups.items()
    ]
    return AdamW(groups, betas=(0.9, 0.95))


def create_scheduler(optimizer: Optimizer, config: TrainingConfig) -> LRScheduler:
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.max_steps,
    )


def create_dataloader(
    dataset: Any,
    collator: Any,
    batch_size: int,
    config: TrainingConfig,
) -> DataLoader[Any]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=config.dataloader_workers,
        pin_memory=True,
        persistent_workers=True,
    )


def evaluate(
    model: nn.Module,
    dataloader: DataLoader[Any],
    accelerator: Accelerator,
    max_batches: int,
) -> float:
    # ---------------------------------------------------------
    # Aggregate validation loss across all distributed workers.
    # ---------------------------------------------------------
    model.eval()
    statistics = torch.zeros(2, device=accelerator.device, dtype=torch.float64)

    for batch_index, batch in enumerate(dataloader):
        if batch_index >= max_batches:
            break

        with torch.no_grad():
            output = model(**batch)

        if output.loss is None:
            raise RuntimeError("The model did not return validation loss.")

        batch_size = batch["input_ids"].shape[0]
        statistics[0] += output.loss.detach().double() * batch_size
        statistics[1] += batch_size

    statistics = accelerator.reduce(statistics, reduction="sum")
    model.train()
    return (statistics[0] / statistics[1]).item()


def save_model(
    model: nn.Module,
    accelerator: Accelerator,
    output_path: Path,
) -> None:
    # ---------------------------------------------------------
    # Save one portable state dict for the next training stage.
    # ---------------------------------------------------------
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    accelerator.save(accelerator.get_state_dict(model), output_path)


def train(
    model: nn.Module,
    train_dataloader: DataLoader[Any],
    validation_dataloader: DataLoader[Any],
    optimizer: Optimizer,
    scheduler: LRScheduler,
    accelerator: Accelerator,
    config: TrainingConfig,
) -> None:
    model.train()
    optimizer.zero_grad()
    completed_steps = 0
    data_pass = 0
    stage_directory = config.output_dir / f"stage-{config.stage}"

    while completed_steps < config.max_steps:
        train_dataloader.set_epoch(data_pass)
        batches_in_pass = 0

        for batch in train_dataloader:
            batches_in_pass += 1
            with accelerator.accumulate(model):
                output = model(**batch)
                if output.loss is None:
                    raise RuntimeError("The model did not return training loss.")

                accelerator.backward(output.loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)

                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                optimizer.zero_grad()

            if not accelerator.sync_gradients:
                continue

            completed_steps += 1
            if completed_steps % config.log_steps == 0:
                mean_loss = accelerator.reduce(output.loss.detach(), reduction="mean")
                accelerator.print(
                    f"stage={config.stage} step={completed_steps} "
                    f"loss={mean_loss.item():.4f}"
                )

            if completed_steps % config.evaluation_steps == 0:
                validation_loss = evaluate(
                    model,
                    validation_dataloader,
                    accelerator,
                    config.validation_batches,
                )
                accelerator.print(
                    f"stage={config.stage} step={completed_steps} "
                    f"validation_loss={validation_loss:.4f}"
                )

            if completed_steps % config.save_steps == 0:
                save_model(model, accelerator, stage_directory / "model.pt")

            if completed_steps >= config.max_steps:
                break

        if batches_in_pass == 0:
            raise RuntimeError("The streaming training dataset returned no samples.")
        data_pass += 1

    save_model(model, accelerator, stage_directory / "model.pt")
