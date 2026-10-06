from typing import Any

import torch
from torch.optim import AdamW
from transformers import get_cosine_with_min_lr_schedule_with_warmup

from src.modeling.lambda_audio import LambdaAudioModel
from src.training.config import TrainingConfig


def create_optimizer_configuration(
    model: LambdaAudioModel,
    config: TrainingConfig,
) -> dict[str, Any]:
    # ---------------------------------------------------------
    # Group parameters by learning rate and weight decay policy.
    # ---------------------------------------------------------
    parameter_groups: dict[tuple[float, float], list[torch.nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        rate = config.adapter_learning_rate if name.startswith("adapter.") or ".cross_attention." in name else config.backbone_learning_rate
        weight_decay = config.weight_decay if parameter.ndim >= 2 else 0.0
        parameter_groups.setdefault((rate, weight_decay), []).append(parameter)

    groups = [
        {
            "params": parameters,
            "lr": rate,
            "weight_decay": weight_decay,
        }
        for (rate, weight_decay), parameters in parameter_groups.items()
    ]

    # ---------------------------------------------------------
    # Create AdamW and its warmup cosine learning-rate schedule.
    # ---------------------------------------------------------
    optimizer = AdamW(groups, betas=(0.9, 0.95), fused=True)
    scheduler = get_cosine_with_min_lr_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.max_steps,
        min_lr_rate=config.min_learning_rate_ratio,
    )
    return {
        "optimizer": optimizer,
        "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
    }
