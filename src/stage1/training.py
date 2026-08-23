from typing import Any

import torch
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from src.shared.lambda_audio import LambdaAudioModel
from src.shared.training import BaseLambdaAudioLightningModule
from src.stage1.config import Stage1Config


class Stage1LightningModule(BaseLambdaAudioLightningModule):
    def __init__(self, model: LambdaAudioModel, config: Stage1Config) -> None:
        super().__init__(model, config)
        self.config = config

    def configure_optimizers(self) -> dict[str, Any]:
        parameter_groups: dict[float, list[torch.nn.Parameter]] = {}
        for parameter in self.model.parameters():
            if not parameter.requires_grad:
                continue

            weight_decay = self.config.weight_decay if parameter.ndim >= 2 else 0.0
            parameter_groups.setdefault(weight_decay, []).append(parameter)

        groups = [
            {
                "params": parameters,
                "lr": self.config.learning_rate,
                "weight_decay": weight_decay,
            }
            for weight_decay, parameters in parameter_groups.items()
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
