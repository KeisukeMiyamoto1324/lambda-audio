from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from src.modeling.zipformer.scaling import ScheduledFloat
from src.stage1.config import Stage1Config
from src.stage1.training import Stage1LightningModule
from src.stage2.config import Stage2Config
from src.stage2.training import Stage2LightningModule


class DummyTrainingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adapter = nn.Linear(2, 2)
        self.encoder = nn.Linear(2, 2)


class DummyScheduledEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dropout = ScheduledFloat((0.0, 0.3), (100.0, 0.1))


class DummyScheduledModel(DummyTrainingModel):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = DummyScheduledEncoder()


class DummyLossModel(DummyTrainingModel):
    def forward(self, **batch: Any) -> SimpleNamespace:
        return SimpleNamespace(loss=torch.tensor(float("nan")))


def test_stage1_optimizer_uses_one_learning_rate() -> None:
    config = Stage1Config(output_dir=Path("outputs"))
    module = Stage1LightningModule(DummyTrainingModel(), config)  # type: ignore[arg-type]
    optimizer = module.configure_optimizers()["optimizer"]
    learning_rates = {group["initial_lr"] for group in optimizer.param_groups}

    assert learning_rates == {config.learning_rate}


def test_stage2_optimizer_uses_separate_learning_rates(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "model.ckpt"
    checkpoint_path.touch()
    config = Stage2Config(
        output_dir=Path("outputs"),
        initialization_checkpoint=checkpoint_path,
    )
    module = Stage2LightningModule(DummyTrainingModel(), config)  # type: ignore[arg-type]
    optimizer = module.configure_optimizers()["optimizer"]
    learning_rates = {group["initial_lr"] for group in optimizer.param_groups}

    assert learning_rates == {
        config.adapter_learning_rate,
        config.backbone_learning_rate,
    }


def test_stage2_optimizer_excludes_frozen_encoder(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "model.ckpt"
    checkpoint_path.touch()
    config = Stage2Config(
        output_dir=Path("outputs"),
        initialization_checkpoint=checkpoint_path,
    )
    model = DummyTrainingModel()
    model.encoder.requires_grad_(False)
    module = Stage2LightningModule(model, config)  # type: ignore[arg-type]

    optimizer = module.configure_optimizers()["optimizer"]
    optimized_parameters = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    assert not any(
        id(parameter) in optimized_parameters
        for parameter in model.encoder.parameters()
    )


def test_training_step_updates_zipformer_schedules() -> None:
    config = Stage1Config(output_dir=Path("outputs"))
    model = DummyScheduledModel()
    module = Stage1LightningModule(model, config)  # type: ignore[arg-type]
    module._trainer = SimpleNamespace(global_step=17)  # type: ignore[assignment]

    module.on_train_batch_start({}, 0)

    assert model.encoder.dropout.batch_count == 17
    assert model.encoder.dropout.name == "encoder.dropout"


def test_training_step_rejects_nonfinite_loss() -> None:
    config = Stage1Config(output_dir=Path("outputs"))
    module = Stage1LightningModule(DummyLossModel(), config)  # type: ignore[arg-type]

    with pytest.raises(FloatingPointError, match="step=0, batch=3"):
        module.training_step({}, 3)


def test_optimizer_step_identifies_nonfinite_gradient() -> None:
    config = Stage1Config(output_dir=Path("outputs"))
    model = DummyTrainingModel()
    module = Stage1LightningModule(model, config)  # type: ignore[arg-type]
    optimizer = torch.optim.AdamW(model.parameters())
    model.encoder.weight.grad = torch.full_like(
        model.encoder.weight,
        float("nan"),
    )

    with pytest.raises(
        FloatingPointError,
        match="parameter=encoder.weight",
    ):
        module.on_before_optimizer_step(optimizer)
