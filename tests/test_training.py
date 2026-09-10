import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from torch import nn

from src.stage1.config import Stage1Config
from src.stage1.training import Stage1LightningModule
from src.stage2.config import Stage2Config
from src.stage2.training import Stage2LightningModule
from src.training import trainer as trainer_module
from src.training.inference_monitor import ValidationInferenceMonitor
from src.training.trainer import create_trainer


class DummyTrainingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adapter = nn.Linear(2, 2)
        self.ctc_output = nn.Linear(2, 2)
        self.encoder = nn.Linear(2, 2)
        self.decoder = nn.Linear(2, 2)


class DummyLossModel(DummyTrainingModel):
    def forward(self, **batch: Any) -> SimpleNamespace:
        return SimpleNamespace(loss=torch.tensor(float("nan")))


def test_stage1_optimizer_uses_one_learning_rate() -> None:
    config = Stage1Config(output_dir=Path("outputs"))
    module = Stage1LightningModule(DummyTrainingModel(), config)  # type: ignore[arg-type]
    optimizer_configuration = module.configure_optimizers()
    optimizer = optimizer_configuration["optimizer"]
    learning_rates = {group["initial_lr"] for group in optimizer.param_groups}
    scheduler = optimizer_configuration["lr_scheduler"]["scheduler"]

    assert learning_rates == {config.learning_rate}
    assert optimizer.defaults["fused"] is True
    assert scheduler.lr_lambdas[0](config.max_steps) == pytest.approx(config.min_learning_rate_ratio)


def test_lightning_module_puts_pretrained_model_in_training_mode() -> None:
    config = Stage1Config(output_dir=Path("outputs"))
    model = DummyTrainingModel().eval()

    Stage1LightningModule(model, config)  # type: ignore[arg-type]

    assert all(module.training for module in model.modules())


def test_stage2_optimizer_uses_separate_learning_rates(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "model.ckpt"
    checkpoint_path.touch()
    config = Stage2Config(
        output_dir=Path("outputs"),
        initialization_checkpoint=checkpoint_path,
    )
    model = DummyTrainingModel()
    module = Stage2LightningModule(model, config)  # type: ignore[arg-type]
    optimizer = module.configure_optimizers()["optimizer"]
    learning_rates = {group["initial_lr"] for group in optimizer.param_groups}

    assert learning_rates == {
        config.adapter_learning_rate,
        config.encoder_learning_rate,
        config.backbone_learning_rate,
    }
    ctc_parameter = model.ctc_output.weight
    ctc_group = next(group for group in optimizer.param_groups if any(parameter is ctc_parameter for parameter in group["params"]))
    assert ctc_group["initial_lr"] == config.adapter_learning_rate


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
    optimized_parameters = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}

    assert not any(id(parameter) in optimized_parameters for parameter in model.encoder.parameters())


def test_training_step_defers_nonfinite_check_to_gradient_clipping(monkeypatch: pytest.MonkeyPatch) -> None:
    config = Stage1Config(output_dir=Path("outputs"))
    module = Stage1LightningModule(DummyLossModel(), config)  # type: ignore[arg-type]
    monkeypatch.setattr(module, "log", lambda *args, **kwargs: None)

    loss = module.training_step({"input_ids": torch.ones(1, 1)}, 3)

    assert torch.isnan(loss)


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
        module.configure_gradient_clipping(optimizer, config.max_grad_norm, "norm")


def test_create_trainer_saves_config_and_full_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trainer_arguments: dict[str, Any] = {}

    def create_dummy_trainer(**kwargs: Any) -> SimpleNamespace:
        trainer_arguments.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(trainer_module.L, "Trainer", create_dummy_trainer)
    config = Stage1Config(output_dir=tmp_path)

    create_trainer(config)

    saved_config = json.loads((tmp_path / "training_config.json").read_text(encoding="utf-8"))
    checkpoint = next(callback for callback in trainer_arguments["callbacks"] if isinstance(callback, ModelCheckpoint))
    assert saved_config == config.model_dump(mode="json")
    assert checkpoint.save_last is True
    assert checkpoint.save_weights_only is False
    assert any(isinstance(callback, ValidationInferenceMonitor) for callback in trainer_arguments["callbacks"])


def test_create_trainer_uses_gradient_bucket_views_for_multiple_gpus(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trainer_arguments: dict[str, Any] = {}

    def create_dummy_trainer(**kwargs: Any) -> SimpleNamespace:
        trainer_arguments.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(trainer_module.L, "Trainer", create_dummy_trainer)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    create_trainer(Stage1Config(output_dir=tmp_path))

    strategy = trainer_arguments["strategy"]
    assert isinstance(strategy, DDPStrategy)
    assert strategy._ddp_kwargs["gradient_as_bucket_view"] is True
