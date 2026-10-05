import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import lightning as L
import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import nn

from src.data.types import AudioTextBatch
from src.modeling.cross_attention import GatedCrossAttention
from src.modeling.lambda_audio import LambdaAudioModel
from src.modeling.zipformer.scaling import ScheduledFloat
from src.training import train as train_module
from src.training import trainer as trainer_module
from src.training.config import TrainingConfig
from src.training.inference_monitor import ValidationInferenceMonitor
from src.training.lightning import LambdaAudioLightningModule
from src.training.performance import TrainingProgressBar
from src.training.trainer import create_trainer


class DummyTrainingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbones_trainable = True
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


def dummy_audio_text_batch() -> AudioTextBatch:
    return cast(
        AudioTextBatch,
        {
            "input_values": torch.empty(1, 1),
            "audio_padding_mask": torch.zeros(1, 1, dtype=torch.bool),
            "input_ids": torch.ones(1, 1, dtype=torch.long),
            "attention_mask": torch.ones(1, 1, dtype=torch.long),
            "labels": torch.ones(1, 1, dtype=torch.long),
            "target_token_count": 1,
            "audio_valid_samples": 1,
            "audio_padded_samples": 1,
        },
    )


def test_optimizer_includes_frozen_parameters_and_separate_rates() -> None:
    config = TrainingConfig(adapter_learning_rate=1e-4, backbone_learning_rate=2e-5)
    model = DummyTrainingModel()
    model.encoder.requires_grad_(False)
    module = LambdaAudioLightningModule(model, config)  # type: ignore[arg-type]
    optimizer = module.configure_optimizers()["optimizer"]
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert optimized == {id(p) for p in model.parameters()}
    assert {group["initial_lr"] for group in optimizer.param_groups} == {1e-4, 2e-5}
    assert optimizer.defaults["fused"] is True


def test_training_step_updates_zipformer_schedules() -> None:
    config = TrainingConfig(output_dir=Path("outputs"))
    model = DummyScheduledModel()
    module = LambdaAudioLightningModule(model, config)  # type: ignore[arg-type]
    module._trainer = SimpleNamespace(global_step=17)  # type: ignore[assignment]

    module.on_train_batch_start({}, 0)

    assert model.encoder.dropout.batch_count == 17
    assert model.encoder.dropout.name == "encoder.dropout"


def test_gradient_clipping_identifies_nonfinite_gradient() -> None:
    config = TrainingConfig(output_dir=Path("outputs"))
    model = DummyTrainingModel()
    module = LambdaAudioLightningModule(model, config)  # type: ignore[arg-type]
    optimizer = torch.optim.AdamW(model.parameters())
    model.encoder.weight.grad = torch.full_like(
        model.encoder.weight,
        float("nan"),
    )

    with pytest.raises(
        FloatingPointError,
        match="parameter=encoder.weight",
    ):
        module.configure_gradient_clipping(optimizer, 1.0, "norm")


def test_gradient_clipping_limits_total_norm() -> None:
    config = TrainingConfig(output_dir=Path("outputs"))
    model = DummyTrainingModel()
    module = LambdaAudioLightningModule(model, config)  # type: ignore[arg-type]
    optimizer = torch.optim.AdamW(model.parameters())
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 10.0)

    module.configure_gradient_clipping(optimizer, 0.5, "norm")

    total_norm = torch.linalg.vector_norm(torch.stack([parameter.grad.norm() for parameter in model.parameters()]))
    assert total_norm.item() == pytest.approx(0.5)


def test_accumulated_gradients_are_normalized_by_token_count() -> None:
    config = TrainingConfig(
        output_dir=Path("outputs"),
        gradient_accumulation_steps=4,
    )
    model = DummyTrainingModel()
    module = LambdaAudioLightningModule(model, config)  # type: ignore[arg-type]
    module._trainer = SimpleNamespace(  # type: ignore[assignment]
        accumulate_grad_batches=4,
        world_size=2,
        strategy=SimpleNamespace(reduce=lambda value, reduce_op: value * 2),
    )
    module.training_token_count = 10
    optimizer = torch.optim.AdamW(model.parameters())
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)

    module.on_before_optimizer_step(optimizer)

    for parameter in model.parameters():
        torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 0.4))
    assert module.training_token_count == 0


def test_create_trainer_saves_config_and_full_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trainer_arguments: dict[str, Any] = {}

    def create_dummy_trainer(**kwargs: Any) -> SimpleNamespace:
        trainer_arguments.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(trainer_module.L, "Trainer", create_dummy_trainer)
    config = TrainingConfig(output_dir=tmp_path / "training")

    create_trainer(config)

    saved_config = json.loads((config.output_dir / "training_config.json").read_text(encoding="utf-8"))
    checkpoint = next(callback for callback in trainer_arguments["callbacks"] if isinstance(callback, ModelCheckpoint))
    assert saved_config == config.model_dump(mode="json")
    assert checkpoint.save_last is True
    assert checkpoint.save_weights_only is False
    assert any(isinstance(callback, ValidationInferenceMonitor) for callback in trainer_arguments["callbacks"])
    progress_bar = next(callback for callback in trainer_arguments["callbacks"] if isinstance(callback, TrainingProgressBar))
    assert progress_bar.refresh_rate == config.log_steps


@pytest.mark.parametrize("entrypoint", ["main", "create_trainer"])
@pytest.mark.parametrize("existing_file", [False, True])
def test_training_rejects_existing_output_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    entrypoint: str,
    existing_file: bool,
) -> None:
    output_dir = tmp_path / "training"
    if existing_file:
        output_dir.write_text("existing output", encoding="utf-8")
        saved_file = output_dir
    else:
        output_dir.mkdir()
        saved_file = output_dir / "training_config.json"
        saved_file.write_text("existing output", encoding="utf-8")
    config = TrainingConfig(output_dir=output_dir)
    monkeypatch.setattr(train_module, "parse_arguments", lambda: config)

    def reject_environment_setup(seed: int) -> None:
        pytest.fail("Training environment must not be initialized for an existing output path.")

    monkeypatch.setattr(train_module, "configure_training_environment", reject_environment_setup)

    with pytest.raises(FileExistsError):
        if entrypoint == "main":
            train_module.main()
        else:
            create_trainer(config)

    assert saved_file.read_text(encoding="utf-8") == "existing output"


class TinyTrainingModel(LambdaAudioModel):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.backbones_trainable = True
        self.encoder = nn.Linear(2, 2)
        self.adapter = nn.Linear(2, 2)
        self.decoder = nn.Module()
        self.decoder.projection = nn.Linear(2, 2)
        self.decoder.cross_attention = GatedCrossAttention(2, 1, 1e-5)

    def forward(self, **kwargs: Any) -> Any:
        return SimpleNamespace(loss=sum(parameter.square().sum() for parameter in self.parameters()))


class TrainingRecorder(L.Callback):
    def __init__(self, checkpoint: Path, stop_step: int | None = None) -> None:
        self.checkpoint = checkpoint
        self.stop_step = stop_step
        self.states: dict[int, dict[str, torch.Tensor]] = {}
        self.trainable: list[tuple[int, bool]] = []

    def on_before_optimizer_step(self, trainer: L.Trainer, pl_module: LambdaAudioLightningModule, optimizer: torch.optim.Optimizer) -> None:
        self.trainable.append((trainer.global_step, pl_module.model.backbones_trainable))
        assert pl_module.model.encoder.training == pl_module.model.backbones_trainable
        assert pl_module.model.decoder.training == pl_module.model.backbones_trainable
        assert pl_module.model.adapter.training
        assert pl_module.model.decoder.cross_attention.training
        for name, parameter in pl_module.model.named_parameters():
            expected = pl_module.model.backbones_trainable or name.startswith("adapter.") or ".cross_attention." in name
            assert (parameter.grad is not None) == expected

    def on_train_batch_end(self, trainer: L.Trainer, pl_module: LambdaAudioLightningModule, outputs: Any, batch: Any, batch_idx: int) -> None:
        self.states[trainer.global_step] = {name: value.clone() for name, value in pl_module.model.state_dict().items()}
        if trainer.global_step == self.stop_step:
            trainer.save_checkpoint(self.checkpoint)
            trainer.should_stop = True


@pytest.mark.parametrize("stop_step", [1, 2, 3])
def test_transition_and_resume_match_continuous_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stop_step: int) -> None:
    config = TrainingConfig(max_steps=4, unfreeze_step=2, warmup_steps=1, gradient_accumulation_steps=2)
    initial = TinyTrainingModel().state_dict()
    runs: list[tuple[LambdaAudioLightningModule, L.Trainer, TrainingRecorder]] = []
    checkpoint = tmp_path / "last.ckpt"
    monkeypatch.setattr(LambdaAudioLightningModule, "on_train_epoch_start", lambda self: None)
    for index in range(3):
        model = TinyTrainingModel()
        model.load_state_dict(initial)
        module = LambdaAudioLightningModule(model, config)
        recorder = TrainingRecorder(checkpoint, stop_step if index == 1 else None)
        trainer = L.Trainer(
            accelerator="cpu",
            devices=1,
            max_steps=4,
            max_epochs=-1,
            accumulate_grad_batches=2,
            gradient_clip_val=1.0,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[recorder],
            val_check_interval=2,
            num_sanity_val_steps=0,
        )
        batches = torch.utils.data.DataLoader([dummy_audio_text_batch() for _ in range(8)], batch_size=None)
        trainer.fit(module, train_dataloaders=batches, val_dataloaders=batches, ckpt_path=checkpoint if index == 2 else None)
        runs.append((module, trainer, recorder))

    continuous, trainer, recorder = runs[0]
    resumed, resumed_trainer, resumed_recorder = runs[2]
    assert recorder.trainable == [(0, False), (1, False), (2, True), (3, True)]
    assert resumed_recorder.trainable[0] == (stop_step, stop_step >= config.unfreeze_step)
    for name, original in initial.items():
        if name.startswith("encoder.") or name.startswith("decoder.projection."):
            torch.testing.assert_close(recorder.states[2][name], original, rtol=0, atol=0)
            assert not torch.equal(recorder.states[3][name], original)
    assert not torch.equal(recorder.states[2]["adapter.weight"], initial["adapter.weight"])
    assert not torch.equal(recorder.states[2]["decoder.cross_attention.norm.weight"], initial["decoder.cross_attention.norm.weight"])
    torch.testing.assert_close(continuous.state_dict(), resumed.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(trainer.optimizers[0].state_dict(), resumed_trainer.optimizers[0].state_dict(), rtol=0, atol=0)
    assert trainer.lr_scheduler_configs[0].scheduler.state_dict() == resumed_trainer.lr_scheduler_configs[0].scheduler.state_dict()


def test_zero_unfreeze_step_and_eval_modes() -> None:
    model = TinyTrainingModel()
    module = LambdaAudioLightningModule(model, TrainingConfig(unfreeze_step=0))
    module._trainer = SimpleNamespace(global_step=0)  # type: ignore[assignment]
    module.on_train_start()
    assert all(parameter.requires_grad for parameter in model.parameters())
    model.set_backbones_trainable(False)
    model.eval()
    assert all(not child.training for child in model.modules())
    model.train()
    assert not model.encoder.training
    assert not model.decoder.training
    assert model.adapter.training
    assert model.decoder.cross_attention.training
    model.set_backbones_trainable(True)
    assert all(child.training for child in model.modules())


def test_early_stopping_starts_after_unfreeze(monkeypatch: pytest.MonkeyPatch) -> None:
    from lightning.pytorch.callbacks import EarlyStopping

    checks: list[int] = []
    monkeypatch.setattr(EarlyStopping, "on_validation_end", lambda self, trainer, module: checks.append(trainer.global_step))
    callback = trainer_module.TrainingEarlyStopping(monitor="validation_loss", check_on_train_epoch_end=False)
    module = LambdaAudioLightningModule(TinyTrainingModel(), TrainingConfig(unfreeze_step=2))
    for step in range(4):
        callback.on_validation_end(SimpleNamespace(global_step=step), module)  # type: ignore[arg-type]
    assert checks == [3]
