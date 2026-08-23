from pathlib import Path

from torch import nn

from src.stage1.config import Stage1Config
from src.stage1.training import Stage1LightningModule
from src.stage2.config import Stage2Config
from src.stage2.training import Stage2LightningModule


class DummyTrainingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adapter = nn.Linear(2, 2)
        self.encoder = nn.Linear(2, 2)


def test_stage1_optimizer_uses_one_learning_rate() -> None:
    config = Stage1Config(output_dir=Path("outputs"))
    module = Stage1LightningModule(DummyTrainingModel(), config)  # type: ignore[arg-type]
    optimizer = module.configure_optimizers()["optimizer"]
    learning_rates = {group["initial_lr"] for group in optimizer.param_groups}

    assert learning_rates == {config.learning_rate}


def test_stage2_optimizer_uses_separate_learning_rates() -> None:
    config = Stage2Config(
        output_dir=Path("outputs"),
        initialization_checkpoint=Path("stage-1/model.pt"),
    )
    module = Stage2LightningModule(DummyTrainingModel(), config)  # type: ignore[arg-type]
    optimizer = module.configure_optimizers()["optimizer"]
    learning_rates = {group["initial_lr"] for group in optimizer.param_groups}

    assert learning_rates == {
        config.adapter_learning_rate,
        config.backbone_learning_rate,
    }
