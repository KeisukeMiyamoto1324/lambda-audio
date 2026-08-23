from pathlib import Path

from torch import nn

from src.config import TrainingConfig
from src.training import LambdaAudioLightningModule


class DummyTrainingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adapter = nn.Linear(2, 2)
        self.encoder = nn.Linear(2, 2)


def test_optimizer_uses_separate_learning_rates() -> None:
    config = TrainingConfig(
        stage=2,
        output_dir=Path("outputs"),
        initialization_checkpoint=Path("stage-1/model.pt"),
    )
    module = LambdaAudioLightningModule(DummyTrainingModel(), config)  # type: ignore[arg-type]
    optimizer = module.configure_optimizers()["optimizer"]
    learning_rates = {group["initial_lr"] for group in optimizer.param_groups}

    assert learning_rates == {
        config.adapter_learning_rate,
        config.backbone_learning_rate,
    }
