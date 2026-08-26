from typing import Any

from src.stage1.config import Stage1Config
from src.training.lightning import BaseLambdaAudioLightningModule
from src.training.optimizer import create_optimizer_configuration


class Stage1LightningModule(BaseLambdaAudioLightningModule[Stage1Config]):
    def configure_optimizers(self) -> dict[str, Any]:
        return create_optimizer_configuration(
            self.model,
            self.config,
            lambda name: self.config.learning_rate,
        )
