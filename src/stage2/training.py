from typing import Any

from src.stage2.config import Stage2Config
from src.training.lightning import BaseLambdaAudioLightningModule
from src.training.optimizer import create_optimizer_configuration


class Stage2LightningModule(BaseLambdaAudioLightningModule[Stage2Config]):
    def configure_optimizers(self) -> dict[str, Any]:
        return create_optimizer_configuration(
            self.model,
            self.config,
            self._learning_rate,
        )

    def _learning_rate(self, name: str) -> float:
        if name.startswith(("adapter.", "ctc_output.")) or ".cross_attention." in name:
            return self.config.adapter_learning_rate
        if name.startswith("encoder."):
            return self.config.encoder_learning_rate
        return self.config.backbone_learning_rate
