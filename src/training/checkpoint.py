from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import Tensor


class BestModelCheckpoint(ModelCheckpoint):
    def __init__(self, output_dir: Path) -> None:
        super().__init__(
            dirpath=output_dir,
            filename="cer-{validation_cer:.5f}-step-{step:08d}",
            monitor="validation_cer",
            mode="min",
            save_top_k=3,
            save_last=False,
            save_weights_only=False,
            save_on_train_epoch_end=False,
            auto_insert_metric_name=False,
            enable_version_counter=False,
        )
        self.alias_path = output_dir / "model.ckpt"

    def _update_best_and_save(
        self,
        current: Tensor,
        trainer: L.Trainer,
        monitor_candidates: dict[str, Tensor],
    ) -> None:
        super()._update_best_and_save(current, trainer, monitor_candidates)
        self._link_checkpoint(
            trainer,
            self.best_model_path,
            str(self.alias_path),
        )
