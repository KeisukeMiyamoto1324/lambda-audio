import csv

import lightning as L

from src.shared.inference_report import InferenceReport
from src.shared.samples import load_inference_samples
from src.shared.transcription import iter_transcriptions
from src.training.config import TrainingConfig
from src.training.data_module import LambdaAudioDataModule
from src.training.lightning import LambdaAudioLightningModule
from src.training.performance import TrainingProgressBar


class ValidationInferenceMonitor(L.Callback):
    def __init__(
        self,
        config: TrainingConfig,
        progress_bar: TrainingProgressBar,
    ) -> None:
        # ---------------------------------------------------------
        # Store inference settings and the append-only history location.
        # ---------------------------------------------------------
        self.config = config
        self.progress_bar = progress_bar
        self.history_path = config.output_dir / "inference_history.csv"

    def on_validation_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        # ---------------------------------------------------------
        # Run inference on the main rank and synchronize all ranks.
        # ---------------------------------------------------------
        if not isinstance(pl_module, LambdaAudioLightningModule):
            raise TypeError("LambdaAudioLightningModule is required.")

        if trainer.is_global_zero:
            self._run_inference(trainer, pl_module)
        trainer.strategy.barrier("validation_inference")

    def _run_inference(
        self,
        trainer: L.Trainer,
        pl_module: LambdaAudioLightningModule,
    ) -> None:
        data_module = trainer.datamodule
        if not isinstance(data_module, LambdaAudioDataModule):
            raise TypeError("LambdaAudioDataModule is required.")

        # ---------------------------------------------------------
        # Transcribe samples and display aligned reference differences.
        # ---------------------------------------------------------
        samples = load_inference_samples(self.config.inference_samples_path)
        report = InferenceReport(f"Validation Inference — Step {trainer.global_step}")
        records: list[dict[str, str | int | float]] = []

        model = pl_module.model
        model.eval()
        for sample, result in iter_transcriptions(
            model,
            data_module.collator.feature_extractor,
            data_module.collator.tokenizer,
            self.config,
            pl_module.device,
            samples,
            self.config.inference_batch_size,
        ):
            report.add_sample(sample.audio_name, sample.reference, result.text)
            records.append(
                {
                    "step": trainer.global_step,
                    "audio": sample.audio_name,
                    "reference": sample.reference,
                    "prediction": result.text,
                    "audio_seconds": result.audio_seconds,
                    "inference_seconds": result.inference_seconds,
                }
            )

        # ---------------------------------------------------------
        # Append one row per sample to compare predictions across steps.
        # ---------------------------------------------------------
        write_header = not self.history_path.exists()
        fieldnames = list(records[0])
        with self.history_path.open("a", encoding="utf-8", newline="") as history_file:
            writer = csv.DictWriter(history_file, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(records)
        self.progress_bar.print_table(report.build_table())
