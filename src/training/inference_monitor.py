import csv
from time import perf_counter

import lightning as L
from rich.table import Table

from src.inference.samples import load_inference_samples
from src.inference.transcription import transcribe
from src.training.config import TrainingConfig
from src.training.data_module import LambdaAudioDataModule
from src.training.lightning import LambdaAudioLightningModule
from src.training.performance import TrainingProgressBar
from src.training.text_diff import create_text_diff


class ValidationInferenceMonitor(L.Callback):
    def __init__(
        self,
        config: TrainingConfig,
        progress_bar: TrainingProgressBar,
    ) -> None:
        self.config = config
        self.progress_bar = progress_bar
        self.history_path = config.output_dir / "inference_history.csv"

    def on_validation_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
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

        samples = load_inference_samples(self.config.inference_samples_path)
        table = Table(title=f"Validation Inference — Step {trainer.global_step}")
        table.add_column("Audio", no_wrap=True)
        table.add_column("正解", overflow="fold")
        table.add_column("推論", overflow="fold")
        records: list[dict[str, str | int | float]] = []

        model = pl_module.model
        model.eval()
        model.decoder.config.use_cache = True
        for sample in samples:
            inference_start = perf_counter()
            result = transcribe(
                model,
                data_module.collator.feature_extractor,
                data_module.collator.tokenizer,
                self.config,
                pl_module.device,
                sample.audio,
            )
            inference_seconds = perf_counter() - inference_start
            reference_text, prediction_text = create_text_diff(
                sample.reference,
                result.text,
            )
            table.add_row(
                sample.audio_name,
                reference_text,
                prediction_text,
            )
            records.append(
                {
                    "step": trainer.global_step,
                    "audio": sample.audio_name,
                    "reference": sample.reference,
                    "prediction": result.text,
                    "audio_seconds": result.audio_seconds,
                    "inference_seconds": inference_seconds,
                }
            )
        model.decoder.config.use_cache = False

        # ---------------------------------------------------------
        # Append one row per sample so predictions remain comparable by step.
        # ---------------------------------------------------------
        write_header = not self.history_path.exists()
        fieldnames = list(records[0])
        with self.history_path.open("a", encoding="utf-8", newline="") as history_file:
            writer = csv.DictWriter(history_file, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(records)
        self.progress_bar.print_table(table)
