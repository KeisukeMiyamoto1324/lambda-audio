import csv
from itertools import batched

import lightning as L

from src.shared.inference_display import display_inference
from src.shared.samples import InferenceSample, load_multilingual_inference_samples
from src.shared.transcription import TranscriptionBatch, prepare_transcription_batch, transcribe_prepared_batch
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
        self.batches: list[tuple[tuple[InferenceSample, ...], TranscriptionBatch]] | None = None

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

        if self.progress_bar.progress is None:
            raise RuntimeError("The training progress display is not initialized.")

        # ---------------------------------------------------------
        # Cache fixed samples and preprocessing on CPU for this training run.
        # ---------------------------------------------------------
        if self.batches is None:
            samples = load_multilingual_inference_samples(self.config.inference_samples_path)
            self.batches = [
                (
                    batch,
                    prepare_transcription_batch(
                        data_module.collator.feature_extractor,
                        data_module.collator.tokenizer,
                        self.config,
                        [sample.audio for sample in batch],
                        [sample.lang for sample in batch],
                    ),
                )
                for batch in batched(samples, self.config.inference_batch_size)
            ]

        model = pl_module.model
        model.eval()
        results = (
            (sample, result)
            for samples, batch in self.batches
            for sample, result in zip(
                samples,
                transcribe_prepared_batch(model, data_module.collator.tokenizer, self.config, pl_module.device, batch),
                strict=True,
            )
        )
        # ---------------------------------------------------------
        # Apply training precision while consuming lazy inference batches.
        # ---------------------------------------------------------
        with trainer.precision_plugin.forward_context():
            completed = display_inference(
                results,
                sum(len(samples) for samples, _ in self.batches),
                f"Validation Inference — Step {trainer.global_step}",
                console=self.progress_bar.progress.console,
            )
        records = [
            {
                "step": trainer.global_step,
                "audio": sample.audio_name,
                "lang": sample.lang,
                "reference": sample.reference,
                "prediction": result.text,
                "audio_seconds": result.audio_seconds,
                "inference_seconds": result.inference_seconds,
            }
            for sample, result in completed
        ]

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
