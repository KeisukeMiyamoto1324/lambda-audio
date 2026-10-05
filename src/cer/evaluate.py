import csv

import torch
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn

from src.cer.config import CerConfig
from src.cer.dataset import iterate_test_samples, load_test_samples
from src.cer.metric import CerMetric
from src.cer.text import normalize_cer_text, should_skip_text
from src.inference.loader import select_device
from src.inference.transcription import transcribe
from src.modeling.checkpoint import load_model_checkpoint
from src.modeling.loading import load_model_components


def evaluate_cer(config: CerConfig, token: str) -> CerMetric:
    device = select_device()
    components = load_model_components(config, token)
    load_model_checkpoint(components.model, config.checkpoint_path)
    components.model.to(device)
    components.model.eval()
    dataset = load_test_samples(config.cache_dir, config.sampling_rate)
    metric = CerMetric()
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TextColumn("CER: {task.fields[cer]:.2%}"),
        TextColumn("Skipped: {task.fields[skipped]}"),
        TimeRemainingColumn(),
    )

    # ---------------------------------------------------------
    # Accumulate corpus-level errors while showing live CER.
    # ---------------------------------------------------------
    with config.output_path.open("w", newline="", encoding="utf-8") as output, progress:
        writer = csv.writer(output)
        writer.writerow(["id", "reference", "prediction", "cer"])
        task_id = progress.add_task("Evaluating", total=len(dataset), cer=0.0, skipped=0)
        skipped = 0
        for sample_id, audio, reference in iterate_test_samples(dataset):
            if should_skip_text(reference):
                skipped += 1
                progress.update(task_id, advance=1, skipped=skipped)
                continue

            result = transcribe(
                components.model,
                components.feature_extractor,
                components.tokenizer,
                config,
                device,
                audio,
                num_beams=config.beam_search,
            )
            sample_metric = CerMetric()
            normalized_reference = normalize_cer_text(reference)
            normalized_prediction = normalize_cer_text(result.text)
            sample_metric.update(normalized_reference, normalized_prediction)
            metric.update(normalized_reference, normalized_prediction)
            writer.writerow([sample_id, reference, result.text, sample_metric.value])
            progress.update(task_id, advance=1, cer=metric.value)

    if device.type == "mps":
        torch.mps.empty_cache()
    return metric
