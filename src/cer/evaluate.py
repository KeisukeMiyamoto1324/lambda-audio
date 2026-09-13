import torch
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn

from src.cer.config import CerConfig
from src.cer.dataset import iterate_test_samples, load_test_samples
from src.cer.metric import CerMetric
from src.data.text import normalize_transcription
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
    dataset = load_test_samples(
        token,
        config.dataset_percentage,
        config.cache_dir,
        config.sampling_rate,
    )
    metric = CerMetric()
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TextColumn("CER: {task.fields[cer]:.2%}"),
        TimeRemainingColumn(),
    )

    # ---------------------------------------------------------
    # Accumulate corpus-level errors while showing live CER.
    # ---------------------------------------------------------
    with progress:
        task_id = progress.add_task("Evaluating", total=len(dataset), cer=0.0)
        for audio, reference in iterate_test_samples(dataset):
            result = transcribe(
                components.model,
                components.feature_extractor,
                components.tokenizer,
                config,
                device,
                audio,
            )
            metric.update(
                normalize_transcription(reference),
                normalize_transcription(result.text),
            )
            progress.update(task_id, advance=1, cer=metric.value)

    if device.type == "mps":
        torch.mps.empty_cache()
    return metric
