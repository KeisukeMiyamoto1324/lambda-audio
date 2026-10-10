from time import perf_counter

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn

from src.environment import load_hf_token
from src.inference.cli import parse_arguments
from src.modeling.checkpoint import load_model_checkpoint
from src.modeling.loading import load_model_components
from src.shared.device import select_device
from src.shared.inference_report import InferenceReport
from src.shared.samples import load_inference_samples
from src.shared.transcription import iter_transcriptions


def main() -> None:
    config = parse_arguments()
    device = select_device()
    components = load_model_components(config, load_hf_token())
    load_model_checkpoint(components.model, config.checkpoint_path)
    components.model.to(device)
    components.model.eval()
    samples = load_inference_samples(config.input_path)
    report = InferenceReport("Inference")

    inference_start = perf_counter()
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TextColumn("CER: {task.fields[cer]}"),
        TextColumn("Skipped: {task.fields[skipped]}"),
        TimeRemainingColumn(),
    ) as progress:
        task_id = progress.add_task("Inference", total=len(samples), cer=report.cer_text, skipped=0)
        for sample, result in iter_transcriptions(
            components.model,
            components.feature_extractor,
            components.tokenizer,
            config,
            device,
            samples,
            config.batch_size,
        ):
            report.add_sample(sample.audio_name, sample.reference, result.text)
            progress.update(task_id, advance=1, cer=report.cer_text, skipped=report.skipped_samples)

    elapsed_seconds = perf_counter() - inference_start
    console = Console()
    console.print(report.build_table())
    console.print(f"Elapsed time: {elapsed_seconds:.2f} seconds")


if __name__ == "__main__":
    main()
