from collections.abc import Iterable
from time import perf_counter

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn

from src.shared.inference_report import InferenceReport
from src.shared.samples import InferenceSample
from src.shared.transcription import TranscriptionResult


def display_inference(
    results: Iterable[tuple[InferenceSample, TranscriptionResult]],
    total_samples: int,
    title: str,
) -> None:
    report = InferenceReport(title)
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
        task_id = progress.add_task("Inference", total=total_samples, cer=report.cer_text, skipped=0)
        for sample, result in results:
            report.add_sample(sample.audio_name, sample.reference, result.text)
            progress.update(task_id, advance=1, cer=report.cer_text, skipped=report.skipped_samples)

    elapsed_seconds = perf_counter() - inference_start
    console = Console()
    console.print(report.build_table())
    console.print(f"Elapsed time: {elapsed_seconds:.2f} seconds")
