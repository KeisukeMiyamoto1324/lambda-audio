from time import perf_counter

from rich.console import Console

from src.cer.cli import parse_arguments
from src.cer.evaluate import evaluate_cer
from src.environment import load_hf_token


def main() -> None:
    config = parse_arguments()
    start_time = perf_counter()
    metric = evaluate_cer(config, load_hf_token())
    elapsed_seconds = perf_counter() - start_time
    console = Console()
    console.print(f"CER: [bold]{metric.value:.2%}[/bold]")
    console.print(f"Elapsed time: [bold]{elapsed_seconds:.2f} seconds[/bold]")


if __name__ == "__main__":
    main()
