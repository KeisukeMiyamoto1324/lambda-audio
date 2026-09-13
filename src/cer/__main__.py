from rich.console import Console

from src.cer.cli import parse_arguments
from src.cer.evaluate import evaluate_cer
from src.environment import load_hf_token


def main() -> None:
    config = parse_arguments()
    metric = evaluate_cer(config, load_hf_token())
    Console().print(f"CER: [bold]{metric.value:.2%}[/bold]")


if __name__ == "__main__":
    main()
