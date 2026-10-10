from src.benchmark.cli import parse_arguments
from src.benchmark.registry import MODEL_LOADERS
from src.environment import load_hf_token
from src.shared.device import select_device
from src.shared.inference_display import display_inference
from src.shared.samples import load_inference_samples
from src.shared.transcription import iter_batch_transcriptions


def main() -> None:
    config = parse_arguments()
    samples = load_inference_samples(config.input_path)
    transcriber = MODEL_LOADERS[config.model](config, select_device(), load_hf_token())
    results = iter_batch_transcriptions(samples, config.batch_size, transcriber)
    display_inference(results, len(samples), config.model)
