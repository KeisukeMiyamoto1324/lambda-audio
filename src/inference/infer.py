from rich.console import Console

from src.environment import load_hf_token
from src.inference.cli import parse_arguments
from src.inference.loader import select_device
from src.inference.samples import load_inference_samples
from src.inference.transcription import transcribe
from src.modeling.checkpoint import load_model_checkpoint
from src.modeling.loading import load_model_components
from src.shared.inference_report import InferenceReport


def main() -> None:
    config = parse_arguments()
    device = select_device()
    components = load_model_components(config, load_hf_token())
    load_model_checkpoint(components.model, config.checkpoint_path)
    components.model.to(device)
    components.model.eval()
    samples = load_inference_samples(config.input_path)
    report = InferenceReport("Inference")

    for sample in samples:
        result = transcribe(
            components.model,
            components.feature_extractor,
            components.tokenizer,
            config,
            device,
            sample.audio,
        )
        report.add_sample(sample.audio_name, sample.reference, result.text)

    Console().print(report.build_table())


if __name__ == "__main__":
    main()
