from src.environment import load_hf_token
from src.inference.cli import parse_arguments
from src.modeling.checkpoint import load_model_checkpoint
from src.modeling.loading import load_model_components
from src.shared.device import select_device
from src.shared.inference_display import display_inference
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
    results = iter_transcriptions(
        components.model,
        components.feature_extractor,
        components.tokenizer,
        config,
        device,
        samples,
        config.batch_size,
    )
    display_inference(results, len(samples), "Inference")


if __name__ == "__main__":
    main()
