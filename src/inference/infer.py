from time import perf_counter

from src.environment import load_hf_token
from src.inference.cli import parse_arguments
from src.inference.loader import select_device
from src.inference.samples import load_inference_samples
from src.inference.transcription import transcribe
from src.modeling.checkpoint import load_model_checkpoint
from src.modeling.loading import load_model_components


def main() -> None:
    config = parse_arguments()
    device = select_device()
    components = load_model_components(config, load_hf_token())
    load_model_checkpoint(components.model, config.checkpoint_path)
    components.model.to(device)
    components.model.eval()
    samples = load_inference_samples(config.input_path)

    for index, sample in enumerate(samples, start=1):
        inference_start = perf_counter()
        result = transcribe(
            components.model,
            components.feature_extractor,
            components.tokenizer,
            config,
            device,
            sample.audio,
        )
        inference_seconds = perf_counter() - inference_start
        print("-" * 72)
        print(f"[{index}/{len(samples)}] {sample.audio_name}")
        print(f"正解({result.audio_seconds:.2g}秒): {sample.reference}")
        print(f"推論({inference_seconds:.2g}秒): {result.text}")


if __name__ == "__main__":
    main()
