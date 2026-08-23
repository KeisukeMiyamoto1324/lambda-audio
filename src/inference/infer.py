import argparse
from pathlib import Path

from src.inference.config import InferenceConfig
from src.inference.loader import load_components, select_device
from src.inference.transcription import transcribe


def parse_arguments() -> InferenceConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    arguments = parser.parse_args()

    if not arguments.checkpoint.is_file():
        parser.error("--checkpoint must be an existing file.")
    if not arguments.audio.is_file():
        parser.error("--audio must be an existing file.")
    if arguments.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1.")

    return InferenceConfig(
        checkpoint_path=arguments.checkpoint,
        audio_path=arguments.audio,
        max_new_tokens=arguments.max_new_tokens,
    )


def main() -> None:
    config = parse_arguments()
    device = select_device()
    model, feature_extractor, tokenizer = load_components(config, device)
    transcription = transcribe(
        model,
        feature_extractor,
        tokenizer,
        config,
        device,
    )
    print(transcription)


if __name__ == "__main__":
    main()
