import argparse
from pathlib import Path
from time import perf_counter

import librosa

from src.inference.config import InferenceConfig
from src.inference.loader import load_components, select_device
from src.inference.samples import load_inference_samples
from src.inference.transcription import transcribe


def parse_arguments() -> InferenceConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    arguments = parser.parse_args()

    if not arguments.checkpoint.is_file():
        parser.error("--checkpoint must be an existing file.")
    if not arguments.input_dir.is_dir():
        parser.error("--input-dir must be an existing directory.")
    if arguments.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1.")

    return InferenceConfig(
        checkpoint_path=arguments.checkpoint,
        input_dir=arguments.input_dir,
        max_new_tokens=arguments.max_new_tokens,
    )


def main() -> None:
    config = parse_arguments()
    device = select_device()
    model, feature_extractor, tokenizer = load_components(config, device)
    samples = load_inference_samples(config.input_dir)

    for index, sample in enumerate(samples, start=1):
        audio_seconds = librosa.get_duration(path=sample.audio_path)
        inference_start = perf_counter()
        prediction = transcribe(
            model,
            feature_extractor,
            tokenizer,
            config,
            device,
            sample.audio_path,
        )
        inference_seconds = perf_counter() - inference_start
        print("-" * 72)
        print(f"[{index}/{len(samples)}] {sample.audio_path.name}")
        print(f"正解({audio_seconds:.2g}秒): {sample.reference}")
        print(f"推論({inference_seconds:.2g}秒): {prediction}")


if __name__ == "__main__":
    main()
