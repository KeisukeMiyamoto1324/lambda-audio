import argparse
from pathlib import Path

from dotenv import dotenv_values

from whisper.config import MODEL_ID, InferenceConfig
from whisper.evaluation.inference_display import display_inference
from whisper.evaluation.samples import load_inference_samples
from whisper.inference import load_engine


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone Whisper MLX inference and timing.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, choices=[1], default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--show-table", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 444:
        parser.error("Maximum new tokens must be between 1 and 444.")
    token = dotenv_values(args.env_file).get("HF_TOKEN")
    if not token:
        parser.error(f"HF_TOKEN is required in {args.env_file}.")
    config = InferenceConfig(max_new_tokens=args.max_new_tokens)
    samples = load_inference_samples(args.input)
    engine = load_engine(config, token)
    # ---------------------------------------------------------
    # Keep sample order and the same timing boundary as LambdaAudio MLX.
    # Model loading and Parquet reading finish before the timer starts.
    # ---------------------------------------------------------
    results = ((sample, engine([sample.audio])[0]) for sample in samples)
    display_inference(results, len(samples), MODEL_ID, show_table=args.show_table)


if __name__ == "__main__":
    main()
