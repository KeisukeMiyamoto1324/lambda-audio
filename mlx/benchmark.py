import argparse
from functools import partial
from itertools import batched
from pathlib import Path

from dotenv import dotenv_values
from huggingface_hub import hf_hub_download

from mlx.config import InferenceConfig
from mlx.evaluation.inference_display import display_inference
from mlx.evaluation.samples import load_inference_samples
from mlx.inference import load_model_components, transcribe


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone MLX LambdaAudio inference and timing.")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model-id", default="KeisukeMiyamoto/lambda-audio-v6")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--show-table", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_new_tokens < 1:
        parser.error("Batch size and maximum new tokens must be positive.")
    token = dotenv_values(args.env_file).get("HF_TOKEN")
    if not token:
        parser.error(f"HF_TOKEN is required in {args.env_file}.")
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = Path(hf_hub_download(args.model_id, "model-mlx-4bit.safetensors", token=token))
    config = InferenceConfig(max_new_tokens=args.max_new_tokens)
    samples = load_inference_samples(args.input)
    components = load_model_components(config, checkpoint, token)
    transcribe_batch = partial(transcribe, components.model, components.feature_extractor, components.tokenizer, config)
    # ---------------------------------------------------------
    # Preserve input order and exclude model loading from elapsed time.
    # Include audio decoding, preprocessing, generation, and CER reporting.
    # ---------------------------------------------------------
    results = (result for batch in batched(samples, args.batch_size) for result in zip(batch, transcribe_batch([sample.audio for sample in batch]), strict=True))
    display_inference(results, len(samples), "lambda-audio-mlx", show_table=args.show_table)


if __name__ == "__main__":
    main()
