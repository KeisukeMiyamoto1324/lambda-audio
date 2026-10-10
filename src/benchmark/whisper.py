import argparse
from pathlib import Path

import torch
from pydantic import ValidationError
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.benchmark.config import WhisperBenchmarkConfig
from src.environment import load_hf_token
from src.shared.device import select_device
from src.shared.inference_display import display_inference
from src.shared.samples import load_inference_samples
from src.shared.transcription import iter_batch_transcriptions
from src.shared.whisper import WhisperTranscriber

MODEL_ID = "openai/whisper-large-v3-turbo"


def parse_arguments() -> WhisperBenchmarkConfig:
    defaults = WhisperBenchmarkConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input_path", type=Path, default=defaults.input_path)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--max-new-tokens", type=int, default=defaults.max_new_tokens)
    try:
        return WhisperBenchmarkConfig.model_validate(vars(parser.parse_args()))
    except ValidationError as error:
        parser.error(str(error))


def main() -> None:
    config = parse_arguments()
    device = select_device()
    token = load_hf_token()
    processor = WhisperProcessor.from_pretrained(MODEL_ID, token=token)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, token=token, dtype=torch.float32, attn_implementation="sdpa")
    model.to(device)
    model.eval()
    samples = load_inference_samples(config.input_path)
    transcriber = WhisperTranscriber(model, processor, config, device)
    results = iter_batch_transcriptions(samples, config.batch_size, transcriber)
    display_inference(results, len(samples), "Whisper large-v3-turbo")


if __name__ == "__main__":
    main()
