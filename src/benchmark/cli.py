import argparse
from pathlib import Path

from pydantic import ValidationError

from src.benchmark.config import BenchmarkConfig
from src.benchmark.registry import MODEL_LOADERS


def parse_arguments() -> BenchmarkConfig:
    defaults = BenchmarkConfig.model_construct()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_LOADERS, required=True)
    parser.add_argument("--checkpoint", dest="checkpoint_path", type=Path)
    parser.add_argument("--input", dest="input_path", type=Path, default=defaults.input_path)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--max-new-tokens", type=int, default=defaults.max_new_tokens)
    try:
        return BenchmarkConfig.model_validate(vars(parser.parse_args()))
    except ValidationError as error:
        parser.error(str(error))
