import argparse
from pathlib import Path

from pydantic import ValidationError

from src.cer.config import CerConfig


def parse_arguments() -> CerConfig:
    defaults = CerConfig.model_construct()
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=Path, help="Path to a model checkpoint file.")
    parser.add_argument("--cache-dir", type=Path, default=defaults.cache_dir)
    parser.add_argument("--output", dest="output_path", type=Path, default=defaults.output_path)
    parser.add_argument("--max-new-tokens", type=int, default=defaults.max_new_tokens)
    parser.add_argument("--beam-search", type=int, default=defaults.beam_search, help="Number of beams (1 uses greedy decoding; default: 5).")
    try:
        return CerConfig.model_validate(vars(parser.parse_args()))
    except ValidationError as error:
        parser.error(str(error))
