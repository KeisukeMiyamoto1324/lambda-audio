import argparse
from pathlib import Path

from pydantic import ValidationError

from src.test.config import EvaluationConfig


def parse_arguments() -> EvaluationConfig:
    defaults = EvaluationConfig.model_construct()
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", dest="checkpoint_path", type=Path, required=True)
    parser.add_argument("--output", dest="output_path", type=Path, default=defaults.output_path)
    parser.add_argument("--max-new-tokens", type=int, default=defaults.max_new_tokens)
    try:
        return EvaluationConfig.model_validate(vars(parser.parse_args()))
    except ValidationError as error:
        parser.error(str(error))
