import argparse
from pathlib import Path

from pydantic import ValidationError

from src.inference.config import InferenceConfig


def parse_arguments() -> InferenceConfig:
    defaults = InferenceConfig.model_construct()
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", dest="checkpoint_path", type=Path, required=True)
    parser.add_argument(
        "--input",
        dest="input_path",
        type=Path,
        default=defaults.input_path,
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    try:
        return InferenceConfig.model_validate(vars(parser.parse_args()))
    except ValidationError as error:
        parser.error(str(error))
