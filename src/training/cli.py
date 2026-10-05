import argparse
from pathlib import Path

from pydantic import ValidationError

from src.training.config import TrainingConfig


def parse_arguments() -> TrainingConfig:
    defaults = TrainingConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=defaults.resume_checkpoint,
    )
    try:
        return TrainingConfig.model_validate(vars(parser.parse_args()))
    except ValidationError as error:
        parser.error(str(error))
