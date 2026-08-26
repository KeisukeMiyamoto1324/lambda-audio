import argparse

from src.stage1.config import Stage1Config
from src.training.cli import add_training_arguments, validated_config


def parse_arguments() -> Stage1Config:
    defaults = Stage1Config()
    parser = argparse.ArgumentParser()
    add_training_arguments(parser, defaults)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=defaults.learning_rate,
    )
    return validated_config(parser, Stage1Config)
