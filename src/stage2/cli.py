import argparse
from pathlib import Path

from src.stage2.config import Stage2Config
from src.training.cli import add_training_arguments, validated_config


def parse_arguments() -> Stage2Config:
    defaults = Stage2Config.model_construct()
    parser = argparse.ArgumentParser()
    add_training_arguments(parser, defaults)
    parser.add_argument(
        "--initialization-checkpoint",
        type=Path,
        default=defaults.initialization_checkpoint,
    )
    parser.add_argument(
        "--adapter-learning-rate",
        type=float,
        default=defaults.adapter_learning_rate,
    )
    parser.add_argument(
        "--backbone-learning-rate",
        type=float,
        default=defaults.backbone_learning_rate,
    )
    return validated_config(parser, Stage2Config)
