import argparse
from pathlib import Path

from pydantic import ValidationError

from src.stage1.config import Stage1Config


def parse_arguments() -> Stage1Config:
    defaults = Stage1Config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=defaults.resume_checkpoint,
    )
    parser.add_argument(
        "--metadata-index-path",
        type=Path,
        default=defaults.metadata_index_path,
    )
    parser.add_argument(
        "--shard-cache-dir",
        type=Path,
        default=defaults.shard_cache_dir,
    )
    parser.add_argument(
        "--bucket-buffer-size",
        type=int,
        default=defaults.bucket_buffer_size,
    )
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=defaults.prefetch_factor,
    )
    parser.add_argument(
        "--train-max-batch-padded-audio-seconds",
        type=float,
        default=defaults.train_max_batch_padded_audio_seconds,
    )
    parser.add_argument(
        "--validation-max-batch-padded-audio-seconds",
        type=float,
        default=defaults.validation_max_batch_padded_audio_seconds,
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=defaults.gradient_accumulation_steps,
    )
    parser.add_argument("--max-steps", type=int, default=defaults.max_steps)
    parser.add_argument("--warmup-steps", type=int, default=defaults.warmup_steps)
    parser.add_argument(
        "--min-learning-rate-ratio",
        type=float,
        default=defaults.min_learning_rate_ratio,
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=defaults.early_stopping_patience,
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=defaults.early_stopping_min_delta,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=defaults.learning_rate,
    )
    try:
        return Stage1Config.model_validate(vars(parser.parse_args()))
    except ValidationError as error:
        parser.error(str(error))
